//! Mixes microphone + system audio, writes the MP3 and feeds the transcriber.
//!
//! Mixing is paced by the wall clock rather than by either device: WASAPI
//! loopback delivers *no* data while nothing is playing, so waiting for both
//! sources would stall. Each tick we emit the audio that "should" exist by now.
//!
//! Every source has a small jitter buffer: it only joins the mix once it has
//! `CUSHION` audio queued, and after an underrun (e.g. loopback going quiet)
//! it re-buffers instead of being chopped into tiny gaps. Choppy input is
//! audible in the MP3 and badly hurts recognition accuracy.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use serde::Serialize;

use super::capture::{SourceBuffer, MIX_RATE};
use super::mp3::Mp3Writer;
use super::resample::Resampler;
use crate::transcribe::{AsrMsg, ASR_RATE};

const TICK: Duration = Duration::from_millis(40);
/// Audio a source must have queued before it is mixed (absorbs delivery jitter).
const CUSHION: usize = MIX_RATE as usize / 10;
/// If a device clock runs fast and its queue grows past this, drop the backlog.
const MAX_BACKLOG: usize = CUSHION + MIX_RATE as usize / 4;
const UI_INTERVAL: Duration = Duration::from_millis(100);

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Levels {
    pub mic: f32,
    pub system: f32,
    /// Seconds of audio recorded so far (pauses excluded).
    pub seconds: f64,
}

pub struct MixerControl {
    pub paused: Arc<AtomicBool>,
    pub stop: Arc<AtomicBool>,
    /// Muted sources ([microphone, computer audio]) are left out of the
    /// recording and the transcript; the recording itself carries on.
    pub muted: [Arc<AtomicBool>; 2],
}

pub fn spawn_mixer(
    mic: Option<Arc<SourceBuffer>>,
    system: Option<Arc<SourceBuffer>>,
    mut mp3: Mp3Writer,
    asr: Sender<AsrMsg>,
    ctl: MixerControl,
    mut on_levels: impl FnMut(Levels) + Send + 'static,
) -> std::thread::JoinHandle<Result<()>> {
    std::thread::Builder::new()
        .name("kalam-mixer".into())
        .spawn(move || {
            let sources: [Option<&SourceBuffer>; 2] = [mic.as_deref(), system.as_deref()];
            let mut down = Resampler::new(MIX_RATE, ASR_RATE);
            let mut clock = Instant::now();
            let mut produced: usize = 0;
            let mut primed = [false; 2];
            let mut recorded: u64 = 0;
            let mut was_paused = false;
            let mut last_ui = Instant::now();
            let mut mix: Vec<f32> = Vec::new();
            let mut tmp: Vec<f32> = Vec::new();
            let mut asr_buf: Vec<f32> = Vec::new();
            let mut mp3_error: Option<anyhow::Error> = None;

            loop {
                let stopping = ctl.stop.load(Ordering::SeqCst);

                if ctl.paused.load(Ordering::SeqCst) && !stopping {
                    if !was_paused {
                        was_paused = true;
                        let _ = asr.send(AsrMsg::Flush);
                    }
                    sources.iter().flatten().for_each(|s| {
                        s.clear();
                        s.take_peak();
                    });
                    if last_ui.elapsed() >= UI_INTERVAL {
                        last_ui = Instant::now();
                        on_levels(Levels { mic: 0.0, system: 0.0, seconds: recorded as f64 / MIX_RATE as f64 });
                    }
                    std::thread::sleep(TICK);
                    continue;
                }
                if was_paused {
                    // Resuming (or stopping while paused): audio captured during
                    // the pause must not end up in the recording.
                    was_paused = false;
                    clock = Instant::now();
                    produced = 0;
                    primed = [false; 2];
                    sources.iter().flatten().for_each(|s| s.clear());
                }

                let need = if stopping {
                    sources.iter().flatten().map(|s| s.len()).max().unwrap_or(0)
                } else {
                    let due = (clock.elapsed().as_secs_f64() * MIX_RATE as f64) as usize;
                    due.saturating_sub(produced)
                };

                if need > 0 {
                    mix.clear();
                    mix.resize(need, 0.0);
                    let mut rms = [0f32; 2];
                    for (i, src) in sources.iter().enumerate() {
                        let Some(src) = src else { continue };
                        if !primed[i] {
                            if stopping || src.len() >= CUSHION {
                                primed[i] = true;
                            } else {
                                continue; // still buffering: silence for now
                            }
                        }
                        tmp.clear();
                        let got = src.take_into(need, &mut tmp);
                        if got < need && !stopping {
                            primed[i] = false; // underrun: re-buffer before continuing
                        }
                        if !stopping && src.len() > MAX_BACKLOG {
                            src.trim_to(CUSHION);
                        }
                        // Muted: the audio is still taken, so the source stays in
                        // step for when it is unmuted, but it is not mixed in.
                        if ctl.muted[i].load(Ordering::Relaxed) {
                            continue;
                        }
                        let mut energy = 0f32;
                        for (m, s) in mix.iter_mut().zip(&tmp) {
                            *m += *s;
                            energy += s * s;
                        }
                        rms[i] = (energy / need as f32).sqrt();
                    }
                    mix.iter_mut().for_each(|m| *m = m.clamp(-1.0, 1.0));

                    if mp3_error.is_none() {
                        if let Err(e) = mp3.write(&mix) {
                            log::error!("MP3 write failed: {e:#}");
                            mp3_error = Some(e);
                        }
                    }
                    produced += need;
                    recorded += need as u64;

                    asr_buf.clear();
                    down.process_into(&mix, &mut asr_buf);
                    let _ = asr.send(AsrMsg::Audio {
                        samples: std::mem::take(&mut asr_buf),
                        mic_rms: rms[0],
                        system_rms: rms[1],
                    });
                }

                if last_ui.elapsed() >= UI_INTERVAL {
                    last_ui = Instant::now();
                    let peak = |i: usize, src: &Option<Arc<SourceBuffer>>| {
                        let p = src.as_ref().map_or(0.0, |s| s.take_peak());
                        if ctl.muted[i].load(Ordering::Relaxed) { 0.0 } else { p }
                    };
                    on_levels(Levels {
                        mic: peak(0, &mic),
                        system: peak(1, &system),
                        seconds: recorded as f64 / MIX_RATE as f64,
                    });
                }

                if stopping {
                    break;
                }
                std::thread::sleep(TICK);
            }

            let _ = asr.send(AsrMsg::Stop);
            let finished = mp3.finish();
            match mp3_error {
                Some(e) => Err(e),
                None => finished,
            }
        })
        .expect("spawn mixer thread")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::capture::{spawn_capture, CaptureRequest};
    use crate::test_util::{play_on_speakers, test_speech};
    use crate::transcribe::{spawn_worker, AsrEvent, WorkerConfig};
    use std::sync::mpsc;
    use std::sync::Mutex;

    /// Muting the microphone mid-recording leaves it out of the mix (MP3 and
    /// transcriber) while computer audio keeps recording, and unmuting brings
    /// it back.
    #[test]
    fn muted_source_is_left_out() {
        let mic = Arc::new(SourceBuffer::default());
        let sys = Arc::new(SourceBuffer::default());
        let done = Arc::new(AtomicBool::new(false));
        // Live-like delivery: 20 ms blocks of a steady 0.5 (mic) and 0.25 (system).
        let feeder = {
            let (mic, sys, done) = (mic.clone(), sys.clone(), done.clone());
            std::thread::spawn(move || {
                while !done.load(Ordering::SeqCst) {
                    mic.push(&[0.5; 960]);
                    sys.push(&[0.25; 960]);
                    std::thread::sleep(Duration::from_millis(20));
                }
            })
        };
        let mp3_path = std::env::temp_dir().join(format!("kalam-mute-test-{}.mp3", std::process::id()));
        let mp3 = Mp3Writer::create(&mp3_path, MIX_RATE, 128).unwrap();
        let (asr_tx, asr_rx) = mpsc::channel::<AsrMsg>();
        let muted = [Arc::new(AtomicBool::new(false)), Arc::new(AtomicBool::new(false))];
        let stop = Arc::new(AtomicBool::new(false));
        let mixer = spawn_mixer(
            Some(mic.clone()),
            Some(sys.clone()),
            mp3,
            asr_tx,
            MixerControl { paused: Arc::new(AtomicBool::new(false)), stop: stop.clone(), muted: muted.clone() },
            |_| {},
        );
        // (mic level, system level, middle sample) of each block sent to the transcriber.
        let blocks = |rx: &mpsc::Receiver<AsrMsg>| -> Vec<(f32, f32, f32)> {
            rx.try_iter()
                .filter_map(|m| match m {
                    AsrMsg::Audio { samples, mic_rms, system_rms } if samples.len() > 20 => {
                        Some((mic_rms, system_rms, samples[samples.len() / 2]))
                    }
                    _ => None,
                })
                .collect()
        };
        let settle = || std::thread::sleep(Duration::from_millis(250));
        let measure = || std::thread::sleep(Duration::from_millis(400));

        settle();
        blocks(&asr_rx);
        measure();
        let both = blocks(&asr_rx);

        muted[0].store(true, Ordering::SeqCst);
        settle();
        blocks(&asr_rx);
        measure();
        let mic_muted = blocks(&asr_rx);

        muted[0].store(false, Ordering::SeqCst);
        settle();
        blocks(&asr_rx);
        measure();
        let unmuted = blocks(&asr_rx);

        stop.store(true, Ordering::SeqCst);
        done.store(true, Ordering::SeqCst);
        mixer.join().unwrap().unwrap();
        feeder.join().unwrap();
        let _ = std::fs::remove_file(&mp3_path);

        let mixed = |v: &[(f32, f32, f32)], mic_on: bool| {
            assert!(!v.is_empty(), "audio keeps flowing");
            let full = v.iter().filter(|b| b.1 > 0.0).count();
            assert!(full * 2 >= v.len(), "system audio recorded throughout: {v:?}");
            for &(mic_rms, _, sample) in v.iter().filter(|b| b.1 > 0.0) {
                if mic_on {
                    assert!(mic_rms > 0.0, "mic recorded: {v:?}");
                } else {
                    assert_eq!(mic_rms, 0.0, "muted mic left out: {v:?}");
                    assert!((sample - 0.25).abs() < 0.05, "only system audio in the mix: {v:?}");
                }
            }
        };
        mixed(&both, true);
        mixed(&mic_muted, false);
        mixed(&unmuted, true);
    }

    /// Plays a speech WAV through the default speakers while recording the
    /// computer's audio (loopback) + microphone, exactly like a session does,
    /// then checks the MP3 and the live transcript.
    ///   KALAM_TEST_WAV=jfk.wav cargo test --release loopback -- --ignored --nocapture
    #[test]
    #[ignore]
    fn loopback_capture_mix_mp3_and_transcribe() {
        let speech = test_speech(0.35);

        // Start capture exactly as Session does.
        let mic = Arc::new(SourceBuffer::default());
        let sys = Arc::new(SourceBuffer::default());
        let (stop_tx, stop_rx) = mpsc::channel();
        let (ready_tx, ready_rx) = mpsc::channel();
        let capture = spawn_capture(
            CaptureRequest { mic: Some((None, mic.clone())), system: Some((None, sys.clone())) },
            stop_rx,
            ready_tx,
        );
        let report = ready_rx.recv().unwrap().expect("capture opens");
        println!("capture: {report:?}");
        assert!(report.system_device.is_some(), "loopback device opened");

        let mp3_path = std::env::temp_dir().join("kalam-loopback-test.mp3");
        let mp3 = Mp3Writer::create(&mp3_path, MIX_RATE, 128).unwrap();
        let (asr_tx, tap_rx) = mpsc::channel::<AsrMsg>();
        let (fwd_tx, asr_rx) = mpsc::channel();
        // Tap: save what the transcriber receives as a WAV for inspection.
        let tap = std::thread::spawn(move || {
            let spec = hound::WavSpec { channels: 3, sample_rate: ASR_RATE, bits_per_sample: 32, sample_format: hound::SampleFormat::Float };
            let path = std::env::temp_dir().join("kalam-loopback-tap.wav");
            let mut w = hound::WavWriter::create(&path, spec).unwrap();
            for msg in tap_rx {
                if let AsrMsg::Audio { samples, mic_rms, system_rms } = &msg {
                    for s in samples {
                        w.write_sample(*s).unwrap();
                        w.write_sample(*mic_rms).unwrap();
                        w.write_sample(*system_rms).unwrap();
                    }
                }
                let _ = fwd_tx.send(msg);
            }
            w.finalize().unwrap();
            println!("tap: {}", path.display());
        });
        let models = crate::test_util::test_models();
        let text = Arc::new(Mutex::new(String::new()));
        let worker = {
            let text = Arc::clone(&text);
            spawn_worker(
                models,
                WorkerConfig::for_test(false, true),
                asr_rx,
                move |ev| {
                    if let AsrEvent::Final(segs) = ev {
                        for s in segs {
                            println!("FINAL [{} ms] {:?}: {}", s.start_ms, s.speaker, s.text);
                            text.lock().unwrap().push_str(&format!(" {}", s.text));
                        }
                    }
                },
            )
        };
        let paused = Arc::new(AtomicBool::new(false));
        let stop = Arc::new(AtomicBool::new(false));
        let peak_sys = Arc::new(Mutex::new(0f32));
        let mixer = {
            let peak_sys = Arc::clone(&peak_sys);
            spawn_mixer(
                Some(mic),
                Some(sys),
                mp3,
                asr_tx,
                MixerControl { paused: paused.clone(), stop: stop.clone(), muted: Default::default() },
                move |l| {
                    let mut p = peak_sys.lock().unwrap();
                    *p = p.max(l.system);
                },
            )
        };

        play_on_speakers(&speech);
        std::thread::sleep(Duration::from_millis(1200));

        let _ = stop_tx.send(());
        capture.join().unwrap();
        stop.store(true, Ordering::SeqCst);
        mixer.join().unwrap().unwrap();
        tap.join().unwrap();
        worker.join().unwrap();

        let mp3_len = std::fs::metadata(&mp3_path).unwrap().len();
        println!("mp3: {} ({} bytes), peak system level {:.3}", mp3_path.display(), mp3_len, peak_sys.lock().unwrap());
        assert!(mp3_len > 100_000, "mp3 has ~12 s of 128 kbps audio");
        assert!(*peak_sys.lock().unwrap() > 0.05, "loopback heard the playback");
        let text = text.lock().unwrap().to_lowercase();
        assert!(text.contains("what your country can do for you"), "transcript: {text}");
    }
}
