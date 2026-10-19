//! One recording session: capture → mixer → (MP3 file, Whisper → DOCX file).

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use serde::Serialize;
use tauri::{AppHandle, Emitter, Runtime};

use crate::audio::capture::{spawn_capture, CaptureRequest, SourceBuffer, MIX_RATE};
use crate::audio::mixer::{spawn_mixer, MixerControl};
use crate::audio::mp3::Mp3Writer;
use crate::docx::{write_transcript, TranscriptMeta};
use crate::settings::{Settings, POLISH_ALWAYS, POLISH_AUTO, SPEAKERS_SOURCE, SPEAKERS_VOICE};
use crate::transcribe::{spawn_worker, AsrEvent, FinishStage, ModelManager, ModelStatus, Segment, SpeakerMode, WorkerConfig};

pub const EV_LEVELS: &str = "kalam://levels";
pub const EV_PREVIEW: &str = "kalam://preview";
pub const EV_SEGMENTS: &str = "kalam://segments";
pub const EV_ERROR: &str = "kalam://error";
pub const EV_POLISH: &str = "kalam://polish";
pub const EV_POLISHED: &str = "kalam://polished";

/// Progress of the finishing pass (polish and/or speakers), for the UI.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PolishUpdate {
    /// "running", "done" or "abandoned".
    pub state: &'static str,
    /// "polish" or "speakers" while running.
    pub stage: &'static str,
    pub progress: f32,
    pub message: Option<String>,
}

const QUALITY_LIVE: &str = "Live transcript";
const QUALITY_FINISHING: &str = "Live transcript (being finished…)";

/// A polish pass still running after its recording stopped.
pub struct Polisher {
    handle: JoinHandle<()>,
    cancel: Arc<AtomicBool>,
}

impl Polisher {
    pub fn is_finished(&self) -> bool {
        self.handle.is_finished()
    }

    /// Wait for the pass to finish.
    pub fn wait(self) {
        let _ = self.handle.join();
    }

    /// Ask the pass to stop after its current window and wait for it.
    pub fn cancel_and_wait(self) {
        self.cancel.store(true, Ordering::SeqCst);
        let _ = self.handle.join();
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionInfo {
    pub mp3_path: String,
    pub docx_path: String,
    pub mic_device: Option<String>,
    pub system_device: Option<String>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SavedFiles {
    pub mp3_path: String,
    pub docx_path: String,
    pub duration_secs: f64,
    pub segments: usize,
    pub warnings: Vec<String>,
    /// The transcript is being polished in the background.
    pub polishing: bool,
}

struct Transcript {
    path: PathBuf,
    meta: TranscriptMeta,
    segments: Vec<Segment>,
}

impl Transcript {
    fn save(&self) -> Result<()> {
        write_transcript(&self.path, &self.meta, &self.segments)
    }
}

/// An audio source that can be muted during a recording.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Source {
    Mic,
    System,
}

pub struct Session {
    pub info: SessionInfo,
    paused: Arc<AtomicBool>,
    /// [microphone, computer audio]; see `MixerControl::muted`.
    muted: [Arc<AtomicBool>; 2],
    stop: Arc<AtomicBool>,
    recorded_ms: Arc<AtomicU64>,
    capture_stop: Sender<()>,
    capture: JoinHandle<()>,
    mixer: JoinHandle<Result<()>>,
    asr: JoinHandle<()>,
    transcript: Arc<Mutex<Transcript>>,
    live_done: mpsc::Receiver<bool>,
    cancel_polish: Arc<AtomicBool>,
}

impl Session {
    /// `voice_model`: the voiceprint model for speaker labels by voice.
    pub fn start<R: Runtime>(app: &AppHandle<R>, settings: &Settings, models: Arc<ModelManager>, voice_model: &Path) -> Result<Self> {
        let out_dir = settings.output_path();
        std::fs::create_dir_all(&out_dir)
            .with_context(|| format!("Cannot create the output folder {}", out_dir.display()))?;
        let started = chrono::Local::now();
        let base = unique_base(&out_dir, &format!("Kalam_{}", started.format("%Y-%m-%d_%H-%M-%S")));
        let mp3_path = base.with_extension("mp3");
        let docx_path = base.with_extension("docx");

        // 1. Open the audio devices.
        let mic_buf = settings.capture_mic.then(|| Arc::new(SourceBuffer::default()));
        let sys_buf = settings.capture_system.then(|| Arc::new(SourceBuffer::default()));
        if mic_buf.is_none() && sys_buf.is_none() {
            return Err(anyhow!("Both audio sources are switched off. Enable the microphone and/or system audio in Settings."));
        }
        let (capture_stop, capture_stop_rx) = mpsc::channel();
        let (ready_tx, ready_rx) = mpsc::channel();
        let capture = spawn_capture(
            CaptureRequest {
                mic: mic_buf.clone().map(|b| (settings.mic_device.clone(), b)),
                system: sys_buf.clone().map(|b| (settings.output_device.clone(), b)),
            },
            capture_stop_rx,
            ready_tx,
        );
        let report = match ready_rx.recv_timeout(Duration::from_secs(15)) {
            Ok(Ok(r)) => r,
            Ok(Err(msg)) => {
                let _ = capture.join();
                return Err(anyhow!(msg));
            }
            Err(_) => {
                let _ = capture_stop.send(());
                return Err(anyhow!("Timed out opening the audio devices"));
            }
        };
        // Only mix sources that actually opened.
        let mic_buf = mic_buf.filter(|_| report.mic_device.is_some());
        let sys_buf = sys_buf.filter(|_| report.system_device.is_some());

        let mp3 = match Mp3Writer::create(&mp3_path, MIX_RATE, settings.mp3_bitrate) {
            Ok(w) => w,
            Err(e) => {
                let _ = capture_stop.send(());
                let _ = capture.join();
                return Err(e);
            }
        };

        // 2. Transcript document, re-saved after every new sentence.
        let sources = [
            report.mic_device.as_ref().map(|d| format!("Microphone ({d})")),
            report.system_device.as_ref().map(|d| format!("Computer audio ({d})")),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join(" + ");
        let engine = match models.status() {
            ModelStatus::Ready { name, on_gpu: true, .. } => format!("Whisper {name} (offline, whisper.cpp, graphics card)"),
            ModelStatus::Ready { name, .. } | ModelStatus::Loading { name } | ModelStatus::Failed { name, .. } => {
                format!("Whisper {name} (offline, whisper.cpp)")
            }
        };
        let transcript = Arc::new(Mutex::new(Transcript {
            path: docx_path.clone(),
            meta: TranscriptMeta {
                started,
                duration_secs: 0.0,
                sources,
                engine,
                audio_file: file_name(&mp3_path),
                in_progress: true,
                quality: None,
            },
            segments: Vec::new(),
        }));
        if let Err(e) = transcript.lock().unwrap().save() {
            log::warn!("Initial transcript save failed: {e:#}");
        }

        // 3. Speech recognition worker.
        let (asr_tx, asr_rx) = mpsc::channel();
        let recorded_ms = Arc::new(AtomicU64::new(0));
        let (live_done_tx, live_done) = mpsc::channel();
        // Polishing costs ~0.25x the meeting length on a graphics card but
        // over 1x on a CPU, so "auto" only polishes with a GPU.
        let polish = settings.polish == POLISH_ALWAYS || (settings.polish == POLISH_AUTO && models.gpu_enabled());
        let cancel_polish = Arc::new(AtomicBool::new(false));
        let asr = {
            let app = app.clone();
            let transcript = Arc::clone(&transcript);
            let recorded_ms = Arc::clone(&recorded_ms);
            let update = |state, stage, progress, message| PolishUpdate { state, stage, progress, message };
            let both_sources = mic_buf.is_some() && sys_buf.is_some();
            let speakers = match settings.speakers.as_str() {
                SPEAKERS_VOICE if voice_model.is_file() => SpeakerMode::Voice(voice_model.to_path_buf()),
                SPEAKERS_VOICE | SPEAKERS_SOURCE if both_sources => SpeakerMode::Source,
                _ => SpeakerMode::Off,
            };
            let finished_quality = match (polish, matches!(speakers, SpeakerMode::Voice(_))) {
                (true, true) => "Polished after recording (beam search, full context); speakers identified by voice",
                (true, false) => "Polished after recording (beam search, full context)",
                (false, true) => "Live transcript; speakers identified by voice after recording",
                (false, false) => QUALITY_LIVE,
            };
            spawn_worker(
                models,
                WorkerConfig {
                    language: settings.language.clone(),
                    threads: settings.threads,
                    live_preview: settings.live_preview,
                    speakers,
                    expected_speakers: (settings.expected_speakers > 0).then_some(settings.expected_speakers as usize),
                    vocabulary: settings.vocabulary.clone(),
                    polish,
                    cancel: Arc::clone(&cancel_polish),
                },
                asr_rx,
                move |event| match event {
                    AsrEvent::Preview(p) => {
                        let _ = app.emit(EV_PREVIEW, p);
                    }
                    AsrEvent::Final(segments) => {
                        let _ = app.emit(EV_SEGMENTS, &segments);
                        let mut t = transcript.lock().unwrap();
                        t.segments.extend(segments);
                        t.meta.duration_secs = recorded_ms.load(Ordering::Relaxed) as f64 / 1000.0;
                        if let Err(e) = t.save() {
                            log::warn!("Transcript autosave failed: {e:#}");
                        }
                    }
                    AsrEvent::Error(msg) => {
                        log::error!("{msg}");
                        let _ = app.emit(EV_ERROR, msg);
                    }
                    AsrEvent::LiveDone { finishing } => {
                        let _ = live_done_tx.send(finishing);
                    }
                    AsrEvent::FinishProgress { stage, progress } => {
                        let stage = match stage {
                            FinishStage::Polish => "polish",
                            FinishStage::Speakers => "speakers",
                        };
                        let _ = app.emit(EV_POLISH, update("running", stage, progress, None));
                    }
                    AsrEvent::Finished(segments) => {
                        let mut t = transcript.lock().unwrap();
                        t.segments = segments;
                        t.meta.quality = Some(finished_quality.into());
                        let message = t.save().err().map(|e| format!("Could not save the finished transcript: {e:#}"));
                        log::info!("Transcript finished: {} segments", t.segments.len());
                        let _ = app.emit(EV_POLISHED, &t.segments);
                        let _ = app.emit(EV_POLISH, update("done", "", 1.0, message));
                    }
                    AsrEvent::FinishAbandoned(message) => {
                        let mut t = transcript.lock().unwrap();
                        t.meta.quality = Some(QUALITY_LIVE.into());
                        if let Err(e) = t.save() {
                            log::warn!("Transcript save failed: {e:#}");
                        }
                        log::info!("{message}");
                        let _ = app.emit(EV_POLISH, update("abandoned", "", 0.0, Some(message)));
                    }
                },
            )
        };

        // 4. Mixer: drives the MP3 encoder and feeds the worker.
        let paused = Arc::new(AtomicBool::new(false));
        let stop = Arc::new(AtomicBool::new(false));
        let muted: [Arc<AtomicBool>; 2] = Default::default();
        let mixer = {
            let app = app.clone();
            let recorded_ms = Arc::clone(&recorded_ms);
            spawn_mixer(
                mic_buf,
                sys_buf,
                mp3,
                asr_tx,
                MixerControl { paused: Arc::clone(&paused), stop: Arc::clone(&stop), muted: muted.clone() },
                move |levels| {
                    recorded_ms.store((levels.seconds * 1000.0) as u64, Ordering::Relaxed);
                    let _ = app.emit(EV_LEVELS, levels);
                },
            )
        };

        let info = SessionInfo {
            mp3_path: mp3_path.to_string_lossy().into_owned(),
            docx_path: docx_path.to_string_lossy().into_owned(),
            mic_device: report.mic_device,
            system_device: report.system_device,
            warnings: report.warnings,
        };
        log::info!("Recording started: {}", info.mp3_path);
        Ok(Self { info, paused, muted, stop, recorded_ms, capture_stop, capture, mixer, asr, transcript, live_done, cancel_polish })
    }

    pub fn set_paused(&self, paused: bool) {
        self.paused.store(paused, Ordering::SeqCst);
    }

    pub fn is_paused(&self) -> bool {
        self.paused.load(Ordering::SeqCst)
    }

    /// Leave a source out of the recording and transcript (or bring it back).
    pub fn set_muted(&self, source: Source, muted: bool) {
        self.muted[source as usize].store(muted, Ordering::SeqCst);
        log::info!("{source:?} {}", if muted { "muted" } else { "unmuted" });
    }

    pub fn is_muted(&self, source: Source) -> bool {
        self.muted[source as usize].load(Ordering::SeqCst)
    }

    /// Stop capture, finish the MP3, transcribe what is still buffered and
    /// write the document. Returns once the live transcript is on disk; if a
    /// polish pass follows, it keeps running in the returned `Polisher`.
    pub fn stop(self) -> Result<(SavedFiles, Option<Polisher>)> {
        let mut warnings = Vec::new();
        let _ = self.capture_stop.send(());
        let _ = self.capture.join();
        self.stop.store(true, Ordering::SeqCst);
        match self.mixer.join() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => warnings.push(format!("The MP3 file may be incomplete: {e:#}")),
            Err(_) => warnings.push("The recorder stopped unexpectedly; the MP3 may be incomplete.".into()),
        }
        // Wait for live transcription to catch up (it may lag on slow PCs).
        let polishing = self.live_done.recv_timeout(Duration::from_secs(1800)).unwrap_or(false);

        let mut t = self.transcript.lock().unwrap();
        t.meta.in_progress = false;
        t.meta.quality = Some(if polishing { QUALITY_FINISHING } else { QUALITY_LIVE }.into());
        t.meta.duration_secs = self.recorded_ms.load(Ordering::Relaxed) as f64 / 1000.0;
        if let Err(e) = t.save() {
            warnings.push(format!("Could not save the transcript: {e:#}"));
        }
        log::info!("Recording saved: {} ({} segments)", self.info.mp3_path, t.segments.len());
        let saved = SavedFiles {
            mp3_path: self.info.mp3_path.clone(),
            docx_path: self.info.docx_path.clone(),
            duration_secs: t.meta.duration_secs,
            segments: t.segments.len(),
            warnings,
            polishing,
        };
        drop(t);
        let polisher = if polishing {
            Some(Polisher { handle: self.asr, cancel: self.cancel_polish })
        } else {
            let _ = self.asr.join();
            None
        };
        Ok((saved, polisher))
    }
}

fn file_name(p: &Path) -> String {
    p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
}

/// `dir/stem`, with " (2)", " (3)"... appended if files already exist.
fn unique_base(dir: &Path, stem: &str) -> PathBuf {
    let taken = |b: &PathBuf| b.with_extension("mp3").exists() || b.with_extension("docx").exists();
    let mut base = dir.join(stem);
    let mut n = 2;
    while taken(&base) {
        base = dir.join(format!("{stem} ({n})"));
        n += 1;
    }
    base
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{play_on_speakers, test_speech};

    /// A full recording exactly as the Record / Pause / Resume / Stop buttons
    /// drive it, inside a mock Tauri app. Plays speech on the speakers.
    ///   KALAM_TEST_WAV=jfk.wav cargo test --release full_session -- --ignored --nocapture
    #[test]
    #[ignore]
    fn full_session_record_pause_resume_stop() {
        let app = tauri::test::mock_app();
        let dir = std::env::temp_dir().join("kalam-session-test");
        let _ = std::fs::remove_dir_all(&dir);
        let settings = Settings {
            output_dir: dir.to_string_lossy().into_owned(),
            polish: crate::settings::POLISH_ALWAYS.into(),
            ..Settings::default()
        };
        let models = crate::test_util::test_models();

        let voice_model = PathBuf::from("resources/speakers/titanet_small.onnx");
        let session = Session::start(app.handle(), &settings, models, &voice_model).expect("session starts");
        println!("started: {:?}", session.info);
        play_on_speakers(&test_speech(0.35));
        session.set_paused(true);
        std::thread::sleep(Duration::from_millis(1500));
        session.set_paused(false);
        std::thread::sleep(Duration::from_millis(1000));
        let (saved, polisher) = session.stop().expect("session stops");
        println!("saved: {saved:?}");
        let live = docx_rs::read_docx(&std::fs::read(&saved.docx_path).unwrap()).unwrap().json();
        assert!(live.contains("being finished"), "live transcript saved before finishing");
        let started = std::time::Instant::now();
        polisher.expect("polish pass runs").wait();
        println!("polished in {:.1}s", started.elapsed().as_secs_f32());

        assert!(saved.warnings.is_empty(), "{:?}", saved.warnings);
        assert!(std::fs::metadata(&saved.mp3_path).unwrap().len() > 100_000);
        // Paused time is excluded: ~11 s speech + ~1.3 s before pause + ~1 s after.
        assert!(saved.duration_secs > 11.0 && saved.duration_secs < 15.5, "{}", saved.duration_secs);
        let doc = docx_rs::read_docx(&std::fs::read(&saved.docx_path).unwrap()).unwrap().json();
        assert!(doc.contains("ask what you can do for your country") || doc.contains("Ask what you can do for your country"), "docx text missing");
        assert!(doc.contains("Speaker 1"), "speaker label missing");
        assert!(!doc.contains("Recording in progress"), "final save happened");
        assert!(doc.contains("Polished after recording"), "polished transcript saved");
    }
}
