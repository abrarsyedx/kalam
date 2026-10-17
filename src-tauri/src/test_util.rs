//! Helpers for the hardware tests (all `#[ignore]`d; see README "Tests").

use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use crate::audio::resample::Resampler;
use crate::transcribe::{ModelManager, ASR_RATE, HEAVY_MODEL_BYTES};

/// Speech from the WAV named by `KALAM_TEST_WAV` as 16 kHz mono, scaled by `gain`.
pub fn test_speech(gain: f32) -> Vec<f32> {
    test_audio().into_iter().map(|s| s * gain).collect()
}

/// Play 16 kHz mono audio on the default output device and block until done.
pub fn play_on_speakers(speech: &[f32]) {
    let host = cpal::default_host();
    let out = host.default_output_device().expect("an output device");
    let cfg = out.default_output_config().unwrap();
    assert_eq!(cfg.sample_format(), cpal::SampleFormat::F32, "test assumes f32 output");
    let config = cfg.config();
    let mut up = Vec::new();
    Resampler::new(ASR_RATE, config.sample_rate).process_into(speech, &mut up);
    let channels = config.channels as usize;
    let mut pos = 0usize;
    let player = out
        .build_output_stream::<f32, _, _>(
            config,
            move |data: &mut [f32], _| {
                for frame in data.chunks_mut(channels) {
                    let s = up.get(pos).copied().unwrap_or(0.0);
                    pos += 1;
                    frame.iter_mut().for_each(|v| *v = s);
                }
            },
            |e| eprintln!("playback error: {e}"),
            None,
        )
        .unwrap();
    player.play().unwrap();
    std::thread::sleep(Duration::from_millis(speech.len() as u64 * 1000 / ASR_RATE as u64 + 300));
}

/// The benchmark audio named by KALAM_TEST_WAV (a .wav or .mp3), as 16 kHz
/// mono. KALAM_TEST_START and KALAM_TEST_SECONDS cut an excerpt (seconds).
pub fn test_audio() -> Vec<f32> {
    let path = std::env::var("KALAM_TEST_WAV").expect("set KALAM_TEST_WAV to a speech recording (.wav or .mp3)");
    let audio = load_audio_16k(&path);
    let secs = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<f64>().ok());
    let start = (secs("KALAM_TEST_START").unwrap_or(0.0) * ASR_RATE as f64) as usize;
    let end = secs("KALAM_TEST_SECONDS").map_or(audio.len(), |s| start + (s * ASR_RATE as f64) as usize);
    audio[start.min(audio.len())..end.min(audio.len())].to_vec()
}

/// Load a .wav or .mp3 file as 16 kHz mono.
pub fn load_audio_16k(path: &str) -> Vec<f32> {
    if path.to_lowercase().ends_with(".mp3") {
        load_mp3_16k(path)
    } else {
        load_wav_16k(path)
    }
}

fn load_mp3_16k(path: &str) -> Vec<f32> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let file = std::fs::File::open(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    hint.with_extension("mp3");
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .expect("readable mp3");
    let mut format = probed.format;
    let track = format.default_track().expect("an audio track").clone();
    let mut decoder = symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default()).unwrap();
    let rate = track.codec_params.sample_rate.unwrap_or(44_100);
    let mut mono = Vec::new();
    while let Ok(packet) = format.next_packet() {
        if packet.track_id() != track.id {
            continue;
        }
        let Ok(decoded) = decoder.decode(&packet) else { continue };
        let spec = *decoded.spec();
        let mut buf = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
        buf.copy_interleaved_ref(decoded);
        let ch = spec.channels.count().max(1);
        mono.extend(buf.samples().chunks(ch).map(|f| f.iter().sum::<f32>() / f.len() as f32));
    }
    let mut out = Vec::new();
    Resampler::new(rate, ASR_RATE).process_into(&mono, &mut out);
    out
}

/// Load any PCM/float WAV (any rate, any channel count) as 16 kHz mono.
pub fn load_wav_16k(path: &str) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path).unwrap();
    let spec = reader.spec();
    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1u64 << (spec.bits_per_sample - 1)) as f32;
            reader.samples::<i32>().map(|s| s.unwrap() as f32 * scale).collect()
        }
    };
    let ch = spec.channels.max(1) as usize;
    let mono: Vec<f32> = interleaved.chunks(ch).map(|f| f.iter().sum::<f32>() / f.len() as f32).collect();
    let mut out = Vec::new();
    Resampler::new(spec.sample_rate, ASR_RATE).process_into(&mono, &mut out);
    out
}

pub const FAST_MODEL: &str = "resources/models/ggml-small.en-q5_1.bin";

/// Models for the hardware tests, selected the way the app does it:
/// KALAM_TEST_MODEL (default: the fast model); a heavy model gets the fast
/// one as its live-preview companion.
pub fn test_models() -> std::sync::Arc<ModelManager> {
    let primary = std::env::var("KALAM_TEST_MODEL").unwrap_or_else(|_| FAST_MODEL.into());
    let heavy = std::fs::metadata(&primary).map(|m| m.len() > HEAVY_MODEL_BYTES).unwrap_or(false);
    let models = ModelManager::new(|s| println!("model: {s:?}"));
    models.set_gpu(test_gpu(), None);
    models.select(primary.into(), heavy.then(|| FAST_MODEL.into()));
    models
}

/// KALAM_TEST_GPU=1 runs the hardware tests on the graphics card.
pub fn test_gpu() -> bool {
    std::env::var("KALAM_TEST_GPU").is_ok_and(|v| v == "1")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cut benchmark excerpts from a long recording: prints how much speech
    /// each minute holds, then writes the most speech-dense 1-minute and
    /// 5-minute stretches as 16 kHz WAV files.
    ///   KALAM_TEST_WAV=meeting.mp3 KALAM_CLIP_DIR=clips cargo test --release extract_clips -- --ignored --nocapture
    #[test]
    #[ignore]
    fn extract_clips() {
        let path = std::env::var("KALAM_TEST_WAV").expect("set KALAM_TEST_WAV");
        let out_dir = std::path::PathBuf::from(std::env::var("KALAM_CLIP_DIR").expect("set KALAM_CLIP_DIR"));
        let audio = load_audio_16k(&path);
        let rate = ASR_RATE as usize;
        println!("{path}: {:.1} min", audio.len() as f64 / rate as f64 / 60.0);
        // Speech activity per second: share of 20 ms frames clearly above silence.
        let active: Vec<f32> = audio
            .chunks(rate)
            .map(|sec| {
                let frames: Vec<f32> = sec.chunks(320).map(|f| (f.iter().map(|s| s * s).sum::<f32>() / f.len() as f32).sqrt()).collect();
                frames.iter().filter(|r| **r > 0.01).count() as f32 / frames.len().max(1) as f32
            })
            .collect();
        for (m, minute) in active.chunks(60).enumerate() {
            let pct = minute.iter().sum::<f32>() / minute.len() as f32 * 100.0;
            println!("minute {m:>2}: {pct:>5.1}% speech {}", "#".repeat((pct / 4.0) as usize));
        }
        let spec = hound::WavSpec { channels: 1, sample_rate: ASR_RATE, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
        for (name, secs) in [("test-1min.wav", 60usize), ("test-5min.wav", 300)] {
            let best = (0..active.len().saturating_sub(secs))
                .max_by(|a, b| active[*a..*a + secs].iter().sum::<f32>().total_cmp(&active[*b..*b + secs].iter().sum::<f32>()))
                .unwrap_or(0);
            let clip = &audio[best * rate..((best + secs) * rate).min(audio.len())];
            let file = out_dir.join(name);
            let mut w = hound::WavWriter::create(&file, spec).unwrap();
            for s in clip {
                w.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16).unwrap();
            }
            w.finalize().unwrap();
            println!("{name}: starts at {:02}:{:02}, {secs} s -> {}", best / 60, best % 60, file.display());
        }
    }
}
