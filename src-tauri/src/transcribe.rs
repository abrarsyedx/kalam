//! Offline, realtime speech-to-text with Whisper (whisper.cpp).
//!
//! Whisper works on whole utterances, so the worker segments the incoming
//! 16 kHz stream with a lightweight energy voice-activity detector:
//!
//! * while someone is speaking, the growing utterance is re-transcribed about
//!   once a second and shown as a live *preview* (cheap: reduced audio context);
//! * when they pause (~0.7 s of silence) or the utterance reaches 15 s, it is
//!   transcribed properly and *committed* as final transcript segments.
//!
//! Silence is never sent to Whisper, which avoids its classic "Thank you."
//! hallucinations on empty audio and keeps CPU use low.
//!
//! After the recording stops, an optional *polish* pass re-transcribes the
//! session's speech in windows of up to 28 s with beam search and the full
//! encoder context, joining sentences the live pass had to cut, and replaces
//! the live transcript with the result.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use serde::Serialize;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState};

use crate::speakers::{self, LiveSpeakers, Voiceprints};

pub const ASR_RATE: u32 = 16_000;
const FRAME: usize = 320; // 20 ms
const FRAME_MS: u64 = 20;
const COMMIT_SILENCE_MS: u64 = 700;
const MIN_UTTERANCE_MS: u64 = 1_000;
const MAX_UTTERANCE_MS: u64 = 15_000;
const MIN_SPEECH_MS: u64 = 240;
const PREVIEW_EVERY: Duration = Duration::from_millis(900);
const PREVIEW_MIN_NEW_MS: u64 = 600;
const PROMPT_CHARS: usize = 220;
/// Names & terms beyond this are ignored, so the prompt stays within
/// Whisper's 224-token limit together with the recent context.
pub const VOCABULARY_CHARS: usize = 400;
/// Polish windows: up to this long, split where the speaker paused longer
/// than POLISH_MAX_GAP_MS.
const POLISH_WINDOW_MS: u64 = 28_000;
const POLISH_MAX_GAP_MS: u64 = 3_000;
const POLISH_BEAM: i32 = 3;

// ---------------------------------------------------------------- model ----

/// Model files larger than this (Whisper medium / large) are "heavy": they get
/// more threads, a trimmed encoder window, and a light companion model for the
/// live preview so the screen stays responsive.
pub const HEAVY_MODEL_BYTES: u64 = 300_000_000;

/// Serialises model loading and all GPU work. whisper.cpp's GPU backends are
/// not safe to set up from two threads at once (loading the preview and main
/// models in parallel crashed the Vulkan backend).
static GPU_LOCK: Mutex<()> = Mutex::new(());

fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
    GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

pub struct Model {
    pub ctx: WhisperContext,
    pub name: String,
    pub english_only: bool,
    pub heavy: bool,
    pub on_gpu: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "state", rename_all = "camelCase")]
pub enum ModelStatus {
    Loading { name: String },
    Ready { name: String, english_only: bool, load_ms: u64, on_gpu: bool },
    Failed { name: String, error: String },
}

struct Slot {
    result: Option<Result<Arc<Model>, String>>,
    status: ModelStatus,
}

#[derive(Default)]
struct Active {
    slots: HashMap<PathBuf, Slot>,
    primary: Option<PathBuf>,
    preview: Option<PathBuf>,
}

/// Loads Whisper models in the background and hands them to sessions. The
/// *primary* model writes the transcript; an optional lighter *preview* model
/// writes the live preview while a heavy primary model is selected.
pub struct ModelManager {
    active: Mutex<Active>,
    ready: Condvar,
    notify: Box<dyn Fn(ModelStatus) + Send + Sync>,
    gpu: Mutex<GpuUse>,
}

#[derive(Clone, Default, PartialEq)]
struct GpuUse {
    enabled: bool,
    /// Crash marker written while a model is loaded and warmed up on the GPU.
    marker: Option<PathBuf>,
}

impl ModelManager {
    /// `notify` is called whenever the primary model changes state.
    pub fn new(notify: impl Fn(ModelStatus) + Send + Sync + 'static) -> Arc<Self> {
        Arc::new(Self {
            active: Mutex::default(),
            ready: Condvar::new(),
            notify: Box::new(notify),
            gpu: Mutex::default(),
        })
    }

    /// Whether models are (to be) loaded on the GPU.
    pub fn gpu_enabled(&self) -> bool {
        self.gpu.lock().unwrap().enabled
    }

    /// Run models on the GPU (or not). Loaded models are dropped when this
    /// changes, so the next `select` reloads them on the right device.
    pub fn set_gpu(&self, enabled: bool, marker: Option<PathBuf>) {
        let next = GpuUse { enabled, marker };
        let mut gpu = self.gpu.lock().unwrap();
        if gpu.enabled != next.enabled {
            self.active.lock().unwrap().slots.clear();
        }
        *gpu = next;
    }

    /// Status of the primary model.
    pub fn status(&self) -> ModelStatus {
        let a = self.active.lock().unwrap();
        a.primary
            .as_ref()
            .and_then(|p| a.slots.get(p))
            .map(|s| s.status.clone())
            .unwrap_or(ModelStatus::Loading { name: String::new() })
    }

    /// Make `primary` (and optionally `preview`) the active models: load the
    /// ones not yet in memory and unload all others.
    pub fn select(self: &Arc<Self>, primary: PathBuf, preview: Option<PathBuf>) {
        let preview = preview.filter(|p| *p != primary);
        let (to_load, status) = {
            let mut a = self.active.lock().unwrap();
            let wanted: Vec<PathBuf> = std::iter::once(primary.clone()).chain(preview.clone()).collect();
            a.slots.retain(|p, _| wanted.contains(p));
            let mut to_load = Vec::new();
            for p in &wanted {
                if !a.slots.contains_key(p) {
                    a.slots.insert(p.clone(), Slot { result: None, status: ModelStatus::Loading { name: model_name(p) } });
                    to_load.push(p.clone());
                }
            }
            a.primary = Some(primary.clone());
            a.preview = preview;
            (to_load, a.slots[&primary].status.clone())
        };
        (self.notify)(status);
        for path in to_load {
            self.spawn_load(path);
        }
    }

    fn spawn_load(self: &Arc<Self>, path: PathBuf) {
        let me = Arc::clone(self);
        let gpu = self.gpu.lock().unwrap().clone();
        std::thread::Builder::new()
            .name("kalam-model-load".into())
            .spawn(move || {
                let name = model_name(&path);
                let started = Instant::now();
                let result = load_model(&path, &name, &gpu).map(Arc::new);
                let status = match &result {
                    Ok(m) => ModelStatus::Ready {
                        name: m.name.clone(),
                        english_only: m.english_only,
                        load_ms: started.elapsed().as_millis() as u64,
                        on_gpu: m.on_gpu,
                    },
                    Err(e) => ModelStatus::Failed { name, error: e.clone() },
                };
                let is_primary = {
                    let mut a = me.active.lock().unwrap();
                    match a.slots.get_mut(&path) {
                        Some(slot) if slot.result.is_none() => {
                            slot.result = Some(result);
                            slot.status = status.clone();
                        }
                        _ => return, // unloaded (or loaded again) in the meantime
                    }
                    a.primary.as_ref() == Some(&path)
                };
                me.ready.notify_all();
                if is_primary {
                    (me.notify)(status);
                }
            })
            .expect("spawn model loader");
    }

    /// Block until the active models are loaded. Returns the primary model
    /// and, if one is selected and loaded fine, the preview model.
    pub fn wait(&self, timeout: Duration) -> Result<(Arc<Model>, Option<Arc<Model>>), String> {
        let pending = |a: &Active, p: &Option<PathBuf>| {
            p.as_ref().and_then(|p| a.slots.get(p)).is_some_and(|s| s.result.is_none())
        };
        let a = self.active.lock().unwrap();
        let (a, _) = self
            .ready
            .wait_timeout_while(a, timeout, |a| a.primary.is_none() || pending(a, &a.primary) || pending(a, &a.preview))
            .unwrap();
        let result = |p: &Option<PathBuf>| p.as_ref().and_then(|p| a.slots.get(p)).and_then(|s| s.result.clone());
        let primary = result(&a.primary).unwrap_or_else(|| Err("Timed out waiting for the speech model to load".into()))?;
        let preview = result(&a.preview).and_then(|r| r.ok());
        Ok((primary, preview))
    }
}

/// Threads to use when the user leaves it on "Automatic". Measured on a
/// 16-core Core Ultra 7: small models peak at 6 threads, large ones at 12;
/// beyond that the slow efficiency cores hold everything back.
pub fn auto_threads(heavy: bool) -> i32 {
    let n = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    if heavy { (n * 55 / 100).clamp(2, 12) as i32 } else { (n / 2).clamp(2, 6) as i32 }
}

fn model_name(path: &Path) -> String {
    let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    stem.strip_prefix("ggml-").unwrap_or(&stem).to_string()
}

fn load_model(path: &Path, name: &str, gpu: &GpuUse) -> Result<Model, String> {
    let _gpu = gpu_lock();
    let size = std::fs::metadata(path).map_err(|_| format!("Model file not found: {}", path.display()))?.len();
    let mut params = WhisperContextParameters::default();
    params.use_gpu(gpu.enabled);
    if let Some(marker) = gpu.marker.as_ref().filter(|_| gpu.enabled) {
        let _ = std::fs::write(marker, name);
    }
    let ctx = WhisperContext::new_with_params(path, params)
        .map_err(|e| format!("Could not load Whisper model {}: {e}", path.display()))?;
    if gpu.enabled {
        // Warm-up: compiles the GPU shaders now rather than on the first
        // sentence of a meeting, and proves the driver copes with this model.
        let warmup = ctx.create_state().and_then(|mut state| {
            let mut p = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
            p.set_print_progress(false);
            p.set_print_realtime(false);
            p.set_max_tokens(4);
            state.full(p, &vec![0.0; ASR_RATE as usize * 2])
        });
        if let Err(e) = warmup {
            log::warn!("GPU warm-up of {name} failed: {e}");
        }
    }
    if let Some(marker) = &gpu.marker {
        let _ = std::fs::remove_file(marker);
    }
    let english_only = name.contains(".en");
    let heavy = size > HEAVY_MODEL_BYTES;
    log::info!("Loaded Whisper model {name} (english_only={english_only}, heavy={heavy}, gpu={})", gpu.enabled);
    Ok(Model { ctx, name: name.to_string(), english_only, heavy, on_gpu: gpu.enabled })
}

// --------------------------------------------------------------- worker ----

pub enum AsrMsg {
    /// 16 kHz mono audio plus the RMS level of each source during this block.
    Audio { samples: Vec<f32>, mic_rms: f32, system_rms: f32 },
    /// Commit whatever is buffered (used on pause).
    Flush,
    /// Commit and exit.
    Stop,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Segment {
    pub start_ms: u64,
    pub end_ms: u64,
    pub speaker: Option<String>,
    pub text: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Preview {
    pub start_ms: u64,
    pub text: String,
}

pub enum AsrEvent {
    Preview(Option<Preview>),
    Final(Vec<Segment>),
    Error(String),
    /// Live transcription has caught up with the end of the recording.
    /// `finishing` says whether a finishing pass (polish and/or speaker
    /// identification) follows.
    LiveDone { finishing: bool },
    /// Finishing progress, 0.0 to 1.0, per stage.
    FinishProgress { stage: FinishStage, progress: f32 },
    /// The finished transcript, replacing the live one.
    Finished(Vec<Segment>),
    /// Finishing was skipped, cancelled or failed; the live transcript stands.
    FinishAbandoned(String),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FinishStage {
    /// Re-transcribing the recording carefully.
    Polish,
    /// Telling the voices apart across the whole recording.
    Speakers,
}

/// How transcript lines are labelled.
#[derive(Debug, Clone, PartialEq)]
pub enum SpeakerMode {
    Off,
    /// "You" (microphone) / "Others" (computer audio), by loudness.
    Source,
    /// "Speaker 1", "Speaker 2", ... by voice, using this voiceprint model.
    Voice(PathBuf),
}

#[derive(Clone)]
pub struct WorkerConfig {
    pub language: String,
    /// Inference threads; 0 = automatic (depends on the model size).
    pub threads: u32,
    pub live_preview: bool,
    pub speakers: SpeakerMode,
    /// How many people spoke, if the user knows (speaker labels by voice).
    pub expected_speakers: Option<usize>,
    /// Names, companies and jargon to steer Whisper's spelling.
    pub vocabulary: String,
    /// Re-transcribe the whole session carefully after it stops.
    pub polish: bool,
    /// Set to stop a running polish pass.
    pub cancel: Arc<AtomicBool>,
}

#[cfg(test)]
impl WorkerConfig {
    /// Settings for tests. KALAM_TEST_THREADS, KALAM_TEST_POLISH=1 and
    /// KALAM_TEST_VOCAB ("a, b") override the defaults.
    /// KALAM_TEST_SPEAKERS=voice labels by voice (Speaker 1, 2...);
    /// KALAM_TEST_EXPECTED_SPEAKERS=n fixes the count.
    pub fn for_test(live_preview: bool, speaker_labels: bool) -> Self {
        let env = |k: &str| std::env::var(k).ok();
        let speakers = if env("KALAM_TEST_SPEAKERS").as_deref() == Some("voice") {
            SpeakerMode::Voice(PathBuf::from("resources/speakers/titanet_small.onnx"))
        } else if speaker_labels {
            SpeakerMode::Source
        } else {
            SpeakerMode::Off
        };
        Self {
            language: "en".into(),
            threads: env("KALAM_TEST_THREADS").and_then(|t| t.parse().ok()).unwrap_or(0),
            live_preview,
            speakers,
            expected_speakers: env("KALAM_TEST_EXPECTED_SPEAKERS").and_then(|v| v.parse().ok()),
            vocabulary: env("KALAM_TEST_VOCAB").unwrap_or_default(),
            polish: env("KALAM_TEST_POLISH").is_some_and(|v| v == "1"),
            cancel: Arc::new(AtomicBool::new(false)),
        }
    }
}

pub fn spawn_worker(
    models: Arc<ModelManager>,
    cfg: WorkerConfig,
    rx: Receiver<AsrMsg>,
    mut emit: impl FnMut(AsrEvent) + Send + 'static,
) -> std::thread::JoinHandle<()> {
    std::thread::Builder::new()
        .name("kalam-asr".into())
        .spawn(move || {
            let engines = models.wait(Duration::from_secs(600)).and_then(|(primary, preview)| {
                let main = Engine::new(primary, &cfg).map_err(|e| format!("{e:#}"))?;
                // A light preview model is optional: without it the main model previews.
                // An English-only preview model is no use for other languages.
                let preview = preview
                    .filter(|m| !m.english_only || cfg.language == "en")
                    .and_then(|m| Engine::new(m, &cfg).ok());
                Ok((main, preview))
            });
            match engines {
                Ok((engine, preview)) => Worker::new(engine, preview, cfg).run(&rx, &mut emit),
                Err(e) => {
                    emit(AsrEvent::Error(format!("Transcription disabled: {e}")));
                    // Keep draining so the recorder never blocks.
                    while !matches!(rx.recv(), Ok(AsrMsg::Stop) | Err(_)) {}
                    emit(AsrEvent::LiveDone { finishing: false });
                }
            }
        })
        .expect("spawn ASR worker")
}

struct Engine {
    /// Keeps the model alive for as long as its state is in use.
    _model: Arc<Model>,
    on_gpu: bool,
    state: WhisperState,
    language: String,
    threads: i32,
    /// Also shorten the encoder context for final passes (not just previews).
    trim_ctx: bool,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Pass {
    /// Quick, cheap pass for the live preview of the sentence being spoken.
    Preview,
    /// The committed transcript.
    Final,
    /// Careful re-transcription after the recording: beam search, full context.
    Polish,
}

struct RawSegment {
    t0_ms: u64,
    t1_ms: u64,
    text: String,
    no_speech: f32,
}

impl Engine {
    fn new(model: Arc<Model>, cfg: &WorkerConfig) -> Result<Self> {
        let state = model.ctx.create_state().map_err(|e| anyhow!("whisper state: {e}"))?;
        let language = if model.english_only { "en".to_string() } else { cfg.language.clone() };
        let threads = if cfg.threads > 0 { cfg.threads as i32 } else { auto_threads(model.heavy) };
        log::info!("{}: {threads} threads", model.name);
        Ok(Self { trim_ctx: model.heavy, on_gpu: model.on_gpu, _model: model, state, language, threads })
    }

    fn transcribe(&mut self, audio: &[f32], prompt: &str, pass: Pass) -> Result<Vec<RawSegment>> {
        let preview = pass == Pass::Preview;
        let mut p = match pass {
            Pass::Preview => FullParams::new(SamplingStrategy::Greedy { best_of: 1 }),
            Pass::Final => FullParams::new(SamplingStrategy::Greedy { best_of: 5 }),
            Pass::Polish => FullParams::new(SamplingStrategy::BeamSearch { beam_size: POLISH_BEAM, patience: -1.0 }),
        };
        p.set_n_threads(self.threads);
        p.set_translate(false);
        p.set_language(Some(if self.language == "auto" { "auto" } else { &self.language }));
        p.set_no_context(true);
        p.set_single_segment(preview);
        p.set_print_special(false);
        p.set_print_progress(false);
        p.set_print_realtime(false);
        p.set_print_timestamps(false);
        p.set_suppress_blank(true);
        p.set_suppress_nst(true);
        p.set_no_speech_thold(0.6);
        let prompt: String = prompt.chars().filter(|c| *c != '\0').collect();
        if !prompt.is_empty() {
            p.set_initial_prompt(&prompt);
        }
        if preview || (self.trim_ctx && pass == Pass::Final) {
            // Encode only as much context as the audio needs (50 frames/s)
            // instead of a full 30 s window: the encoder cost drops in
            // proportion, which is what makes large models realtime on a CPU.
            let secs = audio.len() as f32 / ASR_RATE as f32;
            p.set_audio_ctx(((secs * 50.0) as i32 + 128).clamp(512, 1500));
        }
        if preview {
            p.set_temperature_inc(0.0);
        }

        // whisper.cpp ignores clips shorter than 1 s; pad with silence.
        let min_len = ASR_RATE as usize * 11 / 10;
        let padded;
        let input = if audio.len() < min_len {
            let mut v = audio.to_vec();
            v.resize(min_len, 0.0);
            padded = v;
            &padded[..]
        } else {
            audio
        };

        {
            let _gpu = self.on_gpu.then(gpu_lock);
            self.state.full(p, input).map_err(|e| anyhow!("whisper: {e}"))?;
        }
        let audio_ms = audio.len() as u64 * 1000 / ASR_RATE as u64;
        let mut out = Vec::new();
        for seg in self.state.as_iter() {
            let text = seg.to_str_lossy().map(|t| t.into_owned()).unwrap_or_default();
            let t0 = (seg.start_timestamp().max(0) as u64 * 10).min(audio_ms);
            let t1 = (seg.end_timestamp().max(0) as u64 * 10).clamp(t0, audio_ms);
            out.push(RawSegment { t0_ms: t0, t1_ms: t1, text, no_speech: seg.no_speech_probability() });
        }
        Ok(out)
    }
}

#[derive(Clone, Copy)]
struct EnergyBlock {
    start: u64,
    len: u64,
    mic: f32,
    system: f32,
}

/// "You" if the microphone dominated this time span, "Others" if the
/// computer's audio did.
fn label_speaker(energy: &[EnergyBlock], start_ms: u64, end_ms: u64) -> Option<String> {
    let a = start_ms * ASR_RATE as u64 / 1000;
    let b = (end_ms * ASR_RATE as u64 / 1000).max(a + 1);
    let (mut mic, mut sys) = (0f64, 0f64);
    for blk in energy.iter().filter(|blk| blk.start < b && blk.start + blk.len > a) {
        let overlap = (blk.start + blk.len).min(b).saturating_sub(blk.start.max(a)) as f64;
        mic += blk.mic as f64 * blk.mic as f64 * overlap;
        sys += blk.system as f64 * blk.system as f64 * overlap;
    }
    if mic == 0.0 && sys == 0.0 {
        return None;
    }
    // Speakers leak into the mic, so require the mic to clearly dominate.
    Some(if sys.sqrt() > mic.sqrt() * 0.6 { "Others" } else { "You" }.to_string())
}

struct Worker {
    engine: Engine,
    /// Lighter model for the live preview (used when `engine` is heavy).
    preview_engine: Option<Engine>,
    /// How long the last preview took; previews are spaced out accordingly.
    preview_cost: Duration,
    cfg: WorkerConfig,
    /// Current utterance audio and the absolute sample index of `buf[0]`.
    buf: Vec<f32>,
    buf_start: u64,
    total: u64,
    /// Per-20 ms-frame speech flags for `buf`.
    speech: Vec<bool>,
    noise_floor: f32,
    energy: Vec<EnergyBlock>,
    context: String,
    last_preview: Instant,
    last_preview_len: usize,
    preview_shown: bool,
    /// For the polish pass: the whole session's audio (16-bit), its energy
    /// blocks, and the sample range of every committed utterance.
    history: Vec<i16>,
    all_energy: Vec<EnergyBlock>,
    spans: Vec<(u64, u64)>,
    live_chars: usize,
    /// Everything committed live (the finishing pass starts from this).
    live_segments: Vec<Segment>,
    /// Voiceprint model and live speaker grouping (speaker labels by voice).
    voice: Option<Voiceprints>,
    live_speakers: LiveSpeakers,
    last_speaker: Option<usize>,
}

impl Worker {
    fn new(engine: Engine, preview_engine: Option<Engine>, cfg: WorkerConfig) -> Self {
        let voice = match &cfg.speakers {
            SpeakerMode::Voice(model) => {
                let v = Voiceprints::load(model, 2);
                if v.is_none() {
                    log::warn!("Voiceprint model {} unavailable; no speaker labels", model.display());
                }
                v
            }
            _ => None,
        };
        let expected_speakers = cfg.expected_speakers;
        Self {
            engine,
            preview_engine,
            preview_cost: Duration::ZERO,
            cfg,
            buf: Vec::new(),
            buf_start: 0,
            total: 0,
            speech: Vec::new(),
            noise_floor: 0.002,
            energy: Vec::new(),
            context: String::new(),
            last_preview: Instant::now(),
            last_preview_len: 0,
            preview_shown: false,
            history: Vec::new(),
            all_energy: Vec::new(),
            spans: Vec::new(),
            live_chars: 0,
            live_segments: Vec::new(),
            voice,
            live_speakers: LiveSpeakers::new(expected_speakers),
            last_speaker: None,
        }
    }

    fn run(mut self, rx: &Receiver<AsrMsg>, emit: &mut dyn FnMut(AsrEvent)) {
        loop {
            let mut flush = false;
            let mut stop = false;
            // Wait briefly for audio, then drain everything already queued.
            let mut next = match rx.recv_timeout(Duration::from_millis(120)) {
                Ok(m) => Some(m),
                Err(RecvTimeoutError::Timeout) => None,
                Err(RecvTimeoutError::Disconnected) => Some(AsrMsg::Stop),
            };
            while let Some(msg) = next {
                match msg {
                    AsrMsg::Audio { samples, mic_rms, system_rms } => self.push(samples, mic_rms, system_rms),
                    AsrMsg::Flush => flush = true,
                    AsrMsg::Stop => stop = true,
                }
                next = rx.try_recv().ok();
            }

            if flush || stop {
                self.commit(self.buf.len(), emit);
                if stop {
                    let finishing = !self.spans.is_empty() && (self.cfg.polish || self.voice.is_some());
                    emit(AsrEvent::LiveDone { finishing });
                    if finishing {
                        self.finish(emit);
                    }
                    return;
                }
                continue;
            }
            self.step(emit);
        }
    }

    fn push(&mut self, samples: Vec<f32>, mic: f32, system: f32) {
        let block = EnergyBlock { start: self.total, len: samples.len() as u64, mic, system };
        self.energy.push(block);
        if self.cfg.polish || self.voice.is_some() {
            self.all_energy.push(block);
            self.history.extend(samples.iter().map(|s| (s.clamp(-1.0, 1.0) * 32767.0) as i16));
        }
        self.total += samples.len() as u64;
        self.buf.extend_from_slice(&samples);
        // Classify any newly completed frames.
        while (self.speech.len() + 1) * FRAME <= self.buf.len() {
            let i = self.speech.len() * FRAME;
            let frame = &self.buf[i..i + FRAME];
            let rms = (frame.iter().map(|s| s * s).sum::<f32>() / FRAME as f32).sqrt();
            // Speech = clearly above the background noise, or simply loud. The
            // "loud" rule and the low cap on the floor stop non-stop fast
            // speech from dragging the floor up until speech counts as silence.
            let is_speech = rms > (self.noise_floor * 3.0).max(0.004) || rms > 0.03;
            let rate = if is_speech { 0.0005 } else { 0.05 };
            self.noise_floor = (self.noise_floor * (1.0 - rate) + rms * rate).clamp(0.0003, 0.02);
            self.speech.push(is_speech);
        }
    }

    fn ms(samples: usize) -> u64 {
        samples as u64 * 1000 / ASR_RATE as u64
    }

    fn step(&mut self, emit: &mut dyn FnMut(AsrEvent)) {
        let len_ms = Self::ms(self.buf.len());
        let speech_ms = self.speech.iter().filter(|s| **s).count() as u64 * FRAME_MS;
        let trailing_ms = self.speech.iter().rev().take_while(|s| !**s).count() as u64 * FRAME_MS;

        if speech_ms < MIN_SPEECH_MS {
            // Only silence/noise so far: keep a short pre-roll, drop the rest.
            if len_ms > 3_000 {
                self.discard(self.buf.len() - FRAME * 25); // keep 500 ms
            }
            return;
        }
        if trailing_ms >= COMMIT_SILENCE_MS && len_ms >= MIN_UTTERANCE_MS {
            // Cut inside the pause, at its quietest point: the "silent" tail can
            // already hold the soft onset of the next word, which must not be split.
            let frames = self.speech.len();
            let pause_start = frames - (trailing_ms / FRAME_MS) as usize;
            let cut_frame = self.quietest_frame(pause_start + 2, frames - 4);
            self.commit(cut_frame * FRAME, emit);
        } else if len_ms >= MAX_UTTERANCE_MS {
            // Cut at the quietest frame of the last 4 s so words aren't split.
            let frames = self.speech.len();
            let cut_frame = self.quietest_frame(frames.saturating_sub(200), frames);
            self.commit((cut_frame.max(1)) * FRAME, emit);
        } else if self.cfg.live_preview
            && trailing_ms < 300 // a pause has begun: the final text is moments away
            // Spend at most about half of the time on previews.
            && self.last_preview.elapsed() >= PREVIEW_EVERY.max(self.preview_cost * 2)
            && Self::ms(self.buf.len().saturating_sub(self.last_preview_len)) >= PREVIEW_MIN_NEW_MS
        {
            self.preview(emit);
        }
    }

    fn frame_energy(&self, f: usize) -> f32 {
        let frame = &self.buf[f * FRAME..(f + 1) * FRAME];
        frame.iter().map(|s| s * s).sum::<f32>()
    }

    /// Frame index in `from..to` with the least energy (ties: the earliest).
    fn quietest_frame(&self, from: usize, to: usize) -> usize {
        let to = to.min(self.speech.len());
        if from >= to {
            return to;
        }
        (from..to)
            .min_by(|a, b| self.frame_energy(*a).total_cmp(&self.frame_energy(*b)))
            .unwrap_or(to)
    }

    fn preview(&mut self, emit: &mut dyn FnMut(AsrEvent)) {
        self.last_preview = Instant::now();
        self.last_preview_len = self.buf.len();
        let audio = std::mem::take(&mut self.buf);
        let engine = self.preview_engine.as_mut().unwrap_or(&mut self.engine);
        let prompt = build_prompt(&self.cfg.vocabulary, &self.context);
        let result = engine.transcribe(&audio, &prompt, Pass::Preview);
        self.preview_cost = self.last_preview.elapsed();
        self.buf = audio;
        if let Ok(segs) = result {
            let text = join_clean(segs.iter().filter(|s| s.no_speech < 0.8).map(|s| s.text.as_str()));
            if !text.is_empty() {
                self.preview_shown = true;
                emit(AsrEvent::Preview(Some(Preview { start_ms: self.abs_ms(0), text })));
            }
        }
    }

    /// Transcribe `buf[..cut]` as final text and remove it from the buffer.
    fn commit(&mut self, cut: usize, emit: &mut dyn FnMut(AsrEvent)) {
        let cut = cut.min(self.buf.len());
        let has_speech = self.speech.iter().take(cut / FRAME + 1).filter(|s| **s).count() as u64 * FRAME_MS
            >= MIN_SPEECH_MS;
        if cut > 0 && has_speech {
            let audio: Vec<f32> = self.buf[..cut].to_vec();
            self.spans.push((self.buf_start, self.buf_start + cut as u64));
            let prompt = self.prompt(&self.context);
            match self.engine.transcribe(&audio, &prompt, Pass::Final) {
                Ok(raw) => {
                    let mut segments: Vec<Segment> = Vec::new();
                    for r in raw {
                        let Some(text) = clean_text(&r.text) else { continue };
                        if r.no_speech > 0.8 || (r.no_speech > 0.4 && is_hallucination(&text)) {
                            continue;
                        }
                        let start = self.abs_ms(r.t0_ms);
                        let end = self.abs_ms(r.t1_ms).max(start);
                        let speaker = if self.voice.is_some() {
                            self.voice_label(&audio, r.t0_ms, r.t1_ms)
                        } else {
                            self.speaker(start, end)
                        };
                        segments.push(Segment { start_ms: start, end_ms: end, speaker, text });
                    }
                    if !segments.is_empty() {
                        self.live_chars += segments.iter().map(|s| s.text.len()).sum::<usize>();
                        self.live_segments.extend(segments.iter().cloned());
                        for s in &segments {
                            self.context.push(' ');
                            self.context.push_str(&s.text);
                        }
                        let n = self.context.chars().count();
                        if n > PROMPT_CHARS {
                            self.context = self.context.chars().skip(n - PROMPT_CHARS).collect();
                        }
                        emit(AsrEvent::Final(segments));
                    }
                }
                Err(e) => emit(AsrEvent::Error(format!("{e:#}"))),
            }
        }
        if self.preview_shown {
            self.preview_shown = false;
            emit(AsrEvent::Preview(None));
        }
        self.discard(cut);
    }

    /// Drop `buf[..cut]`. Partial cuts are rounded down to whole frames so the
    /// speech flags stay aligned with the audio.
    fn discard(&mut self, cut: usize) {
        let cut = if cut >= self.buf.len() { self.buf.len() } else { cut - cut % FRAME };
        self.buf.drain(..cut);
        if self.buf.is_empty() {
            self.speech.clear();
        } else {
            self.speech.drain(..cut / FRAME);
        }
        self.buf_start += cut as u64;
        let start = self.buf_start;
        self.energy.retain(|b| b.start + b.len > start);
        self.last_preview_len = 0;
    }

    fn abs_ms(&self, rel_ms: u64) -> u64 {
        self.buf_start * 1000 / ASR_RATE as u64 + rel_ms
    }

    fn speaker(&self, start_ms: u64, end_ms: u64) -> Option<String> {
        match self.cfg.speakers {
            SpeakerMode::Source => label_speaker(&self.energy, start_ms, end_ms),
            _ => None,
        }
    }

    /// Live label by voice for `chunk[t0..t1]` (ms relative to the chunk).
    /// Too-short speech keeps the previous speaker's label.
    fn voice_label(&mut self, chunk: &[f32], t0_ms: u64, t1_ms: u64) -> Option<String> {
        let at = |ms: u64| ((ms * ASR_RATE as u64 / 1000) as usize).min(chunk.len());
        let print = self.voice.as_ref()?.of(&chunk[at(t0_ms)..at(t1_ms)]);
        if let Some(print) = print {
            self.last_speaker = Some(self.live_speakers.identify(print));
        }
        self.last_speaker.map(speakers::label)
    }

    /// After Stop: polish the transcript and/or identify speakers across the
    /// whole recording, then hand over the finished transcript.
    fn finish(&mut self, emit: &mut dyn FnMut(AsrEvent)) {
        let mut segments = self.live_segments.clone();
        let mut improved = false;
        if self.cfg.polish {
            match self.polish(emit) {
                Ok(Some(polished)) => {
                    segments = polished;
                    improved = true;
                }
                Ok(None) => log::info!("The polished transcript looked incomplete; keeping the live text"),
                Err(message) => {
                    emit(AsrEvent::FinishAbandoned(message));
                    return;
                }
            }
        }
        if let Some(voice) = &self.voice {
            let grouping = speakers::Grouping { expected: self.cfg.expected_speakers, ..Default::default() };
            let turns = speakers::session_turns(voice, &self.history, grouping, &self.cfg.cancel, |p| {
                emit(AsrEvent::FinishProgress { stage: FinishStage::Speakers, progress: p })
            });
            match turns {
                Some(turns) => {
                    for s in &mut segments {
                        if let Some(n) = speakers::speaker_at(&turns, s.start_ms, s.end_ms) {
                            s.speaker = Some(speakers::label(n));
                        }
                    }
                    let found = turns.iter().map(|t| t.speaker + 1).max().unwrap_or(0);
                    log::info!("Identified {found} speakers in {} turns", turns.len());
                    improved = true;
                }
                None => {
                    emit(AsrEvent::FinishAbandoned("Finishing skipped; the live transcript is saved.".into()));
                    return;
                }
            }
        }
        if improved {
            emit(AsrEvent::Finished(segments));
        } else {
            emit(AsrEvent::FinishAbandoned("The polished transcript looked incomplete, so the live transcript was kept.".into()));
        }
    }

    /// Whisper's initial prompt: the user's names & terms, then the most
    /// recent transcript text for continuity.
    fn prompt(&self, context: &str) -> String {
        build_prompt(&self.cfg.vocabulary, context)
    }

    /// Re-transcribe the session's speech carefully (see module docs).
    /// `Ok(None)`: the result looked incomplete; `Err`: cancelled or failed.
    fn polish(&mut self, emit: &mut dyn FnMut(AsrEvent)) -> Result<Option<Vec<Segment>>, String> {
        let windows = polish_windows(&self.spans, &self.history);
        let total: u64 = windows.iter().map(|(a, b)| b - a).sum();
        let live_chars = self.live_chars;
        let mut done = 0u64;
        let mut out: Vec<Segment> = Vec::new();
        let mut context = String::new();
        emit(AsrEvent::FinishProgress { stage: FinishStage::Polish, progress: 0.0 });
        for (start, end) in windows {
            if self.cfg.cancel.load(Ordering::SeqCst) {
                return Err("Polishing skipped; the live transcript is saved.".into());
            }
            let audio: Vec<f32> = self.history[start as usize..(end as usize).min(self.history.len())]
                .iter()
                .map(|s| *s as f32 / 32768.0)
                .collect();
            let prompt = self.prompt(&context);
            let raw = match self.engine.transcribe(&audio, &prompt, Pass::Polish) {
                Ok(raw) => raw,
                Err(e) => return Err(format!("Polishing failed ({e:#}); the live transcript is saved.")),
            };
            let offset_ms = start * 1000 / ASR_RATE as u64;
            for r in raw {
                let Some(text) = clean_text(&r.text) else { continue };
                if r.no_speech > 0.8 || (r.no_speech > 0.4 && is_hallucination(&text)) {
                    continue;
                }
                let start_ms = offset_ms + r.t0_ms;
                let end_ms = (offset_ms + r.t1_ms).max(start_ms);
                // Voice labels are added afterwards, across the whole recording.
                let speaker = match self.cfg.speakers {
                    SpeakerMode::Source => label_speaker(&self.all_energy, start_ms, end_ms),
                    _ => None,
                };
                context.push(' ');
                context.push_str(&text);
                out.push(Segment { start_ms, end_ms, speaker, text });
            }
            let n = context.chars().count();
            if n > PROMPT_CHARS {
                context = context.chars().skip(n - PROMPT_CHARS).collect();
            }
            done += end - start;
            emit(AsrEvent::FinishProgress { stage: FinishStage::Polish, progress: done as f32 / total.max(1) as f32 });
        }
        // Safety net: never replace the live transcript with a much shorter one.
        let polished_chars: usize = out.iter().map(|s| s.text.len()).sum();
        Ok((polished_chars * 2 >= live_chars).then_some(out))
    }
}

/// Strip Whisper's non-speech annotations ("[BLANK_AUDIO]", "(music)") and tidy whitespace.
fn clean_text(text: &str) -> Option<String> {
    let mut out = String::with_capacity(text.len());
    let mut depth = 0u32;
    let mut in_stars = false;
    for c in text.chars() {
        match c {
            '[' | '(' if !in_stars => depth += 1,
            ']' | ')' if depth > 0 => depth -= 1,
            '*' if depth == 0 => in_stars = !in_stars,
            _ if depth == 0 && !in_stars => out.push(c),
            _ => {}
        }
    }
    let cleaned = out.split_whitespace().collect::<Vec<_>>().join(" ");
    let cleaned = cleaned.trim_start_matches(|c: char| c == '-' || c.is_whitespace()).to_string();
    if cleaned.chars().any(|c| c.is_alphanumeric()) {
        Some(cleaned)
    } else {
        None
    }
}

/// Whisper prompt: names & terms first, then recent transcript text.
fn build_prompt(vocabulary: &str, context: &str) -> String {
    let terms: Vec<&str> = vocabulary
        .split(|c| c == ',' || c == ';' || c == '\n')
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .collect();
    let mut prompt = String::new();
    if !terms.is_empty() {
        let mut list = terms.join(", ");
        if list.chars().count() > VOCABULARY_CHARS {
            list = list.chars().take(VOCABULARY_CHARS).collect();
        }
        prompt.push_str("Glossary: ");
        prompt.push_str(&list);
        prompt.push('.');
    }
    let context = context.trim();
    if !context.is_empty() {
        if !prompt.is_empty() {
            prompt.push(' ');
        }
        prompt.push_str(context);
    }
    prompt
}

/// Plan the polish windows over the session audio `history` (16 kHz).
///
/// Utterances separated by short pauses are joined into regions of
/// continuous speech; regions longer than POLISH_WINDOW_MS are split at the
/// quietest moment of the last 10 s that fit, so a window rarely starts or
/// ends mid-word. Only region edges (real pauses) get padding, so adjacent
/// windows never overlap and no word is transcribed twice.
fn polish_windows(spans: &[(u64, u64)], history: &[i16]) -> Vec<(u64, u64)> {
    let ms = |ms: u64| ms * ASR_RATE as u64 / 1000;
    let len = history.len() as u64;
    let mut regions: Vec<(u64, u64)> = Vec::new();
    for &(a, b) in spans {
        match regions.last_mut() {
            Some((_, end)) if a.saturating_sub(*end) <= ms(POLISH_MAX_GAP_MS) => *end = b.max(*end),
            _ => regions.push((a, b)),
        }
    }
    let mut out = Vec::new();
    for (i, &(a, b)) in regions.iter().enumerate() {
        // Pad into the surrounding pause, at most halfway to the neighbours.
        let prev_end = if i == 0 { 0 } else { regions[i - 1].1 };
        let next_start = regions.get(i + 1).map_or(len, |r| r.0);
        let start = a.saturating_sub(ms(200).min(a.saturating_sub(prev_end) / 2));
        let end = (b + ms(200).min(next_start.saturating_sub(b) / 2)).min(len);
        let mut cursor = start;
        while end - cursor > ms(POLISH_WINDOW_MS) {
            let cut = quietest_point(history, cursor + ms(POLISH_WINDOW_MS - 10_000), cursor + ms(POLISH_WINDOW_MS));
            out.push((cursor, cut));
            cursor = cut;
        }
        out.push((cursor, end));
    }
    out
}

/// Sample index in `from..to` at the centre of the quietest 100 ms stretch.
fn quietest_point(history: &[i16], from: u64, to: u64) -> u64 {
    const STEP: usize = 320; // 20 ms
    const SPAN: usize = 5; // 100 ms
    let to = (to as usize).min(history.len());
    let from = (from as usize).min(to);
    let energies: Vec<f64> = history[from..to]
        .chunks(STEP)
        .map(|c| c.iter().map(|s| (*s as f64) * (*s as f64)).sum())
        .collect();
    if energies.len() <= SPAN {
        return to as u64;
    }
    let best = (0..=energies.len() - SPAN)
        .min_by(|a, b| {
            let sum = |i: &usize| energies[*i..*i + SPAN].iter().sum::<f64>();
            sum(a).total_cmp(&sum(b))
        })
        .unwrap_or(0);
    (from + (best + SPAN / 2) * STEP) as u64
}

fn join_clean<'a>(parts: impl Iterator<Item = &'a str>) -> String {
    parts.filter_map(clean_text).collect::<Vec<_>>().join(" ")
}

fn is_hallucination(text: &str) -> bool {
    const PHRASES: [&str; 8] = [
        "thank you",
        "thank you.",
        "thanks for watching!",
        "thank you for watching.",
        "thanks for watching.",
        "you",
        "bye.",
        "please subscribe.",
    ];
    let t = text.trim().to_lowercase();
    PHRASES.contains(&t.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cleans_annotations() {
        assert_eq!(clean_text(" [BLANK_AUDIO] "), None);
        assert_eq!(clean_text("(upbeat music)"), None);
        assert_eq!(clean_text("Hello [inaudible] world").as_deref(), Some("Hello world"));
        assert_eq!(clean_text("*laughs* okay").as_deref(), Some("okay"));
        assert_eq!(clean_text(" - So, the budget."), Some("So, the budget.".into()));
    }

    #[test]
    fn polish_windows_split_at_pauses_without_overlap() {
        let rate = ASR_RATE as usize;
        // 40 s of "speech" with a quiet dip at 22.0 to 22.3 s.
        let mut history = vec![8000i16; rate * 40];
        history[rate * 22..rate * 22 + rate * 3 / 10].iter_mut().for_each(|s| *s = 10);
        // Utterances the live pass cut every 15 s (not at the dip).
        let s = |sec: f64| (sec * rate as f64) as u64;
        let spans = [(s(1.0), s(16.0)), (s(16.0), s(31.0)), (s(31.0), s(39.0))];
        let w = polish_windows(&spans, &history);
        assert_eq!(w.len(), 2, "{w:?}");
        assert!(w[0].1 == w[1].0, "adjacent windows touch, never overlap: {w:?}");
        let cut = w[0].1 as f64 / rate as f64;
        assert!((22.0..22.3).contains(&cut), "split inside the pause, got {cut}");
        assert!(w.iter().all(|(a, b)| b - a <= s(28.0)), "{w:?}");
        // Padding only at the outer edges.
        assert_eq!(w[0].0, s(0.8));
        assert_eq!(w[1].1, s(39.2));
    }

    #[test]
    fn prompt_puts_glossary_first() {
        assert_eq!(build_prompt("", ""), "");
        assert_eq!(build_prompt("Kalam, Maria;\n ACME Corp", "we met"), "Glossary: Kalam, Maria, ACME Corp. we met");
        assert!(build_prompt(&"x, ".repeat(400), "").chars().count() < VOCABULARY_CHARS + 20);
    }

    #[test]
    fn detects_hallucinations() {
        assert!(is_hallucination("Thank you."));
        assert!(!is_hallucination("Thank you for joining the call today."));
    }

    /// Speed + output of one or more models on a WAV (any format), for each
    /// kind of pass. Models: KALAM_TEST_MODELS="a.bin;b.bin" (default: bundled).
    ///   KALAM_TEST_WAV=speech.wav cargo test --release bench_models -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_models() {
        let audio = crate::test_util::test_audio();
        let secs = audio.len() as f32 / ASR_RATE as f32;
        let threads: u32 = std::env::var("KALAM_TEST_THREADS").ok().and_then(|t| t.parse().ok()).unwrap_or(0);
        println!("system info: {}", whisper_rs::print_system_info());
        let list = std::env::var("KALAM_TEST_MODELS").unwrap_or_else(|_| "resources/models/ggml-small.en-q5_1.bin".into());
        for path in list.split(';').filter(|p| !p.is_empty()) {
            let models = ModelManager::new(|_| {});
            models.set_gpu(crate::test_util::test_gpu(), None);
            models.select(PathBuf::from(path), None);
            let (model, _) = models.wait(Duration::from_secs(120)).unwrap();
            let mut cfg = WorkerConfig::for_test(true, false);
            cfg.threads = threads;
            let mut engine = Engine::new(model, &cfg).unwrap();
            for (label, pass, trim) in [("preview", Pass::Preview, true), ("final", Pass::Final, false), ("final-trim", Pass::Final, true), ("polish", Pass::Polish, false)] {
                engine.trim_ctx = trim;
                let t = Instant::now();
                let segs = engine.transcribe(&audio, "", pass).unwrap();
                let text: String = segs.iter().map(|s| s.text.as_str()).collect();
                println!("RESULT {path} {label:<10} {:>6.2}s for {secs:.1}s audio (x{:.2} realtime):{text}", t.elapsed().as_secs_f32(), t.elapsed().as_secs_f32() / secs);
            }
        }
    }

    /// Draft transcript for building a reference (ground truth): Kalam's most
    /// careful pipeline (polish pass + speaker labels by voice) over an excerpt,
    /// written as .docx and .txt with times relative to the original file, plus
    /// the excerpt audio. A person corrects the draft; the result is used to
    /// measure word error rates.
    ///   KALAM_TEST_WAV=meeting.mp3 KALAM_TEST_START=597 KALAM_TEST_SECONDS=300 KALAM_DRAFT_DIR=out
    ///   cargo test --release reference_draft -- --ignored --nocapture
    #[test]
    #[ignore]
    fn reference_draft() {
        use crate::docx::{format_clock, write_transcript, TranscriptMeta};
        let out = std::path::PathBuf::from(std::env::var("KALAM_DRAFT_DIR").expect("set KALAM_DRAFT_DIR"));
        std::fs::create_dir_all(&out).unwrap();
        let start_s: f64 = std::env::var("KALAM_TEST_START").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0);
        let offset_ms = (start_s * 1000.0) as u64;
        let audio = crate::test_util::test_audio();
        let history: Vec<i16> = audio.iter().map(|s| (s.clamp(-1.0, 1.0) * 32767.0) as i16).collect();
        let t0 = Instant::now();

        // Speech -> polish windows -> careful transcription (as after Stop).
        let spans: Vec<(u64, u64)> = speakers::utterances(&history).into_iter().map(|(a, b)| (a as u64, b as u64)).collect();
        let windows = polish_windows(&spans, &history);
        let models = crate::test_util::test_models();
        let (model, _) = models.wait(Duration::from_secs(120)).unwrap();
        let cfg = WorkerConfig::for_test(false, false);
        let mut engine = Engine::new(model, &cfg).unwrap();
        let mut segments: Vec<Segment> = Vec::new();
        let mut context = String::new();
        for (a, b) in windows {
            let chunk: Vec<f32> = history[a as usize..(b as usize).min(history.len())].iter().map(|s| *s as f32 / 32768.0).collect();
            let prompt = build_prompt(&cfg.vocabulary, &context);
            for r in engine.transcribe(&chunk, &prompt, Pass::Polish).unwrap() {
                let Some(text) = clean_text(&r.text) else { continue };
                if r.no_speech > 0.8 || (r.no_speech > 0.4 && is_hallucination(&text)) {
                    continue;
                }
                let start_ms = a * 1000 / ASR_RATE as u64 + r.t0_ms;
                let end_ms = (a * 1000 / ASR_RATE as u64 + r.t1_ms).max(start_ms);
                context.push(' ');
                context.push_str(&text);
                segments.push(Segment { start_ms, end_ms, speaker: None, text });
            }
            let n = context.chars().count();
            if n > PROMPT_CHARS {
                context = context.chars().skip(n - PROMPT_CHARS).collect();
            }
        }

        // Speaker labels by voice over the whole excerpt.
        let voice = Voiceprints::load(Path::new("resources/speakers/titanet_small.onnx"), 4).expect("voiceprint model");
        let turns = speakers::session_turns(&voice, &history, speakers::Grouping::default(), &AtomicBool::new(false), |_| {}).unwrap();
        for s in &mut segments {
            s.speaker = speakers::speaker_at(&turns, s.start_ms, s.end_ms).map(speakers::label);
            // Times as in the original recording.
            s.start_ms += offset_ms;
            s.end_ms += offset_ms;
        }
        println!("transcribed {} segments in {:.1}s", segments.len(), t0.elapsed().as_secs_f32());

        let span = format!("{}-{}", format_clock(offset_ms).replace(':', "."), format_clock(offset_ms + audio.len() as u64 * 1000 / ASR_RATE as u64).replace(':', "."));
        let meta = TranscriptMeta {
            started: chrono::Local::now(),
            duration_secs: audio.len() as f64 / ASR_RATE as f64,
            sources: format!("Reference draft, excerpt {span} of the original recording"),
            engine: "Whisper large-v3-turbo-q5_0 (polish pass: beam search, full context) + speaker labels by voice".into(),
            audio_file: format!("reference-excerpt-{span}.wav"),
            in_progress: false,
            quality: Some("DRAFT: please correct the words, then save".into()),
        };
        write_transcript(&out.join(format!("reference-draft-{span}.docx")), &meta, &segments).unwrap();
        let txt: String = segments
            .iter()
            .map(|s| format!("[{}] {}: {}\n", format_clock(s.start_ms), s.speaker.as_deref().unwrap_or("?"), s.text))
            .collect();
        std::fs::write(out.join(format!("reference-draft-{span}.txt")), txt).unwrap();
        let spec = hound::WavSpec { channels: 1, sample_rate: ASR_RATE, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
        let mut w = hound::WavWriter::create(out.join(meta.audio_file), spec).unwrap();
        history.iter().for_each(|s| w.write_sample(*s).unwrap());
        w.finalize().unwrap();
        println!("written to {}", out.display());
    }

    /// Prototype: NVIDIA Parakeet-TDT 0.6B v3 (via sherpa-onnx) over the same
    /// polish windows and speaker labels as `reference_draft`, for comparing
    /// accuracy and speed with Whisper. Writes "[HH:MM:SS] Speaker N: text" lines.
    ///   KALAM_PARAKEET_DIR=folder-with-encoder/decoder/joiner/tokens KALAM_TEST_OUT=out.txt
    ///   KALAM_TEST_WAV=meeting.mp3 KALAM_TEST_START=597 KALAM_TEST_SECONDS=300
    ///   cargo test --release parakeet_bench -- --ignored --nocapture
    /// KALAM_PARAKEET_HOTWORDS=file (one hotword per line, as tokens) tries
    /// modified beam search with hotwords; KALAM_PARAKEET_SCORE sets their boost.
    #[test]
    #[ignore]
    fn parakeet_bench() {
        use crate::docx::format_clock;
        use sherpa_onnx::{OfflineRecognizer, OfflineRecognizerConfig, OfflineTransducerModelConfig};
        let dir = std::path::PathBuf::from(std::env::var("KALAM_PARAKEET_DIR").expect("set KALAM_PARAKEET_DIR"));
        let file = |name: &str| Some(dir.join(name).to_string_lossy().into_owned());
        let threads: i32 = std::env::var("KALAM_TEST_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
        let mut config = OfflineRecognizerConfig::default();
        config.model_config.transducer = OfflineTransducerModelConfig {
            encoder: file("encoder.int8.onnx"),
            decoder: file("decoder.int8.onnx"),
            joiner: file("joiner.int8.onnx"),
        };
        config.model_config.tokens = file("tokens.txt");
        config.model_config.model_type = Some("nemo_transducer".into());
        config.model_config.num_threads = threads;
        if let Ok(hotwords) = std::env::var("KALAM_PARAKEET_HOTWORDS") {
            config.decoding_method = Some("modified_beam_search".into());
            config.hotwords_file = Some(hotwords);
            config.hotwords_score = std::env::var("KALAM_PARAKEET_SCORE").ok().and_then(|v| v.parse().ok()).unwrap_or(1.5);
        }
        let t_load = Instant::now();
        let recognizer = OfflineRecognizer::create(&config).expect("Parakeet loads");
        println!("loaded in {:.1}s ({threads} threads)", t_load.elapsed().as_secs_f32());

        let start_s: f64 = std::env::var("KALAM_TEST_START").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0);
        let offset_ms = (start_s * 1000.0) as u64;
        let audio = crate::test_util::test_audio();
        let history: Vec<i16> = audio.iter().map(|s| (s.clamp(-1.0, 1.0) * 32767.0) as i16).collect();
        let spans: Vec<(u64, u64)> = speakers::utterances(&history).into_iter().map(|(a, b)| (a as u64, b as u64)).collect();
        let windows = polish_windows(&spans, &history);

        // Sentences, timed from the token timestamps.
        let t0 = Instant::now();
        let mut segments: Vec<Segment> = Vec::new();
        for (a, b) in windows {
            let chunk: Vec<f32> = history[a as usize..(b as usize).min(history.len())].iter().map(|s| *s as f32 / 32768.0).collect();
            let stream = recognizer.create_stream();
            stream.accept_waveform(ASR_RATE as i32, &chunk);
            recognizer.decode(&stream);
            let r = stream.get_result().expect("result");
            let base = a * 1000 / ASR_RATE as u64;
            let times = r.timestamps.unwrap_or_default();
            let mut text = String::new();
            let mut first: Option<f32> = None;
            for (k, tok) in r.tokens.iter().enumerate() {
                let t = times.get(k).copied().unwrap_or(0.0);
                first.get_or_insert(t);
                text.push_str(tok);
                let last = k + 1 == r.tokens.len();
                if last || tok.trim_end().ends_with(['.', '?', '!']) {
                    if let Some(clean) = clean_text(&text) {
                        let start_ms = base + (first.unwrap() * 1000.0) as u64;
                        let end_ms = base + (t * 1000.0) as u64 + 80;
                        segments.push(Segment { start_ms, end_ms, speaker: None, text: clean });
                    }
                    text.clear();
                    first = None;
                }
            }
        }
        let asr_s = t0.elapsed().as_secs_f32();
        println!("transcribed {:.0}s of audio in {asr_s:.1}s ({:.3}x real time)", audio.len() as f32 / ASR_RATE as f32, asr_s / (audio.len() as f32 / ASR_RATE as f32));

        let voice = Voiceprints::load(Path::new("resources/speakers/titanet_small.onnx"), 4).expect("voiceprint model");
        let turns = speakers::session_turns(&voice, &history, speakers::Grouping::default(), &AtomicBool::new(false), |_| {}).unwrap();
        let txt: String = segments
            .iter()
            .map(|s| {
                let who = speakers::speaker_at(&turns, s.start_ms, s.end_ms).map(speakers::label).unwrap_or_else(|| "?".into());
                format!("[{}] {who}: {}\n", format_clock(s.start_ms + offset_ms), s.text)
            })
            .collect();
        print!("{txt}");
        if let Ok(out) = std::env::var("KALAM_TEST_OUT") {
            std::fs::write(out, txt).unwrap();
        }
    }

    /// Does saving audio as MP3 change what Whisper hears? Transcribes a clean
    /// WAV directly and again after the same 128 kbps MP3 encode/decode Kalam
    /// uses for recordings, then compares audio and transcripts.
    ///   KALAM_TEST_WAV=clean.wav cargo test --release mp3_round_trip -- --ignored --nocapture
    #[test]
    #[ignore]
    fn mp3_round_trip() {
        use crate::audio::capture::MIX_RATE;
        use crate::audio::mp3::Mp3Writer;
        use crate::audio::resample::Resampler;
        use crate::test_util::{load_audio_16k, load_wav_16k};

        let wav = std::env::var("KALAM_TEST_WAV").expect("set KALAM_TEST_WAV to an uncompressed WAV");
        let kbps: u32 = std::env::var("KALAM_TEST_KBPS").ok().and_then(|v| v.parse().ok()).unwrap_or(128);
        let direct = load_wav_16k(&wav);

        // Round trip exactly as a recording is saved: 48 kHz mono -> LAME -> MP3 file.
        let mut native = Vec::new();
        {
            let mut r = hound::WavReader::open(&wav).unwrap();
            let spec = r.spec();
            let samples: Vec<f32> = match spec.sample_format {
                hound::SampleFormat::Float => r.samples::<f32>().map(|s| s.unwrap()).collect(),
                hound::SampleFormat::Int => {
                    let scale = 1.0 / (1u64 << (spec.bits_per_sample - 1)) as f32;
                    r.samples::<i32>().map(|s| s.unwrap() as f32 * scale).collect()
                }
            };
            let ch = spec.channels.max(1) as usize;
            let mono: Vec<f32> = samples.chunks(ch).map(|f| f.iter().sum::<f32>() / f.len() as f32).collect();
            Resampler::new(spec.sample_rate, MIX_RATE).process_into(&mono, &mut native);
        }
        let mp3_path = std::env::temp_dir().join(format!("kalam-roundtrip-{kbps}.mp3"));
        let mut writer = Mp3Writer::create(&mp3_path, MIX_RATE, kbps).unwrap();
        writer.write(&native).unwrap();
        writer.finish().unwrap();
        let mp3_bytes = std::fs::metadata(&mp3_path).unwrap().len();
        let round_trip = load_audio_16k(mp3_path.to_str().unwrap());

        // Audio difference: align (MP3 adds a short delay), then signal-to-noise ratio.
        let n = direct.len().min(round_trip.len()).saturating_sub(4000);
        let err_at = |lag: usize| -> f64 {
            (0..n).map(|i| (direct[i] as f64 - round_trip[i + lag] as f64).powi(2)).sum::<f64>()
        };
        let lag = (0..2000).min_by(|a, b| err_at(*a).total_cmp(&err_at(*b))).unwrap();
        let signal: f64 = direct[..n].iter().map(|s| (*s as f64).powi(2)).sum();
        let snr = 10.0 * (signal / err_at(lag).max(1e-12)).log10();
        // Same comparison restricted to what Whisper uses: content below 8 kHz
        // is all there is at 16 kHz, so this is already the speech band.
        println!(
            "audio: {:.1} s, MP3 {} kbps = {:.0} KB, delay {:.1} ms, SNR {:.1} dB",
            direct.len() as f32 / ASR_RATE as f32,
            kbps,
            mp3_bytes as f64 / 1024.0,
            lag as f32 * 1000.0 / ASR_RATE as f32,
            snr
        );

        let list = std::env::var("KALAM_TEST_MODELS").unwrap_or_else(|_| {
            "resources/models/ggml-large-v3-turbo-q5_0.bin;resources/models/ggml-small.en-q5_1.bin".into()
        });
        for path in list.split(';').filter(|p| !p.is_empty()) {
            let models = ModelManager::new(|_| {});
            models.set_gpu(crate::test_util::test_gpu(), None);
            models.select(PathBuf::from(path), None);
            let (model, _) = models.wait(Duration::from_secs(120)).unwrap();
            let mut engine = Engine::new(model, &WorkerConfig::for_test(false, false)).unwrap();
            let mut run = |audio: &[f32]| {
                let t = Instant::now();
                let text: String = engine.transcribe(audio, "", Pass::Final).unwrap().iter().map(|s| s.text.as_str()).collect();
                (text.trim().to_string(), t.elapsed().as_secs_f32())
            };
            let (a, ta) = run(&direct);
            let (b, tb) = run(&round_trip[lag..]);
            let words = |s: &str| -> Vec<String> {
                s.split_whitespace()
                    .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'').to_lowercase())
                    .filter(|w| !w.is_empty())
                    .collect()
            };
            let (wa, wb) = (words(&a), words(&b));
            let (edits, diffs) = word_diff(&wa, &wb);
            let model = Path::new(path).file_name().unwrap().to_string_lossy();
            println!("\n== {model}");
            println!("WAV  ({ta:.2}s): {a}");
            println!("MP3  ({tb:.2}s): {b}");
            println!(
                "word differences: {edits} of {} words ({:.1}%){}",
                wa.len(),
                100.0 * edits as f64 / wa.len().max(1) as f64,
                if diffs.is_empty() { String::new() } else { format!(": {}", diffs.join(", ")) }
            );
        }
        std::fs::remove_file(&mp3_path).ok();
    }

    /// Word-level edit distance between two transcripts, plus readable edits.
    fn word_diff(a: &[String], b: &[String]) -> (usize, Vec<String>) {
        let (n, m) = (a.len(), b.len());
        let mut d = vec![vec![0usize; m + 1]; n + 1];
        for i in 0..=n {
            d[i][0] = i;
        }
        for j in 0..=m {
            d[0][j] = j;
        }
        for i in 1..=n {
            for j in 1..=m {
                let sub = d[i - 1][j - 1] + usize::from(a[i - 1] != b[j - 1]);
                d[i][j] = sub.min(d[i - 1][j] + 1).min(d[i][j - 1] + 1);
            }
        }
        let (mut i, mut j, mut out) = (n, m, Vec::new());
        while i > 0 || j > 0 {
            if i > 0 && j > 0 && d[i][j] == d[i - 1][j - 1] + usize::from(a[i - 1] != b[j - 1]) {
                if a[i - 1] != b[j - 1] {
                    out.push(format!("\"{}\" -> \"{}\"", a[i - 1], b[j - 1]));
                }
                i -= 1;
                j -= 1;
            } else if i > 0 && d[i][j] == d[i - 1][j] + 1 {
                out.push(format!("missing \"{}\"", a[i - 1]));
                i -= 1;
            } else {
                out.push(format!("extra \"{}\"", b[j - 1]));
                j -= 1;
            }
        }
        out.reverse();
        (d[n][m], out)
    }

    /// Diagnostic: profile a tap WAV written by the loopback test
    /// (3 channels: audio, mic rms, system rms) and transcribe it in one pass.
    #[test]
    #[ignore]
    fn analyze_tap() {
        let path = std::env::temp_dir().join("kalam-loopback-tap.wav");
        let data: Vec<f32> = hound::WavReader::open(&path).unwrap().samples::<f32>().map(|s| s.unwrap()).collect();
        let audio: Vec<f32> = data.chunks(3).map(|c| c[0]).collect();
        for (i, win) in data.chunks(3 * 8000).enumerate() {
            let a: Vec<f32> = win.chunks(3).map(|c| c[0]).collect();
            let rms = (a.iter().map(|s| s * s).sum::<f32>() / a.len() as f32).sqrt();
            let mic = win.chunks(3).map(|c| c[1]).sum::<f32>() / a.len() as f32;
            let sys = win.chunks(3).map(|c| c[2]).sum::<f32>() / a.len() as f32;
            println!("{:>5.1}s rms={rms:.4} mic={mic:.4} sys={sys:.4}", i as f32 * 0.5);
        }
        let models = crate::test_util::test_models();
        let cfg = WorkerConfig::for_test(false, false);
        let mut engine = Engine::new(models.wait(Duration::from_secs(120)).unwrap().0, &cfg).unwrap();
        for s in engine.transcribe(&audio, "", Pass::Final).unwrap() {
            println!("[{}-{}] ns={:.2} {}", s.t0_ms, s.t1_ms, s.no_speech, s.text);
        }
    }

    /// End-to-end check of the realtime pipeline with the bundled model:
    /// streams a 16 kHz mono WAV in 40 ms blocks at real speed and prints
    /// previews/finals with their latency.
    ///   KALAM_TEST_WAV=path/to/speech.wav cargo test --release -- --ignored --nocapture
    #[test]
    #[ignore]
    fn streams_wav_through_whisper() {
        use std::sync::mpsc;
        let wav = std::env::var("KALAM_TEST_WAV").expect("set KALAM_TEST_WAV");
        let spec = hound::WavReader::open(&wav).ok().map(|r| r.spec());
        let samples: Vec<f32> = if spec.is_some_and(|s| s.sample_format == hound::SampleFormat::Float && s.channels == 3) {
            // A tap file from the loopback test: channel 0 is the audio.
            let mut reader = hound::WavReader::open(&wav).unwrap();
            reader.samples::<f32>().map(|s| s.unwrap()).collect::<Vec<_>>().chunks(3).map(|c| c[0]).collect()
        } else {
            crate::test_util::test_audio()
        };
        // Follow the speech with 2 s of silence, as in a real pause.
        let samples: Vec<f32> = samples.into_iter().chain(std::iter::repeat(0.0).take(32_000)).collect();

        let models = crate::test_util::test_models();
        models.wait(Duration::from_secs(120)).expect("model loads");

        let (tx, rx) = mpsc::channel();
        let finals = Arc::new(Mutex::new(Vec::<Segment>::new()));
        let polished = Arc::new(Mutex::new(Vec::<Segment>::new()));
        let t0 = Instant::now();
        let worker = {
            let finals = Arc::clone(&finals);
            let polished = Arc::clone(&polished);
            spawn_worker(
                models,
                WorkerConfig::for_test(true, false),
                rx,
                move |ev| match ev {
                    AsrEvent::Preview(Some(p)) => println!("{:>6.2}s preview: {}", t0.elapsed().as_secs_f32(), p.text),
                    AsrEvent::Preview(None) => {}
                    AsrEvent::Final(segs) => {
                        for s in &segs {
                            println!("{:>6.2}s FINAL [{}-{} ms] {:?}: {}", t0.elapsed().as_secs_f32(), s.start_ms, s.end_ms, s.speaker, s.text);
                        }
                        finals.lock().unwrap().extend(segs);
                    }
                    AsrEvent::Error(e) => panic!("{e}"),
                    AsrEvent::LiveDone { finishing } => println!("{:>6.2}s live done (finishing: {finishing})", t0.elapsed().as_secs_f32()),
                    AsrEvent::FinishProgress { stage, progress } => {
                        if progress == 0.0 || progress == 1.0 {
                            println!("{:>6.2}s {stage:?} {:.0}%", t0.elapsed().as_secs_f32(), progress * 100.0)
                        }
                    }
                    AsrEvent::Finished(segs) => {
                        for s in &segs {
                            println!("{:>6.2}s POLISHED [{}-{} ms] {:?}: {}", t0.elapsed().as_secs_f32(), s.start_ms, s.end_ms, s.speaker, s.text);
                        }
                        polished.lock().unwrap().extend(segs);
                    }
                    AsrEvent::FinishAbandoned(m) => println!("finishing abandoned: {m}"),
                },
            )
        };
        for block in samples.chunks(640) {
            let rms = (block.iter().map(|s| s * s).sum::<f32>() / block.len() as f32).sqrt();
            tx.send(AsrMsg::Audio { samples: block.to_vec(), mic_rms: rms, system_rms: 0.0 }).unwrap();
            std::thread::sleep(Duration::from_millis(40));
        }
        println!("{:>6.2}s audio finished", t0.elapsed().as_secs_f32());
        tx.send(AsrMsg::Stop).unwrap();
        worker.join().unwrap();
        println!("{:>6.2}s worker done", t0.elapsed().as_secs_f32());

        let polished = polished.lock().unwrap();
        // KALAM_TEST_OUT=prefix: save "<prefix>-live.txt" (and "-polished.txt") as
        // "[HH:MM:SS] Speaker: text" lines, timed from KALAM_TEST_START, for scoring.
        if let Ok(prefix) = std::env::var("KALAM_TEST_OUT") {
            let start_ms = std::env::var("KALAM_TEST_START").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0) * 1000.0;
            let save = |name: &str, segs: &[Segment]| {
                let lines: Vec<String> = segs
                    .iter()
                    .map(|s| {
                        let t = (start_ms as u64 + s.start_ms) / 1000;
                        let who = s.speaker.as_deref().map(|w| format!("{w}: ")).unwrap_or_default();
                        format!("[{:02}:{:02}:{:02}] {who}{}", t / 3600, t / 60 % 60, t % 60, s.text)
                    })
                    .collect();
                std::fs::write(format!("{prefix}-{name}.txt"), lines.join("\n")).unwrap();
            };
            save("live", &finals.lock().unwrap());
            if !polished.is_empty() {
                save("polished", &polished);
            }
        }
        let join = |v: &[Segment]| v.iter().map(|s| s.text.as_str()).collect::<Vec<_>>().join(" ").to_lowercase();
        // With KALAM_TEST_POLISH=1 the polished transcript is what gets saved.
        let text = if polished.is_empty() { join(&finals.lock().unwrap()) } else { join(&polished) };
        // KALAM_TEST_EXPECT: a phrase the transcript must contain (default: JFK's).
        let expect = std::env::var("KALAM_TEST_EXPECT").unwrap_or_else(|_| "ask not what your country can do for you".into());
        assert!(text.contains(&expect.to_lowercase()), "got: {text}");
    }
}
