//! Graphics-card (GPU) acceleration with a safe fallback to the CPU.
//!
//! Kalam is built with Vulkan on Windows (Intel, AMD and NVIDIA graphics) and
//! Metal on macOS. Several layers make sure an unusual PC can never break it:
//!
//! 1. **Bundled Vulkan loader.** The installer ships `vulkan-1.dll`, so Kalam
//!    starts even on PCs without any graphics driver.
//! 2. **Probe in a separate process.** Before a GPU is used, `kalam --gpu-probe`
//!    loads a model and transcribes a test clip on the GPU *and* the CPU. A
//!    driver that crashes or hangs only takes down that throw-away process, and
//!    the GPU is used only if it is clearly faster. The verdict is cached per
//!    app version.
//! 3. **Drivers kept out in CPU mode.** whisper.cpp initialises every compiled
//!    backend, even for CPU-only work. When Kalam runs on the CPU it tells the
//!    Vulkan loader to load no drivers at all (`VK_LOADER_DRIVERS_DISABLE`).
//! 4. **Crash marker.** A marker file exists while a model is loaded and warmed
//!    up on the GPU; if Kalam dies in that window, the next start uses the CPU.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

pub const PROBE_ARG: &str = "--gpu-probe";
const PROBE_PREFIX: &str = "KALAM_GPU_PROBE ";
const PROBE_TIMEOUT: Duration = Duration::from_secs(120);
/// Use the GPU only if it needs at most this fraction of the CPU's time.
const MIN_GAIN: f64 = 0.8;
const CACHE_FILE: &str = "gpu.json";
const MARKER_FILE: &str = "gpu-warmup.marker";

/// What the probe found.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProbeResult {
    pub device: Option<String>,
    pub gpu_ms: Option<u64>,
    pub cpu_ms: Option<u64>,
    pub use_gpu: bool,
    pub note: String,
}

/// Shown in the UI.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "state", rename_all = "camelCase")]
pub enum GpuStatus {
    /// This build has no GPU backend (e.g. Linux without CUDA).
    Unsupported,
    /// Turned off in Settings.
    Disabled,
    Checking,
    Active { device: String, speedup: Option<f32> },
    /// A GPU exists or was tried, but Kalam uses the CPU; `reason` says why.
    Cpu { reason: String },
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Cache {
    app_version: String,
    result: ProbeResult,
}

pub fn compiled_backend() -> Option<&'static str> {
    if cfg!(windows) {
        Some("Vulkan")
    } else if cfg!(target_os = "macos") {
        Some("Metal")
    } else if cfg!(feature = "gpu-cuda") {
        Some("CUDA")
    } else {
        None
    }
}

/// Keep graphics drivers out of this process entirely (CPU mode). Must run
/// before the first model is loaded. Only affects our bundled Vulkan loader.
pub fn keep_drivers_out() {
    if cfg!(windows) {
        std::env::set_var("VK_LOADER_DRIVERS_DISABLE", "*");
    }
}

pub fn drivers_kept_out() -> bool {
    cfg!(windows) && std::env::var_os("VK_LOADER_DRIVERS_DISABLE").is_some()
}

// ------------------------------------------------------- cache & marker ----

pub fn cached(config_dir: &Path, app_version: &str) -> Option<ProbeResult> {
    let txt = std::fs::read_to_string(config_dir.join(CACHE_FILE)).ok()?;
    let cache: Cache = serde_json::from_str(&txt).ok()?;
    (cache.app_version == app_version).then_some(cache.result)
}

pub fn store(config_dir: &Path, app_version: &str, result: &ProbeResult) {
    let cache = Cache { app_version: app_version.into(), result: result.clone() };
    let _ = std::fs::create_dir_all(config_dir);
    if let Ok(txt) = serde_json::to_string_pretty(&cache) {
        let _ = std::fs::write(config_dir.join(CACHE_FILE), txt);
    }
}

pub fn forget(config_dir: &Path) {
    let _ = std::fs::remove_file(config_dir.join(CACHE_FILE));
}

pub fn marker_path(config_dir: &Path) -> PathBuf {
    config_dir.join(MARKER_FILE)
}

/// True (and clears the marker) if Kalam died while warming up the GPU.
pub fn take_crash_marker(config_dir: &Path) -> bool {
    let path = marker_path(config_dir);
    let crashed = path.exists();
    let _ = std::fs::remove_file(path);
    crashed
}

// ----------------------------------------------------------- the probe ----

/// Run the probe in a child process (the same executable with `--gpu-probe`).
pub fn run_probe_process(model: &Path) -> Result<ProbeResult, String> {
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let mut cmd = Command::new(exe);
    cmd.arg(PROBE_ARG).arg(model).stdout(Stdio::piped()).stderr(Stdio::null()).stdin(Stdio::null());
    // The probe must see the real drivers even if this process keeps them out.
    cmd.env_remove("VK_LOADER_DRIVERS_DISABLE");
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    let mut child = cmd.spawn().map_err(|e| format!("could not start the test: {e}"))?;
    let started = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().map_err(|e| e.to_string())? {
            break status;
        }
        if started.elapsed() > PROBE_TIMEOUT {
            let _ = child.kill();
            let _ = child.wait();
            return Err("the graphics card did not respond in time".into());
        }
        std::thread::sleep(Duration::from_millis(200));
    };
    let mut out = String::new();
    if let Some(mut stdout) = child.stdout.take() {
        let _ = stdout.read_to_string(&mut out);
    }
    let line = out.lines().find_map(|l| l.strip_prefix(PROBE_PREFIX));
    match (status.success(), line) {
        (true, Some(json)) => serde_json::from_str(json).map_err(|e| format!("unreadable test result: {e}")),
        _ => Err(format!("the graphics driver crashed during the test ({status})")),
    }
}

/// Entry point of `kalam --gpu-probe <model>`. Prints one result line.
pub fn probe_main(model: &Path) -> i32 {
    match std::env::var("KALAM_GPU_PROBE_SIMULATE").as_deref() {
        Ok("crash") => std::process::abort(),
        Ok("hang") => loop {
            std::thread::sleep(Duration::from_secs(60));
        },
        _ => {}
    }
    let result = probe(model);
    println!("{PROBE_PREFIX}{}", serde_json::to_string(&result).unwrap_or_default());
    0
}

/// GPU devices ggml can use (first = the one whisper.cpp picks).
pub fn gpu_devices() -> Vec<String> {
    use std::ffi::CStr;
    use whisper_rs_sys::*;
    let mut out = Vec::new();
    // SAFETY: plain queries of ggml's device registry; strings are owned by ggml.
    unsafe {
        for i in 0..ggml_backend_dev_count() {
            let dev = ggml_backend_dev_get(i);
            let kind = ggml_backend_dev_type(dev);
            if kind == ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_GPU
                || kind == ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_IGPU
            {
                let desc = ggml_backend_dev_description(dev);
                let name = if desc.is_null() { ggml_backend_dev_name(dev) } else { desc };
                if !name.is_null() {
                    out.push(CStr::from_ptr(name).to_string_lossy().trim().to_string());
                }
            }
        }
    }
    out
}

fn probe(model: &Path) -> ProbeResult {
    let none = |note: &str| ProbeResult { device: None, gpu_ms: None, cpu_ms: None, use_gpu: false, note: note.into() };
    if compiled_backend().is_none() {
        return none("This build has no GPU support.");
    }
    let devices = gpu_devices();
    let Some(device) = devices.first().cloned() else {
        return none("No compatible graphics card was found.");
    };
    let audio = test_clip();
    // First GPU run includes one-off shader compilation; time the second.
    let gpu_ms = match time_transcription(model, true, &audio, 2) {
        Ok(ms) => ms,
        Err(e) => {
            return ProbeResult { device: Some(device), gpu_ms: None, cpu_ms: None, use_gpu: false, note: format!("The graphics card failed the test: {e}") };
        }
    };
    let cpu_ms = time_transcription(model, false, &audio, 1).ok();
    let use_gpu = cpu_ms.map_or(true, |cpu| (gpu_ms as f64) <= cpu as f64 * MIN_GAIN);
    let note = if use_gpu {
        "The graphics card is faster than the processor.".to_string()
    } else {
        "The graphics card is not faster than the processor here.".to_string()
    };
    ProbeResult { device: Some(device), gpu_ms: Some(gpu_ms), cpu_ms, use_gpu, note }
}

fn time_transcription(model: &Path, gpu: bool, audio: &[f32], runs: usize) -> Result<u64, String> {
    use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};
    let mut params = WhisperContextParameters::default();
    params.use_gpu(gpu);
    let ctx = WhisperContext::new_with_params(model, params).map_err(|e| e.to_string())?;
    let mut state = ctx.create_state().map_err(|e| e.to_string())?;
    let mut last = 0;
    for _ in 0..runs {
        let mut p = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        p.set_n_threads(crate::transcribe::auto_threads(false));
        p.set_language(Some("en"));
        p.set_print_progress(false);
        p.set_print_realtime(false);
        p.set_print_special(false);
        p.set_print_timestamps(false);
        p.set_temperature_inc(0.0);
        p.set_max_tokens(16);
        let t = Instant::now();
        state.full(p, audio).map_err(|e| e.to_string())?;
        last = t.elapsed().as_millis() as u64;
    }
    Ok(last)
}

/// 10 s of quiet, speech-band noise bursts: exercises the encoder (the costly
/// part) without needing a bundled audio file.
fn test_clip() -> Vec<f32> {
    let mut seed: u32 = 0x1234_5678;
    (0..crate::transcribe::ASR_RATE as usize * 10)
        .map(|i| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = (seed >> 9) as f32 / (1u32 << 23) as f32 - 0.5;
            let envelope = if (i / 4000) % 2 == 0 { 0.2 } else { 0.02 };
            noise * envelope
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Runs the GPU test in-process and prints the verdict. Simulate a PC
    /// without usable graphics with VK_LOADER_DRIVERS_DISABLE=*.
    ///   cargo test --release gpu_probe -- --ignored --nocapture
    #[test]
    #[ignore]
    fn gpu_probe_reports_a_verdict() {
        let started = Instant::now();
        let result = probe(Path::new(crate::test_util::FAST_MODEL));
        println!("devices: {:?}", gpu_devices());
        println!("probe ({:.1}s): {result:?}", started.elapsed().as_secs_f32());
        assert!(result.use_gpu == result.device.is_some() || !result.use_gpu);
    }
}
