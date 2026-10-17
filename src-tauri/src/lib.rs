//! Kalam ("pen"): offline meeting recorder and live transcriber.

mod audio;
mod commands;
mod docx;
pub mod gpu;
mod session;
mod settings;
mod speakers;
#[cfg(test)]
mod test_util;
mod transcribe;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use tauri::path::BaseDirectory;
use tauri::{AppHandle, Emitter, Manager};

use gpu::{GpuStatus, ProbeResult};
use session::{Polisher, Session};
use settings::Settings;
use transcribe::ModelManager;

/// Bundled speech models (Tauri resources).
pub const FAST_MODEL: &str = "models/ggml-small.en-q5_1.bin";
pub const ACCURATE_MODEL: &str = "models/ggml-large-v3-turbo-q5_0.bin";
/// Voiceprint model for speaker labels by voice (Tauri resource).
pub const VOICE_MODEL: &str = "speakers/titanet_small.onnx";
pub const EV_MODEL: &str = "kalam://model";
pub const EV_GPU: &str = "kalam://gpu";

pub struct AppState {
    pub settings: Mutex<Settings>,
    pub settings_path: PathBuf,
    pub config_dir: PathBuf,
    pub app_version: String,
    pub default_output: PathBuf,
    pub fast_model: PathBuf,
    pub accurate_model: PathBuf,
    pub voice_model: PathBuf,
    pub models: Arc<ModelManager>,
    pub gpu: Mutex<GpuStatus>,
    pub session: Mutex<Option<Session>>,
    /// Polish pass of the last recording, if still running.
    pub polisher: Mutex<Option<Polisher>>,
}

impl AppState {
    /// The model that writes the transcript, and the light model for the live
    /// preview when that one is heavy.
    pub fn model_paths(&self, settings: &Settings) -> (PathBuf, Option<PathBuf>) {
        let primary = match settings.model.as_str() {
            settings::MODEL_FAST => self.fast_model.clone(),
            settings::MODEL_CUSTOM => settings.model_path.as_ref().map(PathBuf::from).unwrap_or_else(|| self.accurate_model.clone()),
            _ => self.accurate_model.clone(),
        };
        let heavy = std::fs::metadata(&primary).is_ok_and(|m| m.len() > transcribe::HEAVY_MODEL_BYTES);
        let preview = (heavy && primary != self.fast_model && self.fast_model.is_file()).then(|| self.fast_model.clone());
        (primary, preview)
    }

    /// Stop a running polish pass (its live transcript stays saved).
    pub fn cancel_polish(&self) {
        let polisher = self.polisher.lock().unwrap().take();
        if let Some(p) = polisher {
            p.cancel_and_wait();
        }
    }
}

/// Load the models chosen in Settings (unloading any others).
pub fn load_model(app: &AppHandle) {
    let state = app.state::<AppState>();
    let (primary, preview) = state.model_paths(&state.settings.lock().unwrap());
    state.models.select(primary, preview);
}

fn set_gpu_status(app: &AppHandle, status: GpuStatus) {
    *app.state::<AppState>().gpu.lock().unwrap() = status.clone();
    let _ = app.emit(EV_GPU, status);
}

/// Decide between graphics card and processor (see `gpu.rs`), then load the
/// models accordingly. `recheck` ignores the cached verdict. Blocking: may run
/// the GPU test, so call it from a background thread.
pub fn decide_gpu(app: &AppHandle, recheck: bool) {
    let state = app.state::<AppState>();
    let off = state.settings.lock().unwrap().gpu == settings::GPU_OFF;
    let cpu = |reason: &str| GpuStatus::Cpu { reason: reason.into() };

    let status = if gpu::compiled_backend().is_none() {
        GpuStatus::Unsupported
    } else if off {
        GpuStatus::Disabled
    } else if gpu::take_crash_marker(&state.config_dir) {
        let reason = "Kalam closed unexpectedly while starting the graphics card last time, so it uses the processor. \
                      Use \u{201c}Test graphics card again\u{201d} in Settings to retry.";
        let result = ProbeResult { device: None, gpu_ms: None, cpu_ms: None, use_gpu: false, note: reason.into() };
        gpu::store(&state.config_dir, &state.app_version, &result);
        log::warn!("GPU crash marker found; using the CPU");
        cpu(reason)
    } else {
        let cached = if recheck { None } else { gpu::cached(&state.config_dir, &state.app_version) };
        let result = cached.unwrap_or_else(|| {
            set_gpu_status(app, GpuStatus::Checking);
            let started = std::time::Instant::now();
            let result = gpu::run_probe_process(&state.fast_model).unwrap_or_else(|e| ProbeResult {
                device: None,
                gpu_ms: None,
                cpu_ms: None,
                use_gpu: false,
                note: format!("The graphics card test failed ({e}), so Kalam uses the processor."),
            });
            log::info!("GPU test took {:.1}s: {result:?}", started.elapsed().as_secs_f32());
            gpu::store(&state.config_dir, &state.app_version, &result);
            result
        });
        match (&result.device, result.use_gpu) {
            (Some(device), true) if gpu::drivers_kept_out() => cpu(&format!("Restart Kalam to start using {device}.")),
            (Some(device), true) => GpuStatus::Active {
                device: device.clone(),
                speedup: result.gpu_ms.zip(result.cpu_ms).map(|(g, c)| c as f32 / g.max(1) as f32),
            },
            _ => cpu(&result.note),
        }
    };

    let use_gpu = matches!(status, GpuStatus::Active { .. });
    if !use_gpu && !matches!(status, GpuStatus::Unsupported) {
        // CPU mode: keep graphics drivers out of this process altogether.
        gpu::keep_drivers_out();
    }
    log::info!("Graphics acceleration: {status:?}");
    state.models.set_gpu(use_gpu, Some(gpu::marker_path(&state.config_dir)));
    set_gpu_status(app, status);
    load_model(app);
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(
            tauri_plugin_log::Builder::new()
                .level(log::LevelFilter::Info)
                .max_file_size(2_000_000)
                .build(),
        )
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            let config_dir = app.path().app_config_dir()?;
            let default_output = app
                .path()
                .document_dir()
                .or_else(|_| app.path().home_dir())?
                .join("Kalam");
            let settings_path = config_dir.join("settings.json");
            let settings = Settings::load(&settings_path, &default_output);
            let fast_model = app.path().resolve(FAST_MODEL, BaseDirectory::Resource)?;
            let accurate_model = app.path().resolve(ACCURATE_MODEL, BaseDirectory::Resource)?;
            let voice_model = app.path().resolve(VOICE_MODEL, BaseDirectory::Resource)?;
            let handle = app.handle().clone();
            let models = ModelManager::new(move |status| {
                let _ = handle.emit(EV_MODEL, status);
            });
            let app_version = app.package_info().version.to_string();
            log::info!("Kalam {app_version} starting; settings at {}", settings_path.display());

            app.manage(AppState {
                settings: Mutex::new(settings),
                settings_path,
                config_dir,
                app_version,
                default_output,
                fast_model,
                accurate_model,
                voice_model,
                models,
                gpu: Mutex::new(GpuStatus::Checking),
                session: Mutex::new(None),
                polisher: Mutex::new(None),
            });
            let handle = app.handle().clone();
            std::thread::Builder::new()
                .name("kalam-gpu-decide".into())
                .spawn(move || decide_gpu(&handle, false))?;
            Ok(())
        })
        .on_window_event(|window, event| {
            // Never lose a recording by closing the window: finish it first.
            if let tauri::WindowEvent::Destroyed = event {
                let state = window.state::<AppState>();
                let session = state.session.lock().unwrap().take();
                if let Some(session) = session {
                    if let Ok((_, Some(polisher))) = session.stop() {
                        polisher.cancel_and_wait();
                    }
                }
                state.cancel_polish();
            }
        })
        .invoke_handler(tauri::generate_handler![
            commands::get_status,
            commands::get_settings,
            commands::save_settings,
            commands::select_model,
            commands::list_devices,
            commands::start_recording,
            commands::pause_recording,
            commands::resume_recording,
            commands::stop_recording,
            commands::cancel_polish,
            commands::recheck_gpu,
            commands::pick_folder,
            commands::pick_model_file,
            commands::open_path,
            commands::reveal_path,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Kalam");
}
