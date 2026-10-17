//! Commands invoked from the UI (ui/app.js).

use std::path::PathBuf;

use serde::Serialize;
use tauri::{AppHandle, Manager, State};
use tauri_plugin_dialog::DialogExt;
use tauri_plugin_opener::OpenerExt;

use crate::audio::capture::{list_devices as enumerate_devices, DeviceList};
use crate::session::{SavedFiles, Session, SessionInfo};
use crate::gpu::GpuStatus;
use crate::settings::{Settings, MODEL_ACCURATE, MODEL_CUSTOM, MODEL_FAST};
use crate::transcribe::ModelStatus;
use crate::AppState;

type CmdResult<T> = Result<T, String>;

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Status {
    recording: bool,
    paused: bool,
    session: Option<SessionInfo>,
    model: ModelStatus,
    default_output_dir: String,
    models: Vec<ModelChoice>,
    gpu: GpuStatus,
    /// The previous recording's transcript is still being polished.
    polishing: bool,
    version: String,
}

/// A bundled speech model the user can switch to.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelChoice {
    id: &'static str,
    name: &'static str,
    detail: &'static str,
    english_only: bool,
    available: bool,
}

#[tauri::command]
pub fn get_status(app: AppHandle, state: State<'_, AppState>) -> Status {
    let session = state.session.lock().unwrap();
    Status {
        recording: session.is_some(),
        paused: session.as_ref().is_some_and(|s| s.is_paused()),
        session: session.as_ref().map(|s| s.info.clone()),
        model: state.models.status(),
        default_output_dir: state.default_output.to_string_lossy().into_owned(),
        models: vec![
            ModelChoice {
                id: MODEL_ACCURATE,
                name: "Accurate",
                detail: "Whisper Large-v3 Turbo",
                english_only: false,
                available: state.accurate_model.is_file(),
            },
            ModelChoice {
                id: MODEL_FAST,
                name: "Fast",
                detail: "Whisper Small (English)",
                english_only: true,
                available: state.fast_model.is_file(),
            },
        ],
        gpu: state.gpu.lock().unwrap().clone(),
        polishing: state.polisher.lock().unwrap().as_ref().is_some_and(|p| !p.is_finished()),
        version: app.package_info().version.to_string(),
    }
}

#[tauri::command]
pub fn get_settings(state: State<'_, AppState>) -> Settings {
    state.settings.lock().unwrap().clone()
}

#[tauri::command]
pub fn save_settings(app: AppHandle, state: State<'_, AppState>, mut settings: Settings) -> CmdResult<Settings> {
    settings.normalize();
    if settings.output_dir.is_empty() {
        settings.output_dir = state.default_output.to_string_lossy().into_owned();
    }
    std::fs::create_dir_all(settings.output_path())
        .map_err(|e| format!("Cannot use the folder \"{}\": {e}", settings.output_dir))?;
    if settings.model == MODEL_CUSTOM {
        if let Some(model) = settings.model_path.as_ref().filter(|m| !PathBuf::from(m).is_file()) {
            return Err(format!("Model file not found: {model}"));
        }
    }
    let (model_changed, gpu_changed) = {
        let mut current = state.settings.lock().unwrap();
        let changed = (state.model_paths(&current) != state.model_paths(&settings), current.gpu != settings.gpu);
        settings.save(&state.settings_path).map_err(err)?;
        *current = settings.clone();
        changed
    };
    if gpu_changed {
        // Re-decides GPU vs CPU and reloads the models on that device.
        std::thread::spawn(move || crate::decide_gpu(&app, false));
    } else if model_changed {
        crate::load_model(&app);
    }
    Ok(settings)
}

/// Switch the speech model ("accurate", "fast" or "custom"). A recording in
/// progress keeps its model; the new one is used from the next recording.
#[tauri::command]
pub fn select_model(app: AppHandle, state: State<'_, AppState>, model: String) -> CmdResult<Settings> {
    let settings = {
        let mut current = state.settings.lock().unwrap();
        let mut next = current.clone();
        next.model = model;
        next.normalize();
        if next == *current {
            return Ok(next);
        }
        next.save(&state.settings_path).map_err(err)?;
        *current = next.clone();
        next
    };
    crate::load_model(&app);
    Ok(settings)
}

#[tauri::command]
pub async fn list_devices() -> CmdResult<DeviceList> {
    tauri::async_runtime::spawn_blocking(enumerate_devices).await.map_err(err)
}

#[tauri::command]
pub async fn start_recording(app: AppHandle) -> CmdResult<SessionInfo> {
    tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AppState>();
        // A new recording takes priority over polishing the previous one.
        state.cancel_polish();
        let mut slot = state.session.lock().unwrap();
        if slot.is_some() {
            return Err("Already recording".to_string());
        }
        let settings = state.settings.lock().unwrap().clone();
        let session =
            Session::start(&app, &settings, state.models.clone(), &state.voice_model).map_err(|e| format!("{e:#}"))?;
        let info = session.info.clone();
        *slot = Some(session);
        Ok(info)
    })
    .await
    .map_err(err)?
}

#[tauri::command]
pub fn pause_recording(state: State<'_, AppState>) -> CmdResult<()> {
    match state.session.lock().unwrap().as_ref() {
        Some(s) => Ok(s.set_paused(true)),
        None => Err("Not recording".into()),
    }
}

#[tauri::command]
pub fn resume_recording(state: State<'_, AppState>) -> CmdResult<()> {
    match state.session.lock().unwrap().as_ref() {
        Some(s) => Ok(s.set_paused(false)),
        None => Err("Not recording".into()),
    }
}

#[tauri::command]
pub async fn stop_recording(app: AppHandle) -> CmdResult<SavedFiles> {
    tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AppState>();
        let session = state.session.lock().unwrap().take();
        match session {
            Some(s) => {
                let (saved, polisher) = s.stop().map_err(|e| format!("{e:#}"))?;
                *state.polisher.lock().unwrap() = polisher;
                Ok(saved)
            }
            None => Err("Not recording".into()),
        }
    })
    .await
    .map_err(err)?
}

/// Skip polishing the last recording; its live transcript stays saved.
#[tauri::command]
pub async fn cancel_polish(app: AppHandle) -> CmdResult<()> {
    tauri::async_runtime::spawn_blocking(move || app.state::<AppState>().cancel_polish())
        .await
        .map_err(err)
}

/// Run the graphics-card test again (e.g. after a driver update).
#[tauri::command]
pub fn recheck_gpu(app: AppHandle, state: State<'_, AppState>) -> CmdResult<()> {
    if state.session.lock().unwrap().is_some() {
        return Err("Stop the recording first.".into());
    }
    crate::gpu::forget(&state.config_dir);
    std::thread::spawn(move || crate::decide_gpu(&app, true));
    Ok(())
}

#[tauri::command]
pub async fn pick_folder(app: AppHandle, current: Option<String>) -> CmdResult<Option<String>> {
    tauri::async_runtime::spawn_blocking(move || {
        let mut dialog = app.dialog().file().set_title("Choose where Kalam saves recordings and transcripts");
        if let Some(dir) = current.filter(|d| PathBuf::from(d).is_dir()) {
            dialog = dialog.set_directory(dir);
        }
        dialog
            .blocking_pick_folder()
            .map(|p| p.into_path().map(|p| p.to_string_lossy().into_owned()).map_err(err))
            .transpose()
    })
    .await
    .map_err(err)?
}

#[tauri::command]
pub async fn pick_model_file(app: AppHandle) -> CmdResult<Option<String>> {
    tauri::async_runtime::spawn_blocking(move || {
        app.dialog()
            .file()
            .set_title("Choose a Whisper ggml model (.bin)")
            .add_filter("Whisper ggml model", &["bin"])
            .blocking_pick_file()
            .map(|p| p.into_path().map(|p| p.to_string_lossy().into_owned()).map_err(err))
            .transpose()
    })
    .await
    .map_err(err)?
}

#[tauri::command]
pub fn open_path(app: AppHandle, path: String) -> CmdResult<()> {
    if !PathBuf::from(&path).exists() {
        return Err(format!("Not found: {path}"));
    }
    app.opener().open_path(path, None::<&str>).map_err(err)
}

#[tauri::command]
pub fn reveal_path(app: AppHandle, path: String) -> CmdResult<()> {
    let p = PathBuf::from(&path);
    if p.is_dir() {
        // Open the folder itself rather than selecting it in its parent.
        return app.opener().open_path(path, None::<&str>).map_err(err);
    }
    app.opener().reveal_item_in_dir(p).map_err(err)
}
