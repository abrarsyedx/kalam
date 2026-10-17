use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// User configuration, persisted as JSON in the platform config directory.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Settings {
    /// Folder where `.mp3` recordings and `.docx` transcripts are written.
    pub output_dir: String,
    pub capture_mic: bool,
    pub capture_system: bool,
    /// cpal device id of the microphone; `None` = system default.
    pub mic_device: Option<String>,
    /// cpal device id of the speaker/headset whose output is recorded; `None` = system default.
    pub output_device: Option<String>,
    /// Speech model: "accurate" / "fast" (both bundled) or "custom" (`model_path`).
    pub model: String,
    /// Path to a ggml Whisper model file, used when `model` is "custom".
    pub model_path: Option<String>,
    /// Whisper language code ("en", "auto", ...). English-only models ignore it.
    pub language: String,
    /// Inference threads; 0 = choose automatically.
    pub threads: u32,
    pub mp3_bitrate: u32,
    /// Transcript speaker labels: "voice" (Speaker 1, Speaker 2... by voice),
    /// "source" (You = microphone / Others = computer audio) or "off".
    pub speakers: String,
    /// How many people speak, if known (labels by voice); 0 = automatic.
    pub expected_speakers: u32,
    /// Show a live, still-changing preview of the sentence being spoken.
    pub live_preview: bool,
    /// Graphics acceleration: "auto" (use a GPU if it passes Kalam's test) or "off".
    pub gpu: String,
    /// Re-transcribe the recording carefully after Stop, for the saved .docx:
    /// "auto" (when a graphics card is in use), "always" or "off".
    pub polish: String,
    /// Names, companies and terms, separated by commas or new lines.
    pub vocabulary: String,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            output_dir: String::new(),
            capture_mic: true,
            capture_system: true,
            mic_device: None,
            output_device: None,
            model: MODEL_ACCURATE.into(),
            model_path: None,
            language: "en".into(),
            threads: 0,
            mp3_bitrate: 128,
            speakers: SPEAKERS_VOICE.into(),
            expected_speakers: 0,
            live_preview: true,
            gpu: GPU_AUTO.into(),
            polish: POLISH_AUTO.into(),
            vocabulary: String::new(),
        }
    }
}

pub const BITRATES: [u32; 6] = [64, 96, 128, 160, 192, 256];
pub const MODEL_ACCURATE: &str = "accurate";
pub const MODEL_FAST: &str = "fast";
pub const MODEL_CUSTOM: &str = "custom";
pub const GPU_AUTO: &str = "auto";
pub const GPU_OFF: &str = "off";
pub const POLISH_AUTO: &str = "auto";
pub const POLISH_ALWAYS: &str = "always";
pub const POLISH_OFF: &str = "off";
pub const SPEAKERS_VOICE: &str = "voice";
pub const SPEAKERS_SOURCE: &str = "source";
pub const SPEAKERS_OFF: &str = "off";

impl Settings {
    pub fn load(path: &Path, default_output: &Path) -> Self {
        let json: Option<serde_json::Value> =
            std::fs::read_to_string(path).ok().and_then(|txt| serde_json::from_str(&txt).ok());
        let mut s: Settings = json.clone().and_then(|v| serde_json::from_value(v).ok()).unwrap_or_default();
        if let Some(v) = &json {
            // Settings from Kalam 1.0 had no "model" key; a model path meant a custom model.
            if v.get("model").is_none() {
                s.model = if s.model_path.is_some() { MODEL_CUSTOM } else { MODEL_ACCURATE }.into();
            }
            // Before 1.3, labels were on/off ("speakerLabels"); "on" now means by voice.
            if v.get("speakers").is_none() && v.get("speakerLabels") == Some(&serde_json::Value::Bool(false)) {
                s.speakers = SPEAKERS_OFF.into();
            }
        }
        if s.output_dir.trim().is_empty() {
            s.output_dir = default_output.to_string_lossy().into_owned();
        }
        s.normalize();
        s
    }

    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(path, serde_json::to_string_pretty(self)?)?;
        Ok(())
    }

    pub fn normalize(&mut self) {
        self.output_dir = self.output_dir.trim().to_string();
        if !BITRATES.contains(&self.mp3_bitrate) {
            self.mp3_bitrate = 128;
        }
        self.threads = self.threads.min(64);
        if ![MODEL_ACCURATE, MODEL_FAST, MODEL_CUSTOM].contains(&self.model.as_str())
            || (self.model == MODEL_CUSTOM && self.model_path.as_deref().map_or(true, |p| p.trim().is_empty()))
        {
            self.model = MODEL_ACCURATE.into();
        }
        if self.gpu != GPU_OFF {
            self.gpu = GPU_AUTO.into();
        }
        if ![SPEAKERS_VOICE, SPEAKERS_SOURCE, SPEAKERS_OFF].contains(&self.speakers.as_str()) {
            self.speakers = SPEAKERS_VOICE.into();
        }
        self.expected_speakers = self.expected_speakers.min(20);
        if self.polish != POLISH_ALWAYS && self.polish != POLISH_OFF {
            self.polish = POLISH_AUTO.into();
        }
        self.vocabulary = self.vocabulary.trim().to_string();
        if self.language.trim().is_empty() {
            self.language = "en".into();
        }
        for opt in [&mut self.mic_device, &mut self.output_device, &mut self.model_path] {
            if opt.as_deref().map(str::trim).map_or(false, str::is_empty) {
                *opt = None;
            }
        }
    }

    pub fn output_path(&self) -> PathBuf {
        PathBuf::from(&self.output_dir)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load_json(json: &str) -> Settings {
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("kalam-settings-{}-{n}.json", std::process::id()));
        std::fs::write(&path, json).unwrap();
        let s = Settings::load(&path, Path::new("out"));
        std::fs::remove_file(&path).ok();
        s
    }

    #[test]
    fn defaults_to_the_accurate_model() {
        assert_eq!(load_json("{}").model, MODEL_ACCURATE);
        assert_eq!(Settings::default().model, MODEL_ACCURATE);
    }

    #[test]
    fn migrates_kalam_1_0_settings() {
        let custom = load_json(r#"{"modelPath":"C:/models/ggml-medium.bin"}"#);
        assert_eq!(custom.model, MODEL_CUSTOM);
        assert_eq!(load_json(r#"{"modelPath":null,"threads":4}"#).model, MODEL_ACCURATE);
    }

    #[test]
    fn migrates_speaker_label_switch() {
        assert_eq!(load_json(r#"{"speakerLabels":true}"#).speakers, SPEAKERS_VOICE);
        assert_eq!(load_json(r#"{"speakerLabels":false}"#).speakers, SPEAKERS_OFF);
        assert_eq!(load_json(r#"{"speakers":"source","speakerLabels":false}"#).speakers, SPEAKERS_SOURCE);
        assert_eq!(load_json("{}").speakers, SPEAKERS_VOICE);
    }

    #[test]
    fn custom_without_a_path_falls_back() {
        assert_eq!(load_json(r#"{"model":"custom","modelPath":"  "}"#).model, MODEL_ACCURATE);
        assert_eq!(load_json(r#"{"model":"fast"}"#).model, MODEL_FAST);
        assert_eq!(load_json(r#"{"model":"bogus"}"#).model, MODEL_ACCURATE);
    }
}
