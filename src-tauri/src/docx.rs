//! Writes the transcript as a Word (.docx) document.

use std::path::Path;

use anyhow::{Context, Result};
use docx_rs::{Docx, LineSpacing, Paragraph, Run, RunFonts};

use crate::transcribe::Segment;

const INK: &str = "0B1F4D";
/// EU flag blue, for the title.
const BLUE: &str = "003399";
const MUTED: &str = "6B6F7A";
/// Dark gold: the EU yellow family, readable on white paper.
const ACCENT: &str = "8A6500";
/// Consecutive lines from the same speaker closer than this become one paragraph.
const MERGE_GAP_MS: u64 = 4_000;

pub struct TranscriptMeta {
    pub started: chrono::DateTime<chrono::Local>,
    pub duration_secs: f64,
    pub sources: String,
    pub engine: String,
    pub audio_file: String,
    pub in_progress: bool,
    /// How the text was produced, e.g. "Polished after recording".
    pub quality: Option<String>,
}

pub fn write_transcript(path: &Path, meta: &TranscriptMeta, segments: &[Segment]) -> Result<()> {
    let mut doc = Docx::new()
        .default_fonts(RunFonts::new().ascii("Calibri").hi_ansi("Calibri").cs("Calibri"))
        .default_size(22)
        .add_paragraph(
            Paragraph::new()
                .add_run(Run::new().add_text("Kalam Transcript").bold().size(40).color(BLUE))
                .line_spacing(LineSpacing::new().after(80)),
        )
        .add_paragraph(meta_line("Date", &meta.started.format("%A, %d %B %Y, %H:%M").to_string()))
        .add_paragraph(meta_line("Duration", &format_clock(meta.duration_secs as u64 * 1000)))
        .add_paragraph(meta_line("Sources", &meta.sources))
        .add_paragraph(meta_line("Recording", &meta.audio_file))
        .add_paragraph(meta_line("Engine", &meta.engine));
    if let Some(quality) = &meta.quality {
        doc = doc.add_paragraph(meta_line("Transcript", quality));
    }
    if meta.in_progress {
        doc = doc.add_paragraph(meta_line("Status", "Recording in progress…"));
    }
    doc = doc.add_paragraph(Paragraph::new().line_spacing(LineSpacing::new().after(120)));

    if segments.is_empty() {
        doc = doc.add_paragraph(
            Paragraph::new().add_run(Run::new().add_text("No speech was detected.").italic().color(MUTED)),
        );
    }

    let mut i = 0;
    while i < segments.len() {
        let first = &segments[i];
        let mut text = first.text.clone();
        let mut last_end = first.end_ms;
        let mut j = i + 1;
        while j < segments.len()
            && segments[j].speaker == first.speaker
            && segments[j].start_ms.saturating_sub(last_end) < MERGE_GAP_MS
        {
            text.push(' ');
            text.push_str(&segments[j].text);
            last_end = segments[j].end_ms;
            j += 1;
        }
        let mut p = Paragraph::new()
            .line_spacing(LineSpacing::new().after(120))
            .add_run(Run::new().add_text(format!("[{}]  ", format_clock(first.start_ms))).size(18).color(MUTED));
        if let Some(speaker) = &first.speaker {
            p = p.add_run(Run::new().add_text(format!("{speaker}: ")).bold().color(speaker_color(speaker)));
        }
        doc = doc.add_paragraph(p.add_run(Run::new().add_text(text).color(INK)));
        i = j;
    }

    // Write to a temp file then swap it in, so a crash never leaves a broken document.
    let tmp = path.with_extension("docx.tmp");
    {
        let file = std::fs::File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
        doc.build().pack(file).context("writing .docx")?;
    }
    std::fs::rename(&tmp, path).with_context(|| {
        let _ = std::fs::remove_file(&tmp);
        format!("saving {} (is it open in Word?)", path.display())
    })?;
    Ok(())
}

/// "Speaker N" gets its own colour (same order as in the app); other labels
/// ("You", "Others") the gold accent.
fn speaker_color(speaker: &str) -> &'static str {
    const PALETTE: [&str; 8] = [BLUE, ACCENT, "0F8A7E", "8E44AD", "C0392B", "2E7D32", "D35400", "5D6D7E"];
    match speaker.strip_prefix("Speaker ").and_then(|n| n.parse::<usize>().ok()) {
        Some(n) if n > 0 => PALETTE[(n - 1) % PALETTE.len()],
        _ => ACCENT,
    }
}

fn meta_line(label: &str, value: &str) -> Paragraph {
    Paragraph::new()
        .line_spacing(LineSpacing::new().after(0))
        .add_run(Run::new().add_text(format!("{label}: ")).bold().size(20).color(MUTED))
        .add_run(Run::new().add_text(value).size(20).color(MUTED))
}

/// `ms` → "HH:MM:SS".
pub fn format_clock(ms: u64) -> String {
    let s = ms / 1000;
    format!("{:02}:{:02}:{:02}", s / 3600, (s / 60) % 60, s % 60)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_a_readable_docx() {
        let dir = std::env::temp_dir().join(format!("kalam-docx-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.docx");
        let seg = |s, e, who: &str, t: &str| Segment {
            start_ms: s,
            end_ms: e,
            speaker: Some(who.into()),
            text: t.into(),
        };
        let meta = TranscriptMeta {
            started: chrono::Local::now(),
            duration_secs: 65.0,
            sources: "Microphone + System audio".into(),
            engine: "Whisper small.en".into(),
            audio_file: "x.mp3".into(),
            in_progress: false,
            quality: Some("Polished after recording".into()),
        };
        let segs = [seg(0, 2000, "You", "Hello there."), seg(2500, 4000, "You", "How are you?"), seg(5000, 7000, "Others", "Fine, thanks.")];
        write_transcript(&path, &meta, &segs).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert!(bytes.starts_with(b"PK"), "docx is a zip file");
        assert_eq!(format_clock(3_725_000), "01:02:05");
        std::fs::remove_dir_all(&dir).ok();
    }
}
