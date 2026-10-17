//! Speaker labels by voice, "who spoke when": Speaker 1, Speaker 2, ...
//!
//! A small voiceprint model (NVIDIA TitaNet-small, run by sherpa-onnx) turns a
//! few seconds of speech into a vector that is similar for the same voice and
//! different for different voices. It runs next to Whisper and never changes
//! the words.
//!
//! * **Live:** every finished sentence's voiceprint is compared with the voices
//!   heard so far ([`LiveSpeakers`]); a new voice becomes a new speaker.
//! * **After Stop:** the whole recording is cut into short utterances, each gets
//!   a voiceprint, and all of them are grouped at once ([`session_turns`]).
//!   Seeing every voice before deciding is far more reliable, so these labels
//!   replace the live ones.
//!
//! Measured on a real 66-minute meeting with 4 speakers: 4 speakers found at
//! every similarity threshold from 0.40 to 0.65, at 0.03x real time on a CPU.
//! pyannote's full segmentation pipeline found 9-17 "speakers" on the same
//! audio and was up to 20x slower, so Kalam does not use it.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use sherpa_onnx::{SpeakerEmbeddingExtractor, SpeakerEmbeddingExtractorConfig};

use crate::transcribe::ASR_RATE;

/// Voiceprints need this much speech to be reliable; shorter bits ("yes")
/// take the label of the speech around them.
const MIN_VOICEPRINT_MS: u64 = 1_000;
/// Longer speech is cut into pieces of at most this length.
const MAX_UTTERANCE_MS: u64 = 6_000;
/// Silence that separates two utterances.
const PAUSE_MS: u64 = 250;
/// Cosine similarity at which voiceprints count as the same speaker.
pub const DEFAULT_SAME_VOICE: f32 = 0.5;
/// Live labelling is stricter about merging: early on, centroids rest on few
/// sentences, and merging two people is worse than briefly splitting one.
pub const LIVE_SAME_VOICE: f32 = 0.55;

/// The voiceprint model.
pub struct Voiceprints {
    extractor: SpeakerEmbeddingExtractor,
}

impl Voiceprints {
    pub fn load(model: &Path, threads: i32) -> Option<Self> {
        if !model.is_file() {
            return None;
        }
        let extractor = SpeakerEmbeddingExtractor::create(&SpeakerEmbeddingExtractorConfig {
            model: Some(model.to_string_lossy().into_owned()),
            num_threads: threads,
            ..Default::default()
        })?;
        Some(Self { extractor })
    }

    /// Normalised voiceprint of 16 kHz speech, or `None` if it is too short.
    pub fn of(&self, audio: &[f32]) -> Option<Vec<f32>> {
        if (audio.len() as u64) * 1000 / (ASR_RATE as u64) < MIN_VOICEPRINT_MS {
            return None;
        }
        let stream = self.extractor.create_stream()?;
        stream.accept_waveform(ASR_RATE as i32, audio);
        stream.input_finished();
        if !self.extractor.is_ready(&stream) {
            return None;
        }
        self.extractor.compute(&stream).map(normalise)
    }
}

/// Live, provisional speaker labels: each voiceprint joins the most similar
/// speaker heard so far, or becomes a new speaker.
pub struct LiveSpeakers {
    /// Running mean voiceprint (normalised) and how many sentences it covers.
    speakers: Vec<(Vec<f32>, f32)>,
    same_voice: f32,
    max_speakers: usize,
}

impl LiveSpeakers {
    pub fn new(max_speakers: Option<usize>) -> Self {
        Self { speakers: Vec::new(), same_voice: LIVE_SAME_VOICE, max_speakers: max_speakers.unwrap_or(12).max(1) }
    }

    /// Speaker index (0-based) for this voiceprint.
    pub fn identify(&mut self, print: Vec<f32>) -> usize {
        let best = self
            .speakers
            .iter()
            .enumerate()
            .map(|(i, (c, _))| (i, dot(c, &print)))
            .max_by(|a, b| a.1.total_cmp(&b.1));
        match best {
            Some((i, sim)) if sim >= self.same_voice || self.speakers.len() >= self.max_speakers => {
                let (centroid, n) = &mut self.speakers[i];
                for (c, x) in centroid.iter_mut().zip(&print) {
                    *c = (*c * *n + x) / (*n + 1.0);
                }
                *centroid = normalise(std::mem::take(centroid));
                *n += 1.0;
                i
            }
            _ => {
                self.speakers.push((print, 1.0));
                self.speakers.len() - 1
            }
        }
    }
}

/// One stretch of speech by one speaker (0-based, in order of appearance).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Turn {
    pub start_ms: u64,
    pub end_ms: u64,
    pub speaker: usize,
}

/// How voiceprints are grouped after the recording.
#[derive(Debug, Clone, Copy)]
pub struct Grouping {
    pub same_voice: f32,
    /// If the user said how many people spoke, group into exactly that many.
    pub expected: Option<usize>,
    /// Groups with less than this share of all speech (or under 10 s) are
    /// folded into the most similar larger group: a cough, a laugh, noise.
    pub min_share: f32,
}

impl Default for Grouping {
    fn default() -> Self {
        Self { same_voice: DEFAULT_SAME_VOICE, expected: None, min_share: 0.02 }
    }
}

/// Speaker turns for a whole recording (16 kHz). Reports progress 0..1 and
/// returns `None` if `cancel` was set.
pub fn session_turns(
    model: &Voiceprints,
    audio: &[i16],
    grouping: Grouping,
    cancel: &AtomicBool,
    mut progress: impl FnMut(f32),
) -> Option<Vec<Turn>> {
    let pieces = utterances(audio);
    let mut prints = Vec::new();
    for (i, &(a, b)) in pieces.iter().enumerate() {
        if cancel.load(Ordering::SeqCst) {
            return None;
        }
        let piece: Vec<f32> = audio[a..b].iter().map(|s| *s as f32 / 32768.0).collect();
        if let Some(v) = model.of(&piece) {
            prints.push(((a, b), v));
        }
        if i % 20 == 0 {
            progress(i as f32 / pieces.len().max(1) as f32);
        }
    }
    progress(1.0);
    Some(turns_from(&prints, grouping))
}

/// Speech stretches (sample ranges) found with a simple energy detector, cut
/// at pauses and into pieces of at most MAX_UTTERANCE_MS.
pub fn utterances(audio: &[i16]) -> Vec<(usize, usize)> {
    const FRAME: usize = 320; // 20 ms
    let frames = |ms: u64| (ms as usize * ASR_RATE as usize / 1000) / FRAME;
    let mut noise = 0.002f32;
    let speech: Vec<bool> = audio
        .chunks(FRAME)
        .map(|f| {
            let rms = (f.iter().map(|s| (*s as f32 / 32768.0).powi(2)).sum::<f32>() / f.len() as f32).sqrt();
            let is_speech = rms > (noise * 3.0).max(0.004) || rms > 0.03;
            let rate = if is_speech { 0.0005 } else { 0.05 };
            noise = (noise * (1.0 - rate) + rms * rate).clamp(0.0003, 0.02);
            is_speech
        })
        .collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < speech.len() {
        if !speech[i] {
            i += 1;
            continue;
        }
        let start = i;
        let mut end = i + 1;
        let mut gap = 0;
        let mut j = i + 1;
        while j < speech.len() && gap < frames(PAUSE_MS) {
            if speech[j] {
                end = j + 1;
                gap = 0;
            } else {
                gap += 1;
            }
            j += 1;
        }
        let pieces = (end - start).div_ceil(frames(MAX_UTTERANCE_MS));
        let len = (end - start).div_ceil(pieces);
        for p in 0..pieces {
            let a = start + p * len;
            let b = (a + len).min(end);
            out.push((a * FRAME, (b * FRAME).min(audio.len())));
        }
        i = j;
    }
    out
}

/// Group voiceprints and merge adjacent pieces of the same speaker into turns.
pub fn turns_from(prints: &[((usize, usize), Vec<f32>)], grouping: Grouping) -> Vec<Turn> {
    let vecs: Vec<Vec<f32>> = prints.iter().map(|(_, v)| v.clone()).collect();
    let weights: Vec<f32> = prints.iter().map(|((a, b), _)| (b - a) as f32 / ASR_RATE as f32).collect();
    let labels = cluster(&vecs, &weights, grouping);
    let ms = |s: usize| s as u64 * 1000 / ASR_RATE as u64;
    let raw: Vec<(u64, u64, usize)> = prints.iter().zip(&labels).map(|(((a, b), _), l)| (ms(*a), ms(*b), *l)).collect();
    let mut turns: Vec<Turn> = Vec::new();
    for t in renumber(&raw) {
        match turns.last_mut() {
            Some(last) if last.speaker == t.speaker && t.start_ms.saturating_sub(last.end_ms) < 2_000 => last.end_ms = t.end_ms,
            _ => turns.push(t),
        }
    }
    turns
}

/// Agglomerative grouping of normalised voiceprints (centroid linkage on
/// duration-weighted means). Returns a group id per voiceprint.
pub fn cluster(prints: &[Vec<f32>], weights: &[f32], grouping: Grouping) -> Vec<usize> {
    let n = prints.len();
    if n == 0 {
        return Vec::new();
    }
    struct Group {
        sum: Vec<f32>,
        weight: f32,
        members: Vec<usize>,
    }
    let mut groups: Vec<Option<Group>> = prints
        .iter()
        .zip(weights)
        .enumerate()
        .map(|(i, (p, w))| Some(Group { sum: p.iter().map(|x| x * w).collect(), weight: *w, members: vec![i] }))
        .collect();
    let mut cents: Vec<Vec<f32>> = prints.to_vec();
    let mut sim = vec![vec![f32::NEG_INFINITY; n]; n];
    for i in 0..n {
        for j in i + 1..n {
            sim[i][j] = dot(&cents[i], &cents[j]);
        }
    }
    let mut alive = n;
    loop {
        let mut best = (f32::NEG_INFINITY, 0, 0);
        for i in (0..n).filter(|&i| groups[i].is_some()) {
            for j in i + 1..n {
                if sim[i][j] > best.0 && groups[j].is_some() {
                    best = (sim[i][j], i, j);
                }
            }
        }
        let (score, a, b) = best;
        let merge = match grouping.expected {
            Some(k) => alive > k.max(1),
            None => score >= grouping.same_voice,
        };
        if !merge || score == f32::NEG_INFINITY {
            break;
        }
        let gb = groups[b].take().unwrap();
        let ga = groups[a].as_mut().unwrap();
        ga.sum.iter_mut().zip(&gb.sum).for_each(|(x, y)| *x += y);
        ga.weight += gb.weight;
        ga.members.extend(gb.members);
        cents[a] = normalise(ga.sum.clone());
        for k in (0..n).filter(|&k| k != a && groups[k].is_some()) {
            let s = dot(&cents[a], &cents[k]);
            if k < a {
                sim[k][a] = s
            } else {
                sim[a][k] = s
            }
        }
        alive -= 1;
    }
    // Fold small groups into the most similar big one (unless a speaker count
    // was given, which already decided how many groups there are).
    if grouping.expected.is_none() {
        let total: f32 = weights.iter().sum();
        let small = |g: &Group| g.weight < (total * grouping.min_share).max(10.0);
        let big: Vec<usize> = (0..n).filter(|&i| groups[i].as_ref().is_some_and(|g| !small(g))).collect();
        if !big.is_empty() {
            for i in 0..n {
                if groups[i].as_ref().is_some_and(|g| small(g)) {
                    let target = *big
                        .iter()
                        .max_by(|x, y| dot(&cents[i], &cents[**x]).total_cmp(&dot(&cents[i], &cents[**y])))
                        .unwrap();
                    let g = groups[i].take().unwrap();
                    groups[target].as_mut().unwrap().members.extend(g.members);
                }
            }
        }
    }
    let mut label = vec![0; n];
    for (gi, g) in groups.iter().enumerate().filter_map(|(i, g)| g.as_ref().map(|g| (i, g))) {
        for &m in &g.members {
            label[m] = gi;
        }
    }
    label
}

/// Renumber speakers 0, 1, 2... in order of first appearance.
fn renumber(raw: &[(u64, u64, usize)]) -> Vec<Turn> {
    let mut order: Vec<usize> = Vec::new();
    raw.iter()
        .map(|&(start_ms, end_ms, id)| {
            let speaker = order.iter().position(|o| *o == id).unwrap_or_else(|| {
                order.push(id);
                order.len() - 1
            });
            Turn { start_ms, end_ms, speaker }
        })
        .collect()
}

/// The speaker who talks most during `start_ms..end_ms`; if nobody's turn
/// overlaps (very short speech), the speaker of the nearest turn within 5 s.
pub fn speaker_at(turns: &[Turn], start_ms: u64, end_ms: u64) -> Option<usize> {
    let mut talk: Vec<u64> = Vec::new();
    for t in turns.iter().filter(|t| t.start_ms < end_ms && t.end_ms > start_ms) {
        let overlap = t.end_ms.min(end_ms) - t.start_ms.max(start_ms);
        if talk.len() <= t.speaker {
            talk.resize(t.speaker + 1, 0);
        }
        talk[t.speaker] += overlap;
    }
    if let Some((s, _)) = talk.iter().enumerate().filter(|(_, ms)| **ms > 0).max_by_key(|(_, ms)| **ms) {
        return Some(s);
    }
    let gap = |t: &Turn| if t.end_ms <= start_ms { start_ms - t.end_ms } else { t.start_ms.saturating_sub(end_ms) };
    turns.iter().filter(|t| gap(t) <= 5_000).min_by_key(|t| gap(t)).map(|t| t.speaker)
}

/// "Speaker 1", "Speaker 2", ...
pub fn label(speaker: usize) -> String {
    format!("Speaker {}", speaker + 1)
}

fn normalise(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    v.iter_mut().for_each(|x| *x /= norm);
    v
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::time::Instant;

    fn unit(v: &[f32]) -> Vec<f32> {
        normalise(v.to_vec())
    }

    #[test]
    fn renumbers_in_order_of_appearance() {
        let turns = renumber(&[(0, 1000, 3), (1000, 2000, 1), (2000, 3000, 3), (3000, 4000, 0)]);
        assert_eq!(turns.iter().map(|t| t.speaker).collect::<Vec<_>>(), vec![0, 1, 0, 2]);
    }

    #[test]
    fn picks_the_speaker_who_talks_most_or_the_nearest() {
        let t = |a, b, s| Turn { start_ms: a, end_ms: b, speaker: s };
        let turns = [t(0, 4000, 0), t(4000, 5000, 1), t(5000, 9000, 2)];
        assert_eq!(speaker_at(&turns, 3000, 6000), Some(2));
        assert_eq!(speaker_at(&turns, 3400, 4500), Some(0));
        assert_eq!(speaker_at(&turns, 9500, 9900), Some(2), "nearest turn");
        assert_eq!(speaker_at(&turns, 20_000, 21_000), None, "nothing nearby");
    }

    #[test]
    fn groups_voices_and_folds_noise() {
        // Two clear voices plus one 2-second oddity close to voice A.
        let a = [unit(&[1.0, 0.1, 0.0]), unit(&[0.95, 0.15, 0.05]), unit(&[1.0, 0.0, 0.1])];
        let b = [unit(&[0.0, 1.0, 0.1]), unit(&[0.1, 0.95, 0.0])];
        let odd = unit(&[0.7, 0.0, 0.7]);
        let prints: Vec<Vec<f32>> = a.iter().chain(&b).chain(std::iter::once(&odd)).cloned().collect();
        let weights = [20.0, 20.0, 20.0, 20.0, 20.0, 2.0];
        let labels = cluster(&prints, &weights, Grouping::default());
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_ne!(labels[0], labels[3]);
        assert_eq!(labels[5], labels[0], "the oddity joins the closest real voice");
        // With a known speaker count the grouping obeys it.
        let one = cluster(&prints, &weights, Grouping { expected: Some(1), ..Default::default() });
        assert!(one.iter().all(|l| *l == one[0]));
    }

    #[test]
    fn live_speakers_recognise_returning_voices() {
        let mut live = LiveSpeakers::new(None);
        assert_eq!(live.identify(unit(&[1.0, 0.0, 0.0])), 0);
        assert_eq!(live.identify(unit(&[0.0, 1.0, 0.0])), 1);
        assert_eq!(live.identify(unit(&[0.95, 0.1, 0.0])), 0);
        let mut capped = LiveSpeakers::new(Some(1));
        capped.identify(unit(&[1.0, 0.0, 0.0]));
        assert_eq!(capped.identify(unit(&[0.0, 1.0, 0.0])), 0, "respects the speaker count");
    }

    #[test]
    fn utterances_split_at_pauses() {
        let rate = ASR_RATE as usize;
        let tone = |secs: f32| (0..(secs * rate as f32) as usize).map(|i| (i as f32 * 0.3).sin() * 0.2).collect::<Vec<f32>>();
        let silence = |secs: f32| vec![0.0f32; (secs * rate as f32) as usize];
        let audio: Vec<i16> = [silence(0.5), tone(2.0), silence(1.0), tone(14.0)].concat().iter().map(|s| (s * 32767.0) as i16).collect();
        let u = utterances(&audio);
        assert_eq!(u.len(), 1 + 3, "2 s utterance, then 14 s cut into 3 pieces: {u:?}");
        assert!(u.iter().all(|(a, b)| (b - a) <= rate * 6 + 320));
    }

    /// Voiceprint models x grouping thresholds on a recording (voiceprints are
    /// computed once per model; grouping is cheap).
    ///   KALAM_TEST_WAV=meeting.mp3 cargo test --release voiceprint_bench -- --ignored --nocapture
    #[test]
    #[ignore]
    fn voiceprint_bench() {
        let dir = PathBuf::from(std::env::var("KALAM_DIAR_DIR").unwrap_or_else(|_| "resources/speakers".into()));
        let audio: Vec<i16> = crate::test_util::test_audio().iter().map(|s| (s.clamp(-1.0, 1.0) * 32767.0) as i16).collect();
        let secs = audio.len() as f32 / ASR_RATE as f32;
        let threads = std::env::var("KALAM_TEST_THREADS").ok().and_then(|t| t.parse().ok()).unwrap_or(4);
        let models = std::env::var("KALAM_DIAR_EMBEDDINGS").unwrap_or_else(|_| "titanet_small.onnx".into());
        let thresholds: Vec<f32> = std::env::var("KALAM_DIAR_THRESHOLDS")
            .unwrap_or_else(|_| "0.4;0.45;0.5;0.55;0.6;0.65".into())
            .split(';')
            .filter_map(|t| t.parse().ok())
            .collect();
        println!("UTTER {} utterances in {secs:.0}s", utterances(&audio).len());
        for name in models.split(';') {
            let vp = Voiceprints::load(&dir.join(name), threads).expect("load voiceprint model");
            let t = Instant::now();
            let prints: Vec<_> = utterances(&audio)
                .into_iter()
                .filter_map(|(a, b)| {
                    let piece: Vec<f32> = audio[a..b].iter().map(|s| *s as f32 / 32768.0).collect();
                    vp.of(&piece).map(|v| ((a, b), v))
                })
                .collect();
            let took = t.elapsed().as_secs_f32();
            println!("PRINTS {name}: {} voiceprints in {took:.1}s (x{:.3} realtime)", prints.len(), took / secs);
            for &th in &thresholds {
                let turns = turns_from(&prints, Grouping { same_voice: th, ..Default::default() });
                let n = turns.iter().map(|t| t.speaker + 1).max().unwrap_or(0);
                let mut talk = vec![0u64; n];
                for t in &turns {
                    talk[t.speaker] += t.end_ms - t.start_ms;
                }
                let total = talk.iter().sum::<u64>().max(1) as f32;
                let share: Vec<String> = talk.iter().map(|ms| format!("{:.0}%", 100.0 * *ms as f32 / total)).collect();
                println!("GROUP {name:<20} same_voice={th:.2}: {n:>2} speakers, {:>4} turns  talk: {}", turns.len(), share.join(" "));
            }
        }
    }
}
