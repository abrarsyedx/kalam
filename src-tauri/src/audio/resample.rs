//! Small streaming resampler: windowed-sinc low-pass FIR followed by linear
//! interpolation. The low-pass removes everything above ~45% of the lower of
//! the two rates, so the interpolation step cannot alias. Plenty for speech
//! and meeting audio, with no external dependency.

use std::f32::consts::PI;

const TAPS: usize = 63;

pub struct Resampler {
    step: f64,
    passthrough: bool,
    fir: Vec<f32>,
    /// Last `TAPS - 1` raw input samples (filter memory).
    history: Vec<f32>,
    /// Last filtered sample of the previous block (index -1 for interpolation).
    prev: f32,
    /// Position of the next output sample, in filtered-input sample units,
    /// relative to `prev` at index 0.
    pos: f64,
    scratch: Vec<f32>,
}

impl Resampler {
    pub fn new(in_rate: u32, out_rate: u32) -> Self {
        let passthrough = in_rate == out_rate;
        let cutoff = 0.45 * in_rate.min(out_rate) as f32 / in_rate as f32; // cycles/sample
        Self {
            step: in_rate as f64 / out_rate as f64,
            passthrough,
            fir: if passthrough { Vec::new() } else { design_lowpass(cutoff) },
            history: vec![0.0; TAPS - 1],
            prev: 0.0,
            pos: 0.0,
            scratch: Vec::new(),
        }
    }

    /// Resample one block of mono samples, appending the result to `out`.
    pub fn process_into(&mut self, input: &[f32], out: &mut Vec<f32>) {
        if self.passthrough {
            out.extend_from_slice(input);
            return;
        }
        if input.is_empty() {
            return;
        }
        // 1. Low-pass filter at the input rate.
        self.scratch.clear();
        self.scratch.reserve(input.len() + 1);
        self.scratch.push(self.prev);
        let hist_len = self.history.len();
        self.history.extend_from_slice(input);
        for i in 0..input.len() {
            let window = &self.history[i..i + TAPS];
            let acc: f32 = window.iter().zip(&self.fir).map(|(x, h)| x * h).sum();
            self.scratch.push(acc);
        }
        self.history.drain(..self.history.len() - hist_len);

        // 2. Linear interpolation; scratch[0] is the previous block's last sample.
        let n = input.len();
        while (self.pos as usize) < n {
            let i = self.pos as usize;
            let frac = (self.pos - i as f64) as f32;
            let s = self.scratch[i] * (1.0 - frac) + self.scratch[i + 1] * frac;
            out.push(s);
            self.pos += self.step;
        }
        self.pos -= n as f64;
        self.prev = self.scratch[n];
    }
}

/// Hann-windowed sinc low-pass with unity DC gain. `cutoff` in cycles/sample.
fn design_lowpass(cutoff: f32) -> Vec<f32> {
    let m = (TAPS - 1) as f32;
    let mut h: Vec<f32> = (0..TAPS)
        .map(|i| {
            let x = i as f32 - m / 2.0;
            let sinc = if x == 0.0 { 2.0 * cutoff } else { (2.0 * PI * cutoff * x).sin() / (PI * x) };
            let window = 0.5 - 0.5 * (2.0 * PI * i as f32 / m).cos();
            sinc * window
        })
        .collect();
    let sum: f32 = h.iter().sum();
    h.iter_mut().for_each(|v| *v /= sum);
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(rate: u32, hz: f32, secs: f32) -> Vec<f32> {
        (0..(rate as f32 * secs) as usize)
            .map(|i| (2.0 * PI * hz * i as f32 / rate as f32).sin())
            .collect()
    }

    fn rms(x: &[f32]) -> f32 {
        (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt()
    }

    #[test]
    fn output_length_matches_ratio_across_blocks() {
        let mut r = Resampler::new(44_100, 16_000);
        let mut out = Vec::new();
        let input = tone(44_100, 440.0, 2.0);
        for block in input.chunks(441) {
            r.process_into(block, &mut out);
        }
        let expected = input.len() as f64 * 16_000.0 / 44_100.0;
        assert!((out.len() as f64 - expected).abs() <= 2.0, "{} vs {}", out.len(), expected);
    }

    #[test]
    fn keeps_speech_band_and_removes_alias_band() {
        // 1 kHz passes, 12 kHz (above 8 kHz Nyquist of 16 kHz) is suppressed.
        let mut pass = Vec::new();
        Resampler::new(48_000, 16_000).process_into(&tone(48_000, 1_000.0, 1.0), &mut pass);
        let mut stop = Vec::new();
        Resampler::new(48_000, 16_000).process_into(&tone(48_000, 12_000.0, 1.0), &mut stop);
        assert!(rms(&pass[200..]) > 0.65, "passband rms {}", rms(&pass[200..]));
        assert!(rms(&stop[200..]) < 0.02, "stopband rms {}", rms(&stop[200..]));
    }

    #[test]
    fn passthrough_is_identity() {
        let mut r = Resampler::new(48_000, 48_000);
        let input = tone(48_000, 300.0, 0.1);
        let mut out = Vec::new();
        r.process_into(&input, &mut out);
        assert_eq!(out, input);
    }
}
