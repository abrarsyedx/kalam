//! Device enumeration and capture streams.
//!
//! Microphone: an ordinary input stream.
//! System audio: an *input* stream opened on an *output* device. cpal turns
//! that into WASAPI loopback on Windows and a Core Audio process tap on
//! macOS 14.6+, which captures everything the computer plays (Teams, Zoom,
//! Google Meet in the browser, ...).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, DeviceId, FromSample, SampleFormat, SizedSample, Stream, StreamConfig};
use serde::Serialize;

use super::resample::Resampler;

/// Everything is converted to mono at this rate before mixing.
pub const MIX_RATE: u32 = 48_000;
/// Cap on queued audio per source (protects memory if the mixer stalls).
const MAX_QUEUE: usize = MIX_RATE as usize * 5;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DeviceInfo {
    pub id: String,
    pub name: String,
    pub is_default: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DeviceList {
    pub inputs: Vec<DeviceInfo>,
    pub outputs: Vec<DeviceInfo>,
}

pub fn list_devices() -> DeviceList {
    let host = cpal::default_host();
    let collect = |devices: Option<Vec<Device>>, default: Option<Device>| -> Vec<DeviceInfo> {
        let default_id = default.and_then(|d| d.id().ok());
        devices
            .unwrap_or_default()
            .into_iter()
            .filter_map(|d| {
                let id = d.id().ok()?;
                Some(DeviceInfo {
                    is_default: Some(&id) == default_id.as_ref(),
                    id: id.to_string(),
                    name: device_name(&d),
                })
            })
            .collect()
    };
    DeviceList {
        inputs: collect(
            host.input_devices().ok().map(|it| it.collect()),
            host.default_input_device(),
        ),
        outputs: collect(
            host.output_devices().ok().map(|it| it.collect()),
            host.default_output_device(),
        ),
    }
}

pub fn device_name(d: &Device) -> String {
    d.description()
        .map(|desc| desc.name().to_string())
        .unwrap_or_else(|_| d.to_string())
}

/// Thread-safe queue of mono `MIX_RATE` samples filled by one capture stream.
#[derive(Default)]
pub struct SourceBuffer {
    queue: Mutex<VecDeque<f32>>,
    /// Peak level of recent audio, stored as f32 bits (for the UI meters).
    peak: AtomicU32,
}

impl SourceBuffer {
    fn push(&self, samples: &[f32]) {
        let peak = samples.iter().fold(0f32, |m, s| m.max(s.abs()));
        let prev = f32::from_bits(self.peak.load(Ordering::Relaxed));
        self.peak.store(prev.max(peak).to_bits(), Ordering::Relaxed);
        let mut q = self.queue.lock().unwrap();
        q.extend(samples);
        let excess = q.len().saturating_sub(MAX_QUEUE);
        q.drain(..excess);
    }

    pub fn len(&self) -> usize {
        self.queue.lock().unwrap().len()
    }

    /// Move up to `max` queued samples into `out`; returns how many were taken.
    pub fn take_into(&self, max: usize, out: &mut Vec<f32>) -> usize {
        let mut q = self.queue.lock().unwrap();
        let n = max.min(q.len());
        out.extend(q.drain(..n));
        n
    }

    /// Drop queued audio so only the newest `keep` samples remain.
    pub fn trim_to(&self, keep: usize) {
        let mut q = self.queue.lock().unwrap();
        let excess = q.len().saturating_sub(keep);
        q.drain(..excess);
    }

    pub fn clear(&self) {
        self.queue.lock().unwrap().clear();
    }

    /// Read and reset the peak level.
    pub fn take_peak(&self) -> f32 {
        f32::from_bits(self.peak.swap(0, Ordering::Relaxed))
    }
}

#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CaptureReport {
    pub mic_device: Option<String>,
    pub system_device: Option<String>,
    pub warnings: Vec<String>,
}

pub struct CaptureRequest {
    pub mic: Option<(Option<String>, Arc<SourceBuffer>)>,
    pub system: Option<(Option<String>, Arc<SourceBuffer>)>,
}

/// Open the requested streams on a dedicated thread (streams are kept on the
/// thread that created them) and keep them running until `stop` fires.
/// Reports which devices were opened, or an error if nothing could be opened.
pub fn spawn_capture(
    req: CaptureRequest,
    stop: Receiver<()>,
    ready: Sender<Result<CaptureReport, String>>,
) -> std::thread::JoinHandle<()> {
    std::thread::Builder::new()
        .name("kalam-capture".into())
        .spawn(move || {
            let host = cpal::default_host();
            let mut report = CaptureReport::default();
            let mut streams: Vec<Stream> = Vec::new();

            if let Some((id, sink)) = req.mic {
                match open_mic(&host, id.as_deref(), sink) {
                    Ok((name, s)) => {
                        report.mic_device = Some(name);
                        streams.push(s);
                    }
                    Err(e) => report.warnings.push(format!("Microphone unavailable: {e:#}")),
                }
            }
            if let Some((id, sink)) = req.system {
                match open_system(&host, id.as_deref(), sink) {
                    Ok((name, s)) => {
                        report.system_device = Some(name);
                        streams.push(s);
                    }
                    Err(e) => report.warnings.push(format!("System audio unavailable: {e:#}")),
                }
            }

            if streams.is_empty() {
                let msg = if report.warnings.is_empty() {
                    "No audio source selected. Enable the microphone and/or system audio in Settings.".to_string()
                } else {
                    report.warnings.join("\n")
                };
                let _ = ready.send(Err(msg));
                return;
            }
            let _ = ready.send(Ok(report));
            // Block until the session ends; dropping the streams stops capture.
            let _ = stop.recv();
            drop(streams);
        })
        .expect("spawn capture thread")
}

fn find_device(host: &cpal::Host, id: Option<&str>, output: bool) -> Result<Device> {
    if let Some(id) = id {
        if let Ok(parsed) = id.parse::<DeviceId>() {
            if let Some(d) = host.device_by_id(&parsed) {
                return Ok(d);
            }
        }
        log::warn!("Saved device {id} not found, falling back to the default device");
    }
    let d = if output { host.default_output_device() } else { host.default_input_device() };
    d.ok_or_else(|| anyhow!("no default {} device", if output { "output" } else { "input" }))
}

fn open_mic(host: &cpal::Host, id: Option<&str>, sink: Arc<SourceBuffer>) -> Result<(String, Stream)> {
    let device = find_device(host, id, false)?;
    let supported = device.default_input_config().context("reading microphone format")?;
    let stream = build_stream(&device, supported.config(), supported.sample_format(), sink)?;
    Ok((device_name(&device), stream))
}

fn open_system(host: &cpal::Host, id: Option<&str>, sink: Arc<SourceBuffer>) -> Result<(String, Stream)> {
    let device = find_device(host, id, true)?;
    // Loopback streams use the output (render) format of the device.
    let supported = device.default_output_config().context("reading output device format")?;
    let stream = build_stream(&device, supported.config(), supported.sample_format(), sink)
        .context(if cfg!(target_os = "macos") {
            "opening system audio tap (needs macOS 14.6+ and the \"System Audio Recording\" permission)"
        } else {
            "opening loopback capture"
        })?;
    Ok((device_name(&device), stream))
}

fn build_stream(
    device: &Device,
    config: StreamConfig,
    format: SampleFormat,
    sink: Arc<SourceBuffer>,
) -> Result<Stream> {
    let stream = match format {
        SampleFormat::F32 => typed_stream::<f32>(device, config, sink),
        SampleFormat::I16 => typed_stream::<i16>(device, config, sink),
        SampleFormat::I32 => typed_stream::<i32>(device, config, sink),
        SampleFormat::U16 => typed_stream::<u16>(device, config, sink),
        SampleFormat::U8 => typed_stream::<u8>(device, config, sink),
        SampleFormat::I8 => typed_stream::<i8>(device, config, sink),
        SampleFormat::F64 => typed_stream::<f64>(device, config, sink),
        other => return Err(anyhow!("unsupported sample format {other:?}")),
    }?;
    stream.play().context("starting audio stream")?;
    Ok(stream)
}

fn typed_stream<T>(device: &Device, config: StreamConfig, sink: Arc<SourceBuffer>) -> Result<Stream>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let channels = config.channels.max(1) as usize;
    let mut resampler = Resampler::new(config.sample_rate, MIX_RATE);
    let mut mono: Vec<f32> = Vec::new();
    let mut resampled: Vec<f32> = Vec::new();
    let stream = device.build_input_stream::<T, _, _>(
        config,
        move |data: &[T], _info| {
            mono.clear();
            mono.extend(data.chunks(channels).map(|frame| {
                frame.iter().map(|s| s.to_sample::<f32>()).sum::<f32>() / frame.len() as f32
            }));
            resampled.clear();
            resampler.process_into(&mono, &mut resampled);
            sink.push(&resampled);
        },
        |err| log::error!("audio stream error: {err}"),
        Some(Duration::from_secs(5)),
    )?;
    Ok(stream)
}
