//! Streaming MP3 writer (LAME). Audio is encoded as it arrives, so a long
//! meeting never has to fit in memory and a crash still leaves a playable file.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use mp3lame_encoder::{Bitrate, Builder, Encoder, FlushNoGap, MonoPcm, Quality};

const CHUNK: usize = 8192;

pub struct Mp3Writer {
    encoder: Encoder,
    out: BufWriter<File>,
    buf: Vec<u8>,
    pcm: Vec<f32>,
}

impl Mp3Writer {
    pub fn create(path: &Path, sample_rate: u32, kbps: u32) -> Result<Self> {
        let mut b = Builder::new().ok_or_else(|| anyhow!("could not create LAME encoder"))?;
        b.set_num_channels(1).map_err(|e| anyhow!("mp3 channels: {e:?}"))?;
        b.set_sample_rate(sample_rate).map_err(|e| anyhow!("mp3 sample rate: {e:?}"))?;
        b.set_brate(bitrate(kbps)).map_err(|e| anyhow!("mp3 bitrate: {e:?}"))?;
        b.set_quality(Quality::Good).map_err(|e| anyhow!("mp3 quality: {e:?}"))?;
        let encoder = b.build().map_err(|e| anyhow!("mp3 encoder init: {e:?}"))?;
        let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
        Ok(Self { encoder, out: BufWriter::new(file), buf: Vec::new(), pcm: Vec::new() })
    }

    pub fn write(&mut self, samples: &[f32]) -> Result<()> {
        for chunk in samples.chunks(CHUNK) {
            self.pcm.clear();
            self.pcm.extend(chunk.iter().map(|s| s.clamp(-1.0, 1.0)));
            self.buf.clear();
            self.buf.reserve(mp3lame_encoder::max_required_buffer_size(self.pcm.len()));
            self.encoder
                .encode_to_vec(MonoPcm(self.pcm.as_slice()), &mut self.buf)
                .map_err(|e| anyhow!("mp3 encode: {e:?}"))?;
            self.out.write_all(&self.buf)?;
        }
        Ok(())
    }

    pub fn finish(mut self) -> Result<()> {
        self.buf.clear();
        self.buf.reserve(7200 * 2);
        self.encoder
            .flush_to_vec::<FlushNoGap>(&mut self.buf)
            .map_err(|e| anyhow!("mp3 flush: {e:?}"))?;
        self.out.write_all(&self.buf)?;
        self.out.flush()?;
        Ok(())
    }
}

fn bitrate(kbps: u32) -> Bitrate {
    match kbps {
        0..=64 => Bitrate::Kbps64,
        65..=96 => Bitrate::Kbps96,
        97..=128 => Bitrate::Kbps128,
        129..=160 => Bitrate::Kbps160,
        161..=192 => Bitrate::Kbps192,
        _ => Bitrate::Kbps256,
    }
}
