//! Reading and writing RIFF/WAVE files.
//!
//! Enough of the format for a vocoder's input and output: uncompressed PCM
//! (8, 16, 24 and 32 bit integers) and 32-bit float samples, any channel
//! count (mixed down to mono on request), and a 16-bit PCM writer. Anything
//! else (a compressed codec, a truncated file, a chunk that runs past the end)
//! is an error that says what it found, never a guess.

use crate::error::{Result, VocoderError};

/// Format tags of the `fmt ` chunk.
const FORMAT_PCM: u16 = 1;
const FORMAT_FLOAT: u16 = 3;
const FORMAT_EXTENSIBLE: u16 = 0xFFFE;

/// A decoded WAVE file: interleaved float samples in `[-1, 1]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Wav {
    /// Samples per second, per channel.
    pub sample_rate: u32,
    /// Interleaved channels.
    pub channels: u16,
    /// `frames * channels` samples.
    pub samples: Vec<f32>,
}

impl Wav {
    /// Samples per channel.
    pub fn frames(&self) -> usize {
        self.samples.len() / usize::from(self.channels.max(1))
    }

    /// The channels averaged into one.
    pub fn mono(&self) -> Vec<f32> {
        let channels = usize::from(self.channels.max(1));
        if channels == 1 {
            return self.samples.clone();
        }
        self.samples
            .chunks_exact(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect()
    }
}

fn audio_error(message: impl Into<String>) -> VocoderError {
    VocoderError::Audio(message.into())
}

fn le_u16(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([bytes[at], bytes[at + 1]])
}

fn le_u32(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

/// Decode a WAVE file.
pub fn read_wav(bytes: &[u8]) -> Result<Wav> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(audio_error("not a RIFF/WAVE file"));
    }
    let mut format: Option<(u16, u16, u32, u16)> = None; // tag, channels, rate, bits
    let mut data: Option<&[u8]> = None;
    let mut at = 12;
    while at + 8 <= bytes.len() {
        let id = &bytes[at..at + 4];
        let size = le_u32(bytes, at + 4) as usize;
        let body = at + 8;
        // A data chunk may declare more than the file holds when a recorder
        // was cut off; the samples that are there are still good, a whole
        // number of frames of them being checked below.
        let end = body.saturating_add(size).min(bytes.len());
        if id == b"fmt " {
            if end - body < 16 {
                return Err(audio_error("the fmt chunk is shorter than 16 bytes"));
            }
            let mut tag = le_u16(bytes, body);
            if tag == FORMAT_EXTENSIBLE {
                if end - body < 26 {
                    return Err(audio_error(
                        "an extensible fmt chunk without its sub-format",
                    ));
                }
                tag = le_u16(bytes, body + 24);
            }
            format = Some((
                tag,
                le_u16(bytes, body + 2),
                le_u32(bytes, body + 4),
                le_u16(bytes, body + 14),
            ));
        } else if id == b"data" {
            data = Some(&bytes[body..end]);
        }
        // Chunks are padded to an even length.
        at = body.saturating_add(size).saturating_add(size & 1);
    }
    let (tag, channels, sample_rate, bits) = format.ok_or_else(|| audio_error("no fmt chunk"))?;
    let data = data.ok_or_else(|| audio_error("no data chunk"))?;
    if channels == 0 || sample_rate == 0 {
        return Err(audio_error(format!(
            "{channels} channels at {sample_rate} Hz is not an audio format"
        )));
    }

    let samples: Vec<f32> = match (tag, bits) {
        (FORMAT_PCM, 8) => data
            .iter()
            .map(|&b| (f32::from(b) - 128.0) / 128.0)
            .collect(),
        (FORMAT_PCM, 16) => data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f32::from(i16::from_le_bytes(*c)) / 32768.0)
            .collect(),
        (FORMAT_PCM, 24) => data
            .as_chunks::<3>()
            .0
            .iter()
            .map(|c| {
                let value = i32::from_le_bytes([0, c[0], c[1], c[2]]) >> 8;
                value as f32 / 8_388_608.0
            })
            .collect(),
        (FORMAT_PCM, 32) => data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| i32::from_le_bytes(*c) as f32 / 2_147_483_648.0)
            .collect(),
        (FORMAT_FLOAT, 32) => data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
        _ => {
            return Err(audio_error(format!(
                "unsupported sample format (tag {tag}, {bits} bits): expected 8, 16, 24 or \
                 32-bit PCM, or 32-bit float"
            )));
        }
    };
    let channels_usize = usize::from(channels);
    let whole = samples.len() - samples.len() % channels_usize;
    let mut samples = samples;
    samples.truncate(whole);
    Ok(Wav {
        sample_rate,
        channels,
        samples,
    })
}

/// A mono 16-bit PCM WAVE file of `samples`.
pub fn write_wav_pcm16(samples: &[i16], sample_rate: u32) -> Vec<u8> {
    let data_len = (samples.len() * 2) as u32;
    let mut out = Vec::with_capacity(44 + samples.len() * 2);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&FORMAT_PCM.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
    out.extend_from_slice(&2u16.to_le_bytes()); // block align
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for sample in samples {
        out.extend_from_slice(&sample.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A WAVE file of `data` in the given format, with an extra chunk before
    /// the data, as encoders write.
    fn wav(tag: u16, channels: u16, rate: u32, bits: u16, data: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(b"WAVE");
        out.extend_from_slice(b"LIST");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&[1, 2, 3, 0]); // odd length, padded
        out.extend_from_slice(b"fmt ");
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&tag.to_le_bytes());
        out.extend_from_slice(&channels.to_le_bytes());
        out.extend_from_slice(&rate.to_le_bytes());
        out.extend_from_slice(&(rate * u32::from(channels) * u32::from(bits / 8)).to_le_bytes());
        out.extend_from_slice(&(channels * (bits / 8)).to_le_bytes());
        out.extend_from_slice(&bits.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(data);
        out
    }

    #[test]
    fn pcm16_round_trips_through_the_writer() {
        let samples = [0i16, 1, -1, 32767, -32768, 1234];
        let file = write_wav_pcm16(&samples, 24_000);
        let decoded = read_wav(&file).unwrap();
        assert_eq!((decoded.sample_rate, decoded.channels), (24_000, 1));
        assert_eq!(decoded.frames(), samples.len());
        for (got, want) in decoded.samples.iter().zip(samples) {
            assert!(
                (got - f32::from(want) / 32768.0).abs() < 1e-7,
                "{got} {want}"
            );
        }
    }

    #[test]
    fn every_supported_sample_format_decodes_to_the_same_value() {
        // Half scale in each format.
        let formats: [(u16, u16, Vec<u8>); 5] = [
            (FORMAT_PCM, 8, vec![192]),
            (FORMAT_PCM, 16, 16384i16.to_le_bytes().to_vec()),
            (FORMAT_PCM, 24, vec![0x00, 0x00, 0x40]),
            (FORMAT_PCM, 32, 0x4000_0000i32.to_le_bytes().to_vec()),
            (FORMAT_FLOAT, 32, 0.5f32.to_le_bytes().to_vec()),
        ];
        for (tag, bits, data) in formats {
            let decoded = read_wav(&wav(tag, 1, 16_000, bits, &data)).unwrap();
            assert_eq!(decoded.samples.len(), 1, "{bits}-bit {tag}");
            assert!((decoded.samples[0] - 0.5).abs() < 1e-6, "{bits}-bit {tag}");
        }
        // Negative full scale of 24 bits sign-extends.
        let low = read_wav(&wav(FORMAT_PCM, 1, 16_000, 24, &[0, 0, 0x80])).unwrap();
        assert_eq!(low.samples, [-1.0]);
    }

    #[test]
    fn stereo_mixes_down_to_the_mean_of_its_channels() {
        let mut data = Vec::new();
        for s in [16384i16, -16384, 16384, 16384] {
            data.extend_from_slice(&s.to_le_bytes());
        }
        let decoded = read_wav(&wav(FORMAT_PCM, 2, 8_000, 16, &data)).unwrap();
        assert_eq!(decoded.frames(), 2);
        assert_eq!(decoded.mono(), [0.0, 0.5]);
    }

    #[test]
    fn a_file_that_is_not_audio_we_can_read_is_an_error_that_says_so() {
        assert!(
            read_wav(b"not a wav at all")
                .unwrap_err()
                .to_string()
                .contains("RIFF")
        );
        let compressed = wav(2, 1, 8_000, 4, &[0; 4]); // ADPCM
        let err = read_wav(&compressed).unwrap_err().to_string();
        assert!(
            err.contains("unsupported") && err.contains("tag 2"),
            "{err}"
        );
        let mut no_data = wav(FORMAT_PCM, 1, 8_000, 16, &[0; 4]);
        no_data.truncate(no_data.len() - 12);
        assert!(read_wav(&no_data).unwrap_err().to_string().contains("data"));
    }

    #[test]
    fn a_cut_off_recording_keeps_its_whole_frames() {
        let mut file = wav(FORMAT_PCM, 1, 8_000, 16, &[1, 0, 2, 0, 3]);
        // The header promises 5 bytes; the last sample is half there.
        let decoded = read_wav(&file).unwrap();
        assert_eq!(decoded.samples.len(), 2);
        // A declared length beyond the end of the file is cut to the file.
        let at = file.len() - 5 - 4;
        file[at..at + 4].copy_from_slice(&1000u32.to_le_bytes());
        assert_eq!(read_wav(&file).unwrap().samples.len(), 2);
    }
}
