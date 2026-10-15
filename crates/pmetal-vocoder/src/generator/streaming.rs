//! Streaming and chunked synthesis.
//!
//! BigVGAN is fully convolutional, so each output sample depends on a bounded
//! window of mel frames (its receptive field). A stream can therefore be
//! synthesised in pieces that reproduce the one-shot result exactly: give every
//! piece enough left context to cover the receptive field, hold back the last
//! frames until enough right context has arrived, and discard the context's own
//! output. Memory stays proportional to the piece, not the recording.

use super::BigVGAN;
use crate::error::{Result, VocoderError};
use pmetal_runtime::compat::{Array, ops};

/// Incremental mel-to-waveform synthesis with exact overlap handling.
///
/// Feed mel frames with [`push`](Self::push); each call returns the audio that
/// is now final (everything whose right context has arrived). Call
/// [`finish`](Self::finish) at the end of the stream to flush the tail, whose
/// right edge is the real end of the signal. The concatenated output equals
/// [`BigVGAN::forward`] on the whole mel up to floating-point rounding.
///
/// The latency of the stream is the model's receptive-field radius in frames
/// ([`BigVGAN::receptive_radius_frames`]).
#[derive(Debug)]
pub struct StreamingSynthesizer<'a> {
    model: &'a BigVGAN,
    /// Buffered mel frames `[batch, frames, n_mels]` from `window_start` on.
    buffer: Option<Array>,
    /// Absolute index of the first buffered frame.
    window_start: usize,
    /// Absolute number of frames whose audio has been returned.
    emitted: usize,
    /// Absolute number of frames received.
    received: usize,
    batch: i32,
}

impl<'a> StreamingSynthesizer<'a> {
    /// Start a stream for `batch` parallel signals.
    pub fn new(model: &'a BigVGAN, batch: i32) -> Self {
        Self {
            model,
            buffer: None,
            window_start: 0,
            emitted: 0,
            received: 0,
            batch,
        }
    }

    /// Frames of delay between a mel frame arriving and its audio being final.
    pub fn latency_frames(&self) -> usize {
        self.model.receptive_radius_frames() as usize
    }

    /// Total frames received so far.
    pub fn frames_received(&self) -> usize {
        self.received
    }

    /// Add `mel` (`[batch, n_mels, frames]`) and return the newly final audio
    /// (`[batch, 1, samples]`), or `None` if no frame is final yet.
    pub fn push(&mut self, mel: &Array) -> Result<Option<Array>> {
        let config = &self.model.config;
        if mel.ndim() != 3 || mel.dim(0) != self.batch || mel.dim(1) != config.num_mels {
            return Err(VocoderError::ShapeMismatch {
                expected: vec![self.batch, config.num_mels, -1],
                actual: mel.shape().to_vec(),
            });
        }
        let chunk = mel.transpose_axes(&[0, 2, 1]);
        self.received += chunk.dim(1) as usize;
        self.buffer = Some(match self.buffer.take() {
            Some(b) => ops::concatenate_axis(&[&b, &chunk], 1),
            None => chunk,
        });
        let ready = self.received.saturating_sub(self.latency_frames());
        self.emit_until(ready)
    }

    /// Flush the remaining audio at the end of the stream.
    pub fn finish(&mut self) -> Result<Option<Array>> {
        self.emit_until(self.received)
    }

    /// Return the audio of frames `[emitted, until)`.
    fn emit_until(&mut self, until: usize) -> Result<Option<Array>> {
        if until <= self.emitted {
            return Ok(None);
        }
        let buffer = self.buffer.as_ref().expect("frames were received");
        let model = self.model;
        let hop = model.config.hop_size;

        let audio = model.forward_nlc(buffer)?; // [batch, buffered * hop, 1]
        let from = (self.emitted - self.window_start) as i32 * hop;
        let to = (until - self.window_start) as i32 * hop;
        let piece = ops::slice_axis(&audio, 1, from, to).transpose_axes(&[0, 2, 1]);
        // Realise the piece now so the context's graph does not pile up.
        piece.eval();

        self.emitted = until;
        // Keep `latency` frames of left context for the next piece.
        let keep_from = self.emitted.saturating_sub(self.latency_frames());
        if keep_from > self.window_start {
            let drop = (keep_from - self.window_start) as i32;
            self.buffer = Some(ops::slice_axis(buffer, 1, drop, buffer.dim(1)));
            self.window_start = keep_from;
        }
        Ok(Some(piece))
    }
}

impl BigVGAN {
    /// Synthesise `mel` (`[batch, n_mels, frames]`) piece by piece, at most
    /// `chunk_frames` new frames at a time, and return the whole waveform
    /// `[batch, 1, samples]`. The result equals [`forward`](Self::forward) up
    /// to rounding; peak memory follows the chunk, not the input.
    pub fn synthesize_chunked(&self, mel: &Array, chunk_frames: i32) -> Result<Array> {
        if chunk_frames < 1 || mel.ndim() != 3 {
            return Err(VocoderError::Config(
                "synthesize_chunked needs a [batch, n_mels, frames] mel and chunk_frames >= 1"
                    .into(),
            ));
        }
        let frames = mel.dim(2);
        let mut stream = StreamingSynthesizer::new(self, mel.dim(0));
        let mut pieces = Vec::new();
        let mut start = 0;
        while start < frames {
            let end = (start + chunk_frames).min(frames);
            let chunk = ops::slice_axis(mel, 2, start, end);
            pieces.extend(stream.push(&chunk)?);
            start = end;
        }
        pieces.extend(stream.finish()?);
        if pieces.is_empty() {
            return Err(VocoderError::Audio("the mel has no frames".into()));
        }
        let refs: Vec<&Array> = pieces.iter().collect();
        Ok(ops::concatenate_axis(&refs, 2))
    }
}
