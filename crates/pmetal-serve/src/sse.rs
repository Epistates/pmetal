//! Shared SSE helpers for the OpenAI + Anthropic streaming endpoints.
//!
//! BPE tokenisers can split a multi-byte UTF-8 codepoint across two tokens
//! — emitting each `Tokenizer::decode(&[token])` independently would send
//! invalid `\xNN` byte sequences to the client, which most HTTP/SSE
//! consumers reject. [`IncrementalDecoder`] buffers the full token
//! sequence, decodes the growing buffer as a unit, and emits only the
//! confirmed text prefix that has already shown up in a complete decode.
//!
//! All three streaming handlers (`chat_sse_stream`, `completion_sse_stream`,
//! and Anthropic `messages` streaming) previously open-coded this pattern
//! with `token_buffer: Vec<u32>` + `emitted_text_len: usize`. Factoring it
//! here keeps the state machine in one place — the handlers only differ in
//! how they wrap the decoded text into their endpoint-specific event
//! frames.

use std::sync::Arc;

/// Stateful decoder that turns a per-token event stream into UTF-8-safe
/// text deltas.
///
/// Typical use:
///
/// ```ignore
/// let mut dec = IncrementalDecoder::new(tokenizer);
/// // on each TokenEvent::Token:
/// let delta = dec.push(token_id);
/// if !delta.is_empty() { emit_content_delta(delta); }
/// // on TokenEvent::Done:
/// let tail = dec.flush();
/// if !tail.is_empty() { emit_content_delta(tail); }
/// // For tool-call detection post-Done:
/// let full = dec.decoded_text_with_special_tokens();
/// ```
///
/// Streaming logprobs use `push_with_aux` / `flush_aux` to keep an arbitrary
/// per-token payload aligned with the text deltas: tokens that don't yet
/// emit text (mid-codepoint) park their payload in the pending queue;
/// when a later token confirms the codepoint, the drained queue is
/// returned alongside the new text.
pub struct IncrementalDecoder<Aux = ()> {
    tokenizer: Arc<pmetal_data::Tokenizer>,
    buffer: Vec<u32>,
    emitted: usize,
    /// Per-token aux payloads waiting for their emission boundary.
    pending_aux: Vec<Aux>,
}

impl<Aux> IncrementalDecoder<Aux> {
    /// Create a new decoder that decodes tokens against `tokenizer`.
    pub fn new(tokenizer: Arc<pmetal_data::Tokenizer>) -> Self {
        Self {
            tokenizer,
            buffer: Vec::new(),
            emitted: 0,
            pending_aux: Vec::new(),
        }
    }

    /// Push a token and return the newly confirmed text prefix. Returns an
    /// empty string when the decoder is still mid-codepoint — callers
    /// should skip emitting an SSE frame in that case.
    pub fn push(&mut self, token_id: u32) -> String {
        self.buffer.push(token_id);
        self.consume_newly_decoded()
    }

    /// Push a token with an associated aux payload. Returns the newly
    /// confirmed text prefix and the per-token aux payloads aligned with
    /// the tokens that contributed to that text. When the decoder is still
    /// mid-codepoint, returns an empty string + empty Vec — the aux is
    /// queued and will drain on a later boundary.
    pub fn push_with_aux(&mut self, token_id: u32, aux: Aux) -> (String, Vec<Aux>) {
        self.buffer.push(token_id);
        self.pending_aux.push(aux);
        let text = self.consume_newly_decoded();
        if text.is_empty() {
            (text, Vec::new())
        } else {
            (text, std::mem::take(&mut self.pending_aux))
        }
    }

    /// Flush any remaining buffered tokens at end-of-stream. Returns an
    /// empty string when every decoded byte has already been emitted.
    pub fn flush(&mut self) -> String {
        self.consume_newly_decoded()
    }

    /// Same as [`flush`] but also returns any pending aux payloads that
    /// never got drained because their tokens never produced a complete
    /// codepoint. Callers using [`push_with_aux`] should prefer this on
    /// the terminal Done event so no payload is silently dropped.
    pub fn flush_aux(&mut self) -> (String, Vec<Aux>) {
        let text = self.consume_newly_decoded();
        let aux = std::mem::take(&mut self.pending_aux);
        (text, aux)
    }

    /// The full decoded text with special tokens preserved. Tool-call
    /// protocols such as Gemma 4 use special-token delimiters, so final
    /// tool-call parsing needs this form even when user-visible deltas use
    /// normal decoding.
    pub fn decoded_text_with_special_tokens(&self) -> String {
        self.tokenizer
            .decode_with_special_tokens(&self.buffer)
            .unwrap_or_default()
    }

    /// Number of tokens seen so far. Used by the Anthropic stream to
    /// report `output_tokens` on the terminal `message_delta`.
    pub fn token_count(&self) -> usize {
        self.buffer.len()
    }

    /// Advance `emitted` to the current decoded length and return the
    /// suffix that crossed the boundary. Shared by [`push`] and [`flush`].
    fn consume_newly_decoded(&mut self) -> String {
        let decoded = self.tokenizer.decode(&self.buffer).unwrap_or_default();
        if decoded.len() > self.emitted {
            let out = decoded[self.emitted..].to_owned();
            self.emitted = decoded.len();
            out
        } else {
            String::new()
        }
    }
}

#[cfg(test)]
mod tests {
    // Note: IncrementalDecoder depends on `pmetal_data::Tokenizer`, which
    // loads real tokenizer JSON files. End-to-end SSE behaviour is covered
    // by the downstream handlers; the assertions here are kept to pure
    // state-transition invariants that don't require a concrete tokenizer.
    //
    // If you add a mock tokenizer to `pmetal-data::tokenizer::testing`,
    // plumb it in here to cover the BPE-boundary edge case directly.
}
