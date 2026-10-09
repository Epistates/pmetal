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

/// The text one or more generated tokens added, split into the model's
/// reasoning and its answer.
#[derive(Debug)]
pub struct ChannelDelta<Aux> {
    /// New reasoning text (empty when none).
    pub reasoning: String,
    /// New answer text (empty when none).
    pub content: String,
    /// Aux payloads of the answer tokens behind `content`.
    pub aux: Vec<Aux>,
}

/// An [`IncrementalDecoder`] per channel, fed by the
/// [`ReasoningSplitter`](pmetal_data::stream_format::ReasoningSplitter): a
/// chat stream's reasoning and answer come out apart, with the markers
/// between them dropped. Each channel's leading whitespace (the newlines
/// after `<think>` and `</think>`) is dropped too. Aux payloads (logprobs)
/// follow the answer.
pub struct ChannelDecoder<Aux = ()> {
    splitter: pmetal_data::stream_format::ReasoningSplitter,
    reasoning: IncrementalDecoder<()>,
    answer: IncrementalDecoder<Aux>,
    reasoning_started: bool,
    answer_started: bool,
    tokens: usize,
}

impl<Aux> ChannelDecoder<Aux> {
    /// A decoder for the tokens generated after `prompt`, whose markers
    /// set the channel generation starts in.
    pub fn new(tokenizer: Arc<pmetal_data::Tokenizer>, prompt: &[u32]) -> Self {
        let mut splitter = pmetal_data::stream_format::ReasoningSplitter::new(&tokenizer);
        splitter.prime(prompt);
        Self {
            splitter,
            reasoning: IncrementalDecoder::new(Arc::clone(&tokenizer)),
            answer: IncrementalDecoder::new(tokenizer),
            reasoning_started: false,
            answer_started: false,
            tokens: 0,
        }
    }

    /// Push a generated token; its aux payload is kept when it belongs to
    /// the answer and dropped otherwise.
    pub fn push_with_aux(&mut self, token_id: u32, aux: Aux) -> ChannelDelta<Aux> {
        use pmetal_data::stream_format::Route;
        self.tokens += 1;
        let (reasoning, content, aux) = match self.splitter.route(token_id) {
            Route::Reasoning => (self.reasoning.push(token_id), String::new(), Vec::new()),
            Route::Answer => {
                let (text, aux) = self.answer.push_with_aux(token_id, aux);
                (String::new(), text, aux)
            }
            Route::Markup => (String::new(), String::new(), Vec::new()),
        };
        self.delta(reasoning, content, aux)
    }

    /// Flush both channels at end of stream.
    pub fn flush_aux(&mut self) -> ChannelDelta<Aux> {
        let reasoning = self.reasoning.flush();
        let (content, aux) = self.answer.flush_aux();
        self.delta(reasoning, content, aux)
    }

    /// The answer decoded with special tokens kept, for tool-call parsing.
    pub fn answer_with_special_tokens(&self) -> String {
        self.answer.decoded_text_with_special_tokens()
    }

    /// Number of tokens generated so far, markers and reasoning included.
    pub fn token_count(&self) -> usize {
        self.tokens
    }

    fn delta(&mut self, reasoning: String, content: String, aux: Vec<Aux>) -> ChannelDelta<Aux> {
        ChannelDelta {
            reasoning: trim_channel_start(reasoning, &mut self.reasoning_started),
            content: trim_channel_start(content, &mut self.answer_started),
            aux,
        }
    }
}

impl ChannelDecoder<()> {
    /// [`push_with_aux`](Self::push_with_aux) without a payload.
    pub fn push(&mut self, token_id: u32) -> ChannelDelta<()> {
        self.push_with_aux(token_id, ())
    }
}

/// `text` with leading whitespace dropped until a channel's first visible
/// character; `started` records that it has come.
fn trim_channel_start(text: String, started: &mut bool) -> String {
    if *started {
        return text;
    }
    let trimmed = text.trim_start();
    if trimmed.is_empty() {
        return String::new();
    }
    *started = true;
    trimmed.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_data::tokenizer::testing::word_level;

    #[test]
    fn channels_stream_apart_and_aux_follows_the_answer() {
        let tok = Arc::new(word_level(
            &["[UNK]", "assistant", "plan", "add", "Paris", "is", "here"],
            &[
                ("<|im_start|>", true),
                ("<think>", false),
                ("</think>", false),
            ],
        ));
        let ids = |text: &str| tok.encode_with_special_tokens(text).unwrap();
        // The template opened the thinking block.
        let mut decoder: ChannelDecoder<u32> =
            ChannelDecoder::new(Arc::clone(&tok), &ids("<|im_start|> assistant <think>"));
        let (mut reasoning, mut content, mut aux) = (String::new(), String::new(), Vec::new());
        for (i, token) in ids("plan add </think> Paris is here")
            .into_iter()
            .enumerate()
        {
            let delta = decoder.push_with_aux(token, i as u32);
            reasoning.push_str(&delta.reasoning);
            content.push_str(&delta.content);
            aux.extend(delta.aux);
        }
        let tail = decoder.flush_aux();
        assert!(tail.reasoning.is_empty() && tail.content.is_empty());
        assert_eq!(reasoning, "plan add");
        assert_eq!(content, "Paris is here");
        // Only the answer's tokens (3, 4, 5) carry their payloads through.
        assert_eq!(aux, vec![3, 4, 5]);
        assert_eq!(decoder.token_count(), 6);
        assert_eq!(decoder.answer_with_special_tokens(), "Paris is here");
    }
}
