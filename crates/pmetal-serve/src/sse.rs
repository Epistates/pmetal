//! Shared SSE helpers for the OpenAI + Anthropic streaming endpoints.
//!
//! BPE tokenisers can split a multi-byte UTF-8 codepoint across two tokens
//! — emitting each `Tokenizer::decode(&[token])` independently would send
//! invalid `\xNN` byte sequences to the client, which most HTTP/SSE
//! consumers reject. [`IncrementalDecoder`] decodes a window of the newest
//! tokens as a unit and emits only the text confirmed complete.

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
/// ```
///
/// Each step decodes a window of the newest tokens, not the whole
/// generation: the window starts at the tokens behind the last emission (the
/// context that makes their text decode as it does in the full sequence) and
/// ends at the newest token, so a step costs the same at the thirty-thousandth
/// token as at the first. Text is held back while the window ends in a
/// partial UTF-8 sequence (U+FFFD); [`flush`](Self::flush) releases it.
///
/// Streaming logprobs use `push_with_aux` / `flush_aux` to keep an arbitrary
/// per-token payload aligned with the text deltas: tokens that don't yet
/// emit text (mid-codepoint) park their payload in the pending queue;
/// when a later token confirms the codepoint, the drained queue is
/// returned alongside the new text.
pub struct IncrementalDecoder<Aux = ()> {
    tokenizer: Arc<pmetal_data::Tokenizer>,
    /// Keep special tokens in the text (tool-call protocols are written in
    /// them) instead of skipping them.
    keep_special: bool,
    buffer: Vec<u32>,
    /// Where the decode window starts: the first token of the last emission.
    prefix_offset: usize,
    /// How many tokens' text has been emitted.
    read_offset: usize,
    /// Per-token aux payloads waiting for their emission boundary.
    pending_aux: Vec<Aux>,
}

impl<Aux> IncrementalDecoder<Aux> {
    /// Create a new decoder that decodes tokens against `tokenizer`.
    pub fn new(tokenizer: Arc<pmetal_data::Tokenizer>) -> Self {
        Self {
            tokenizer,
            keep_special: false,
            buffer: Vec::new(),
            prefix_offset: 0,
            read_offset: 0,
            pending_aux: Vec::new(),
        }
    }

    /// Decode special tokens into the text rather than dropping them.
    #[must_use]
    pub fn keeping_special_tokens(mut self, keep: bool) -> Self {
        self.keep_special = keep;
        self
    }

    /// Push a token and return the newly confirmed text prefix. Returns an
    /// empty string when the decoder is still mid-codepoint — callers
    /// should skip emitting an SSE frame in that case.
    pub fn push(&mut self, token_id: u32) -> String {
        self.buffer.push(token_id);
        self.consume_newly_decoded(false)
    }

    /// Push a token with an associated aux payload. Returns the newly
    /// confirmed text prefix and the per-token aux payloads aligned with
    /// the tokens that contributed to that text. When the decoder is still
    /// mid-codepoint, returns an empty string + empty Vec — the aux is
    /// queued and will drain on a later boundary.
    pub fn push_with_aux(&mut self, token_id: u32, aux: Aux) -> (String, Vec<Aux>) {
        self.buffer.push(token_id);
        self.pending_aux.push(aux);
        let text = self.consume_newly_decoded(false);
        if text.is_empty() {
            (text, Vec::new())
        } else {
            (text, std::mem::take(&mut self.pending_aux))
        }
    }

    /// Flush any remaining buffered tokens at end-of-stream. Returns an
    /// empty string when every decoded byte has already been emitted.
    pub fn flush(&mut self) -> String {
        self.consume_newly_decoded(true)
    }

    /// Same as [`flush`](Self::flush) but also returns any pending aux
    /// payloads that never got drained because their tokens never produced a
    /// complete codepoint. Callers using
    /// [`push_with_aux`](Self::push_with_aux) should prefer this on the
    /// terminal Done event so no payload is silently dropped.
    pub fn flush_aux(&mut self) -> (String, Vec<Aux>) {
        let text = self.consume_newly_decoded(true);
        let aux = std::mem::take(&mut self.pending_aux);
        (text, aux)
    }

    fn decode(&self, ids: &[u32]) -> String {
        let decoded = if self.keep_special {
            self.tokenizer.decode_with_special_tokens(ids)
        } else {
            self.tokenizer.decode(ids)
        };
        decoded.unwrap_or_default()
    }

    /// The text the window adds to what was emitted, advancing the window
    /// past it. A window that ends in a partial codepoint (U+FFFD) adds
    /// nothing until `flush`.
    fn consume_newly_decoded(&mut self, flush: bool) -> String {
        if self.read_offset >= self.buffer.len() {
            return String::new();
        }
        let emitted = self.decode(&self.buffer[self.prefix_offset..self.read_offset]);
        let window = self.decode(&self.buffer[self.prefix_offset..]);
        if window.len() <= emitted.len() || (!flush && window.ends_with('\u{fffd}')) {
            return String::new();
        }
        let Some(delta) = window.get(emitted.len()..) else {
            // The window decodes differently from its prefix (a merge across
            // the boundary): nothing consistent to emit yet.
            return String::new();
        };
        let delta = delta.to_owned();
        self.prefix_offset = self.read_offset;
        self.read_offset = self.buffer.len();
        delta
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
    reasoning_tokens: usize,
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
            reasoning_tokens: 0,
        }
    }

    /// Keep special tokens in the answer's text (a tool-call protocol
    /// written in them needs to see them).
    #[must_use]
    pub fn keeping_special_tokens(mut self, keep: bool) -> Self {
        self.answer = self.answer.keeping_special_tokens(keep);
        self
    }

    /// Push a generated token; its aux payload is kept when it belongs to
    /// the answer and dropped otherwise.
    pub fn push_with_aux(&mut self, token_id: u32, aux: Aux) -> ChannelDelta<Aux> {
        use pmetal_data::stream_format::Route;
        let (reasoning, content, aux) = match self.splitter.route(token_id) {
            Route::Reasoning => {
                self.reasoning_tokens += 1;
                (self.reasoning.push(token_id), String::new(), Vec::new())
            }
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

    /// How many of the tokens pushed were reasoning.
    pub fn reasoning_token_count(&self) -> usize {
        self.reasoning_tokens
    }

    fn delta(&mut self, reasoning: String, content: String, aux: Vec<Aux>) -> ChannelDelta<Aux> {
        ChannelDelta {
            reasoning: trim_channel_start(reasoning, &mut self.reasoning_started),
            content: trim_channel_start(content, &mut self.answer_started),
            aux,
        }
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
    use pmetal_data::tokenizer::testing::{byte_level_bpe, word_level};

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
        assert_eq!(decoder.reasoning_token_count(), 2);
    }

    /// Streaming a byte-level generation gives exactly the text decoding it
    /// whole gives, however the bytes of a multi-byte character fall across
    /// tokens.
    #[test]
    fn deltas_concatenate_to_the_full_decode() {
        let tok = Arc::new(byte_level_bpe(&["h i", "\u{120} t"], &["<s>"], None));
        let text = "hi there, caf\u{e9} \u{4e16}\u{754c} \u{1f980} t\u{e9}";
        let ids = tok.encode(text).unwrap();
        // Split the bytes so every multi-byte character spans tokens: one
        // token per byte (the 256 byte tokens are the first ids).
        let per_byte: Vec<u32> = text.bytes().map(|b| byte_token(&tok, b)).collect();
        for tokens in [ids, per_byte] {
            let mut decoder: IncrementalDecoder = IncrementalDecoder::new(Arc::clone(&tok));
            let mut streamed = String::new();
            for token in &tokens {
                let delta = decoder.push(*token);
                assert!(
                    !delta.contains('\u{fffd}'),
                    "a partial character leaked: {delta:?}"
                );
                streamed.push_str(&delta);
            }
            streamed.push_str(&decoder.flush());
            assert_eq!(streamed, tok.decode(&tokens).unwrap());
            assert_eq!(streamed, text);
        }
    }

    fn byte_token(tok: &pmetal_data::Tokenizer, byte: u8) -> u32 {
        let ids = tok.inner().get_vocab(true);
        // GPT-2's byte alphabet: printable bytes stand for themselves.
        let printable = |b: u8| matches!(b, 0x21..=0x7e | 0xa1..=0xac | 0xae..=0xff);
        let c = if printable(byte) {
            byte as char
        } else {
            let rank = (0..=255u8)
                .filter(|b| !printable(*b))
                .position(|b| b == byte)
                .unwrap();
            char::from_u32(256 + rank as u32).unwrap()
        };
        ids[&c.to_string()]
    }

    /// A step costs the same deep into a generation as at its start: 30k
    /// tokens used to decode the whole generation again at every token.
    #[test]
    fn a_long_generation_streams_in_linear_time() {
        let tok = Arc::new(byte_level_bpe(&[], &[], None));
        let mut decoder: IncrementalDecoder = IncrementalDecoder::new(Arc::clone(&tok));
        let token = byte_token(&tok, b'a');
        let started = std::time::Instant::now();
        let mut len = 0;
        for _ in 0..30_000 {
            len += decoder.push(token).len();
        }
        len += decoder.flush().len();
        assert_eq!(len, 30_000);
        assert!(
            started.elapsed() < std::time::Duration::from_secs(5),
            "30k tokens took {:?}",
            started.elapsed()
        );
    }

    #[test]
    fn special_tokens_are_kept_on_request() {
        let tok = Arc::new(byte_level_bpe(&[], &["<tool_call>"], None));
        let special = tok.inner().token_to_id("<tool_call>").unwrap();
        let mut plain: IncrementalDecoder = IncrementalDecoder::new(Arc::clone(&tok));
        assert_eq!(plain.push(special), "");
        let mut keeping: IncrementalDecoder =
            IncrementalDecoder::new(Arc::clone(&tok)).keeping_special_tokens(true);
        assert_eq!(keeping.push(special), "<tool_call>");
    }
}
