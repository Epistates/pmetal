//! Streaming output formatter for chat models with reasoning channels.
//!
//! Several modern instruct models (Gemma 4, Qwen 3, DeepSeek R1, GPT-OSS,
//! Phi 4 reasoning) emit a dedicated "thinking" channel alongside the
//! final answer. When a chat template has `enable_thinking=True`, the
//! decoded token stream is shaped like:
//!
//!   - Gemma 4: `<|channel>thought\n…reasoning…\n<channel|>…answer…`
//!   - Qwen 3 : `<think>…reasoning…</think>…answer…`
//!
//! The channel markers are special tokens in some tokenizers (Gemma 4)
//! and regular added tokens in others (Qwen 3's `<think>` decodes as a
//! literal string even with `skip_special_tokens=true`). Either way, the
//! downstream CLI ends up printing the reasoning body inline with the
//! final answer — indistinguishable from the "real" response. This
//! formatter fixes that by:
//!
//!  1. Resolving the enter/exit marker IDs up-front from the tokenizer.
//!  2. Tracking per-token whether we're inside a reasoning channel.
//!  3. Optionally suppressing the reasoning body (`show_thinking=false`),
//!     or labelling it with a visual separator when shown.
//!
//! Internally, the formatter keeps its own *decode-only* token buffer so
//! marker tokens and suppressed-thinking tokens never enter the decode
//! pipeline — no leaked `<think>` text, and `streamed_bytes` stays a
//! monotonic cursor over visible output only.

use crate::tokenizer::Tokenizer;

/// Names of channel-open marker tokens — all candidates are probed and
/// every one that resolves to an id in the tokenizer is recognised at
/// runtime. Multiple entries coexist so the same formatter handles
/// models that use different vocabularies.
const OPEN_MARKER_NAMES: &[&str] = &[
    "<|channel>", // Gemma 4
    "<think>",    // Qwen 3 / DeepSeek R1 Distill
];

/// Names of channel-close marker tokens.
const CLOSE_MARKER_NAMES: &[&str] = &[
    "<channel|>", // Gemma 4
    "</think>",   // Qwen 3 / DeepSeek R1 Distill
];

/// Streaming formatter that folds reasoning-channel markers into visual
/// separators so the end-user never sees raw `<|channel>thought` tokens
/// in their terminal output.
pub struct StreamFormatter {
    open_ids: Vec<u32>,
    close_ids: Vec<u32>,
    show_thinking: bool,

    /// Token buffer used for *decoding* only. Channel-marker tokens and
    /// suppressed-thinking tokens never enter this buffer — that way
    /// multi-byte character boundaries stay consistent while the marker
    /// text never leaks into `out`.
    decode_buf: Vec<u32>,
    /// Byte cursor into the decoded string: the amount of text we've
    /// already emitted from the buffer.
    streamed_bytes: usize,

    in_thinking: bool,
    thinking_label_printed: bool,
    answer_label_printed: bool,
}

impl StreamFormatter {
    /// Create a new formatter. Pass `show_thinking=false` to suppress the
    /// thinking-channel body entirely (matches the existing
    /// `--hide-thinking` flag in the CLI).
    pub fn new(tokenizer: &Tokenizer, show_thinking: bool) -> Self {
        let resolve = |name: &&str| -> Option<u32> { tokenizer.inner().token_to_id(name) };
        let open_ids: Vec<u32> = OPEN_MARKER_NAMES.iter().filter_map(resolve).collect();
        let close_ids: Vec<u32> = CLOSE_MARKER_NAMES.iter().filter_map(resolve).collect();
        tracing::debug!(
            "StreamFormatter: open_ids={:?} close_ids={:?} show_thinking={}",
            open_ids,
            close_ids,
            show_thinking
        );
        Self {
            open_ids,
            close_ids,
            show_thinking,
            decode_buf: Vec::new(),
            streamed_bytes: 0,
            in_thinking: false,
            thinking_label_printed: false,
            answer_label_printed: false,
        }
    }

    /// Called for every sampled token. Returns the incremental text the
    /// caller should emit to stdout (may be empty, may include labels).
    pub fn push_token(&mut self, tokenizer: &Tokenizer, token_id: u32) -> String {
        let mut out = String::new();

        // 1. Channel-enter: swap visual state, inject label, drop marker
        //    from the decode buffer.
        if self.open_ids.contains(&token_id) && !self.in_thinking {
            self.in_thinking = true;
            if self.show_thinking && !self.thinking_label_printed {
                if !self.is_at_start() {
                    out.push('\n');
                }
                out.push_str("[thinking]\n");
                self.thinking_label_printed = true;
            }
            return out;
        }

        // 2. Channel-exit: swap back, inject label / newline, drop marker.
        if self.close_ids.contains(&token_id) && self.in_thinking {
            self.in_thinking = false;
            if self.show_thinking && !self.answer_label_printed {
                out.push_str("\n\n[answer]\n");
                self.answer_label_printed = true;
            } else if !self.answer_label_printed && self.streamed_bytes > 0 {
                out.push('\n');
                self.answer_label_printed = true;
            }
            return out;
        }

        // 3. Suppressed thinking content never enters the decode buffer.
        if self.in_thinking && !self.show_thinking {
            return out;
        }

        // 4. Visible token: push to decode buffer, emit the delta.
        self.decode_buf.push(token_id);
        if let Ok(text) = tokenizer.decode(&self.decode_buf) {
            if text.len() > self.streamed_bytes {
                let idx = self.streamed_bytes.min(text.len());
                let start = (idx..=text.len())
                    .find(|&i| text.is_char_boundary(i))
                    .unwrap_or(text.len());
                if start < text.len() {
                    out.push_str(&text[start..]);
                }
                self.streamed_bytes = text.len();
            }
        }
        out
    }

    /// Returns true when no visible output has been emitted yet.
    fn is_at_start(&self) -> bool {
        self.streamed_bytes == 0
    }
}
