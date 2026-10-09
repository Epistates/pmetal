//! Streaming output formatter for chat models with reasoning channels.
//!
//! Several modern instruct models (Gemma 4, Qwen 3, DeepSeek R1, GPT-OSS,
//! Phi 4 reasoning) emit a dedicated "thinking" channel alongside the
//! final answer. When a chat template has `enable_thinking=True`, the
//! decoded token stream is shaped like:
//!
//!   - Gemma 4: `<|channel>thought\n…reasoning…\n<channel|>…answer…`
//!   - Qwen 3 : `<think>…reasoning…</think>…answer…`
//!   - gpt-oss: `<|channel|>analysis<|message|>…reasoning…<|end|>`
//!     `<|start|>assistant<|channel|>final<|message|>…answer…`
//!
//! The channel markers are special tokens in some tokenizers (Gemma 4,
//! gpt-oss) and regular added tokens in others (Qwen 3's `<think>` decodes
//! as a literal string even with `skip_special_tokens=true`). Either way,
//! printed as decoded, the reasoning body runs inline with the final answer,
//! indistinguishable from the "real" response.
//!
//! [`ReasoningSplitter`] resolves the marker IDs up front from the tokenizer
//! and sorts every token into the reasoning channel, the answer, or the
//! markup between them, which nobody should see. A chat template may open
//! the reasoning itself (Qwen3.5's thinking prompt ends in `<think>\n`), so
//! the splitter can be [`prime`](ReasoningSplitter::prime)d with the prompt
//! to start in the state the prompt leaves. The CLI's [`StreamFormatter`]
//! and the server's `reasoning_content` both read the same splitter.

use crate::tokenizer::Tokenizer;

/// Names of reasoning-open marker tokens: every one that resolves to an id
/// in the tokenizer is recognised, so one splitter handles several
/// vocabularies.
const THINK_OPEN_NAMES: &[&str] = &[
    "<think>", // Qwen 3, DeepSeek R1, Phi-4-reasoning, Nemotron
    "[THINK]", // Magistral
];

/// Names of reasoning-close marker tokens.
const THINK_CLOSE_NAMES: &[&str] = &["</think>", "[/THINK]"];

/// Tokens that start a new turn: whatever the turn before left open, the
/// next one starts outside the reasoning.
const TURN_START_NAMES: &[&str] = &[
    "<|im_start|>",
    "<|turn>", // Gemma 4
    "<start_of_turn>",
    "<|start_header_id|>",
    "<｜User｜>",
    "<｜Assistant｜>",
];

/// Where a generated token belongs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Route {
    /// The model's reasoning.
    Reasoning,
    /// The answer.
    Answer,
    /// Channel markup (a marker, a channel header): shown to nobody.
    Markup,
}

/// A channel header being read: tokens after a channel-open marker that
/// name the channel rather than belong to it.
#[derive(Debug, Clone, Copy)]
enum Header {
    /// Gemma 4 `<|channel>NAME\n`: ends at the newline.
    Gemma { name: Option<u32>, len: usize },
    /// gpt-oss `<|start|>ROLE<|channel|>NAME …<|message|>`: ends at
    /// `<|message|>`. `name` is the first token after `<|channel|>`.
    Harmony {
        after_channel: bool,
        name: Option<u32>,
    },
}

/// Longest Gemma 4 channel header read before giving up on its newline.
const GEMMA_HEADER_MAX_TOKENS: usize = 4;

/// Sorts a chat model's tokens into reasoning, answer and markup.
#[derive(Debug, Clone)]
pub struct ReasoningSplitter {
    think_open: Vec<u32>,
    think_close: Vec<u32>,
    turn_starts: Vec<u32>,
    gemma_open: Option<u32>,
    gemma_close: Option<u32>,
    gemma_thought: Option<u32>,
    newline: Option<u32>,
    harmony_start: Option<u32>,
    harmony_channel: Option<u32>,
    harmony_message: Option<u32>,
    harmony_end: Option<u32>,
    harmony_analysis: Option<u32>,
    in_reasoning: bool,
    header: Option<Header>,
}

impl ReasoningSplitter {
    /// Resolve the marker tokens `tokenizer` has. A tokenizer with none sends
    /// every token to the answer.
    pub fn new(tokenizer: &Tokenizer) -> Self {
        let id = |name: &str| tokenizer.inner().token_to_id(name);
        let ids = |names: &[&str]| names.iter().filter_map(|n| id(n)).collect::<Vec<_>>();
        let gemma_open = id("<|channel>");
        let harmony_channel = id("<|channel|>");
        let splitter = Self {
            think_open: ids(THINK_OPEN_NAMES),
            think_close: ids(THINK_CLOSE_NAMES),
            turn_starts: ids(TURN_START_NAMES),
            gemma_open,
            gemma_close: id("<channel|>"),
            gemma_thought: gemma_open.and_then(|_| id("thought")),
            newline: id("\n").or_else(|| id("Ċ")),
            harmony_start: harmony_channel.and_then(|_| id("<|start|>")),
            harmony_channel,
            harmony_message: harmony_channel.and_then(|_| id("<|message|>")),
            harmony_end: harmony_channel.and_then(|_| id("<|end|>")),
            harmony_analysis: harmony_channel.and_then(|_| id("analysis")),
            in_reasoning: false,
            header: None,
        };
        tracing::debug!(
            think_open = ?splitter.think_open,
            think_close = ?splitter.think_close,
            gemma = ?splitter.gemma_open,
            harmony = ?splitter.harmony_channel,
            "ReasoningSplitter markers"
        );
        splitter
    }

    /// Run the prompt through the splitter, so generation starts in the
    /// state the prompt leaves: inside the reasoning when the chat template
    /// opened it, in a channel header when it opened one.
    pub fn prime(&mut self, prompt: &[u32]) {
        for &token in prompt {
            self.route(token);
        }
    }

    /// Whether the next token belongs to the reasoning (outside any header).
    pub fn in_reasoning(&self) -> bool {
        self.in_reasoning && self.header.is_none()
    }

    /// Where `token` belongs, advancing the state.
    pub fn route(&mut self, token: u32) -> Route {
        if let Some(header) = self.header.as_mut() {
            match header {
                Header::Gemma { .. } if Some(token) == self.gemma_close => {
                    self.in_reasoning = false;
                    self.header = None;
                }
                Header::Gemma { name, len } => {
                    name.get_or_insert(token);
                    *len += 1;
                    if Some(token) == self.newline || *len >= GEMMA_HEADER_MAX_TOKENS {
                        self.in_reasoning = name.is_some() && *name == self.gemma_thought;
                        self.header = None;
                    }
                }
                Header::Harmony {
                    after_channel,
                    name,
                } => {
                    if Some(token) == self.harmony_message {
                        self.in_reasoning = name.is_some_and(|n| Some(n) == self.harmony_analysis);
                        self.header = None;
                    } else if Some(token) == self.harmony_channel {
                        *after_channel = true;
                    } else if *after_channel && name.is_none() {
                        *name = Some(token);
                    }
                }
            }
            return Route::Markup;
        }

        if self.turn_starts.contains(&token) {
            self.in_reasoning = false;
            return Route::Markup;
        }
        if Some(token) == self.harmony_start || Some(token) == self.harmony_channel {
            self.in_reasoning = false;
            self.header = Some(Header::Harmony {
                after_channel: Some(token) == self.harmony_channel,
                name: None,
            });
            return Route::Markup;
        }
        if Some(token) == self.gemma_open {
            self.header = Some(Header::Gemma { name: None, len: 0 });
            return Route::Markup;
        }
        if self.think_open.contains(&token) {
            self.in_reasoning = true;
            return Route::Markup;
        }
        if self.think_close.contains(&token)
            || Some(token) == self.gemma_close
            || Some(token) == self.harmony_end
        {
            self.in_reasoning = false;
            return Route::Markup;
        }
        if self.in_reasoning {
            Route::Reasoning
        } else {
            Route::Answer
        }
    }
}

/// A finished generation split into its reasoning and its answer.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SplitOutput {
    /// The reasoning's tokens, markup excluded.
    pub reasoning: Vec<u32>,
    /// The answer's tokens, markup excluded.
    pub answer: Vec<u32>,
    /// Where each generated token went, in order.
    pub routes: Vec<Route>,
}

impl SplitOutput {
    /// Split `generated`, the tokens that followed `prompt`.
    pub fn split(tokenizer: &Tokenizer, prompt: &[u32], generated: &[u32]) -> Self {
        let mut splitter = ReasoningSplitter::new(tokenizer);
        splitter.prime(prompt);
        let mut out = Self::default();
        for &token in generated {
            let route = splitter.route(token);
            match route {
                Route::Reasoning => out.reasoning.push(token),
                Route::Answer => out.answer.push(token),
                Route::Markup => {}
            }
            out.routes.push(route);
        }
        out
    }

    /// The reasoning as text, trimmed; `None` when there was none.
    pub fn reasoning_text(&self, tokenizer: &Tokenizer) -> Option<String> {
        let text = tokenizer.decode(&self.reasoning).ok()?;
        let text = text.trim();
        (!text.is_empty()).then(|| text.to_string())
    }
}

/// Streaming formatter that folds reasoning-channel markers into visual
/// separators so the end-user never sees raw `<|channel>thought` tokens
/// in their terminal output.
pub struct StreamFormatter {
    splitter: ReasoningSplitter,
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
        Self {
            splitter: ReasoningSplitter::new(tokenizer),
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
        let route = self.splitter.route(token_id);
        let now_thinking = self.splitter.in_reasoning();

        // 1. Channel-enter: inject the label.
        if now_thinking && !self.in_thinking {
            self.in_thinking = true;
            if self.show_thinking && !self.thinking_label_printed {
                if !self.is_at_start() {
                    out.push('\n');
                }
                out.push_str("[thinking]\n");
                self.thinking_label_printed = true;
            }
        }

        // 2. Channel-exit: inject the label / newline.
        if !now_thinking && self.in_thinking && route != Route::Reasoning {
            self.in_thinking = false;
            if self.show_thinking && !self.answer_label_printed {
                out.push_str("\n\n[answer]\n");
                self.answer_label_printed = true;
            } else if !self.answer_label_printed && self.streamed_bytes > 0 {
                out.push('\n');
                self.answer_label_printed = true;
            }
        }

        // 3. Markup and suppressed thinking never enter the decode buffer.
        if route == Route::Markup || (route == Route::Reasoning && !self.show_thinking) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::testing::word_level as tokenizer;

    fn ids(tok: &Tokenizer, text: &str) -> Vec<u32> {
        tok.encode_with_special_tokens(text).unwrap()
    }

    fn split(tok: &Tokenizer, prompt: &str, generated: &str) -> (Option<String>, String) {
        let out = SplitOutput::split(tok, &ids(tok, prompt), &ids(tok, generated));
        (
            out.reasoning_text(tok),
            tok.decode(&out.answer).unwrap().trim().to_string(),
        )
    }

    const WORDS: &[&str] = &[
        "[UNK]",
        "user",
        "assistant",
        "hi",
        "plan",
        "add",
        "Paris",
        "4",
        "is",
        "thought",
        "analysis",
        "final",
    ];

    #[test]
    fn think_tags_split_whether_the_model_or_the_template_opens_them() {
        let tok = tokenizer(
            WORDS,
            &[
                ("<|im_start|>", true),
                ("<think>", false),
                ("</think>", false),
            ],
        );
        // Qwen3: the model opens the block.
        assert_eq!(
            split(
                &tok,
                "<|im_start|> user hi <|im_start|> assistant",
                "<think> plan add </think> Paris"
            ),
            (Some("plan add".into()), "Paris".into())
        );
        // Qwen3.5: the template opens it, the model only closes it.
        assert_eq!(
            split(
                &tok,
                "<|im_start|> assistant <think>",
                "plan </think> Paris"
            ),
            (Some("plan".into()), "Paris".into())
        );
        // Thinking off: the template closes an empty block.
        assert_eq!(
            split(&tok, "<|im_start|> assistant <think> </think>", "Paris"),
            (None, "Paris".into())
        );
        // A block left open in an earlier turn doesn't leak into this one.
        assert_eq!(
            split(
                &tok,
                "<|im_start|> user <think> hi <|im_start|> assistant",
                "Paris"
            ),
            (None, "Paris".into())
        );
        // Cut off mid-thought: all reasoning, no answer.
        assert_eq!(
            split(&tok, "<|im_start|> assistant <think>", "plan add"),
            (Some("plan add".into()), String::new())
        );
    }

    #[test]
    fn harmony_channels_split_on_their_names() {
        let tok = tokenizer(
            WORDS,
            &[
                ("<|start|>", true),
                ("<|channel|>", true),
                ("<|message|>", true),
                ("<|end|>", true),
                ("<|return|>", true),
            ],
        );
        assert_eq!(
            split(
                &tok,
                "<|start|> user <|message|> hi <|end|> <|start|> assistant",
                "<|channel|> analysis <|message|> plan add <|end|> \
                 <|start|> assistant <|channel|> final <|message|> 4"
            ),
            (Some("plan add".into()), "4".into())
        );
    }

    #[test]
    fn gemma_channel_header_names_the_thought() {
        let tok = tokenizer(
            WORDS,
            &[
                ("<|turn>", true),
                ("<|channel>", true),
                ("<channel|>", true),
                ("\n", false),
            ],
        );
        assert_eq!(
            split(
                &tok,
                "<|turn> user hi <|turn> assistant",
                "<|channel> thought \n plan <channel|> Paris"
            ),
            (Some("plan".into()), "Paris".into())
        );
        // Gemma 4's non-thinking prompt closes an empty thought channel.
        assert_eq!(
            split(
                &tok,
                "<|turn> assistant <|channel> thought \n <channel|>",
                "Paris"
            ),
            (None, "Paris".into())
        );
    }

    #[test]
    fn a_tokenizer_without_markers_sends_everything_to_the_answer() {
        let tok = tokenizer(WORDS, &[]);
        assert_eq!(
            split(&tok, "user hi", "Paris is 4"),
            (None, "Paris is 4".into())
        );
    }

    #[test]
    fn formatter_labels_and_hides_reasoning_from_the_splitter() {
        let tok = tokenizer(WORDS, &[("<think>", false), ("</think>", false)]);
        let render = |show| {
            let mut f = StreamFormatter::new(&tok, show);
            ids(&tok, "<think> plan </think> Paris")
                .into_iter()
                .map(|t| f.push_token(&tok, t))
                .collect::<String>()
        };
        assert_eq!(render(true), "[thinking]\nplan\n\n[answer]\n Paris");
        assert_eq!(render(false), "Paris");
    }
}
