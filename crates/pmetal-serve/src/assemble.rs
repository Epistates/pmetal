//! From generated tokens to a reply.
//!
//! An [`Assembler`] takes one choice's tokens as the engine produces them and
//! gives back what a client is shown, in order: the model's reasoning, its
//! answer text, and its tool calls. Every endpoint, streamed or not, builds
//! its reply from these [`Piece`]s, so what counts as the answer is decided
//! in one place:
//!
//! 1. the tokens are decoded to text, UTF-8 safe, the reasoning split from
//!    the answer (`ChannelDecoder`);
//! 2. the answer is cut at the first stop sequence, in the text, however the
//!    tokens fell ([`StopFilter`]);
//! 3. when the request offered tools, the answer is split into text and
//!    parsed calls ([`ToolStream`]), so tool-call markup is never content;
//! 4. the logprobs of the tokens behind each piece of answer text travel
//!    with it, and only for text the client is shown.
//!
//! [`run_choice`] drives an assembler from the engine's event channel and
//! ends the generation as soon as a stop sequence completes or the client has
//! gone.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;

use futures::stream::{self, Stream};
use tokio::sync::mpsc::Receiver;

use crate::engine::{RequestMetrics, TokenEvent, TokenLogprobEntry};
use crate::sse::{ChannelDecoder, ChannelDelta, HarmonyDecoder, ReplyDecoder};
use crate::stop::StopFilter;
use crate::tools::{ParsedCall, ToolEvent, ToolFormat, ToolSet, ToolStream};

/// What an [`Assembler`] is told about the request.
pub struct AssemblerConfig {
    pub tokenizer: Arc<pmetal_data::Tokenizer>,
    /// The prompt's tokens: their markers say whether generation starts in
    /// a reasoning block.
    pub prompt: Vec<u32>,
    pub stops: Vec<String>,
    /// The tool-call family and tools offered, when the request offered any.
    pub tools: Option<(ToolFormat, ToolSet)>,
    /// Keep only the first tool call (`parallel_tool_calls: false`).
    pub single_tool_call: bool,
}

/// A step of a reply.
#[derive(Debug, Clone)]
pub enum Piece {
    Reasoning(String),
    /// Answer text, with the logprobs of the tokens behind it.
    Content {
        text: String,
        logprobs: Vec<TokenLogprobEntry>,
    },
    ToolCall(ParsedCall),
}

/// How a choice ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finish {
    /// `stop`, `length` or `tool_calls`.
    pub reason: &'static str,
    /// The stop sequence the reply ended at, when it did.
    pub stop_sequence: Option<String>,
    /// Tokens generated, the one that completed a stop sequence included.
    pub completion_tokens: usize,
    /// Of those, the tokens of reasoning.
    pub reasoning_tokens: usize,
}

pub struct Assembler {
    decoder: ReplyDecoder<Option<TokenLogprobEntry>>,
    filter: StopFilter,
    tools: Option<ToolStream>,
    single_tool_call: bool,
    tokens: usize,
    /// Bytes of answer text decoded, and bytes the stop filter has released.
    answer_in: usize,
    released: usize,
    /// Logprobs of decoded answer text, by `(start, end)` byte range of the
    /// text, waiting for the filter to release it.
    batches: VecDeque<(usize, usize, Vec<TokenLogprobEntry>)>,
    /// Logprobs released with text that produced no content piece (it went
    /// into a tool call or is held), for the next content piece.
    carry: Vec<TokenLogprobEntry>,
    calls: usize,
}

impl Assembler {
    pub fn new(config: AssemblerConfig) -> Self {
        let keep_special = config.tools.is_some();
        // Harmony's calls are messages the decoder reads itself; every other
        // family writes them into the answer text for the tool stream.
        let harmony = matches!(config.tools, Some((ToolFormat::Harmony, _)));
        let decoder = if harmony {
            ReplyDecoder::Harmony(HarmonyDecoder::new(config.tokenizer))
        } else {
            ReplyDecoder::Channels(
                ChannelDecoder::new(config.tokenizer, &config.prompt)
                    .keeping_special_tokens(keep_special),
            )
        };
        Self {
            decoder,
            filter: StopFilter::new(&config.stops),
            tools: config
                .tools
                .filter(|_| !harmony)
                .map(|(format, tools)| ToolStream::new(format, tools)),
            single_tool_call: config.single_tool_call,
            tokens: 0,
            answer_in: 0,
            released: 0,
            batches: VecDeque::new(),
            carry: Vec::new(),
            calls: 0,
        }
    }

    /// Whether a stop sequence has been seen: generation can end.
    pub fn is_done(&self) -> bool {
        self.filter.stopped()
    }

    /// Take in the next generated token (and its logprobs, if asked for).
    pub fn push(&mut self, token: u32, logprob: Option<TokenLogprobEntry>) -> Vec<Piece> {
        let mut out = Vec::new();
        if self.filter.stopped() {
            return out;
        }
        self.tokens += 1;
        let delta = self.decoder.push_with_aux(token, logprob);
        self.take(delta, &mut out);
        out
    }

    /// The generation is over (`engine_reason` is why the engine stopped):
    /// release what is held and say how the choice ended.
    pub fn finish(&mut self, engine_reason: &str) -> (Vec<Piece>, Finish) {
        let mut out = Vec::new();
        if !self.filter.stopped() {
            let delta = self.decoder.flush_aux();
            self.take(delta, &mut out);
        }
        if !self.filter.stopped() {
            let rest = self.filter.finish();
            self.release(rest, &mut out);
        }
        if let Some(tools) = self.tools.as_mut() {
            for event in tools.finish() {
                Self::tool_event(
                    event,
                    &mut self.carry,
                    &mut self.calls,
                    self.single_tool_call,
                    &mut out,
                );
            }
        }
        let reason = if self.calls > 0 {
            "tool_calls"
        } else if self.filter.stopped() || engine_reason == "stop" {
            "stop"
        } else if engine_reason == "length" {
            "length"
        } else {
            // Cancelled: nobody is reading.
            "stop"
        };
        (
            out,
            Finish {
                reason,
                stop_sequence: self.filter.matched().map(str::to_owned),
                completion_tokens: self.tokens,
                reasoning_tokens: self.decoder.reasoning_token_count(),
            },
        )
    }

    fn take(&mut self, delta: ChannelDelta<Option<TokenLogprobEntry>>, out: &mut Vec<Piece>) {
        if !delta.reasoning.is_empty() {
            out.push(Piece::Reasoning(delta.reasoning));
        }
        for call in delta.calls {
            if self.single_tool_call && self.calls >= 1 {
                continue;
            }
            self.calls += 1;
            out.push(Piece::ToolCall(call));
        }
        if delta.content.is_empty() {
            return;
        }
        let start = self.answer_in;
        self.answer_in += delta.content.len();
        let entries: Vec<TokenLogprobEntry> = delta.aux.into_iter().flatten().collect();
        if !entries.is_empty() {
            self.batches.push_back((start, self.answer_in, entries));
        }
        let released = self.filter.push(&delta.content);
        self.release(released, out);
    }

    /// Text the stop filter let through: it goes to the tool splitter, and
    /// the logprobs of the tokens it covers go with it.
    fn release(&mut self, text: String, out: &mut Vec<Piece>) {
        self.released += text.len();
        // A batch is the logprobs of the tokens of one decoded delta; it is
        // shown once all its text is. When a stop sequence cut the text, the
        // batch it cut into is shown too: its token was generated.
        let stopped = self.filter.stopped();
        while let Some((start, end, _)) = self.batches.front() {
            if *end <= self.released || (stopped && *start < self.released) {
                let (.., entries) = self.batches.pop_front().expect("just seen");
                self.carry.extend(entries);
            } else {
                break;
            }
        }
        if stopped {
            self.batches.clear();
        }
        if text.is_empty() {
            return;
        }
        match self.tools.as_mut() {
            None => {
                let logprobs = std::mem::take(&mut self.carry);
                out.push(Piece::Content { text, logprobs });
            }
            Some(tools) => {
                for event in tools.push(&text) {
                    Self::tool_event(
                        event,
                        &mut self.carry,
                        &mut self.calls,
                        self.single_tool_call,
                        out,
                    );
                }
            }
        }
    }

    fn tool_event(
        event: ToolEvent,
        carry: &mut Vec<TokenLogprobEntry>,
        calls: &mut usize,
        single: bool,
        out: &mut Vec<Piece>,
    ) {
        match event {
            ToolEvent::Text(text) => out.push(Piece::Content {
                text,
                logprobs: std::mem::take(carry),
            }),
            ToolEvent::Call(call) => {
                if single && *calls >= 1 {
                    return;
                }
                *calls += 1;
                out.push(Piece::ToolCall(call));
            }
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Driving an assembler from the engine
// ────────────────────────────────────────────────────────────────────────────

/// Why a choice failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Failure {
    /// The request ran past the server's time limit (seconds).
    Timeout(u64),
    /// The engine reported an error, or ended without a result.
    Engine(String),
}

/// How a finished choice is accounted.
#[derive(Debug, Clone)]
pub struct ChoiceDone {
    pub finish: Finish,
    /// The engine's timing for the request, when it reported any.
    pub metrics: Option<RequestMetrics>,
}

#[derive(Debug, Clone)]
pub enum ChoiceEvent {
    Piece(Piece),
    Done(ChoiceDone),
    Failed(Failure),
}

struct Run {
    rx: Option<Receiver<TokenEvent>>,
    assembler: Assembler,
    queue: VecDeque<ChoiceEvent>,
    deadline: Option<(Instant, u64)>,
    finished: bool,
}

/// The events of one choice: its pieces as they come, then `Done` (or
/// `Failed`). Dropping the stream drops the engine's channel, which cancels
/// the generation; so does a stop sequence completing, which ends the stream
/// at once.
pub fn run_choice(
    rx: Receiver<TokenEvent>,
    assembler: Assembler,
    deadline: Option<(Instant, u64)>,
) -> impl Stream<Item = ChoiceEvent> + Send + 'static {
    let run = Run {
        rx: Some(rx),
        assembler,
        queue: VecDeque::new(),
        deadline,
        finished: false,
    };
    stream::unfold(run, |mut run| async move {
        loop {
            if let Some(event) = run.queue.pop_front() {
                return Some((event, run));
            }
            if run.finished {
                return None;
            }
            let rx = run.rx.as_mut()?;
            let received = match run.deadline {
                Some((at, secs)) => match tokio::time::timeout_at(at.into(), rx.recv()).await {
                    Ok(received) => received,
                    Err(_) => {
                        run.finished = true;
                        run.rx = None;
                        run.queue
                            .push_back(ChoiceEvent::Failed(Failure::Timeout(secs)));
                        continue;
                    }
                },
                None => rx.recv().await,
            };
            match received {
                Some(TokenEvent::Token { id, logprob }) => {
                    let pieces = run.assembler.push(id, logprob);
                    run.queue.extend(pieces.into_iter().map(ChoiceEvent::Piece));
                    if run.assembler.is_done() {
                        // The stop sequence is complete: nothing more is
                        // wanted, and the engine need not go on.
                        run.rx = None;
                        run.finish("stop", None);
                    }
                }
                Some(TokenEvent::Done {
                    finish_reason,
                    metrics,
                    ..
                }) => {
                    run.rx = None;
                    run.finish(&finish_reason, Some(metrics));
                }
                Some(TokenEvent::Error(message)) => {
                    run.finished = true;
                    run.rx = None;
                    run.queue
                        .push_back(ChoiceEvent::Failed(Failure::Engine(message)));
                }
                None => {
                    run.finished = true;
                    run.rx = None;
                    run.queue.push_back(ChoiceEvent::Failed(Failure::Engine(
                        "generation ended without a result".into(),
                    )));
                }
            }
        }
    })
}

impl Run {
    fn finish(&mut self, engine_reason: &str, metrics: Option<RequestMetrics>) {
        let (pieces, finish) = self.assembler.finish(engine_reason);
        self.queue
            .extend(pieces.into_iter().map(ChoiceEvent::Piece));
        self.queue
            .push_back(ChoiceEvent::Done(ChoiceDone { finish, metrics }));
        self.finished = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_data::tokenizer::testing::byte_level_bpe;

    fn tokenizer() -> Arc<pmetal_data::Tokenizer> {
        Arc::new(byte_level_bpe(
            &["h i"],
            &[
                "<|im_start|>",
                "<think>",
                "</think>",
                "<tool_call>",
                "</tool_call>",
            ],
            None,
        ))
    }

    fn assembler(stops: &[&str], tools: bool) -> Assembler {
        let tok = tokenizer();
        let prompt = tok.encode_with_special_tokens("<|im_start|>").unwrap();
        Assembler::new(AssemblerConfig {
            tokenizer: tok,
            prompt,
            stops: stops.iter().map(|s| s.to_string()).collect(),
            tools: tools.then(|| (ToolFormat::Hermes, ToolSet::default())),
            single_tool_call: false,
        })
    }

    fn entry(token: u32) -> TokenLogprobEntry {
        TokenLogprobEntry {
            token,
            logprob: -1.0,
            top_logprobs: Vec::new(),
        }
    }

    /// Feed `text`, one byte per token, and collect the answer.
    fn run(asm: &mut Assembler, text: &str, engine_reason: &str) -> (String, Vec<u32>, Finish) {
        let tok = tokenizer();
        let mut content = String::new();
        let mut logprob_tokens = Vec::new();
        let mut take = |pieces: Vec<Piece>| {
            for piece in pieces {
                if let Piece::Content { text, logprobs } = piece {
                    content.push_str(&text);
                    logprob_tokens.extend(logprobs.iter().map(|e| e.token));
                }
            }
        };
        for id in tok.encode(text).unwrap() {
            take(asm.push(id, Some(entry(id))));
            if asm.is_done() {
                break;
            }
        }
        let (rest, finish) = asm.finish(engine_reason);
        take(rest);
        (content, logprob_tokens, finish)
    }

    #[test]
    fn a_stop_sequence_cuts_the_text_and_ends_the_choice() {
        let mut asm = assembler(&["END"], false);
        let (content, _, finish) = run(&mut asm, "say hi END and more", "length");
        assert_eq!(content, "say hi ");
        assert_eq!(finish.reason, "stop");
        assert!(asm.is_done());
        // Tokens generated: up to and including the one that completed it.
        assert!(finish.completion_tokens < "say hi END and more".len());
    }

    #[test]
    fn without_a_stop_the_engine_reason_stands() {
        let mut asm = assembler(&["END"], false);
        let (content, _, finish) = run(&mut asm, "say hi", "length");
        assert_eq!(content, "say hi");
        assert_eq!(finish.reason, "length");
        let mut asm = assembler(&[], false);
        assert_eq!(run(&mut asm, "ok", "stop").2.reason, "stop");
    }

    #[test]
    fn logprobs_follow_only_the_text_the_client_sees() {
        let tok = tokenizer();
        let mut asm = assembler(&["END"], false);
        let (content, logprob_tokens, _) = run(&mut asm, "ab END", "length");
        assert_eq!(content, "ab ");
        let expect: Vec<u32> = tok.encode("ab ").unwrap();
        // The tokens behind "ab " (END's first byte sits in the held tail).
        assert_eq!(logprob_tokens, expect);
    }

    #[test]
    fn tool_markup_is_not_content() {
        let mut asm = assembler(&[], true);
        let text = "<tool_call>{\"name\": \"f\", \"arguments\": {}}</tool_call>";
        let tok = tokenizer();
        let mut ids = tok.encode_with_special_tokens(text).unwrap();
        // The test tokenizer holds the tags as special tokens.
        ids.retain(|_| true);
        let mut pieces = Vec::new();
        for id in ids {
            pieces.extend(asm.push(id, None));
        }
        let (rest, finish) = asm.finish("stop");
        pieces.extend(rest);
        assert!(
            pieces
                .iter()
                .all(|p| !matches!(p, Piece::Content { text, .. } if !text.trim().is_empty())),
            "{pieces:?}"
        );
        let calls: Vec<_> = pieces
            .iter()
            .filter_map(|p| match p {
                Piece::ToolCall(c) => Some(c.name.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(calls, vec!["f".to_string()]);
        assert_eq!(finish.reason, "tool_calls");
    }

    #[test]
    fn reasoning_is_split_off_and_counted() {
        let tok = tokenizer();
        let prompt = tok
            .encode_with_special_tokens("<|im_start|><think>")
            .unwrap();
        let mut asm = Assembler::new(AssemblerConfig {
            tokenizer: Arc::clone(&tok),
            prompt,
            stops: Vec::new(),
            tools: None,
            single_tool_call: false,
        });
        let mut reasoning = String::new();
        let mut content = String::new();
        for id in tok.encode_with_special_tokens("hmm</think>hi").unwrap() {
            for piece in asm.push(id, None) {
                match piece {
                    Piece::Reasoning(t) => reasoning.push_str(&t),
                    Piece::Content { text, .. } => content.push_str(&text),
                    Piece::ToolCall(_) => unreachable!(),
                }
            }
        }
        let (rest, finish) = asm.finish("stop");
        for piece in rest {
            if let Piece::Content { text, .. } = piece {
                content.push_str(&text);
            }
        }
        assert_eq!(reasoning, "hmm");
        assert_eq!(content, "hi");
        assert!(finish.reasoning_tokens >= 3);
    }

    // ── gpt-oss: Harmony messages ────────────────────────────────────────────

    fn harmony_tokenizer() -> Arc<pmetal_data::Tokenizer> {
        Arc::new(byte_level_bpe(
            &[],
            &[
                "<|start|>",
                "<|end|>",
                "<|channel|>",
                "<|message|>",
                "<|call|>",
                "<|return|>",
                "<|constrain|>",
            ],
            None,
        ))
    }

    /// Run `text` through a Harmony assembler; returns reasoning, content, calls and how it finished.
    fn harmony(text: &str) -> (String, String, Vec<ParsedCall>, Finish) {
        let tok = harmony_tokenizer();
        let prompt = tok
            .encode_with_special_tokens("<|start|>assistant")
            .unwrap();
        let tools = ToolSet::new(&[]);
        let mut asm = Assembler::new(AssemblerConfig {
            tokenizer: Arc::clone(&tok),
            prompt,
            stops: Vec::new(),
            tools: Some((ToolFormat::Harmony, tools)),
            single_tool_call: false,
        });
        let ids = tok.encode_with_special_tokens(text).unwrap();
        let (mut reasoning, mut content, mut calls) = (String::new(), String::new(), Vec::new());
        let mut take = |pieces: Vec<Piece>| {
            for piece in pieces {
                match piece {
                    Piece::Reasoning(t) => reasoning.push_str(&t),
                    Piece::Content { text, .. } => content.push_str(&text),
                    Piece::ToolCall(c) => calls.push(c),
                }
            }
        };
        for id in ids {
            take(asm.push(id, None));
        }
        let (rest, finish) = asm.finish("stop");
        take(rest);
        (reasoning, content, calls, finish)
    }

    #[test]
    fn harmony_channels_split_into_reasoning_and_answer() {
        let (reasoning, content, calls, finish) = harmony(
            "<|channel|>analysis<|message|>think hard<|end|><|start|>assistant\
             <|channel|>final<|message|>Hi there<|return|>",
        );
        assert_eq!(reasoning, "think hard");
        assert_eq!(content, "Hi there");
        assert!(calls.is_empty());
        assert_eq!(finish.reason, "stop");
    }

    #[test]
    fn a_harmony_function_call_is_a_call_not_text() {
        for text in [
            // The engine stops at `<|call|>` before emitting it ...
            "<|channel|>analysis<|message|>need weather<|end|><|start|>assistant\
             <|channel|>commentary to=functions.get_weather <|constrain|>json\
             <|message|>{\"city\": \"SF\"}",
            // ... and a model that does emit it ends the message there.
            "<|channel|>commentary to=functions.get_weather <|constrain|>json\
             <|message|>{\"city\": \"SF\"}<|call|>",
        ] {
            let (_, content, calls, finish) = harmony(text);
            assert_eq!(content, "", "{text}");
            assert_eq!(
                calls,
                vec![ParsedCall {
                    name: "get_weather".into(),
                    arguments: r#"{"city":"SF"}"#.into()
                }]
            );
            assert_eq!(finish.reason, "tool_calls");
        }
    }

    #[test]
    fn a_commentary_preamble_is_text() {
        let (_, content, calls, _) = harmony(
            "<|channel|>commentary<|message|>Let me check.<|end|><|start|>assistant\
             <|channel|>commentary to=functions.f <|constrain|>json<|message|>{}<|call|>",
        );
        assert_eq!(content, "Let me check.");
        assert_eq!(calls.len(), 1);
    }
}
