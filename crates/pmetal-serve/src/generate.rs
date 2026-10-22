//! Running a request's choices.
//!
//! A request asks for `n` replies. Each is a generation of its own on the
//! engine, with its own seed, and each is read through an
//! [`Assembler`]; this module starts them, hands back their event streams
//! to be merged into SSE frames, or drains them into finished replies for a
//! JSON body.

use std::sync::Arc;
use std::time::Instant;

use futures::Stream;
use futures::future::join_all;
use futures::stream::StreamExt;

use crate::assemble::{Assembler, ChoiceDone, ChoiceEvent, Failure, Piece, run_choice};
use crate::engine::{PreparedPrompt, SamplingParams, TokenLogprobEntry};
use crate::error::{ServeError, ServeResult};
use crate::request::{ResolvedStops, SamplingRequest, sampling_params};
use crate::routes::{AppState, chat_constraint, stream_tokens};
use crate::tools::{ParsedCall, ToolPlan};
use crate::types::{ChatMessage, ResponseFormat};
use pmetal_data::chat_templates::ChatTemplateKwargs;

/// A chat request made ready to generate: its prompt, and the sampling it
/// asks for with the grammar it must follow.
pub(crate) struct Prepared {
    pub prompt: PreparedPrompt,
    pub params: SamplingParams,
}

/// Render `messages` (with the tools of `plan`) into a prompt, compile the
/// grammar the request demands (a forced tool call or a `response_format`),
/// and settle the sampling parameters against the context that is left. The
/// three chat-shaped endpoints (chat completions, messages, responses) share
/// this.
pub(crate) async fn prepare(
    state: &Arc<AppState>,
    messages: &[ChatMessage],
    plan: &ToolPlan,
    kwargs: &ChatTemplateKwargs,
    sampling: &SamplingRequest<'_>,
    stops: &ResolvedStops,
    response_format: Option<&ResponseFormat>,
) -> ServeResult<Prepared> {
    let prompt = state
        .engine
        .prepare_chat(messages, plan.prompt.as_deref(), kwargs)
        .await?;
    let thinking = state.engine.thinks_with(kwargs);
    let constraint = chat_constraint(
        &state.engine,
        response_format,
        plan,
        &prompt.input_ids,
        thinking,
    )
    .await?;
    let mut params = sampling_params(
        &state.engine,
        sampling,
        stops,
        thinking,
        prompt.input_ids.len(),
    )?;
    params.constraint = constraint;
    state.engine.validate_sampling_params(&params)?;
    Ok(Prepared { prompt, params })
}

/// Start `n` generations of `prompt`; stream `i` is read through
/// `assemblers(i)`. Seeds step by one per choice, so a seeded request is
/// repeatable and its choices differ.
pub(crate) fn start_choices(
    state: &Arc<AppState>,
    prompt: &PreparedPrompt,
    params: &SamplingParams,
    n: usize,
    mut assembler: impl FnMut(usize) -> Assembler,
) -> Vec<impl Stream<Item = ChoiceEvent> + Send + 'static> {
    let deadline = state.deadline();
    (0..n)
        .map(|i| {
            let mut params = params.clone();
            params.seed = params.seed.map(|seed| seed.wrapping_add(i as u64));
            let rx = stream_tokens(&state.engine, prompt, params);
            let metrics = Arc::clone(state);
            run_choice(rx, assembler(i), deadline).inspect(move |event| {
                if let ChoiceEvent::Done(ChoiceDone {
                    metrics: Some(m), ..
                }) = event
                {
                    metrics.record_request(m);
                }
            })
        })
        .collect()
}

/// A finished reply.
#[derive(Debug, Default)]
pub(crate) struct Collected {
    pub reasoning: String,
    pub content: String,
    pub tool_calls: Vec<ParsedCall>,
    pub logprobs: Vec<TokenLogprobEntry>,
    pub done: Option<ChoiceDone>,
}

impl Collected {
    pub fn finish_reason(&self) -> &'static str {
        self.done.as_ref().map_or("stop", |d| d.finish.reason)
    }

    pub fn completion_tokens(&self) -> usize {
        self.done.as_ref().map_or(0, |d| d.finish.completion_tokens)
    }

    /// Prompt tokens the prefix cache supplied to this choice.
    pub fn cached_tokens(&self) -> usize {
        self.done
            .as_ref()
            .and_then(|d| d.metrics.as_ref())
            .map_or(0, |m| m.cached_prompt_tokens)
    }

    pub fn reasoning_tokens(&self) -> usize {
        self.done.as_ref().map_or(0, |d| d.finish.reasoning_tokens)
    }
}

impl From<Failure> for ServeError {
    fn from(failure: Failure) -> Self {
        match failure {
            Failure::Timeout(secs) => ServeError::Timeout(secs),
            Failure::Engine(message) => ServeError::Internal(message),
        }
    }
}

/// Drain one choice's events into its reply.
pub(crate) async fn collect(
    events: impl Stream<Item = ChoiceEvent>,
) -> Result<Collected, ServeError> {
    futures::pin_mut!(events);
    let mut reply = Collected::default();
    while let Some(event) = events.next().await {
        match event {
            ChoiceEvent::Piece(Piece::Reasoning(text)) => reply.reasoning.push_str(&text),
            ChoiceEvent::Piece(Piece::Content { text, logprobs }) => {
                reply.content.push_str(&text);
                reply.logprobs.extend(logprobs);
            }
            ChoiceEvent::Piece(Piece::ToolCall(call)) => reply.tool_calls.push(call),
            ChoiceEvent::Done(done) => reply.done = Some(done),
            ChoiceEvent::Failed(failure) => return Err(failure.into()),
        }
    }
    Ok(reply)
}

/// Drain every choice, concurrently, into replies in order.
pub(crate) async fn collect_all(
    streams: Vec<impl Stream<Item = ChoiceEvent> + Send + 'static>,
) -> Result<Vec<Collected>, ServeError> {
    join_all(streams.into_iter().map(collect))
        .await
        .into_iter()
        .collect()
}

/// A reply's usage, summed over its choices.
pub(crate) fn usage_of(
    prompt_tokens: usize,
    cached_tokens: usize,
    choices: impl IntoIterator<Item = (usize, usize)>,
) -> crate::types::Usage {
    let (completion, reasoning) = choices
        .into_iter()
        .fold((0, 0), |(c, r), (cc, rr)| (c + cc, r + rr));
    crate::types::Usage::new(prompt_tokens, completion, cached_tokens, reasoning)
}

/// How long a request has, from now.
pub(crate) fn deadline_after(timeout: Option<std::time::Duration>) -> Option<(Instant, u64)> {
    timeout.map(|t| (Instant::now() + t, t.as_secs()))
}
