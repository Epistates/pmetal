//! From a read example to text, and from text to tokens and labels.
//!
//! The loss mask is decided in bytes of the rendered text and carried to the
//! tokens by their offsets, so it cannot drift from what the tokenizer
//! actually produced: the old way (tokenize the prompt on its own, mask that
//! many tokens) was wrong whenever the prompt encoded differently alone than
//! in place, e.g. under a post-processor that appends an end-of-sequence
//! token to every encoding.

use std::borrow::Cow;
use std::ops::Range;
use std::sync::Once;

use pmetal_core::{PMetalError, Result};

use super::format::{Conversation, Example};
use crate::chat_templates::{ChatTemplate, Message};
use crate::tokenizer::Tokenizer;

use super::IGNORE_INDEX;

/// Which assistant turns of a conversation the loss covers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum AssistantLoss {
    /// Every assistant turn (the default): a conversation is several
    /// examples of the assistant answering, and training on one throws the
    /// others away.
    #[default]
    AllTurns,
    /// Only the final assistant turn; the turns before it are context.
    LastTurn,
}

/// A sample rendered to text, before tokenization.
#[derive(Debug, Clone)]
pub struct TextSample {
    /// The full text to tokenize.
    pub text: String,
    /// The part of `text` before the first trained byte (for SFT, the
    /// prompt); `None` when all of `text` is trained.
    pub prompt: Option<String>,
    /// Byte ranges of `text` the loss covers. `None` takes them from `prompt`:
    /// everything after it, or the whole text when there is none.
    pub trained: Option<Vec<Range<usize>>>,
    /// Whether a chat template rendered `text`. Its special tokens (BOS,
    /// end-of-turn) are then already in the text, so the tokenizer adds none
    /// of its own and no end-of-sequence token is appended.
    pub templated: bool,
}

impl TextSample {
    /// A document, trained whole.
    pub fn document(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            prompt: None,
            trained: None,
            templated: false,
        }
    }

    /// A prompt and the completion that follows it; only the completion is
    /// trained.
    pub fn completion(prompt: impl Into<String>, completion: &str) -> Self {
        let prompt = prompt.into();
        Self {
            text: format!("{prompt}{completion}"),
            prompt: Some(prompt),
            trained: None,
            templated: false,
        }
    }

    /// `text` with exactly the byte ranges `trained` carrying loss.
    pub fn with_spans(text: String, trained: Vec<Range<usize>>, templated: bool) -> Self {
        let prompt = trained
            .first()
            .map(|first| text[..first.start].to_string())
            .filter(|p| !p.is_empty());
        Self {
            text,
            prompt,
            trained: Some(trained),
            templated,
        }
    }

    /// The trained byte ranges, `None` for all of the text.
    fn spans(&self) -> Option<Cow<'_, [Range<usize>]>> {
        if let Some(spans) = &self.trained {
            return Some(Cow::Borrowed(spans));
        }
        let prompt = self.prompt.as_deref()?;
        self.text
            .starts_with(prompt)
            .then(|| Cow::Owned(std::iter::once(prompt.len()..self.text.len()).collect()))
    }
}

/// Render an example to text and the bytes of it the loss covers.
///
/// Usually one sample. A conversation whose chat template renders an earlier
/// assistant turn differently in context than alone (see
/// [`render_templated`]) yields one more for each such turn: the
/// conversation up to that turn, with the turn as the answer, which is
/// exactly how the template renders it for generation.
pub(crate) fn render(
    example: Example,
    template: Option<&ChatTemplate>,
    loss: AssistantLoss,
) -> Vec<TextSample> {
    match example {
        Example::Text(text) => vec![TextSample::document(text)],
        Example::Completion { prompt, completion } => {
            vec![TextSample::completion(prompt, &completion)]
        }
        Example::Chat(conversation) => render_chat(&conversation, template, loss),
    }
}

/// The indices of the messages the loss covers.
fn trained_messages(conversation: &Conversation, loss: AssistantLoss) -> Vec<usize> {
    let mut trained: Vec<usize> = conversation
        .messages
        .iter()
        .enumerate()
        .filter(|(i, m)| {
            m.role == "assistant" && conversation.trained.get(*i).copied().flatten() != Some(false)
        })
        .map(|(i, _)| i)
        .collect();
    if loss == AssistantLoss::LastTurn {
        // The last assistant turn of the conversation, whether or not it is
        // the last message.
        let last_assistant = conversation
            .messages
            .iter()
            .rposition(|m| m.role == "assistant");
        trained.retain(|&i| Some(i) == last_assistant);
    }
    trained
}

fn render_chat(
    conversation: &Conversation,
    template: Option<&ChatTemplate>,
    loss: AssistantLoss,
) -> Vec<TextSample> {
    let trained = trained_messages(conversation, loss);
    match template {
        Some(template) => render_templated(conversation, template, &trained),
        None => vec![render_plain(&conversation.messages, &trained)],
    }
}

/// Render through the chat template.
///
/// The template reports where the *last* message's text begins
/// ([`FormattedChat::response_start`](crate::chat_templates::FormattedChat)),
/// so assistant turn `i` is found by rendering the conversation up to it: its
/// text runs from there to the end of that rendering. That holds in the full
/// conversation only if the rendering is a prefix of it. A template that
/// rewrites earlier turns (Qwen3's drops the reasoning of turns before the
/// last question) breaks that, and the turn's bytes in the full text cannot
/// be told from guesses; rather than train on guesses or lose the turn, it is
/// rendered as a sample of its own, the conversation up to it, which is how
/// the template renders it when asked to generate it.
fn render_templated(
    conversation: &Conversation,
    template: &ChatTemplate,
    trained: &[usize],
) -> Vec<TextSample> {
    let tools = (!conversation.tools.is_empty()).then_some(conversation.tools.as_slice());
    let messages = &conversation.messages;
    let full = template.apply_with_tools(messages, tools);

    let mut spans: Vec<Range<usize>> = Vec::with_capacity(trained.len());
    let mut separate: Vec<TextSample> = Vec::new();
    for &i in trained {
        let (start, end) = if i + 1 == messages.len() {
            (full.response_start, full.text.len())
        } else {
            let upto = template.apply_with_tools(&messages[..=i], tools);
            if !full.text.starts_with(&upto.text) {
                static NOTE: Once = Once::new();
                NOTE.call_once(|| {
                    tracing::info!(
                        "the chat template does not render a conversation's earlier turns as \
                         it renders them alone; each such assistant turn becomes a sample of \
                         its own (train on the last turn only to avoid the extra samples)"
                    );
                });
                if upto.response_start < upto.text.len()
                    && upto.text.is_char_boundary(upto.response_start)
                {
                    let span = upto.response_start..upto.text.len();
                    separate.push(TextSample::with_spans(upto.text, vec![span], true));
                }
                continue;
            }
            (upto.response_start, upto.text.len())
        };
        if start < end && full.text.is_char_boundary(start) {
            spans.push(start..end);
        }
    }
    let mut samples = Vec::with_capacity(1 + separate.len());
    if !spans.is_empty() || separate.is_empty() {
        samples.push(TextSample::with_spans(full.text, spans, true));
    }
    samples.extend(separate);
    samples
}

/// Without a chat template: `Role: content` blocks, the assistant's content
/// and the blank line after it trained, its `Assistant: ` header not (that is
/// the prompt a model is given to answer).
fn render_plain(messages: &[Message], trained: &[usize]) -> TextSample {
    let mut text = String::new();
    let mut spans = Vec::new();
    for (i, message) in messages.iter().enumerate() {
        let label = match message.role.as_str() {
            "user" => "User",
            "assistant" => "Assistant",
            "system" => "System",
            "tool" => "Tool",
            other => other,
        };
        text.push_str(label);
        text.push_str(": ");
        let start = text.len();
        text.push_str(&message.content);
        if let Some(calls) = &message.tool_calls {
            if let Ok(calls) = serde_json::to_string(calls) {
                if !message.content.is_empty() {
                    text.push('\n');
                }
                text.push_str(&calls);
            }
        }
        text.push_str("\n\n");
        if trained.contains(&i) {
            spans.push(start..text.len());
        }
    }
    TextSample::with_spans(text, spans, false)
}

/// Render `messages` through `template` and tokenize them: the ids and the
/// labels (unshifted; [`IGNORE_INDEX`] outside the assistant turns `loss`
/// names), cut to `max_length`.
pub(crate) fn tokenize_messages(
    messages: &[Message],
    template: &ChatTemplate,
    tokenizer: &Tokenizer,
    max_length: usize,
    loss: AssistantLoss,
) -> Result<(Vec<u32>, Vec<i64>)> {
    let conversation = Conversation {
        messages: messages.to_vec(),
        tools: Vec::new(),
        trained: vec![None; messages.len()],
    };
    // The first sample holds the turns the template lets us locate in place:
    // all of them under `LastTurn`, which is what this is for.
    let sample = render_chat(&conversation, Some(template), loss)
        .into_iter()
        .next()
        .ok_or_else(|| PMetalError::InvalidArgument("a conversation with no messages".into()))?;
    let tokenized = tokenize(tokenizer, &sample, max_length)?;
    Ok((tokenized.input_ids, tokenized.labels))
}

/// A tokenized sample.
#[derive(Debug, Clone)]
pub(crate) struct Tokenized {
    pub input_ids: Vec<u32>,
    /// Aligned with `input_ids`: the token itself where the loss covers it,
    /// [`IGNORE_INDEX`] elsewhere. Unshifted; the loss shifts.
    pub labels: Vec<i64>,
    /// Length before truncation.
    pub original_len: usize,
}

impl Tokenized {
    /// Does the loss cover any token?
    pub fn has_trained_tokens(&self) -> bool {
        self.labels.iter().any(|&l| l != IGNORE_INDEX)
    }
}

/// Tokenize `sample`, label the tokens its trained spans cover, and cut to
/// `max_length`.
///
/// A token carries loss when it *starts* inside a trained span. (A token
/// that starts in the prompt and runs into the answer belongs to the prompt:
/// training on it would teach the model to produce text the prompt contains.)
/// Tokens the tokenizer adds that stand for no text (a BOS, an appended EOS)
/// follow the token before them, so a BOS is never trained and the EOS after a
/// trained answer is.
///
/// Text that was not rendered by a chat template is encoded with the
/// tokenizer's special tokens (BOS and so on) and given an end-of-sequence
/// token if it does not end in one; templated text is encoded as it is.
pub(crate) fn tokenize(
    tokenizer: &Tokenizer,
    sample: &TextSample,
    max_length: usize,
) -> Result<Tokenized> {
    let encoding = tokenizer
        .inner()
        .encode(sample.text.as_str(), !sample.templated)
        .map_err(|e| PMetalError::Tokenizer(e.to_string()))?;
    let mut ids = encoding.get_ids().to_vec();
    let mut offsets = encoding.get_offsets().to_vec();

    if !sample.templated {
        if let Some(eos) = tokenizer.eos_token_id() {
            if ids.last() != Some(&eos) {
                ids.push(eos);
                offsets.push((sample.text.len(), sample.text.len()));
            }
        }
    }

    let original_len = ids.len();
    ids.truncate(max_length);

    let labels = match sample.spans() {
        None => ids.iter().map(|&id| i64::from(id)).collect(),
        Some(spans) => {
            let mut carries = false;
            ids.iter()
                .zip(&offsets)
                .map(|(&id, &(start, end))| {
                    if end > start {
                        carries = spans.iter().any(|s| s.start <= start && start < s.end);
                    }
                    if carries { i64::from(id) } else { IGNORE_INDEX }
                })
                .collect()
        }
    };
    Ok(Tokenized {
        input_ids: ids,
        labels,
        original_len,
    })
}

#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)] // one trained span is the common case
mod tests {
    use super::*;

    #[test]
    fn a_prompt_alone_trains_what_follows_it() {
        let s = TextSample::completion("Q: ", "A");
        assert_eq!(s.spans().unwrap().into_owned(), [3..4]);
        assert_eq!(TextSample::document("x").spans(), None);
        // A prompt that is not a prefix of the text asks for nothing special.
        let odd = TextSample {
            text: "abc".into(),
            prompt: Some("zzz".into()),
            trained: None,
            templated: false,
        };
        assert_eq!(odd.spans(), None);
    }

    #[test]
    fn spans_name_the_prompt() {
        let s = TextSample::with_spans("hello world".into(), vec![6..11], false);
        assert_eq!(s.prompt.as_deref(), Some("hello "));
        let s = TextSample::with_spans("hello".into(), vec![0..5], false);
        assert_eq!(s.prompt, None);
    }
}
