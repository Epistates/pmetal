//! Logits processors: the stage of the sampler between the model's logits
//! and the temperature and truncation filters, where a generation's
//! constraints live.
//!
//! A [`LogitsProcessor`] rewrites the `[1, vocab]` f32 logits of one step.
//! [`DecodeSampler`](crate::decode::DecodeSampler) runs its processors after
//! the repetition, frequency and presence penalties and before
//! [`filtered_log_probs`](crate::decode::filtered_log_probs), the order of
//! transformers' `LogitsProcessorList` followed by its warpers, so a mask that
//! rules a token out holds whatever the temperature or top-p does after it.
//!
//! Two processors ship: [`LogitBias`] (the OpenAI `logit_bias` parameter) and
//! [`AllowedTokens`] (a token whitelist). A grammar or JSON-schema engine
//! plugs in through the same trait: it computes the legal-token set of its
//! state in [`process`](LogitsProcessor::process), advances the state in
//! [`accept`](LogitsProcessor::accept), and snapshots it in
//! [`fork`](LogitsProcessor::fork) so a speculative decoder can follow its
//! drafts on a copy.

use crate::InlineArray;
use crate::compat::Dtype;

/// One step of constraint on a generation's logits. See the module docs.
pub trait LogitsProcessor {
    /// `[1, vocab]` f32 logits with this processor's constraint applied.
    /// Called once per distribution the sampler builds, in the order of
    /// the generation's tokens; a processor that tracks the generation
    /// reads it from [`accept`](Self::accept).
    fn process(&mut self, logits_2d: &InlineArray) -> InlineArray;

    /// The generation emitted `token`. Called, in order, for every token the
    /// sampler records, when [`needs_host_tokens`](Self::needs_host_tokens).
    fn accept(&mut self, _token: u32) {}

    /// Whether [`process`](Self::process) depends on the tokens
    /// [`accept`](Self::accept) was given. The sampler then reads each token
    /// back before the next step instead of recording it lazily, which gives
    /// up the decode pipeline's overlap of sampling with the next forward.
    fn needs_host_tokens(&self) -> bool {
        false
    }

    /// A copy that continues independently, for a speculative decoder that
    /// follows its drafts on a fork and keeps only what the target accepts.
    fn fork(&self) -> Box<dyn LogitsProcessor>;
}

/// A dense `[1, vocab]` f32 row built on the first logits, which is the
/// first time the vocabulary size is known.
#[derive(Clone, Default)]
struct DenseRow(Option<InlineArray>);

impl DenseRow {
    fn get(&mut self, vocab: i32, build: impl FnOnce(&mut [f32])) -> &InlineArray {
        if self.0.as_ref().is_none_or(|row| row.dim(-1) != vocab) {
            let mut values = vec![0.0f32; vocab.max(0) as usize];
            build(&mut values);
            self.0 = Some(InlineArray::from_f32_slice(&values, &[1, vocab]));
        }
        self.0.as_ref().expect("just built")
    }
}

fn as_f32(logits: &InlineArray) -> InlineArray {
    logits.as_dtype(Dtype::Float32.as_i32())
}

/// Added to chosen tokens' logits before sampling: OpenAI's `logit_bias`
/// (`-100` all but bans a token, `100` all but forces it; any finite value
/// works). A token the logits' vocabulary does not cover is ignored, and a
/// token listed twice takes its last bias.
#[derive(Clone)]
pub struct LogitBias {
    entries: Vec<(u32, f32)>,
    row: DenseRow,
}

impl LogitBias {
    /// The bias of `entries`, `(token, bias)`; `None` when there is none to
    /// apply.
    pub fn new(entries: &[(u32, f32)]) -> Option<Self> {
        let entries: Vec<(u32, f32)> = entries.iter().copied().filter(|&(_, b)| b != 0.0).collect();
        (!entries.is_empty()).then(|| Self {
            entries,
            row: DenseRow::default(),
        })
    }
}

impl LogitsProcessor for LogitBias {
    fn process(&mut self, logits_2d: &InlineArray) -> InlineArray {
        let logits = as_f32(logits_2d);
        let entries = &self.entries;
        let row = self.row.get(logits.dim(-1), |values| {
            for &(token, bias) in entries {
                if let Some(slot) = values.get_mut(token as usize) {
                    *slot = bias;
                }
            }
        });
        logits.add(row)
    }

    fn fork(&self) -> Box<dyn LogitsProcessor> {
        Box::new(self.clone())
    }
}

/// Rules out every token but the ones allowed: their logits are untouched
/// and every other token's goes to `-inf`, so a generation can only emit
/// from the set. A grammar engine that has the legal tokens of its state as
/// a list sets them with [`set`](Self::set) between steps.
///
/// An empty set, or one holding no token of the vocabulary, would leave
/// nothing to draw; it is left as it is rather than masking everything.
#[derive(Clone)]
pub struct AllowedTokens {
    allowed: Vec<u32>,
    row: DenseRow,
}

impl AllowedTokens {
    pub fn new(allowed: &[u32]) -> Self {
        let mut this = Self {
            allowed: Vec::new(),
            row: DenseRow::default(),
        };
        this.set(allowed);
        this
    }

    /// Replace the allowed set.
    pub fn set(&mut self, allowed: &[u32]) {
        self.allowed = allowed.to_vec();
        self.row = DenseRow::default();
    }
}

impl LogitsProcessor for AllowedTokens {
    fn process(&mut self, logits_2d: &InlineArray) -> InlineArray {
        let logits = as_f32(logits_2d);
        let allowed = &self.allowed;
        let vocab = logits.dim(-1);
        if !allowed.iter().any(|&t| (t as i64) < vocab as i64) {
            return logits;
        }
        let row = self.row.get(vocab, |values| {
            values.fill(f32::NEG_INFINITY);
            for &token in allowed {
                if let Some(slot) = values.get_mut(token as usize) {
                    *slot = 0.0;
                }
            }
        });
        logits.add(row)
    }

    fn fork(&self) -> Box<dyn LogitsProcessor> {
        Box::new(self.clone())
    }
}
