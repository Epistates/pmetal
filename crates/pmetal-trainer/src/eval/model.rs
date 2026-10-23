//! What task evaluation asks of a model, and the model that answers it.
//!
//! Two operations are all a benchmark needs: the log-likelihood of a
//! continuation given a context (multiple-choice tasks), and a greedy
//! generation (generation tasks). [`EvalModel`] is those two, over token ids,
//! so the tasks' prompt construction and scoring are written against it and run
//! without a GPU in tests. [`LoadedModel`] is the real one: the inference
//! path's own [`DynamicModel`], with an adapter attached when asked.
//!
//! Scoring runs sequences in batches, longest first, right-padded: a causal
//! model's earlier positions never see padding that follows them, so a padded
//! row scores what it would alone. Choices that are one token each (a letter,
//! as MMLU's are) after one context are scored from a single forward of that
//! context.

use std::collections::HashMap;

use pmetal_lora::DynamicLoraModel;
use pmetal_models::{DynamicModel, GenerationConfig};
use pmetal_runtime::compat::indexing::take_along_axis;
use pmetal_runtime::compat::ops;
use pmetal_runtime::compat::{Array, Dtype};

use super::EvalError;
use crate::logprob_utils::selective_log_softmax;

/// A continuation to score after a context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlRequest {
    /// The context's tokens (not empty).
    pub context: Vec<u32>,
    /// The continuation's tokens (not empty).
    pub continuation: Vec<u32>,
}

/// How a model scored a continuation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LlResult {
    /// The sum of the continuation's tokens' log-probabilities given the
    /// context.
    pub logprob: f64,
    /// Whether greedy decoding from the context would produce exactly the
    /// continuation.
    pub greedy: bool,
}

/// A model that can be evaluated.
pub trait EvalModel {
    /// The log-likelihood of each request's continuation.
    ///
    /// # Errors
    ///
    /// [`EvalError::Model`] if the model fails.
    fn loglikelihood(&mut self, requests: &[LlRequest]) -> Result<Vec<LlResult>, EvalError>;

    /// Greedy generation from `prompt` of at most `max_new_tokens` tokens. After
    /// each token `stop` is asked with everything generated so far, and a `true`
    /// ends the generation. The tokens are returned, a stop token included.
    ///
    /// # Errors
    ///
    /// [`EvalError::Model`] if the model fails.
    fn generate(
        &mut self,
        prompt: &[u32],
        max_new_tokens: usize,
        stop: &mut dyn FnMut(&[u32]) -> bool,
    ) -> Result<Vec<u32>, EvalError>;
}

/// The model a run evaluates: a checkpoint, with an adapter or without.
enum Held {
    Base(DynamicModel),
    Adapted(DynamicLoraModel),
}

/// An evaluated model: [`DynamicModel`], the one `pmetal infer` and `serve`
/// run.
pub struct LoadedModel {
    model: Held,
    /// Most sequences in a scoring batch.
    batch_size: usize,
    /// Most tokens (sequences times the longest's length) in a scoring batch.
    token_budget: usize,
    /// The longest sequence scored; a longer one loses the start of its context.
    max_len: usize,
    /// Tokens that end a generation (the model's end-of-turn and end-of-text).
    stop_tokens: Vec<u32>,
}

/// Token id of the right padding, which no scored position reads.
const PAD: i32 = 0;
/// The label that marks a position as not scored.
const IGNORE: i32 = -100;

impl LoadedModel {
    /// Evaluate `model` as it is.
    pub fn base(model: DynamicModel) -> Self {
        Self::new(Held::Base(model))
    }

    /// Evaluate `model` with its adapter attached.
    pub fn adapted(model: DynamicLoraModel) -> Self {
        Self::new(Held::Adapted(model))
    }

    fn new(model: Held) -> Self {
        Self {
            model,
            batch_size: 16,
            token_budget: 8192,
            max_len: 8192,
            stop_tokens: Vec::new(),
        }
    }

    /// At most `size` sequences to a scoring batch.
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size.max(1);
        self
    }

    /// At most `tokens` (a batch's sequences times its longest's length) to a
    /// scoring batch: what bounds the logits' memory.
    pub fn with_token_budget(mut self, tokens: usize) -> Self {
        self.token_budget = tokens.max(1);
        self
    }

    /// Score no sequence longer than `len` tokens.
    pub fn with_max_len(mut self, len: usize) -> Self {
        self.max_len = len.max(2);
        self
    }

    /// The tokens that end a generation.
    pub fn with_stop_tokens(mut self, tokens: Vec<u32>) -> Self {
        self.stop_tokens = tokens;
        self
    }

    fn dynamic(&mut self) -> &mut DynamicModel {
        match &mut self.model {
            Held::Base(model) => model,
            Held::Adapted(model) => model.model_mut(),
        }
    }

    /// One request's tokens as scored: the context's end and the whole
    /// continuation, within [`max_len`](Self::with_max_len).
    fn fit(&self, request: &LlRequest) -> Result<(Vec<u32>, usize), EvalError> {
        if request.context.is_empty() || request.continuation.is_empty() {
            return Err(EvalError::Model(
                "a log-likelihood request needs a context and a continuation".into(),
            ));
        }
        let mut full: Vec<u32> = request.context.clone();
        full.extend(&request.continuation);
        let excess = full.len().saturating_sub(self.max_len);
        if excess >= request.context.len() {
            return Err(EvalError::Model(format!(
                "a continuation of {} tokens does not fit the {} token limit",
                request.continuation.len(),
                self.max_len
            )));
        }
        Ok((full[excess..].to_vec(), request.context.len() - excess))
    }

    /// The batches of `lengths` (indices into the caller's list) to run: longest
    /// first, within the batch and token limits.
    fn batches(&self, lengths: &[usize]) -> Vec<Vec<usize>> {
        let mut order: Vec<usize> = (0..lengths.len()).collect();
        order.sort_by(|&a, &b| lengths[b].cmp(&lengths[a]).then(a.cmp(&b)));
        let mut batches: Vec<Vec<usize>> = Vec::new();
        let mut current: Vec<usize> = Vec::new();
        let mut longest = 0;
        for index in order {
            let len = lengths[index];
            // Sorted longest first, so a batch's first entry is its longest.
            let tokens = current.first().map_or(len, |&first| lengths[first]) * (current.len() + 1);
            if !current.is_empty()
                && (current.len() >= self.batch_size || tokens > self.token_budget)
            {
                batches.push(std::mem::take(&mut current));
                longest = 0;
            }
            longest = longest.max(len);
            current.push(index);
        }
        let _ = longest;
        if !current.is_empty() {
            batches.push(current);
        }
        batches
    }

    /// Score requests whose continuation is more than one token, or whose
    /// context nobody else shares: a forward of the whole sequence.
    fn score_sequences(
        &mut self,
        sequences: &[(Vec<u32>, usize)],
        which: &[usize],
        out: &mut [LlResult],
    ) -> Result<(), EvalError> {
        let lengths: Vec<usize> = which.iter().map(|&i| sequences[i].0.len() - 1).collect();
        for batch in self.batches(&lengths) {
            let rows: Vec<usize> = batch.iter().map(|&b| which[b]).collect();
            let width = rows
                .iter()
                .map(|&r| sequences[r].0.len() - 1)
                .max()
                .unwrap_or(0);
            let mut inputs = vec![PAD; rows.len() * width];
            let mut labels = vec![IGNORE; rows.len() * width];
            for (b, &r) in rows.iter().enumerate() {
                let (full, context) = (&sequences[r].0, sequences[r].1);
                for (p, &token) in full[..full.len() - 1].iter().enumerate() {
                    inputs[b * width + p] = token as i32;
                }
                // Position p predicts token p + 1; the continuation starts at `context`.
                for p in context - 1..full.len() - 1 {
                    labels[b * width + p] = full[p + 1] as i32;
                }
            }
            let shape = [rows.len() as i32, width as i32];
            let inputs = Array::from_slice(&inputs, &shape);
            let labels = Array::from_slice(&labels, &shape);
            let logits = self
                .dynamic()
                .forward(&inputs, None)
                .map_err(|e| EvalError::Model(format!("forward pass failed: {e}")))?;
            let (logps, valid) = selective_log_softmax(&logits, &labels)
                .map_err(|e| EvalError::Model(e.to_string()))?;
            let total = ops::sum_axis(&logps, 1, false);
            let predicted = ops::argmax_axis(&logits, -1);
            let wrong = ops::sum_axis(
                &ops::not_equal(&predicted, &labels)
                    .as_dtype(Dtype::Float32.as_i32())
                    .multiply(&valid),
                1,
                false,
            );
            total.eval();
            wrong.eval();
            pmetal_runtime::check_unobserved_error()
                .map_err(|e| EvalError::Model(e.to_string()))?;
            let (total, wrong) = (total.as_slice::<f32>(), wrong.as_slice::<f32>());
            for (b, &r) in rows.iter().enumerate() {
                if !total[b].is_finite() {
                    return Err(EvalError::Model(format!(
                        "a sequence scored {} (a request's log-likelihood must be finite)",
                        total[b]
                    )));
                }
                out[r] = LlResult {
                    logprob: f64::from(total[b]),
                    greedy: wrong[b] == 0.0,
                };
            }
        }
        Ok(())
    }

    /// Score requests with a one-token continuation from one forward per
    /// distinct context: the logits at the context's last position hold every
    /// such continuation's score.
    fn score_next_tokens(
        &mut self,
        sequences: &[(Vec<u32>, usize)],
        which: &[usize],
        out: &mut [LlResult],
    ) -> Result<(), EvalError> {
        // The distinct contexts, and for each request the context's row.
        let mut rows: HashMap<&[u32], usize> = HashMap::new();
        let mut contexts: Vec<&[u32]> = Vec::new();
        let mut row_of: Vec<usize> = Vec::with_capacity(which.len());
        for &r in which {
            let context = &sequences[r].0[..sequences[r].1];
            let row = *rows.entry(context).or_insert_with(|| {
                contexts.push(context);
                contexts.len() - 1
            });
            row_of.push(row);
        }
        let lengths: Vec<usize> = contexts.iter().map(|c| c.len()).collect();
        for batch in self.batches(&lengths) {
            let width = batch.iter().map(|&c| contexts[c].len()).max().unwrap_or(0);
            let mut inputs = vec![PAD; batch.len() * width];
            let mut last = Vec::with_capacity(batch.len());
            for (b, &c) in batch.iter().enumerate() {
                for (p, &token) in contexts[c].iter().enumerate() {
                    inputs[b * width + p] = token as i32;
                }
                last.push(contexts[c].len() as i32 - 1);
            }
            let rows_in_batch: HashMap<usize, usize> =
                batch.iter().enumerate().map(|(b, &c)| (c, b)).collect();
            let inputs = Array::from_slice(&inputs, &[batch.len() as i32, width as i32]);
            let logits = self
                .dynamic()
                .forward(&inputs, None)
                .map_err(|e| EvalError::Model(format!("forward pass failed: {e}")))?;
            let vocab = logits.shape()[2];
            // The last context position of every row, [B, V], as f32 log-probs.
            let at = Array::from_slice(&last, &[batch.len() as i32, 1, 1]);
            let at = ops::broadcast_to(&at, &[batch.len() as i32, 1, vocab]);
            let picked = take_along_axis(&logits, &at, 1)
                .reshape(&[batch.len() as i32, vocab])
                .as_dtype(Dtype::Float32.as_i32());
            let log_probs = picked.subtract(&ops::logsumexp_axis_keepdims(&picked, -1, true));
            let best = ops::argmax_axis(&log_probs, -1);

            let wanted: Vec<(usize, usize)> = which
                .iter()
                .zip(&row_of)
                .filter_map(|(&r, &row)| rows_in_batch.get(&row).map(|&b| (r, b)))
                .collect();
            let batch_rows: Vec<i32> = wanted.iter().map(|&(_, b)| b as i32).collect();
            let tokens: Vec<i32> = wanted
                .iter()
                .map(|&(r, _)| sequences[r].0[sequences[r].1] as i32)
                .collect();
            let n = wanted.len() as i32;
            let rows_arr = Array::from_slice(&batch_rows, &[n]);
            let scores = take_along_axis(
                &ops::take_axis(&log_probs, &rows_arr, 0),
                &Array::from_slice(&tokens, &[n, 1]),
                1,
            )
            .reshape(&[n]);
            let greedy = ops::equal(
                &ops::take_axis(&best, &rows_arr, 0),
                &Array::from_slice(&tokens, &[n]).as_dtype(best.dtype().as_i32()),
            )
            .as_dtype(Dtype::Float32.as_i32());
            scores.eval();
            greedy.eval();
            pmetal_runtime::check_unobserved_error()
                .map_err(|e| EvalError::Model(e.to_string()))?;
            let (scores, greedy) = (scores.as_slice::<f32>(), greedy.as_slice::<f32>());
            for (i, &(r, _)) in wanted.iter().enumerate() {
                if !scores[i].is_finite() {
                    return Err(EvalError::Model(format!(
                        "a continuation scored {} (a log-likelihood must be finite)",
                        scores[i]
                    )));
                }
                out[r] = LlResult {
                    logprob: f64::from(scores[i]),
                    greedy: greedy[i] > 0.5,
                };
            }
        }
        Ok(())
    }
}

impl EvalModel for LoadedModel {
    fn loglikelihood(&mut self, requests: &[LlRequest]) -> Result<Vec<LlResult>, EvalError> {
        let sequences: Vec<(Vec<u32>, usize)> = requests
            .iter()
            .map(|request| self.fit(request))
            .collect::<Result<_, _>>()?;
        let mut out = vec![
            LlResult {
                logprob: f64::NEG_INFINITY,
                greedy: false
            };
            requests.len()
        ];
        // A continuation of one token, after a context some other request shares,
        // is read off that context's forward. Anything else is its own sequence.
        let mut shared: HashMap<&[u32], usize> = HashMap::new();
        for (full, context) in &sequences {
            if full.len() == context + 1 {
                *shared.entry(&full[..*context]).or_default() += 1;
            }
        }
        let (next, whole): (Vec<usize>, Vec<usize>) = (0..requests.len()).partition(|&r| {
            let (full, context) = &sequences[r];
            full.len() == context + 1 && shared[&full[..*context]] > 1
        });
        if !next.is_empty() {
            self.score_next_tokens(&sequences, &next, &mut out)?;
        }
        if !whole.is_empty() {
            self.score_sequences(&sequences, &whole, &mut out)?;
        }
        Ok(out)
    }

    fn generate(
        &mut self,
        prompt: &[u32],
        max_new_tokens: usize,
        stop: &mut dyn FnMut(&[u32]) -> bool,
    ) -> Result<Vec<u32>, EvalError> {
        if prompt.is_empty() {
            return Err(EvalError::Model("an empty prompt".into()));
        }
        let config =
            GenerationConfig::greedy(max_new_tokens).with_stop_tokens(self.stop_tokens.clone());
        let mut generated: Vec<u32> = Vec::new();
        self.dynamic()
            .generate_streaming(prompt, config, |token| {
                generated.push(token);
                !stop(&generated)
            })
            .map_err(|e| EvalError::Model(format!("generation failed: {e}")))?;
        Ok(generated)
    }
}
