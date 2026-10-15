//! DFlash speculative-decoding loop.
//!
//! Glues a DFlash draft model of either generation
//! ([`crate::dflash_drafts::DFlashDraft`]) together with a Qwen3 / Qwen3.5
//! target so a single call to [`DFlashDecoder::generate`] runs the full
//! draft→verify→accept→rollback cycle.
//!
//! # Verification
//!
//! The target runs one forward pass over the whole proposed block and draws
//! its own token at every position of it, exactly as plain decoding would
//! draw them one at a time: through the generation's
//! [`DecodeSampler`] (temperature,
//! top-k, top-p, min-p, penalties, bias, and the seed's `n`-th key for the
//! `n`-th token). The drafts that equal the target's tokens are kept, the
//! first one that doesn't is replaced by the target's token, and a block
//! that is kept whole earns one more from the target (the bonus token). The
//! draft is deterministic (each position's argmax), a point mass, for which
//! "keep the draft if it is what the target draws" is exactly the
//! rejection-sampling rule of Leviathan et al. (arXiv 2211.17192): every
//! emitted token is distributed as the target alone would draw it, and a
//! seeded run draws the same tokens as the seeded plain run
//! ([`DFlashDecoder::generate_plain`]), up to bf16 ties: the verify's
//! block-shaped forward rounds differently from one-token decoding, and where
//! the target's top two logits are a rounding apart the two can pick
//! differently ([`DFlashDecoder::next_token_top_two`] measures it).
//!
//! The other upstream verification modes (stream, chunked,
//! parallel-lazy-logits, parallel-greedy-argmax) are straight-line variants
//! of the same plumbing and can be added as needed without touching the
//! target trait.
//!
//! # Target model support
//!
//! Targets plug in via the [`DFlashTarget`] trait: Qwen3 and Qwen3.5 on both
//! engines. On Qwen3.5 a rejected tail is rewound out of the GDN layers'
//! recurrent state as well as the attention caches, by replaying the
//! verify's recurrence inputs over the accepted prefix (a [`MambaCache`] on
//! the `DynamicModel` engine, the native target's own on the native one).
//!
//! # Context
//!
//! The draft cross-attends to the target's tapped hidden states of every
//! context position. The decoder hands it the states of the positions
//! decided since the last draft (`target_hidden`) and the draft keeps the
//! earlier ones in its own cache, so each draft sees the whole context:
//! the accepted positions of every round, not only the last.

use pmetal_runtime::compat::{Array, Dtype, Exception, Module, ops};
use pmetal_runtime::decode::DecodeSampler;

use pmetal_cache::kv_cache::{KVCache, KVCacheConfig, MambaCache};
use pmetal_cache::speculative::SpecCapture;

use crate::dflash_drafts::DFlashDraft;
use crate::generation::GenerationConfig;
use crate::traits::ModelConfig;

// ----------------------------------------------------------------------------
// Target trait
// ----------------------------------------------------------------------------

/// Target model contract for DFlash speculative decoding.
///
/// A target implementation supplies token embedding, hidden-state-capturing
/// forward, lm_head projection, and KV cache construction. The DFlash loop
/// never touches the internal architecture beyond these hooks, which makes
/// it trivial to add new architectures (Llama, Gemma, …) once their forward
/// pass grows a `SpecCapture` tap point.
pub trait DFlashTarget {
    /// Embed a `[B, T]` token id tensor into `[B, T, hidden]`.
    fn embed_tokens(&mut self, input_ids: &Array) -> Result<Array, Exception>;

    /// Forward pass over a token sequence that updates `kv_cache` (and,
    /// for hybrid architectures, `mamba_cache`) in place and records
    /// hidden states / GDN verify inputs into `capture`.
    ///
    /// Returns logits of shape `[B, T, vocab_size]`.
    fn forward_with_capture(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        kv_cache: Option<&mut KVCache>,
        mamba_cache: Option<&mut MambaCache>,
        capture: &mut SpecCapture,
    ) -> Result<Array, Exception>;

    /// [`Self::forward_with_capture`] over a prompt, returning the logits of
    /// its last position alone, `[B, 1, vocab_size]`: the only row a prefill
    /// reads, where the logits of every position of a long prompt are
    /// gigabytes. The default runs the full forward and slices; a target
    /// that can skip the LM head on the other positions overrides it.
    fn prefill_with_capture(
        &mut self,
        input_ids: &Array,
        kv_cache: Option<&mut KVCache>,
        mamba_cache: Option<&mut MambaCache>,
        capture: &mut SpecCapture,
    ) -> Result<Array, Exception> {
        let logits = self.forward_with_capture(input_ids, None, kv_cache, mamba_cache, capture)?;
        let t = logits.dim(1);
        Ok(logits.slice(&[0, t - 1, 0], &[logits.dim(0), t, logits.dim(2)]))
    }

    /// Forward over a drafted block (the last accepted token, then the
    /// guesses), which [`Self::rollback_rejected`] may then partly undo.
    /// Defaults to [`Self::forward_with_capture`]; a target that owns
    /// recurrent state it can't simply trim records here what it needs to
    /// rewind.
    fn verify_with_capture(
        &mut self,
        input_ids: &Array,
        kv_cache: Option<&mut KVCache>,
        mamba_cache: Option<&mut MambaCache>,
        capture: &mut SpecCapture,
    ) -> Result<Array, Exception> {
        self.forward_with_capture(input_ids, None, kv_cache, mamba_cache, capture)
    }

    /// Apply the target's lm_head to hidden states of shape `[B, T, hidden]`.
    fn lm_head_project(&mut self, hidden: &Array) -> Result<Array, Exception>;

    /// Hidden size — the last axis of the embedding / lm_head input.
    fn target_hidden_size(&self) -> i32;

    /// Number of transformer layers (used to size the KV cache).
    fn target_num_layers(&self) -> usize;

    /// Number of KV heads per layer.
    fn target_num_kv_heads(&self) -> i32;

    /// Head dimension of the target's attention.
    fn target_head_dim(&self) -> i32;

    /// Whether this target has any linear-attention (GDN / Mamba) layers
    /// and therefore needs a [`MambaCache`] alongside its KV cache.
    fn target_needs_mamba_cache(&self) -> bool {
        false
    }

    /// Forget any state the target keeps itself between forwards, before a
    /// new generation. The default keeps none: its caches are the ones
    /// [`Self::make_kv_cache`] and [`Self::make_mamba_cache`] hand out.
    fn reset_state(&mut self) {}

    /// Construct a fresh KV cache sized for `max_seq_len` tokens.
    fn make_kv_cache(&self, max_seq_len: usize) -> KVCache {
        let config = KVCacheConfig::new(
            self.target_num_layers(),
            max_seq_len,
            self.target_num_kv_heads() as usize,
            self.target_head_dim() as usize,
        );
        KVCache::new(config)
    }

    /// Construct a fresh Mamba cache for hybrid architectures, or `None`
    /// for pure-attention targets. Default: `None`.
    fn make_mamba_cache(&self) -> Option<MambaCache> {
        None
    }

    /// Rollback the target's KV state by `n` tokens after a partial-accept
    /// verify step. Default: delegates to [`KVCache::rollback`] on the
    /// provided external cache.
    ///
    /// Native-engine targets own their own cache and override this to
    /// rewind the internal state (recurrent layers included, from what
    /// [`Self::verify_with_capture`] recorded), ignoring the external
    /// `kv_cache` arg.
    /// Such targets should also return a dummy cache from
    /// [`DFlashTarget::make_kv_cache`] that is never touched.
    ///
    /// A cache that can't give back all `n` tokens (a sliding window that
    /// has already dropped them) is an error: the rejected tokens would stay
    /// in it and every later token would attend to them.
    fn rollback_rejected(&mut self, kv_cache: &mut KVCache, n: usize) -> Result<(), Exception> {
        let removed = kv_cache.rollback(n);
        if removed != n {
            return Err(Exception::custom(format!(
                "DFlash: the target's KV cache rolled back {removed} of {n} rejected tokens"
            )));
        }
        Ok(())
    }

    /// Whether this target supports tree-verify — per-position RoPE +
    /// custom additive attention mask on the forward pass. The default
    /// is `false`; the [`crate::dflash_native_target::NativeQwen3Target`]
    /// override returns `true`.
    fn supports_tree_verify(&self) -> bool {
        false
    }

    /// The dtype of the target's attention (the raw runtime dtype id), which
    /// the tree-verify mask must match: the runtime's attention rejects a
    /// mask of another dtype. Default: bf16.
    fn tree_mask_dtype(&self) -> i32 {
        Dtype::Bfloat16.as_i32()
    }

    /// Tree-verify forward pass. `position_ids` carries the per-token
    /// absolute positions and `attention_mask` is the additive mask
    /// (`0` visible / `-inf` hidden) built by `ddtree::compile_tree`.
    /// Default impl returns `Unsupported`, so implementations that
    /// opt in must override both this and [`Self::supports_tree_verify`].
    fn forward_tree_verify(
        &mut self,
        _input_ids: &Array,
        _position_ids: &Array,
        _attention_mask: &Array,
        _capture: &mut SpecCapture,
    ) -> Result<Array, Exception> {
        Err(Exception::custom(
            "DFlashTarget: tree verify not supported for this target",
        ))
    }

    /// Compact the target's KV cache after a tree-verify round: the
    /// `tree_length` rows at `[past_length..past_length+tree_length]`
    /// are reduced to `accepted_indices.len()` rows (in the same
    /// order). Default: no-op. Only the native-engine target
    /// implements this today.
    fn compact_tree_cache(
        &mut self,
        _past_length: usize,
        _tree_length: usize,
        _accepted_indices: &[usize],
    ) {
        // default: no-op
    }
}

// ----------------------------------------------------------------------------
// Configuration / outputs
// ----------------------------------------------------------------------------

/// Runtime configuration for [`DFlashDecoder::generate`].
#[derive(Debug, Clone)]
pub struct DFlashConfig {
    /// How many tokens to generate, which tokens stop the generation, and how
    /// each is drawn: the target's own sampling, as plain decoding would
    /// draw it. [`GenerationConfig::greedy`] is greedy decoding.
    pub generation: GenerationConfig,
    /// Optional override for the draft block size. `None` uses the value
    /// stored in the draft model config.
    pub speculative_tokens: Option<usize>,
    /// Whether tree verification ([`DFlashDecoder::generate_ddtree`])
    /// shrinks its tree while the draft keeps winning. It never grows past
    /// the budget it is given.
    pub adaptive_tree_budget: bool,
}

impl DFlashConfig {
    /// Decode as `generation` says.
    pub fn new(generation: GenerationConfig) -> Self {
        Self {
            generation,
            speculative_tokens: None,
            adaptive_tree_budget: true,
        }
    }
}

impl Default for DFlashConfig {
    fn default() -> Self {
        Self::new(GenerationConfig::greedy(128))
    }
}

/// End-of-run metrics for one [`DFlashDecoder::generate`] call.
#[derive(Debug, Clone, Default)]
pub struct DFlashMetrics {
    /// Accepted-token counts per speculative step. One entry per iteration.
    pub acceptance_lengths: Vec<usize>,
    /// Total new tokens produced (not counting the prompt).
    pub num_generated: usize,
    /// How many tokens were ever drafted (`block_size * num_iterations`).
    pub total_drafted: usize,
    /// How many drafted tokens the verifier accepted.
    pub total_accepted: usize,
}

impl DFlashMetrics {
    /// Average acceptance length. Returns 0.0 for a run with no iterations.
    pub fn avg_acceptance_length(&self) -> f32 {
        if self.acceptance_lengths.is_empty() {
            0.0
        } else {
            let sum: usize = self.acceptance_lengths.iter().sum();
            sum as f32 / self.acceptance_lengths.len() as f32
        }
    }

    /// Acceptance rate = accepted / drafted. Zero when nothing was drafted.
    pub fn acceptance_rate(&self) -> f32 {
        if self.total_drafted == 0 {
            0.0
        } else {
            self.total_accepted as f32 / self.total_drafted as f32
        }
    }
}

/// Result of a [`DFlashDecoder::generate`] call.
#[derive(Debug, Clone)]
pub struct DFlashOutput {
    /// Token ids, including the prompt. A stop token ends the generation and
    /// is not part of the output.
    pub tokens: Vec<u32>,
    /// Metrics collected during the run.
    pub metrics: DFlashMetrics,
}

// ----------------------------------------------------------------------------
// Decoder
// ----------------------------------------------------------------------------

/// One generation in flight: the target's caches, the draft's context, the
/// tokens so far and the sampler that draws them.
struct Run {
    kv_cache: KVCache,
    mamba_cache: Option<MambaCache>,
    capture: SpecCapture,
    draft_cache: Vec<KVCache>,
    sampler: DecodeSampler,
    /// The prompt, then what has been generated.
    tokens: Vec<u32>,
    /// The target's tapped hidden states of the positions decided since the
    /// last draft.
    target_hidden: Array,
    metrics: DFlashMetrics,
    prompt_len: usize,
    /// `prompt_len + max_new_tokens`.
    total_max: usize,
    /// A stop token was drawn, or the length is reached.
    finished: bool,
    /// Where the last step's tokens start, to count only those kept.
    last_step_from: usize,
}

impl Run {
    /// Append drawn tokens, stopping at a stop token (which is dropped) or
    /// the length limit.
    fn emit(&mut self, drawn: &[u32], config: &GenerationConfig) {
        self.last_step_from = self.tokens.len();
        for &token in drawn {
            if config.is_stop_token(token) || self.tokens.len() >= self.total_max {
                self.finished = true;
                return;
            }
            self.tokens.push(token);
        }
        self.finished = self.tokens.len() >= self.total_max;
    }

    fn finish(mut self) -> DFlashOutput {
        // A stop token or the length limit can cut the last step short;
        // count the tokens it kept, as the reference does.
        if let Some(last) = self.metrics.acceptance_lengths.last_mut() {
            *last = self.tokens.len().saturating_sub(self.last_step_from);
        }
        self.metrics.num_generated = self.tokens.len() - self.prompt_len;
        DFlashOutput {
            tokens: self.tokens,
            metrics: self.metrics,
        }
    }
}

/// DFlash speculative decoder.
///
/// Holds a target model, a DFlash draft, and a reusable [`SpecCapture`]
/// buffer. One instance is intended to service many `generate` calls — the
/// draft / target weights stay resident and only the per-request KV caches
/// are allocated fresh.
pub struct DFlashDecoder<T: DFlashTarget> {
    target: T,
    draft: DFlashDraft,
    target_layer_ids: Vec<usize>,
}

impl<T: DFlashTarget> DFlashDecoder<T> {
    /// Create a new decoder.
    ///
    /// `target_layer_ids` tells the loop which target layers to tap for the
    /// draft's conditioning input; it must match the
    /// `dflash_config.target_layer_ids` the draft was trained with, which
    /// it takes from the draft (either generation).
    pub fn new(target: T, draft: impl Into<DFlashDraft>) -> Self {
        let draft = draft.into();
        let target_layer_ids = draft.target_layer_ids();
        Self {
            target,
            draft,
            target_layer_ids,
        }
    }

    /// Borrow the underlying target model (mutably).
    pub fn target_mut(&mut self) -> &mut T {
        &mut self.target
    }

    /// Borrow the underlying draft model.
    pub fn draft(&self) -> &DFlashDraft {
        &self.draft
    }

    /// The draft block size a run uses: the override, within what the draft
    /// was trained for.
    fn block_size(&self, config: &DFlashConfig) -> usize {
        config
            .speculative_tokens
            .unwrap_or_else(|| self.draft.block_size())
            .max(1)
            .min(self.draft.block_size())
    }

    /// Start a generation: reset the target, prefill the prompt (recording
    /// the tapped hidden states the draft conditions on) and draw the first
    /// token. `headroom` is how many tokens past the last emitted one a
    /// verify step writes before it rolls the rejected tail back.
    fn begin(
        &mut self,
        prompt: &[u32],
        config: &DFlashConfig,
        headroom: usize,
    ) -> Result<Run, Exception> {
        if prompt.is_empty() {
            return Err(Exception::custom("DFlash: the prompt has no tokens"));
        }
        let total_max = prompt.len() + config.generation.max_new_tokens;
        self.target.reset_state();
        let mut kv_cache = self.target.make_kv_cache(total_max + headroom);

        // Defensive sanity check: rolling back across a sliding-window
        // eviction boundary would silently produce wrong outputs because
        // the rolled-back positions may already have been dropped from
        // the cache. Refuse to run if the cache window is smaller than the
        // block size we may need to rewind.
        let cache_window = match kv_cache.config().mode {
            pmetal_cache::kv_cache::CacheMode::SlidingWindow { window_size } => Some(window_size),
            pmetal_cache::kv_cache::CacheMode::Rotating { max_size, .. } => Some(max_size),
            _ => None,
        };
        if let Some(window) = cache_window
            && window < headroom + 1
        {
            return Err(Exception::custom(format!(
                "DFlashDecoder: target cache window ({window}) must be larger than \
                 the speculative block ({headroom}) so a rejected-tail rollback \
                 can land on positions that are still in the window."
            )));
        }
        let mut mamba_cache = self.target.make_mamba_cache();
        let mut capture = SpecCapture::with_layers(self.target_layer_ids.clone());
        let draft_cache = self.draft.make_cache(total_max);

        // The target processes the whole prompt in one pass and records
        // hidden states at every tapped layer. The first "free" token is
        // drawn from the last prompt position, and the tapped hidden states
        // are the draft's first `target_hidden`.
        let prompt_arr = array_from_row(prompt);
        let last_logits = self.target.prefill_with_capture(
            &prompt_arr,
            Some(&mut kv_cache),
            mamba_cache.as_mut(),
            &mut capture,
        )?;
        let mut sampler = config.generation.sampler(prompt);
        let first = read_token(&sampler.sample(&last_row(&last_logits)))?;
        let target_hidden = capture.stack_hidden()?;

        let mut run = Run {
            kv_cache,
            mamba_cache,
            capture,
            draft_cache,
            sampler,
            tokens: prompt.to_vec(),
            target_hidden,
            metrics: DFlashMetrics::default(),
            prompt_len: prompt.len(),
            total_max,
            finished: config.generation.max_new_tokens == 0,
            last_step_from: prompt.len(),
        };
        if !run.finished {
            run.emit(&[first], &config.generation);
        }
        Ok(run)
    }

    /// Run the DFlash draft/verify loop from `prompt`.
    pub fn generate(
        &mut self,
        prompt: &[u32],
        config: &DFlashConfig,
    ) -> Result<DFlashOutput, Exception> {
        let block_size = self.block_size(config);
        if block_size < 2 {
            // A block of one has nothing to guess.
            return self.generate_plain(prompt, config);
        }
        let mut run = self.begin(prompt, config, block_size)?;

        while !run.finished {
            // Draft: start with the latest accepted token, fill the rest
            // with mask-token noise.
            let seed_token = *run.tokens.last().expect("the prompt is not empty");
            let mut block = Vec::with_capacity(block_size);
            block.push(seed_token);
            block.resize(block_size, self.draft.mask_token_id() as u32);

            // noise_embedding is the target's token embedding of the block.
            let noise_embedding = self.target.embed_tokens(&array_from_row(&block))?;

            // `target_hidden` holds only the positions since the last draft;
            // the draft cache holds the rest of the context.
            let target = &mut self.target;
            let drafted = self.draft.propose(
                &noise_embedding,
                seed_token as i32,
                &run.target_hidden,
                &mut run.draft_cache,
                &mut |hidden| target.lm_head_project(hidden),
                block_size - 1,
            )?;
            for (slot, token) in block[1..].iter_mut().zip(read_tokens(&drafted)?) {
                *slot = token;
            }

            // Snapshot the Mamba cache BEFORE verify so a partial-accept
            // rollback can replay the GDN recurrence through only the
            // accepted prefix. Pure-attention targets (Qwen3) have no
            // mamba_cache and this is a no-op.
            let mamba_snapshot: Option<Vec<pmetal_cache::kv_cache::MambaSnapshot>> =
                run.mamba_cache.as_ref().map(|c| c.snapshot());

            // Verify: one target forward pass over the whole block.
            run.capture.clear();
            let verify_logits = self.target.verify_with_capture(
                &array_from_row(&block),
                Some(&mut run.kv_cache),
                run.mamba_cache.as_mut(),
                &mut run.capture,
            )?;
            let verifier_hidden_all = run.capture.stack_hidden()?;

            // What the target draws at every position of the block, in one
            // block on a fork: the positions past the first disagreement are
            // never used, and only the kept tokens are committed.
            let rows = verify_logits.reshape(&[block_size as i32, verify_logits.dim(-1)]);
            let mut fork = run.sampler.fork();
            let drawn = read_tokens(&fork.sample_block(&rows))?;

            // Accept the longest prefix of the drafts the target drew too,
            // plus the token it drew where they part (or after the last).
            let matched = longest_prefix_match(&block[1..], &drawn[..block_size - 1]);
            let accepted_inputs = matched + 1;
            tracing::trace!(
                drafted = ?&block[1..],
                drawn = ?&drawn,
                matched,
                "dflash round"
            );

            run.metrics.acceptance_lengths.push(accepted_inputs);
            run.metrics.total_drafted += block_size;
            run.metrics.total_accepted += matched;

            // Rollback rejected positions in the target's KV cache.
            let rejected = block_size - accepted_inputs;
            if rejected > 0 {
                self.target.rollback_rejected(&mut run.kv_cache, rejected)?;
                if let (Some(mamba), Some(snaps)) =
                    (run.mamba_cache.as_mut(), mamba_snapshot.as_ref())
                {
                    // Build per-layer verify inputs in cache layer order so
                    // `rewind_from_snapshots` can replay each GDN layer
                    // forward through the accepted prefix.
                    let num_layers = mamba.num_layers();
                    let mut per_layer: Vec<Option<pmetal_cache::kv_cache::GdnVerifyInputs>> =
                        Vec::with_capacity(num_layers);
                    for layer_idx in 0..num_layers {
                        per_layer.push(run.capture.gdn_inputs.remove(&layer_idx));
                    }
                    mamba.rewind_from_snapshots(snaps, &per_layer, accepted_inputs)?;
                }
            }

            // Commit: the accepted drafts equal the target's first `matched`
            // draws, and the draw where they part comes after them.
            let emitted = &drawn[..accepted_inputs];
            run.sampler.commit(emitted);
            run.emit(emitted, &config.generation);

            // The next draft step conditions on the tapped hidden states of
            // the accepted positions.
            run.target_hidden = slice_axis_1(&verifier_hidden_all, 0, accepted_inputs as i32);
        }
        Ok(run.finish())
    }

    /// Plain decoding with the same target and sampler, one token per forward
    /// and no draft: what [`Self::generate`] must reproduce, greedy or
    /// seeded, and the speed it is measured against. The metrics count one
    /// token per step.
    pub fn generate_plain(
        &mut self,
        prompt: &[u32],
        config: &DFlashConfig,
    ) -> Result<DFlashOutput, Exception> {
        let mut run = self.begin(prompt, config, 1)?;
        let mut metrics = DFlashMetrics::default();
        while !run.finished {
            run.capture.clear();
            let input = array_from_row(&run.tokens[run.tokens.len() - 1..]);
            let logits = self.target.forward_with_capture(
                &input,
                None,
                Some(&mut run.kv_cache),
                run.mamba_cache.as_mut(),
                &mut run.capture,
            )?;
            let token = read_token(&run.sampler.sample(&last_row(&logits)))?;
            run.emit(&[token], &config.generation);
            metrics.acceptance_lengths.push(1);
        }
        let mut output = run.finish();
        // One token per step, the step that drew a stop token included.
        output.metrics.acceptance_lengths = metrics.acceptance_lengths;
        Ok(output)
    }

    /// The target's two likeliest next tokens after `prefix` and their
    /// logits, from one forward over it: how close a call it was where two
    /// decodings of the same target part ways.
    pub fn next_token_top_two(&mut self, prefix: &[u32]) -> Result<[(u32, f32); 2], Exception> {
        self.target.reset_state();
        let mut kv = self.target.make_kv_cache(prefix.len() + 1);
        let mut mamba = self.target.make_mamba_cache();
        let mut capture = SpecCapture::with_layers(Vec::new());
        let logits = self.target.prefill_with_capture(
            &array_from_row(prefix),
            Some(&mut kv),
            mamba.as_mut(),
            &mut capture,
        )?;
        let last = last_row(&logits).as_dtype(Dtype::Float32.as_i32());
        let _ = last.eval();
        pmetal_runtime::check_unobserved_error()
            .map_err(|e| Exception::custom(format!("top two: {e}")))?;
        let values = last.as_slice::<f32>();
        let mut best = [(0u32, f32::NEG_INFINITY); 2];
        for (token, &logit) in values.iter().enumerate() {
            if logit > best[0].1 {
                best = [(token as u32, logit), best[0]];
            } else if logit > best[1].1 {
                best[1] = (token as u32, logit);
            }
        }
        Ok(best)
    }

    /// Tree-verify variant of [`Self::generate`]. For each draft round
    /// we build a budget-bounded candidate tree from the draft's
    /// output logits (via [`crate::ddtree::build_tree`]), compile it
    /// into verify-friendly inputs, run the target forward with the
    /// per-position RoPE and tree-visibility mask, then walk the
    /// posterior argmax through the tree to accept whichever path the
    /// target selects. This gives the target MULTIPLE siblings to
    /// choose from at each position instead of forcing it to accept a
    /// single linear prefix — significantly higher expected
    /// acceptance length per round when the draft is only moderately
    /// off (which is exactly pmetal's bf16-noise situation on iter
    /// 2+ of linear DFlash).
    ///
    /// **Adaptive budget.** The `tree_budget` argument is treated
    /// as the BASE; an EMA over recent matched counts may shrink
    /// the effective budget per round when the draft is winning
    /// consistently (a smaller tree captures the same useful
    /// candidates at lower per-round cost). The budget is never
    /// grown past the base — empirically the throughput plateau
    /// runs out around `base = 12-24` on Qwen3-4B, and growing
    /// past that costs more than it gains. Turn it off with
    /// [`DFlashConfig::adaptive_tree_budget`] for benchmarking.
    ///
    /// **Numerics note.** The tree forward routes through MLX's
    /// `has_mask=true` SDPA kernel specialization, while greedy
    /// decode and linear DFlash use `do_causal=true`. These
    /// specializations are mathematically equivalent but produce
    /// slightly different bf16 numerics due to different fused
    /// instruction orderings inside the steel attention kernel
    /// (see `mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h`
    /// in ml-explore/mlx). The K/V values
    /// written to cache stay bit-exact (they come from RoPE +
    /// projection, both kernel-identical to the linear path), so
    /// the drift only affects which TOKEN the target picks at
    /// argmax — and only on positions where two top tokens are
    /// near-tied. At temperature > 0 this is invisible; at
    /// temperature = 0 it can produce 1–3 token differences from
    /// the canonical greedy baseline over a 128-token generation.
    /// The trade is roughly +25% throughput vs linear DFlash on
    /// Qwen3-4B at base budget 12. Callers that need strict
    /// greedy reproducibility should use [`Self::generate`]
    /// (linear) or set `tree_budget = 0`.
    ///
    /// Falls back to linear [`Self::generate`] when the target does
    /// not override [`DFlashTarget::supports_tree_verify`] (i.e.,
    /// anything that's not `NativeQwen3Target` today), for a DFlash 2
    /// draft, whose selector proposes one path rather than per-position
    /// distributions to branch on, and for any decoding that is not the
    /// plain argmax (a temperature, a penalty, a bias): the tree walks the
    /// target's argmax at every node, and a sampled token or one shaped by
    /// the history of its path has no such per-node reading.
    pub fn generate_ddtree(
        &mut self,
        prompt: &[u32],
        config: &DFlashConfig,
        tree_budget: usize,
    ) -> Result<DFlashOutput, Exception> {
        let greedy = config.generation.sampler(prompt).is_plain_argmax();
        if !self.target.supports_tree_verify()
            || tree_budget == 0
            || self.draft.is_dflash2()
            || !greedy
        {
            return self.generate(prompt, config);
        }
        let block_size = self.block_size(config);
        // Tree horizon = `block_size - 1` (the root eats one slot, the
        // other positions are drafted). Matches DDTree upstream.
        let draft_horizon = block_size.saturating_sub(1);
        if draft_horizon == 0 {
            return self.generate(prompt, config);
        }
        let tree_budget = tree_budget.max(1);
        let max_tree_nodes = 1 + tree_budget; // +1 for the root

        // Every round writes `max_tree_nodes` tokens and keeps at most
        // `max_tree_nodes - 1` (tree_budget) after compaction, so the cache
        // needs room for `prompt + max_new_tokens + max_tree_nodes` at peak.
        let mut run = self.begin(prompt, config, max_tree_nodes)?;
        let mut past_length = prompt.len(); // cache length after prefill

        // ── Adaptive tree budget state ────────────────────────────────
        // The budget passed in is the BASE — it caps how big the tree
        // can be. We may shrink the EFFECTIVE budget per round when
        // the recent acceptance rate suggests the draft is winning
        // (in which case a smaller tree captures the same useful
        // candidates at lower per-round cost). We only ever shrink:
        // growing past `base_budget` consistently hurts throughput
        // because compute scales linearly with tree size while accept
        // length plateaus. Shrink-only adaptation is purely a Pareto
        // improvement.
        let base_budget = tree_budget;
        let min_budget = (base_budget / 3).max(4);
        let mut effective_budget = base_budget;
        // EMA over per-round `matched` (= accepted_count - 1). The
        // optimistic prior of 1.5 keeps the budget at base for the
        // first few rounds while the EMA warms up.
        let mut ema_matched: f32 = 1.5;
        let ema_alpha: f32 = 0.35;

        while !run.finished {
            let seed_token = *run.tokens.last().expect("the prompt is not empty");

            // ── Draft: build noise embedding and run draft forward ──
            let mut block = Vec::with_capacity(block_size);
            block.push(seed_token);
            block.resize(block_size, self.draft.mask_token_id() as u32);
            let noise_embedding = self.target.embed_tokens(&array_from_row(&block))?;
            // [1, draft_horizon, vocab], the root position sliced away.
            let target = &mut self.target;
            let draft_logits = self
                .draft
                .guess_logits(
                    &noise_embedding,
                    &run.target_hidden,
                    &mut run.draft_cache,
                    &mut |hidden| target.lm_head_project(hidden),
                    draft_horizon,
                )?
                .ok_or_else(|| Exception::custom("DFlash 2 drafts don't tree-verify"))?;
            // [draft_horizon, vocab]
            let draft_logits_2d =
                draft_logits.reshape(&[draft_horizon as i32, draft_logits.dim(2)]);

            // ── Tree build + compile ────────────────────────────────
            let tree = crate::ddtree::build_tree(&draft_logits_2d, effective_budget);
            let compiled = crate::ddtree::compile_tree(
                &tree,
                seed_token as i32,
                past_length as i32,
                past_length as i32,
                self.target.tree_mask_dtype(),
            );
            let tree_length = compiled.current_length;

            // ── Verify: target forward with tree inputs ─────────────
            run.capture.clear();
            let verify_logits = self.target.forward_tree_verify(
                &compiled.verify_input_ids,
                &compiled.verify_position_ids,
                &compiled.attention_mask,
                &mut run.capture,
            )?;
            let verifier_tokens = argmax_last_axis(&verify_logits)?;
            let verifier_hidden_all = run.capture.stack_hidden()?;

            // ── Walk the verified path ──────────────────────────────
            let (accepted_indices, bonus_token) =
                crate::ddtree::follow_verified_tree(&tree.child_maps, &verifier_tokens);
            let accepted_count = accepted_indices.len();
            let matched = accepted_count.saturating_sub(1);

            tracing::trace!(
                budget = effective_budget,
                tree_length,
                accepted = accepted_count,
                matched,
                bonus = bonus_token,
                ema = ema_matched,
                "ddtree round"
            );

            // ── Adaptive budget update ──────────────────────────────
            // Update EMA, then pick the next round's budget. Three
            // tiers: high-confidence (≥2 matched on average) shrinks
            // hardest, mid (≥1.2) shrinks modestly, low restores the
            // base. The tiers were tuned on Qwen3-4B where the sweep
            // peak sits at ~12 nodes; reduce_factors here translate
            // to budgets like {12, 8, 12} which all sit on the
            // throughput plateau. Capped at min_budget so the floor
            // is never below the smallest useful tree.
            if config.adaptive_tree_budget {
                ema_matched = ema_matched * (1.0 - ema_alpha) + matched as f32 * ema_alpha;
                effective_budget = if ema_matched >= 2.0 {
                    (base_budget / 2).max(min_budget)
                } else if ema_matched >= 1.2 {
                    (base_budget * 2 / 3).max(min_budget)
                } else {
                    base_budget
                };
            }

            run.metrics.acceptance_lengths.push(accepted_count);
            run.metrics.total_drafted += tree_length.saturating_sub(1);
            run.metrics.total_accepted += matched;

            // ── Commit accepted path ────────────────────────────────
            // `accepted_indices[0]` is the root — the seed we already
            // have in `run.tokens`. For positions 1..accepted_count
            // we pull the tree's token id.
            let mut emitted: Vec<u32> = accepted_indices[1..]
                .iter()
                .map(|&idx| tree.node_token_ids[idx - 1] as u32)
                .collect();
            emitted.push(bonus_token as u32);
            run.sampler.commit(&emitted);
            run.emit(&emitted, &config.generation);

            // ── Compact the cache to keep only accepted positions ──
            self.target
                .compact_tree_cache(past_length, tree_length, &accepted_indices);
            past_length += accepted_count;

            // ── Next round's target_hidden = selected rows ─────────
            // `verifier_hidden_all` is [1, tree_length, hidden*taps].
            // Gather the accepted rows via take_axis on the seq dim.
            let accepted_i32: Vec<i32> = accepted_indices.iter().map(|&i| i as i32).collect();
            let idx_arr = Array::from_slice(&accepted_i32, &[accepted_count as i32]);
            run.target_hidden = verifier_hidden_all.take_axis(&idx_arr, 1);
        }
        Ok(run.finish())
    }
}

// ----------------------------------------------------------------------------
// Array <-> Vec helpers
// ----------------------------------------------------------------------------

fn array_from_row(tokens: &[u32]) -> Array {
    let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
    Array::from_slice(&ids, &[1, ids.len() as i32])
}

fn slice_axis_1(arr: &Array, start: i32, stop: i32) -> Array {
    let rank = arr.shape().len();
    let mut start_v = vec![0i32; rank];
    let mut stop_v: Vec<i32> = arr.shape().to_vec();
    start_v[1] = start;
    stop_v[1] = stop;
    arr.slice(&start_v, &stop_v)
}

/// `[B, 1, vocab]` logits of one position as `[B, vocab]`.
fn last_row(logits: &Array) -> Array {
    logits.reshape(&[logits.dim(0), logits.dim(-1)])
}

/// The one token of an evaluated-on-demand `[1]` or scalar array.
fn read_token(token: &Array) -> Result<u32, Exception> {
    token
        .try_eval()
        .map_err(|e| Exception::custom(format!("DFlash sampling: {e}")))?;
    Ok(token.item_u32())
}

/// Materialize a lazy array of unsigned token ids.
fn read_tokens(ids: &Array) -> Result<Vec<u32>, Exception> {
    let ids = ids.as_dtype(Dtype::Uint32.as_i32());
    let _ = ids.eval();
    pmetal_runtime::check_unobserved_error()
        .map_err(|e| Exception::custom(format!("DFlash tokens: {e}")))?;
    Ok(ids.as_slice::<u32>().to_vec())
}

/// Compute `argmax(logits, axis=-1)` and materialize the result as token ids.
fn argmax_last_axis(logits: &Array) -> Result<Vec<i32>, Exception> {
    let argmax = ops::argmax_axis(logits, -1);
    let _ = argmax.eval();
    pmetal_runtime::check_unobserved_error()
        .map_err(|e| Exception::custom(format!("DFlash verify argmax: {e}")))?;
    Ok(argmax.as_slice::<u32>().iter().map(|&u| u as i32).collect())
}

fn longest_prefix_match(drafted: &[u32], verifier: &[u32]) -> usize {
    drafted
        .iter()
        .zip(verifier.iter())
        .take_while(|(d, v)| d == v)
        .count()
}

// ----------------------------------------------------------------------------
// Qwen3 target implementation
// ----------------------------------------------------------------------------

use crate::DynamicModel;
use crate::architectures::qwen3::Qwen3ForCausalLM;
use crate::architectures::qwen3_next::Qwen3NextForCausalLM;

impl DFlashTarget for Qwen3ForCausalLM {
    fn embed_tokens(&mut self, input_ids: &Array) -> Result<Array, Exception> {
        Module::forward(&mut self.model.embed_tokens, input_ids)
    }

    fn forward_with_capture(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        kv_cache: Option<&mut KVCache>,
        _mamba_cache: Option<&mut MambaCache>,
        capture: &mut SpecCapture,
    ) -> Result<Array, Exception> {
        // Qwen3 has no linear-attention layers — mamba_cache is unused.
        Qwen3ForCausalLM::forward_with_capture(self, input_ids, mask, kv_cache, capture)
    }

    fn lm_head_project(&mut self, hidden: &Array) -> Result<Array, Exception> {
        if let Some(ref mut lm_head) = self.lm_head {
            Ok(lm_head.forward(hidden)?)
        } else {
            Ok(self.model.embed_tokens.as_linear(hidden))
        }
    }

    fn target_hidden_size(&self) -> i32 {
        self.config.hidden_size
    }

    fn target_num_layers(&self) -> usize {
        self.config.num_hidden_layers as usize
    }

    fn target_num_kv_heads(&self) -> i32 {
        self.config.num_kv_heads()
    }

    fn target_head_dim(&self) -> i32 {
        self.config.get_head_dim()
    }
}

// ----------------------------------------------------------------------------
// DynamicModel target implementation
// ----------------------------------------------------------------------------

/// Apply Gemma 4-style final logit softcap: `cap * tanh(x / cap)`.
/// Returns the input unchanged when `cap` is `None`.
fn apply_logit_softcap(logits: &Array, cap: Option<f32>) -> Array {
    match cap {
        Some(c) => {
            // Pre-cast the scalar to the logit dtype. A bare `from_f32`
            // here would promote bf16 logits to f32 for the divide/tanh
            // round-trip (same footgun as gemma4_native). Cheap post-
            // forward op, but the rule applies everywhere.
            let c_arr = Array::from_f32(c).as_dtype(logits.dtype().as_i32());
            let scaled = logits.divide(&c_arr);
            let tanh = pmetal_runtime::compat::ops::tanh(&scaled);
            tanh.multiply(&c_arr)
        }
        None => logits.clone(),
    }
}

/// Return an explicit "not yet wired" error for architectures that do not
/// yet have a `forward_with_capture` entry point. Callers should see a
/// clear message telling them which model type is missing capture support.
fn dflash_architecture_unsupported(name: &str) -> Exception {
    Exception::custom(format!(
        "DFlashTarget: architecture {name} does not yet expose `forward_with_capture`. \
         Qwen3 and Qwen3.5 are wired today; other dense / hybrid models follow the same \
         pattern — add a `forward_with_capture` method on the Model / ForCausalLM wrapper \
         and a branch below. See `impl DFlashTarget for DynamicModel` in dflash_decoder.rs."
    ))
}

impl DFlashTarget for DynamicModel {
    fn embed_tokens(&mut self, input_ids: &Array) -> Result<Array, Exception> {
        match self {
            Self::Qwen3(m) => Module::forward(&mut m.model.embed_tokens, input_ids),
            Self::Qwen3Next(m) => Module::forward(&mut m.model.embed_tokens, input_ids),
            Self::Llama(m) => Module::forward(&mut m.model.embed_tokens, input_ids),
            Self::Qwen2(m) => Module::forward(&mut m.model.embed_tokens, input_ids),
            Self::Gemma(m) => Module::forward(&mut m.model.embed_tokens, input_ids),
            Self::Mistral(m) => Module::forward(&mut m.model.embed_tokens, input_ids),
            Self::Phi(m) => Module::forward(&mut m.model.embed_tokens, input_ids),
            Self::Phi4(m) => Module::forward(&mut m.model.embed_tokens, input_ids),
            Self::DeepSeek(m) => Module::forward(&mut m.model.embed_tokens, input_ids),
            Self::Qwen3MoE(m) => Module::forward(&mut m.model.embed_tokens, input_ids),
            Self::Gemma4(m) => Ok(m.model.embed_tokens.forward(input_ids)),
            Self::GptOss(m) => {
                // GptOss's `Embedding` forward is `&self` only; use the
                // explicit forward rather than `Module::forward` to avoid
                // borrow issues.
                Ok(m.model.embed_tokens.forward(input_ids))
            }
            Self::Llama4(_) => Err(dflash_architecture_unsupported("Llama4")),
            Self::Cohere(_) => Err(dflash_architecture_unsupported("Cohere")),
            Self::Granite(_) => Err(dflash_architecture_unsupported("Granite")),
            Self::NemotronH(_) => Err(dflash_architecture_unsupported("NemotronH")),
            Self::Qwen4Exp(_) => Err(dflash_architecture_unsupported("Qwen4Exp")),
            // Mllama's decoder interleaves cross-attention layers whose KV
            // comes from images, so a text-only draft model cannot mirror it.
            Self::Mllama(_) => Err(dflash_architecture_unsupported("Mllama")),
            Self::Flux(_) | Self::Bert(_) | Self::DiffusionGemma(_) => Err(Exception::custom(
                "DFlashTarget: Flux / BERT / DiffusionGemma are not causal LMs and cannot serve as DFlash targets",
            )),
        }
    }

    fn forward_with_capture(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        kv_cache: Option<&mut KVCache>,
        mamba_cache: Option<&mut MambaCache>,
        capture: &mut SpecCapture,
    ) -> Result<Array, Exception> {
        match self {
            Self::Qwen3(m) => <Qwen3ForCausalLM as DFlashTarget>::forward_with_capture(
                m,
                input_ids,
                mask,
                kv_cache,
                mamba_cache,
                capture,
            ),
            Self::Qwen3Next(m) => <Qwen3NextForCausalLM as DFlashTarget>::forward_with_capture(
                m,
                input_ids,
                mask,
                kv_cache,
                mamba_cache,
                capture,
            ),
            Self::Llama(m) => {
                // Pure attention — mamba_cache is unused.
                let _ = mamba_cache;
                m.forward_with_capture(input_ids, mask, kv_cache, capture)
            }
            Self::Qwen2(m) => {
                let _ = mamba_cache;
                m.forward_with_capture(input_ids, mask, kv_cache, capture)
            }
            Self::Gemma(m) => {
                let _ = mamba_cache;
                m.forward_with_capture(input_ids, mask, kv_cache, capture)
            }
            Self::Mistral(m) => {
                let _ = mamba_cache;
                m.forward_with_capture(input_ids, mask, kv_cache, capture)
            }
            Self::Phi(m) => {
                let _ = mamba_cache;
                m.forward_with_capture(input_ids, mask, kv_cache, capture)
            }
            Self::Phi4(m) => {
                let _ = mamba_cache;
                m.forward_with_capture(input_ids, mask, kv_cache, capture)
            }
            Self::DeepSeek(m) => {
                let _ = mamba_cache;
                m.forward_with_capture(input_ids, mask, kv_cache, capture)
            }
            Self::Qwen3MoE(m) => {
                let _ = mamba_cache;
                m.forward_with_capture(input_ids, mask, kv_cache, capture)
            }
            Self::Gemma4(m) => {
                let _ = mamba_cache;
                m.forward_with_capture(input_ids, mask, kv_cache, capture)
            }
            Self::GptOss(m) => {
                let _ = mamba_cache;
                m.forward_with_capture(input_ids, mask, kv_cache, capture)
            }
            Self::Llama4(_) => Err(dflash_architecture_unsupported("Llama4")),
            Self::Cohere(_) => Err(dflash_architecture_unsupported("Cohere")),
            Self::Granite(_) => Err(dflash_architecture_unsupported("Granite")),
            Self::NemotronH(_) => Err(dflash_architecture_unsupported("NemotronH")),
            Self::Qwen4Exp(_) => Err(dflash_architecture_unsupported("Qwen4Exp")),
            Self::Mllama(_) => Err(dflash_architecture_unsupported("Mllama")),
            Self::Flux(_) | Self::Bert(_) | Self::DiffusionGemma(_) => Err(Exception::custom(
                "DFlashTarget: Flux / BERT / DiffusionGemma are not causal LMs",
            )),
        }
    }

    fn lm_head_project(&mut self, hidden: &Array) -> Result<Array, Exception> {
        match self {
            Self::Qwen3(m) => <Qwen3ForCausalLM as DFlashTarget>::lm_head_project(m, hidden),
            Self::Qwen3Next(m) => {
                <Qwen3NextForCausalLM as DFlashTarget>::lm_head_project(m, hidden)
            }
            Self::Llama(m) => match &mut m.lm_head {
                Some(lm) => Ok(Module::forward(lm, hidden)?),
                None => Ok(m.model.embed_tokens.as_linear(hidden)),
            },
            Self::Qwen2(m) => match &mut m.lm_head {
                Some(lm) => Ok(Module::forward(lm, hidden)?),
                None => Ok(m.model.embed_tokens.as_linear(hidden)),
            },
            Self::Gemma(m) => Ok(m.model.embed_tokens.as_linear(hidden)),
            Self::Mistral(m) => match &mut m.lm_head {
                Some(lm) => Ok(Module::forward(lm, hidden)?),
                None => Ok(m.model.embed_tokens.as_linear(hidden)),
            },
            // Through `project_logits`, which handles the tied-head case.
            // Phi-4-mini ties, so a direct `lm_head` projection here draws on
            // a head the checkpoint never filled.
            Self::Phi(m) | Self::Phi4(m) => m.project_logits(hidden),
            Self::DeepSeek(m) => Ok(m.lm_head_forward(hidden)),
            Self::Qwen3MoE(m) => match &mut m.lm_head {
                Some(lm) => Ok(Module::forward(lm, hidden)?),
                None => {
                    let embed_w = m.model.embed_tokens.weight.value.as_ref();
                    Ok(hidden.matmul(&embed_w.t()))
                }
            },
            Self::Gemma4(m) => {
                // Gemma 4 ties embeddings + applies final-logit softcap.
                let logits = m.model.embed_tokens.as_linear(hidden);
                Ok(apply_logit_softcap(
                    &logits,
                    m.config.final_logit_softcapping,
                ))
            }
            _ => Err(dflash_architecture_unsupported("lm_head_project")),
        }
    }

    fn target_hidden_size(&self) -> i32 {
        DynamicModel::hidden_size(self)
    }

    fn target_num_layers(&self) -> usize {
        // Delegate to create_cache which carries num_layers on the returned
        // config. A tiny allocation, but only called once at generate start.
        self.create_cache(1).config().num_layers
    }

    fn target_num_kv_heads(&self) -> i32 {
        self.create_cache(1).config().num_kv_heads as i32
    }

    fn target_head_dim(&self) -> i32 {
        self.create_cache(1).config().head_dim as i32
    }

    fn target_needs_mamba_cache(&self) -> bool {
        matches!(self, Self::Qwen3Next(_) | Self::NemotronH(_))
    }

    fn make_kv_cache(&self, max_seq_len: usize) -> KVCache {
        DynamicModel::create_cache(self, max_seq_len)
    }

    fn make_mamba_cache(&self) -> Option<MambaCache> {
        DynamicModel::create_mamba_cache(self)
    }
}

impl DFlashTarget for Qwen3NextForCausalLM {
    fn embed_tokens(&mut self, input_ids: &Array) -> Result<Array, Exception> {
        Module::forward(&mut self.model.embed_tokens, input_ids)
    }

    fn forward_with_capture(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        kv_cache: Option<&mut KVCache>,
        mamba_cache: Option<&mut MambaCache>,
        capture: &mut SpecCapture,
    ) -> Result<Array, Exception> {
        let h = self.model.forward_with_cache_and_capture(
            input_ids,
            mask,
            kv_cache,
            None,
            mamba_cache,
            Some(capture),
        )?;
        // lm_head projection.
        if let Some(ref mut lm_head) = self.lm_head {
            Ok(lm_head.forward(&h)?)
        } else {
            Ok(self.model.embed_tokens.as_linear(&h))
        }
    }

    fn lm_head_project(&mut self, hidden: &Array) -> Result<Array, Exception> {
        if let Some(ref mut lm_head) = self.lm_head {
            Ok(lm_head.forward(hidden)?)
        } else {
            Ok(self.model.embed_tokens.as_linear(hidden))
        }
    }

    fn target_hidden_size(&self) -> i32 {
        self.config.hidden_size
    }

    fn target_num_layers(&self) -> usize {
        self.config.num_hidden_layers as usize
    }

    fn target_num_kv_heads(&self) -> i32 {
        self.config.num_kv_heads()
    }

    fn target_head_dim(&self) -> i32 {
        self.config.head_dim()
    }

    fn target_needs_mamba_cache(&self) -> bool {
        true
    }

    fn make_mamba_cache(&self) -> Option<MambaCache> {
        Some(MambaCache::new(self.config.num_hidden_layers as usize))
    }
}

// ----------------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::architectures::dflash_draft::{DFlashDraftConfig, DFlashDraftModel, DFlashExtras};
    use crate::architectures::qwen3::Qwen3Config;
    use serial_test::serial;

    /// A small vocabulary so that a random draft is right often enough for
    /// a run to accept some of its guesses as well as reject others.
    const VOCAB: i32 = 9;

    fn tiny_target() -> Qwen3ForCausalLM {
        let config = Qwen3Config {
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 4,
            num_attention_heads: 2,
            num_key_value_heads: Some(1),
            head_dim: 16,
            vocab_size: VOCAB,
            ..Default::default()
        };
        Qwen3ForCausalLM::new(config).unwrap()
    }

    fn tiny_draft(block_size: i32) -> DFlashDraftModel {
        let config = DFlashDraftConfig {
            model_type: "dflash_qwen3".to_string(),
            hidden_size: 32,
            num_hidden_layers: 2,
            intermediate_size: 64,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            rms_norm_eps: 1e-6,
            vocab_size: VOCAB,
            max_position_embeddings: 64,
            rope_theta: Some(10_000.0),
            head_dim: 16,
            tie_word_embeddings: true,
            block_size,
            dflash_config: DFlashExtras {
                // Tap the last two layers of the tiny 4-layer target.
                target_layer_ids: vec![2, 3],
                mask_token_id: 5,
                ..Default::default()
            },
            ..Default::default()
        };
        DFlashDraftModel::new(config).unwrap()
    }

    fn config(generation: GenerationConfig) -> DFlashConfig {
        DFlashConfig::new(generation)
    }

    #[test]
    #[serial]
    fn test_dflash_decoder_generates_expected_number_of_tokens() {
        // Random weights are fine — we're validating the *plumbing*:
        //   1. prefill captures hidden states at the tapped layers
        //   2. draft produces a block of tokens (via the target lm_head)
        //   3. verify returns logits, argmax acceptance advances the sequence
        //   4. rollback on partial accept does not break subsequent forwards
        //   5. the loop terminates at `max_new_tokens`
        let mut decoder = DFlashDecoder::new(tiny_target(), tiny_draft(4));
        let output = decoder
            .generate(&[1, 2, 3, 4, 5], &config(GenerationConfig::greedy(12)))
            .unwrap();

        assert_eq!(
            output.tokens.len(),
            5 + 12,
            "should produce max_new_tokens after prompt"
        );
        assert_eq!(output.metrics.num_generated, 12);
        assert!(
            output.metrics.acceptance_lengths.iter().all(|&a| a >= 1),
            "every speculative step must accept at least the bonus token"
        );
        assert!(
            output.metrics.total_drafted >= 4,
            "at least one draft block"
        );
    }

    #[test]
    #[serial]
    fn a_stop_token_ends_the_decode_and_is_left_out() {
        let mut decoder = DFlashDecoder::new(tiny_target(), tiny_draft(4));
        let prompt = [1u32, 2, 3];
        let plain = decoder
            .generate_plain(&prompt, &config(GenerationConfig::greedy(10)))
            .unwrap();
        let made = &plain.tokens[prompt.len()..];
        let stop = made[4];
        let at = made.iter().position(|&t| t == stop).unwrap();

        let stopping = config(GenerationConfig::greedy(32).with_stop_tokens(vec![stop]));
        let speculative = decoder.generate(&prompt, &stopping).unwrap();
        let plain_stopped = decoder.generate_plain(&prompt, &stopping).unwrap();
        for out in [&speculative, &plain_stopped] {
            assert_eq!(out.tokens, plain.tokens[..prompt.len() + at]);
            assert_eq!(out.metrics.num_generated, at);
        }
    }

    /// Speculation changes the speed and not the tokens: greedy DFlash is
    /// greedy decoding, the draft being right or wrong, and so is a seeded
    /// DFlash run that draws through the filters, the penalties and a bias,
    /// the target drawing at each position as plain decoding draws it. The
    /// draft here is random, so most blocks are rejected after a token or
    /// two and every round rolls the target's cache back.
    #[test]
    #[serial]
    fn dflash_decodes_the_tokens_plain_decoding_does() {
        let prompt = [1u32, 2, 3, 4, 5];
        let configs = [
            GenerationConfig::greedy(40),
            GenerationConfig::greedy(40)
                .with_repetition_penalty(1.3)
                .with_presence_penalty(0.4)
                .with_logit_bias(3, 2.0),
            GenerationConfig::sampling(40, 1.0)
                .with_top_k(0)
                .with_top_p(1.0)
                .with_min_p(0.0)
                .with_repetition_penalty(1.0)
                .with_seed(13),
            GenerationConfig::sampling(40, 0.8)
                .with_top_k(5)
                .with_top_p(0.9)
                .with_frequency_penalty(0.3)
                .with_seed(5),
        ];
        let mut accepted_somewhere = false;
        for block_size in [2, 3, 5] {
            let mut decoder = DFlashDecoder::new(tiny_target(), tiny_draft(block_size));
            for (i, generation) in configs.iter().enumerate() {
                let config = config(generation.clone());
                let plain = decoder.generate_plain(&prompt, &config).unwrap();
                let speculative = decoder.generate(&prompt, &config).unwrap();
                pmetal_runtime::check_unobserved_error().expect("no runtime op failed");
                assert_eq!(
                    speculative.tokens, plain.tokens,
                    "block {block_size}, config {i}"
                );
                assert_eq!(speculative.metrics.num_generated, 40);
                accepted_somewhere |= speculative.metrics.total_accepted > 0;
            }
        }
        assert!(
            accepted_somewhere,
            "no draft was ever accepted: the acceptance path went untested"
        );
    }

    /// A seed fixes the draw of every position, so a seeded run repeats and
    /// another seed gives other tokens.
    #[test]
    #[serial]
    fn a_seeded_dflash_run_repeats() {
        let mut decoder = DFlashDecoder::new(tiny_target(), tiny_draft(4));
        let seeded = |seed| {
            config(
                GenerationConfig::sampling(24, 1.2)
                    .with_top_k(0)
                    .with_top_p(1.0)
                    .with_min_p(0.0)
                    .with_repetition_penalty(1.0)
                    .with_seed(seed),
            )
        };
        let first = decoder.generate(&[1, 2, 3], &seeded(7)).unwrap();
        assert_eq!(
            decoder.generate(&[1, 2, 3], &seeded(7)).unwrap().tokens,
            first.tokens
        );
        assert_ne!(
            decoder.generate(&[1, 2, 3], &seeded(8)).unwrap().tokens,
            first.tokens
        );
    }

    #[test]
    #[serial]
    fn degenerate_requests() {
        let mut decoder = DFlashDecoder::new(tiny_target(), tiny_draft(4));
        assert!(decoder.generate(&[], &DFlashConfig::default()).is_err());
        let none = decoder
            .generate(&[1, 2], &config(GenerationConfig::greedy(0)))
            .unwrap();
        assert_eq!(none.tokens, vec![1, 2]);
        // A block of one is plain decoding.
        let one = DFlashConfig {
            speculative_tokens: Some(1),
            ..config(GenerationConfig::greedy(6))
        };
        let out = decoder.generate(&[1, 2], &one).unwrap();
        assert_eq!(out.metrics.num_generated, 6);
    }
}
