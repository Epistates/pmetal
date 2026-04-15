//! DFlash speculative-decoding loop.
//!
//! Glues a [`crate::architectures::dflash_draft::DFlashDraftModel`] together
//! with a Qwen3 / Qwen3.5 target so a single call to [`DFlashDecoder::generate`]
//! runs the full draft→verify→accept→rollback cycle.
//!
//! # Verification mode
//!
//! This implementation uses the `parallel-replay` verification mode from the
//! upstream dflash-mlx Python reference (`dflash_mlx/runtime.py`): the target
//! runs one forward pass over the whole proposed block, the verifier's argmax
//! at every position is compared with the drafted tokens, and the longest
//! matching prefix is accepted plus one bonus correction token. At
//! `temperature = 0` this produces output that is bit-identical to greedy
//! baseline decoding.
//!
//! The other four upstream verification modes (stream, chunked,
//! parallel-lazy-logits, parallel-greedy-argmax) are straight-line variants
//! of the same plumbing and can be added as needed without touching the
//! target trait.
//!
//! # Target model support
//!
//! Targets plug in via the [`DFlashTarget`] trait. Qwen3 is the primary
//! target today — its [`crate::architectures::qwen3::Qwen3ForCausalLM`]
//! implementation is at the bottom of this file. Qwen3.5 (`qwen3_next`) will
//! implement the same trait once its GDN verify-input capture is wired
//! through the mixer; the loop above is architecture-agnostic.

use std::path::Path;

use pmetal_bridge::compat::{Array, Dtype, Exception, Module, ops};

use pmetal_mlx::kv_cache::{KVCache, KVCacheConfig, MambaCache};
use pmetal_mlx::speculative::SpecCapture;

use crate::architectures::dflash_draft::{DFlashDraftConfig, DFlashDraftModel};
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
    /// Native-bridge targets own their own cache and override this to
    /// rewind the internal state, ignoring the external `kv_cache` arg.
    /// Such targets should also return a dummy cache from
    /// [`DFlashTarget::make_kv_cache`] that is never touched.
    fn rollback_rejected(&mut self, kv_cache: &mut KVCache, n: usize) {
        kv_cache.rollback(n);
    }

    /// Whether this target supports tree-verify — per-position RoPE +
    /// custom additive attention mask on the forward pass. The default
    /// is `false`; the [`crate::dflash_native_target::NativeQwen3Target`]
    /// override returns `true`.
    fn supports_tree_verify(&self) -> bool {
        false
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
    /// order). Default: no-op. Only the native-bridge target
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

/// Which device runs the DFlash draft model.
///
/// Today only [`DraftBackend::Gpu`] is supported. [`DraftBackend::Ane`] is
/// the roadmap path — it would offload the draft to the Apple Neural
/// Engine while the target keeps running on the GPU, removing contention
/// on the draft→target critical path. The ANE MIL compilation of
/// `DFlashAttention`'s cross-attention (which has to consume the
/// `target_hidden` tensor alongside the draft hidden states) is not
/// plumbed yet; requesting `Ane` returns an explicit error so callers can
/// feature-detect rather than silently fall back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DraftBackend {
    /// Run the draft on the GPU via mlx-rs (the default today).
    Gpu,
    /// Run the draft on the Apple Neural Engine — not yet implemented.
    Ane,
}

impl Default for DraftBackend {
    fn default() -> Self {
        DraftBackend::Gpu
    }
}

/// Runtime configuration for [`DFlashDecoder::generate`].
#[derive(Debug, Clone)]
pub struct DFlashConfig {
    /// Upper bound on new tokens produced after the prompt.
    pub max_new_tokens: usize,
    /// Sampling temperature. `0.0` = greedy (bit-identical to baseline).
    pub temperature: f32,
    /// Token ids whose appearance ends generation.
    pub stop_tokens: Vec<i32>,
    /// Optional override for the draft block size. `None` uses the value
    /// stored in the draft model config.
    pub speculative_tokens: Option<usize>,
    /// Which device runs the draft. See [`DraftBackend`].
    pub draft_backend: DraftBackend,
}

impl Default for DFlashConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 128,
            temperature: 0.0,
            stop_tokens: Vec::new(),
            speculative_tokens: None,
            draft_backend: DraftBackend::default(),
        }
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
    /// Token ids, including the prompt.
    pub tokens: Vec<i32>,
    /// Metrics collected during the run.
    pub metrics: DFlashMetrics,
}

// ----------------------------------------------------------------------------
// Decoder
// ----------------------------------------------------------------------------

/// DFlash speculative decoder.
///
/// Holds a target model, a DFlash draft, and a reusable [`SpecCapture`]
/// buffer. One instance is intended to service many `generate` calls — the
/// draft / target weights stay resident and only the per-request KV caches
/// are allocated fresh.
pub struct DFlashDecoder<T: DFlashTarget> {
    target: T,
    draft: DFlashDraftModel,
    target_layer_ids: Vec<usize>,
}

impl<T: DFlashTarget> DFlashDecoder<T> {
    /// Create a new decoder.
    ///
    /// `target_layer_ids` tells the loop which target layers to tap for the
    /// draft's conditioning input; it must match the
    /// `dflash_config.target_layer_ids` the draft was trained with. Use
    /// [`DFlashDraftModel::config`]`.target_layer_ids()` to obtain this list
    /// from the draft checkpoint.
    pub fn new(target: T, draft: DFlashDraftModel) -> Self {
        let target_layer_ids = draft.config.target_layer_ids();
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
    pub fn draft(&self) -> &DFlashDraftModel {
        &self.draft
    }

    /// Run the DFlash draft/verify loop starting from `prompt_ids`
    /// (shape `[1, prompt_len]`, `Int32` dtype).
    pub fn generate(
        &mut self,
        prompt_ids: &Array,
        config: &DFlashConfig,
    ) -> Result<DFlashOutput, Exception> {
        if config.draft_backend == DraftBackend::Ane {
            return Err(Exception::custom(
                "DFlashDecoder: draft_backend=Ane is not yet implemented. \
                 The DFlashAttention cross-attention (target_hidden || hidden_states) \
                 needs ANE MIL compilation; tracked in the project roadmap.",
            ));
        }
        let prompt_len = prompt_ids.dim(1) as usize;
        let total_max_tokens = prompt_len + config.max_new_tokens;
        let block_size = config
            .speculative_tokens
            .unwrap_or_else(|| self.draft.block_size())
            .max(1)
            .min(self.draft.block_size());

        // Verify steps temporarily extend the cache by `block_size` tokens
        // before rolling back the rejected tail, so size the cache with
        // enough headroom to absorb one full speculative block past the
        // final-emitted-token mark.
        let cache_max_tokens = total_max_tokens + block_size;
        let mut target_cache = self.target.make_kv_cache(cache_max_tokens);

        // Defensive sanity check: rolling back across a sliding-window
        // eviction boundary would silently produce wrong outputs because
        // the rolled-back positions may already have been dropped from
        // the cache. Refuse to run if the cache window is smaller than the
        // block size we may need to rewind. (Inspired by mlx-lm's
        // `can_trim_prompt_cache` guard, commit f56d997.)
        let cache_window = match target_cache.config().mode {
            pmetal_mlx::kv_cache::CacheMode::SlidingWindow { window_size } => Some(window_size),
            pmetal_mlx::kv_cache::CacheMode::Rotating { max_size, .. } => Some(max_size),
            _ => None,
        };
        if let Some(window) = cache_window
            && (window as usize) < block_size + 1
        {
            return Err(Exception::custom(format!(
                "DFlashDecoder: target cache window ({window}) must be larger than \
                 speculative block_size ({block_size}) so a rejected-tail rollback \
                 can land on positions that are still in the window."
            )));
        }
        let mut mamba_cache: Option<MambaCache> = self.target.make_mamba_cache();
        let mut capture = SpecCapture::with_layers(self.target_layer_ids.clone());

        // ── Prefill ───────────────────────────────────────────────────────
        // The target processes the whole prompt in one pass and records
        // hidden states at every tapped layer. We sample the first "free"
        // token from the last prompt position and keep the tapped hidden
        // states as initial `target_hidden` for the draft.
        let prefill_logits = self.target.forward_with_capture(
            prompt_ids,
            None,
            Some(&mut target_cache),
            mamba_cache.as_mut(),
            &mut capture,
        )?;
        let last_logits = slice_last_time_step(&prefill_logits)?;
        let first_token = sample_token_argmax(&last_logits)?;

        let mut output_tokens: Vec<i32> = tokens_from_array(prompt_ids)?;
        output_tokens.push(first_token);

        let mut target_hidden = capture.stack_hidden()?;

        let mut metrics = DFlashMetrics::default();

        // ── Decode loop ───────────────────────────────────────────────────
        while output_tokens.len() < total_max_tokens {
            // Draft: start with the latest accepted token, fill the rest
            // with mask-token noise.
            let seed_token = *output_tokens.last().unwrap();
            let mut block_tokens = Vec::with_capacity(block_size);
            block_tokens.push(seed_token);
            for _ in 1..block_size {
                block_tokens.push(self.draft.mask_token_id());
            }
            let block_input = array_from_i32_row(&block_tokens);

            // noise_embedding is the target's token embedding of the block.
            let noise_embedding = self.target.embed_tokens(&block_input)?;

            // Cacheless draft pass: simpler to get right than the cached
            // path and plenty fast for a ~4-layer DFlash checkpoint. Cache
            // reuse can be added later for long generations.
            let draft_hidden = self.draft.forward(&noise_embedding, &target_hidden, None)?;
            let draft_suffix = slice_axis_1(&draft_hidden, 1, block_size as i32);
            let draft_logits = self.target.lm_head_project(&draft_suffix)?;
            let drafted_tokens = argmax_last_axis(&draft_logits)?;
            for (i, tok) in drafted_tokens.into_iter().enumerate() {
                block_tokens[i + 1] = tok;
            }

            // Snapshot the Mamba cache BEFORE verify so a partial-accept
            // rollback can replay the GDN recurrence through only the
            // accepted prefix. Pure-attention targets (Qwen3) have no
            // mamba_cache and this is a no-op.
            let mamba_snapshot: Option<Vec<pmetal_mlx::kv_cache::MambaSnapshot>> =
                mamba_cache.as_ref().map(|c| c.snapshot());

            // Verify: one target forward pass over the whole block.
            capture.clear();
            let verify_input = array_from_i32_row(&block_tokens);
            let verify_logits = self.target.forward_with_capture(
                &verify_input,
                None,
                Some(&mut target_cache),
                mamba_cache.as_mut(),
                &mut capture,
            )?;
            let verifier_tokens = argmax_over_sequence(&verify_logits)?;
            let verifier_hidden_all = capture.stack_hidden()?;

            // Accept longest matching prefix + one bonus token.
            let matched = longest_prefix_match(&block_tokens[1..], &verifier_tokens[..block_size - 1]);
            let accepted_inputs = matched + 1;
            let bonus_token = verifier_tokens[matched];

            if std::env::var_os("PMETAL_DFLASH_TRACE").is_some() {
                eprintln!(
                    "[dflash trace] drafted={:?} verifier={:?} matched={}",
                    &block_tokens[1..],
                    &verifier_tokens[..block_size.min(verifier_tokens.len())],
                    matched,
                );
            }

            metrics.acceptance_lengths.push(accepted_inputs);
            metrics.total_drafted += block_size;
            metrics.total_accepted += matched;

            // Rollback rejected positions in the target's KV cache.
            let rejected = block_size - accepted_inputs;
            if rejected > 0 {
                self.target.rollback_rejected(&mut target_cache, rejected);
                if let (Some(ref mut mamba), Some(snaps)) =
                    (mamba_cache.as_mut(), mamba_snapshot.as_ref())
                {
                    // Build per-layer verify inputs in cache layer order so
                    // `rewind_from_snapshots` can replay each GDN layer
                    // forward through the accepted prefix.
                    let num_layers = mamba.num_layers();
                    let mut per_layer: Vec<Option<pmetal_mlx::kv_cache::GdnVerifyInputs>> =
                        Vec::with_capacity(num_layers);
                    for layer_idx in 0..num_layers {
                        per_layer.push(capture.gdn_inputs.remove(&layer_idx));
                    }
                    mamba.rewind_from_snapshots(snaps, &per_layer, accepted_inputs)?;
                }
            }

            // Commit: append the accepted block tokens and the bonus.
            let accepted_slice = &block_tokens[1..accepted_inputs];
            output_tokens.extend_from_slice(accepted_slice);
            output_tokens.push(bonus_token);

            // The next draft step conditions on the tapped hidden states of
            // the accepted positions.
            target_hidden = slice_axis_1(&verifier_hidden_all, 0, accepted_inputs as i32);

            // Stop-token check over the newly emitted tail.
            if !config.stop_tokens.is_empty() {
                let tail_start = output_tokens.len().saturating_sub(accepted_inputs + 1);
                let mut hit = None;
                for (offset, tok) in output_tokens[tail_start..].iter().enumerate() {
                    if config.stop_tokens.contains(tok) {
                        hit = Some(tail_start + offset);
                        break;
                    }
                }
                if let Some(end_idx) = hit {
                    output_tokens.truncate(end_idx + 1);
                    break;
                }
            }
        }

        if output_tokens.len() > total_max_tokens {
            output_tokens.truncate(total_max_tokens);
        }
        metrics.num_generated = output_tokens.len().saturating_sub(prompt_len);

        Ok(DFlashOutput {
            tokens: output_tokens,
            metrics,
        })
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
    /// past that costs more than it gains. Disable with
    /// `PMETAL_DFLASH_ADAPTIVE=0` for benchmarking.
    ///
    /// **Numerics note.** The tree forward routes through MLX's
    /// `has_mask=true` SDPA kernel specialization, while greedy
    /// decode and linear DFlash use `do_causal=true`. These
    /// specializations are mathematically equivalent but produce
    /// slightly different bf16 numerics due to different fused
    /// instruction orderings inside the steel attention kernel
    /// (see `.strategy/mlx/.../steel_attention.h`). The K/V values
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
    /// anything that's not `NativeQwen3Target` today).
    pub fn generate_ddtree(
        &mut self,
        prompt_ids: &Array,
        config: &DFlashConfig,
        tree_budget: usize,
    ) -> Result<DFlashOutput, Exception> {
        if !self.target.supports_tree_verify() || tree_budget == 0 {
            return self.generate(prompt_ids, config);
        }
        if config.draft_backend == DraftBackend::Ane {
            return Err(Exception::custom(
                "DFlashDecoder::generate_ddtree: draft_backend=Ane is not yet implemented",
            ));
        }
        let prompt_len = prompt_ids.dim(1) as usize;
        let total_max_tokens = prompt_len + config.max_new_tokens;
        let block_size = config
            .speculative_tokens
            .unwrap_or_else(|| self.draft.block_size())
            .max(1)
            .min(self.draft.block_size());
        // Tree horizon = `block_size - 1` (the root eats one slot, the
        // other positions are drafted). Matches DDTree upstream.
        let draft_horizon = block_size.saturating_sub(1);
        if draft_horizon == 0 {
            return self.generate(prompt_ids, config);
        }
        let tree_budget = tree_budget.max(1);
        let max_tree_nodes = 1 + tree_budget; // +1 for the root

        // Allocate enough cache for the worst case: every round writes
        // `max_tree_nodes` tokens and keeps at most `max_tree_nodes - 1`
        // (tree_budget) after compaction. So the cache needs to hold
        // `prompt + max_new_tokens + max_tree_nodes` tokens at peak.
        let cache_max_tokens = total_max_tokens + max_tree_nodes;
        let mut target_cache = self.target.make_kv_cache(cache_max_tokens);

        let mut mamba_cache: Option<MambaCache> = self.target.make_mamba_cache();
        let mut capture = SpecCapture::with_layers(self.target_layer_ids.clone());

        // ── Prefill ───────────────────────────────────────────────────
        let prefill_logits = self.target.forward_with_capture(
            prompt_ids,
            None,
            Some(&mut target_cache),
            mamba_cache.as_mut(),
            &mut capture,
        )?;
        let last_logits = slice_last_time_step(&prefill_logits)?;
        let first_token = sample_token_argmax(&last_logits)?;
        let mut output_tokens: Vec<i32> = tokens_from_array(prompt_ids)?;
        output_tokens.push(first_token);

        let mut target_hidden = capture.stack_hidden()?;
        let mut metrics = DFlashMetrics::default();
        let mut past_length = prompt_len; // cache length after prefill
        let mut seed_token = first_token;

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
        //
        // Disable with `PMETAL_DFLASH_ADAPTIVE=0` for benchmarking
        // against the fixed-budget baseline.
        let adaptive_enabled = std::env::var("PMETAL_DFLASH_ADAPTIVE")
            .map(|v| v != "0")
            .unwrap_or(true);
        let base_budget = tree_budget;
        let min_budget = (base_budget / 3).max(4);
        let mut effective_budget = base_budget;
        // EMA over per-round `matched` (= accepted_count - 1). The
        // optimistic prior of 1.5 keeps the budget at base for the
        // first few rounds while the EMA warms up.
        let mut ema_matched: f32 = 1.5;
        let ema_alpha: f32 = 0.35;
        // ── Iter-1 fast-path ───────────────────────────────────────
        // The very first round after prefill has the LEAST signal:
        // the draft has no recently-emitted hidden states to anchor
        // against, only the prompt's prefill capture. Some workloads
        // benefit from skipping the tree on iter 1 and just doing a
        // single greedy step. Off by default — empirically iter 1 is
        // not worse than steady-state for our (Qwen3 / DFlash-b16)
        // pair, but the toggle lets users A/B test their own pair.
        let skip_iter1 = std::env::var_os("PMETAL_DFLASH_SKIP_ITER1").is_some();
        let mut round_index: usize = 0;

        while output_tokens.len() < total_max_tokens {
            round_index += 1;

            // ── Iter-1 skip: greedy step in lieu of tree round ─────
            // One target forward over [seed], no draft, no tree.
            // Commits a single bonus token then proceeds to iter 2
            // which uses the full tree path.
            if skip_iter1 && round_index == 1 {
                let seed_arr = array_from_i32_row(&[seed_token]);
                capture.clear();
                let logits = self.target.forward_with_capture(
                    &seed_arr,
                    None,
                    Some(&mut target_cache),
                    mamba_cache.as_mut(),
                    &mut capture,
                )?;
                let argmax = argmax_over_sequence(&logits)?;
                let bonus = argmax[0];
                output_tokens.push(bonus);
                past_length += 1;
                target_hidden = capture.stack_hidden()?;
                seed_token = bonus;
                metrics.acceptance_lengths.push(1);
                continue;
            }

            // ── Draft: build noise embedding and run draft forward ──
            let mut block_tokens: Vec<i32> = Vec::with_capacity(block_size);
            block_tokens.push(seed_token);
            for _ in 1..block_size {
                block_tokens.push(self.draft.mask_token_id());
            }
            let block_input = array_from_i32_row(&block_tokens);
            let noise_embedding = self.target.embed_tokens(&block_input)?;
            let draft_hidden = self.draft.forward(&noise_embedding, &target_hidden, None)?;
            // [1, draft_horizon, hidden] — slice away the root position.
            let draft_suffix =
                slice_axis_1(&draft_hidden, 1, (1 + draft_horizon) as i32);
            // [1, draft_horizon, vocab]
            let draft_logits = self.target.lm_head_project(&draft_suffix)?;
            // [draft_horizon, vocab]
            let draft_logits_2d = draft_logits.reshape(&[
                draft_horizon as i32,
                draft_logits.dim(2),
            ]);

            // ── Tree build + compile ────────────────────────────────
            let tree = crate::ddtree::build_tree(&draft_logits_2d, effective_budget);
            let compiled = crate::ddtree::compile_tree(
                &tree,
                seed_token,
                past_length as i32,
                past_length as i32,
                self.target_dtype_hint(),
            );
            let tree_length = compiled.current_length;

            // ── Verify: target forward with tree inputs ─────────────
            capture.clear();
            let verify_logits = self.target.forward_tree_verify(
                &compiled.verify_input_ids,
                &compiled.verify_position_ids,
                &compiled.attention_mask,
                &mut capture,
            )?;
            let verifier_tokens = argmax_last_axis(&verify_logits)?;
            let verifier_hidden_all = capture.stack_hidden()?;

            // ── Walk the verified path ──────────────────────────────
            let (accepted_indices, bonus_token) =
                crate::ddtree::follow_verified_tree(&tree.child_maps, &verifier_tokens);
            let accepted_count = accepted_indices.len();
            let matched = accepted_count.saturating_sub(1);

            if std::env::var_os("PMETAL_DFLASH_TRACE").is_some() {
                eprintln!(
                    "[ddtree trace] budget={effective_budget} tree_len={tree_length} \
                     accepted={accepted_count} matched={matched} bonus={bonus_token} \
                     ema={ema_matched:.2}"
                );
            }

            // ── Adaptive budget update ──────────────────────────────
            // Update EMA, then pick the next round's budget. Three
            // tiers: high-confidence (≥2 matched on average) shrinks
            // hardest, mid (≥1.2) shrinks modestly, low restores the
            // base. The tiers were tuned on Qwen3-4B where the sweep
            // peak sits at ~12 nodes; reduce_factors here translate
            // to budgets like {12, 8, 12} which all sit on the
            // throughput plateau. Capped at min_budget so the floor
            // is never below the smallest useful tree.
            if adaptive_enabled {
                ema_matched = ema_matched * (1.0 - ema_alpha) + matched as f32 * ema_alpha;
                effective_budget = if ema_matched >= 2.0 {
                    (base_budget / 2).max(min_budget)
                } else if ema_matched >= 1.2 {
                    (base_budget * 2 / 3).max(min_budget)
                } else {
                    base_budget
                };
            }

            metrics.acceptance_lengths.push(accepted_count);
            metrics.total_drafted += tree_length.saturating_sub(1);
            metrics.total_accepted += matched;

            // ── Commit accepted path ────────────────────────────────
            // `accepted_indices[0]` is the root — the seed we already
            // have in `output_tokens`. For positions 1..accepted_count
            // we pull the tree's token id.
            for &idx in &accepted_indices[1..] {
                let t = tree.node_token_ids[idx - 1];
                output_tokens.push(t);
            }
            output_tokens.push(bonus_token);

            // ── Compact the cache to keep only accepted positions ──
            self.target.compact_tree_cache(
                past_length,
                tree_length,
                &accepted_indices,
            );
            past_length += accepted_count;

            // ── Next round's target_hidden = selected rows ─────────
            // `verifier_hidden_all` is [1, tree_length, hidden*taps].
            // Gather the accepted rows via take_axis on the seq dim.
            let accepted_i32: Vec<i32> =
                accepted_indices.iter().map(|&i| i as i32).collect();
            let idx_arr = Array::from_slice(&accepted_i32, &[accepted_count as i32]);
            target_hidden = verifier_hidden_all.take_axis(&idx_arr, 1);

            // Next iteration's seed is the bonus token.
            seed_token = bonus_token;

            // ── Stop-token check over the newly emitted tail ───────
            if !config.stop_tokens.is_empty() {
                let tail_start = output_tokens.len().saturating_sub(accepted_count + 1);
                let mut hit = None;
                for (offset, tok) in output_tokens[tail_start..].iter().enumerate() {
                    if config.stop_tokens.contains(tok) {
                        hit = Some(tail_start + offset);
                        break;
                    }
                }
                if let Some(end_idx) = hit {
                    output_tokens.truncate(end_idx + 1);
                    break;
                }
            }
        }

        if output_tokens.len() > total_max_tokens {
            output_tokens.truncate(total_max_tokens);
        }
        metrics.num_generated = output_tokens.len().saturating_sub(prompt_len);
        Ok(DFlashOutput {
            tokens: output_tokens,
            metrics,
        })
    }

    /// Dtype for tree attention mask. Must match (or promote to) the
    /// model's working dtype — MLX's SDPA enforces this constraint.
    /// For Qwen3 (bf16) we use bf16. The bf16 representation of -inf
    /// is exact (same exponent encoding as f32) so we don't lose
    /// information at the mask values themselves; any precision
    /// difference vs the linear-causal path comes from the explicit-
    /// mask SDPA kernel, not the mask itself.
    fn target_dtype_hint(&self) -> i32 {
        pmetal_bridge::compat::Dtype::Bfloat16.as_i32()
    }
}

// ----------------------------------------------------------------------------
// Array <-> Vec helpers
// ----------------------------------------------------------------------------

fn tokens_from_array(ids: &Array) -> Result<Vec<i32>, Exception> {
    if ids.ndim() != 2 {
        return Err(Exception::custom(format!(
            "expected [B, T] token tensor, got shape {:?}",
            ids.shape()
        )));
    }
    let evaled = ids.clone();
    let _ = evaled.eval();
    Ok(evaled.as_slice::<i32>().to_vec())
}

fn array_from_i32_row(tokens: &[i32]) -> Array {
    Array::from_slice(tokens, &[1, tokens.len() as i32])
}

fn slice_axis_1(arr: &Array, start: i32, stop: i32) -> Array {
    let rank = arr.shape().len();
    let mut start_v = vec![0i32; rank];
    let mut stop_v: Vec<i32> = arr.shape().to_vec();
    start_v[1] = start;
    stop_v[1] = stop;
    arr.slice(&start_v, &stop_v)
}

/// Slice out the final sequence step: `logits[..., -1:, :]` with a squeeze
/// along the sequence axis, returning `[B, vocab]`.
fn slice_last_time_step(logits: &Array) -> Result<Array, Exception> {
    if logits.ndim() != 3 {
        return Err(Exception::custom(format!(
            "expected [B, T, V] logits, got shape {:?}",
            logits.shape()
        )));
    }
    let t = logits.dim(1);
    if t <= 0 {
        return Err(Exception::custom("logits sequence length must be > 0"));
    }
    let b = logits.dim(0);
    let v = logits.dim(2);
    let slice = logits.slice(&[0, t - 1, 0], &[b, t, v]);
    Ok(slice.reshape(&[b, v]))
}

fn sample_token_argmax(logits_2d: &Array) -> Result<i32, Exception> {
    if logits_2d.ndim() != 2 {
        return Err(Exception::custom(format!(
            "expected [B, V] logits, got shape {:?}",
            logits_2d.shape()
        )));
    }
    let tokens = argmax_last_axis(logits_2d)?;
    tokens
        .into_iter()
        .next()
        .ok_or_else(|| Exception::custom("argmax returned empty tensor"))
}

/// Compute `argmax(logits, axis=-1)` and materialize the result to a
/// `Vec<i32>`. MLX's argmax returns `uint32`, so we cast to the i32-friendly
/// token-id form on the CPU after eval.
fn argmax_last_axis(logits: &Array) -> Result<Vec<i32>, Exception> {
    let argmax = ops::argmax_axis(logits, -1);
    let _ = argmax.eval();
    Ok(argmax.as_slice::<u32>().iter().map(|&u| u as i32).collect())
}

fn argmax_over_sequence(logits: &Array) -> Result<Vec<i32>, Exception> {
    if logits.ndim() != 3 {
        return Err(Exception::custom(format!(
            "expected [B, T, V] verify logits, got shape {:?}",
            logits.shape()
        )));
    }
    argmax_last_axis(logits)
}

fn longest_prefix_match(drafted: &[i32], verifier: &[i32]) -> usize {
    drafted
        .iter()
        .zip(verifier.iter())
        .take_while(|(d, v)| d == v)
        .count()
}

// ----------------------------------------------------------------------------
// Qwen3 target implementation
// ----------------------------------------------------------------------------

use crate::architectures::qwen3::Qwen3ForCausalLM;

/// Quantization mode applied to a DFlash draft after loading.
///
/// DFlash speedup is sensitive to draft quality — 4-bit weight quantization
/// alone drops acceptance rates. `Fp8` keeps ±240 range which covers every
/// Linear layer in the upstream Qwen3-DFlash checkpoints and shaves ~50%
/// off draft memory vs BF16 with minimal acceptance-rate regression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DFlashDraftQuant {
    /// Keep weights at their loaded dtype (typically BF16).
    None,
    /// Quantize every `.weight` tensor to FP8 (E4M3) in-place via
    /// [`crate::fp8_utils::quantize_model_linears`]. Leaves biases and
    /// normalisation scales alone.
    Fp8,
}

impl Default for DFlashDraftQuant {
    fn default() -> Self {
        DFlashDraftQuant::None
    }
}

/// Load a DFlash draft model from a local directory.
///
/// Equivalent to
/// [`load_dflash_draft_from_dir_quantized`]`(dir, DFlashDraftQuant::None)`.
pub fn load_dflash_draft_from_dir(
    model_dir: impl AsRef<Path>,
) -> Result<
    (
        DFlashDraftModel,
        crate::architectures::dflash_draft::LoadReport,
    ),
    Exception,
> {
    load_dflash_draft_from_dir_quantized(model_dir, DFlashDraftQuant::None)
}

/// Load a DFlash draft model from a local directory with an optional
/// post-load weight quantization pass.
///
/// Expects:
/// * `config.json` with the `DFlashDraftConfig` schema (including the
///   `dflash_config.target_layer_ids` / `mask_token_id` fields).
/// * One or more `*.safetensors` files whose weight names follow the
///   upstream naming described in [`DFlashDraftModel::load_weights`].
///
/// Returns the assembled draft plus a [`crate::architectures::dflash_draft::LoadReport`]
/// describing which parameter tensors were actually assigned.
pub fn load_dflash_draft_from_dir_quantized(
    model_dir: impl AsRef<Path>,
    quant: DFlashDraftQuant,
) -> Result<
    (
        DFlashDraftModel,
        crate::architectures::dflash_draft::LoadReport,
    ),
    Exception,
> {
    let dir = model_dir.as_ref();

    // Parse config.json.
    let cfg_path = dir.join("config.json");
    let cfg_bytes = std::fs::read(&cfg_path).map_err(|e| {
        Exception::custom(format!(
            "DFlash draft: failed to read {}: {e}",
            cfg_path.display()
        ))
    })?;
    let cfg: DFlashDraftConfig = serde_json::from_slice(&cfg_bytes).map_err(|e| {
        Exception::custom(format!(
            "DFlash draft: failed to parse {}: {e}",
            cfg_path.display()
        ))
    })?;

    // Build the model skeleton.
    let mut draft = DFlashDraftModel::new(cfg)?;

    // Load weights from safetensors.
    let weights = crate::loader::load_weights(dir)
        .map_err(|e| Exception::custom(format!("DFlash draft: weight load failed: {e}")))?;
    let report = draft.load_weights(&weights)?;

    // Optional post-load quantization.
    match quant {
        DFlashDraftQuant::None => {}
        DFlashDraftQuant::Fp8 => {
            crate::fp8_utils::quantize_model_linears(&mut draft)?;
        }
    }

    Ok((draft, report))
}

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
// Qwen3.5 (qwen3_next) target implementation
// ----------------------------------------------------------------------------

use crate::DynamicModel;
use crate::architectures::qwen3_next::Qwen3NextForCausalLM;

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
            let tanh = pmetal_bridge::compat::ops::tanh(&scaled);
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
            Self::StarCoder2(m) => Ok(m.embed_tokens.forward(input_ids)),
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
            Self::RecurrentGemma(_) => Err(dflash_architecture_unsupported("RecurrentGemma")),
            Self::Jamba(_) => Err(dflash_architecture_unsupported("Jamba")),
            Self::FalconH1(_) => Err(dflash_architecture_unsupported("FalconH1")),
            Self::Flux(_) | Self::Bert(_) => Err(Exception::custom(
                "DFlashTarget: Flux / BERT are not causal LMs and cannot serve as DFlash targets",
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
                m, input_ids, mask, kv_cache, mamba_cache, capture,
            ),
            Self::Qwen3Next(m) => <Qwen3NextForCausalLM as DFlashTarget>::forward_with_capture(
                m, input_ids, mask, kv_cache, mamba_cache, capture,
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
            Self::StarCoder2(m) => {
                let _ = mamba_cache;
                m.forward_with_capture(input_ids, mask, kv_cache, Some(capture))
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
            Self::RecurrentGemma(_) => Err(dflash_architecture_unsupported("RecurrentGemma")),
            Self::Jamba(_) => Err(dflash_architecture_unsupported("Jamba")),
            Self::FalconH1(_) => Err(dflash_architecture_unsupported("FalconH1")),
            Self::Flux(_) | Self::Bert(_) => Err(Exception::custom(
                "DFlashTarget: Flux / BERT are not causal LMs",
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
            Self::Phi(m) => Ok(Module::forward(&mut m.lm_head, hidden)?),
            Self::Phi4(m) => Ok(Module::forward(&mut m.lm_head, hidden)?),
            Self::DeepSeek(m) => Ok(Module::forward(&mut m.lm_head, hidden)?),
            Self::Qwen3MoE(m) => match &mut m.lm_head {
                Some(lm) => Ok(Module::forward(lm, hidden)?),
                None => {
                    let embed_w = m.model.embed_tokens.weight.value.as_ref();
                    Ok(hidden.matmul(&embed_w.t()))
                }
            },
            Self::StarCoder2(m) => Ok(Module::forward(&mut m.lm_head, hidden)?),
            Self::Gemma4(m) => {
                // Gemma 4 ties embeddings + applies final-logit softcap.
                let logits = m.model.embed_tokens.as_linear(hidden);
                Ok(apply_logit_softcap(&logits, m.config.final_logit_softcapping))
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
        matches!(
            self,
            Self::Qwen3Next(_) | Self::NemotronH(_) | Self::FalconH1(_) | Self::Jamba(_)
        )
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
    use crate::architectures::dflash_draft::{DFlashDraftConfig, DFlashExtras, DFlashDraftModel};
    use crate::architectures::qwen3::Qwen3Config;
    use serial_test::serial;

    fn tiny_target() -> Qwen3ForCausalLM {
        let config = Qwen3Config {
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 4,
            num_attention_heads: 2,
            num_key_value_heads: Some(1),
            head_dim: 16,
            vocab_size: 64,
            ..Default::default()
        };
        Qwen3ForCausalLM::new(config).unwrap()
    }

    fn tiny_draft() -> DFlashDraftModel {
        let config = DFlashDraftConfig {
            model_type: "dflash_qwen3".to_string(),
            hidden_size: 32,
            num_hidden_layers: 2,
            intermediate_size: 64,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            rms_norm_eps: 1e-6,
            vocab_size: 64,
            max_position_embeddings: 64,
            rope_theta: 10_000.0,
            head_dim: 16,
            tie_word_embeddings: true,
            attention_bias: false,
            rope_scaling: None,
            block_size: 4,
            dflash_config: DFlashExtras {
                // Tap the last two layers of the tiny 4-layer target.
                target_layer_ids: vec![2, 3],
                mask_token_id: 5,
            },
        };
        DFlashDraftModel::new(config).unwrap()
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
        let target = tiny_target();
        let draft = tiny_draft();
        let mut decoder = DFlashDecoder::new(target, draft);

        let prompt = Array::from_slice(&[1_i32, 2, 3, 4, 5], &[1, 5]);
        let config = DFlashConfig {
            max_new_tokens: 12,
            temperature: 0.0,
            stop_tokens: vec![],
            speculative_tokens: None,
            ..Default::default()
        };
        let output = decoder.generate(&prompt, &config).unwrap();

        assert_eq!(output.tokens.len(), 5 + 12, "should produce max_new_tokens after prompt");
        assert_eq!(output.metrics.num_generated, 12);
        assert!(
            output.metrics.acceptance_lengths.iter().all(|&a| a >= 1),
            "every speculative step must accept at least the bonus token"
        );
        assert!(output.metrics.total_drafted >= 4, "at least one draft block");
    }

    #[test]
    #[serial]
    fn test_dflash_decoder_stops_at_stop_token() {
        // Force the first token of the output to act as a stop — after the
        // prefill we sample greedily, so a vocab full of identical rows
        // would pick token 0 every time. We instead inject a stop set that
        // is guaranteed to contain the first sampled token by running once
        // first to observe it, then running again with it as a stop.
        let target = tiny_target();
        let draft = tiny_draft();
        let mut decoder = DFlashDecoder::new(target, draft);

        let prompt = Array::from_slice(&[1_i32, 2, 3], &[1, 3]);

        let observe_config = DFlashConfig {
            max_new_tokens: 2,
            temperature: 0.0,
            stop_tokens: vec![],
            speculative_tokens: None,
            ..Default::default()
        };
        let observed = decoder.generate(&prompt, &observe_config).unwrap();
        let first_generated = observed.tokens[3];

        // Re-run with the first generated token set as a stop — the loop
        // should halt as soon as it is produced.
        let stop_config = DFlashConfig {
            max_new_tokens: 32,
            temperature: 0.0,
            stop_tokens: vec![first_generated],
            speculative_tokens: None,
            ..Default::default()
        };
        let stopped = decoder.generate(&prompt, &stop_config).unwrap();
        assert_eq!(
            *stopped.tokens.last().unwrap(),
            first_generated,
            "stop token must be the final emitted token"
        );
    }
}
