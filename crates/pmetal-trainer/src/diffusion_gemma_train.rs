//! DiffusionGemma block-diffusion training objective primitives.
//!
//! DiffusionGemma is a *block-autoregressive discrete-diffusion* model: a causal
//! encoder commits a prompt to a read-only KV cache, and a bidirectional decoder
//! denoises a fixed-length canvas conditioned on that cache. transformers ships
//! the modeling + generation code but **no training loop** (the top-level
//! `forward` returns logits, never a loss), so the objective here is derived
//! from the architecture, mirroring how the generation sampler was.
//!
//! Two differences from the LLaDA masked-diffusion path in [`crate::diffusion`]:
//!
//! 1. **Corruption kernel is uniform-categorical, not masking.** A corrupted
//!    canvas position is replaced by a *uniformly random vocabulary token*
//!    (there is no `[MASK]`/absorbing state), matching the generation loop's
//!    uniform noise init + renoise.
//! 2. **The forward is `DiffusionGemmaForBlockDiffusion::forward_train`** — an
//!    encoder→decoder→tied-head pass over a `(context, noised canvas)` pair,
//!    not a single causal `forward(ids)`.
//!
//! The objective is the decoder **denoising cross-entropy** (predict the clean
//! canvas from its corrupted version), optionally restricted to corrupted
//! positions, plus an optional **encoder autoregressive** next-token loss (the
//! encoder is causal; transformers exposes its hidden states "to compute an
//! autoregressive loss on the encoder during training"). The full training step
//! (t-schedule sampling, self-conditioning dropout, optimiser) lives in the
//! training loop that consumes these primitives.

use std::collections::HashMap;

use pmetal_bridge::compat::nn::value_and_grad_explicit;
use pmetal_bridge::compat::{Array, Dtype, Exception, eval, ops, random};
use pmetal_bridge::training::per_token_cross_entropy_loss;
use pmetal_models::architectures::diffusion_gemma::DiffusionGemmaForBlockDiffusion;
use rand::{RngExt as _, SeedableRng, rngs::StdRng};

/// Corrupt a clean canvas with the uniform-categorical forward kernel.
///
/// Each position is kept with probability `keep_prob` (= `alpha_t`) and
/// otherwise replaced by a uniformly random token in `[0, vocab)`. Returns
/// `(x_t, corrupted_mask)` where `x_t` is `[B, S]` int32 and `corrupted_mask`
/// is `[B, S]` f32 (1.0 at replaced positions, 0.0 where the original was kept).
///
/// The bridge RNG is process-global and stateless; seed the trajectory with
/// [`pmetal_bridge::compat::random::seed`] before calling (the `seed` argument
/// is accepted for call-site clarity and forward compatibility).
pub fn uniform_categorical_noise(
    x0: &Array,
    keep_prob: f32,
    vocab: i32,
    seed: Option<u64>,
) -> (Array, Array) {
    let _ = seed; // bridge RNG is seeded globally via random::seed
    let shape = x0.shape();
    let uniform = random::uniform_range(0.0_f32, 1.0_f32, shape, Dtype::Float32);
    // keep where u < keep_prob; corrupt otherwise.
    let keep_bool = uniform.lt(&Array::from_f32(keep_prob));
    let random_tokens = random::randint(0, vocab, shape, Dtype::Int32);
    let x0_i32 = x0.as_dtype(Dtype::Int32.as_i32());
    let x_t = ops::r#where(&keep_bool, &x0_i32, &random_tokens);
    let corrupted_mask = Array::from_f32(1.0).subtract(&keep_bool.as_type::<f32>());
    (x_t, corrupted_mask)
}

/// Decoder denoising cross-entropy: predict the clean canvas `targets` from the
/// decoder `logits` `[B, S, V]`.
///
/// When `corrupted_mask` is `Some`, the loss is averaged over corrupted
/// positions only (the standard denoising ELBO term counts the reconstruction
/// of noised tokens); when `None`, it is the mean over all canvas positions.
/// `ignore_index` positions in `targets` never contribute.
pub fn diffusion_denoising_loss(
    logits: &Array,
    targets: &Array,
    corrupted_mask: Option<&Array>,
    ignore_index: i32,
) -> Array {
    let v = logits.dim(logits.ndim() - 1);
    let n = targets.dim(0) * targets.dim(1);
    let per_token = per_token_cross_entropy_loss(
        &logits.reshape(&[n, v]),
        &targets.reshape(&[n]),
        ignore_index,
    );

    match corrupted_mask {
        Some(mask) => {
            let mflat = mask.reshape(&[n]).as_type::<f32>();
            let weighted = per_token.multiply(&mflat);
            // Guard the empty-corruption case (all kept) against divide-by-zero.
            let denom = mflat.sum(None).maximum(&Array::from_f32(1.0));
            weighted.sum(None).divide(&denom)
        }
        None => per_token.mean(None),
    }
}

/// Encoder autoregressive (next-token) cross-entropy over the prompt.
///
/// `encoder_logits` `[B, S, V]` are the tied-head logits of the causal encoder;
/// this shifts by one (`logits[:, :-1]` predict `input_ids[:, 1:]`) and returns
/// the mean CE. Optional auxiliary term — the primary objective is the decoder
/// denoising loss.
pub fn encoder_ar_loss(encoder_logits: &Array, input_ids: &Array, ignore_index: i32) -> Array {
    let b = encoder_logits.dim(0);
    let s = encoder_logits.dim(1);
    let v = encoder_logits.dim(2);
    debug_assert!(s >= 2, "encoder AR loss needs at least 2 positions");
    let shifted = ops::slice_axis(encoder_logits, 1, 0, s - 1);
    let labels = ops::slice_axis(input_ids, 1, 1, s);
    let n = b * (s - 1);
    per_token_cross_entropy_loss(
        &shifted.reshape(&[n, v]),
        &labels.reshape(&[n]),
        ignore_index,
    )
    .mean(None)
}

/// Hyperparameters for the DiffusionGemma LoRA training loop.
///
/// The optimiser is decoupled-weight-decay AdamW applied *only* to the bake-in
/// LoRA adapters (`Gemma4Attention::lora`), which live outside the model's
/// `ModuleParameters` tree by design. Gradients are obtained with
/// [`value_and_grad_explicit`] over the explicit adapter array list rather than
/// the module tree, so the frozen base weights never receive gradients.
#[derive(Debug, Clone)]
pub struct DiffusionGemmaTrainConfig {
    /// AdamW learning rate.
    pub learning_rate: f32,
    /// AdamW β₁ (first-moment decay).
    pub beta1: f32,
    /// AdamW β₂ (second-moment decay).
    pub beta2: f32,
    /// AdamW ε (denominator floor).
    pub eps: f32,
    /// Decoupled weight decay (0.0 = none; LoRA is typically trained without it).
    pub weight_decay: f32,
    /// Lowest noise level sampled per step; `t ~ U(min_noise_level, 1]`.
    pub min_noise_level: f32,
    /// Label id excluded from the denoising cross-entropy.
    pub ignore_index: i32,
    /// Restrict the denoising loss to corrupted canvas positions (standard ELBO
    /// reconstruction term). When `false`, average over the whole canvas.
    pub corrupted_only: bool,
    /// Global gradient-norm clip threshold (0.0 disables clipping).
    pub max_grad_norm: f32,
    /// Seed for the per-step noise-level sampler.
    pub seed: u64,
}

impl Default for DiffusionGemmaTrainConfig {
    fn default() -> Self {
        Self {
            learning_rate: 2e-4,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            min_noise_level: 1e-3,
            ignore_index: -100,
            corrupted_only: true,
            max_grad_norm: 1.0,
            seed: 42,
        }
    }
}

/// Statistics for a single DiffusionGemma LoRA training step.
#[derive(Debug, Clone)]
pub struct DiffusionGemmaStepStats {
    /// 1-based optimiser step counter.
    pub step: u64,
    /// Denoising loss at the parameters used to compute the gradient.
    pub loss: f32,
    /// Pre-clip global gradient norm (`None` when clipping is disabled).
    pub grad_norm: Option<f32>,
}

/// DiffusionGemma block-diffusion LoRA trainer.
///
/// Owns the AdamW moment buffers (keyed by adapter parameter name) and the
/// noise-level RNG. Call [`Self::train_step`] with a clean canvas to run the
/// full stochastic objective (sample `t`, corrupt, denoise, update), or
/// [`Self::train_on_batch`] with a caller-supplied corruption for a
/// deterministic step (reproducible runs, curriculum, overfit checks).
///
/// The model must have adapters attached (`model.attach_lora(..)`) before the
/// first step; otherwise there are no trainable parameters and the step errors.
pub struct DiffusionGemmaTrainer {
    config: DiffusionGemmaTrainConfig,
    vocab: i32,
    /// AdamW first moments, keyed by adapter parameter name.
    m: HashMap<String, Array>,
    /// AdamW second moments, keyed by adapter parameter name.
    v: HashMap<String, Array>,
    /// Completed optimiser steps (drives bias correction).
    step: u64,
    /// Noise-level sampler.
    rng: StdRng,
}

impl DiffusionGemmaTrainer {
    /// Create a trainer for a model whose vocabulary size is `vocab`.
    pub fn new(config: DiffusionGemmaTrainConfig, vocab: i32) -> Self {
        let rng = StdRng::seed_from_u64(config.seed);
        Self {
            config,
            vocab,
            m: HashMap::new(),
            v: HashMap::new(),
            step: 0,
            rng,
        }
    }

    /// Completed optimiser steps.
    pub fn step(&self) -> u64 {
        self.step
    }

    /// Full stochastic training step: sample a noise level `t ~ U(min, 1]`,
    /// corrupt `canvas_x0` with the uniform-categorical kernel, denoise it
    /// conditioned on `context_ids`, and take one AdamW step on the adapters.
    pub fn train_step(
        &mut self,
        model: &mut DiffusionGemmaForBlockDiffusion,
        context_ids: &Array,
        canvas_x0: &Array,
    ) -> Result<DiffusionGemmaStepStats, Exception> {
        let t: f32 = self.rng.random_range(self.config.min_noise_level..=1.0);
        let keep_prob = 1.0 - t;
        let (mut x_t, mut mask) = uniform_categorical_noise(canvas_x0, keep_prob, self.vocab, None);
        // Realise the corruption before it feeds the differentiable forward.
        x_t.eval();
        mask.eval();
        let corrupted_mask = if self.config.corrupted_only {
            Some(&mask)
        } else {
            None
        };
        self.train_on_batch(model, context_ids, &x_t, canvas_x0, corrupted_mask)
    }

    /// Deterministic training step over a caller-supplied corruption.
    ///
    /// Computes the denoising loss of `noised_canvas` against the clean
    /// `targets` (both `[B, canvas]` int ids), differentiates it w.r.t. the
    /// attached LoRA adapters, clips the global gradient norm, and applies one
    /// decoupled-AdamW update. `corrupted_mask` (`Some` = ELBO reconstruction
    /// over corrupted positions only) mirrors [`diffusion_denoising_loss`].
    pub fn train_on_batch(
        &mut self,
        model: &mut DiffusionGemmaForBlockDiffusion,
        context_ids: &Array,
        noised_canvas: &Array,
        targets: &Array,
        corrupted_mask: Option<&Array>,
    ) -> Result<DiffusionGemmaStepStats, Exception> {
        // Snapshot adapter parameters in a stable order. Cloning detaches the
        // owned array *handles* (same underlying nodes) so the model borrow ends
        // before the differentiable closure re-borrows it mutably.
        let (keys, param_arrays): (Vec<String>, Vec<Array>) = model
            .lora_parameters()
            .into_iter()
            .map(|(k, a)| (k, a.clone()))
            .unzip();
        if keys.is_empty() {
            return Err(Exception::custom(
                "DiffusionGemmaTrainer: no LoRA adapters attached (call model.attach_lora first)",
            ));
        }
        let n_params = keys.len();
        let ignore_index = self.config.ignore_index;

        // Loss closure: `all[..n_params]` are the traced adapter leaves. Inject
        // them into the model's adapter slots so `forward_train`'s LoRA deltas
        // are computed from the differentiated arrays, then denoise + CE.
        let loss_fn = |all: &[Array]| -> Array {
            for (i, (_, slot)) in model.lora_parameters_mut().into_iter().enumerate() {
                *slot = all[i].clone();
            }
            match model.forward_train(context_ids, noised_canvas, None) {
                Ok(logits) => {
                    diffusion_denoising_loss(&logits, targets, corrupted_mask, ignore_index)
                }
                Err(_) => Array::from_f32(f32::NAN),
            }
        };
        let (mut loss, grads) = value_and_grad_explicit(loss_fn, &param_arrays, &[])?;
        // Model borrow released here; loss/grads reference the traced leaves.
        loss.eval();
        let loss_val = loss.item_f32();

        // Global-norm gradient clipping.
        let (grads, grad_norm) = if self.config.max_grad_norm > 0.0 {
            let mut norm_sq = Array::from_f32(0.0);
            for g in &grads {
                norm_sq = norm_sq.add(&g.multiply(g).sum(None));
            }
            let mut norm = ops::sqrt(&norm_sq);
            norm.eval();
            let total = norm.item_f32();
            let max_norm = self.config.max_grad_norm;
            if total > max_norm {
                let scale = Array::from_f32(max_norm / (total + 1e-6));
                let clipped = grads.iter().map(|g| g.multiply(&scale)).collect();
                (clipped, Some(total))
            } else {
                (grads, Some(total))
            }
        } else {
            (grads, None)
        };

        // AdamW update on the adapter arrays, then write the new values back.
        self.step += 1;
        let new_params = self.adamw_step(&keys, &param_arrays, &grads)?;
        for ((_, slot), np) in model.lora_parameters_mut().into_iter().zip(&new_params) {
            *slot = np.clone();
        }
        eval(model.lora_parameters().into_iter().map(|(_, a)| a))?;

        Ok(DiffusionGemmaStepStats {
            step: self.step,
            loss: loss_val,
            grad_norm,
        })
    }

    /// One decoupled-weight-decay AdamW update. Reads/writes the per-key moment
    /// buffers and returns the new parameter values (already evaluated).
    fn adamw_step(
        &mut self,
        keys: &[String],
        params: &[Array],
        grads: &[Array],
    ) -> Result<Vec<Array>, Exception> {
        let (lr, b1, b2, eps, wd) = (
            self.config.learning_rate,
            self.config.beta1,
            self.config.beta2,
            self.config.eps,
            self.config.weight_decay,
        );
        let t = self.step as f32;
        let bc1 = 1.0 - b1.powf(t);
        let bc2 = 1.0 - b2.powf(t);
        let sc = |x: f32| Array::from_f32(x);
        let zeros_like = |a: &Array| a.multiply(&sc(0.0));

        let mut new_params = Vec::with_capacity(keys.len());
        let mut new_m = Vec::with_capacity(keys.len());
        let mut new_v = Vec::with_capacity(keys.len());
        for i in 0..keys.len() {
            let g = &grads[i];
            let m_prev = self
                .m
                .get(&keys[i])
                .cloned()
                .unwrap_or_else(|| zeros_like(g));
            let v_prev = self
                .v
                .get(&keys[i])
                .cloned()
                .unwrap_or_else(|| zeros_like(g));

            let m_new = m_prev.multiply(&sc(b1)).add(&g.multiply(&sc(1.0 - b1)));
            let v_new = v_prev
                .multiply(&sc(b2))
                .add(&g.multiply(g).multiply(&sc(1.0 - b2)));

            let m_hat = m_new.divide(&sc(bc1));
            let v_hat = v_new.divide(&sc(bc2));
            let denom = ops::sqrt(&v_hat).add(&sc(eps));
            let update = m_hat.divide(&denom);

            // Decoupled weight decay: p ← p·(1 − lr·wd) − lr·update.
            let new_p = params[i]
                .multiply(&sc(1.0 - lr * wd))
                .subtract(&update.multiply(&sc(lr)));

            new_params.push(new_p);
            new_m.push(m_new);
            new_v.push(v_new);
        }

        // Evaluate the whole update in one pass so the graph does not grow
        // across steps, then persist the moments.
        eval(new_params.iter().chain(new_m.iter()).chain(new_v.iter()))?;
        for (k, (m, v)) in keys.iter().zip(new_m.into_iter().zip(new_v)) {
            self.m.insert(k.clone(), m);
            self.v.insert(k.clone(), v);
        }
        Ok(new_params)
    }
}

/// Save the model's attached LoRA adapters to a safetensors file, keyed by the
/// `{encoder|decoder}.layers.{i}.self_attn.{proj}.lora_{a,b}` namespace.
pub fn save_lora_adapters(
    model: &DiffusionGemmaForBlockDiffusion,
    path: impl AsRef<std::path::Path>,
) -> Result<(), Exception> {
    let map: HashMap<String, Array> = model
        .lora_parameters()
        .into_iter()
        .map(|(k, a)| (k, a.clone()))
        .collect();
    pmetal_lora::save_safetensors_map(path, &map).map_err(|e| Exception::custom(e.to_string()))
}

/// Load LoRA adapter weights from a safetensors file into the model's attached
/// adapters (matching by name). Adapters must already be attached; missing keys
/// are left at their current value.
pub fn load_lora_adapters(
    model: &mut DiffusionGemmaForBlockDiffusion,
    path: impl AsRef<std::path::Path>,
) -> Result<(), Exception> {
    let loaded =
        pmetal_lora::load_safetensors_map(path).map_err(|e| Exception::custom(e.to_string()))?;
    for (name, slot) in model.lora_parameters_mut() {
        if let Some(a) = loaded.get(&name) {
            *slot = a.clone();
        }
    }
    eval(model.lora_parameters().into_iter().map(|(_, a)| a))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    fn item(a: &Array) -> f32 {
        let mut a = a.clone();
        a.to_f32_vec(1).expect("scalar vec")[0]
    }

    #[test]
    #[serial]
    fn noise_rate_matches_keep_prob() {
        random::seed(7);
        let vocab = 32;
        let x0 = random::randint(0, vocab, &[4, 64], Dtype::Int32);

        // keep_prob = 1.0 -> nothing corrupted, x_t == x0.
        let (x_t, mask) = uniform_categorical_noise(&x0, 1.0, vocab, None);
        assert_eq!(
            item(&mask.sum(None)),
            0.0,
            "keep_prob=1 must corrupt nothing"
        );
        let same = x_t.as_type::<f32>().subtract(&x0.as_type::<f32>());
        assert_eq!(item(&same.abs().sum(None)), 0.0, "x_t must equal x0");

        // keep_prob = 0.0 -> everything corrupted.
        let (_x_t0, mask0) = uniform_categorical_noise(&x0, 0.0, vocab, None);
        assert_eq!(
            item(&mask0.sum(None)),
            4.0 * 64.0,
            "keep_prob=0 corrupts all"
        );

        // keep_prob = 0.5 -> ~half corrupted (256 positions, generous band).
        let (_x_th, maskh) = uniform_categorical_noise(&x0, 0.5, vocab, None);
        let corrupted = item(&maskh.sum(None));
        assert!(
            (96.0..160.0).contains(&corrupted),
            "keep_prob=0.5 corrupted {corrupted}/256, expected ~128"
        );
    }

    #[test]
    #[serial]
    fn denoising_loss_zero_for_perfect_logits() {
        let targets = Array::from_slice(&[0i32, 1, 2, 3], &[1, 4]);
        // One-hot-ish logits: huge at the target class, ~0 elsewhere (V=4).
        let mut data = vec![0.0f32; 4 * 4];
        for (pos, &t) in [0, 1, 2, 3].iter().enumerate() {
            data[pos * 4 + t as usize] = 30.0;
        }
        let logits = Array::from_slice(&data, &[1, 4, 4]);
        let loss = diffusion_denoising_loss(&logits, &targets, None, -100);
        assert!(item(&loss) < 1e-3, "perfect logits should give ~0 loss");
    }

    #[test]
    #[serial]
    fn denoising_loss_uniform_is_ln_vocab() {
        let targets = Array::from_slice(&[0i32, 1, 2, 3], &[1, 4]);
        let logits = Array::from_slice(&vec![0.0f32; 4 * 4], &[1, 4, 4]); // uniform over V=4
        let loss = diffusion_denoising_loss(&logits, &targets, None, -100);
        let expected = (4.0f32).ln();
        assert!(
            (item(&loss) - expected).abs() < 1e-3,
            "uniform logits loss {} != ln(4) {expected}",
            item(&loss)
        );
    }

    use pmetal_core::LoraConfig;
    use pmetal_models::architectures::diffusion_gemma::DiffusionGemmaTextConfig;

    /// Tiny DiffusionGemma with q/k/v/o LoRA attached — mirrors the model-crate
    /// `tiny_config` so the trainer exercises the real architecture.
    fn tiny_model_with_lora(moe_intermediate: i32) -> (DiffusionGemmaForBlockDiffusion, i32, i32) {
        let cfg = DiffusionGemmaTextConfig {
            vocab_size: 64,
            hidden_size: 32,
            intermediate_size: 48,
            num_hidden_layers: 3,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 8,
            global_head_dim: 16,
            num_global_key_value_heads: Some(1),
            sliding_window: 8,
            sliding_window_pattern: 3,
            num_experts: 4,
            top_k_experts: 2,
            moe_intermediate_size: moe_intermediate,
            canvas_length: 8,
            ..Default::default()
        };
        let vocab = cfg.vocab_size;
        let canvas = cfg.canvas_length;
        let mut model = DiffusionGemmaForBlockDiffusion::new(cfg).unwrap();
        let lora_cfg = LoraConfig {
            r: 4,
            alpha: 8.0,
            target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            ..Default::default()
        };
        model.attach_lora(&lora_cfg).unwrap();
        (model, vocab, canvas)
    }

    fn logits_vec(
        model: &mut DiffusionGemmaForBlockDiffusion,
        ctx: &Array,
        canvas: &Array,
    ) -> Vec<f32> {
        let mut l = model.forward_train(ctx, canvas, None).unwrap();
        let n = (l.dim(1) * l.dim(2)) as usize;
        l.to_f32_vec(n).unwrap()
    }

    #[test]
    #[serial]
    fn train_on_batch_reduces_overfit_loss() {
        let (mut model, vocab, canvas) = tiny_model_with_lora(16);

        let ctx_ids: Vec<i32> = (0..5).map(|i| (i * 3 + 1) % vocab).collect();
        let context = Array::from_slice(&ctx_ids, &[1, 5]);
        // Clean targets and a fixed corruption of a few canvas positions.
        let clean: Vec<i32> = (0..canvas).map(|i| (i * 7 + 2) % vocab).collect();
        let targets = Array::from_slice(&clean, &[1, canvas]);
        let mut noised = clean.clone();
        for &p in &[1usize, 3, 6] {
            noised[p] = (noised[p] + 17) % vocab;
        }
        let x_t = Array::from_slice(&noised, &[1, canvas]);

        let config = DiffusionGemmaTrainConfig {
            learning_rate: 3e-3,
            corrupted_only: false, // full-canvas loss ⇒ deterministic, monotone signal
            ..Default::default()
        };
        let mut trainer = DiffusionGemmaTrainer::new(config, vocab);

        let mut first = f32::NAN;
        let mut last = f32::NAN;
        for i in 0..60 {
            let stats = trainer
                .train_on_batch(&mut model, &context, &x_t, &targets, None)
                .unwrap();
            assert!(stats.loss.is_finite(), "step {i} produced non-finite loss");
            if i == 0 {
                first = stats.loss;
            }
            last = stats.loss;
        }
        assert!(
            last < first,
            "overfit loss did not decrease: first={first}, last={last}"
        );
        // Expect a clear reduction, not just numerical drift.
        assert!(
            first - last > 0.05,
            "overfit loss barely moved: first={first}, last={last}"
        );
    }

    #[test]
    #[serial]
    fn trains_lora_on_quantized_base() {
        // QLoRA: 4-bit frozen base (attention + MoE experts) + f32 LoRA. The
        // trainer's gradients flow only to the adapters, so the loss must still
        // decrease even though the base is quantized. moe_intermediate = 32 to
        // satisfy the minimum quantization group size.
        let (mut model, vocab, canvas) = tiny_model_with_lora(32);
        model.quantize_base(32, 4, true).unwrap();

        let ctx_ids: Vec<i32> = (0..5).map(|i| (i * 3 + 1) % vocab).collect();
        let context = Array::from_slice(&ctx_ids, &[1, 5]);
        let clean: Vec<i32> = (0..canvas).map(|i| (i * 7 + 2) % vocab).collect();
        let targets = Array::from_slice(&clean, &[1, canvas]);
        let mut noised = clean.clone();
        for &p in &[1usize, 3, 6] {
            noised[p] = (noised[p] + 17) % vocab;
        }
        let x_t = Array::from_slice(&noised, &[1, canvas]);

        let config = DiffusionGemmaTrainConfig {
            learning_rate: 3e-3,
            corrupted_only: false,
            ..Default::default()
        };
        let mut trainer = DiffusionGemmaTrainer::new(config, vocab);

        let mut first = f32::NAN;
        let mut last = f32::NAN;
        for i in 0..60 {
            let stats = trainer
                .train_on_batch(&mut model, &context, &x_t, &targets, None)
                .unwrap();
            assert!(stats.loss.is_finite(), "step {i} produced non-finite loss");
            if i == 0 {
                first = stats.loss;
            }
            last = stats.loss;
        }
        assert!(
            last < first && first - last > 0.02,
            "LoRA-on-quantized-base did not train: first={first}, last={last}"
        );
    }

    #[test]
    #[serial]
    fn lora_checkpoint_save_load_roundtrip() {
        let (mut model, vocab, canvas) = tiny_model_with_lora(16);

        let ctx_ids: Vec<i32> = (0..5).map(|i| (i * 3 + 1) % vocab).collect();
        let context = Array::from_slice(&ctx_ids, &[1, 5]);
        let clean: Vec<i32> = (0..canvas).map(|i| (i * 7 + 2) % vocab).collect();
        let targets = Array::from_slice(&clean, &[1, canvas]);
        let mut noised = clean.clone();
        noised[2] = (noised[2] + 9) % vocab;
        let x_t = Array::from_slice(&noised, &[1, canvas]);

        // Train a few steps so the adapters are meaningfully non-zero.
        let mut trainer = DiffusionGemmaTrainer::new(
            DiffusionGemmaTrainConfig {
                learning_rate: 5e-3,
                corrupted_only: false,
                ..Default::default()
            },
            vocab,
        );
        for _ in 0..8 {
            trainer
                .train_on_batch(&mut model, &context, &x_t, &targets, None)
                .unwrap();
        }
        let trained = logits_vec(&mut model, &context, &x_t);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dg_lora.safetensors");
        save_lora_adapters(&model, &path).unwrap();

        // Zero the live adapters so their effect is removed, then confirm the
        // forward actually changed (the mutation took hold).
        for (_, slot) in model.lora_parameters_mut() {
            *slot = slot.multiply(&Array::from_f32(0.0));
        }
        eval(model.lora_parameters().into_iter().map(|(_, a)| a)).unwrap();
        let zeroed = logits_vec(&mut model, &context, &x_t);
        let mutated_diff = trained
            .iter()
            .zip(&zeroed)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            mutated_diff > 1e-5,
            "zeroing adapters should change the forward (diff={mutated_diff})"
        );

        // Restore from the checkpoint and confirm the trained forward returns.
        load_lora_adapters(&mut model, &path).unwrap();
        let restored = logits_vec(&mut model, &context, &x_t);
        let restore_diff = trained
            .iter()
            .zip(&restored)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            restore_diff < 1e-6,
            "checkpoint did not restore adapters exactly (diff={restore_diff})"
        );
    }

    #[test]
    #[serial]
    fn corrupted_mask_restricts_loss_to_corrupted_positions() {
        // Position 0 perfect, position 1 wrong; mask counts only position 1.
        let targets = Array::from_slice(&[0i32, 1], &[1, 2]);
        let logits = Array::from_slice(
            &[
                30.0f32, 0.0, 0.0, 0.0, // pos 0: perfect for class 0
                0.0, 0.0, 0.0, 0.0, // pos 1: uniform (loss ln 4)
            ],
            &[1, 2, 4],
        );
        let mask = Array::from_slice(&[0.0f32, 1.0], &[1, 2]);
        let loss = diffusion_denoising_loss(&logits, &targets, Some(&mask), -100);
        assert!(
            (item(&loss) - (4.0f32).ln()).abs() < 1e-3,
            "masked loss should equal the corrupted position's ln(4), got {}",
            item(&loss)
        );
    }
}
