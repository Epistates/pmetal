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

use pmetal_bridge::compat::{Array, Dtype, ops, random};
use pmetal_bridge::training::per_token_cross_entropy_loss;

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
