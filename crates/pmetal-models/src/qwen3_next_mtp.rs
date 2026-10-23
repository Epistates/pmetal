//! Qwen3Next bundled MTP speculative decoding.
//!
//! The decode loop is [`crate::mtp`]'s. This module supplies the predictor as
//! an [`MtpDrafter`] and the three targets it runs against: the dynamic model
//! (adapted or not), and the native bridge engine `pmetal infer` uses.

use std::path::Path;

use pmetal_cache::{
    kv_cache::{GdnVerifyInputs, KVCache, MambaCache, MambaSnapshot},
    speculative::SpecCapture,
};
use pmetal_runtime::compat::{
    Array, Exception,
    ops::{concatenate_axis, slice_axis},
};
use pmetal_runtime::decode::{last_token_hidden, last_token_logits, prompt_tokens_to_input};
use pmetal_runtime::qwen3_native::{self, GdnReplay, NativeCache, NativeWeights};

use crate::{
    architectures::{Qwen3NextConfig, Qwen3NextForCausalLM, Qwen3NextMtpForCausalLM},
    generation::{GenerationConfig, GenerationOutput},
    mtp::{MtpDrafter, MtpTarget, generate_mtp_streaming},
};

#[derive(Debug, Clone)]
pub struct Qwen3NextMtpConfig {
    pub num_draft_tokens: usize,
}

impl Default for Qwen3NextMtpConfig {
    fn default() -> Self {
        Self {
            // Three draft tokens per step (draft_max=3).
            num_draft_tokens: 3,
        }
    }
}

/// Check that `mtp` can draft for a target with `target_config`.
pub fn validate_qwen3_next_mtp_pair(
    target_config: &Qwen3NextConfig,
    mtp: &Qwen3NextMtpForCausalLM,
) -> Result<(), Exception> {
    if target_config.hidden_size != mtp.config.hidden_size {
        return Err(Exception::custom(format!(
            "Qwen MTP hidden_size ({}) must match target hidden_size ({})",
            mtp.config.hidden_size, target_config.hidden_size
        )));
    }
    if target_config.vocab_size != mtp.config.vocab_size {
        return Err(Exception::custom(format!(
            "Qwen MTP vocab_size ({}) must match target vocab_size ({})",
            mtp.config.vocab_size, target_config.vocab_size
        )));
    }
    if mtp.config.mtp_num_hidden_layers() == 0 {
        return Err(Exception::custom("Qwen MTP has no predictor layers"));
    }
    Ok(())
}

/// MTP decode of a dynamic Qwen3Next target (adapted or not).
#[allow(clippy::too_many_arguments)]
pub fn generate_qwen3_next_mtp_streaming<F>(
    target: &mut Qwen3NextForCausalLM,
    mtp: &mut Qwen3NextMtpForCausalLM,
    input_ids: &[u32],
    gen_config: GenerationConfig,
    target_cache: &mut KVCache,
    target_mamba_cache: &mut MambaCache,
    mtp_cache: &mut KVCache,
    mtp_config: Qwen3NextMtpConfig,
    on_token: F,
) -> Result<GenerationOutput, Exception>
where
    F: FnMut(u32) -> bool,
{
    validate_qwen3_next_mtp_pair(&target.config, mtp)?;
    let mut target = Qwen3NextMtpVerifier::new(target, target_cache, target_mamba_cache);
    let mut drafter = Qwen3NextMtpDrafter::new(mtp, mtp_cache);
    generate_mtp_streaming(
        &mut target,
        &mut drafter,
        input_ids,
        &gen_config,
        mtp_config.num_draft_tokens,
        on_token,
    )
}

/// The Qwen3Next MTP predictor as an [`MtpDrafter`]. It keeps its own KV
/// cache, one entry per `(token, hidden state)` pair: entry `i` pairs
/// `x[i + 1]` with the target's `h[i]`. A round's draft steps add entries
/// built on the predictor's own hidden states; the round's commit puts the
/// cache back as it was and adds the kept tokens with the target's.
pub struct Qwen3NextMtpDrafter<'a> {
    mtp: &'a mut Qwen3NextMtpForCausalLM,
    cache: &'a mut KVCache,
    before_round: Option<KVCache>,
}

impl<'a> Qwen3NextMtpDrafter<'a> {
    pub fn new(mtp: &'a mut Qwen3NextMtpForCausalLM, cache: &'a mut KVCache) -> Self {
        Self {
            mtp,
            cache,
            before_round: None,
        }
    }

    /// Run `tokens` paired with `hidden` (`[1, T, H]`) into the cache.
    fn append(
        &mut self,
        tokens: &[u32],
        hidden: &Array,
        step: usize,
    ) -> Result<(Array, Array), Exception> {
        let input = prompt_tokens_to_input(tokens);
        self.mtp
            .forward_logits(&input, hidden, None, Some(self.cache), step)
    }
}

impl<T: MtpTarget> MtpDrafter<T> for Qwen3NextMtpDrafter<'_> {
    fn prefill(&mut self, _: &mut T, prompt: &[u32], hidden: &Array) -> Result<(), Exception> {
        self.cache.reset();
        let n = prompt.len();
        if n > 1 {
            let head = slice_axis(hidden, 1, 0, n as i32 - 1);
            self.append(&prompt[1..], &head, 0)?;
        }
        Ok(())
    }

    fn step(
        &mut self,
        _: &mut T,
        token: u32,
        hidden: &Array,
        step: usize,
        _position: usize,
    ) -> Result<(Array, Array), Exception> {
        if step == 0 {
            // A clone shares the cached arrays; appending makes new ones.
            self.before_round = Some(self.cache.clone());
        }
        let (hidden, logits) = self.append(&[token], hidden, step)?;
        Ok((last_token_hidden(&hidden), last_token_logits(&logits)))
    }

    fn commit(
        &mut self,
        _: &mut T,
        pending: u32,
        seed: &Array,
        drafts: &[u32],
        accepted: usize,
        verify_hidden: &Array,
    ) -> Result<(), Exception> {
        if let Some(before) = self.before_round.take() {
            *self.cache = before;
        }
        // `pending` was predicted by `seed`, and kept draft `j` by the
        // target's hidden state at the block's position `j`.
        let mut tokens = Vec::with_capacity(accepted + 1);
        tokens.push(pending);
        tokens.extend_from_slice(&drafts[..accepted]);
        let hidden = if accepted == 0 {
            seed.clone()
        } else {
            concatenate_axis(
                &[seed, &slice_axis(verify_hidden, 1, 0, accepted as i32)],
                1,
            )
        };
        self.append(&tokens, &hidden, 0)?;
        Ok(())
    }
}

/// A dynamic Qwen3Next model as an [`MtpTarget`]. Its attention layers roll
/// back by their KV caches; its GDN layers are rewound from a snapshot taken
/// before the verify, replaying the kept tokens' recorded inputs.
pub struct Qwen3NextMtpVerifier<'a> {
    model: &'a mut Qwen3NextForCausalLM,
    cache: &'a mut KVCache,
    mamba: &'a mut MambaCache,
    capture: SpecCapture,
    snapshot: Vec<MambaSnapshot>,
    verified: usize,
}

impl<'a> Qwen3NextMtpVerifier<'a> {
    pub fn new(
        model: &'a mut Qwen3NextForCausalLM,
        cache: &'a mut KVCache,
        mamba: &'a mut MambaCache,
    ) -> Self {
        Self {
            model,
            cache,
            mamba,
            capture: SpecCapture::with_layers_and_embedding(Vec::new(), false),
            snapshot: Vec::new(),
            verified: 0,
        }
    }
}

impl MtpTarget for Qwen3NextMtpVerifier<'_> {
    fn reset(&mut self) -> Result<(), Exception> {
        self.cache.reset();
        self.mamba.reset();
        Ok(())
    }

    fn forward(&mut self, tokens: &[u32]) -> Result<(Array, Array), Exception> {
        self.capture.clear();
        self.model.forward_hidden_with_capture(
            &prompt_tokens_to_input(tokens),
            None,
            Some(self.cache),
            Some(self.mamba),
            &mut self.capture,
        )
    }

    fn verify(&mut self, tokens: &[u32]) -> Result<(Array, Array), Exception> {
        self.snapshot = self.mamba.snapshot();
        self.verified = tokens.len();
        self.forward(tokens)
    }

    fn keep(&mut self, kept: usize) -> Result<(), Exception> {
        let rejected = self.verified.saturating_sub(kept);
        if rejected > 0 {
            self.cache.rollback(rejected);
            let per_layer: Vec<Option<GdnVerifyInputs>> = (0..self.mamba.num_layers())
                .map(|layer| self.capture.gdn_inputs.remove(&layer))
                .collect();
            self.mamba
                .rewind_from_snapshots(&self.snapshot, &per_layer, kept)?;
        }
        Ok(())
    }
}

/// MTP decode of the native bridge Qwen engine, loaded from `model_dir`.
pub fn generate_qwen3_native_mtp_streaming<F>(
    model_dir: &Path,
    mtp: &mut Qwen3NextMtpForCausalLM,
    mtp_cache: &mut KVCache,
    input_ids: &[u32],
    gen_config: GenerationConfig,
    mtp_config: Qwen3NextMtpConfig,
    on_token: F,
) -> Result<GenerationOutput, Exception>
where
    F: FnMut(u32) -> bool,
{
    let mut target = NativeQwen3MtpVerifier::load(model_dir)?;
    if mtp.config.hidden_size != target.hidden_size {
        return Err(Exception::custom(format!(
            "Qwen MTP hidden_size ({}) must match target hidden_size ({})",
            mtp.config.hidden_size, target.hidden_size
        )));
    }
    let mut drafter = Qwen3NextMtpDrafter::new(mtp, mtp_cache);
    generate_mtp_streaming(
        &mut target,
        &mut drafter,
        input_ids,
        &gen_config,
        mtp_config.num_draft_tokens,
        on_token,
    )
}

/// The native bridge Qwen engine as an [`MtpTarget`]. Its attention layers
/// roll back by their cache offsets; on Qwen3.5 a verify records each GDN
/// layer's recurrence inputs, and a partial accept replays them over the
/// kept prefix ([`qwen3_native::rewind_verify`], as the DFlash target
/// does), rather than running the kept tokens through the model again.
pub struct NativeQwen3MtpVerifier {
    weights: NativeWeights,
    cache: NativeCache,
    /// The last verify's GDN recurrence inputs, one per GDN layer.
    gdn_replays: Vec<GdnReplay>,
    /// Tokens in the last verify.
    verified: usize,
    hidden_size: i32,
}

impl NativeQwen3MtpVerifier {
    /// Load a Qwen3 / Qwen3.5 checkpoint into the native engine.
    pub fn load(model_dir: &Path) -> Result<Self, Exception> {
        let config = qwen3_native::load_config(model_dir).map_err(Exception::from)?;
        let weights = qwen3_native::load_model(model_dir, &config).map_err(Exception::from)?;
        tracing::debug!(
            layers = config.num_hidden_layers,
            hidden = config.hidden_size,
            "native Qwen MTP verifier loaded"
        );
        Ok(Self {
            cache: NativeCache::new_empty(&weights),
            weights,
            gdn_replays: Vec::new(),
            verified: 0,
            hidden_size: config.hidden_size,
        })
    }

    /// Run `tokens`, recording what rewinding them takes when `rewindable`.
    fn run(&mut self, tokens: &[u32], rewindable: bool) -> (Array, Array) {
        self.verified = if rewindable { tokens.len() } else { 0 };
        self.gdn_replays.clear();
        let replays =
            (rewindable && !self.cache.gdn_caches.is_empty()).then_some(&mut self.gdn_replays);
        qwen3_native::forward_step_hidden_recording(
            &self.weights,
            &prompt_tokens_to_input(tokens),
            &mut self.cache,
            replays,
        )
    }
}

impl MtpTarget for NativeQwen3MtpVerifier {
    fn reset(&mut self) -> Result<(), Exception> {
        self.cache = NativeCache::new_empty(&self.weights);
        self.gdn_replays.clear();
        self.verified = 0;
        Ok(())
    }

    fn forward(&mut self, tokens: &[u32]) -> Result<(Array, Array), Exception> {
        Ok(self.run(tokens, false))
    }

    fn verify(&mut self, tokens: &[u32]) -> Result<(Array, Array), Exception> {
        // A one-token block is never cut back, and recording would take it
        // off the GDN layers' compiled single-token step.
        Ok(self.run(tokens, tokens.len() > 1))
    }

    fn keep(&mut self, kept: usize) -> Result<(), Exception> {
        let verified = std::mem::take(&mut self.verified);
        if kept < verified {
            qwen3_native::rewind_verify(
                &mut self.cache,
                &self.gdn_replays,
                verified as i32,
                kept as i32,
            )
            .map_err(|e| Exception::custom(format!("native Qwen MTP: {e}")))?;
        }
        self.gdn_replays.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_qwen_mtp_rejects_mismatched_hidden_size() {
        let mut target_config = crate::architectures::Qwen3NextConfig {
            hidden_size: 4,
            intermediate_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            num_key_value_heads: Some(1),
            head_dim: Some(4),
            // Two rotary channels: the default quarter of a 4-wide head is
            // one, which no rotation pairs.
            partial_rotary_factor: 0.5,
            vocab_size: 16,
            num_experts: 0,
            mtp_num_hidden_layers: Some(1),
            ..Default::default()
        };
        let target =
            crate::architectures::Qwen3NextForCausalLM::new(target_config.clone()).unwrap();
        target_config.hidden_size = 8;
        let mtp = crate::architectures::Qwen3NextMtpForCausalLM::new(target_config).unwrap();
        assert!(validate_qwen3_next_mtp_pair(&target.config, &mtp).is_err());
    }

    #[test]
    fn validate_qwen_mtp_accepts_multiple_predictor_layers() {
        let config = crate::architectures::Qwen3NextConfig {
            hidden_size: 4,
            intermediate_size: 8,
            num_hidden_layers: 2,
            num_attention_heads: 1,
            num_key_value_heads: Some(1),
            head_dim: Some(4),
            // Two rotary channels: the default quarter of a 4-wide head is
            // one, which no rotation pairs.
            partial_rotary_factor: 0.5,
            vocab_size: 16,
            num_experts: 0,
            mtp_num_hidden_layers: Some(2),
            ..Default::default()
        };
        let target = crate::architectures::Qwen3NextForCausalLM::new(config.clone()).unwrap();
        let mtp = crate::architectures::Qwen3NextMtpForCausalLM::new(config).unwrap();

        validate_qwen3_next_mtp_pair(&target.config, &mtp).unwrap();
        assert_eq!(mtp.create_cache(16).config().num_layers, 2);
    }

    /// The tokens of a sampled MTP decode on a tiny random target and
    /// predictor. With `disturb`, every emitted token also draws from the
    /// thread's MLX key, as another stream sharing the thread would.
    fn sampled_mtp_tokens(seed: Option<u64>, disturb: bool) -> Vec<u32> {
        use pmetal_cache::kv_cache::KVCacheConfig;
        use pmetal_runtime::compat::{Dtype, random};

        let config = crate::architectures::Qwen3NextConfig {
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: Some(1),
            head_dim: Some(8),
            partial_rotary_factor: 0.5,
            vocab_size: 64,
            num_experts: 0,
            mtp_num_hidden_layers: Some(1),
            ..Default::default()
        };
        random::seed(5);
        let mut target = Qwen3NextForCausalLM::new(config.clone()).unwrap();
        let mut mtp = Qwen3NextMtpForCausalLM::new(config.clone()).unwrap();
        let mut target_cache = KVCache::new(KVCacheConfig::new(2, 128, 1, 8));
        let mut mamba = MambaCache::new(2);
        let mut mtp_cache = mtp.create_cache(128);
        let gen_config = GenerationConfig {
            max_new_tokens: 32,
            temperature: 1.0,
            do_sample: true,
            seed,
            ..Default::default()
        };
        let out = generate_qwen3_next_mtp_streaming(
            &mut target,
            &mut mtp,
            &[1, 2, 3, 4],
            gen_config,
            &mut target_cache,
            &mut mamba,
            &mut mtp_cache,
            Qwen3NextMtpConfig {
                num_draft_tokens: 3,
            },
            |_| {
                if disturb {
                    random::uniform(&[8], Dtype::Float32).eval();
                }
                true
            },
        )
        .expect("generate");
        pmetal_runtime::check_unobserved_error().expect("no bridge op failed");
        out.token_ids
    }

    /// A seed fixes the tokens of a sampled MTP decode, drafts, acceptance
    /// tests, corrections and bonus tokens alike, whatever else draws from
    /// the thread's key meanwhile and whichever thread runs it.
    #[test]
    #[serial_test::serial]
    fn a_seeded_mtp_decode_draws_the_same_tokens_whatever_else_draws() {
        let first = sampled_mtp_tokens(Some(7), false);
        assert_eq!(first.len(), 4 + 32);
        assert_eq!(
            sampled_mtp_tokens(Some(7), true),
            first,
            "draws on the thread's key moved the seeded tokens"
        );
        let elsewhere = std::thread::spawn(|| sampled_mtp_tokens(Some(7), true))
            .join()
            .expect("thread");
        assert_eq!(elsewhere, first, "another thread drew other tokens");
        assert_ne!(
            sampled_mtp_tokens(Some(8), false),
            first,
            "the seed is unused"
        );
    }
}
