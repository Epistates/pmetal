//! NEFTune noise, end to end, on every architecture that can carry it.
//!
//! NEFTune (Jain et al., 2023) adds `U(-mag, mag)` to the embedding output
//! during training, with `mag = alpha / sqrt(seq_len * dims)`. Every way of
//! getting it wrong is silent: noise that never gets added, noise with the
//! wrong sign distribution, noise at the wrong scale, or noise in the wrong
//! dtype. A run finishes and reports the same loss either way, so each of those
//! gets its own assertion here.

use pmetal_bridge::compat::{Array, Dtype, nn};
use pmetal_core::LoraConfig;
use pmetal_lora::{AdaptedModel, TrainableModel};
use pmetal_models::dispatcher::DynamicModel;

/// One config per architecture family the dispatcher can put an alpha on.
const CASES: &[(&str, &str)] = &[
    (
        "llama",
        r#"{
            "model_type": "llama",
            "vocab_size": 128, "hidden_size": 64, "intermediate_size": 128,
            "num_hidden_layers": 2, "num_attention_heads": 4,
            "num_key_value_heads": 2, "max_position_embeddings": 256,
            "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
            "tie_word_embeddings": false
        }"#,
    ),
    (
        "qwen3",
        r#"{
            "model_type": "qwen3",
            "vocab_size": 128, "hidden_size": 64, "intermediate_size": 128,
            "num_hidden_layers": 2, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 16,
            "max_position_embeddings": 256,
            "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
            "tie_word_embeddings": false
        }"#,
    ),
    (
        // Gemma scales the embedding by sqrt(hidden_size) after the lookup, so
        // the noise is multiplied up with it. That is what TRL does too, since
        // its hook fires on the embedding module's output.
        "gemma",
        r#"{
            "model_type": "gemma",
            "vocab_size": 128, "hidden_size": 64, "intermediate_size": 128,
            "num_hidden_layers": 2, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 16,
            "max_position_embeddings": 256,
            "rms_norm_eps": 1e-6, "rope_theta": 10000.0
        }"#,
    ),
    (
        "cohere",
        r#"{
            "model_type": "cohere",
            "vocab_size": 128, "hidden_size": 64, "intermediate_size": 128,
            "num_hidden_layers": 2, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 16,
            "max_position_embeddings": 256,
            "layer_norm_eps": 1e-5, "rope_theta": 10000.0,
            "logit_scale": 0.0625
        }"#,
    ),
    (
        "phi3",
        r#"{
            "model_type": "phi3",
            "vocab_size": 128, "hidden_size": 64, "intermediate_size": 128,
            "num_hidden_layers": 2, "num_attention_heads": 4,
            "num_key_value_heads": 4, "max_position_embeddings": 256,
            "rms_norm_eps": 1e-5, "rope_theta": 10000.0,
            "hidden_act": "silu", "tie_word_embeddings": false
        }"#,
    ),
    (
        "gpt_oss",
        r#"{
            "model_type": "gpt_oss",
            "vocab_size": 128, "hidden_size": 64, "intermediate_size": 32,
            "num_hidden_layers": 2, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 16,
            "num_local_experts": 4, "num_experts_per_tok": 2,
            "max_position_embeddings": 256,
            "rms_norm_eps": 1e-5, "rope_theta": 10000.0,
            "sliding_window": 8, "tie_word_embeddings": false
        }"#,
    ),
];

const SEQ_LEN: i32 = 16;
const ALPHA: f32 = 5.0;

fn build(config_json: &str) -> AdaptedModel {
    let config = LoraConfig {
        r: 4,
        alpha: 8.0,
        target_modules: vec!["q_proj".into(), "v_proj".into()],
        ..Default::default()
    };
    let model = DynamicModel::from_config(config_json).expect("architecture should build");
    AdaptedModel::attach(model, config).expect("adapters should attach")
}

fn input_ids() -> Array {
    let tokens: Vec<i32> = (0..SEQ_LEN).map(|i| (i * 5 + 1) % 128).collect();
    Array::from_slice(&tokens, &[1, SEQ_LEN])
}

#[test]
fn noise_reaches_every_architecture_that_claims_it() {
    let ids = input_ids();
    for (name, config_json) in CASES {
        let mut model = build(config_json);

        let clean = TrainableModel::forward(&mut model, &ids, None).unwrap();
        let noised = model.forward_noised(&ids, None, ALPHA).unwrap();

        let difference = clean.subtract(&noised).abs().max(None).item_f32();
        assert!(
            difference > 0.0,
            "{name}: forward_noised produced exactly the regular forward, so the alpha \
             was accepted and dropped"
        );
        assert!(
            difference.is_finite(),
            "{name}: forward_noised produced a non-finite output"
        );
    }
}

#[test]
fn the_alpha_does_not_outlive_the_call() {
    // Eval and generation share this model. If `forward_noised` left the alpha
    // set, every later pass would be perturbed too.
    let ids = input_ids();
    let mut model = build(CASES[0].1);

    let before = TrainableModel::forward(&mut model, &ids, None).unwrap();
    let _ = model.forward_noised(&ids, None, ALPHA).unwrap();
    let after = TrainableModel::forward(&mut model, &ids, None).unwrap();

    let drift = before.subtract(&after).abs().max(None).item_f32();
    assert_eq!(
        drift, 0.0,
        "a plain forward changed after a noised one, so the alpha stayed set"
    );
}

#[test]
fn noise_is_zero_mean_and_scaled_by_alpha_over_sqrt_of_the_size() {
    // The scale is the whole content of the method. TRL computes
    // `mag = alpha / sqrt(seq_len * dims)` and draws `U(-mag, mag)`; an
    // implementation drawing `U(0, mag)` instead would still "add noise", still
    // train, and still be wrong, because it biases every dimension upward.
    let seq_len = 64;
    let dims = 128;
    let mut embedding = nn::Embedding::new(256, dims).unwrap();
    embedding.weight.value = Array::zeros_f32(&[256, dims]);

    let tokens: Vec<i32> = (0..seq_len).map(|i| i % 256).collect();
    let ids = Array::from_slice(&tokens, &[1, seq_len]);

    embedding.neftune_alpha = Some(ALPHA);
    let noise = embedding.forward(&ids);
    noise.eval();

    let expected_magnitude = ALPHA / ((seq_len * dims) as f32).sqrt();
    let observed_max = noise.abs().max(None).item_f32();
    assert!(
        observed_max <= expected_magnitude,
        "noise reached {observed_max}, above the {expected_magnitude} bound"
    );
    assert!(
        observed_max > expected_magnitude * 0.5,
        "noise peaked at {observed_max}, far below the {expected_magnitude} bound, \
         so the scale is wrong"
    );

    let mean = noise.mean(None).item_f32();
    let tolerance = expected_magnitude / ((seq_len * dims) as f32).sqrt() * 8.0;
    assert!(
        mean.abs() < tolerance,
        "noise mean {mean} is not centred on zero (bound {tolerance}), which is what \
         drawing from U(0, mag) instead of U(-mag, mag) looks like"
    );
}

#[test]
fn noise_does_not_change_the_dtype() {
    // Drawing the noise in f32 against a bf16 checkpoint promotes the sum, and
    // MLX carries that through every op downstream, so the rest of the model
    // silently runs in f32.
    let dims = 32;
    let mut embedding = nn::Embedding::new(64, dims).unwrap();
    embedding.weight.value = Array::zeros_f32(&[64, dims]).as_dtype(Dtype::Bfloat16.as_i32());
    embedding.neftune_alpha = Some(ALPHA);

    let ids = Array::from_slice(&[1i32, 2, 3, 4], &[1, 4]);
    let out = embedding.forward(&ids);
    out.eval();

    assert_eq!(
        out.dtype(),
        Dtype::Bfloat16,
        "NEFTune promoted a bf16 embedding, which spreads to the whole forward pass"
    );
}

#[test]
fn a_tied_lm_head_gets_no_noise() {
    // Models with tied embeddings run the LM head through `as_linear`. Noise
    // belongs on the input lookup, not on the output projection.
    let dims = 16;
    let mut embedding = nn::Embedding::new(32, dims).unwrap();
    embedding.neftune_alpha = Some(ALPHA);

    let hidden = Array::ones(&[1, 4, dims], Dtype::Float32.as_i32());
    let first = embedding.as_linear(&hidden);
    let second = embedding.as_linear(&hidden);

    let difference = first.subtract(&second).abs().max(None).item_f32();
    assert_eq!(difference, 0.0, "the tied LM head picked up NEFTune noise");
}

#[test]
fn an_unadapted_embedding_is_untouched() {
    // The default has to be off: inference shares this layer.
    let dims = 16;
    let mut embedding = nn::Embedding::new(32, dims).unwrap();
    assert!(embedding.neftune_alpha.is_none());

    let ids = Array::from_slice(&[1i32, 2, 3], &[1, 3]);
    let first = embedding.forward(&ids);
    let second = embedding.forward(&ids);
    let difference = first.subtract(&second).abs().max(None).item_f32();
    assert_eq!(difference, 0.0, "a fresh embedding added noise");

    // An alpha of zero is a disabled alpha, not a zero-width draw.
    embedding.neftune_alpha = Some(0.0);
    let third = embedding.forward(&ids);
    let difference = first.subtract(&third).abs().max(None).item_f32();
    assert_eq!(difference, 0.0, "alpha = 0 perturbed the embedding");
}

#[test]
fn the_architectures_without_a_token_embedding_say_so() {
    // BERT is encoder-only with its own embedding stack, and Flux has no token
    // embedding at all. Both must decline rather than silently no-op.
    let mut model = DynamicModel::from_config(CASES[0].1).unwrap();
    assert!(
        model.set_neftune_alpha(Some(ALPHA)),
        "llama should accept a NEFTune alpha"
    );
    assert!(model.token_embedding_mut().is_some());
    model.set_neftune_alpha(None);
    assert_eq!(
        model.token_embedding_mut().unwrap().neftune_alpha,
        None,
        "clearing the alpha did not take"
    );
}
