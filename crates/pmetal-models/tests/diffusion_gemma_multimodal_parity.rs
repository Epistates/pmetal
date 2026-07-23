//! Numerical-parity test for the DiffusionGemma **multimodal encoder** —
//! the text tower + Gemma 4 vision tower + vision→text projector + image-token
//! merge ([`DiffusionGemmaEncoderModel::forward_multimodal`]).
//!
//! The oracle is the authoritative HuggingFace `transformers`
//! `DiffusionGemmaEncoderModel`. `.strategy/parity/dump_diffusion_gemma_multimodal_reference.py`
//! builds a tiny seeded model, runs the encoder over a prompt with image
//! placeholders + one image, and commits the encoder hidden states.
//!
//! The `pooled_causal` fixture is the plain-causal path (no `mm_token_type_ids`)
//! — end-to-end validation of the vision tower, projector, and masked-scatter
//! merge feeding the causal text stack. The bidirectional-vision band can't be
//! parity-checked (the oracle discards it — see the second test), so it is
//! covered behaviourally there and by the mask unit test.

mod common;

use common::{fixture_path, load_shard, ref_tensor};
use pmetal_mlx::test_utils::{ParityReport, Tolerance, print_report_table};

use pmetal_models::architectures::diffusion_gemma::{
    DiffusionGemmaEncoderModel, DiffusionGemmaTextConfig,
    load_diffusion_gemma_encoder_multimodal_weights,
};
use pmetal_models::architectures::gemma4_vision::Gemma4VisionConfig;

fn text_config() -> DiffusionGemmaTextConfig {
    DiffusionGemmaTextConfig {
        vocab_size: 512,
        hidden_size: 64,
        intermediate_size: 128,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 16,
        global_head_dim: 32,
        num_global_key_value_heads: Some(1),
        sliding_window: 8,
        sliding_window_pattern: 2,
        rms_norm_eps: 1e-6,
        final_logit_softcapping: Some(30.0),
        num_experts: 4,
        top_k_experts: 2,
        moe_intermediate_size: 16,
        canvas_length: 8,
        use_bidirectional_attention: Some("vision".to_string()),
        ..Default::default()
    }
}

fn vision_config() -> Gemma4VisionConfig {
    Gemma4VisionConfig {
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 4,
        head_dim: 8,
        pooling_kernel_size: 2,
        patch_size: 4,
        position_embedding_size: 64,
        rope_theta: 100.0,
        ..Default::default()
    }
}

/// Build the encoder, attach vision, and load the multimodal fixture weights.
fn build_loaded_encoder(
    weights: &std::collections::HashMap<String, pmetal_bridge::compat::Array>,
) -> DiffusionGemmaEncoderModel {
    let mut enc = DiffusionGemmaEncoderModel::new(text_config()).expect("encoder builds");
    enc.attach_vision(&vision_config(), 500)
        .expect("vision attaches");
    let report =
        load_diffusion_gemma_encoder_multimodal_weights(&mut enc, weights).expect("weights load");
    assert!(
        report.skipped.is_empty(),
        "multimodal loader skipped tensors: {:?}",
        report.skipped
    );
    assert!(report.loaded > 0, "multimodal loader loaded 0 tensors");
    enc
}

/// `pooled_causal`: no `mm_token_type_ids` → the whole merged sequence is causal.
/// End-to-end validation of the vision tower, projector, and masked-scatter
/// merge feeding the causal text stack.
#[test]
fn diffusion_gemma_multimodal_causal_parity() {
    let ref_shard = load_shard(&fixture_path(
        "diffusion_gemma_multimodal_reference.safetensors",
    ));
    let weights = load_shard(&fixture_path(
        "diffusion_gemma_multimodal_weights.safetensors",
    ));
    let mut enc = build_loaded_encoder(&weights);

    let (hidden, _kvs) = enc
        .forward_multimodal(
            ref_tensor(&ref_shard, "input_ids"),
            ref_tensor(&ref_shard, "pixel_values"),
            ref_tensor(&ref_shard, "image_position_ids"),
            false,
        )
        .expect("multimodal forward runs");

    let report = ParityReport::compute_with_per_position(
        "pooled_causal",
        &hidden,
        ref_tensor(&ref_shard, "pooled_causal"),
        Tolerance::new(8e-3, 4e-3),
    );
    println!("\n== DiffusionGemma multimodal (causal) parity ==");
    print_report_table(std::slice::from_ref(&report));
    assert!(
        report.passed(),
        "multimodal causal parity failed (max abs {:.3e})",
        report.max_abs_diff
    );
}

/// The bidirectional-image band (`use_bidirectional_attention="vision"`) is the
/// architecture's intent, but it **cannot be parity-checked against the
/// oracle**: `transformers` (≤ 5.10.0.dev0) discards the bidirectional mask it
/// builds, so its encoder is silently causal (`pooled_bidir == pooled_causal`,
/// verified to 0.0 in the fixture). pmetal implements the intended behaviour and
/// deliberately diverges here (see `forward_multimodal`).
///
/// This is therefore a behavioural test, not a parity test: with the band on,
/// (a) tokens *before* any image are unchanged (they never attend forward into
/// the image span), and (b) image-token positions change relative to the causal
/// path (they now see the rest of their image block). The mask pattern itself is
/// unit-tested in the module tests.
#[test]
fn diffusion_gemma_multimodal_bidirectional_band_behaviour() {
    let ref_shard = load_shard(&fixture_path(
        "diffusion_gemma_multimodal_reference.safetensors",
    ));
    let weights = load_shard(&fixture_path(
        "diffusion_gemma_multimodal_weights.safetensors",
    ));

    let input_ids = ref_tensor(&ref_shard, "input_ids").clone();
    let pixel_values = ref_tensor(&ref_shard, "pixel_values").clone();
    let image_position_ids = ref_tensor(&ref_shard, "image_position_ids").clone();

    let mut enc = build_loaded_encoder(&weights);
    let (causal, _) = enc
        .forward_multimodal(&input_ids, &pixel_values, &image_position_ids, false)
        .expect("causal forward");
    let (mut bidir, _) = enc
        .forward_multimodal(&input_ids, &pixel_values, &image_position_ids, true)
        .expect("bidirectional forward");

    // input_ids = [2, 10, 500, 500, 500, 500, 11, 3]: positions 0,1 are text
    // before the image; 2..=5 are image tokens.
    let diff = pmetal_bridge::compat::ops::abs(&causal.subtract(&bidir));
    let per_pos: Vec<f32> = (0..8)
        .map(|t| {
            pmetal_bridge::compat::ops::slice_axis(&diff, 1, t, t + 1)
                .mean_all()
                .item_f32()
        })
        .collect();
    println!("per-position mean |causal − bidir|: {per_pos:?}");

    assert!(
        bidir
            .to_f32_vec(8 * 64)
            .unwrap()
            .iter()
            .all(|x| x.is_finite())
    );
    // Text-before-image is identical; image tokens change.
    assert!(
        per_pos[0] < 1e-4 && per_pos[1] < 1e-4,
        "pre-image text changed: {per_pos:?}"
    );
    assert!(
        per_pos[2] > 1e-3,
        "image token t=2 did not gain bidirectional context"
    );
}
