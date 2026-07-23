//! Numerical-parity test for the Rust Gemma 4 vision tower port
//! ([`Gemma4VisionModel`], `model_type: gemma4_vision`) — DiffusionGemma's
//! image backbone.
//!
//! The oracle is the authoritative HuggingFace `transformers` implementation of
//! `gemma4_vision` — the reference the released weights are defined against.
//! `.strategy/parity/dump_gemma4_vision_reference.py` builds a tiny seeded
//! `Gemma4VisionModel` (norms + ones-init `position_embedding_table` perturbed
//! so parity isn't blind to them), runs a forward over a regular 4×4 patch
//! grid, and commits the patch-embedder output, the encoder `last_hidden_state`,
//! and the pooled soft-token features as a fixture.
//!
//! This loads the same weights into the Rust tower and diffs each checkpoint,
//! exercising the linear patch embed + 2-D position table, the multidimensional
//! RoPE + GQA attention, the gelu-tanh SwiGLU MLP, and the position-based
//! average pooler with `√hidden` scaling.

mod common;

use common::{fixture_path, load_shard, ref_tensor};
use pmetal_bridge::compat::{Array, ops};
use pmetal_mlx::test_utils::{ParityReport, Tolerance, print_report_table};

use pmetal_models::architectures::gemma4_vision::{
    Gemma4VisionConfig, Gemma4VisionModel, load_gemma4_vision_weights,
};

/// Synthetic config — mirrors `SYNTHETIC_ARGS` in the Python dumper.
fn synthetic_config() -> Gemma4VisionConfig {
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

/// `(position_ids == -1).all(axis=-1)` → `[B, N]` bool padding mask (matches the
/// computation inside [`Gemma4VisionModel::forward`]).
fn padding_positions(position_ids: &Array) -> Array {
    let neg_one = Array::from_f32(-1.0).as_type::<i32>();
    let is_pad = ops::equal(position_ids, &neg_one).as_type::<f32>();
    ops::greater(&is_pad.sum_axis(2, false), &Array::from_f32(1.5))
}

fn compare(name: &str, rust: &Array, reference: &Array, tol: Tolerance) -> ParityReport {
    ParityReport::compute_with_per_position(name, rust, reference, tol)
}

#[test]
fn gemma4_vision_synthetic_parity() {
    let ref_shard = load_shard(&fixture_path("gemma4_vision_reference.safetensors"));
    let weights = load_shard(&fixture_path("gemma4_vision_weights.safetensors"));

    let pixel_values = ref_tensor(&ref_shard, "pixel_values").clone();
    let position_ids = ref_tensor(&ref_shard, "position_ids").clone();

    let config = synthetic_config();
    let mut model = Gemma4VisionModel::new(&config).expect("model builds");
    let report =
        load_gemma4_vision_weights(&mut model, &weights).expect("vision weight loader runs");
    assert!(
        report.skipped.is_empty(),
        "vision weight loader skipped tensors: {:?}",
        report.skipped
    );
    assert!(report.loaded > 0, "vision weight loader loaded 0 tensors");

    // Checkpoint walk through the public sub-modules (the same calls
    // `Gemma4VisionModel::forward` makes).
    let pad = padding_positions(&position_ids);
    let patch_embeds_rust = model
        .patch_embedder
        .forward(&pixel_values, &position_ids, &pad);
    let encoder_out_rust = model
        .encoder
        .forward(&patch_embeds_rust, &position_ids, None)
        .expect("encoder forward runs");

    // Full path → pooled soft tokens. The reference strips the batch dim
    // (`hidden_states[pooler_mask]`), so squeeze the single-image batch axis.
    let pooled_rust = model
        .forward(&pixel_values, &position_ids)
        .expect("vision forward runs")
        .squeeze(0);

    let reports = vec![
        compare(
            "patch_embeds",
            &patch_embeds_rust,
            ref_tensor(&ref_shard, "patch_embeds"),
            Tolerance::new(1e-4, 1e-4),
        ),
        compare(
            "encoder_out",
            &encoder_out_rust,
            ref_tensor(&ref_shard, "encoder_out"),
            Tolerance::new(6e-3, 3e-3),
        ),
        // Pooling scales activations by √hidden (≈5.66×), so the absolute
        // tolerance is scaled up accordingly.
        compare(
            "pooled",
            &pooled_rust,
            ref_tensor(&ref_shard, "pooled"),
            Tolerance::new(1.5e-2, 5e-3),
        ),
    ];

    println!("\n== Gemma4 vision synthetic parity report ==");
    print_report_table(&reports);

    for r in &reports {
        assert!(
            r.passed(),
            "checkpoint {} failed parity (max abs {:.3e})",
            r.name,
            r.max_abs_diff
        );
    }
}
