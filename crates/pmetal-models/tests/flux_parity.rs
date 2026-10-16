//! Numerical parity of the FLUX.1 transformer against `diffusers`'
//! `FluxTransformer2DModel`.
//!
//! Two tiny seeded transformers, two double-stream and two single-stream
//! blocks each: a guidance-distilled one (`dev`) and a timestep-distilled one
//! (`schnell`). Each fixture holds the rotary tables, the conditioning vector,
//! the states after the last block of each kind, and the output. The same
//! weights are also written in the original (single-file) layout, which
//! `diffusers`' own converter turns back into the `diffusers` tensors, so that
//! loader is held to the same outputs.

mod common;

use std::collections::HashMap;

use common::{fixture_path, load_shard, ref_tensor};
use pmetal_cache::test_utils::{ParityReport, Tolerance, max_abs_diff, print_report_table};
use pmetal_models::architectures::flux::{FluxConfig, FluxDiT};
use pmetal_models::loader::{LoadError, load_flux_weights};
use pmetal_runtime::compat::{Array, Dtype, ops};

fn config(guidance_embeds: bool) -> FluxConfig {
    FluxConfig {
        in_channels: 16,
        num_layers: 2,
        num_single_layers: 2,
        attention_head_dim: 16,
        num_attention_heads: 4,
        joint_attention_dim: 32,
        pooled_projection_dim: 24,
        guidance_embeds,
        axes_dims_rope: vec![4, 6, 6],
        ..Default::default()
    }
}

fn weights(variant: &str, layout: &str) -> HashMap<String, Array> {
    load_shard(&fixture_path(&format!(
        "flux_{variant}{layout}_weights.safetensors"
    )))
}

fn reference(variant: &str) -> HashMap<String, Array> {
    load_shard(&fixture_path(&format!(
        "flux_{variant}_reference.safetensors"
    )))
}

fn model(variant: &str, layout: &str) -> FluxDiT {
    let mut model = FluxDiT::new(config(variant == "dev")).expect("model builds");
    load_flux_weights(&mut model, weights(variant, layout)).expect("weights load");
    model
}

fn run(model: &mut FluxDiT, reference: &HashMap<String, Array>) -> Array {
    model
        .forward(
            ref_tensor(reference, "hidden_states"),
            ref_tensor(reference, "timestep"),
            ref_tensor(reference, "encoder_hidden_states"),
            ref_tensor(reference, "pooled_projections"),
            reference.get("guidance"),
            ref_tensor(reference, "txt_ids"),
            ref_tensor(reference, "img_ids"),
        )
        .expect("forward")
}

fn report(name: &str, got: &Array, want: &Array) -> ParityReport {
    ParityReport::compute(name, got, want, Tolerance::new(1e-4, 1e-5))
}

fn check(variant: &str, layout: &str) {
    let reference = reference(variant);
    let mut model = model(variant, layout);
    let output = run(&mut model, &reference);
    pmetal_runtime::check_unobserved_error().expect("no runtime op failed");

    // The pieces: conditioning, then the stream states after the last block of
    // each kind, walked as the forward does.
    let tables = {
        let ids = ops::concatenate_axis(
            &[
                ref_tensor(&reference, "txt_ids"),
                ref_tensor(&reference, "img_ids"),
            ],
            0,
        );
        model.rope_tables(&ids).expect("rope tables")
    };
    let temb = model
        .time_text_embed
        .forward(
            &ref_tensor(&reference, "timestep").mul_scalar(1000.0),
            reference
                .get("guidance")
                .map(|g| g.mul_scalar(1000.0))
                .as_ref(),
            ref_tensor(&reference, "pooled_projections"),
            Dtype::Float32,
        )
        .expect("conditioning");
    let mut image = model
        .x_embedder
        .forward(ref_tensor(&reference, "hidden_states"));
    let mut text = model
        .context_embedder
        .forward(ref_tensor(&reference, "encoder_hidden_states"));
    for block in &mut model.transformer_blocks {
        (image, text) = block
            .forward(&image, &text, &temb, &tables)
            .expect("double block");
    }
    let text_len = text.dim(1);
    let mut joint = ops::concatenate_axis(&[&text, &image], 1);
    let after_double_text = text.clone();
    let after_double_image = image.clone();
    for block in &mut model.single_transformer_blocks {
        joint = block.forward(&joint, &temb, &tables).expect("single block");
    }
    let after_single_image = ops::slice_axis_from(&joint, 1, text_len);

    // The reference's tables repeat each angle for its channel pair.
    let pair = |t: &Array| {
        let half = t.dim(1) / 2;
        ops::select_axis(&t.reshape(&[t.dim(0), half, 2]), 0, 2)
    };
    let reports = [
        report(
            "rope_cos",
            &tables.cos,
            &pair(ref_tensor(&reference, "rope_cos")),
        ),
        report(
            "rope_sin",
            &tables.sin,
            &pair(ref_tensor(&reference, "rope_sin")),
        ),
        report("temb", &temb, ref_tensor(&reference, "temb")),
        report(
            "double_last_text",
            &after_double_text,
            ref_tensor(&reference, "double_last_text"),
        ),
        report(
            "double_last_image",
            &after_double_image,
            ref_tensor(&reference, "double_last_image"),
        ),
        report(
            "single_last_image",
            &after_single_image,
            ref_tensor(&reference, "single_last_image"),
        ),
        report("output", &output, ref_tensor(&reference, "output")),
    ];
    print_report_table(&reports);
    for report in &reports {
        assert!(
            report.passed(),
            "{variant}{layout} {}: {report:?}",
            report.name
        );
    }
}

#[test]
fn a_guidance_distilled_transformer_matches_diffusers() {
    check("dev", "");
}

#[test]
fn a_timestep_distilled_transformer_matches_diffusers() {
    check("schnell", "");
}

#[test]
fn an_original_layout_checkpoint_matches_diffusers() {
    check("dev", "_original");
    check("schnell", "_original");
}

/// The comparison above must be able to fail. Rotating by one id fewer, or
/// feeding the noise level unscaled, moves the output well past tolerance.
#[test]
fn the_fixture_is_sensitive_to_position_and_timestep() {
    let reference = reference("dev");
    let mut model = model("dev", "");
    let want = ref_tensor(&reference, "output");

    // Image tokens on a row-0 line instead of the grid.
    let flat_ids = ref_tensor(&reference, "img_ids")
        .multiply(&Array::from_slice(&[0.0f32, 0.0, 1.0], &[1, 3]));
    let moved = model
        .forward(
            ref_tensor(&reference, "hidden_states"),
            ref_tensor(&reference, "timestep"),
            ref_tensor(&reference, "encoder_hidden_states"),
            ref_tensor(&reference, "pooled_projections"),
            reference.get("guidance"),
            ref_tensor(&reference, "txt_ids"),
            &flat_ids,
        )
        .expect("forward");
    assert!(
        max_abs_diff(&moved, want) > 1e-2,
        "the row id is not exercised"
    );

    let other_time = ref_tensor(&reference, "timestep").mul_scalar(0.5);
    let moved = model
        .forward(
            ref_tensor(&reference, "hidden_states"),
            &other_time,
            ref_tensor(&reference, "encoder_hidden_states"),
            ref_tensor(&reference, "pooled_projections"),
            reference.get("guidance"),
            ref_tensor(&reference, "txt_ids"),
            ref_tensor(&reference, "img_ids"),
        )
        .expect("forward");
    assert!(
        max_abs_diff(&moved, want) > 1e-2,
        "the timestep is not exercised"
    );
}

/// A guidance-distilled model needs a guidance scale and a timestep-distilled
/// one refuses it, each naming the cause; `diffusers` fails on both too, with a
/// less helpful message.
#[test]
fn guidance_must_match_the_model() {
    let reference = reference("dev");
    let mut dev = model("dev", "");
    let hidden = ref_tensor(&reference, "hidden_states");
    let go = |model: &mut FluxDiT, guidance: Option<&Array>| {
        model.forward(
            hidden,
            ref_tensor(&reference, "timestep"),
            ref_tensor(&reference, "encoder_hidden_states"),
            ref_tensor(&reference, "pooled_projections"),
            guidance,
            ref_tensor(&reference, "txt_ids"),
            ref_tensor(&reference, "img_ids"),
        )
    };
    let err = go(&mut dev, None).expect_err("dev needs guidance");
    assert!(err.to_string().contains("guidance"), "{err}");

    let mut schnell = model("schnell", "");
    let err =
        go(&mut schnell, ref_tensor(&reference, "guidance").into()).expect_err("schnell has none");
    assert!(err.to_string().contains("guidance"), "{err}");
}

/// A bf16 transformer stays bf16 and its output stays near the f32 one.
#[test]
fn a_bf16_transformer_stays_in_bf16() {
    let reference = reference("dev");
    let mut model = FluxDiT::new(config(true)).expect("model builds");
    let half: HashMap<String, Array> = weights("dev", "")
        .into_iter()
        .map(|(k, v)| (k, v.as_dtype(Dtype::Bfloat16.as_i32())))
        .collect();
    load_flux_weights(&mut model, half).expect("weights load");
    let cast = |name: &str| ref_tensor(&reference, name).as_dtype(Dtype::Bfloat16.as_i32());
    let output = model
        .forward(
            &cast("hidden_states"),
            ref_tensor(&reference, "timestep"),
            &cast("encoder_hidden_states"),
            &cast("pooled_projections"),
            reference.get("guidance"),
            ref_tensor(&reference, "txt_ids"),
            ref_tensor(&reference, "img_ids"),
        )
        .expect("forward");
    assert_eq!(output.dtype(), Dtype::Bfloat16);
    pmetal_runtime::check_unobserved_error().expect("no runtime op failed");
    let gap = max_abs_diff(&output, ref_tensor(&reference, "output"));
    assert!(gap < 0.3, "bf16 output is {gap:.2e} from the f32 reference");
}

#[test]
fn loading_is_strict() {
    let mut extra = weights("dev", "");
    extra.insert(
        "single_transformer_blocks.0.attn.mystery.weight".into(),
        Array::from_slice(&[1.0f32], &[1]),
    );
    let mut m = FluxDiT::new(config(true)).unwrap();
    let err = load_flux_weights(&mut m, extra).expect_err("a stray tensor");
    assert!(err.to_string().contains("mystery"), "{err}");

    let mut short = weights("dev", "");
    short.remove("single_transformer_blocks.1.proj_mlp.bias");
    let mut m = FluxDiT::new(config(true)).unwrap();
    let err = load_flux_weights(&mut m, short).expect_err("a missing tensor");
    assert!(
        err.to_string()
            .contains("single_transformer_blocks.1.proj_mlp.bias"),
        "{err}"
    );

    // A dev checkpoint into a model without a guidance embedder.
    let mut m = FluxDiT::new(config(false)).unwrap();
    let err = load_flux_weights(&mut m, weights("dev", "")).expect_err("guidance without a home");
    assert!(err.to_string().contains("guidance_embedder"), "{err}");

    // ... and the reverse.
    let mut m = FluxDiT::new(config(true)).unwrap();
    let err = load_flux_weights(&mut m, weights("schnell", "")).expect_err("guidance unfilled");
    assert!(err.to_string().contains("guidance_embedder"), "{err}");

    let mut wrong = weights("dev", "");
    wrong.insert(
        "x_embedder.weight".into(),
        Array::from_slice(&vec![0.0f32; 64 * 8], &[64, 8]),
    );
    let mut m = FluxDiT::new(config(true)).unwrap();
    let err = load_flux_weights(&mut m, wrong).expect_err("a wrong shape");
    assert!(matches!(err, LoadError::ShapeMismatch { .. }), "{err}");

    // An original-layout file that lost a block's fused projection.
    let mut cut = weights("dev", "_original");
    cut.remove("double_blocks.1.img_attn.qkv.weight");
    let mut m = FluxDiT::new(config(true)).unwrap();
    let err = load_flux_weights(&mut m, cut).expect_err("a missing fused projection");
    assert!(
        err.to_string()
            .contains("transformer_blocks.1.attn.to_q.weight"),
        "{err}"
    );
}

#[test]
fn a_config_must_rotate_every_channel_of_a_head() {
    let err = FluxDiT::new(FluxConfig {
        axes_dims_rope: vec![4, 6, 4],
        ..config(false)
    })
    .expect_err("14 of 16 channels");
    assert!(err.to_string().contains("axes_dims_rope"), "{err}");
}

#[test]
fn a_config_parses_the_way_released_configs_spell_it() {
    // FLUX.1-dev's transformer/config.json.
    let config: FluxConfig = serde_json::from_str(
        r#"{"_class_name": "FluxTransformer2DModel", "attention_head_dim": 128,
            "guidance_embeds": true, "in_channels": 64, "joint_attention_dim": 4096,
            "num_attention_heads": 24, "num_layers": 19, "num_single_layers": 38,
            "patch_size": 1, "pooled_projection_dim": 768}"#,
    )
    .unwrap();
    assert_eq!(config.hidden_size(), 3072);
    assert!(config.guidance_embeds);
    assert_eq!(config.axes_dims_rope, vec![16, 56, 56]);
}
