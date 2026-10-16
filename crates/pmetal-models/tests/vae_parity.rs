//! Numerical parity of the `AutoencoderKL` against `diffusers`.
//!
//! Two tiny seeded autoencoders with three resolution levels: the FLUX/SD3
//! layout (no quant convs, a latent shift and scale) and the Stable Diffusion
//! one (both quant convs, two layers per block, no shift). Both carry the
//! mid-block attention. Images and latents are channels-last in the fixtures.

mod common;

use std::collections::HashMap;

use common::{fixture_path, load_shard, ref_tensor};
use pmetal_cache::test_utils::{ParityReport, Tolerance, max_abs_diff, print_report_table};
use pmetal_models::architectures::vae::{AutoencoderKL, VAEConfig};
use pmetal_models::loader::{LoadError, load_vae_weights};
use pmetal_runtime::compat::{Array, Dtype};

fn flux_like() -> VAEConfig {
    VAEConfig {
        latent_channels: 8,
        block_out_channels: vec![16, 32, 32],
        layers_per_block: 1,
        norm_num_groups: 8,
        use_quant_conv: false,
        use_post_quant_conv: false,
        scaling_factor: 0.3611,
        shift_factor: Some(0.1159),
        ..Default::default()
    }
}

fn sd_like() -> VAEConfig {
    VAEConfig {
        latent_channels: 4,
        block_out_channels: vec![16, 32, 32],
        layers_per_block: 2,
        norm_num_groups: 8,
        use_quant_conv: true,
        use_post_quant_conv: true,
        scaling_factor: 0.18215,
        shift_factor: None,
        ..Default::default()
    }
}

fn weights(variant: &str) -> HashMap<String, Array> {
    load_shard(&fixture_path(&format!("vae_{variant}_weights.safetensors")))
}

fn reference(variant: &str) -> HashMap<String, Array> {
    load_shard(&fixture_path(&format!(
        "vae_{variant}_reference.safetensors"
    )))
}

fn model(variant: &str, config: VAEConfig) -> AutoencoderKL {
    let mut model = AutoencoderKL::new(config).expect("model builds");
    load_vae_weights(&mut model, weights(variant)).expect("weights load");
    model
}

fn check(variant: &str, config: VAEConfig) {
    let reference = reference(variant);
    let mut vae = model(variant, config);
    let image = ref_tensor(&reference, "image");
    let z = ref_tensor(&reference, "z");

    let distribution = vae.encode_distribution(image).expect("encode");
    let scaled = vae.encode(image, false).expect("encode mean");
    let decoded = vae.decode_raw(z).expect("decode");
    let round_trip = vae
        .decode(ref_tensor(&reference, "scaled_latents"))
        .expect("decode scaled");
    pmetal_runtime::check_unobserved_error().expect("no runtime op failed");

    let reports = [
        ParityReport::compute_with_per_position(
            "mean",
            &distribution.mean,
            ref_tensor(&reference, "mean"),
            Tolerance::new(3e-5, 3e-6),
        ),
        ParityReport::compute(
            "logvar",
            &distribution.logvar,
            ref_tensor(&reference, "logvar"),
            Tolerance::new(3e-5, 3e-6),
        ),
        ParityReport::compute(
            "scaled_latents",
            &scaled,
            ref_tensor(&reference, "scaled_latents"),
            Tolerance::new(3e-5, 3e-6),
        ),
        ParityReport::compute_with_per_position(
            "decoded",
            &decoded,
            ref_tensor(&reference, "decoded"),
            Tolerance::new(3e-5, 3e-6),
        ),
        ParityReport::compute(
            "decoded_from_scaled",
            &round_trip,
            ref_tensor(&reference, "decoded_from_scaled"),
            Tolerance::new(3e-5, 3e-6),
        ),
    ];
    print_report_table(&reports);
    for report in &reports {
        assert!(report.passed(), "{variant} {}: {report:?}", report.name);
    }
}

#[test]
fn the_flux_layout_matches_diffusers() {
    check("flux_like", flux_like());
}

#[test]
fn the_stable_diffusion_layout_matches_diffusers() {
    check("sd_like", sd_like());
}

/// The downsampler pads the bottom and right only. The same weights with a
/// symmetric padding of one sample a different grid, so the encoder's output
/// would not match; this pins the fixture to that, not just the code.
#[test]
fn the_fixture_is_sensitive_to_the_downsampler_padding() {
    let reference = reference("flux_like");
    let mut vae = model("flux_like", flux_like());
    for block in &mut vae.encoder.down_blocks {
        for downsampler in &mut block.downsamplers {
            downsampler.conv.padding = [1, 1];
        }
    }
    // A padded stride-2 conv on 16x16 gives 8x8, then 4x4: the same shape, a
    // different grid.
    let wrong = vae
        .encode_distribution(ref_tensor(&reference, "image"))
        .expect("encode")
        .mean;
    let gap = max_abs_diff(&wrong, ref_tensor(&reference, "mean"));
    assert!(gap > 1e-2, "symmetric padding moves the mean by {gap:.2e}");
}

/// Scoring the attention's queries a few at a time gives the whole
/// attention's rows.
#[test]
fn chunked_attention_equals_whole_attention() {
    let mut vae = model("flux_like", flux_like());
    let attention = &mut vae.encoder.mid_block.attentions[0];
    pmetal_runtime::compat::random::seed(4);
    let x = pmetal_runtime::compat::random::normal(&[2, 4, 4, 32], Dtype::Float32);
    let whole = attention.forward_in_chunks(&x, 16).expect("whole");
    for rows in [1, 3, 5, 15] {
        let chunked = attention.forward_in_chunks(&x, rows).expect("chunked");
        let gap = max_abs_diff(&chunked, &whole);
        assert!(gap < 1e-5, "{rows} rows per chunk differ by {gap:.2e}");
    }
    pmetal_runtime::check_unobserved_error().expect("no runtime op failed");
}

#[test]
fn a_bf16_autoencoder_stays_in_bf16() {
    let reference = reference("flux_like");
    let mut vae = AutoencoderKL::new(flux_like()).expect("model builds");
    let half: HashMap<String, Array> = weights("flux_like")
        .into_iter()
        .map(|(k, v)| (k, v.as_dtype(Dtype::Bfloat16.as_i32())))
        .collect();
    load_vae_weights(&mut vae, half).expect("weights load");

    let image = ref_tensor(&reference, "image").as_dtype(Dtype::Bfloat16.as_i32());
    let latents = vae.encode(&image, false).expect("encode");
    let decoded = vae.decode(&latents).expect("decode");
    assert_eq!(latents.dtype(), Dtype::Bfloat16);
    assert_eq!(decoded.dtype(), Dtype::Bfloat16);
    pmetal_runtime::check_unobserved_error().expect("no runtime op failed");
    let report = ParityReport::compute(
        "bf16 decode",
        &decoded,
        ref_tensor(&reference, "decoded_from_scaled"),
        Tolerance::new(0.15, 0.15),
    );
    assert!(report.passed(), "{report:?}");
}

#[test]
fn the_pre_0_14_attention_layout_loads() {
    // query/key/value/proj_attn as 1x1 convolutions, the layout `diffusers`
    // converts on load.
    let mut old = weights("flux_like");
    for prefix in ["encoder", "decoder"] {
        for (new, legacy) in [
            ("to_q", "query"),
            ("to_k", "key"),
            ("to_v", "value"),
            ("to_out.0", "proj_attn"),
        ] {
            for suffix in ["weight", "bias"] {
                let from = format!("{prefix}.mid_block.attentions.0.{new}.{suffix}");
                let to = format!("{prefix}.mid_block.attentions.0.{legacy}.{suffix}");
                let tensor = old.remove(&from).expect("fixture key");
                let tensor = if suffix == "weight" {
                    tensor.reshape(&[tensor.dim(0), tensor.dim(1), 1, 1])
                } else {
                    tensor
                };
                old.insert(to, tensor);
            }
        }
    }
    let reference = reference("flux_like");
    let mut vae = AutoencoderKL::new(flux_like()).expect("model builds");
    load_vae_weights(&mut vae, old).expect("legacy attention loads");
    let decoded = vae
        .decode_raw(ref_tensor(&reference, "z"))
        .expect("decode");
    let gap = max_abs_diff(&decoded, ref_tensor(&reference, "decoded"));
    assert!(gap < 1e-4, "legacy layout decodes {gap:.2e} off");
}

#[test]
fn loading_is_strict() {
    let mut extra = weights("flux_like");
    extra.insert(
        "decoder.up_blocks.0.resnets.0.mystery.weight".into(),
        Array::from_slice(&[1.0f32], &[1]),
    );
    let mut vae = AutoencoderKL::new(flux_like()).unwrap();
    let err = load_vae_weights(&mut vae, extra).expect_err("a stray tensor");
    assert!(err.to_string().contains("mystery"), "{err}");

    let mut short = weights("flux_like");
    short.remove("decoder.conv_out.bias");
    let mut vae = AutoencoderKL::new(flux_like()).unwrap();
    let err = load_vae_weights(&mut vae, short).expect_err("a missing tensor");
    assert!(err.to_string().contains("decoder.conv_out.bias"), "{err}");

    // An SD-layout checkpoint into a model built without its quant convs.
    let mut vae = AutoencoderKL::new(flux_like()).unwrap();
    let err = load_vae_weights(&mut vae, weights("sd_like")).expect_err("wrong layout");
    assert!(
        matches!(err, LoadError::ShapeMismatch { .. } | LoadError::Incomplete { .. }),
        "{err}"
    );
}

#[test]
fn a_config_parses_the_way_released_configs_spell_it() {
    // FLUX.1's vae/config.json, abridged.
    let config: VAEConfig = serde_json::from_str(
        r#"{"_class_name": "AutoencoderKL", "act_fn": "silu",
            "block_out_channels": [128, 256, 512, 512], "force_upcast": true,
            "in_channels": 3, "latent_channels": 16, "latents_mean": null,
            "layers_per_block": 2, "mid_block_add_attention": true,
            "norm_num_groups": 32, "out_channels": 3, "sample_size": 1024,
            "scaling_factor": 0.3611, "shift_factor": 0.1159,
            "use_post_quant_conv": false, "use_quant_conv": false}"#,
    )
    .unwrap();
    let flux = VAEConfig::flux();
    assert_eq!(config.block_out_channels, flux.block_out_channels);
    assert_eq!(config.shift_factor, flux.shift_factor);
    assert!(!config.use_quant_conv && !config.use_post_quant_conv);
    // A config that names none of it is `diffusers`' default: SD's quant convs.
    let bare: VAEConfig = serde_json::from_str("{}").unwrap();
    assert!(bare.use_quant_conv && bare.use_post_quant_conv);
    assert_eq!(bare.shift_factor, None);
}

#[test]
fn channels_that_do_not_divide_into_groups_are_refused() {
    let err = AutoencoderKL::new(VAEConfig {
        block_out_channels: vec![20],
        norm_num_groups: 8,
        ..Default::default()
    })
    .expect_err("20 channels in 8 groups");
    assert!(err.to_string().contains("20"), "{err}");
}
