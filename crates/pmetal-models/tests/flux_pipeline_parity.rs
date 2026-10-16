//! The FLUX.1 sampling loop and decode against `diffusers`' `FluxPipeline`.
//!
//! A tiny guidance-distilled transformer and a tiny two-level VAE, both seeded
//! in `diffusers`, are run through `FluxPipeline.__call__` with the prompt
//! embeddings and the initial (packed) noise given, for three steps of the
//! dynamically shifted schedule. The fixture keeps the schedule, the image
//! position ids, the final packed latents and the decoded image.

mod common;

use common::{fixture_path, load_shard, ref_tensor};
use pmetal_cache::test_utils::{ParityReport, Tolerance, max_abs_diff, print_report_table};
use pmetal_models::architectures::clip::CLIPConfig;
use pmetal_models::architectures::flux::FluxConfig;
use pmetal_models::architectures::t5::T5Config;
use pmetal_models::architectures::vae::VAEConfig;
use pmetal_models::loader::{load_flux_weights, load_vae_weights};
use pmetal_models::pipelines::{
    FluxPipeline, PromptEmbeddings, latent_image_ids, pack_latents, unpack_latents,
};
use pmetal_models::sampling::diffusion::FlowMatchConfig;
use pmetal_runtime::compat::Array;

const HEIGHT: usize = 32;
const WIDTH: usize = 48;
const STEPS: usize = 3;
const GUIDANCE: f32 = 3.5;

fn pipeline() -> FluxPipeline {
    let clip = CLIPConfig {
        vocab_size: 10,
        embed_dim: 8,
        num_layers: 1,
        num_heads: 1,
        intermediate_size: 8,
        max_position_embeddings: 4,
        ..Default::default()
    };
    let t5 = T5Config {
        vocab_size: 10,
        d_model: 8,
        d_ff: 8,
        d_kv: 4,
        num_layers: 1,
        num_heads: 2,
        ..Default::default()
    };
    let flux = FluxConfig {
        in_channels: 16,
        num_layers: 1,
        num_single_layers: 1,
        attention_head_dim: 16,
        num_attention_heads: 2,
        joint_attention_dim: 32,
        pooled_projection_dim: 24,
        guidance_embeds: true,
        axes_dims_rope: vec![4, 6, 6],
        ..Default::default()
    };
    let vae = VAEConfig {
        latent_channels: 4,
        block_out_channels: vec![8, 16],
        layers_per_block: 1,
        norm_num_groups: 4,
        use_quant_conv: false,
        use_post_quant_conv: false,
        scaling_factor: 0.3611,
        shift_factor: Some(0.1159),
        ..Default::default()
    };
    let mut pipeline = FluxPipeline::new(clip, t5, flux, vae, FlowMatchConfig::flux_dev())
        .expect("pipeline builds");
    load_flux_weights(
        &mut pipeline.dit,
        load_shard(&fixture_path(
            "flux_pipeline_transformer_weights.safetensors",
        )),
    )
    .expect("transformer loads");
    load_vae_weights(
        &mut pipeline.vae,
        load_shard(&fixture_path("flux_pipeline_vae_weights.safetensors")),
    )
    .expect("vae loads");
    pipeline
}

#[test]
fn the_sampling_loop_and_decode_match_diffusers() {
    let reference = load_shard(&fixture_path("flux_pipeline_reference.safetensors"));
    let mut pipeline = pipeline();
    let embeddings = PromptEmbeddings {
        prompt_embeds: ref_tensor(&reference, "prompt_embeds").clone(),
        pooled_prompt_embeds: ref_tensor(&reference, "pooled_prompt_embeds").clone(),
    };
    let noise = ref_tensor(&reference, "noise");

    let packed = pipeline
        .sample_latents(&embeddings, noise, STEPS, GUIDANCE)
        .expect("sampling");
    assert_eq!(pipeline.pixels_per_token(), 4);
    let (rows, cols) = (2 * (HEIGHT as i32 / 4), 2 * (WIDTH as i32 / 4));
    let image = pipeline
        .decode_latents(&packed, rows, cols)
        .expect("decode");
    pmetal_runtime::check_unobserved_error().expect("no runtime op failed");

    // The reference's decoded image is channels-first, as the pipeline returns it.
    let decoded = ref_tensor(&reference, "decoded");
    let sigmas = Array::from_slice(pipeline.scheduler.sigmas(), &[(STEPS + 1) as i32]);
    let reports = [
        ParityReport::compute(
            "sigmas",
            &sigmas,
            ref_tensor(&reference, "sigmas"),
            Tolerance::new(2e-6, 2e-6),
        ),
        ParityReport::compute(
            "final_packed",
            &packed,
            ref_tensor(&reference, "final_packed"),
            Tolerance::new(5e-5, 1e-5),
        ),
        ParityReport::compute("decoded", &image, decoded, Tolerance::new(1e-4, 1e-5)),
    ];
    print_report_table(&reports);
    for report in &reports {
        assert!(report.passed(), "{}: {report:?}", report.name);
    }
}

#[test]
fn the_image_position_ids_match_diffusers() {
    let reference = load_shard(&fixture_path("flux_pipeline_reference.safetensors"));
    let ids = latent_image_ids(16, 24);
    // diffusers keeps a batch axis in the fixture's ids on the first dim.
    let want = ref_tensor(&reference, "latent_image_ids");
    assert!(max_abs_diff(&ids, &want.reshape(&[8 * 12, 3])) == 0.0);
}

/// Packing is a pure rearrangement: it round-trips, and `diffusers`' packed
/// noise is what packing our channels-last noise gives.
#[test]
fn packing_matches_diffusers_and_round_trips() {
    let reference = load_shard(&fixture_path("flux_pipeline_reference.safetensors"));
    let packed = pack_latents(ref_tensor(&reference, "noise")).expect("pack");
    assert_eq!(
        max_abs_diff(&packed, ref_tensor(&reference, "packed_noise")),
        0.0
    );
    let back = unpack_latents(&packed, 16, 24).expect("unpack");
    assert_eq!(max_abs_diff(&back, ref_tensor(&reference, "noise")), 0.0);
}

#[test]
fn unpacking_tokens_that_are_not_the_grid_is_an_error() {
    let tokens = Array::from_slice(&vec![0.0f32; 2 * 10 * 8], &[2, 10, 8]);
    assert!(unpack_latents(&tokens, 4, 6).is_err());
    assert!(pack_latents(&Array::from_slice(&[0.0f32; 3 * 3 * 4], &[1, 3, 3, 4])).is_err());
}
