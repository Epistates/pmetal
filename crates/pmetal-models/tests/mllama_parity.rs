//! Numerical-parity test for the Rust Mllama (Llama 3.2 Vision) port.
//!
//! The oracle is the authoritative HuggingFace
//! `transformers.models.mllama.modeling_mllama`. There is no mlx-lm port to
//! cross-check against, and the implementation this replaced was a
//! non-functional skeleton that still produced plausible-looking tensors — so
//! this fixture is the only thing that distinguishes "runs" from "correct".
//!
//! `.strategy/parity/dump_mllama_reference.py` builds a tiny seeded
//! `MllamaForConditionalGeneration`, sets every scalar gate to 0.7 (the shipped
//! inits of 0 and π/4 would leave `tanh(gate)` small enough to hide the entire
//! gated global transformer and all of cross-attention), and commits the
//! `state_dict` alongside three checkpoints:
//!
//! * `vision_out` — the tower's `[batch, images, tiles, patches, 7680-analogue]`
//!   output, which catches the tile/position embeddings, the gated global
//!   stack, the 8-multiple patch padding, and the intermediate-layer concat.
//! * `cross_states` — the projector output, isolating the text boundary.
//! * `logits` — end to end, with cross-attention live.
//!
//! The fixture's inputs are chosen to make the awkward paths load-bearing: two
//! of the four tile slots are padding, and the last text row may attend to no
//! tile at all (`full_text_row_masked_out_mask`).

mod common;

use common::{fixture_path, load_shard, ref_tensor};
use pmetal_mlx::test_utils::{ParityReport, Tolerance, print_report_table};

use pmetal_models::architectures::llama::LlamaConfig;
use pmetal_models::architectures::mllama::{
    MllamaConfig, MllamaForConditionalGeneration, MllamaTextConfig, MllamaVisionConfig,
    MllamaVisionInputs, load_mllama_weights,
};

const REFERENCE: &str = "mllama_reference.safetensors";
const WEIGHTS: &str = "mllama_weights.safetensors";

/// Forward-pass tolerance, set from the measured worst case (2.6e-6 on
/// `vision_out`, over a 4.5 reference magnitude) with modest headroom. Note
/// `ParityReport::passed` is `atol || rtol`, so both have to stay tight for this
/// to mean anything.
const TOL: Tolerance = Tolerance::new(5e-6, 1e-6);

/// Mirrors `VISION_ARGS` / `TEXT_ARGS` in the Python dumper.
fn fixture_config() -> MllamaConfig {
    MllamaConfig {
        vision_config: MllamaVisionConfig {
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 2,
            num_global_layers: 1,
            attention_heads: 4,
            num_channels: 3,
            image_size: 28,
            patch_size: 14,
            hidden_act: "gelu".to_string(),
            norm_eps: 1e-5,
            max_num_tiles: 4,
            intermediate_layers_indices: vec![1],
            vision_output_dim: 64,
            supported_aspect_ratios: None,
        },
        text_config: MllamaTextConfig {
            llama: LlamaConfig {
                vocab_size: 64,
                hidden_size: 64,
                intermediate_size: 128,
                num_hidden_layers: 3,
                num_attention_heads: 4,
                num_key_value_heads: Some(2),
                head_dim: None,
                max_position_embeddings: 128,
                rms_norm_eps: 1e-5,
                rope_theta: 10000.0,
                tie_word_embeddings: false,
                ..Default::default()
            },
            cross_attention_layers: vec![1],
        },
        ..Default::default()
    }
}

#[test]
fn mllama_synthetic_parity() {
    let reference = load_shard(&fixture_path(REFERENCE));
    let weights = load_shard(&fixture_path(WEIGHTS));

    let config = fixture_config();
    let mut model = MllamaForConditionalGeneration::new(config).expect("model builds");

    let load = load_mllama_weights(&mut model, &weights).expect("weight loader runs");
    // Every key the loader looks for must exist. A typo here would otherwise
    // leave a random-init tensor in place and merely widen the diff.
    assert!(
        load.skipped.is_empty(),
        "weight loader skipped tensors: {:?}",
        load.skipped
    );
    // ...and the reverse: nothing in the checkpoint may go unclaimed, which is
    // what would happen if a module were missing from the port entirely.
    assert_eq!(
        load.loaded,
        weights.len(),
        "loaded {} of {} checkpoint tensors",
        load.loaded,
        weights.len()
    );

    let pixel_values = ref_tensor(&reference, "pixel_values").clone();
    let aspect_ratio_ids = ref_tensor(&reference, "aspect_ratio_ids").clone();
    let aspect_ratio_mask = ref_tensor(&reference, "aspect_ratio_mask").clone();
    let input_ids = ref_tensor(&reference, "input_ids").clone();
    let cross_attention_mask = ref_tensor(&reference, "cross_attention_mask").clone();

    let vision = MllamaVisionInputs {
        pixel_values: &pixel_values,
        aspect_ratio_ids: &aspect_ratio_ids,
        aspect_ratio_mask: &aspect_ratio_mask,
    };

    // Checkpoint 1: the vision tower alone.
    let vision_out = model
        .vision_model
        .forward(&pixel_values, &aspect_ratio_ids, &aspect_ratio_mask)
        .expect("vision tower runs");

    // Checkpoint 2: the projected features the text side consumes. Ours are
    // already collapsed to [images·tiles, patches, hidden]; the reference keeps
    // the leading axes, so restore them for the shape comparison.
    let reference_states = ref_tensor(&reference, "cross_states");
    let cross_states = model
        .prepare_cross_attention(vision, Some(&cross_attention_mask))
        .expect("projector runs")
        .states
        .reshape(reference_states.shape());

    // Checkpoint 3: end to end.
    let logits = model
        .forward(&input_ids, Some(vision), Some(&cross_attention_mask))
        .expect("full forward runs");

    let reports = vec![
        ParityReport::compute(
            "vision_out",
            &vision_out,
            ref_tensor(&reference, "vision_out"),
            TOL,
        ),
        ParityReport::compute("cross_states", &cross_states, reference_states, TOL),
        ParityReport::compute("logits", &logits, ref_tensor(&reference, "logits"), TOL),
    ];

    println!("\n== Mllama synthetic parity report ==");
    print_report_table(&reports);

    for report in &reports {
        assert!(
            report.passed(),
            "checkpoint {} failed parity (max abs {:.3e}, shapes {:?} vs {:?})",
            report.name,
            report.max_abs_diff,
            report.shape_rust,
            report.shape_ref
        );
    }
}

/// A text-only forward must skip the cross-attention layers rather than run them
/// against zeros — and must therefore differ from the image-conditioned logits.
#[test]
fn text_only_forward_skips_cross_attention() {
    let reference = load_shard(&fixture_path(REFERENCE));
    let weights = load_shard(&fixture_path(WEIGHTS));

    let mut model = MllamaForConditionalGeneration::new(fixture_config()).expect("model builds");
    load_mllama_weights(&mut model, &weights).expect("weight loader runs");

    let input_ids = ref_tensor(&reference, "input_ids").clone();
    let logits = model
        .forward(&input_ids, None, None)
        .expect("text-only forward runs");

    let with_images = ref_tensor(&reference, "logits");
    assert_eq!(logits.shape(), with_images.shape());

    let report = ParityReport::compute("text_only_vs_conditioned", &logits, with_images, TOL);
    assert!(
        !report.passed(),
        "text-only logits matched the image-conditioned ones to {:.3e} — the \
         cross-attention layer is contributing nothing, so the fixture is not \
         actually testing image conditioning",
        report.max_abs_diff
    );

    let mut logits = logits;
    logits.eval();
    let n = logits.size();
    assert!(
        logits.to_f32_vec(n).unwrap().iter().all(|x| x.is_finite()),
        "text-only logits are not finite"
    );
}

/// End-to-end load of a real released checkpoint, gated behind
/// `PMETAL_MLLAMA_CHECKPOINT`.
///
/// The synthetic fixture above proves the *math*, but it is generated by the
/// installed `transformers`, which nests both towers under a `MllamaModel`
/// called `model`. The released weights predate that and use
/// `vision_model.` / `language_model.model.` — a layout no synthetic fixture
/// exercises. This checks the loader against the real thing.
///
/// To run:
///
/// ```text
/// PMETAL_MLLAMA_CHECKPOINT=/path/to/Llama-3.2-11B-Vision-Instruct \
///     cargo test -p pmetal-models --test mllama_parity -- --ignored --nocapture
/// ```
#[test]
#[ignore = "needs a ~20 GB Llama 3.2 Vision checkpoint; set PMETAL_MLLAMA_CHECKPOINT"]
fn real_checkpoint_loads_and_runs() {
    use pmetal_models::dispatcher::{DynamicModel, ModelArchitecture};

    let dir = match std::env::var("PMETAL_MLLAMA_CHECKPOINT") {
        Ok(v) => std::path::PathBuf::from(v),
        Err(_) => {
            eprintln!("PMETAL_MLLAMA_CHECKPOINT not set — skipping");
            return;
        }
    };
    assert!(
        dir.exists(),
        "PMETAL_MLLAMA_CHECKPOINT points at {dir:?}, which does not exist"
    );

    // MLX ops that throw route through a thread-local error channel and return
    // a placeholder array rather than unwinding. A degenerate placeholder then
    // *broadcasts* through the residual stream, so shape and finiteness checks
    // alone will happily pass a forward in which attention computed nothing —
    // which is exactly what happened before the mask-dtype fix. Drain the
    // channel after every stage.
    fn assert_no_bridge_error(stage: &str) {
        if let Err(e) = pmetal_bridge::check_last_error() {
            panic!("{stage} raised a bridge exception: {e}");
        }
    }

    let mut model = DynamicModel::load(&dir).expect("real checkpoint loads");
    assert_no_bridge_error("loading");
    assert_eq!(
        model.architecture(),
        ModelArchitecture::Mllama,
        "architecture detection picked the wrong arch — `mllama` must be matched \
         before the generic `llama` arms"
    );
    assert_eq!(model.vocab_size(), 128_256);
    assert_eq!(model.hidden_size(), 4096);

    // A short text-only prompt. Degenerate weights (a silently-skipped tensor
    // left at random init) show up here as non-finite or absurd logits.
    let input_ids =
        pmetal_bridge::compat::Array::from_slice(&[128_000_i32, 9906, 11, 1917, 0], &[1, 5]);
    let mut logits = model.forward(&input_ids, None).expect("text forward runs");
    logits.eval();
    assert_no_bridge_error("the text forward");
    assert_eq!(logits.shape(), &[1, 5, 128_256]);

    let values = logits.to_f32_vec(logits.size()).expect("logits readable");
    assert!(
        values.iter().all(|x| x.is_finite()),
        "real logits are not finite"
    );
    let peak = values.iter().fold(0.0_f32, |a, &b| a.max(b.abs()));
    assert!(
        (1.0..100.0).contains(&peak),
        "peak |logit| {peak} is outside the range a correctly-loaded Llama sits in; \
         a tensor was probably left at random init"
    );

    // And the vision tower at its real geometry, on the processor's layout.
    let mllama = model.as_mllama_mut().expect("Mllama variant");
    let vc = mllama.config.vision_config.clone();
    assert_eq!(vc.image_size, 560, "released checkpoints use 560px tiles");
    assert_eq!(vc.vision_output_dim, 7680);

    let pixels = pmetal_bridge::compat::random::normal(
        &[1, 1, vc.max_num_tiles, 3, vc.image_size, vc.image_size],
        pmetal_bridge::compat::Dtype::Float32,
    );
    let ids = pmetal_bridge::compat::Array::from_slice(&[2_i32], &[1, 1]);
    let mask =
        pmetal_bridge::compat::Array::from_slice(&[1_i32, 1, 0, 0], &[1, 1, vc.max_num_tiles]);
    let mut features = mllama
        .vision_model
        .forward(&pixels, &ids, &mask)
        .expect("real vision tower runs");
    features.eval();
    assert_no_bridge_error("the vision tower");
    assert_eq!(
        features.shape(),
        &[
            1,
            1,
            vc.max_num_tiles,
            vc.num_patches(),
            vc.vision_output_dim
        ]
    );
    let features = features
        .to_f32_vec(features.size())
        .expect("features readable");
    assert!(
        features.iter().all(|x| x.is_finite()),
        "real vision features are not finite"
    );
}
