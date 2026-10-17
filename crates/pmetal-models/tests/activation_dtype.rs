//! A half-precision model's activations stay half precision.
//!
//! The GELUs in `pmetal_runtime::compat::nn` evaluate in f32 and return f32 for
//! a bf16 input. PyTorch (and so transformers) rounds the result back to the
//! input's dtype, so a gated MLP there is `down(round(gelu(gate)) * up)` in
//! bf16 throughout. An architecture that fed the f32 result straight on ran its
//! `[batch, seq, intermediate]` activation, and the backward pass's copy of it,
//! at twice the width, and (where a product followed) rounded once instead of
//! twice, a difference the reference does not have.

use pmetal_models::architectures::diffusion_gemma::{
    DiffusionGemmaSelfConditioning, DiffusionGemmaTextConfig,
};
use pmetal_models::architectures::gemma4_vision::{Gemma4VisionConfig, Gemma4VisionMLP};
use pmetal_models::architectures::utils::resolve_activation;
use pmetal_runtime::compat::{
    Array, Dtype, ModuleParameters, ModuleParametersExt, fast, nn, random,
};

fn bf16() -> i32 {
    Dtype::Bfloat16.as_i32()
}

/// Largest absolute difference, NaN-aware.
fn max_diff(a: &Array, b: &Array) -> f32 {
    let d = a
        .as_dtype(Dtype::Float32.as_i32())
        .subtract(&b.as_dtype(Dtype::Float32.as_i32()))
        .abs();
    let m = d.max(None);
    m.eval();
    let m = m.item_f32();
    assert!(!m.is_nan(), "difference is NaN");
    m
}

fn to_bf16(model: &mut impl ModuleParameters) {
    for (_, param) in model.flatten_params_mut() {
        *param = param.as_dtype(bf16());
    }
}

fn drain() {
    pmetal_runtime::check_unobserved_error().expect("a runtime op threw");
}

/// Every name `resolve_activation` knows returns the dtype it was given.
#[test]
fn resolved_activations_keep_the_input_dtype() {
    for name in [
        "gelu",
        "gelu_python",
        "gelu_new",
        "gelu_pytorch_tanh",
        "gelu_python_tanh",
        "gelu_fast",
        "gelu_accurate",
        "quick_gelu",
        "relu",
        "silu",
        "swish",
        "tanh",
        "sigmoid",
    ] {
        let act = resolve_activation(name).unwrap_or_else(|| panic!("{name} does not resolve"));
        for dtype in [Dtype::Bfloat16, Dtype::Float16, Dtype::Float32] {
            let x =
                Array::from_f32_slice(&[-2.0, -0.5, 0.5, 3.0], &[1, 4]).as_dtype(dtype.as_i32());
            let y = act(&x);
            assert_eq!(
                y.dtype(),
                dtype,
                "{name} on {dtype:?} returned {:?}",
                y.dtype()
            );
        }
    }
    drain();
}

/// Gemma 4's vision MLP is `down(round(gelu_tanh(gate)) * up)` in bf16, as the
/// reference runs it.
#[test]
fn gemma4_vision_mlp_rounds_the_activation_before_the_product() {
    random::seed(11);
    let config: Gemma4VisionConfig =
        serde_json::from_str(r#"{"hidden_size": 32, "intermediate_size": 96}"#).unwrap();
    let mut mlp = Gemma4VisionMLP::new(&config).unwrap();
    to_bf16(&mut mlp);
    let x = random::normal(&[2, 5, 32], Dtype::Float32).as_dtype(bf16());

    let got = mlp.forward(&x).unwrap();

    let gate = mlp.gate_proj.forward(&x);
    let up = mlp.up_proj.forward(&x);
    let rounded = nn::gelu_tanh_approximate(&gate).as_dtype(bf16());
    let want = mlp.down_proj.forward(&rounded.multiply(&up));
    assert_eq!(got.dtype(), Dtype::Bfloat16);
    assert_eq!(
        max_diff(&got, &want),
        0.0,
        "differs from the bf16 reference"
    );

    // The fixture can tell the two orders apart: the f32 stream differs.
    let f32_stream = mlp
        .down_proj
        .forward(&nn::gelu_tanh_approximate(&gate).multiply(&up));
    assert!(
        max_diff(&got, &f32_stream) > 0.0,
        "rounding the activation changed nothing, so this test proves nothing"
    );
    drain();
}

/// The same for DiffusionGemma's self-conditioning MLP.
#[test]
fn diffusion_gemma_self_conditioning_rounds_the_activation_before_the_product() {
    random::seed(12);
    let config = DiffusionGemmaTextConfig {
        hidden_size: 32,
        intermediate_size: 96,
        ..Default::default()
    };
    let mut sc = DiffusionGemmaSelfConditioning::new(&config).unwrap();
    to_bf16(&mut sc);
    let signal = random::normal(&[2, 5, 32], Dtype::Float32).as_dtype(bf16());
    let embeds = Array::zeros(&[2, 5, 32], bf16());

    let got = sc.forward(&embeds, &signal).unwrap();

    let normed = sc.pre_norm.forward(&signal);
    let gate = nn::gelu_tanh_approximate(&sc.gate_proj.forward(&normed)).as_dtype(bf16());
    let delta = sc
        .down_proj
        .forward(&gate.multiply(&sc.up_proj.forward(&normed)));
    let want = fast::rms_norm_opt(&embeds.add(&delta), None, sc.eps);
    assert_eq!(got.dtype(), Dtype::Bfloat16);
    assert_eq!(
        max_diff(&got, &want),
        0.0,
        "differs from the bf16 reference"
    );
    drain();
}
