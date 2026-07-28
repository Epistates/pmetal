//! Every architecture applies the GELU variant its reference config names.
//!
//! HuggingFace's `ACT2FN` spells three different functions with confusingly
//! similar names — exact erf (`"gelu"`), the tanh approximation (`"gelu_new"`,
//! `"gelu_pytorch_tanh"`) and the sigmoid fast-approx (`"quick_gelu"`). They
//! differ by up to ~1.9e-2, against forward passes that are expected to agree
//! with the reference to ~1e-6, so substituting one for another is a
//! correctness bug that no shape or finiteness assertion can see.
//!
//! Each test below drives a real module with identity weights, so the module's
//! output *is* its activation applied to the input, and compares against the
//! function the reference would have picked. Bit-exact equality is the right
//! bar here: both sides run the same kernel on the same input, so any
//! difference means a different function was chosen.

use pmetal_bridge::compat::{Array, Dtype, Param, nn, ops};
use pmetal_models::architectures::bert::{BertConfig, BertIntermediate};
use pmetal_models::architectures::clip::{CLIPConfig, CLIPMLP};
use pmetal_models::architectures::phi::{PhiActivation, PhiConfig, PhiMLP};
use pmetal_models::architectures::t5::{T5Config, T5DenseGatedActDense};
use pmetal_models::architectures::utils::Activation;

/// Width of the square identity projections used throughout.
const N: usize = 8;

/// Probe points chosen where the three GELUs disagree most.
const PROBES: [f32; N] = [-4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 7.0];

fn probe_input() -> Array {
    Array::from_f32_slice(&PROBES, &[1, N as i32])
}

fn values(mut a: Array) -> Vec<f32> {
    a.eval();
    a.to_f32_vec(N).expect("to_f32_vec")
}

/// Turn a linear layer into a no-op so a module's output is exactly its
/// activation applied to the input.
fn make_identity(linear: &mut nn::Linear) {
    linear.weight = Param::new(ops::eye(N as i32, Dtype::Float32));
    if linear.bias.value.is_some() {
        linear.bias = Param::new(Some(Array::zeros(&[N as i32], Dtype::Float32.as_i32())));
    }
}

/// What the reference activation produces on the probe points.
fn expected(act: Activation) -> Vec<f32> {
    values(act(&probe_input()))
}

#[test]
fn bert_intermediate_applies_the_activation_its_config_names() {
    // BERT's default `hidden_act` is `"gelu"` — `GELUActivation`, the exact erf
    // definition. This used to fall through a `_ => nn::gelu(&h)` catch-all to
    // the sigmoid approximation, which also silently swallowed every
    // unrecognized name.
    for (hidden_act, want) in [
        ("gelu", nn::gelu_erf as Activation),
        ("gelu_new", nn::gelu_tanh_approximate),
        ("quick_gelu", nn::gelu),
        ("relu", nn::relu),
    ] {
        let config = BertConfig {
            hidden_size: N,
            intermediate_size: N,
            hidden_act: hidden_act.to_string(),
            ..Default::default()
        };
        let mut intermediate = BertIntermediate::new(&config).expect("BertIntermediate::new");
        make_identity(&mut intermediate.dense);

        assert_eq!(
            values(intermediate.forward(&probe_input()).expect("forward")),
            expected(want),
            "bert hidden_act {hidden_act:?} applied the wrong function"
        );
    }
}

#[test]
fn bert_rejects_an_activation_it_does_not_implement() {
    let config = BertConfig {
        hidden_act: "mish".to_string(),
        ..Default::default()
    };
    let err = BertIntermediate::new(&config).expect_err("mish is not implemented");
    assert!(
        format!("{err:?}").contains("mish"),
        "error should name the unsupported activation, got: {err:?}"
    );
}

#[test]
fn clip_mlp_distinguishes_all_three_gelus() {
    // OpenAI's CLIP is `"quick_gelu"`, but LAION re-trains ship `"gelu"` and
    // some variants `"gelu_new"`. This was a `use_quick_gelu: bool`, so both
    // non-quick spellings collapsed onto the same (wrong) function.
    for (hidden_act, want) in [
        ("quick_gelu", nn::gelu as Activation),
        ("gelu", nn::gelu_erf),
        ("gelu_new", nn::gelu_tanh_approximate),
    ] {
        let config = CLIPConfig {
            embed_dim: N,
            intermediate_size: N,
            hidden_act: hidden_act.to_string(),
            ..Default::default()
        };
        let mut mlp = CLIPMLP::new(&config);
        make_identity(&mut mlp.fc1);
        make_identity(&mut mlp.fc2);

        assert_eq!(
            values(mlp.forward(&probe_input()).expect("forward")),
            expected(want),
            "clip hidden_act {hidden_act:?} applied the wrong function"
        );
    }
}

#[test]
fn t5_gated_dense_uses_the_tanh_gelu() {
    // `T5Config.__init__` maps `feed_forward_proj: "gated-gelu"` to
    // `dense_act_fn: "gelu_new"` — the tanh approximation, NOT the exact erf
    // GELU a bare `"gelu"` would mean. T5-XXL (FLUX's text encoder) is
    // `gated-gelu`, so this is the path that actually runs.
    let config = T5Config {
        d_model: N,
        d_ff: N,
        ..Default::default()
    };
    assert_eq!(config.dense_act_fn(), "gelu_new");

    let mut dense = T5DenseGatedActDense::new(&config);
    make_identity(&mut dense.wi_0);
    make_identity(&mut dense.wi_1);
    make_identity(&mut dense.wo);

    // With identity projections the block reduces to `act(x) * x`.
    let want: Vec<f32> = expected(nn::gelu_tanh_approximate)
        .iter()
        .zip(PROBES)
        .map(|(a, x)| a * x)
        .collect();
    assert_eq!(
        values(dense.forward(&probe_input()).expect("forward")),
        want,
        "t5 gated-gelu applied the wrong function"
    );
}

#[test]
fn t5_derives_dense_act_fn_the_way_the_reference_does() {
    // The reference splits on `-` and takes the last segment, with `gated-gelu`
    // special-cased to `gelu_new`.
    for (feed_forward_proj, want) in [
        ("gated-gelu", "gelu_new"),
        ("relu", "relu"),
        ("gated-silu", "silu"),
        ("gelu", "gelu"),
    ] {
        let config = T5Config {
            feed_forward_proj: feed_forward_proj.to_string(),
            ..Default::default()
        };
        assert_eq!(
            config.dense_act_fn(),
            want,
            "feed_forward_proj {feed_forward_proj:?} resolved wrongly"
        );
    }
}

#[test]
fn phi_mlp_distinguishes_its_two_gelus() {
    // `PhiActivation::GeluExact` used to map to the sigmoid approximation,
    // contradicting its own name, and `GeluApprox` mapped there too — so the
    // two variants were indistinguishable.
    for (activation, want) in [
        (PhiActivation::GeluExact, nn::gelu_erf as Activation),
        (PhiActivation::GeluApprox, nn::gelu_tanh_approximate),
    ] {
        let config = PhiConfig {
            hidden_size: N as i32,
            intermediate_size: N as i32,
            hidden_act: activation,
            ..Default::default()
        };
        let mut mlp = PhiMLP::new(&config).expect("PhiMLP::new");
        make_identity(&mut mlp.gate_up_proj);
        make_identity(&mut mlp.down_proj);

        assert_eq!(
            values(mlp.forward(&probe_input()).expect("forward")),
            expected(want),
            "phi {activation:?} applied the wrong function"
        );
    }
}

#[test]
fn phi_deserializes_the_activation_spelling_real_checkpoints_use() {
    // Phi-2's released `config.json` says `"gelu_new"`. Before that was
    // aliased, the enum only accepted `"gelu_approx"` — a spelling no
    // HuggingFace config uses — so loading a stock Phi-2 config failed on an
    // unknown variant.
    for spelling in ["gelu_new", "gelu_approx", "gelu_pytorch_tanh", "gelu_fast"] {
        let parsed: PhiActivation = serde_json::from_str(&format!("\"{spelling}\""))
            .unwrap_or_else(|e| panic!("{spelling:?} should deserialize: {e}"));
        assert_eq!(parsed, PhiActivation::GeluApprox);
    }

    let exact: PhiActivation = serde_json::from_str("\"gelu\"").expect("gelu should deserialize");
    assert_eq!(exact, PhiActivation::GeluExact);
}
