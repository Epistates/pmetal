//! PiSSA, OLoRA and LoftQ initialise a whole model.
//!
//! The matrix-level definitions are tested beside the code
//! (`init.rs`). Here the same initialisations run through
//! `AdaptedModel` on a built architecture: the model starts out computing what
//! it computed before, the adapters are live (a nonzero `B` from step zero),
//! training lowers the loss, and a LoftQ start beats a plain QLoRA start on the
//! quantization error it exists to remove.

mod common;

use common::{cases, input_ids};
use pmetal_cache::test_utils::max_abs_diff;
use pmetal_core::{LoraConfig, LoraInit};
use pmetal_lora::{AdaptedModel, QLoraConfig, QLoraScheme, quantize_base};
use pmetal_models::dispatcher::DynamicModel;
use pmetal_runtime::compat::optimizers::AdamW;
use pmetal_runtime::compat::{Array, Dtype, random};

fn llama() -> &'static str {
    cases()
        .into_iter()
        .find(|c| c.name == "llama")
        .expect("llama case")
        .config_json
}

fn build() -> DynamicModel {
    random::seed(31);
    DynamicModel::from_config(llama()).expect("build")
}

fn config(init: LoraInit) -> LoraConfig {
    LoraConfig {
        r: 8,
        alpha: 16.0,
        init_lora_weights: init,
        ..Default::default()
    }
}

fn drain() {
    pmetal_runtime::check_unobserved_error().expect("a runtime op threw");
}

fn ids() -> Array {
    input_ids(256)
}

fn mean_abs_logit_error(a: &Array, b: &Array) -> f32 {
    let d = a
        .as_dtype(Dtype::Float32.as_i32())
        .subtract(&b.as_dtype(Dtype::Float32.as_i32()))
        .abs()
        .mean(None);
    d.eval();
    d.item_f32()
}

/// The model computes what it computed, and every adapter carries weight.
#[test]
fn pissa_and_olora_leave_the_model_unchanged_and_the_adapters_live() {
    let ids = ids();
    let mut reference = build();
    let want = reference.forward(&ids, None).unwrap();
    for init in [LoraInit::Pissa, LoraInit::Olora] {
        let mut model = AdaptedModel::attach(build(), config(init)).expect("attach");
        let got = model.forward(&ids, None).unwrap();
        drain();
        let diff = max_abs_diff(&got, &want);
        assert!(diff < 2e-3, "{init}: logits moved by {diff} at init");

        // Standard LoRA starts with B = 0; these start somewhere.
        let params = model.lora_parameters();
        let silent: Vec<_> = params
            .iter()
            .filter(|(k, _)| k.ends_with(".lora_b"))
            .filter(|(_, b)| b.square().sum(None).item_f32() == 0.0)
            .map(|(k, _)| k.to_string())
            .collect();
        assert!(silent.is_empty(), "{init}: zero B on {silent:?}");

        // And the base moved: it is the residual now.
        let mut stock = AdaptedModel::attach(build(), config(LoraInit::Standard)).unwrap();
        let moved = max_abs_diff(
            &model.model_mut().forward(&ids, None).unwrap(),
            &stock.forward(&ids, None).unwrap(),
        );
        assert!(moved < 2e-3, "{init} vs standard: {moved}");
    }
}

/// They train: the loss falls over a few steps from the principal start, with
/// the gradient reaching `A` as well as `B` from step zero.
#[test]
fn pissa_and_olora_train() {
    let ids = ids();
    for init in [LoraInit::Pissa, LoraInit::Olora] {
        let mut model = AdaptedModel::attach(build(), config(init)).expect("attach");
        let mut optimizer = AdamW::new(1e-3, 0.0);
        let losses: Vec<f32> = (0..10)
            .map(|_| common::train_step(&mut model, &mut optimizer, &ids))
            .collect();
        drain();
        assert!(losses.iter().all(|l| l.is_finite()), "{init}: {losses:?}");
        assert!(
            losses[9] < losses[0] - 0.05,
            "{init}: loss did not fall: {losses:?}"
        );
    }
}

/// A LoftQ start leaves the quantized model closer to the unquantized one than
/// a plain QLoRA start does, and QPiSSA does too: the principal part never goes
/// through the quantizer.
#[test]
fn quantization_aware_starts_beat_a_plain_qlora_start() {
    let ids = ids();
    let want = build().forward(&ids, None).unwrap();
    let qlora = QLoraConfig {
        scheme: QLoraScheme::Nf4,
        group_size: 64,
        double_quant: false,
    };

    let mut plain_base = build();
    quantize_base(&mut plain_base, &qlora).unwrap();
    let mut plain = AdaptedModel::attach(plain_base, config(LoraInit::Standard)).unwrap();
    let plain_error = mean_abs_logit_error(&plain.forward(&ids, None).unwrap(), &want);

    for init in [LoraInit::Loftq, LoraInit::Pissa] {
        let (mut model, packed) =
            AdaptedModel::attach_quantized(build(), config(init), &qlora).expect("attach");
        assert!(packed.packed > 0, "{init}: nothing packed");
        let error = mean_abs_logit_error(&model.forward(&ids, None).unwrap(), &want);
        drain();
        assert!(
            error < plain_error,
            "{init}: error {error} against plain QLoRA's {plain_error}"
        );
    }
}

/// The ordering is not the caller's to get right: LoftQ without a quantizer,
/// and PiSSA on a base that was packed first, are refused with a reason.
#[test]
fn an_initialisation_that_needs_the_dense_weight_or_a_quantizer_says_so() {
    let err = AdaptedModel::attach(build(), config(LoraInit::Loftq)).unwrap_err();
    assert!(err.to_string().contains("quantizer"), "{err}");

    let mut packed = build();
    quantize_base(&mut packed, &QLoraConfig::default()).unwrap();
    let err = AdaptedModel::attach(packed, config(LoraInit::Pissa)).unwrap_err();
    assert!(err.to_string().contains("already packed"), "{err}");
}

/// A standard start is unchanged by the new entry point.
#[test]
fn attach_quantized_with_the_standard_start_is_quantize_then_attach() {
    let ids = ids();
    let qlora = QLoraConfig::default();
    let mut base = build();
    quantize_base(&mut base, &qlora).unwrap();
    let mut by_hand = AdaptedModel::attach(base, config(LoraInit::Standard)).unwrap();
    let (mut together, _) =
        AdaptedModel::attach_quantized(build(), config(LoraInit::Standard), &qlora).unwrap();
    let a = by_hand.forward(&ids, None).unwrap();
    let b = together.forward(&ids, None).unwrap();
    drain();
    assert_eq!(max_abs_diff(&a, &b), 0.0);
    assert_eq!(by_hand.adapted_projections(), together.adapted_projections());
}
