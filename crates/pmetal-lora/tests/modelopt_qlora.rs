//! QLoRA on an NVIDIA ModelOpt NVFP4 checkpoint, with no dense copy of the
//! base ever made.
//!
//! The checkpoint's projections load packed (see
//! `pmetal-models/tests/modelopt_packed_linears.rs`); an adapter on one is
//! QLoRA by construction. This holds that path to the adapter contract: a
//! fresh adapter changes nothing, the gradient reaches `A` and `B` through the
//! quantized matmul and matches a finite difference, training lowers the
//! loss, and folding the adapters in dequantizes the layer and agrees.

mod common;

use std::collections::HashMap;
use std::rc::Rc;

use common::input_ids;
use pmetal_cache::test_utils::max_abs_diff;
use pmetal_core::LoraConfig;
use pmetal_lora::AdaptedModel;
use pmetal_models::dispatcher::DynamicModel;
use pmetal_runtime::QuantizedMode;
use pmetal_runtime::compat::{Array, Dtype, VisitLinears, random};
use pmetal_runtime::inline_array::value_and_grad;

const LLAMA: &str = r#"{
    "model_type": "llama", "vocab_size": 256, "hidden_size": 64, "intermediate_size": 128,
    "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
    "max_position_embeddings": 512, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
    "tie_word_embeddings": false
}"#;

/// A tiny Llama with every projection but the head written the way ModelOpt
/// writes NVFP4.
fn modelopt_llama() -> tempfile::TempDir {
    use pmetal_runtime::compat::ModuleParametersExt;
    random::seed(95);
    let mut model = DynamicModel::from_config(LLAMA).expect("build");
    let mut projections = HashMap::new();
    model.visit_linears_mut("", &mut |path, linear| {
        if path != "lm_head" {
            projections.insert(path.to_string(), linear.weight.value.clone());
        }
    });
    let mut out: Vec<(String, Array)> = Vec::new();
    for (name, value) in model.flatten_params() {
        let name = name.to_string();
        let module = name.strip_suffix(".weight");
        match module.and_then(|m| projections.get(m).map(|w| (m, w))) {
            Some((module, w)) => {
                let (packed, scales) = w.quantize_weights_mode(16, 4, QuantizedMode::Nvfp4);
                out.push((
                    format!("{module}.weight"),
                    packed.view(pmetal_runtime::dtype::U8),
                ));
                out.push((format!("{module}.weight_scale"), scales));
                out.push((
                    format!("{module}.weight_scale_2"),
                    Array::from_f32_slice(&[0.5], &[]),
                ));
            }
            None => out.push((name, value)),
        }
    }
    let dir = tempfile::tempdir().unwrap();
    let entries: Vec<(&str, &Array)> = out.iter().map(|(k, v)| (k.as_str(), v)).collect();
    Array::save_safetensors(
        dir.path().join("model.safetensors").to_str().unwrap(),
        &entries,
    )
    .unwrap();
    std::fs::write(dir.path().join("config.json"), LLAMA).unwrap();
    dir
}

fn lora() -> LoraConfig {
    LoraConfig {
        r: 4,
        alpha: 8.0,
        ..Default::default()
    }
}

fn drain() {
    pmetal_runtime::check_unobserved_error().expect("a runtime op threw");
}

fn randomize(model: &mut AdaptedModel) {
    random::seed(5);
    let mut params: Vec<_> = model.lora_parameters().into_iter().collect();
    params.sort_by(|a, b| a.0.cmp(&b.0));
    let set: HashMap<Rc<str>, Array> = params
        .into_iter()
        .map(|(k, p)| {
            (
                k,
                random::uniform_range(-0.08, 0.08, p.shape(), Dtype::Float32),
            )
        })
        .collect();
    model.set_lora_parameters(&set);
}

#[test]
fn adapters_on_a_packed_modelopt_base_are_qlora() {
    let dir = modelopt_llama();
    let ids = input_ids(256);

    let mut base = DynamicModel::load(dir.path()).unwrap();
    let want = base.forward(&ids, None).unwrap();
    let mut model = AdaptedModel::attach(DynamicModel::load(dir.path()).unwrap(), lora()).unwrap();
    let mut packed = 0;
    model.model_mut().visit_linears_mut("", &mut |_, linear| {
        packed += usize::from(linear.quant.is_some() && linear.is_adapted());
    });
    assert_eq!(
        packed,
        model.adapted_projections().len(),
        "adapters sit on packed layers"
    );
    assert_eq!(packed, 14);

    let fresh = model.forward(&ids, None).unwrap();
    assert!(
        max_abs_diff(&fresh, &want) < 1e-5,
        "fresh adapters moved the logits"
    );

    // The gradient goes through the quantized matmul to both factors, and
    // matches the central difference of the loss along itself.
    randomize(&mut model);
    let weights = random::uniform_range(-1.0, 1.0, &[1, 256], Dtype::Float32);
    let objective = |model: &mut AdaptedModel| {
        let logits = model.forward(&ids, None).unwrap();
        let last = logits.slice(&[0, 23, 0], &[1, 24, 256]).reshape(&[1, 256]);
        last.as_dtype(Dtype::Float32.as_i32())
            .multiply(&weights)
            .sum_all()
    };
    let mut names: Vec<Rc<str>> = model.lora_parameters().into_keys().collect();
    names.sort();
    let live = model.lora_parameters();
    let params: Vec<Array> = names.iter().map(|n| live[n].clone()).collect();
    let (loss, grads) = value_and_grad(
        |arrays| {
            let set: HashMap<Rc<str>, Array> =
                names.iter().cloned().zip(arrays.iter().cloned()).collect();
            model.set_lora_parameters(&set);
            objective(&mut model)
        },
        &params,
        &[],
    );
    loss.eval();
    for g in &grads {
        g.eval();
    }
    model.set_lora_parameters(&live);
    assert!(loss.item_f32().is_finite());
    let g_norm = grads
        .iter()
        .map(|g| g.square().sum(None).item_f32())
        .sum::<f32>()
        .sqrt();
    assert!(g_norm > 1e-4, "the gradient is {g_norm}");
    let h = 1e-2_f32;
    let mut shifted = |sign: f32| {
        let set: HashMap<Rc<str>, Array> = names
            .iter()
            .zip(params.iter().zip(&grads))
            .map(|(n, (p, g))| (n.clone(), p.add(&g.mul_scalar(sign * h / g_norm))))
            .collect();
        model.set_lora_parameters(&set);
        let l = objective(&mut model).item_f32();
        model.set_lora_parameters(&live);
        l
    };
    let slope = (shifted(1.0) - shifted(-1.0)) / (2.0 * h);
    assert!(
        (slope - g_norm).abs() / g_norm < 0.05,
        "autograd {g_norm}, finite differences {slope}"
    );

    // Folding in dequantizes each adapted layer and reproduces the output.
    let adapted = model.forward(&ids, None).unwrap();
    model.merge().unwrap();
    let merged = model.forward(&ids, None).unwrap();
    assert!(max_abs_diff(&adapted, &merged) < 2e-3, "merge");
    drain();
}

/// A released ModelOpt NVFP4 checkpoint (`PMETAL_MODELOPT_DIR`, for example
/// `nvidia/Qwen3.8-27B-NVFP4`) trains with its base packed: the loss over a
/// repeated sequence falls, and the memory it takes is the packed model's plus
/// activations.
#[test]
#[ignore = "needs a ModelOpt NVFP4 checkpoint on disk and a lot of memory"]
fn a_released_modelopt_checkpoint_trains() {
    use pmetal_lora::TrainableModel;
    let dir = std::env::var("PMETAL_MODELOPT_DIR").expect("PMETAL_MODELOPT_DIR");
    let model = DynamicModel::load(std::path::Path::new(&dir)).expect("load");
    let config = LoraConfig {
        r: 8,
        alpha: 16.0,
        ..Default::default()
    };
    let mut model = AdaptedModel::attach(model, config).expect("attach");
    if model.supports_gradient_checkpointing() {
        model.enable_gradient_checkpointing(1);
    }
    let ids: Vec<i32> = (0..48).map(|i| (i * 131 + 7) % 1000).collect();
    let ids = Array::from_slice(&ids, &[1, 48]);
    // LoRA's usual learning rate; a larger one overshoots on a 27B model.
    let mut optimizer = pmetal_runtime::compat::optimizers::AdamW::new(2e-4, 0.0);
    let losses: Vec<f32> = (0..10)
        .map(|_| common::train_step(&mut model, &mut optimizer, &ids))
        .collect();
    drain();
    println!(
        "{} adapted projections; losses {losses:?}; peak {:.2} GB",
        model.adapted_projections().len(),
        pmetal_runtime::inline_array::get_peak_memory() as f64 / 1e9
    );
    assert!(losses.iter().all(|l| l.is_finite()), "{losses:?}");
    assert!(losses[9] < losses[0] - 0.05, "{losses:?}");
}

#[test]
fn a_packed_modelopt_base_trains() {
    let dir = modelopt_llama();
    let ids = input_ids(256);
    let mut model = AdaptedModel::attach(DynamicModel::load(dir.path()).unwrap(), lora()).unwrap();
    let mut optimizer = pmetal_runtime::compat::optimizers::AdamW::new(1e-2, 0.0);
    let losses: Vec<f32> = (0..10)
        .map(|_| common::train_step(&mut model, &mut optimizer, &ids))
        .collect();
    drain();
    assert!(losses.iter().all(|l| l.is_finite()), "{losses:?}");
    assert!(losses[9] < losses[0] - 0.05, "{losses:?}");
}
