//! LoRA on the encoder models that produce embeddings (BERT-family and
//! ModernBERT).
//!
//! An encoder is not a causal language model, but it is a `DynamicModel` of
//! `Linear`s like any other, so `AdaptedModel` adapts it, and its `forward`
//! returns the hidden states the pooling step reads, with the attention mask
//! an encoder takes (`[batch, length]`, non-zero for real tokens). These tests
//! hold it to the same contract as the decoders: a fresh adapter changes
//! nothing, the gradient reaches every adapter and matches finite differences,
//! folding the adapters in gives the adapted output, padding stays masked, and
//! a contrastive objective lowers.

use std::collections::HashMap;
use std::rc::Rc;

use pmetal_cache::test_utils::max_abs_diff;
use pmetal_core::LoraConfig;
use pmetal_lora::AdaptedModel;
use pmetal_models::dispatcher::DynamicModel;
use pmetal_runtime::compat::{Array, Dtype, random};
use pmetal_runtime::inline_array::value_and_grad;

const BERT: &str = r#"{"model_type": "bert", "vocab_size": 120, "hidden_size": 32,
    "intermediate_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
    "max_position_embeddings": 64, "type_vocab_size": 2}"#;

const MODERNBERT: &str = r#"{"model_type": "modernbert", "vocab_size": 120, "hidden_size": 32,
    "intermediate_size": 48, "num_hidden_layers": 3, "num_attention_heads": 4,
    "local_attention": 8}"#;

const SEQ: i32 = 12;

fn build(config: &str) -> DynamicModel {
    random::seed(81);
    DynamicModel::from_config(config).expect("build")
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

fn raw_ids() -> Vec<i32> {
    (0..2 * SEQ).map(|i| (i * 7 + 3) % 120).collect()
}

/// Two sequences, the second padded after `valid` tokens.
fn batch(valid: i32) -> (Array, Array) {
    let ids = raw_ids();
    let mask: Vec<i32> = (0..2 * SEQ)
        .map(|i| i32::from(i < SEQ || i % SEQ < valid))
        .collect();
    (
        Array::from_slice(&ids, &[2, SEQ]),
        Array::from_slice(&mask, &[2, SEQ]),
    )
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
                random::uniform_range(-0.1, 0.1, p.shape(), Dtype::Float32),
            )
        })
        .collect();
    model.set_lora_parameters(&set);
}

/// Mean-pooled, unit-length embeddings of the real tokens.
fn embed(hidden: &Array, mask: &Array) -> Array {
    let m = mask.as_dtype(Dtype::Float32.as_i32()).expand_dims(-1);
    let sum = hidden
        .as_dtype(Dtype::Float32.as_i32())
        .multiply(&m)
        .sum_axis(1, false);
    let mean = sum.divide(&m.sum_axis(1, false));
    let norm = mean.square().sum_axis(-1, true).sqrt();
    mean.divide(&norm)
}

#[test]
fn encoders_take_adapters_and_start_unchanged() {
    for (name, config) in [("bert", BERT), ("modernbert", MODERNBERT)] {
        let (ids, mask) = batch(7);
        let want = build(config).forward(&ids, Some(&mask)).unwrap();
        let mut model = AdaptedModel::attach(build(config), lora()).expect(name);
        // Attention and feed-forward projections both.
        let adapted = model.adapted_projections();
        assert!(adapted.len() >= 6, "{name}: {adapted:?}");
        let got = model.forward(&ids, Some(&mask)).unwrap();
        assert_eq!(got.shape(), want.shape(), "{name}");
        assert!(
            max_abs_diff(&got, &want) < 1e-6,
            "{name}: fresh adapters moved the output"
        );
        drain();
    }
}

/// Every adapter moves the output, and padding stays masked: changing a
/// padded token changes nothing a real token sees.
#[test]
fn encoder_adapters_reach_the_output_and_padding_stays_masked() {
    for (name, config) in [("bert", BERT), ("modernbert", MODERNBERT)] {
        let (ids, mask) = batch(7);
        let mut model = AdaptedModel::attach(build(config), lora()).unwrap();
        randomize(&mut model);
        let hidden = model.forward(&ids, Some(&mask)).unwrap();
        let reference = embed(&hidden, &mask);

        // Each adapter on its own: nudge B and the embedding must move.
        let params = model.lora_parameters();
        for (key, b) in params.iter().filter(|(k, _)| k.ends_with(".lora_b")) {
            let nudged = HashMap::from([(
                key.clone(),
                b.add(&random::uniform_range(0.2, 0.4, b.shape(), Dtype::Float32)),
            )]);
            model.set_lora_parameters(&nudged);
            let moved = max_abs_diff(
                &embed(&model.forward(&ids, Some(&mask)).unwrap(), &mask),
                &reference,
            );
            model.set_lora_parameters(&HashMap::from([(key.clone(), b.clone())]));
            assert!(moved.is_finite() && moved > 0.0, "{name}: {key} is silent");
        }

        // Padding: replace the padded tokens of row 1.
        let mut other = raw_ids();
        for j in 7..SEQ {
            other[(SEQ + j) as usize] = 99;
        }
        let other = Array::from_slice(&other, &[2, SEQ]);
        let again = embed(&model.forward(&other, Some(&mask)).unwrap(), &mask);
        assert!(
            max_abs_diff(&again, &reference) < 1e-5,
            "{name}: a padded token reached a real one"
        );
        drain();
    }
}

#[test]
fn encoder_adapters_merge_exactly() {
    for (name, config) in [("bert", BERT), ("modernbert", MODERNBERT)] {
        let (ids, mask) = batch(9);
        let mut model = AdaptedModel::attach(build(config), lora()).unwrap();
        randomize(&mut model);
        let adapted = model.forward(&ids, Some(&mask)).unwrap();
        model.merge().unwrap();
        let merged = model.forward(&ids, Some(&mask)).unwrap();
        assert!(max_abs_diff(&adapted, &merged) < 1e-4, "{name}: merge");
        model.fuse().unwrap();
        let fused = model.forward(&ids, Some(&mask)).unwrap();
        assert!(max_abs_diff(&adapted, &fused) < 1e-4, "{name}: fuse");
        drain();
    }
}

/// The gradient of a pooled-embedding objective with respect to every adapter,
/// against the central difference of the objective itself along the gradient.
#[test]
fn encoder_adapter_gradients_match_finite_differences() {
    const H: f32 = 1e-2;
    for (name, config) in [("bert", BERT), ("modernbert", MODERNBERT)] {
        let (ids, mask) = batch(8);
        let mut model = AdaptedModel::attach(build(config), lora()).unwrap();
        randomize(&mut model);
        let mut names: Vec<Rc<str>> = model.lora_parameters().into_keys().collect();
        names.sort();
        let live = model.lora_parameters();
        let params: Vec<Array> = names.iter().map(|n| live[n].clone()).collect();

        let weights = random::uniform_range(-1.0, 1.0, &[2, 32], Dtype::Float32);
        let objective = |model: &mut AdaptedModel| {
            let hidden = model.forward(&ids, Some(&mask)).unwrap();
            embed(&hidden, &mask).multiply(&weights).sum_all()
        };
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
        let base_loss = loss.item_f32();
        model.set_lora_parameters(&live);
        drain();
        assert!(base_loss.is_finite(), "{name}");

        let norm = |arrays: &[Array]| {
            arrays
                .iter()
                .map(|a| a.square().sum(None).item_f32())
                .sum::<f32>()
                .sqrt()
        };
        let g_norm = norm(&grads);
        assert!(g_norm > 1e-4, "{name}: the gradient is {g_norm}");
        // The step along the gradient's own direction: the loss changes at the
        // rate `|g|`.
        let mut shift = |sign: f32| {
            let set: HashMap<Rc<str>, Array> = names
                .iter()
                .zip(params.iter().zip(&grads))
                .map(|(n, (p, g))| (n.clone(), p.add(&g.mul_scalar(sign * H / g_norm))))
                .collect();
            model.set_lora_parameters(&set);
            let l = objective(&mut model).item_f32();
            model.set_lora_parameters(&live);
            l
        };
        let slope = (shift(1.0) - shift(-1.0)) / (2.0 * H);
        drain();
        assert!(
            (slope - g_norm).abs() / g_norm < 0.03,
            "{name}: autograd says {g_norm}, finite differences {slope}"
        );
    }
}

/// A contrastive objective (InfoNCE over in-batch negatives) lowers over a few
/// steps with the adapters alone trainable.
#[test]
fn encoder_adapters_train_on_a_contrastive_objective() {
    for (name, config) in [("bert", BERT), ("modernbert", MODERNBERT)] {
        let (ids_a, mask) = batch(SEQ);
        // The positive of each row is the same row with a few tokens changed.
        let mut b = raw_ids();
        for row in 0..2usize {
            for j in [2usize, 5, 9] {
                b[row * SEQ as usize + j] = (b[row * SEQ as usize + j] + 17) % 120;
            }
        }
        let ids_b = Array::from_slice(&b, &[2, SEQ]);

        let mut model = AdaptedModel::attach(build(config), lora()).unwrap();
        let mut optimizer = pmetal_runtime::compat::optimizers::AdamW::new(5e-3, 0.0);
        let mut losses = Vec::new();
        for _ in 0..12 {
            losses.push(step(&mut model, &mut optimizer, &ids_a, &ids_b, &mask));
        }
        drain();
        assert!(losses.iter().all(|l| l.is_finite()), "{name}: {losses:?}");
        assert!(
            losses[11] < losses[0] - 0.01,
            "{name}: the contrastive loss did not fall: {losses:?}"
        );
    }
}

fn step(
    model: &mut AdaptedModel,
    optimizer: &mut pmetal_runtime::compat::optimizers::AdamW,
    ids_a: &Array,
    ids_b: &Array,
    mask: &Array,
) -> f32 {
    use pmetal_runtime::compat::ModuleParametersExt;
    use pmetal_runtime::compat::optimizers::Optimizer;
    let mut names: Vec<Rc<str>> = model.flatten_trainable_params().into_keys().collect();
    names.sort();
    let live = model.flatten_trainable_params();
    let params: Vec<Array> = names.iter().map(|n| live[n].clone()).collect();
    drop(live);
    let (loss, grads) = value_and_grad(
        |arrays| {
            let set: HashMap<Rc<str>, Array> =
                names.iter().cloned().zip(arrays.iter().cloned()).collect();
            model.set_lora_parameters(&set);
            let ea = embed(&model.forward(ids_a, Some(mask)).unwrap(), mask);
            let eb = embed(&model.forward(ids_b, Some(mask)).unwrap(), mask);
            // InfoNCE: each row's positive is the matching row.
            // A soft temperature, so the objective starts with room to fall.
            let logits = ea.matmul(&eb.t()).mul_scalar(1.5);
            let log_probs = logits.log_softmax(-1);
            let diagonal = log_probs.multiply(&pmetal_runtime::compat::ops::eye(2, Dtype::Float32));
            diagonal.sum_all().mul_scalar(-0.5)
        },
        &params,
        &[],
    );
    optimizer.advance_step();
    let mut flat = model.flatten_params_mut();
    for (name, grad) in names.iter().zip(&grads) {
        let slot = flat.get_mut(name.as_ref()).expect("adapter parameter");
        optimizer.update_single(name, grad, slot).expect("update");
    }
    drop(flat);
    loss.item_f32()
}
