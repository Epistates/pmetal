//! `pmetal fuse` folds per-expert adapters into a mixture-of-experts model's
//! weights, in whichever layout the base checkpoint stores its experts.
//!
//! The reference is the adapted model's own forward: the fused checkpoint,
//! loaded as a plain model, has to compute what the base plus its adapters
//! computed. A checkpoint holds experts one tensor each
//! (`mlp.experts.{e}.{proj}.weight`, the released layout), fused into two
//! (`mlp.experts.gate_up_proj` and `mlp.experts.down_proj`, transformers 5), or
//! stacked (`mlp.switch_mlp.{proj}.weight`, an MLX conversion).

#![cfg(all(feature = "lora", feature = "merge"))]

use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

use pmetal::lora::{AdaptedModel, save_safetensors_map};
use pmetal::merge::{AccurateMergeConfig, streaming_lora_merge};
use pmetal::models::dispatcher::DynamicModel;
use pmetal::runtime::compat::{Array, Dtype, ModuleParametersExt, eval, ops, random};
use pmetal_core::LoraConfig;

const CONFIG: &str = r#"{
    "model_type": "qwen3_moe",
    "vocab_size": 256,
    "hidden_size": 64,
    "intermediate_size": 128,
    "moe_intermediate_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "decoder_sparse_step": 1,
    "max_position_embeddings": 512,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "tie_word_embeddings": false
}"#;

const EXPERTS: i32 = 4;

#[derive(Clone, Copy, Debug)]
enum Layout {
    Stacked,
    PerExpert,
    Fused,
}

fn drain() {
    pmetal::runtime::check_unobserved_error().expect("a runtime op threw");
}

fn input_ids() -> Array {
    let ids: Vec<i32> = (0..24).map(|i| (i * 7 + 3) % 256).collect();
    Array::from_slice(&ids, &[1, 24])
}

fn max_abs_diff(a: &Array, b: &Array) -> f32 {
    let diff = a.subtract(b).abs().max(None);
    eval([&diff]).unwrap();
    diff.item_f32()
}

/// The base model's tensors in the stacked layout the loader holds them in,
/// which is what `flatten_params` names them.
fn base_tensors() -> HashMap<String, Array> {
    random::seed(61);
    let model = DynamicModel::from_config(CONFIG).expect("build");
    let params = model.flatten_params();
    eval(params.values()).unwrap();
    params
        .into_iter()
        .map(|(name, value)| (name.to_string(), value))
        .collect()
}

/// The same tensors as a checkpoint of `layout` stores them.
fn in_layout(stacked: &HashMap<String, Array>, layout: Layout) -> HashMap<Rc<str>, Array> {
    let mut out: HashMap<Rc<str>, Array> = HashMap::new();
    let mut put = |key: String, value: Array| {
        out.insert(Rc::from(key.as_str()), value);
    };
    for (key, value) in stacked {
        let Some((layer, projection)) = key
            .split_once(".mlp.switch_mlp.")
            .and_then(|(layer, rest)| Some((layer, rest.strip_suffix(".weight")?)))
        else {
            put(key.clone(), value.clone());
            continue;
        };
        match layout {
            Layout::Stacked => put(key.clone(), value.clone()),
            Layout::PerExpert => {
                for expert in 0..EXPERTS {
                    let one = ops::slice_axis(value, 0, expert, expert + 1);
                    let matrix = one.reshape(&[value.dim(1), value.dim(2)]);
                    put(
                        format!("{layer}.mlp.experts.{expert}.{projection}.weight"),
                        matrix,
                    );
                }
            }
            Layout::Fused => match projection {
                "down_proj" => put(format!("{layer}.mlp.experts.down_proj"), value.clone()),
                "gate_proj" => {
                    let up = &stacked[&format!("{layer}.mlp.switch_mlp.up_proj.weight")];
                    put(
                        format!("{layer}.mlp.experts.gate_up_proj"),
                        ops::concatenate_axis(&[value, up], 1),
                    );
                }
                _ => {}
            },
        }
    }
    out
}

fn write_checkpoint(dir: &Path, tensors: &HashMap<Rc<str>, Array>) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("config.json"), CONFIG).unwrap();
    save_safetensors_map(dir.join("model.safetensors"), tensors).unwrap();
}

/// The adapted model's logits, and its adapters saved in `dir`.
fn adapted_logits(dir: &Path) -> Array {
    random::seed(61);
    let base = DynamicModel::from_config(CONFIG).expect("build");
    let config = LoraConfig {
        r: 8,
        alpha: 16.0,
        ..Default::default()
    };
    let mut model = AdaptedModel::attach(base, config).expect("attach");
    random::seed(9);
    let mut params: Vec<_> = model.lora_parameters().into_iter().collect();
    params.sort_by(|a, b| a.0.cmp(&b.0));
    let set: HashMap<Rc<str>, Array> = params
        .into_iter()
        .map(|(name, p)| {
            (
                name,
                random::uniform_range(-0.1, 0.1, p.shape(), Dtype::Float32),
            )
        })
        .collect();
    model.set_lora_parameters(&set);
    let logits = model.forward(&input_ids(), None).unwrap();
    eval([&logits]).unwrap();
    model.save_adapter(dir, None).unwrap();
    logits
}

#[test]
fn expert_adapters_fold_into_every_layout_a_checkpoint_stores_experts_in() {
    let root = tempfile::tempdir().unwrap();
    let adapter = root.path().join("adapter");
    let want = adapted_logits(&adapter);
    let stacked = base_tensors();

    // Folding changed the model: the base alone computes something else.
    random::seed(61);
    let mut base_only = DynamicModel::from_config(CONFIG).unwrap();
    let unadapted = base_only.forward(&input_ids(), None).unwrap();
    assert!(
        max_abs_diff(&unadapted, &want) > 1e-3,
        "the adapters changed nothing"
    );

    for layout in [Layout::Stacked, Layout::PerExpert, Layout::Fused] {
        let base = root.path().join(format!("base-{layout:?}"));
        write_checkpoint(&base, &in_layout(&stacked, layout));
        let fused = root.path().join(format!("fused-{layout:?}"));
        let stats = streaming_lora_merge(&AccurateMergeConfig::new(&base, &adapter, &fused))
            .unwrap_or_else(|e| panic!("{layout:?}: {e}"));
        assert!(stats.tensors_merged > 0, "{layout:?}");

        let mut loaded = DynamicModel::load(&fused).unwrap_or_else(|e| panic!("{layout:?}: {e}"));
        let got = loaded.forward(&input_ids(), None).unwrap();
        let diff = max_abs_diff(&got, &want);
        assert!(
            diff < 5e-4,
            "{layout:?}: the fused model differs from the adapted one by {diff}"
        );
        drain();
    }
}
