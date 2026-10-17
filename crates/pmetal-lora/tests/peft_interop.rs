//! Adapters in the PEFT layout: what is written, what is read, and that they
//! round-trip.
//!
//! The tests that need `peft` itself (an adapter it wrote, loaded here; one
//! written here, loaded there) are `#[ignore]`d and read a directory named by
//! `PMETAL_PEFT_ORACLE`, which a script builds with `transformers` and `peft`
//! (see `oracle_agrees_with_peft`).

mod common;

use std::collections::{BTreeSet, HashMap};
use std::path::Path;
use std::rc::Rc;

use common::{cases, input_ids};
use pmetal_cache::test_utils::max_abs_diff;
use pmetal_core::LoraConfig;
use pmetal_lora::peft::{self, ADAPTER_CONFIG, ADAPTER_WEIGHTS};
use pmetal_lora::{AdaptedModel, load_safetensors_map, save_safetensors_map};
use pmetal_models::dispatcher::DynamicModel;
use pmetal_runtime::compat::{Array, Dtype, random};

fn case(name: &str) -> &'static str {
    cases()
        .into_iter()
        .find(|c| c.name == name)
        .unwrap_or_else(|| panic!("no `{name}` case"))
        .config_json
}

fn build(config_json: &str) -> DynamicModel {
    random::seed(41);
    DynamicModel::from_config(config_json).expect("build")
}

fn lora(use_dora: bool, use_rslora: bool) -> LoraConfig {
    LoraConfig {
        r: 4,
        alpha: 12.0,
        use_dora,
        use_rslora,
        target_modules: vec!["all-linear".to_string()],
        ..Default::default()
    }
}

fn drain() {
    pmetal_runtime::check_unobserved_error().expect("a runtime op threw");
}

/// An adapted model whose adapters all carry weight.
fn trained(config_json: &str, config: LoraConfig) -> AdaptedModel {
    let mut model = AdaptedModel::attach(build(config_json), config).expect("attach");
    random::seed(5);
    let mut params: Vec<_> = model.lora_parameters().into_iter().collect();
    params.sort_by(|a, b| a.0.cmp(&b.0));
    let set: HashMap<Rc<str>, Array> = params
        .into_iter()
        .map(|(name, p)| {
            let fresh = if name.ends_with(".lora_magnitude") {
                // Keep the magnitude near its seed so the layer stays sane.
                p.multiply(&random::uniform_range(0.9, 1.1, p.shape(), Dtype::Float32))
            } else {
                random::uniform_range(-0.08, 0.08, p.shape(), Dtype::Float32)
            };
            (name, fresh)
        })
        .collect();
    model.set_lora_parameters(&set);
    model
}

fn same_adapters(a: &AdaptedModel, b: &AdaptedModel) {
    let (pa, pb) = (a.lora_parameters(), b.lora_parameters());
    assert_eq!(pa.len(), pb.len(), "different adapter sets");
    for (name, value) in &pa {
        let other = pb.get(name).unwrap_or_else(|| panic!("{name} missing"));
        assert_eq!(max_abs_diff(value, other), 0.0, "{name} differs");
    }
}

fn tensor_names(path: &Path) -> BTreeSet<String> {
    load_safetensors_map(path)
        .expect("readable")
        .into_keys()
        .collect()
}

/// An adapter directory holds exactly the tensors PEFT names, with PEFT's
/// shapes: `lora_A.weight` `[r, in]`, `lora_B.weight` `[out, r]`, and a DoRA
/// magnitude as one value per output row.
#[test]
fn a_saved_adapter_carries_peft_names_and_shapes() {
    let model = trained(case("llama"), lora(true, false));
    let dir = tempfile::tempdir().unwrap();
    model.save_adapter(dir.path(), Some("org/base")).unwrap();

    let tensors = load_safetensors_map(dir.path().join(ADAPTER_WEIGHTS)).unwrap();
    let q = "base_model.model.model.layers.0.self_attn.q_proj";
    let down = "base_model.model.model.layers.1.mlp.down_proj";
    // The tiny Llama: hidden 64, 4 heads of 16, 2 kv heads, intermediate 128.
    assert_eq!(tensors[&format!("{q}.lora_A.weight")].shape(), &[4, 64]);
    assert_eq!(tensors[&format!("{q}.lora_B.weight")].shape(), &[64, 4]);
    assert_eq!(tensors[&format!("{q}.lora_magnitude_vector")].shape(), &[64]);
    assert_eq!(tensors[&format!("{down}.lora_A.weight")].shape(), &[4, 128]);
    assert_eq!(tensors[&format!("{down}.lora_B.weight")].shape(), &[64, 4]);

    // Seven projections a layer, two layers, three tensors each; no head.
    assert_eq!(tensors.len(), 2 * 7 * 3);
    assert!(tensors.keys().all(|k| k.starts_with("base_model.model.")));
    assert!(tensors.keys().all(|k| !k.contains("lm_head")));
    drain();
}

/// `adapter_config.json` has what PEFT's `LoraConfig` reads, and lists the
/// modules by their full names, as PEFT does for `all-linear`.
#[test]
fn the_config_is_what_peft_reads() {
    let model = trained(case("llama"), lora(false, true));
    let dir = tempfile::tempdir().unwrap();
    model.save_adapter(dir.path(), Some("org/base")).unwrap();
    let config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.path().join(ADAPTER_CONFIG)).unwrap())
            .unwrap();
    assert_eq!(config["peft_type"], "LORA");
    assert_eq!(config["task_type"], "CAUSAL_LM");
    assert_eq!(config["r"], 4);
    assert_eq!(config["lora_alpha"], 12.0);
    assert_eq!(config["use_rslora"], true);
    assert_eq!(config["use_dora"], false);
    assert_eq!(config["bias"], "none");
    assert_eq!(config["base_model_name_or_path"], "org/base");
    let targets: Vec<&str> = config["target_modules"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert_eq!(targets.len(), 14);
    assert!(targets.contains(&"model.layers.0.self_attn.q_proj"));
    assert!(targets.contains(&"model.layers.1.mlp.up_proj"));
    assert!(!targets.contains(&"lm_head"));
}

/// Save, rebuild the model from the saved config, load: the same adapters and
/// the same logits. For plain LoRA, rsLoRA and DoRA.
#[test]
fn an_adapter_round_trips_through_its_directory() {
    let ids = input_ids(256);
    for (dora, rslora) in [(false, false), (false, true), (true, false)] {
        let mut model = trained(case("llama"), lora(dora, rslora));
        let want = model.forward(&ids, None).unwrap();
        let dir = tempfile::tempdir().unwrap();
        model.save_adapter(dir.path(), None).unwrap();

        let config_json: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dir.path().join(ADAPTER_CONFIG)).unwrap(),
        )
        .unwrap();
        let config = peft::lora_config_from_peft(&config_json).unwrap();
        let mut loaded = AdaptedModel::attach(build(case("llama")), config).unwrap();
        loaded.load_lora_weights(dir.path()).unwrap();
        let got = loaded.forward(&ids, None).unwrap();

        same_adapters(&model, &loaded);
        assert_eq!(max_abs_diff(&want, &got), 0.0, "dora={dora} rslora={rslora}");
        drain();
    }
}

/// Adapters written before the PEFT layout, `<module>.lora_a` in a file called
/// `lora_weights.safetensors`, still load, by file or by directory.
#[test]
fn the_earlier_layout_still_loads() {
    let model = trained(case("llama"), lora(true, false));
    let dir = tempfile::tempdir().unwrap();
    let legacy = dir.path().join(peft::LEGACY_ADAPTER_WEIGHTS);
    save_safetensors_map(&legacy, &model.lora_parameters()).unwrap();
    assert!(tensor_names(&legacy).contains("layers.0.self_attn.q_proj.lora_a"));

    for path in [legacy.as_path(), dir.path()] {
        let mut loaded = AdaptedModel::attach(build(case("llama")), lora(true, false)).unwrap();
        loaded.load_lora_weights(path).unwrap();
        same_adapters(&model, &loaded);
    }
}

/// Another rank, another module set, or a directory with no adapter are
/// refused with the reason, rather than loaded into a model that then serves
/// something other than what was trained.
#[test]
fn an_adapter_that_does_not_fit_is_refused_with_a_reason() {
    let model = trained(case("llama"), lora(false, false));
    let dir = tempfile::tempdir().unwrap();
    model.save_adapter(dir.path(), None).unwrap();

    let wide = LoraConfig {
        r: 8,
        ..lora(false, false)
    };
    let mut other_rank = AdaptedModel::attach(build(case("llama")), wide).unwrap();
    let err = other_rank.load_lora_weights(dir.path()).unwrap_err();
    assert!(err.to_string().contains("rank"), "{err}");

    let few = LoraConfig {
        target_modules: vec!["q_proj".to_string()],
        ..lora(false, false)
    };
    let mut fewer = AdaptedModel::attach(build(case("llama")), few).unwrap();
    let err = fewer.load_lora_weights(dir.path()).unwrap_err();
    assert!(err.to_string().contains("no adapter on"), "{err}");

    let empty = tempfile::tempdir().unwrap();
    let mut any = AdaptedModel::attach(build(case("llama")), lora(false, false)).unwrap();
    let err = any.load_lora_weights(empty.path()).unwrap_err();
    assert!(err.to_string().contains(ADAPTER_WEIGHTS), "{err}");
}

/// A multimodal wrapper names the text model `model.language_model.*`; an
/// adapter written against it loads onto the text model's modules.
#[test]
fn a_language_model_prefix_is_accepted() {
    let model = trained(case("llama"), lora(false, false));
    let dir = tempfile::tempdir().unwrap();
    model.save_adapter(dir.path(), None).unwrap();
    let weights = dir.path().join(ADAPTER_WEIGHTS);
    let renamed: HashMap<String, Array> = load_safetensors_map(&weights)
        .unwrap()
        .into_iter()
        .map(|(k, v)| {
            (
                k.replace("base_model.model.model.", "base_model.model.model.language_model."),
                v,
            )
        })
        .collect();
    save_safetensors_map(&weights, &renamed).unwrap();

    let mut loaded = AdaptedModel::attach(build(case("llama")), lora(false, false)).unwrap();
    loaded.load_lora_weights(dir.path()).unwrap();
    same_adapters(&model, &loaded);
}

/// Gemma holds its layers under `layers.gemma1.N` / `layers.gemma2.N`; the
/// checkpoint, and so PEFT, does not.
#[test]
fn gemma_variant_prefixes_do_not_reach_the_file() {
    let model = trained(case("gemma2"), lora(false, false));
    let dir = tempfile::tempdir().unwrap();
    model.save_adapter(dir.path(), None).unwrap();
    let names = tensor_names(&dir.path().join(ADAPTER_WEIGHTS));
    assert!(
        names.iter().all(|n| !n.contains("gemma1") && !n.contains("gemma2")),
        "{names:?}"
    );
    assert!(names.contains("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"));

    let mut loaded = AdaptedModel::attach(build(case("gemma2")), lora(false, false)).unwrap();
    loaded.load_lora_weights(dir.path()).unwrap();
    same_adapters(&model, &loaded);
    drain();
}

/// A config PEFT wrote for `all-linear` lists full module names; they select
/// those modules and no others.
#[test]
fn full_module_names_select_their_modules() {
    let config = LoraConfig {
        target_modules: vec![
            "model.layers.0.self_attn.q_proj".to_string(),
            "layers.1.mlp.down_proj".to_string(),
        ],
        ..lora(false, false)
    };
    let model = AdaptedModel::attach(build(case("llama")), config).unwrap();
    let mut adapted = model.adapted_projections().to_vec();
    adapted.sort();
    assert_eq!(
        adapted,
        ["model.layers.0.self_attn.q_proj", "model.layers.1.mlp.down_proj"]
    );
}

/// Agreement with `peft`, both ways, on a tiny Llama. Run with
/// `PMETAL_PEFT_ORACLE` set to a directory holding:
///
/// - `base/`: a `transformers` Llama (`config.json`, `model.safetensors`,
///   `logits.safetensors` for the base model) with the tiny config of the
///   `llama` case in `common`, float32;
/// - `lora/`, `rslora/`, `dora/`: `peft` adapters on it (rank 8, alpha 24,
///   `target_modules="all-linear"`, with every tensor randomised) saved with
///   `save_pretrained`, each with a `logits.safetensors` of the adapted
///   model's logits on `input_ids(256)` (token `(7 i + 3) % 256` at position
///   `i`, 24 tokens).
///
/// The adapters are loaded here and must reproduce the logits.
#[test]
#[ignore = "needs a directory of adapters written by peft"]
fn oracle_agrees_with_peft() {
    let root = std::env::var("PMETAL_PEFT_ORACLE").expect("PMETAL_PEFT_ORACLE");
    let root = Path::new(&root);
    let ids = input_ids(256);
    for variant in ["lora", "rslora", "dora"] {
        let dir = root.join(variant);
        let config_json: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join(ADAPTER_CONFIG)).expect("adapter_config.json"),
        )
        .unwrap();
        let config = peft::lora_config_from_peft(&config_json).expect("config");
        let base = DynamicModel::load(&root.join("base")).expect("load base");
        let mut model = AdaptedModel::attach(base, config).expect("attach");
        model.load_lora_weights(&dir).expect("load the PEFT adapter");
        let got = model.forward(&ids, None).unwrap();
        let want = load_safetensors_map(dir.join("logits.safetensors")).unwrap()["logits"].clone();
        let diff = max_abs_diff(&got, &want);
        println!("{variant}: max |logit difference| against peft = {diff:.3e}");
        assert!(diff < 2e-3, "{variant}: {diff}");

        // The other way: this adapter, written here, for peft to load.
        let out = root.join(format!("rust_{variant}"));
        model.save_adapter(&out, None).expect("save");
        save_safetensors_map(
            out.join("logits.safetensors"),
            &HashMap::from([("logits", got)]),
        )
        .unwrap();
    }
    drain();
}
