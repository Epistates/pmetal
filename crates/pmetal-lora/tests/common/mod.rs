//! Tiny random configs for every architecture the dispatcher builds, and a
//! staged checkpoint for each, shared by the integration tests that hold the
//! training path against the inference path.

// Each test binary compiles its own copy of this module and uses a subset of
// it, so `dead_code` fires per binary. `expect` would go unfulfilled in
// `base_parity`, which uses every item.
#![allow(
    dead_code,
    reason = "shared by several test binaries, each using a subset"
)]

use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;

use pmetal_bridge::compat::optimizers::{AdamW, Optimizer};
use pmetal_bridge::compat::{Array, Dtype, ModuleParametersExt, eval, random};
use pmetal_bridge::inline_array::value_and_grad;
use pmetal_core::LoraConfig;
use pmetal_lora::{AdaptedModel, TrainableModel, save_safetensors_map};
use pmetal_mlx::test_utils::max_abs_diff;
use pmetal_models::dispatcher::DynamicModel;

/// One architecture under test.
pub struct ArchCase {
    /// Display name, also the temp-directory discriminator.
    pub name: &'static str,
    /// Minimal `config.json`. Every config struct is `#[serde(default)]`, so
    /// only the fields that shape the forward pass need stating.
    pub config_json: &'static str,
    /// Set when the training path is *known* to compute something else, with
    /// the reason. `None` means the two must agree.
    pub known_divergence: Option<&'static str>,
    /// Whether the architecture loads the checkpoint [`stage`] writes, which
    /// is keyed by pmetal's own parameter paths. Nemotron-H's loader reads
    /// the reference `mixer` layout per layer type instead, so its case
    /// builds from `from_config` but cannot round-trip through a file.
    pub stages: bool,
}

/// Sequence fed to both paths. Long enough to cross a sliding-window boundary
/// in the cases that set one, short enough that a 2-layer model is quick.
pub const SEQ_LEN: i32 = 24;

pub fn cases() -> Vec<ArchCase> {
    vec![
        ArchCase {
            name: "llama",
            config_json: r#"{
                "model_type": "llama",
                "vocab_size": 256,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "max_position_embeddings": 512,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0,
                "tie_word_embeddings": false
            }"#,
            known_divergence: None,
            stages: true,
        },
        ArchCase {
            name: "mistral",
            config_json: r#"{
                "model_type": "mistral",
                "vocab_size": 256,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "max_position_embeddings": 512,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0,
                "tie_word_embeddings": false
            }"#,
            known_divergence: None,
            stages: true,
        },
        ArchCase {
            name: "qwen3",
            config_json: r#"{
                "model_type": "qwen3",
                "vocab_size": 256,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "max_position_embeddings": 512,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0,
                "tie_word_embeddings": false
            }"#,
            known_divergence: None,
            stages: true,
        },
        // Gemma 1: uniformly causal, so this case isolates the GeGLU / embedding
        // scaling path from the window handling exercised by `gemma2` below.
        ArchCase {
            name: "gemma",
            config_json: r#"{
                "model_type": "gemma",
                "vocab_size": 256,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "max_position_embeddings": 512,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0
            }"#,
            known_divergence: None,
            stages: true,
        },
        // Gemma 2 alternates local and global attention. `sliding_window` is
        // deliberately below SEQ_LEN so the window actually bites: with a full
        // causal mask every layer sees the whole prefix and the divergence is
        // invisible.
        ArchCase {
            name: "gemma2",
            config_json: r#"{
                "model_type": "gemma2",
                "vocab_size": 256,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "max_position_embeddings": 512,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0,
                "sliding_window": 8,
                "attn_logit_softcapping": 50.0,
                "final_logit_softcapping": 30.0,
                "query_pre_attn_scalar": 16
            }"#,
            known_divergence: None,
            stages: true,
        },
        // Phi-3 with LongRoPE. `max_position_embeddings` exceeds
        // `original_max_position_embeddings`, which is what selects the long
        // factor table in the inference path.
        ArchCase {
            name: "phi3_longrope",
            config_json: r#"{
                "model_type": "phi3",
                "vocab_size": 256,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "max_position_embeddings": 512,
                "original_max_position_embeddings": 128,
                "rms_norm_eps": 1e-5,
                "rope_theta": 10000.0,
                "hidden_act": "silu",
                "tie_word_embeddings": false,
                "rope_scaling": {
                    "type": "longrope",
                    "short_factor": [1.0, 1.02, 1.04, 1.06, 1.08, 1.10, 1.12, 1.14],
                    "long_factor": [1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.4, 3.8]
                }
            }"#,
            known_divergence: None,
            stages: true,
        },
        ArchCase {
            name: "cohere",
            config_json: r#"{
                "model_type": "cohere",
                "vocab_size": 256,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "max_position_embeddings": 512,
                "layer_norm_eps": 1e-5,
                "rope_theta": 10000.0,
                "logit_scale": 0.0625
            }"#,
            known_divergence: None,
            stages: true,
        },
        ArchCase {
            name: "granite",
            config_json: r#"{
                "model_type": "granite",
                "vocab_size": 256,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "max_position_embeddings": 512,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0,
                "attention_multiplier": 0.125,
                "embedding_multiplier": 12.0,
                "residual_multiplier": 0.22,
                "logits_scaling": 8.0,
                "tie_word_embeddings": false
            }"#,
            known_divergence: None,
            stages: true,
        },
        // Granite 4.0-H: Mamba-2 and NoPE attention, routed experts plus a
        // shared MLP.
        ArchCase {
            name: "granitemoehybrid",
            config_json: r#"{
                "model_type": "granitemoehybrid",
                "vocab_size": 256,
                "hidden_size": 64,
                "intermediate_size": 32,
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "max_position_embeddings": 512,
                "rms_norm_eps": 1e-5,
                "layer_types": ["mamba", "attention", "mamba", "mamba"],
                "position_embedding_type": "nope",
                "num_local_experts": 4,
                "num_experts_per_tok": 2,
                "shared_intermediate_size": 48,
                "mamba_n_heads": 8,
                "mamba_d_state": 16,
                "mamba_chunk_size": 8,
                "attention_multiplier": 0.125,
                "embedding_multiplier": 12.0,
                "residual_multiplier": 0.22,
                "logits_scaling": 6.0,
                "tie_word_embeddings": true
            }"#,
            known_divergence: None,
            stages: true,
        },
        ArchCase {
            name: "qwen3_moe",
            config_json: r#"{
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
            }"#,
            known_divergence: None,
            stages: true,
        },
        ArchCase {
            name: "gpt_oss",
            config_json: r#"{
                "model_type": "gpt_oss",
                "vocab_size": 256,
                "hidden_size": 64,
                "intermediate_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "num_local_experts": 4,
                "num_experts_per_tok": 2,
                "max_position_embeddings": 512,
                "rms_norm_eps": 1e-5,
                "rope_theta": 10000.0,
                "sliding_window": 8,
                "tie_word_embeddings": false
            }"#,
            known_divergence: None,
            stages: true,
        },
        // Gemma 3 makes every layer local except one in `sliding_window_pattern`,
        // a different interleave from Gemma 2.
        ArchCase {
            name: "gemma3",
            config_json: r#"{
                "model_type": "gemma3",
                "vocab_size": 256,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "max_position_embeddings": 512,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0,
                "rope_local_base_freq": 10000.0,
                "sliding_window": 8,
                "sliding_window_pattern": 2
            }"#,
            known_divergence: None,
            stages: true,
        },
        ArchCase {
            name: "gemma4",
            config_json: r#"{
                "model_type": "gemma4_text",
                "vocab_size": 256,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "max_position_embeddings": 512,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0,
                "sliding_window": 8
            }"#,
            known_divergence: None,
            stages: true,
        },
        ArchCase {
            name: "llama4",
            config_json: r#"{
                "model_type": "llama4_text",
                "vocab_size": 256,
                "hidden_size": 64,
                "intermediate_size": 32,
                "intermediate_size_mlp": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "num_local_experts": 4,
                "num_experts_per_tok": 2,
                "interleave_moe_layer_step": 1,
                "max_position_embeddings": 512,
                "rms_norm_eps": 1e-5,
                "rope_theta": 10000.0,
                "attention_chunk_size": 8192,
                "tie_word_embeddings": false
            }"#,
            // Llama 4's training and inference paths used to name the MoE
            // block differently (`feed_forward` against `moe`) and disagree
            // about the router's key, so neither could load a real checkpoint
            // and the two could not agree. There is one forward pass now, so
            // the question no longer arises. `llama4::sanitize_checkpoint`
            // splits a real checkpoint's fused `gate_up_proj` / `down_proj`
            // into the per-expert Linears this forward uses.
            known_divergence: None,
            stages: true,
        },
        // Dense DeepSeek (no routed experts) isolates MLA from the MoE.
        ArchCase {
            name: "deepseek_dense",
            config_json: r#"{
                "model_type": "deepseek_v3",
                "vocab_size": 256,
                "hidden_size": 64,
                "intermediate_size": 128,
                "moe_intermediate_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "num_experts_per_tok": 1,
                "n_group": 1,
                "topk_group": 1,
                "first_k_dense_replace": 8,
                "routed_scaling_factor": 1.0,
                "topk_method": "greedy",
                "scoring_func": "softmax",
                "norm_topk_prob": true,
                "attention_bias": false,
                "moe_layer_freq": 1,
                "kv_lora_rank": 16,
                "q_lora_rank": null,
                "qk_rope_head_dim": 8,
                "qk_nope_head_dim": 8,
                "v_head_dim": 16,
                "max_position_embeddings": 512,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0,
                "tie_word_embeddings": false
            }"#,
            known_divergence: None,
            stages: true,
        },
        ArchCase {
            name: "deepseek",
            config_json: r#"{
                "model_type": "deepseek_v3",
                "vocab_size": 256,
                "hidden_size": 64,
                "intermediate_size": 128,
                "moe_intermediate_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "n_routed_experts": 4,
                "n_shared_experts": 1,
                "num_experts_per_tok": 2,
                "n_group": 1,
                "topk_group": 1,
                "first_k_dense_replace": 0,
                "routed_scaling_factor": 1.0,
                "topk_method": "greedy",
                "scoring_func": "softmax",
                "norm_topk_prob": true,
                "attention_bias": false,
                "moe_layer_freq": 1,
                "kv_lora_rank": 16,
                "q_lora_rank": null,
                "qk_rope_head_dim": 8,
                "qk_nope_head_dim": 8,
                "v_head_dim": 16,
                "max_position_embeddings": 512,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0,
                "tie_word_embeddings": false
            }"#,
            known_divergence: None,
            stages: true,
        },
        // Nemotron-H: Mamba-2, attention, a relu² MLP and a MoE block with a
        // shared expert, one layer of each.
        ArchCase {
            name: "nemotron_h",
            config_json: r#"{
                "model_type": "nemotron_h",
                "vocab_size": 256,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 4,
                "hybrid_override_pattern": "M*-E",
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "mamba_num_heads": 4,
                "mamba_head_dim": 16,
                "ssm_state_size": 16,
                "conv_kernel": 4,
                "n_groups": 2,
                "mlp_hidden_act": "relu2",
                "layer_norm_epsilon": 1e-5,
                "use_conv_bias": true,
                "moe_intermediate_size": 32,
                "moe_shared_expert_intermediate_size": 64,
                "n_routed_experts": 4,
                "n_shared_experts": 1,
                "num_experts_per_tok": 2,
                "max_position_embeddings": 512,
                "rope_theta": 10000.0,
                "tie_word_embeddings": false
            }"#,
            known_divergence: None,
            stages: false,
        },
        ArchCase {
            name: "qwen3_next",
            config_json: r#"{
                "model_type": "qwen3_next",
                "vocab_size": 256,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "max_position_embeddings": 512,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0,
                "linear_num_value_heads": 4,
                "linear_num_key_heads": 2,
                "linear_key_head_dim": 32,
                "linear_value_head_dim": 16,
                "linear_conv_kernel_dim": 4,
                "full_attention_interval": 4,
                "num_experts": 0,
                "num_experts_per_tok": 0,
                "moe_intermediate_size": 32,
                "shared_expert_intermediate_size": 128,
                "partial_rotary_factor": 0.25,
                "tie_word_embeddings": false
            }"#,
            known_divergence: None,
            stages: true,
        },
    ]
}

/// Read an integer out of the case's `config.json`.
pub fn config_int(case: &ArchCase, key: &str) -> Result<i32, String> {
    let value: serde_json::Value =
        serde_json::from_str(case.config_json).map_err(|e| format!("parse config.json: {e}"))?;
    value[key]
        .as_i64()
        .map(|v| v as i32)
        .ok_or_else(|| format!("config.json has no integer `{key}`"))
}

/// Stage `config.json` and a placeholder checkpoint, then materialise a real
/// one from the architecture's own random init.
///
/// Returns the staged directory. The caller removes it.
pub fn stage(case: &ArchCase) -> Result<PathBuf, String> {
    let dir = std::env::temp_dir().join(format!(
        "pmetal_lora_base_parity_{}_{}",
        case.name,
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).map_err(|e| format!("create temp dir: {e}"))?;
    std::fs::write(dir.join("config.json"), case.config_json)
        .map_err(|e| format!("write config.json: {e}"))?;

    // `from_config` is the sized constructor, so every parameter already has
    // its real shape and a random value. `flatten_params` then keys them by the
    // parameter paths, which for most architectures *are* the checkpoint keys.
    let donor =
        DynamicModel::from_config(case.config_json).map_err(|e| format!("build donor: {e}"))?;
    let params = donor.flatten_params();
    if params.is_empty() {
        return Err("donor model exposed no parameters".to_string());
    }
    eval(params.values()).map_err(|e| format!("eval checkpoint tensors: {e}"))?;

    // Which `mlp` prefixes are DeepSeek MoE blocks, recognised by the `w1/w2/w3`
    // expert naming `deepseek_param_name` produces. Keying on the rename's own
    // signature rather than on "has experts" keeps other MoE architectures,
    // whose parameter paths already are checkpoint keys, out of the rewrite.
    let moe_prefixes: std::collections::HashSet<String> = params
        .keys()
        .filter_map(|path| {
            let idx = path.find(".mlp.experts.")?;
            let (prefix, tail) = path.split_at(idx + ".mlp.".len());
            let member = tail.strip_prefix("experts.")?.split_once('.')?.1;
            matches!(member, "w1.weight" | "w2.weight" | "w3.weight").then(|| prefix.to_string())
        })
        .collect();

    let checkpoint: HashMap<Rc<str>, Array> = params
        .into_iter()
        .map(|(path, value)| {
            let key = checkpoint_key(&path, |prefix| moe_prefixes.contains(prefix));
            (Rc::from(key.as_str()), value)
        })
        .collect();
    save_safetensors_map(dir.join("model.safetensors"), &checkpoint)
        .map_err(|e| format!("write checkpoint: {e}"))?;

    Ok(dir)
}

/// Rewrite a pmetal parameter path into the key a checkpoint would use.
///
/// The two are the same almost everywhere, which is what lets
/// `assign_loaded_weights` match by exact name. Two architectures differ, and
/// both handle it with a bespoke loader that walks the struct instead of
/// matching names, so nothing in production notices — but it does mean their
/// flattened parameters are not a valid checkpoint, and this test has to bridge
/// the gap itself.
///
/// **Gemma.** `GemmaLayers` holds `gemma1` and `gemma2` as separate fields so
/// one struct can carry either layer shape, and `impl_module_params!` puts that
/// field name into the path: `model.layers.gemma1.0.self_attn.q_proj.weight`
/// where the checkpoint says `model.layers.0.self_attn.q_proj.weight`.
///
/// **DeepSeek.** `deepseek_param_name` renames three things inside a MoE `mlp`
/// on the way in; this is its inverse. `is_moe_layer` distinguishes the shared
/// expert (which pmetal merges into `mlp` with no prefix) from a dense MLP,
/// which would otherwise be the same path.
pub fn checkpoint_key(param_path: &str, is_moe_layer: impl Fn(&str) -> bool) -> String {
    for variant in ["model.layers.gemma1.", "model.layers.gemma2."] {
        if let Some(rest) = param_path.strip_prefix(variant) {
            return format!("model.layers.{rest}");
        }
    }

    let Some(idx) = param_path.find(".mlp.") else {
        return param_path.to_string();
    };
    let (prefix, tail) = param_path.split_at(idx + ".mlp.".len());
    if !is_moe_layer(prefix) {
        return param_path.to_string();
    }

    // `mlp.weight.weight` is the router: `DeepSeekMoEGate`'s own Linear field is
    // called `weight`, so the flattened path double-nests.
    if tail == "weight.weight" {
        return format!("{prefix}gate.weight");
    }
    if let Some(rest) = tail.strip_prefix("experts.") {
        if let Some((index, member)) = rest.split_once('.') {
            let renamed = match member {
                "w1.weight" => Some("gate_proj.weight"),
                "w3.weight" => Some("up_proj.weight"),
                "w2.weight" => Some("down_proj.weight"),
                _ => None,
            };
            if let Some(renamed) = renamed {
                return format!("{prefix}experts.{index}.{renamed}");
            }
        }
        return param_path.to_string();
    }
    // Anything else directly under a MoE layer's `mlp` is the shared expert,
    // which pmetal merges in with no prefix.
    format!("{prefix}shared_experts.{tail}")
}

/// Deterministic token ids inside the configured vocab.
pub fn input_ids(vocab_size: i32) -> Array {
    let ids: Vec<i32> = (0..SEQ_LEN).map(|i| (i * 7 + 3) % vocab_size).collect();
    Array::from_slice(&ids, &[1, SEQ_LEN])
}

/// Adapters that don't move the logits when their `B` is nudged, as
/// `"case: path"`, after `prepare` has had the base model.
///
/// An architecture that multiplies by a projection's weight directly, rather
/// than through `Linear::forward`, never sees the adapter on it.
pub fn silent_adapters(prepare: impl Fn(&mut DynamicModel)) -> Vec<String> {
    // Every projection, as the QLoRA paper adapts them.
    let lora = LoraConfig {
        r: 4,
        alpha: 8.0,
        dropout: 0.0,
        target_modules: Vec::new(),
        ..Default::default()
    };
    let mut silent = Vec::new();
    for case in cases() {
        let Ok(mut base) = DynamicModel::from_config(case.config_json) else {
            continue;
        };
        prepare(&mut base);
        let mut model = AdaptedModel::attach(base, lora.clone()).expect("attach");
        let ids = input_ids(config_int(&case, "vocab_size").expect("vocab"));
        let Ok(Ok(before)) =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| model.forward(&ids, None)))
        else {
            let _ = pmetal_bridge::check_last_error();
            silent.push(format!("{}: the model doesn't run", case.name));
            continue;
        };

        // One adapter at a time: each must move the logits.
        let adapters: Vec<String> = model.adapted_projections().to_vec();
        let params = model.lora_parameters();
        for path in adapters {
            let key = path.strip_prefix("model.").unwrap_or(&path).to_string();
            let b_key: Rc<str> = Rc::from(format!("{key}.lora_b").as_str());
            let b = params.get(&b_key).unwrap_or_else(|| panic!("{key}.lora_b"));
            let nudged: HashMap<Rc<str>, Array> = HashMap::from([(
                b_key.clone(),
                random::uniform_range(0.5, 1.0, b.shape(), Dtype::Float32),
            )]);
            model.set_lora_parameters(&nudged);
            let after = model.forward(&ids, None).expect("forward");
            pmetal_bridge::check_last_error().expect("no bridge error");
            let moved = max_abs_diff(&before, &after);
            // NaN compares unequal to everything, so a broken forward would
            // pass for one the adapter reached.
            if !(moved.is_finite() && moved > 0.0) {
                silent.push(format!(
                    "{}: {path} (moved the logits by {moved})",
                    case.name
                ));
            }
            model.set_lora_parameters(&HashMap::from([(b_key, b.clone())]));
        }
    }
    silent
}

/// One step of causal-LM training on the adapters alone. Returns the loss.
pub fn train_step(model: &mut AdaptedModel, optimizer: &mut AdamW, ids: &Array) -> f32 {
    let mut names: Vec<Rc<str>> = model.flatten_trainable_params().into_keys().collect();
    names.sort();
    let live = model.flatten_trainable_params();
    let params: Vec<Array> = names.iter().map(|name| live[name].clone()).collect();
    drop(live);

    let (loss, grads) = value_and_grad(
        |arrays| {
            let restored: HashMap<Rc<str>, Array> = names
                .iter()
                .cloned()
                .zip(arrays[..names.len()].iter().cloned())
                .collect();
            model.set_lora_parameters(&restored);
            let input = &arrays[names.len()];
            let logits = TrainableModel::forward(model, input, None).unwrap();
            pmetal_bridge::training::causal_lm_loss(&logits, input, -100)
        },
        &params,
        std::slice::from_ref(ids),
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
