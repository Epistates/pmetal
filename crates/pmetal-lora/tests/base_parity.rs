//! Holds the **training** forward pass against the **inference** forward pass.
//!
//! `pmetal-lora` does not wrap `pmetal-models`; it re-derives every
//! architecture from the same `Config` struct. That means each architecture has
//! two independent implementations, and a fix landed in one does not reach the
//! other. Nothing in the suite asked whether they still agree.
//!
//! A freshly constructed LoRA model initialises `lora_b` to zeros, so the
//! adapter contributes exactly nothing and the wrapped model *is* the base
//! model. Feed both paths the same checkpoint and the logits must match to
//! floating-point noise. Anything above that is the training path computing a
//! different function from the one that will serve the adapter.
//!
//! ## How a case is staged
//!
//! Weights are generated rather than committed, so this test needs no fixture
//! and carries no upstream model license:
//!
//! 1. [`DynamicModel::from_config`] builds the architecture from the case's
//!    `config.json` and random-initialises it. This is the *sized* constructor,
//!    not the placeholder one `load` uses.
//! 2. `flatten_params` yields that random init keyed by parameter path, which
//!    (after [`checkpoint_key`]) is the checkpoint format.
//! 3. Write it as `model.safetensors`. Both paths now load one identical set of
//!    weights through their own production loaders.
//!
//! Step 1 is what keeps this architecture-agnostic: no per-architecture weight
//! generator, and a checkpoint that is correct by construction.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use pmetal_bridge::compat::{Array, ModuleParametersExt, eval};
use pmetal_core::LoraConfig;
use pmetal_lora::{DynamicLoraModel, TrainableModel, save_safetensors_map};
use pmetal_mlx::test_utils::{
    ParityReport, Tolerance, argmax_last_axis, max_abs_value, print_report_table,
};
use pmetal_models::dispatcher::DynamicModel;

/// One architecture under test.
struct ArchCase {
    /// Display name, also the temp-directory discriminator.
    name: &'static str,
    /// Minimal `config.json`. Every config struct is `#[serde(default)]`, so
    /// only the fields that shape the forward pass need stating.
    config_json: &'static str,
    /// Set when the training path is *known* to compute something else, with
    /// the reason. `None` means the two must agree.
    known_divergence: Option<&'static str>,
}

/// Sequence fed to both paths. Long enough to cross a sliding-window boundary
/// in the cases that set one, short enough that a 2-layer model is quick.
const SEQ_LEN: i32 = 24;

/// Both paths run the same ops in f32 over the same weights, so the only
/// legitimate difference is op-ordering noise (a fused SDPA on one side and an
/// explicit softmax on the other reassociate the same sums). The relative part
/// carries the gate; `atol` only keeps near-zero logits from tripping it.
const TOLERANCE: Tolerance = Tolerance::new(1e-3, 1e-3);

fn cases() -> Vec<ArchCase> {
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
            known_divergence: Some(
                "The two paths disagree about where Llama 4's MoE block lives in a \
                 checkpoint. pmetal-models names the field `moe`, so its parameter \
                 path is `model.layers.N.moe.experts.J.…`; llama4_lora.rs reads \
                 `…feed_forward.experts.J.…`, which is what transformers' \
                 Llama4TextDecoderLayer calls it. The string `feed_forward` appears \
                 nowhere in pmetal-models, and the Llama 4 load arm remaps only the \
                 `model.language_model.` prefix, so at most one of the two can match \
                 a released checkpoint. Needs a real Llama 4 checkpoint to settle",
            ),
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
                "linear_key_head_dim": 16,
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
        },
    ]
}

/// Read an integer out of the case's `config.json`.
fn config_int(case: &ArchCase, key: &str) -> Result<i32, String> {
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
fn stage(case: &ArchCase) -> Result<PathBuf, String> {
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
fn checkpoint_key(param_path: &str, is_moe_layer: impl Fn(&str) -> bool) -> String {
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
fn input_ids(vocab_size: i32) -> Array {
    let ids: Vec<i32> = (0..SEQ_LEN).map(|i| (i * 7 + 3) % vocab_size).collect();
    Array::from_slice(&ids, &[1, SEQ_LEN])
}

/// Run one architecture and return its report, or the reason it could not run.
fn run_case(case: &ArchCase) -> Result<ParityReport, String> {
    let dir = stage(case)?;
    let result = compare(case, &dir);
    let _ = std::fs::remove_dir_all(&dir);
    result
}

fn compare(case: &ArchCase, dir: &Path) -> Result<ParityReport, String> {
    let vocab = config_int(case, "vocab_size")?;
    let ids = input_ids(vocab);

    let mut reference = DynamicModel::load(dir).map_err(|e| format!("reference load: {e}"))?;
    let ref_logits = reference
        .forward(&ids, None)
        .map_err(|e| format!("reference forward: {e}"))?;

    // Two identically-degenerate outputs would sail through any diff. Insist
    // the reference is a real distribution over the configured vocab before
    // comparing anything to it.
    if ref_logits.shape() != [1, SEQ_LEN, vocab] {
        return Err(format!(
            "reference logits are {:?}, expected [1, {SEQ_LEN}, {vocab}] — the staged \
             checkpoint did not populate this architecture",
            ref_logits.shape()
        ));
    }
    if max_abs_value(&ref_logits) < 1e-3 {
        return Err("reference logits are all but zero, so the comparison is vacuous".to_string());
    }

    // Rank 8 with the stock zero-init on `lora_b`: adapters are present and
    // trainable, and contribute nothing until the first optimizer step.
    let lora_config = LoraConfig {
        r: 8,
        alpha: 16.0,
        dropout: 0.0,
        ..Default::default()
    };
    let mut lora = DynamicLoraModel::from_pretrained(dir, lora_config)
        .map_err(|e| format!("lora load: {e}"))?;
    let lora_logits =
        TrainableModel::forward(&mut lora, &ids, None).map_err(|e| format!("lora forward: {e}"))?;

    if ref_logits.shape() != lora_logits.shape() {
        return Err(format!(
            "shape mismatch: inference {:?} vs training {:?}",
            ref_logits.shape(),
            lora_logits.shape()
        ));
    }

    let report = ParityReport::compute(case.name, &lora_logits, &ref_logits, TOLERANCE);
    Ok(report)
}

/// Every architecture reachable from both dispatchers must agree.
///
/// Reported as a table so a regression names the architecture that broke rather
/// than failing on whichever case happens to run first.
///
/// Architectures carrying a [`ArchCase::known_divergence`] are held to the
/// opposite assertion: they *must still fail*. That keeps the gate honest in
/// both directions. A new divergence breaks the build, and repairing a listed
/// one also breaks the build until its entry is deleted, so the list cannot rot
/// into a permanent suppression.
#[test]
fn lora_forward_matches_base_forward() {
    let mut reports = Vec::new();
    let mut failures = Vec::new();
    let mut fixed = Vec::new();

    for case in cases() {
        match run_case(&case) {
            Ok(report) => {
                let detail = format!(
                    "max_abs={:.3e} mean_abs={:.3e} cos={:.6}",
                    report.max_abs_diff, report.mean_abs_diff, report.cosine_similarity
                );
                match (report.passed(), case.known_divergence) {
                    (false, None) => failures.push(format!("{}: {detail}", case.name)),
                    (true, Some(note)) => fixed.push(format!(
                        "{}: now agrees ({detail}). Delete its `known_divergence`: {note}",
                        case.name
                    )),
                    _ => {}
                }
                reports.push(report);
            }
            Err(reason) => failures.push(format!("{}: {reason}", case.name)),
        }
    }

    print_report_table(&reports);
    for case in cases() {
        if let Some(note) = case.known_divergence {
            println!("known divergence — {}: {note}", case.name);
        }
    }

    assert!(
        fixed.is_empty(),
        "an architecture listed as divergent now agrees:\n  {}",
        fixed.join("\n  ")
    );
    assert!(
        failures.is_empty(),
        "the training forward pass diverges from the inference forward pass:\n  {}",
        failures.join("\n  ")
    );
}

/// The zero-init contract the test above rests on.
///
/// If `lora_b` ever stopped being zero-initialised, `lora_forward_matches_base_forward`
/// would fail for a reason that has nothing to do with architecture drift. This
/// separates the two so the diagnosis is immediate.
#[test]
fn fresh_adapters_contribute_nothing() {
    let case = &cases()[0];
    let dir = stage(case).expect("stage llama");

    let lora_config = LoraConfig {
        r: 8,
        alpha: 16.0,
        dropout: 0.0,
        ..Default::default()
    };
    let lora = DynamicLoraModel::from_pretrained(&dir, lora_config).expect("load lora");
    let params = lora.lora_parameters();
    let _ = std::fs::remove_dir_all(&dir);

    assert!(!params.is_empty(), "no adapter parameters were created");

    let mut b_count = 0;
    for (name, value) in &params {
        if name.ends_with("lora_b") {
            b_count += 1;
            assert_eq!(
                max_abs_value(value),
                0.0,
                "{name} is not zero-initialised, so a fresh adapter perturbs the base model"
            );
        }
    }
    assert!(
        b_count > 0,
        "no lora_b parameters found among {} params",
        params.len()
    );
}

/// Guard against a tolerance that passes because both sides produce garbage.
///
/// A model whose logits are all zero, or whose argmax is constant, would sail
/// through a diff-based comparison while telling us nothing.
#[test]
fn reference_forward_is_non_degenerate() {
    let case = &cases()[0];
    let dir = stage(case).expect("stage llama");
    let mut model = DynamicModel::load(&dir).expect("load reference");
    let logits = model.forward(&input_ids(256), None).expect("forward");
    let _ = std::fs::remove_dir_all(&dir);

    assert!(
        max_abs_value(&logits) > 1e-3,
        "reference logits are all but zero, so the parity comparison is vacuous"
    );
    let argmax = argmax_last_axis(&logits);
    assert!(
        argmax.iter().any(|&t| t != argmax[0]),
        "reference argmax is constant across every position"
    );
}
