//! Native inference — all models through pmetal-bridge, zero mlx-rs.
//!
//! Each architecture has its own `run_{arch}` function that owns the full
//! pipeline: load config → load weights → prefill → decode.  The module is
//! intentionally self-contained; the only external dependencies are
//! `pmetal_bridge` and `serde_json`.

use std::{collections::HashMap, path::Path};

use pmetal_bridge::compat::{Array, ops::select_axis};
use pmetal_bridge::turboquant::TurboQuantConfig;
use pmetal_mlx::kv_cache::KVCache;
use pmetal_models::{
    Qwen3NextMtpConfig,
    architectures::Qwen3NextMtpForCausalLM,
    generation::{
        GenerationConfig, GenerationOutput, SpeculativeDecodeMetrics, sample_from_log_probs,
        sampling_log_probs_with_counts, token_probability_from_log_probs,
    },
};

fn ensure_native_bridge_metal_available() -> Result<(), String> {
    if pmetal_metal::context::MetalContext::device_available() {
        Ok(())
    } else {
        Err("Native bridge inference requires Metal: No Metal device found. Ensure running on Apple Silicon or macOS with Metal support.".to_string())
    }
}

// ============================================================================
// Architecture enum
// ============================================================================

/// Supported model architectures for native inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeArch {
    /// Dense Qwen3 (`model_type = "qwen3"`).
    Qwen3,
    /// Qwen3.5 dense or MoE (`qwen3_5*` and `qwen3_5_moe*` model types).
    Qwen3_5,
    /// Llama 4 (`model_type = "llama4"` / `"llama4_text"`).
    Llama4,
    /// DeepSeek V3/R1 (`model_type = "deepseek_v3"`).
    DeepSeek,
    /// GPT-OSS (`model_type = "gpt_oss"`).
    GptOss,
    /// Gemma 4 (`model_type = "gemma4"` / `"gemma4_text"`). Text-only dense
    /// path, including E2B/E4B per-layer-input gating and KV sharing. MoE /
    /// multimodal towers remain unsupported here.
    Gemma4,
}

impl NativeArch {
    pub fn supports_turboquant(self) -> bool {
        // qwen3_native owns the most-tested TurboQuant integration (incl. the
        // hot/cold split). gpt_oss_native and llama4_native got the same path
        // wired through the shared dispatch helper; deepseek's MLA layout is
        // structurally different and remains a follow-up.
        matches!(
            self,
            Self::Qwen3 | Self::Qwen3_5 | Self::GptOss | Self::Llama4
        )
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Qwen3 => "Qwen3",
            Self::Qwen3_5 => "Qwen3.5",
            Self::Llama4 => "Llama4",
            Self::DeepSeek => "DeepSeek",
            Self::GptOss => "GPT-OSS",
            Self::Gemma4 => "Gemma4",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeBridgeInfo {
    pub arch: NativeArch,
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub value_head_dim: usize,
    pub supports_turboquant: bool,
}

// ============================================================================
// Architecture detection
// ============================================================================

/// Detect architecture from `config.json`.
///
/// Checks `text_config.model_type` first (multi-modal configs), then falls
/// back to the top-level `model_type` field.
pub fn detect_arch(model_path: &Path) -> Option<NativeArch> {
    // Escape hatch for comparing the native engine against the generic
    // `pmetal-models` path on the same checkpoint, which is the only way to
    // tell which of the two is wrong when they disagree.
    if std::env::var_os("PMETAL_DISABLE_NATIVE_BRIDGE").is_some() {
        return None;
    }
    let data = std::fs::read_to_string(model_path.join("config.json")).ok()?;
    let v: serde_json::Value = serde_json::from_str(&data).ok()?;

    let mt = v
        .get("text_config")
        .and_then(|tc| tc.get("model_type"))
        .or_else(|| v.get("model_type"))
        .and_then(|mv| mv.as_str())?;

    match mt {
        "qwen3" | "qwen3dense" => Some(NativeArch::Qwen3),
        "qwen3_5" | "qwen3_5_text" | "qwen3_5_moe" | "qwen3_5_moe_text" | "qwen3_6"
        | "qwen3_6_text" | "qwen3_6_moe" | "qwen3_6_moe_text" => Some(NativeArch::Qwen3_5),
        "llama4" | "llama4_text" => Some(NativeArch::Llama4),
        "deepseek_v3" => Some(NativeArch::DeepSeek),
        "gpt_oss" => Some(NativeArch::GptOss),
        "gemma4" | "gemma4_text" | "gemma4_unified" | "gemma4_unified_text" => {
            Some(NativeArch::Gemma4)
        }
        _ => None,
    }
}

pub fn load_native_bridge_info(model_path: &Path) -> Result<Option<NativeBridgeInfo>, String> {
    let Some(arch) = detect_arch(model_path) else {
        return Ok(None);
    };

    let info = match arch {
        NativeArch::Qwen3 | NativeArch::Qwen3_5 => {
            let config = pmetal_bridge::qwen3_native::load_config(model_path)?;
            NativeBridgeInfo {
                arch,
                num_layers: config.num_hidden_layers as usize,
                num_kv_heads: config.get_num_kv_heads() as usize,
                head_dim: config.get_head_dim() as usize,
                value_head_dim: config.get_head_dim() as usize,
                supports_turboquant: true,
            }
        }
        NativeArch::Llama4 => {
            let config = pmetal_bridge::llama4_native::load_config(model_path)?;
            NativeBridgeInfo {
                arch,
                num_layers: config.text().num_hidden_layers as usize,
                num_kv_heads: config.num_kv_heads() as usize,
                head_dim: config.head_dim() as usize,
                value_head_dim: config.head_dim() as usize,
                supports_turboquant: true,
            }
        }
        NativeArch::DeepSeek => {
            let config = pmetal_bridge::deepseek_native::load_config(model_path)?;
            NativeBridgeInfo {
                arch,
                num_layers: config.num_hidden_layers as usize,
                num_kv_heads: config.num_attention_heads as usize,
                head_dim: config.q_head_dim() as usize,
                value_head_dim: config.v_head_dim as usize,
                supports_turboquant: false,
            }
        }
        NativeArch::GptOss => {
            let config = pmetal_bridge::gpt_oss_native::load_config(model_path)?;
            NativeBridgeInfo {
                arch,
                num_layers: config.num_hidden_layers as usize,
                num_kv_heads: config.num_key_value_heads as usize,
                head_dim: config.head_dim as usize,
                value_head_dim: config.head_dim as usize,
                supports_turboquant: true,
            }
        }
        NativeArch::Gemma4 => {
            let config = pmetal_bridge::gemma4_native::load_config(model_path)?;
            // Use the full-attention KV-head count since it's the smaller of
            // the two per-layer-type dimensions — callers only use this to
            // estimate KV-cache memory budgets.
            let n_kv_full = config
                .num_global_key_value_heads
                .unwrap_or(config.num_key_value_heads);
            let head_dim_full = config.global_head_dim.unwrap_or(config.head_dim);
            NativeBridgeInfo {
                arch,
                num_layers: config.num_hidden_layers as usize,
                num_kv_heads: n_kv_full.max(config.num_key_value_heads) as usize,
                head_dim: head_dim_full.max(config.head_dim) as usize,
                value_head_dim: head_dim_full.max(config.head_dim) as usize,
                supports_turboquant: false,
            }
        }
    };

    Ok(Some(info))
}

#[cfg(test)]
mod tests {
    use super::{NativeArch, detect_arch, load_native_bridge_info};
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn write_temp_config(json: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("pmetal-native-inference-{unique}"));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("config.json"), json).unwrap();
        dir
    }

    #[test]
    fn detects_qwen35_moe_from_top_level_model_type() {
        let dir = write_temp_config(r#"{"model_type":"qwen3_5_moe"}"#);
        assert_eq!(detect_arch(&dir), Some(NativeArch::Qwen3_5));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn detects_qwen35_moe_from_nested_text_model_type() {
        let dir = write_temp_config(
            r#"{"model_type":"qwen3_5_moe","text_config":{"model_type":"qwen3_5_moe_text"}}"#,
        );
        assert_eq!(detect_arch(&dir), Some(NativeArch::Qwen3_5));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn detects_qwen36_moe_from_nested_text_model_type() {
        let dir = write_temp_config(
            r#"{"model_type":"qwen3_6_moe","text_config":{"model_type":"qwen3_6_moe_text"}}"#,
        );
        assert_eq!(detect_arch(&dir), Some(NativeArch::Qwen3_5));
        let _ = fs::remove_dir_all(dir);
    }

    /// `mlx-community/gemma-4-12B-it-bf16` states `gemma4_unified_text`, which
    /// used to match nothing here, so it fell through to the generic path and
    /// decoded garbage while the 31B (`gemma4_text`) took the native engine.
    #[test]
    fn detects_gemma4_unified_from_nested_text_model_type() {
        let dir = write_temp_config(
            r#"{"model_type":"gemma4_unified","text_config":{"model_type":"gemma4_unified_text"}}"#,
        );
        assert_eq!(detect_arch(&dir), Some(NativeArch::Gemma4));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn load_native_bridge_info_tracks_qwen_turboquant_support() {
        let dir = write_temp_config(
            r#"{
                "model_type":"qwen3_5",
                "text_config":{
                    "model_type":"qwen3_5_text",
                    "hidden_size":1536,
                    "num_hidden_layers":28,
                    "num_attention_heads":12,
                    "num_key_value_heads":2,
                    "head_dim":128
                }
            }"#,
        );
        let info = load_native_bridge_info(&dir).unwrap().unwrap();
        assert_eq!(info.arch, NativeArch::Qwen3_5);
        assert_eq!(info.num_layers, 28);
        assert_eq!(info.num_kv_heads, 2);
        assert_eq!(info.head_dim, 128);
        assert_eq!(info.value_head_dim, 128);
        assert!(info.supports_turboquant);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn load_native_bridge_info_uses_attention_dims_not_linear_dims_for_qwen35_moe() {
        let dir = write_temp_config(
            r#"{
                "model_type":"qwen3_5_moe",
                "text_config":{
                    "model_type":"qwen3_5_moe_text",
                    "hidden_size":2048,
                    "num_hidden_layers":40,
                    "num_attention_heads":16,
                    "num_key_value_heads":2,
                    "head_dim":256,
                    "linear_key_head_dim":128,
                    "linear_value_head_dim":128
                }
            }"#,
        );
        let info = load_native_bridge_info(&dir).unwrap().unwrap();
        assert_eq!(info.arch, NativeArch::Qwen3_5);
        assert_eq!(info.num_layers, 40);
        assert_eq!(info.num_kv_heads, 2);
        assert_eq!(info.head_dim, 256);
        assert_eq!(info.value_head_dim, 256);
        assert!(info.supports_turboquant);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn load_native_bridge_info_tracks_gpt_oss_turboquant_support() {
        let dir = write_temp_config(
            r#"{
                "model_type":"gpt_oss",
                "hidden_size":2880,
                "num_hidden_layers":24,
                "num_attention_heads":64,
                "num_key_value_heads":8,
                "head_dim":64,
                "num_local_experts":32,
                "num_experts_per_tok":4,
                "sliding_window":128,
                "layer_types":["full_attention"],
                "intermediate_size":2880
            }"#,
        );
        let info = load_native_bridge_info(&dir).unwrap().unwrap();
        assert_eq!(info.arch, NativeArch::GptOss);
        assert!(info.supports_turboquant);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn load_native_bridge_info_tracks_llama4_turboquant_support() {
        let dir = write_temp_config(
            r#"{
                "model_type":"llama4_text",
                "hidden_size":5120,
                "intermediate_size":8192,
                "intermediate_size_mlp":16384,
                "num_hidden_layers":48,
                "num_attention_heads":40,
                "num_key_value_heads":8,
                "head_dim":128,
                "num_local_experts":16,
                "num_experts_per_tok":1,
                "vocab_size":128256,
                "attention_chunk_size":8192,
                "max_position_embeddings":131072,
                "use_qk_norm":true
            }"#,
        );
        let info = load_native_bridge_info(&dir).unwrap().unwrap();
        assert_eq!(info.arch, NativeArch::Llama4);
        assert!(info.supports_turboquant);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn load_native_bridge_info_tracks_deepseek_asymmetric_dims() {
        let dir = write_temp_config(
            r#"{
                "model_type":"deepseek_v3",
                "hidden_size":7168,
                "intermediate_size":18432,
                "num_experts_per_tok":8,
                "num_hidden_layers":61,
                "num_attention_heads":128,
                "kv_lora_rank":512,
                "qk_rope_head_dim":64,
                "v_head_dim":128,
                "qk_nope_head_dim":128
            }"#,
        );
        let info = load_native_bridge_info(&dir).unwrap().unwrap();
        assert_eq!(info.arch, NativeArch::DeepSeek);
        assert_eq!(info.num_layers, 61);
        assert_eq!(info.num_kv_heads, 128);
        assert_eq!(info.head_dim, 192);
        assert_eq!(info.value_head_dim, 128);
        assert!(!info.supports_turboquant);
        let _ = fs::remove_dir_all(dir);
    }
}

// ============================================================================
// Output type
// ============================================================================

/// Output produced by a native generation run.
pub struct NativeGenerationOutput {
    /// All token IDs: prompt + generated.
    pub token_ids: Vec<u32>,
    /// Number of tokens generated (excludes prompt).
    pub num_generated: usize,
    /// True when generation stopped because `on_token` returned `false`
    /// (i.e. an EOS or stop token was hit).
    pub stopped_by_token: bool,
    /// True when generation stopped because `max_tokens` was exhausted.
    pub stopped_by_length: bool,
    /// Decode throughput metrics from the bridge generate loop.
    /// `None` when fewer than 20 decode steps were measured.
    pub decode_metrics: Option<pmetal_bridge::decode::DecodeMetrics>,
}

#[derive(Debug, Clone)]
pub struct MlxLmBenchmarkTrial {
    pub prompt_tps: f64,
    pub generation_tps: f64,
    pub peak_memory_gb: f64,
}

// ============================================================================
// Top-level dispatch
// ============================================================================

/// Run native inference end-to-end: load → prefill → generate.
///
/// `on_token(id)` is called for every generated token; return `false` to stop
/// early (EOS, stop token, or user cancel).
///
/// Returns `Err` if the architecture is unsupported or if loading fails.
pub fn run_native_inference(
    model_path: &Path,
    input_ids: &[u32],
    max_tokens: usize,
    temperature: f32,
    turboquant: Option<TurboQuantConfig>,
    mut on_token: impl FnMut(u32) -> bool,
) -> Result<NativeGenerationOutput, String> {
    run_native_inference_ext(
        model_path,
        input_ids,
        max_tokens,
        pmetal_bridge::decode::SamplingParams::new(temperature),
        turboquant,
        None,
        &mut on_token,
    )
}

/// Extended native inference with optional affine KV cache quantization.
pub fn run_native_inference_ext(
    model_path: &Path,
    input_ids: &[u32],
    max_tokens: usize,
    params: pmetal_bridge::decode::SamplingParams,
    turboquant: Option<TurboQuantConfig>,
    quant_config: Option<pmetal_bridge::qwen3_native::QuantCacheConfig>,
    mut on_token: impl FnMut(u32) -> bool,
) -> Result<NativeGenerationOutput, String> {
    let arch = detect_arch(model_path)
        .ok_or_else(|| "unsupported architecture for native inference".to_string())?;

    ensure_native_bridge_metal_available()?;

    if turboquant.is_some() && !arch.supports_turboquant() {
        return Err(format!(
            "TurboQuant native cache is not yet supported for {}",
            arch.label()
        ));
    }

    match arch {
        NativeArch::Qwen3 | NativeArch::Qwen3_5 => run_qwen3(
            model_path,
            input_ids,
            max_tokens,
            params,
            turboquant,
            quant_config,
            &mut on_token,
        ),
        NativeArch::Llama4 => run_llama4(
            model_path,
            input_ids,
            max_tokens,
            params,
            turboquant,
            quant_config,
            &mut on_token,
        ),
        NativeArch::DeepSeek => run_deepseek(
            model_path,
            input_ids,
            max_tokens,
            params,
            quant_config,
            &mut on_token,
        ),
        NativeArch::GptOss => run_gpt_oss(
            model_path,
            input_ids,
            max_tokens,
            params,
            turboquant,
            quant_config,
            &mut on_token,
        ),
        NativeArch::Gemma4 => run_gemma4(model_path, input_ids, max_tokens, params, &mut on_token),
    }
}

/// Run Qwen3Next/Qwen3.6 MTP with the bridge-native Qwen target as verifier.
///
/// This is the native/compiled architecture proof: the large target/verifier
/// uses `pmetal_bridge::qwen3_native` and its compiled decode kernels, while
/// the small trained MTP predictor remains the existing Rust/MLX module. For
/// correctness with Qwen's GDN recurrent state, each verify round snapshots the
/// native cache and replays only committed tokens after partial acceptance.
#[allow(clippy::too_many_arguments)]
pub fn run_qwen3_native_mtp_inference_ext(
    model_path: &Path,
    mtp: &mut Qwen3NextMtpForCausalLM,
    mtp_cache: &mut KVCache,
    input_ids: &[u32],
    gen_config: GenerationConfig,
    mtp_config: Qwen3NextMtpConfig,
    mut on_token: impl FnMut(u32) -> bool,
) -> Result<GenerationOutput, String> {
    use pmetal_bridge::qwen3_native;

    if input_ids.is_empty() {
        return Err("Qwen native MTP requires a non-empty prompt".to_string());
    }
    if gen_config.max_new_tokens == 0 {
        return Ok(GenerationOutput {
            token_ids: input_ids.to_vec(),
            num_generated: 0,
            stopped_by_token: false,
            stopped_by_length: true,
            decode_metrics: None,
            speculative_metrics: None,
        });
    }

    ensure_native_bridge_metal_available()?;

    if let Some(seed) = gen_config.seed {
        pmetal_bridge::inline_array::random_seed(seed);
        seed_acceptance_rng(seed);
    }

    let config = qwen3_native::load_config(model_path)?;
    tracing::debug!(
        "Qwen native MTP verifier: {} layers, hidden={}",
        config.num_hidden_layers,
        config.hidden_size
    );

    let t0 = std::time::Instant::now();
    let weights = qwen3_native::load_model(model_path, &config)?;
    tracing::info!(
        elapsed_s = format!("{:.1}", t0.elapsed().as_secs_f64()),
        active_mb = format!(
            "{:.0}",
            pmetal_bridge::inline_array::get_active_memory() as f64 / 1e6
        ),
        "Native Qwen verifier loaded for MTP",
    );

    let mut target_cache = qwen3_native::NativeCache::new_empty(&weights);
    mtp_cache.reset();

    let prompt = token_array(input_ids);
    let (target_hidden, target_logits) =
        qwen3_native::forward_step_hidden(&weights, &prompt, &mut target_cache);
    let mut prev_target_logits = select_last_logits(&target_logits);

    let (mtp_hidden, mtp_logits) = mtp
        .forward_logits(&prompt, &target_hidden, None, Some(mtp_cache), 0)
        .map_err(|e| format!("native Qwen MTP prompt forward: {e}"))?;
    let mut last_mtp_hidden = select_last_sequence(&mtp_hidden);
    let mut prev_mtp_logits = select_last_logits(&mtp_logits);

    let mut token_ids = input_ids.to_vec();
    let mut generated_counts = HashMap::new();
    let mut num_generated = 0usize;
    let mut stopped_by_token = false;
    let draft_budget = mtp_config.num_draft_tokens.max(1);
    let mut speculative_metrics = SpeculativeDecodeMetrics::default();

    while num_generated < gen_config.max_new_tokens {
        let remaining = gen_config.max_new_tokens - num_generated;
        let max_draft = draft_budget.min(remaining);
        let base_mtp_hidden = last_mtp_hidden.clone();
        let base_mtp_logits = prev_mtp_logits.clone();

        let mut draft_tokens = Vec::with_capacity(max_draft);
        let mut draft_log_probs = Vec::with_capacity(max_draft);
        let mut draft_history = token_ids.clone();
        let mut draft_counts = generated_counts.clone();
        let mut draft_hidden = last_mtp_hidden.clone();
        let mut draft_logits = prev_mtp_logits.clone();

        for draft_idx in 0..max_draft {
            let draft_token = if gen_config.do_sample {
                let log_probs = sampling_log_probs_with_counts(
                    &draft_logits,
                    &draft_history,
                    &draft_counts,
                    &gen_config,
                )
                .map_err(|e| format!("native Qwen MTP draft sampling: {e}"))?;
                let token = sample_from_log_probs(&log_probs)
                    .map_err(|e| format!("native Qwen MTP draft sample: {e}"))?;
                draft_log_probs.push(log_probs);
                token
            } else {
                greedy_token(&draft_logits)
            };
            draft_tokens.push(draft_token);
            draft_history.push(draft_token);
            increment_count(&mut draft_counts, draft_token);

            let (next_hidden, next_logits) =
                mtp_step(mtp, draft_token, &draft_hidden, mtp_cache, draft_idx + 1)?;
            draft_hidden = next_hidden;
            draft_logits = next_logits;
            if gen_config.stop_tokens.contains(&draft_token) || draft_idx + 1 >= max_draft {
                break;
            }
        }

        let target_snapshot = target_cache.fork();
        let verify_input = token_array(&draft_tokens);
        let (verify_hidden, verify_logits) =
            qwen3_native::forward_step_hidden(&weights, &verify_input, &mut target_cache);

        let (accepted, correction) = if gen_config.do_sample {
            accept_sampled_draft(
                &draft_tokens,
                &draft_log_probs,
                &prev_target_logits,
                &verify_logits,
                &token_ids,
                &generated_counts,
                &gen_config,
            )?
        } else {
            accept_greedy_draft(&draft_tokens, &prev_target_logits, &verify_logits)
        };
        speculative_metrics.record_verify_step(draft_tokens.len(), accepted);

        let all_accepted = accepted == draft_tokens.len();
        let mut planned = Vec::with_capacity(draft_tokens.len() + 1);
        planned.extend_from_slice(&draft_tokens[..accepted]);
        let append_after_emit = if all_accepted {
            if num_generated + planned.len() < gen_config.max_new_tokens {
                let row = select_axis(&verify_logits, (draft_tokens.len() - 1) as i32, 1);
                let bonus = if gen_config.do_sample {
                    let mut verify_history = token_ids.clone();
                    let mut verify_counts = generated_counts.clone();
                    for &token in &draft_tokens {
                        verify_history.push(token);
                        increment_count(&mut verify_counts, token);
                    }
                    let log_probs = sampling_log_probs_with_counts(
                        &row,
                        &verify_history,
                        &verify_counts,
                        &gen_config,
                    )
                    .map_err(|e| format!("native Qwen MTP bonus sampling: {e}"))?;
                    sample_from_log_probs(&log_probs)
                        .map_err(|e| format!("native Qwen MTP bonus sample: {e}"))?
                } else {
                    greedy_token(&row)
                };
                planned.push(bonus);
                Some(bonus)
            } else {
                None
            }
        } else {
            let correction = correction.ok_or_else(|| {
                "native Qwen MTP rejected a draft without correction token".to_string()
            })?;
            planned.push(correction);
            Some(correction)
        };

        let stop_pos = planned
            .iter()
            .position(|token| gen_config.stop_tokens.contains(token));
        let planned_len = stop_pos.map(|idx| idx + 1).unwrap_or(planned.len());

        let mut continue_stream = true;
        let mut emitted_planned_count = 0usize;
        for &token in &planned[..planned_len] {
            continue_stream = emit_token(
                token,
                &mut token_ids,
                &mut num_generated,
                &mut stopped_by_token,
                &gen_config,
                &mut on_token,
            );
            emitted_planned_count += 1;
            increment_count(&mut generated_counts, token);
            if !continue_stream || stopped_by_token || num_generated >= gen_config.max_new_tokens {
                break;
            }
        }

        let emitted_draft_count = emitted_planned_count.min(accepted);
        speculative_metrics.emitted_draft_tokens += emitted_draft_count;
        if emitted_planned_count > emitted_draft_count {
            if all_accepted {
                speculative_metrics.bonus_tokens += 1;
            } else {
                speculative_metrics.correction_tokens += 1;
            }
        }

        if emitted_draft_count < draft_tokens.len() {
            target_cache = target_snapshot;
            for &token in &draft_tokens[..emitted_draft_count] {
                let input = token_array(&[token]);
                let _ = qwen3_native::forward_step_hidden(&weights, &input, &mut target_cache);
            }
        }

        mtp_cache.rollback(draft_tokens.len());
        let (mut next_mtp_hidden, mut next_mtp_logits) =
            (base_mtp_hidden.clone(), base_mtp_logits.clone());
        if emitted_draft_count > 0 {
            (next_mtp_hidden, next_mtp_logits) = replay_mtp_with_target_hidden(
                mtp,
                mtp_cache,
                &draft_tokens[..emitted_draft_count],
                &verify_hidden,
            )?;
        }
        last_mtp_hidden = next_mtp_hidden;
        prev_mtp_logits = next_mtp_logits;

        if !continue_stream
            || stopped_by_token
            || num_generated >= gen_config.max_new_tokens
            || stop_pos.is_some()
        {
            break;
        }

        if let Some(token) = append_after_emit
            && emitted_draft_count < planned_len
        {
            let (target_hidden, target_logits) =
                target_step_native(&weights, &mut target_cache, token);
            prev_target_logits = target_logits;
            let (mtp_hidden, mtp_logits) = mtp_step(
                mtp,
                token,
                &target_hidden,
                mtp_cache,
                emitted_draft_count + 1,
            )?;
            last_mtp_hidden = mtp_hidden;
            prev_mtp_logits = mtp_logits;
        } else if emitted_draft_count > 0 {
            let last_idx = emitted_draft_count - 1;
            prev_target_logits = select_axis(&verify_logits, last_idx as i32, 1);
        }
    }

    Ok(GenerationOutput {
        token_ids,
        num_generated,
        stopped_by_token,
        stopped_by_length: num_generated >= gen_config.max_new_tokens && !stopped_by_token,
        decode_metrics: None,
        speculative_metrics: Some(speculative_metrics),
    })
}

fn target_step_native(
    weights: &pmetal_bridge::qwen3_native::NativeWeights,
    cache: &mut pmetal_bridge::qwen3_native::NativeCache,
    token: u32,
) -> (Array, Array) {
    let input = token_array(&[token]);
    let (hidden, logits) = pmetal_bridge::qwen3_native::forward_step_hidden(weights, &input, cache);
    (select_last_sequence(&hidden), select_last_logits(&logits))
}

fn mtp_step(
    mtp: &mut Qwen3NextMtpForCausalLM,
    token: u32,
    hidden: &Array,
    mtp_cache: &mut KVCache,
    step_idx: usize,
) -> Result<(Array, Array), String> {
    let input = token_array(&[token]);
    let (hidden, logits) = mtp
        .forward_logits(&input, hidden, None, Some(mtp_cache), step_idx)
        .map_err(|e| format!("native Qwen MTP step: {e}"))?;
    Ok((select_last_sequence(&hidden), select_last_logits(&logits)))
}

fn replay_mtp_with_target_hidden(
    mtp: &mut Qwen3NextMtpForCausalLM,
    mtp_cache: &mut KVCache,
    tokens: &[u32],
    target_hidden: &Array,
) -> Result<(Array, Array), String> {
    let mut last_hidden = None;
    let mut last_logits = None;
    for (idx, &token) in tokens.iter().enumerate() {
        let hidden = select_axis(target_hidden, idx as i32, 1).reshape(&[1, 1, -1]);
        let (h, logits) = mtp_step(mtp, token, &hidden, mtp_cache, idx)?;
        last_hidden = Some(h);
        last_logits = Some(logits);
    }
    Ok((
        last_hidden.ok_or_else(|| "native Qwen MTP replay received no tokens".to_string())?,
        last_logits.ok_or_else(|| "native Qwen MTP replay received no tokens".to_string())?,
    ))
}

fn accept_greedy_draft(
    draft_tokens: &[u32],
    prev_target_logits: &Array,
    verify_logits: &Array,
) -> (usize, Option<u32>) {
    let mut matched = 0usize;
    while matched < draft_tokens.len() {
        let row = if matched == 0 {
            prev_target_logits.clone()
        } else {
            select_axis(verify_logits, (matched - 1) as i32, 1)
        };
        if greedy_token(&row) != draft_tokens[matched] {
            break;
        }
        matched += 1;
    }
    if matched == draft_tokens.len() {
        (matched, None)
    } else {
        let row = if matched == 0 {
            prev_target_logits.clone()
        } else {
            select_axis(verify_logits, (matched - 1) as i32, 1)
        };
        (matched, Some(greedy_token(&row)))
    }
}

#[allow(clippy::too_many_arguments)]
fn accept_sampled_draft(
    draft_tokens: &[u32],
    draft_log_probs: &[Array],
    prev_target_logits: &Array,
    verify_logits: &Array,
    token_ids: &[u32],
    generated_counts: &HashMap<u32, usize>,
    gen_config: &GenerationConfig,
) -> Result<(usize, Option<u32>), String> {
    let mut accepted = 0usize;
    let mut verify_history = token_ids.to_vec();
    let mut verify_counts = generated_counts.clone();

    while accepted < draft_tokens.len() {
        let row = if accepted == 0 {
            prev_target_logits.clone()
        } else {
            select_axis(verify_logits, (accepted - 1) as i32, 1)
        };
        let target_log_probs =
            sampling_log_probs_with_counts(&row, &verify_history, &verify_counts, gen_config)
                .map_err(|e| format!("native Qwen MTP target sampling log-probs: {e}"))?;
        let draft_log_probs_row = &draft_log_probs[accepted];
        let token = draft_tokens[accepted];
        let p_target = token_probability_from_log_probs(&target_log_probs, token)
            .map_err(|e| format!("native Qwen MTP target probability: {e}"))?;
        let p_draft = token_probability_from_log_probs(draft_log_probs_row, token)
            .map_err(|e| format!("native Qwen MTP draft probability: {e}"))?;
        let accept_prob = if p_draft > 0.0 {
            (p_target / p_draft).min(1.0)
        } else {
            0.0
        };

        if rand_uniform() < accept_prob {
            accepted += 1;
            verify_history.push(token);
            increment_count(&mut verify_counts, token);
        } else {
            let correction_log_probs =
                correction_log_probs(&target_log_probs, draft_log_probs_row)?;
            return Ok((
                accepted,
                Some(
                    sample_from_log_probs(&correction_log_probs)
                        .map_err(|e| format!("native Qwen MTP correction sample: {e}"))?,
                ),
            ));
        }
    }

    Ok((accepted, None))
}

fn correction_log_probs(
    target_log_probs: &Array,
    draft_log_probs: &Array,
) -> Result<Array, String> {
    let target_probs = target_log_probs.exp();
    let draft_probs = draft_log_probs.exp();
    let diff = target_probs.subtract(&draft_probs);
    let clipped = diff.maximum(&Array::from_f32(0.0));
    let total = clipped.sum_axis(-1, true);
    let total_value = total.item::<f32>();
    if !total_value.is_finite() || total_value <= 1e-20 {
        return Ok(target_log_probs.clone());
    }
    Ok(clipped.divide(&total).log())
}

fn emit_token<F>(
    token: u32,
    token_ids: &mut Vec<u32>,
    num_generated: &mut usize,
    stopped_by_token: &mut bool,
    gen_config: &GenerationConfig,
    on_token: &mut F,
) -> bool
where
    F: FnMut(u32) -> bool,
{
    token_ids.push(token);
    *num_generated += 1;
    if gen_config.stop_tokens.contains(&token) {
        *stopped_by_token = true;
    }
    on_token(token)
}

fn token_array(tokens: &[u32]) -> Array {
    let data: Vec<i32> = tokens.iter().map(|token| *token as i32).collect();
    Array::from_i32_slice(&data).reshape(&[1, data.len() as i32])
}

fn select_last_sequence(values: &Array) -> Array {
    let seq_len = values.dim(1);
    select_axis(values, seq_len - 1, 1).reshape(&[1, 1, -1])
}

fn select_last_logits(logits: &Array) -> Array {
    let seq_len = logits.dim(1);
    select_axis(logits, seq_len - 1, 1)
}

fn greedy_token(logits: &Array) -> u32 {
    logits.argmax(-1).item::<u32>()
}

fn increment_count(token_counts: &mut HashMap<u32, usize>, token: u32) {
    *token_counts.entry(token).or_insert(0) += 1;
}

thread_local! {
    static ACCEPTANCE_RNG_STATE: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

fn seed_acceptance_rng(seed: u64) {
    ACCEPTANCE_RNG_STATE.with(|state| {
        state.set(if seed == 0 {
            0xdead_beef_cafe_1234
        } else {
            seed
        });
    });
}

fn rand_uniform() -> f32 {
    ACCEPTANCE_RNG_STATE.with(|state| {
        let mut x = state.get();
        if x == 0 {
            x = seed_from_time();
        }
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        state.set(x);
        (x >> 40) as f32 / (1u64 << 24) as f32
    })
}

fn seed_from_time() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or(0x1234_5678_9abc_def0);
    nanos ^ 0xa5a5_5a5a_dead_beef
}

// ============================================================================
// Shared helper
// ============================================================================

fn finish_with_bridge_generate(
    prompt: &[u32],
    first_tok: u32,
    max_tokens: usize,
    on_token: &mut dyn FnMut(u32) -> bool,
    generate_tail: impl FnOnce(
        &mut dyn FnMut(u32) -> bool,
    ) -> (Vec<u32>, Option<pmetal_bridge::decode::DecodeMetrics>),
) -> NativeGenerationOutput {
    let prompt_len = prompt.len();
    let mut all_tokens = prompt.to_vec();
    all_tokens.push(first_tok);

    if !on_token(first_tok) {
        return NativeGenerationOutput {
            token_ids: all_tokens,
            num_generated: 1,
            stopped_by_token: true,
            stopped_by_length: false,
            decode_metrics: None,
        };
    }

    let remaining = max_tokens.saturating_sub(1);
    let (generated_tail, decode_metrics) = generate_tail(on_token);
    let stopped_by_token = generated_tail.len() < remaining;
    all_tokens.extend(generated_tail);
    let num_generated = all_tokens.len() - prompt_len;

    NativeGenerationOutput {
        token_ids: all_tokens,
        num_generated,
        stopped_by_token,
        stopped_by_length: !stopped_by_token && num_generated >= max_tokens,
        decode_metrics,
    }
}

#[allow(clippy::too_many_arguments)]
fn run_bridge_inference<Config, Weights, Cache>(
    model_path: &Path,
    input_ids: &[u32],
    max_tokens: usize,
    params: pmetal_bridge::decode::SamplingParams,
    on_token: &mut dyn FnMut(u32) -> bool,
    load_config: impl Fn(&Path) -> Result<Config, String>,
    describe_config: impl Fn(&Config) -> String,
    load_model: impl Fn(&Path, &Config) -> Result<Weights, String>,
    build_cache: impl Fn(&Weights, &Config) -> Cache,
    prefill_first_token: impl Fn(&Weights, &mut Cache, &[u32], f32) -> u32,
    generate: impl Fn(
        &Weights,
        &Config,
        &mut Cache,
        u32,
        usize,
        pmetal_bridge::decode::SamplingParams,
        &mut dyn FnMut(u32) -> bool,
    ) -> (Vec<u32>, Option<pmetal_bridge::decode::DecodeMetrics>),
) -> Result<NativeGenerationOutput, String> {
    let config = load_config(model_path)?;
    tracing::debug!("{}", describe_config(&config));

    let t0 = std::time::Instant::now();
    let weights = load_model(model_path, &config)?;
    tracing::info!(
        elapsed_s = format!("{:.1}", t0.elapsed().as_secs_f64()),
        active_mb = format!(
            "{:.0}",
            pmetal_bridge::inline_array::get_active_memory() as f64 / 1e6
        ),
        "Model loaded",
    );

    let temperature = params.temperature;
    let mut cache = build_cache(&weights, &config);
    let first_tok = prefill_first_token(&weights, &mut cache, input_ids, temperature);

    Ok(finish_with_bridge_generate(
        input_ids,
        first_tok,
        max_tokens,
        on_token,
        |on_token| {
            generate(
                &weights,
                &config,
                &mut cache,
                first_tok,
                max_tokens.saturating_sub(1),
                params,
                on_token,
            )
        },
    ))
}

fn mlx_lm_trial_metrics(
    trial: pmetal_bridge::decode::BenchmarkTrial,
    prompt_tokens: usize,
    generation_tokens: usize,
) -> MlxLmBenchmarkTrial {
    MlxLmBenchmarkTrial {
        prompt_tps: prompt_tokens as f64 / trial.prompt_secs.max(f64::MIN_POSITIVE),
        generation_tps: generation_tokens as f64 / trial.generation_secs.max(f64::MIN_POSITIVE),
        peak_memory_gb: trial.peak_memory_bytes as f64 / 1e9,
    }
}

fn run_benchmark_trials(
    prompt_tokens: usize,
    generation_tokens: usize,
    num_trials: usize,
    mut run_once: impl FnMut() -> pmetal_bridge::decode::BenchmarkTrial,
) -> Vec<MlxLmBenchmarkTrial> {
    let _warmup = run_once();

    let mut trials = Vec::with_capacity(num_trials);
    for _ in 0..num_trials {
        trials.push(mlx_lm_trial_metrics(
            run_once(),
            prompt_tokens,
            generation_tokens,
        ));
    }
    trials
}

// ============================================================================
// Qwen3 / Qwen3.5
// ============================================================================

fn run_qwen3(
    model_path: &Path,
    input_ids: &[u32],
    max_tokens: usize,
    params: pmetal_bridge::decode::SamplingParams,
    turboquant: Option<TurboQuantConfig>,
    quant_config: Option<pmetal_bridge::qwen3_native::QuantCacheConfig>,
    on_token: &mut dyn FnMut(u32) -> bool,
) -> Result<NativeGenerationOutput, String> {
    use pmetal_bridge::qwen3_native;

    run_bridge_inference(
        model_path,
        input_ids,
        max_tokens,
        params,
        on_token,
        qwen3_native::load_config,
        |config| {
            format!(
                "Qwen3{}: {} layers, hidden={}{}",
                if config.is_moe() { " MoE" } else { "" },
                config.num_hidden_layers,
                config.hidden_size,
                if config.is_qwen3_dense() {
                    " (Qwen3 dense)"
                } else {
                    ""
                },
            )
        },
        |path, config| {
            let mut weights = qwen3_native::load_model(path, config)?;
            // Apply Hadamard preconditioning when affine KV cache quantization is enabled.
            // Absorbs random rotation into Q/K/V/O weights for better quantization quality.
            if quant_config.is_some() {
                qwen3_native::apply_kv_preconditioning(&mut weights);
            }
            // Apply outlier channel permutation for mixed-bit presets (TurboQuant v2).
            // This absorbs the permutation into Q/K/V/O projection weights at load time,
            // moving high-magnitude channels to the front of each head with zero runtime cost.
            if let Some(qcfg) = quant_config {
                if let Some(mb) = qcfg.mixed_bit {
                    let outlier_fraction = mb.outlier_count as f32 / config.get_head_dim() as f32;
                    qwen3_native::apply_outlier_permutation(&mut weights, outlier_fraction);
                }
                // Generate QJL projection matrix when QJL residual correction is enabled.
                // Must be called after apply_kv_preconditioning so S is in the same space as R.
                if qcfg.qjl {
                    qwen3_native::apply_qjl_matrix(&mut weights);
                }
            }
            Ok(weights)
        },
        |weights, _| build_qwen3_cache_with_quant(weights, turboquant, quant_config),
        qwen3_native::prefill_first_token,
        |weights, config, cache, first_tok, remaining, params, on_token| {
            qwen3_native::generate_canonical(
                weights, cache, config, first_tok, remaining, params, turboquant, on_token,
            )
        },
    )
}

fn build_qwen3_cache_with_quant(
    weights: &pmetal_bridge::qwen3_native::NativeWeights,
    turboquant: Option<TurboQuantConfig>,
    quant_config: Option<pmetal_bridge::qwen3_native::QuantCacheConfig>,
) -> pmetal_bridge::qwen3_native::NativeCache {
    let mut cache = match turboquant {
        Some(config) => {
            pmetal_bridge::qwen3_native::NativeCache::new_with_turboquant(weights, Some(config))
        }
        None => pmetal_bridge::qwen3_native::NativeCache::new_empty(weights),
    };
    // Apply zero-overhead affine quantization config to all KV layers
    if let Some(qcfg) = quant_config {
        for kv in &mut cache.kv_caches {
            kv.quant_config = Some(qcfg);
        }
    }
    cache
}

// ============================================================================
// Gemma 4
// ============================================================================

fn run_gemma4(
    model_path: &Path,
    input_ids: &[u32],
    max_tokens: usize,
    params: pmetal_bridge::decode::SamplingParams,
    on_token: &mut dyn FnMut(u32) -> bool,
) -> Result<NativeGenerationOutput, String> {
    use pmetal_bridge::gemma4_native;

    run_bridge_inference(
        model_path,
        input_ids,
        max_tokens,
        params,
        on_token,
        gemma4_native::load_config,
        |config| {
            format!(
                "Gemma4: {} layers, hidden={}, head_dim={}, global_head_dim={:?}",
                config.num_hidden_layers,
                config.hidden_size,
                config.head_dim,
                config.global_head_dim
            )
        },
        gemma4_native::load_model,
        gemma4_native::build_cache,
        gemma4_native::prefill_first_token,
        |weights, config, cache, first_tok, remaining, params, on_token| {
            gemma4_native::generate(
                weights, config, cache, first_tok, remaining, params, on_token,
            )
        },
    )
}

/// Benchmark full prompt + generation throughput using the same workload shape
/// as `mlx_lm.benchmark`: fixed prompt token ids, one warmup, EOS disabled, and
/// repeated generations from a fresh cache.
pub fn benchmark_native_mlx_lm(
    model_path: &Path,
    prompt_ids: &[u32],
    generation_tokens: usize,
    turboquant: Option<TurboQuantConfig>,
    num_trials: usize,
) -> Result<Vec<MlxLmBenchmarkTrial>, String> {
    use pmetal_bridge::qwen3_native;

    if prompt_ids.is_empty() {
        return Err("MLX-LM parity benchmark requires prompt_tokens > 0".to_string());
    }
    if generation_tokens == 0 {
        return Err("MLX-LM parity benchmark requires generation_tokens > 0".to_string());
    }

    ensure_native_bridge_metal_available()?;

    let arch = detect_arch(model_path)
        .ok_or_else(|| "unsupported architecture for native inference".to_string())?;
    if turboquant.is_some() && !arch.supports_turboquant() {
        return Err(format!(
            "TurboQuant native benchmark is only supported for Qwen3/Qwen3.5, not {}",
            arch.label()
        ));
    }

    if num_trials == 0 {
        return Ok(Vec::new());
    }

    let trials = {
        match arch {
            NativeArch::Qwen3 | NativeArch::Qwen3_5 => {
                let config = qwen3_native::load_config(model_path)?;
                let weights = qwen3_native::load_model(model_path, &config)?;
                run_benchmark_trials(prompt_ids.len(), generation_tokens, num_trials, || {
                    qwen3_native::benchmark_mlx_lm_trial_canonical(
                        &weights,
                        &config,
                        prompt_ids,
                        generation_tokens,
                        turboquant,
                    )
                })
            }
            NativeArch::Llama4 => {
                use pmetal_bridge::llama4_native;
                let config = llama4_native::load_config(model_path)?;
                let weights = llama4_native::load_model(model_path, &config)?;
                run_benchmark_trials(prompt_ids.len(), generation_tokens, num_trials, || {
                    llama4_native::benchmark_mlx_lm_trial(&weights, prompt_ids, generation_tokens)
                })
            }
            NativeArch::DeepSeek => {
                use pmetal_bridge::deepseek_native;
                let config = deepseek_native::load_config(model_path)?;
                let weights = deepseek_native::load_model(model_path, &config)?;
                run_benchmark_trials(prompt_ids.len(), generation_tokens, num_trials, || {
                    deepseek_native::benchmark_mlx_lm_trial(&weights, prompt_ids, generation_tokens)
                })
            }
            NativeArch::GptOss => {
                use pmetal_bridge::gpt_oss_native;
                let config = gpt_oss_native::load_config(model_path)?;
                let weights = gpt_oss_native::load_model(model_path, &config)?;
                run_benchmark_trials(prompt_ids.len(), generation_tokens, num_trials, || {
                    gpt_oss_native::benchmark_mlx_lm_trial(&weights, prompt_ids, generation_tokens)
                })
            }
            NativeArch::Gemma4 => {
                return Err("Gemma 4 native benchmark mode is not implemented yet".to_string());
            }
        }
    };

    pmetal_bridge::inline_array::synchronize();
    pmetal_bridge::inline_array::clear_cache();

    Ok(trials)
}

// ============================================================================
// Llama 4
// ============================================================================

fn run_llama4(
    model_path: &Path,
    input_ids: &[u32],
    max_tokens: usize,
    params: pmetal_bridge::decode::SamplingParams,
    turboquant: Option<TurboQuantConfig>,
    quant_config: Option<pmetal_bridge::qwen3_native::QuantCacheConfig>,
    on_token: &mut dyn FnMut(u32) -> bool,
) -> Result<NativeGenerationOutput, String> {
    use pmetal_bridge::llama4_native;

    run_bridge_inference(
        model_path,
        input_ids,
        max_tokens,
        params,
        on_token,
        llama4_native::load_config,
        |config| {
            let tc = config.text();
            format!(
                "Llama4 MoE: {} layers, hidden={}, experts={}/tok={}",
                tc.num_hidden_layers, tc.hidden_size, tc.num_local_experts, tc.num_experts_per_tok
            )
        },
        llama4_native::load_model,
        move |weights, _| build_llama4_cache(weights, turboquant, quant_config),
        llama4_native::prefill_first_token,
        |weights, _config, cache, first_tok, remaining, params, on_token| {
            llama4_native::generate(weights, cache, first_tok, remaining, params, on_token)
        },
    )
}

fn build_llama4_cache(
    weights: &pmetal_bridge::llama4_native::NativeWeights,
    turboquant: Option<TurboQuantConfig>,
    quant_config: Option<pmetal_bridge::qwen3_native::QuantCacheConfig>,
) -> pmetal_bridge::llama4_native::NativeCache {
    use pmetal_bridge::llama4_native;
    let mut cache = match turboquant {
        Some(cfg) => llama4_native::NativeCache::new_with_turboquant(weights, Some(cfg)),
        None => llama4_native::NativeCache::new_empty(weights),
    };
    if let Some(qcfg) = quant_config {
        for kv in &mut cache.kv_caches {
            kv.quant_config = Some(qcfg);
        }
    }
    cache
}

// ============================================================================
// DeepSeek V3/R1
// ============================================================================

fn run_deepseek(
    model_path: &Path,
    input_ids: &[u32],
    max_tokens: usize,
    params: pmetal_bridge::decode::SamplingParams,
    quant_config: Option<pmetal_bridge::qwen3_native::QuantCacheConfig>,
    on_token: &mut dyn FnMut(u32) -> bool,
) -> Result<NativeGenerationOutput, String> {
    use pmetal_bridge::deepseek_native;

    run_bridge_inference(
        model_path,
        input_ids,
        max_tokens,
        params,
        on_token,
        deepseek_native::load_config,
        |config| {
            format!(
                "DeepSeek V3: {} layers, hidden={}, experts={}/tok={}",
                config.num_hidden_layers,
                config.hidden_size,
                config.n_routed_experts.unwrap_or(0),
                config.num_experts_per_tok,
            )
        },
        deepseek_native::load_model,
        move |_, config| {
            deepseek_native::NativeCache::new_with_quant(
                config.num_hidden_layers as usize,
                quant_config,
            )
        },
        deepseek_native::prefill_first_token,
        |weights, _config, cache, first_tok, remaining, params, on_token| {
            deepseek_native::generate(weights, cache, first_tok, remaining, params, on_token)
        },
    )
}

// ============================================================================
// GPT-OSS
// ============================================================================

fn run_gpt_oss(
    model_path: &Path,
    input_ids: &[u32],
    max_tokens: usize,
    params: pmetal_bridge::decode::SamplingParams,
    turboquant: Option<TurboQuantConfig>,
    quant_config: Option<pmetal_bridge::qwen3_native::QuantCacheConfig>,
    on_token: &mut dyn FnMut(u32) -> bool,
) -> Result<NativeGenerationOutput, String> {
    use pmetal_bridge::gpt_oss_native;

    run_bridge_inference(
        model_path,
        input_ids,
        max_tokens,
        params,
        on_token,
        gpt_oss_native::load_config,
        |config| {
            format!(
                "GPT-OSS: {} layers, hidden={}, experts={}/tok={}",
                config.num_hidden_layers,
                config.hidden_size,
                config.num_local_experts,
                config.experts_per_tok(),
            )
        },
        gpt_oss_native::load_model,
        move |weights, _| build_gpt_oss_cache(weights, turboquant, quant_config),
        gpt_oss_native::prefill_first_token,
        |weights, _config, cache, first_tok, remaining, params, on_token| {
            gpt_oss_native::generate(weights, cache, first_tok, remaining, params, on_token)
        },
    )
}

fn build_gpt_oss_cache(
    weights: &pmetal_bridge::gpt_oss_native::NativeWeights,
    turboquant: Option<TurboQuantConfig>,
    quant_config: Option<pmetal_bridge::qwen3_native::QuantCacheConfig>,
) -> pmetal_bridge::gpt_oss_native::NativeCache {
    use pmetal_bridge::gpt_oss_native;
    let mut cache = match turboquant {
        Some(cfg) => gpt_oss_native::NativeCache::new_with_turboquant(weights, Some(cfg)),
        None => gpt_oss_native::NativeCache::new_empty(weights),
    };
    // Affine quant on full-attention layers only — sliding layers stay bf16.
    if let Some(qcfg) = quant_config {
        for kv in &mut cache.kv_caches {
            if !kv.is_sliding {
                kv.quant_config = Some(qcfg);
            }
        }
    }
    cache
}
