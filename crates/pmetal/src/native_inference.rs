//! Native inference — all models through the pmetal-bridge native engines.
//!
//! Each architecture has its own `run_{arch}` function that owns the full
//! pipeline: load config → load weights → prefill → decode.  The module is
//! intentionally self-contained; the only external dependencies are
//! `pmetal_bridge` and `serde_json`.

use std::path::Path;

use pmetal_bridge::turboquant::TurboQuantConfig;

pub(crate) fn ensure_native_bridge_metal_available() -> Result<(), String> {
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
    /// Qwen3 and Qwen3-MoE (`model_type = "qwen3"` / `"qwen3_moe"`).
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
        // hot/cold split). llama4_native got the same path wired through the
        // shared dispatch helper; deepseek's MLA layout is structurally
        // different and remains a follow-up. gpt-oss's attention sinks have
        // no place in TurboQuant's attention kernels.
        matches!(self, Self::Qwen3 | Self::Qwen3_5 | Self::Llama4)
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

    // A `*_mtp` checkpoint is the base architecture plus bundled
    // multi-token-prediction layers; `--mtp` loads those separately as a
    // speculative draft, and the target runs as its own architecture.
    let mt = mt.strip_suffix("_mtp").unwrap_or(mt);

    match mt {
        "qwen3" | "qwen3dense" | "qwen3_moe" => Some(NativeArch::Qwen3),
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
                // Attention sinks: TurboQuant's kernels have no sink term.
                supports_turboquant: false,
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
    fn detects_qwen3_moe_as_the_qwen3_engine() {
        let dir = write_temp_config(r#"{"model_type":"qwen3_moe"}"#);
        assert_eq!(detect_arch(&dir), Some(NativeArch::Qwen3));
        let _ = fs::remove_dir_all(dir);
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

    /// `mlx-community/gemma-4-12B-it-qat-4bit` stores every projection and the
    /// embedding as packed `uint32` beside `.scales`/`.biases`, and puts the
    /// MLP at 8 bits through per-module overrides. The native engine reads all
    /// of that now, so handing it to the generic loader would dequantize a
    /// 10 GB checkpoint back up to ~27 GB for nothing.
    #[test]
    fn claims_a_quantized_gemma4_checkpoint() {
        let dir = write_temp_config(
            r#"{
                "model_type":"gemma4_unified",
                "text_config":{"model_type":"gemma4_unified_text"},
                "quantization":{
                    "group_size":64,
                    "bits":4,
                    "language_model.model.layers.0.mlp.up_proj":{"group_size":64,"bits":8}
                }
            }"#,
        );
        assert_eq!(detect_arch(&dir), Some(NativeArch::Gemma4));
        let _ = fs::remove_dir_all(dir);
    }

    /// Every native architecture reads `.scales`/`.biases` itself, so
    /// quantization is not a reason to hand one to the generic loader.
    #[test]
    fn a_quantized_checkpoint_still_reaches_the_other_native_engines() {
        let dir = write_temp_config(
            r#"{"model_type":"qwen3","quantization":{"group_size":64,"bits":4}}"#,
        );
        assert_eq!(detect_arch(&dir), Some(NativeArch::Qwen3));
        let _ = fs::remove_dir_all(dir);
    }

    /// `mlx-community/Qwen3.8-27B-MTP-4bit` states `qwen3_5_mtp` and carries no
    /// `architectures` array, so the bundled-MTP checkpoints matched nothing
    /// and fell off the native engine entirely. The MTP layers are a draft that
    /// `--mtp` loads separately; the target is still a Qwen3.5.
    #[test]
    fn a_bundled_mtp_checkpoint_runs_as_its_target_architecture() {
        let dir = write_temp_config(r#"{"model_type":"qwen3_5_mtp"}"#);
        assert_eq!(detect_arch(&dir), Some(NativeArch::Qwen3_5));
        let _ = fs::remove_dir_all(dir);

        let moe = write_temp_config(r#"{"model_type":"qwen3_6_moe_mtp"}"#);
        assert_eq!(detect_arch(&moe), Some(NativeArch::Qwen3_5));
        let _ = fs::remove_dir_all(moe);
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
        // Its attention sinks have no place in TurboQuant's kernels.
        assert!(!info.supports_turboquant);
        assert!(!NativeArch::GptOss.supports_turboquant());
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
pub struct BenchmarkTrialMetrics {
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
            pmetal_bridge::qwen3_native::prefill_first_token,
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
            quant_config,
            &mut on_token,
        ),
        NativeArch::Gemma4 => run_gemma4(model_path, input_ids, max_tokens, params, &mut on_token),
    }
}

/// [`run_native_inference_ext`] for a Qwen3.5-family prompt carrying images
/// or videos.
///
/// `input_ids` is the expanded prompt (one token per merged patch); `images`
/// and `videos` are preprocessed, each in prompt order. The vision tower runs
/// first and is dropped before the text weights load; the prompt is then
/// prefilled from the merged embeddings at its 3-D positions and decoded on the
/// ordinary native path, from one past the prompt's largest position.
#[allow(clippy::too_many_arguments)]
pub fn run_native_multimodal_inference_ext(
    model_path: &Path,
    input_ids: &[u32],
    images: &[pmetal_data::qwen_vl_processing::ProcessedMedia],
    videos: &[pmetal_data::qwen_vl_processing::ProcessedMedia],
    max_tokens: usize,
    params: pmetal_bridge::decode::SamplingParams,
    turboquant: Option<TurboQuantConfig>,
    quant_config: Option<pmetal_bridge::qwen3_native::QuantCacheConfig>,
    mut on_token: impl FnMut(u32) -> bool,
) -> Result<NativeGenerationOutput, String> {
    use pmetal_bridge::qwen3_native;
    use pmetal_models::architectures::qwen3_5_vision::Qwen3_5Vision;

    if detect_arch(model_path) != Some(NativeArch::Qwen3_5) {
        return Err("images need a Qwen3.5-family model".into());
    }
    ensure_native_bridge_metal_available()?;
    let started = std::time::Instant::now();
    let vision = Qwen3_5Vision::load(model_path).map_err(|e| e.to_string())?;
    let encoded = vision
        .encode(input_ids, images, videos)
        .map_err(|e| e.to_string())?;
    let multimodal = vision.config.clone();
    drop(vision);
    tracing::info!(
        elapsed_s = format!("{:.1}", started.elapsed().as_secs_f64()),
        "Vision tower encoded {} images and {} videos",
        images.len(),
        videos.len()
    );
    let config = qwen3_native::load_config(model_path)?;
    let tables = config.mrope_tables(&encoded.positions.array());
    let prompt_ids = input_ids.to_vec();
    run_qwen3(
        model_path,
        input_ids,
        max_tokens,
        params,
        turboquant,
        quant_config,
        &mut on_token,
        move |weights, cache, ids, params| {
            // One pass: the merged embeddings and their 3-D positions cover the
            // whole prompt, so it is not split into chunks.
            pmetal_bridge::decode::prefill_first_token_chunked(
                weights,
                cache,
                ids,
                params,
                ids.len(),
                |weights, prompt, cache| {
                    let text = qwen3_native::embed_tokens(weights, prompt);
                    let embeddings = encoded
                        .merge(&text, &prompt_ids, &multimodal)
                        .expect("media token counts were checked by the vision encode");
                    qwen3_native::forward_embeddings_last(
                        weights,
                        &embeddings,
                        &tables,
                        encoded.positions.next_position,
                        cache,
                    )
                },
            )
        },
    )
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
    prefill_first_token: impl Fn(
        &Weights,
        &mut Cache,
        &[u32],
        pmetal_bridge::decode::SamplingParams,
    ) -> (u32, pmetal_bridge::decode::DecodeSampler),
    generate: impl Fn(
        &Weights,
        &Config,
        &mut Cache,
        u32,
        usize,
        pmetal_bridge::decode::DecodeSampler,
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

    let mut cache = build_cache(&weights, &config);
    let (first_tok, sampler) = prefill_first_token(&weights, &mut cache, input_ids, params);

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
                sampler,
                on_token,
            )
        },
    ))
}

fn trial_metrics(
    trial: pmetal_bridge::decode::BenchmarkTrial,
    prompt_tokens: usize,
    generation_tokens: usize,
) -> BenchmarkTrialMetrics {
    BenchmarkTrialMetrics {
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
) -> Vec<BenchmarkTrialMetrics> {
    let _warmup = run_once();

    let mut trials = Vec::with_capacity(num_trials);
    for _ in 0..num_trials {
        trials.push(trial_metrics(run_once(), prompt_tokens, generation_tokens));
    }
    trials
}

// ============================================================================
// Qwen3 / Qwen3.5
// ============================================================================

#[allow(clippy::too_many_arguments)]
fn run_qwen3(
    model_path: &Path,
    input_ids: &[u32],
    max_tokens: usize,
    params: pmetal_bridge::decode::SamplingParams,
    turboquant: Option<TurboQuantConfig>,
    quant_config: Option<pmetal_bridge::qwen3_native::QuantCacheConfig>,
    on_token: &mut dyn FnMut(u32) -> bool,
    prefill_first_token: impl Fn(
        &pmetal_bridge::qwen3_native::NativeWeights,
        &mut pmetal_bridge::qwen3_native::NativeCache,
        &[u32],
        pmetal_bridge::decode::SamplingParams,
    ) -> (u32, pmetal_bridge::decode::DecodeSampler),
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
        prefill_first_token,
        |weights, _config, cache, first_tok, remaining, sampler, on_token| {
            qwen3_native::generate(weights, cache, first_tok, remaining, sampler, on_token)
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
        |weights, config, cache, first_tok, remaining, sampler, on_token| {
            gemma4_native::generate(
                weights, config, cache, first_tok, remaining, sampler, on_token,
            )
        },
    )
}

/// Benchmark full prompt + generation throughput with a fixed workload shape:
/// fixed prompt token ids, one warmup, EOS disabled, and
/// repeated generations from a fresh cache.
pub fn benchmark_native(
    model_path: &Path,
    prompt_ids: &[u32],
    generation_tokens: usize,
    turboquant: Option<TurboQuantConfig>,
    num_trials: usize,
) -> Result<Vec<BenchmarkTrialMetrics>, String> {
    use pmetal_bridge::qwen3_native;

    if prompt_ids.is_empty() {
        return Err("Inference benchmark requires prompt_tokens > 0".to_string());
    }
    if generation_tokens == 0 {
        return Err("Inference benchmark requires generation_tokens > 0".to_string());
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
                    qwen3_native::benchmark_trial(
                        &weights,
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
                    llama4_native::benchmark_trial(&weights, prompt_ids, generation_tokens)
                })
            }
            NativeArch::DeepSeek => {
                use pmetal_bridge::deepseek_native;
                let config = deepseek_native::load_config(model_path)?;
                let weights = deepseek_native::load_model(model_path, &config)?;
                run_benchmark_trials(prompt_ids.len(), generation_tokens, num_trials, || {
                    deepseek_native::benchmark_trial(&weights, prompt_ids, generation_tokens)
                })
            }
            NativeArch::GptOss => {
                use pmetal_bridge::gpt_oss_native;
                let config = gpt_oss_native::load_config(model_path)?;
                let weights = gpt_oss_native::load_model(model_path, &config)?;
                run_benchmark_trials(prompt_ids.len(), generation_tokens, num_trials, || {
                    gpt_oss_native::benchmark_trial(&weights, prompt_ids, generation_tokens)
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
        |weights, _config, cache, first_tok, remaining, sampler, on_token| {
            llama4_native::generate(weights, cache, first_tok, remaining, sampler, on_token)
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
        move |weights, _| deepseek_native::NativeCache::new_with_quant(weights, quant_config),
        deepseek_native::prefill_first_token,
        |weights, _config, cache, first_tok, remaining, sampler, on_token| {
            deepseek_native::generate(weights, cache, first_tok, remaining, sampler, on_token)
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
        // Affine quant on full-attention layers only — sliding layers stay
        // in the model dtype.
        move |weights, _| gpt_oss_native::NativeCache::new_with_quant(weights, quant_config),
        gpt_oss_native::prefill_first_token,
        |weights, _config, cache, first_tok, remaining, sampler, on_token| {
            gpt_oss_native::generate(weights, cache, first_tok, remaining, sampler, on_token)
        },
    )
}
