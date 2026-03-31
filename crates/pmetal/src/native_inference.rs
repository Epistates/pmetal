//! Native inference — all models through pmetal-bridge, zero mlx-rs.
//!
//! Each architecture has its own `run_{arch}` function that owns the full
//! pipeline: load config → load weights → prefill → decode.  The module is
//! intentionally self-contained; the only external dependencies are
//! `pmetal_bridge` and `serde_json`.

use std::path::Path;

use pmetal_bridge::turboquant::TurboQuantConfig;

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
}

impl NativeArch {
    pub fn supports_turboquant(self) -> bool {
        matches!(self, Self::Qwen3 | Self::Qwen3_5)
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Qwen3 => "Qwen3",
            Self::Qwen3_5 => "Qwen3.5",
            Self::Llama4 => "Llama4",
            Self::DeepSeek => "DeepSeek",
            Self::GptOss => "GPT-OSS",
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
    let data = std::fs::read_to_string(model_path.join("config.json")).ok()?;
    let v: serde_json::Value = serde_json::from_str(&data).ok()?;

    let mt = v
        .get("text_config")
        .and_then(|tc| tc.get("model_type"))
        .or_else(|| v.get("model_type"))
        .and_then(|mv| mv.as_str())?;

    match mt {
        "qwen3" | "qwen3dense" => Some(NativeArch::Qwen3),
        "qwen3_5" | "qwen3_5_text" | "qwen3_5_moe" | "qwen3_5_moe_text" => {
            Some(NativeArch::Qwen3_5)
        }
        "llama4" | "llama4_text" => Some(NativeArch::Llama4),
        "deepseek_v3" => Some(NativeArch::DeepSeek),
        "gpt_oss" => Some(NativeArch::GptOss),
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
                supports_turboquant: false,
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
    let arch = detect_arch(model_path)
        .ok_or_else(|| "unsupported architecture for native inference".to_string())?;

    ensure_native_bridge_metal_available()?;

    if turboquant.is_some() && !arch.supports_turboquant() {
        return Err(format!(
            "TurboQuant native cache is only supported for Qwen3/Qwen3.5, not {}",
            arch.label()
        ));
    }

    match arch {
        NativeArch::Qwen3 | NativeArch::Qwen3_5 => run_qwen3(
            model_path,
            input_ids,
            max_tokens,
            temperature,
            turboquant,
            &mut on_token,
        ),
        NativeArch::Llama4 => run_llama4(
            model_path,
            input_ids,
            max_tokens,
            temperature,
            &mut on_token,
        ),
        NativeArch::DeepSeek => run_deepseek(
            model_path,
            input_ids,
            max_tokens,
            temperature,
            &mut on_token,
        ),
        NativeArch::GptOss => run_gpt_oss(
            model_path,
            input_ids,
            max_tokens,
            temperature,
            &mut on_token,
        ),
    }
}

// ============================================================================
// Shared helper
// ============================================================================

fn finish_with_bridge_generate(
    prompt: &[u32],
    first_tok: u32,
    max_tokens: usize,
    on_token: &mut dyn FnMut(u32) -> bool,
    generate_tail: impl FnOnce(&mut dyn FnMut(u32) -> bool) -> Vec<u32>,
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
        };
    }

    let remaining = max_tokens.saturating_sub(1);
    let generated_tail = generate_tail(on_token);
    let stopped_by_token = generated_tail.len() < remaining;
    all_tokens.extend(generated_tail);
    let num_generated = all_tokens.len() - prompt_len;

    NativeGenerationOutput {
        token_ids: all_tokens,
        num_generated,
        stopped_by_token,
        stopped_by_length: !stopped_by_token && num_generated >= max_tokens,
    }
}

fn run_bridge_inference<Config, Weights, Cache>(
    model_path: &Path,
    input_ids: &[u32],
    max_tokens: usize,
    temperature: f32,
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
        f32,
        &mut dyn FnMut(u32) -> bool,
    ) -> Vec<u32>,
) -> Result<NativeGenerationOutput, String> {
    let config = load_config(model_path)?;
    eprintln!("[NATIVE] {}", describe_config(&config));

    let t0 = std::time::Instant::now();
    let weights = load_model(model_path, &config)?;
    eprintln!(
        "[NATIVE] Loaded in {:.1}s, active={:.0}MB",
        t0.elapsed().as_secs_f64(),
        pmetal_bridge::inline_array::get_active_memory() as f64 / 1e6,
    );

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
                temperature,
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
    temperature: f32,
    turboquant: Option<TurboQuantConfig>,
    on_token: &mut dyn FnMut(u32) -> bool,
) -> Result<NativeGenerationOutput, String> {
    use pmetal_bridge::qwen3_native;

    run_bridge_inference(
        model_path,
        input_ids,
        max_tokens,
        temperature,
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
        qwen3_native::load_model,
        |weights, _| build_qwen3_cache(weights, turboquant),
        qwen3_native::prefill_first_token,
        |weights, config, cache, first_tok, remaining, temperature, on_token| {
            qwen3_native::generate_canonical(
                weights,
                cache,
                config,
                first_tok,
                remaining,
                temperature,
                turboquant,
                on_token,
            )
        },
    )
}

fn build_qwen3_cache(
    weights: &pmetal_bridge::qwen3_native::NativeWeights,
    turboquant: Option<TurboQuantConfig>,
) -> pmetal_bridge::qwen3_native::NativeCache {
    match turboquant {
        Some(config) => {
            pmetal_bridge::qwen3_native::NativeCache::new_with_turboquant(weights, Some(config))
        }
        None => pmetal_bridge::qwen3_native::NativeCache::new_empty(weights),
    }
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
    temperature: f32,
    on_token: &mut dyn FnMut(u32) -> bool,
) -> Result<NativeGenerationOutput, String> {
    use pmetal_bridge::llama4_native;

    run_bridge_inference(
        model_path,
        input_ids,
        max_tokens,
        temperature,
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
        |weights, _| llama4_native::NativeCache::new_empty(weights),
        llama4_native::prefill_first_token,
        |weights, _config, cache, first_tok, remaining, temperature, on_token| {
            llama4_native::generate(weights, cache, first_tok, remaining, temperature, on_token)
        },
    )
}

// ============================================================================
// DeepSeek V3/R1
// ============================================================================

fn run_deepseek(
    model_path: &Path,
    input_ids: &[u32],
    max_tokens: usize,
    temperature: f32,
    on_token: &mut dyn FnMut(u32) -> bool,
) -> Result<NativeGenerationOutput, String> {
    use pmetal_bridge::deepseek_native;

    run_bridge_inference(
        model_path,
        input_ids,
        max_tokens,
        temperature,
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
        |_, config| deepseek_native::NativeCache::new_empty(config.num_hidden_layers as usize),
        deepseek_native::prefill_first_token,
        |weights, _config, cache, first_tok, remaining, temperature, on_token| {
            deepseek_native::generate(weights, cache, first_tok, remaining, temperature, on_token)
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
    temperature: f32,
    on_token: &mut dyn FnMut(u32) -> bool,
) -> Result<NativeGenerationOutput, String> {
    use pmetal_bridge::gpt_oss_native;

    run_bridge_inference(
        model_path,
        input_ids,
        max_tokens,
        temperature,
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
        |weights, _| gpt_oss_native::NativeCache::new_empty(weights),
        gpt_oss_native::prefill_first_token,
        |weights, _config, cache, first_tok, remaining, temperature, on_token| {
            gpt_oss_native::generate(weights, cache, first_tok, remaining, temperature, on_token)
        },
    )
}
