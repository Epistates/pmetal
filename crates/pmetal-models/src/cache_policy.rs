//! One policy for the KV cache a generation runs on, and for how much room
//! the model and its cache need on the device.
//!
//! `pmetal infer`, the GUI and `pmetal serve` all choose a cache the same way:
//! an explicit request (`--kv-quant`, `--kv-turboquant`, a TurboQuant preset,
//! `--no-kv-quant`) is honoured as asked; otherwise the cache is fp16 unless
//! the weights plus a full-length fp16 cache would take more than
//! [`WORKING_SET_FRACTION`] of what the GPU may keep resident, and then it is
//! 8-bit. The weights are counted from the files the directory loads (an FP8
//! checkpoint, or `fp8` quantization at load, halves them), not from a
//! parameter count that a config may not state.
//!
//! The server's default context length ([`default_serve_context_len`]) draws on
//! the same budget, so the context it offers is one the cache selection then
//! leaves in fp16.

use std::collections::HashSet;
use std::path::Path;

use pmetal_cache::kv_cache::{
    CacheMode, KVCacheConfig, TurboQuantConfig, sanitize_cache_mode_for_config,
};
use pmetal_runtime::compat::Dtype;

/// The fraction of the device's recommended working set the weights and the
/// fp16 KV cache may fill before the cache goes to 8 bits (and the server's
/// default context stops growing).
pub const WORKING_SET_FRACTION: f64 = 0.70;

pub fn explicit_cache_mode_override(
    base_cache_config: &KVCacheConfig,
    request: CacheModeRequest,
) -> Option<CacheMode> {
    let explicit = request.no_kv_quant
        || request.kv_quant == Some(0)
        || request.kv_turboquant
        || request.kv_turboquant_preset.is_some()
        || request.kv_quant.is_some()
        || request.kv_k_bits.is_some()
        || request.kv_v_bits.is_some();

    if !explicit {
        return None;
    }

    let mode = if request.no_kv_quant || request.kv_quant == Some(0) {
        CacheMode::Standard
    } else {
        resolve_cache_mode(
            base_cache_config.head_dim,
            base_cache_config.value_head_dim,
            request,
        )
    };

    Some(sanitize_cache_mode_for_config(base_cache_config, mode))
}

fn resolve_cache_mode(
    key_head_dim: usize,
    value_head_dim: usize,
    request: CacheModeRequest,
) -> CacheMode {
    if request.no_kv_quant || request.kv_quant == Some(0) {
        return CacheMode::Standard;
    }

    if let Some(preset) = request.kv_turboquant_preset {
        if request.kv_quant.is_some() || request.kv_k_bits.is_some() || request.kv_v_bits.is_some()
        {
            tracing::warn!(
                preset = preset.as_str(),
                "TurboQuant preset overrides explicit KV bit flags"
            );
        }
        return CacheMode::TurboQuant {
            config: preset.config(key_head_dim, value_head_dim),
        };
    }

    let kv_quant = request.kv_quant.unwrap_or(8);
    match (request.kv_k_bits, request.kv_v_bits) {
        (Some(k), Some(v)) => {
            if request.kv_turboquant {
                CacheMode::TurboQuant {
                    config: TurboQuantConfig::uniform(k, v),
                }
            } else {
                CacheMode::AsymmetricQuantized {
                    key_bits: k,
                    value_bits: v,
                    group_size: request.kv_group_size,
                }
            }
        }
        (Some(k), None) | (None, Some(k)) => {
            let v = request.kv_v_bits.unwrap_or(kv_quant);
            let k_final = request.kv_k_bits.unwrap_or(kv_quant);
            let _ = k; // suppress unused
            if request.kv_turboquant {
                CacheMode::TurboQuant {
                    config: TurboQuantConfig::uniform(k_final, v),
                }
            } else if k_final == v {
                CacheMode::Quantized {
                    bits: k_final,
                    group_size: request.kv_group_size,
                }
            } else {
                CacheMode::AsymmetricQuantized {
                    key_bits: k_final,
                    value_bits: v,
                    group_size: request.kv_group_size,
                }
            }
        }
        (None, None) => {
            if request.kv_turboquant {
                CacheMode::TurboQuant {
                    config: TurboQuantConfig::uniform(kv_quant, kv_quant),
                }
            } else {
                CacheMode::Quantized {
                    bits: kv_quant,
                    group_size: request.kv_group_size,
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheModeSource {
    ForcedFp16,
    Explicit,
    AutoFp16,
    AutoQ8,
}

impl CacheModeSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ForcedFp16 => "forced-fp16",
            Self::Explicit => "explicit",
            Self::AutoFp16 => "auto-fp16",
            Self::AutoQ8 => "auto-q8",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheModeSelection {
    pub mode: CacheMode,
    pub source: CacheModeSource,
    pub estimated_weight_bytes: u64,
    pub estimated_fp16_kv_bytes: u64,
    pub working_set_bytes: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurboQuantPreset {
    Q2_5,
    Q3_5,
}

impl std::str::FromStr for TurboQuantPreset {
    type Err = String;

    fn from_str(name: &str) -> Result<Self, String> {
        match name {
            "q2_5" => Ok(Self::Q2_5),
            "q3_5" => Ok(Self::Q3_5),
            other => Err(format!("unknown preset '{other}'; valid: q2_5, q3_5")),
        }
    }
}

impl TurboQuantPreset {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Q2_5 => "q2_5",
            Self::Q3_5 => "q3_5",
        }
    }

    fn config(self, key_head_dim: usize, value_head_dim: usize) -> TurboQuantConfig {
        let key_config = match self {
            Self::Q2_5 => TurboQuantConfig::preset_q2_5(key_head_dim).keys,
            Self::Q3_5 => TurboQuantConfig::preset_q3_5(key_head_dim).keys,
        };
        let value_config = match self {
            Self::Q2_5 => TurboQuantConfig::preset_q2_5(value_head_dim).values,
            Self::Q3_5 => TurboQuantConfig::preset_q3_5(value_head_dim).values,
        };
        TurboQuantConfig {
            keys: key_config,
            values: value_config,
            ..TurboQuantConfig::uniform(8, 8)
        }
    }
}

/// What a caller asks of the KV cache; [`Default`] asks for nothing, which
/// leaves the choice to the memory budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheModeRequest {
    pub kv_quant: Option<u8>,
    pub kv_k_bits: Option<u8>,
    pub kv_v_bits: Option<u8>,
    pub kv_group_size: usize,
    pub kv_turboquant: bool,
    pub kv_turboquant_preset: Option<TurboQuantPreset>,
    pub no_kv_quant: bool,
    pub fp8: bool,
}

impl Default for CacheModeRequest {
    fn default() -> Self {
        Self {
            kv_quant: None,
            kv_k_bits: None,
            kv_v_bits: None,
            kv_group_size: 64,
            kv_turboquant: false,
            kv_turboquant_preset: None,
            no_kv_quant: false,
            fp8: false,
        }
    }
}

pub fn select_cache_mode_for_model(
    base_cache_config: &KVCacheConfig,
    model_path: &Path,
    param_count: usize,
    request: CacheModeRequest,
) -> CacheModeSelection {
    let working_set_bytes = pmetal_metal::context::MetalContext::global()
        .ok()
        .map(|ctx| ctx.properties().recommended_working_set_size);
    let estimated_weight_bytes = estimate_weight_bytes(model_path, param_count, request.fp8);

    select_cache_mode_with_working_set(
        base_cache_config,
        estimated_weight_bytes,
        request,
        working_set_bytes,
    )
}

pub fn select_cache_mode_with_working_set(
    base_cache_config: &KVCacheConfig,
    estimated_weight_bytes: u64,
    request: CacheModeRequest,
    working_set_bytes: Option<u64>,
) -> CacheModeSelection {
    let estimated_fp16_kv_bytes = estimate_fp16_kv_cache_bytes(base_cache_config);

    if request.no_kv_quant || request.kv_quant == Some(0) {
        return CacheModeSelection {
            mode: CacheMode::Standard,
            source: CacheModeSource::ForcedFp16,
            estimated_weight_bytes,
            estimated_fp16_kv_bytes,
            working_set_bytes,
        };
    }

    if let Some(mode) = explicit_cache_mode_override(base_cache_config, request) {
        return CacheModeSelection {
            mode,
            source: CacheModeSource::Explicit,
            estimated_weight_bytes,
            estimated_fp16_kv_bytes,
            working_set_bytes,
        };
    }

    let estimated_total_fp16 = estimated_weight_bytes.saturating_add(estimated_fp16_kv_bytes);
    let prefer_q8 = working_set_bytes.is_some_and(|working_set| {
        working_set > 0
            && estimated_total_fp16 > ((working_set as f64) * WORKING_SET_FRACTION) as u64
    });

    CacheModeSelection {
        mode: if prefer_q8 {
            CacheMode::Quantized {
                bits: 8,
                group_size: request.kv_group_size,
            }
        } else {
            CacheMode::Standard
        },
        source: if prefer_q8 {
            CacheModeSource::AutoQ8
        } else {
            CacheModeSource::AutoFp16
        },
        estimated_weight_bytes,
        estimated_fp16_kv_bytes,
        working_set_bytes,
    }
}

pub fn estimate_weight_bytes(model_path: &Path, param_count: usize, fp8: bool) -> u64 {
    let static_fp8_checkpoint = model_uses_weight_scale_inv_fp8(model_path);
    let param_estimate =
        estimate_weight_bytes_from_param_count(param_count, fp8 || static_fp8_checkpoint);
    estimate_local_model_weight_bytes(model_path, fp8, static_fp8_checkpoint)
        .map(|bytes| bytes.max(param_estimate))
        .unwrap_or(param_estimate)
}

pub(crate) fn estimate_weight_bytes_from_param_count(param_count: usize, fp8: bool) -> u64 {
    // Single source of truth for the fp8/fp16 weight byte rates —
    // see pmetal_core::constants::bytes_per_param.
    let quant = if fp8 { "fp8" } else { "fp16" };
    let bpp = pmetal_core::constants::bytes_per_param(quant);
    (param_count as f64 * bpp) as u64
}

/// The context a server holds per sequence when none is given.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DefaultContextLength {
    /// The chosen length.
    pub tokens: usize,
    /// The model's context window, when its config names one.
    pub context_window: Option<usize>,
    /// What the device's memory holds after the weights, when known.
    pub memory_cap: Option<usize>,
}

/// Shortest default context: what every model this server runs takes.
pub const MIN_DEFAULT_CONTEXT: usize = 4096;

/// The default per-sequence context of `pmetal serve`: the model's context
/// window ([`context_window`](pmetal_data::inference_config::context_window),
/// YaRN included), capped so the weights plus `sequences` full-length fp16 KV
/// caches stay within 70% of the device's recommended working set (the line
/// above which the server already switches the cache to 8 bits), rounded
/// down to a multiple of 1024 and never under [`MIN_DEFAULT_CONTEXT`]. A
/// request's cache grows with its sequence, so a long window costs memory
/// only when a request fills it.
///
/// The KV estimate is the hub's: a sliding layer stops growing at its window
/// and recurrent layers keep none.
pub fn default_serve_context_len(model_path: &Path, sequences: usize) -> DefaultContextLength {
    let context_window = pmetal_data::inference_config::context_window(model_path);
    let working_set = pmetal_metal::context::MetalContext::global()
        .ok()
        .map(|ctx| ctx.properties().recommended_working_set_size);
    let weights = sum_model_weight_file_bytes(model_path).unwrap_or(0);
    let memory_cap = match (working_set, kv_shape(model_path)) {
        (Some(ws), Some(kv)) => {
            let budget = ((ws as f64) * WORKING_SET_FRACTION) as u64;
            Some(context_that_fits(
                &kv,
                budget.saturating_sub(weights),
                sequences,
            ))
        }
        _ => None,
    };
    let tokens = match (context_window, memory_cap) {
        (Some(c), Some(m)) => c.min(m),
        (Some(c), None) => c,
        (None, Some(m)) => m,
        (None, None) => MIN_DEFAULT_CONTEXT,
    };
    let tokens = (tokens / 1024 * 1024).max(MIN_DEFAULT_CONTEXT);
    DefaultContextLength {
        tokens,
        context_window,
        memory_cap,
    }
}

/// How the model's KV cache grows with context, from its `config.json`
/// ([`pmetal_hub::fit::model_spec_from_dir`], the one definition of it:
/// latent attention, hybrid layer patterns and sliding windows included).
pub(crate) fn kv_shape(model_path: &Path) -> Option<pmetal_hub::fit::KvShape> {
    pmetal_hub::fit::model_spec_from_dir(model_path)?.kv
}

/// The longest context whose fp16 KV caches, one per sequence, fit in `free`
/// bytes.
pub(crate) fn context_that_fits(
    kv: &pmetal_hub::fit::KvShape,
    free: u64,
    sequences: usize,
) -> usize {
    // Contexts past a million tokens are not modelled (the hub clamps there).
    const LONGEST: usize = 1 << 20;
    let sequences = sequences.max(1) as f64;
    let fits = |context: usize| kv.bytes(context as u32, 2.0) * sequences <= free as f64;
    let (mut low, mut high) = (0usize, LONGEST);
    while low < high {
        let middle = (low + high).div_ceil(2);
        if fits(middle) {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    low
}

fn estimate_local_model_weight_bytes(
    model_path: &Path,
    fp8: bool,
    static_fp8_checkpoint: bool,
) -> Option<u64> {
    let on_disk_bytes = sum_model_weight_file_bytes(model_path)?;
    Some(if static_fp8_checkpoint {
        ((on_disk_bytes as f64) * 1.05).ceil() as u64
    } else if fp8 {
        ((on_disk_bytes as f64) * (1.05 / 2.0)).ceil() as u64
    } else {
        on_disk_bytes
    })
}

fn model_uses_weight_scale_inv_fp8(model_path: &Path) -> bool {
    let config_path = model_path.join("config.json");
    if let Ok(content) = std::fs::read_to_string(config_path) {
        if let Ok(config) = serde_json::from_str::<serde_json::Value>(&content) {
            if json_quantization_config_is_fp8(config.get("quantization_config"))
                || json_quantization_config_is_fp8(config.get("quantization"))
                || config.get("text_config").is_some_and(|text_config| {
                    json_quantization_config_is_fp8(text_config.get("quantization_config"))
                        || json_quantization_config_is_fp8(text_config.get("quantization"))
                })
            {
                return true;
            }
        }
    }

    let index_path = model_path.join(pmetal_core::safetensors_index::INDEX_FILE);
    pmetal_core::safetensors_index::ShardIndex::read(&index_path).is_ok_and(|index| {
        index
            .weight_map
            .keys()
            .any(|key| key.ends_with("weight_scale_inv"))
    })
}

fn json_quantization_config_is_fp8(value: Option<&serde_json::Value>) -> bool {
    value
        .and_then(|value| value.get("quant_method"))
        .and_then(serde_json::Value::as_str)
        .is_some_and(|method| method.eq_ignore_ascii_case("fp8"))
}

/// Bytes of weights `model_path` holds: the file itself for a GGUF path, else
/// the one set of weights the directory loads
/// ([`pmetal_hub::verify_model_dir`], which follows the Hub cache's symlinks
/// and never counts two formats of the same model twice).
fn sum_model_weight_file_bytes(model_path: &Path) -> Option<u64> {
    let bytes = if model_path.is_file() {
        std::fs::metadata(model_path).ok()?.len()
    } else {
        // Two names for one blob are one file's bytes.
        let mut seen = HashSet::new();
        pmetal_hub::verify_model_dir(model_path)
            .weight_files
            .iter()
            .filter(|file| seen.insert(file.canonicalize().unwrap_or_else(|_| (*file).clone())))
            .filter_map(|file| std::fs::metadata(file).ok())
            .map(|meta| meta.len())
            .sum()
    };
    (bytes > 0).then_some(bytes)
}

pub fn estimate_fp16_kv_cache_bytes(base_cache_config: &KVCacheConfig) -> u64 {
    base_cache_config
        .clone()
        .with_dtype(Dtype::Float16)
        .with_mode(CacheMode::Standard)
        .memory_footprint() as u64
}

pub fn log_cache_selection(selection: &CacheModeSelection, max_seq_len: usize) {
    let estimated_weight_gb = selection.estimated_weight_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let estimated_fp16_kv_gb =
        selection.estimated_fp16_kv_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let working_set_gb = selection
        .working_set_bytes
        .map(|bytes| bytes as f64 / (1024.0 * 1024.0 * 1024.0));

    tracing::info!(
        mode = %selection.mode.describe(),
        source = selection.source.as_str(),
        tokens = max_seq_len,
        estimated_weight_gb = format!("{estimated_weight_gb:.2}"),
        estimated_fp16_kv_gb = format!("{estimated_fp16_kv_gb:.2}"),
        working_set_gb = working_set_gb.map(|value| format!("{value:.2}")),
        "KV cache"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn qwen3_cache_config(max_seq_len: usize) -> KVCacheConfig {
        KVCacheConfig::new(28, max_seq_len, 8, 128)
    }

    #[test]
    fn presets_parse_by_the_name_the_command_line_uses() {
        assert_eq!("q2_5".parse(), Ok(TurboQuantPreset::Q2_5));
        assert_eq!("q3_5".parse(), Ok(TurboQuantPreset::Q3_5));
        for preset in [TurboQuantPreset::Q2_5, TurboQuantPreset::Q3_5] {
            assert_eq!(preset.as_str().parse(), Ok(preset));
        }
        assert!("Q2_5".parse::<TurboQuantPreset>().is_err());
    }

    #[test]
    fn auto_cache_mode_prefers_fp16_when_model_fits_comfortably() {
        let selection = select_cache_mode_with_working_set(
            &qwen3_cache_config(256),
            estimate_weight_bytes_from_param_count(620_000_000, false),
            CacheModeRequest {
                kv_quant: None,
                kv_k_bits: None,
                kv_v_bits: None,
                kv_group_size: 64,
                kv_turboquant: false,
                kv_turboquant_preset: None,
                no_kv_quant: false,
                fp8: false,
            },
            Some(48 * 1024 * 1024 * 1024),
        );

        assert_eq!(selection.mode, CacheMode::Standard);
        assert_eq!(selection.source, CacheModeSource::AutoFp16);
    }

    #[test]
    fn auto_cache_mode_promotes_to_q8_when_budget_is_tight() {
        let selection = select_cache_mode_with_working_set(
            &qwen3_cache_config(4096),
            estimate_weight_bytes_from_param_count(7_000_000_000, false),
            CacheModeRequest {
                kv_quant: None,
                kv_k_bits: None,
                kv_v_bits: None,
                kv_group_size: 64,
                kv_turboquant: false,
                kv_turboquant_preset: None,
                no_kv_quant: false,
                fp8: false,
            },
            Some(12 * 1024 * 1024 * 1024),
        );

        assert_eq!(
            selection.mode,
            CacheMode::Quantized {
                bits: 8,
                group_size: 64
            }
        );
        assert_eq!(selection.source, CacheModeSource::AutoQ8);
    }

    #[test]
    fn explicit_cache_mode_overrides_auto_selection() {
        let selection = select_cache_mode_with_working_set(
            &qwen3_cache_config(4096),
            estimate_weight_bytes_from_param_count(620_000_000, false),
            CacheModeRequest {
                kv_quant: Some(4),
                kv_k_bits: None,
                kv_v_bits: None,
                kv_group_size: 64,
                kv_turboquant: false,
                kv_turboquant_preset: None,
                no_kv_quant: false,
                fp8: false,
            },
            Some(12 * 1024 * 1024 * 1024),
        );

        assert_eq!(
            selection.mode,
            CacheMode::Quantized {
                bits: 4,
                group_size: 64
            }
        );
        assert_eq!(selection.source, CacheModeSource::Explicit);
    }

    #[test]
    fn explicit_turboquant_mode_uses_turboquant_variant() {
        let selection = select_cache_mode_with_working_set(
            &qwen3_cache_config(4096),
            estimate_weight_bytes_from_param_count(620_000_000, false),
            CacheModeRequest {
                kv_quant: Some(4),
                kv_k_bits: None,
                kv_v_bits: Some(3),
                kv_group_size: 64,
                kv_turboquant: true,
                kv_turboquant_preset: None,
                no_kv_quant: false,
                fp8: false,
            },
            Some(12 * 1024 * 1024 * 1024),
        );

        assert_eq!(
            selection.mode,
            CacheMode::TurboQuant {
                config: TurboQuantConfig::uniform(4, 3)
            }
        );
        assert_eq!(selection.source, CacheModeSource::Explicit);
    }

    #[test]
    fn turboquant_flag_without_bits_defaults_to_q8_turboquant() {
        let selection = select_cache_mode_with_working_set(
            &qwen3_cache_config(4096),
            estimate_weight_bytes_from_param_count(620_000_000, false),
            CacheModeRequest {
                kv_quant: None,
                kv_k_bits: None,
                kv_v_bits: None,
                kv_group_size: 64,
                kv_turboquant: true,
                kv_turboquant_preset: None,
                no_kv_quant: false,
                fp8: false,
            },
            Some(48 * 1024 * 1024 * 1024),
        );

        assert_eq!(
            selection.mode,
            CacheMode::TurboQuant {
                config: TurboQuantConfig::uniform(8, 8)
            }
        );
        assert_eq!(selection.source, CacheModeSource::Explicit);
    }

    #[test]
    fn turboquant_preset_uses_mixed_config() {
        let selection = select_cache_mode_with_working_set(
            &qwen3_cache_config(4096),
            estimate_weight_bytes_from_param_count(620_000_000, false),
            CacheModeRequest {
                kv_quant: None,
                kv_k_bits: None,
                kv_v_bits: None,
                kv_group_size: 64,
                kv_turboquant: false,
                kv_turboquant_preset: Some(TurboQuantPreset::Q2_5),
                no_kv_quant: false,
                fp8: false,
            },
            Some(48 * 1024 * 1024 * 1024),
        );

        assert_eq!(
            selection.mode,
            CacheMode::TurboQuant {
                config: TurboQuantConfig::preset_q2_5(128)
            }
        );
        assert_eq!(selection.source, CacheModeSource::Explicit);
    }

    #[test]
    fn explicit_cache_mode_override_uses_real_key_and_value_dims() {
        let base = KVCacheConfig::new(40, 4096, 2, 256)
            .with_value_head_dim(128)
            .with_dtype(Dtype::Float16);
        let mode = explicit_cache_mode_override(
            &base,
            CacheModeRequest {
                kv_quant: None,
                kv_k_bits: None,
                kv_v_bits: None,
                kv_group_size: 64,
                kv_turboquant: false,
                kv_turboquant_preset: Some(TurboQuantPreset::Q2_5),
                no_kv_quant: false,
                fp8: false,
            },
        )
        .expect("explicit turboquant preset should produce a cache override");

        let expected = TurboQuantConfig {
            keys: TurboQuantConfig::preset_q2_5(256).keys,
            values: TurboQuantConfig::preset_q2_5(128).values,
            ..TurboQuantConfig::uniform(8, 8)
        };
        assert_eq!(mode, CacheMode::TurboQuant { config: expected });
    }

    #[test]
    fn no_kv_quant_forces_fp16_even_when_budget_is_tight() {
        let selection = select_cache_mode_with_working_set(
            &qwen3_cache_config(4096),
            estimate_weight_bytes_from_param_count(7_000_000_000, false),
            CacheModeRequest {
                kv_quant: None,
                kv_k_bits: None,
                kv_v_bits: None,
                kv_group_size: 64,
                kv_turboquant: false,
                kv_turboquant_preset: None,
                no_kv_quant: true,
                fp8: false,
            },
            Some(12 * 1024 * 1024 * 1024),
        );

        assert_eq!(selection.mode, CacheMode::Standard);
        assert_eq!(selection.source, CacheModeSource::ForcedFp16);
    }

    #[test]
    fn estimate_weight_bytes_prefers_real_model_files_when_param_count_is_missing() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("model.safetensors"), vec![0u8; 4096]).unwrap();
        std::fs::write(dir.path().join("tokenizer.json"), vec![0u8; 8192]).unwrap();

        assert_eq!(estimate_weight_bytes(dir.path(), 0, false), 4096);
    }

    #[test]
    fn estimate_weight_bytes_scales_disk_estimate_for_fp8() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("model.safetensors"), vec![0u8; 4000]).unwrap();

        assert_eq!(estimate_weight_bytes(dir.path(), 0, true), 2100);
    }

    #[test]
    fn estimate_weight_bytes_accounts_for_static_fp8_mxfp8_runtime_weights() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("model.safetensors"), vec![0u8; 4000]).unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"quantization_config":{"quant_method":"fp8","fmt":"e4m3"}}"#,
        )
        .unwrap();

        assert_eq!(estimate_weight_bytes(dir.path(), 0, false), 4200);
    }

    #[test]
    fn estimate_weight_bytes_detects_weight_scale_inv_index_sidecars() {
        let dir = tempdir().unwrap();
        std::fs::write(
            dir.path().join("model-00001-of-00001.safetensors"),
            vec![0u8; 4096],
        )
        .unwrap();
        std::fs::write(
            dir.path().join("model.safetensors.index.json"),
            r#"{"weight_map":{"model.layers.0.mlp.gate_proj.weight_scale_inv":"model-00001-of-00001.safetensors"}}"#,
        )
        .unwrap();

        assert_eq!(estimate_weight_bytes(dir.path(), 0, false), 4301);
    }

    /// How many tokens of context `config.json`'s model fits in `free` bytes
    /// (one sequence).
    fn context_for(config: &str, free: u64) -> Option<usize> {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), config).unwrap();
        Some(context_that_fits(&kv_shape(dir.path())?, free, 1))
    }

    /// The KV cost a token adds is the hub's one definition of it: latent
    /// attention caches a compressed vector (not every head's keys and
    /// values), a hybrid keeps a cache only in its attention layers, and a
    /// sliding layer stops growing at its window.
    #[test]
    fn the_context_that_fits_follows_the_models_attention_layout() {
        let gib = 1u64 << 30;
        // DeepSeek-V3: 61 layers, each caching kv_lora_rank + qk_rope_head_dim
        // = 576 values (fp16: 2 bytes) per token.
        let mla = r#"{"num_hidden_layers": 61, "num_attention_heads": 128, "hidden_size": 7168,
                "num_key_value_heads": 128, "kv_lora_rank": 512, "qk_rope_head_dim": 64}"#;
        assert_eq!(context_for(mla, gib), Some((gib / (61 * 576 * 2)) as usize));

        // Nemotron-H: only the `*` layers of the pattern attend.
        let hybrid = r#"{"num_hidden_layers": 8, "num_attention_heads": 8, "hidden_size": 512,
                "num_key_value_heads": 2, "head_dim": 64, "hybrid_override_pattern": "M-M*M-M*"}"#;
        assert_eq!(
            context_for(hybrid, gib),
            Some((gib / (2 * (2 * 2 * 64) * 2)) as usize)
        );

        // Plain grouped-query attention: every layer, keys and values.
        let gqa = r#"{"num_hidden_layers": 4, "num_attention_heads": 8, "hidden_size": 512,
                "num_key_value_heads": 2}"#;
        assert_eq!(
            context_for(gqa, gib),
            Some((gib / (4 * (2 * 2 * 64) * 2)) as usize)
        );

        // Sliding layers stop growing at their window, so a model of them
        // fits far more than one that caches everything.
        let sliding = r#"{"num_hidden_layers": 4, "num_attention_heads": 8, "hidden_size": 512,
                "num_key_value_heads": 2, "sliding_window": 128}"#;
        assert!(context_for(sliding, gib) > context_for(gqa, gib));
        assert_eq!(context_for("{}", gib), None);
    }

    #[test]
    fn estimate_weight_bytes_dedupes_symlinked_hf_snapshot_files() {
        use std::os::unix::fs::symlink;

        let dir = tempdir().unwrap();
        let blobs = dir.path().join("blobs");
        let snapshot = dir.path().join("snapshots").join("1234");
        std::fs::create_dir_all(&blobs).unwrap();
        std::fs::create_dir_all(&snapshot).unwrap();

        let blob = blobs.join("weights");
        std::fs::write(&blob, vec![0u8; 3072]).unwrap();
        symlink(&blob, snapshot.join("model-00001-of-00002.safetensors")).unwrap();
        symlink(&blob, snapshot.join("model-duplicate.safetensors")).unwrap();
        std::fs::write(snapshot.join("config.json"), b"{}").unwrap();

        assert_eq!(estimate_weight_bytes(&snapshot, 0, false), 3072);
    }
}
