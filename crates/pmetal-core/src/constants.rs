//! Shared numeric constants — quantisation byte tables and the like.
//!
//! These tables used to live as magic numbers inside individual crates
//! (`pmetal-hub::fit`, `pmetal-cli::inference_runner`, …), each duplicating
//! a partial view of the same lookup. The April 2026 audit flagged the
//! drift risk: the inference runner only knew about fp8 vs fp16, while
//! the fit estimator covered fp32/fp16/fp8/q8/q6/q5/q4/q3/q2. Centralising
//! here keeps the tables honest and makes model-size estimation consistent
//! across the CLI, TUI, serve, hub, and Python surfaces.

/// Returns bytes per weight parameter for a given quantisation format.
///
/// Designed for inference memory estimation on Apple Silicon / MLX. The
/// values account for scale + bias overhead on block-quantised formats
/// (q*_k), so `q4_k_m ≈ 0.58 bytes/param` rather than the nominal
/// `4 bits / 8 = 0.5`.
///
/// Matching is case-insensitive and accepts synonyms from the GGUF,
/// MLX-LM, AWQ, and GPTQ ecosystems. Unknown strings fall back to fp16
/// (2.0 bytes/param), matching pmetal's default model load path.
///
/// # Examples
/// ```
/// use pmetal_core::constants::bytes_per_param;
/// assert_eq!(bytes_per_param("fp16"), 2.0);
/// assert!((bytes_per_param("q4_k_m") - 0.58).abs() < 1e-9);
/// ```
pub fn bytes_per_param(quantization: &str) -> f64 {
    match quantization.to_lowercase().as_str() {
        "fp32" | "f32" | "float32" => 4.0,
        "fp16" | "f16" | "bf16" | "bfloat16" | "float16" | "" => 2.0,
        "fp8" | "f8" | "e4m3" | "e5m2" => 1.05,
        "q8_0" | "int8" | "8bit" | "mlx-8bit" | "w8a16" => 1.05,
        "q6_k" => 0.80,
        "q5_k_m" | "q5_k_s" | "q5_0" | "q5_1" => 0.68,
        "q4_k_m" | "q4_k_s" | "q4_0" | "q4_1" | "4bit" | "mlx-4bit" | "nf4" | "awq" | "gptq"
        | "w4a16" => 0.58,
        "q3_k_m" | "q3_k_s" | "q3_k_l" => 0.48,
        "q2_k" | "q2_k_s" | "2bit" => 0.37,
        _ => 2.0, // default to fp16
    }
}

/// Returns bytes per KV-cache element for a given cache quantisation bit
/// width.
///
/// `None` (default fp16) → `2.0`. Quantised paths add a small fixed
/// overhead for per-group scale / bias metadata:
///
/// | bits | bytes/elem | notes                            |
/// |------|-----------:|----------------------------------|
/// | `None` | 2.0      | fp16 / bf16 default              |
/// | 8    | 1.1        | q8: ~50 % reduction + overhead   |
/// | 4    | 0.6        | q4: ~75 % reduction + overhead   |
/// | 2    | 0.35       | q2: ~87 % reduction + overhead   |
///
/// Any other value falls back to fp16 — unrecognised bit widths are
/// treated the same as absent quantisation rather than silently producing
/// misleading size estimates.
pub fn kv_bytes_per_element(bits: Option<u8>) -> f64 {
    match bits {
        Some(8) => 1.1,
        Some(4) => 0.6,
        Some(2) => 0.35,
        _ => 2.0,
    }
}

/// Heuristically maps a Hugging Face model ID (e.g. `"TheBloke/Llama-2-7B-GGUF"`
/// or `"mlx-community/Qwen3-32B-4bit"`) to a canonical quantisation label
/// compatible with [`bytes_per_param`]. Matches the GGUF, MLX-LM, AWQ, and
/// GPTQ conventions; falls back to `"fp16"` when no marker is present.
///
/// This function is deliberately string-in / string-out so the result can
/// be passed straight into [`bytes_per_param`] for memory estimation,
/// carried through serialised [`ModelSpec`]-shaped structures, or shown to
/// the user in the TUI / CLI.
///
/// # Examples
/// ```
/// use pmetal_core::constants::detect_quantization_from_id;
/// assert_eq!(detect_quantization_from_id("meta-llama/Llama-2-7B"), "fp16");
/// assert_eq!(detect_quantization_from_id("mlx-community/Qwen3-32B-4bit"), "mlx-4bit");
/// assert_eq!(
///     detect_quantization_from_id("TheBloke/llama-2-7b-chat.Q4_K_M.gguf"),
///     "Q4_K_M",
/// );
/// ```
///
/// [`ModelSpec`]: ../../pmetal_hub/fit/struct.ModelSpec.html
pub fn detect_quantization_from_id(model_id: &str) -> String {
    let lower = model_id.to_lowercase();
    if lower.contains("q2_k") {
        "Q2_K".to_string()
    } else if lower.contains("q3_k") {
        "Q3_K_M".to_string()
    } else if lower.contains("q4_k") || lower.contains("q4_0") {
        "Q4_K_M".to_string()
    } else if lower.contains("q5_k") {
        "Q5_K_M".to_string()
    } else if lower.contains("q6_k") {
        "Q6_K".to_string()
    } else if lower.contains("q8_0") || lower.contains("int8") || lower.contains("8bit") {
        "Q8_0".to_string()
    } else if lower.contains("fp8") || lower.contains("f8") {
        "fp8".to_string()
    } else if lower.contains("4bit") || lower.contains("mlx-4bit") || lower.contains("w4a16") {
        "mlx-4bit".to_string()
    } else if lower.contains("gptq") {
        "gptq".to_string()
    } else if lower.contains("awq") {
        "awq".to_string()
    } else if lower.contains("gguf") {
        "Q4_K_M".to_string() // GGUF default assumption
    } else {
        "fp16".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_per_param_matches_audit_table() {
        // Reference table from the April 2026 BPP audit (pmetal-hub/src/fit.rs).
        for (q, expected) in [
            ("fp32", 4.0),
            ("fp16", 2.0),
            ("bf16", 2.0),
            ("fp8", 1.05),
            ("q8_0", 1.05),
            ("q6_k", 0.80),
            ("q5_k_m", 0.68),
            ("q4_k_m", 0.58),
            ("nf4", 0.58),
            ("awq", 0.58),
            ("q3_k_m", 0.48),
            ("q2_k", 0.37),
        ] {
            assert!(
                (bytes_per_param(q) - expected).abs() < 1e-9,
                "{q} → {}, expected {expected}",
                bytes_per_param(q),
            );
        }
    }

    #[test]
    fn bytes_per_param_default_is_fp16() {
        assert_eq!(bytes_per_param(""), 2.0);
        assert_eq!(bytes_per_param("totally-unknown-format"), 2.0);
    }

    #[test]
    fn bytes_per_param_case_insensitive() {
        assert_eq!(bytes_per_param("Q4_K_M"), bytes_per_param("q4_k_m"));
        assert_eq!(bytes_per_param("FP16"), bytes_per_param("fp16"));
    }

    #[test]
    fn kv_bytes_per_element_covers_quant_modes() {
        assert_eq!(kv_bytes_per_element(None), 2.0);
        assert_eq!(kv_bytes_per_element(Some(8)), 1.1);
        assert_eq!(kv_bytes_per_element(Some(4)), 0.6);
        assert_eq!(kv_bytes_per_element(Some(2)), 0.35);
        // Unknown bit widths fall back to fp16 rather than extrapolating.
        assert_eq!(kv_bytes_per_element(Some(3)), 2.0);
        assert_eq!(kv_bytes_per_element(Some(16)), 2.0);
    }

    #[test]
    fn detect_quantization_recognises_gguf_suffixes() {
        assert_eq!(
            detect_quantization_from_id("TheBloke/Llama-2-7B-Chat-GGUF.Q4_K_M"),
            "Q4_K_M"
        );
        assert_eq!(
            detect_quantization_from_id("TheBloke/Llama-2-7B-Chat-GGUF.q2_k"),
            "Q2_K"
        );
        assert_eq!(
            detect_quantization_from_id("unsloth/Llama-3-70B.Q8_0"),
            "Q8_0"
        );
    }

    #[test]
    fn detect_quantization_recognises_mlx_and_modern_formats() {
        assert_eq!(
            detect_quantization_from_id("mlx-community/Qwen3-32B-4bit"),
            "mlx-4bit"
        );
        assert_eq!(
            detect_quantization_from_id("some-org/llama-fp8-serving"),
            "fp8"
        );
        assert_eq!(
            detect_quantization_from_id("casperhansen/llama-2-7b-awq"),
            "awq"
        );
        assert_eq!(
            detect_quantization_from_id("TheBloke/Llama-2-7B-GPTQ"),
            "gptq"
        );
    }

    #[test]
    fn detect_quantization_default_is_fp16() {
        assert_eq!(detect_quantization_from_id("meta-llama/Llama-2-7B"), "fp16");
        assert_eq!(
            detect_quantization_from_id("mistralai/Mistral-7B-v0.1"),
            "fp16"
        );
    }

    #[test]
    fn detect_quantization_gguf_without_suffix_assumes_q4_k_m() {
        // Bare "GGUF" means "some GGUF file"; we pick the common default.
        assert_eq!(
            detect_quantization_from_id("TheBloke/Llama-2-7B-GGUF"),
            "Q4_K_M"
        );
    }

    #[test]
    fn detect_and_bpp_roundtrip_consistently() {
        // Every label `detect_quantization_from_id` can emit must be a
        // recognised key in `bytes_per_param`, so downstream memory
        // estimation never falls silently onto the fp16 default.
        let ids = [
            "foo.Q2_K",
            "foo.Q3_K_M",
            "foo.Q4_K_M",
            "foo.q5_k",
            "foo.Q6_K",
            "foo.Q8_0",
            "foo-fp8",
            "foo-4bit",
            "foo-gptq",
            "foo-awq",
            "foo.gguf",
        ];
        for id in ids {
            let label = detect_quantization_from_id(id);
            // Non-trivial answer (not the fp16 fallback) …
            assert_ne!(label, "fp16", "{id} should map to a specific quant");
            // …and a bytes-per-param entry exists for it.
            let bpp = bytes_per_param(&label);
            assert!(
                bpp < 2.0,
                "{id} ({label}) should have a sub-fp16 BPP; got {bpp}"
            );
        }
    }
}
