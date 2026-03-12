//! HuggingFace Hub model search API.
//!
//! Provides async search for models on HuggingFace Hub using the HTTP API,
//! with structured results including size, downloads, and architecture info.

use pmetal_core::{Result, SecretString};
use serde::{Deserialize, Serialize};

/// A search result from HuggingFace Hub.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfSearchResult {
    /// Model ID (e.g., "Qwen/Qwen3-0.6B").
    pub model_id: String,
    /// Total download count.
    pub downloads: u64,
    /// Like count.
    pub likes: u64,
    /// Pipeline tag (e.g., "text-generation").
    pub pipeline_tag: Option<String>,
    /// Last modified timestamp.
    pub last_modified: String,
    /// Total safetensors weight size in bytes (if available).
    pub safetensors_bytes: Option<u64>,
    /// Model tags.
    pub tags: Vec<String>,
    /// Estimated parameter count in billions (from safetensors size or tags).
    pub estimated_params_b: Option<f64>,
}

/// Detailed model metadata fetched from HuggingFace Hub.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfModelMeta {
    /// Model ID.
    pub model_id: String,
    /// Hidden size.
    pub hidden_size: Option<u64>,
    /// Number of transformer layers.
    pub num_hidden_layers: Option<u64>,
    /// FFN intermediate size.
    pub intermediate_size: Option<u64>,
    /// Vocabulary size.
    pub vocab_size: Option<u64>,
    /// Number of attention heads.
    pub num_attention_heads: Option<u64>,
    /// Number of KV heads (GQA/MQA).
    pub num_key_value_heads: Option<u64>,
    /// Head dimension.
    pub head_dim: Option<u64>,
    /// Model architecture type.
    pub model_type: Option<String>,
    /// Max sequence / position length.
    pub max_position_embeddings: Option<u64>,
    /// MoE: number of experts.
    pub num_local_experts: Option<u64>,
    /// MoE: active experts per token.
    pub num_experts_per_tok: Option<u64>,
    /// Estimated parameter count in billions.
    pub estimated_params_b: f64,
    /// Total safetensors size in bytes.
    pub safetensors_bytes: Option<u64>,
    /// Detected quantization format.
    pub quantization: String,
    /// Raw config.json as parsed Value (for building ModelSpec).
    pub config: serde_json::Value,
}

/// HF API response types (internal).
/// We parse the JSON manually to handle the varied response format robustly.
type HfApiResponse = Vec<serde_json::Value>;

/// Maximum response body size (4 MB) to prevent heap exhaustion from rogue API responses.
const MAX_RESPONSE_BYTES: usize = 4 * 1024 * 1024;

/// Read a bounded response body and deserialize as JSON.
async fn bounded_json<T: serde::de::DeserializeOwned>(
    resp: reqwest::Response,
) -> std::result::Result<T, pmetal_core::PMetalError> {
    let bytes = resp
        .bytes()
        .await
        .map_err(|e| pmetal_core::PMetalError::Hub(format!("Failed to read response: {e}")))?;
    if bytes.len() > MAX_RESPONSE_BYTES {
        return Err(pmetal_core::PMetalError::Hub(format!(
            "Response too large ({} bytes, max {})",
            bytes.len(),
            MAX_RESPONSE_BYTES
        )));
    }
    serde_json::from_slice(&bytes)
        .map_err(|e| pmetal_core::PMetalError::Hub(format!("Failed to parse JSON: {e}")))
}

/// Validate that a model ID looks like a valid HF model path (org/name).
/// Rejects path traversal, URL injection, and other malformed values.
fn is_valid_model_id(id: &str) -> bool {
    let parts: Vec<&str> = id.splitn(2, '/').collect();
    parts.len() == 2
        && parts.iter().all(|p| {
            !p.is_empty()
                && p.chars()
                    .all(|c| c.is_alphanumeric() || matches!(c, '-' | '_' | '.'))
        })
}

/// Build HTTP client with optional HF token.
fn build_client(token: Option<&SecretString>) -> Result<reqwest::Client> {
    let mut builder = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .user_agent(concat!("pmetal/", env!("CARGO_PKG_VERSION")));

    if let Some(secret) = token {
        use reqwest::header;
        let mut headers = header::HeaderMap::new();
        let val = header::HeaderValue::from_str(&format!("Bearer {}", secret.expose_secret()))
            .map_err(|e| pmetal_core::PMetalError::Hub(format!("Invalid token: {e}")))?;
        headers.insert(header::AUTHORIZATION, val);
        builder = builder.default_headers(headers);
    }

    builder
        .build()
        .map_err(|e| pmetal_core::PMetalError::Hub(e.to_string()))
}

/// Search HuggingFace Hub for text-generation models.
///
/// Returns results sorted by download count (most popular first).
pub async fn search_models(
    query: &str,
    limit: usize,
    token: Option<&SecretString>,
) -> Result<Vec<HfSearchResult>> {
    let client = build_client(token)?;

    let url = format!(
        "https://huggingface.co/api/models?search={}&filter=text-generation&sort=downloads&direction=-1&limit={}&full=true",
        urlencoding::encode(query),
        limit.min(50)
    );

    let response = client
        .get(&url)
        .send()
        .await
        .map_err(|e| pmetal_core::PMetalError::Hub(format!("Search request failed: {e}")))?;

    if !response.status().is_success() {
        return Err(pmetal_core::PMetalError::Hub(format!(
            "HF API returned status {}",
            response.status()
        )));
    }

    let models: HfApiResponse = bounded_json(response).await?;

    let results = models
        .into_iter()
        .filter_map(|m| {
            let model_id = m
                .get("modelId")
                .or_else(|| m.get("id"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            // Skip results with invalid/malformed model IDs
            if !is_valid_model_id(&model_id) {
                return None;
            }

            let downloads = m.get("downloads").and_then(|v| v.as_u64()).unwrap_or(0);

            let likes = m.get("likes").and_then(|v| v.as_u64()).unwrap_or(0);

            let pipeline_tag = m
                .get("pipeline_tag")
                .and_then(|v| v.as_str())
                .map(String::from);

            let last_modified = m
                .get("lastModified")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let tags: Vec<String> = m
                .get("tags")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();

            let safetensors_bytes = m
                .get("safetensors")
                .and_then(|s| s.get("total"))
                .and_then(|t| t.as_u64());

            // Estimate params: try safetensors size first, then parse from model name
            let estimated_params_b = safetensors_bytes
                .map(|bytes| bytes as f64 / 2.0 / 1e9)
                .or_else(|| estimate_params_from_name(&model_id));

            Some(HfSearchResult {
                model_id,
                downloads,
                likes,
                pipeline_tag,
                last_modified,
                safetensors_bytes,
                tags,
                estimated_params_b,
            })
        })
        .collect();

    Ok(results)
}

/// Fetch detailed model metadata including config.json.
///
/// Makes two requests: one for the model info (safetensors size) and one for config.json.
pub async fn fetch_model_meta(model_id: &str, token: Option<&SecretString>) -> Result<HfModelMeta> {
    if !is_valid_model_id(model_id) {
        return Err(pmetal_core::PMetalError::Hub(format!(
            "Invalid model ID: {model_id}"
        )));
    }

    let client = build_client(token)?;

    // Fetch config.json
    let config_url = format!(
        "https://huggingface.co/{}/resolve/main/config.json",
        model_id
    );

    let config: serde_json::Value = match client.get(&config_url).send().await {
        Ok(resp) if resp.status().is_success() => bounded_json(resp).await.unwrap_or_default(),
        _ => serde_json::Value::Object(serde_json::Map::new()),
    };

    // Fetch model info for safetensors size
    let info_url = format!("https://huggingface.co/api/models/{}", model_id);
    let safetensors_bytes = match client.get(&info_url).send().await {
        Ok(resp) if resp.status().is_success() => {
            let info: serde_json::Value = bounded_json(resp).await.unwrap_or_default();
            info.get("safetensors")
                .and_then(|s| s.get("total"))
                .and_then(|t| t.as_u64())
        }
        _ => None,
    };

    let model_spec = crate::fit::model_spec_from_config(&config, safetensors_bytes);
    let quantization = crate::fit::detect_quantization_from_id(model_id);

    let quantization = if model_spec.quantization != "fp16" {
        model_spec.quantization.clone()
    } else {
        quantization
    };

    Ok(HfModelMeta {
        model_id: model_id.to_string(),
        hidden_size: config["hidden_size"].as_u64(),
        num_hidden_layers: config["num_hidden_layers"].as_u64(),
        intermediate_size: config["intermediate_size"].as_u64(),
        vocab_size: config["vocab_size"].as_u64(),
        num_attention_heads: config["num_attention_heads"].as_u64(),
        num_key_value_heads: config["num_key_value_heads"].as_u64(),
        head_dim: config["head_dim"].as_u64(),
        model_type: config["model_type"].as_str().map(String::from),
        max_position_embeddings: config["max_position_embeddings"].as_u64(),
        num_local_experts: config["num_local_experts"].as_u64(),
        num_experts_per_tok: config["num_experts_per_tok"].as_u64(),
        estimated_params_b: model_spec.params_b,
        safetensors_bytes,
        quantization,
        config,
    })
}

/// Estimate parameter count from a model name/ID.
///
/// Parses common patterns like "Qwen3-0.6B", "Llama-3.2-1B", "Mistral-7B-Instruct".
fn estimate_params_from_name(model_id: &str) -> Option<f64> {
    let name = model_id.rsplit('/').next().unwrap_or(model_id);
    let lower = name.to_lowercase();

    // Find patterns like "0.6b", "7b", "70b", "1.5b", "8x7b"
    // Scan for a number followed by 'b' (case insensitive)
    let chars: Vec<char> = lower.chars().collect();
    let len = chars.len();

    for i in 0..len {
        // Look for digit (possibly with decimal) followed by 'b'
        if !chars[i].is_ascii_digit() {
            continue;
        }

        // Collect the number
        let start = i;
        let mut j = i;
        while j < len && (chars[j].is_ascii_digit() || chars[j] == '.') {
            j += 1;
        }

        // Check if followed by 'b'
        if j < len && chars[j] == 'b' {
            // Make sure the 'b' isn't part of a longer word (like "base", "bert")
            let after_b = j + 1;
            let is_boundary = after_b >= len || !chars[after_b].is_ascii_alphabetic();

            // Check it's actually a param count (not version like "3.2")
            // Param counts typically appear after a hyphen or start of name
            let before_ok = start == 0
                || chars[start - 1] == '-'
                || chars[start - 1] == '_'
                || chars[start - 1] == 'x'; // for MoE like "8x7b"

            if is_boundary && before_ok {
                let num_str: String = chars[start..j].iter().collect();
                if let Ok(val) = num_str.parse::<f64>() {
                    // Check for MoE multiplier (e.g., "8x7b")
                    if start >= 2 && chars[start - 1] == 'x' {
                        let mut k = start - 2;
                        while k > 0 && chars[k].is_ascii_digit() {
                            k -= 1;
                        }
                        if !chars[k].is_ascii_digit() {
                            k += 1;
                        }
                        let mult_str: String = chars[k..start - 1].iter().collect();
                        if let Ok(mult) = mult_str.parse::<f64>() {
                            return Some(val * mult);
                        }
                    }
                    return Some(val);
                }
            }
        }
    }
    None
}

/// Format a download count for display (e.g., 1234567 → "1.2M").
pub fn format_downloads(count: u64) -> String {
    if count >= 1_000_000_000 {
        format!("{:.1}B", count as f64 / 1e9)
    } else if count >= 1_000_000 {
        format!("{:.1}M", count as f64 / 1e6)
    } else if count >= 1_000 {
        format!("{:.1}K", count as f64 / 1e3)
    } else {
        count.to_string()
    }
}
