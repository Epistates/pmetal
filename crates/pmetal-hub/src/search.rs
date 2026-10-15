//! HuggingFace Hub model search, model metadata and model cards.
//!
//! Search asks the Hub for exactly the fields it needs (tensor dtypes, GGUF
//! summary, tags, downloads) in one request, so each result can be sized for
//! this Mac without opening the repository. Detailed metadata reads the
//! model's `config.json` as well, for an exact KV-cache layout.

use std::collections::BTreeMap;

use pmetal_core::{PMetalError, Result, SecretString};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::api::get_json;
use crate::client::HubOptions;
use crate::fit::{
    DeviceSpec, FitEstimate, ModelSpec, detect_quantization_from_id, estimate_fit,
    model_spec_from_config,
};
use crate::repo::{RepoHandle, validate_repo_id};
use crate::select::{GgufVariant, gguf_variants};
use crate::tensors::dtype_bits;

/// A search result from HuggingFace Hub.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfSearchResult {
    /// Model ID (e.g., "Qwen/Qwen3-0.6B").
    pub model_id: String,
    /// Downloads in the last 30 days.
    pub downloads: u64,
    /// Like count.
    pub likes: u64,
    /// Pipeline tag (e.g., "text-generation").
    pub pipeline_tag: Option<String>,
    /// Last modified timestamp.
    pub last_modified: String,
    /// Size of the safetensors weights in bytes, summed over the tensors'
    /// element types as the Hub reports them (`None` for repositories with no
    /// safetensors, such as GGUF-only ones).
    pub safetensors_bytes: Option<u64>,
    /// Model tags.
    pub tags: Vec<String>,
    /// Parameter count in billions: the Hub's own count when it has one,
    /// else read off the model name. Packed quantized checkpoints count their
    /// storage words, so prefer `safetensors_bytes` for memory.
    pub estimated_params_b: Option<f64>,
    /// Parameter count as the Hub reports it (safetensors or GGUF).
    #[serde(default)]
    pub parameters: Option<u64>,
    /// The library the model is for (`transformers`, `mlx`, ...).
    #[serde(default)]
    pub library_name: Option<String>,
    /// License id from the model's tags (e.g. `apache-2.0`).
    #[serde(default)]
    pub license: Option<String>,
    /// Whether access requires accepting terms first.
    #[serde(default)]
    pub gated: bool,
    /// For a GGUF repository, the model architecture it declares.
    #[serde(default)]
    pub gguf_architecture: Option<String>,
    /// For a GGUF repository, the context length it declares.
    #[serde(default)]
    pub gguf_context_length: Option<u64>,
}

impl HfSearchResult {
    /// A `ModelSpec` for this result at `context_length` tokens: the Hub's
    /// measured weight size when it has one, the declared context, a mixture
    /// of experts recognized from its tags or `…-A3B` name.
    pub fn model_spec(&self, context_length: u32) -> ModelSpec {
        let params_b = self.estimated_params_b.unwrap_or(0.0);
        let mut spec = ModelSpec::new(params_b, detect_quantization_from_id(&self.model_id))
            .with_context(context_length);
        spec.weight_bytes = self.safetensors_bytes;
        spec.max_context_length = self
            .gguf_context_length
            .map(|c| c.min(u64::from(u32::MAX)) as u32);
        if let Some((total, active)) = moe_sizes_from_name(&self.model_id) {
            spec = spec.with_active_params(active);
            if params_b <= 0.0 {
                spec.params_b = total;
            }
        } else if self.tags.iter().any(|t| t == "moe") {
            spec.is_moe = true;
        }
        spec
    }

    /// How this result would run on `device` at `context_length` tokens.
    /// `None` when the Hub gave no size at all (neither tensor sizes nor a
    /// parameter count, and none in the name): there is nothing to estimate.
    pub fn fit(&self, device: &DeviceSpec, context_length: u32) -> Option<FitEstimate> {
        let spec = self.model_spec(context_length);
        (spec.weight_bytes.is_some() || spec.params_b > 0.0).then(|| estimate_fit(&spec, device))
    }
}

/// How to order search results.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SearchSort {
    /// Most downloaded in the last 30 days first.
    #[default]
    Downloads,
    /// Most liked first.
    Likes,
    /// Trending first.
    Trending,
    /// Most recently updated first.
    Updated,
}

impl SearchSort {
    fn api_name(self) -> &'static str {
        match self {
            SearchSort::Downloads => "downloads",
            SearchSort::Likes => "likes",
            SearchSort::Trending => "trendingScore",
            SearchSort::Updated => "lastModified",
        }
    }
}

/// A model search.
#[derive(Debug, Clone)]
pub struct SearchOptions {
    /// Text matched against model names.
    pub query: String,
    /// Most results to return (at most 100).
    pub limit: usize,
    /// Pipeline tag to filter by, e.g. `text-generation`. `None` does not
    /// filter on it.
    pub pipeline_tag: Option<String>,
    /// Only models from this user or organization.
    pub author: Option<String>,
    /// Extra tags every result must have: `gguf`, `mlx`, `license:apache-2.0`,
    /// `moe`, a language, ...
    pub tags: Vec<String>,
    /// Result order.
    pub sort: SearchSort,
    /// Connection settings (token, endpoint, retries).
    pub hub: HubOptions,
}

impl SearchOptions {
    /// A text-generation search for `query` with the default limit and order.
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            limit: 20,
            pipeline_tag: Some("text-generation".to_owned()),
            author: None,
            tags: Vec::new(),
            sort: SearchSort::Downloads,
            hub: HubOptions::default(),
        }
    }
}

/// The largest search response read.
const MAX_SEARCH_BYTES: usize = 16 * 1024 * 1024;
/// The most results the Hub is asked for.
const MAX_RESULTS: usize = 100;

/// The fields a result needs; asking for them by name keeps responses small
/// and includes the ones `full` leaves out (tensor dtypes, GGUF summary).
const SEARCH_FIELDS: &[&str] = &[
    "downloads",
    "likes",
    "tags",
    "pipeline_tag",
    "lastModified",
    "library_name",
    "gated",
    "safetensors",
    "gguf",
];

fn search_query(options: &SearchOptions) -> Vec<(String, String)> {
    let mut query = vec![
        ("search".to_owned(), options.query.clone()),
        (
            "limit".to_owned(),
            options.limit.clamp(1, MAX_RESULTS).to_string(),
        ),
        ("sort".to_owned(), options.sort.api_name().to_owned()),
        ("direction".to_owned(), "-1".to_owned()),
    ];
    if let Some(tag) = &options.pipeline_tag {
        query.push(("pipeline_tag".to_owned(), tag.clone()));
    }
    if let Some(author) = &options.author {
        query.push(("author".to_owned(), author.clone()));
    }
    for tag in &options.tags {
        query.push(("filter".to_owned(), tag.clone()));
    }
    for field in SEARCH_FIELDS {
        query.push(("expand[]".to_owned(), (*field).to_owned()));
    }
    query
}

/// Search HuggingFace Hub for models.
///
/// Returns results sorted by download count (most popular first). See
/// [`search_models_with`] for filters.
pub async fn search_models(
    query: &str,
    limit: usize,
    token: Option<&SecretString>,
) -> Result<Vec<HfSearchResult>> {
    search_models_with_tag(query, limit, token, None).await
}

/// Search HuggingFace Hub for models, optionally filtered by pipeline tag.
///
/// `pipeline_tag` filters results by pipeline type. Pass `None` to use the
/// default `"text-generation"` filter, or `Some("text-to-speech")` etc. to
/// search a different pipeline category.
///
/// Returns results sorted by download count (most popular first).
pub async fn search_models_with_tag(
    query: &str,
    limit: usize,
    token: Option<&SecretString>,
    pipeline_tag: Option<&str>,
) -> Result<Vec<HfSearchResult>> {
    let mut options = SearchOptions::new(query);
    options.limit = limit;
    options.hub = HubOptions::from_token(token);
    if let Some(tag) = pipeline_tag {
        options.pipeline_tag = Some(tag.to_owned());
    }
    search_models_with(&options).await
}

/// Search HuggingFace Hub for models with explicit [`SearchOptions`].
pub async fn search_models_with(options: &SearchOptions) -> Result<Vec<HfSearchResult>> {
    let body = get_json(
        &options.hub,
        "/api/models",
        &search_query(options),
        MAX_SEARCH_BYTES,
        Some("model search"),
    )
    .await?;
    parse_search_results(&body)
}

/// Parse a model-list response. Entries with malformed ids are skipped, and a
/// field that is missing or the wrong type is absent, not an error: the Hub
/// adds and renames fields, and one odd entry must not hide the rest.
pub(crate) fn parse_search_results(body: &[u8]) -> Result<Vec<HfSearchResult>> {
    let entries: Vec<Value> = serde_json::from_slice(body).map_err(|e| {
        PMetalError::Hub(format!(
            "the Hub's search response is not a model list: {e}"
        ))
    })?;
    Ok(entries.iter().filter_map(parse_search_entry).collect())
}

fn parse_search_entry(entry: &Value) -> Option<HfSearchResult> {
    let model_id = entry
        .get("id")
        .or_else(|| entry.get("modelId"))
        .and_then(Value::as_str)?
        .to_owned();
    // Anything that is not a plain repository id could mean something else
    // in a path or URL downstream.
    validate_repo_id(&model_id).ok()?;
    let number = |key: &str| entry.get(key).and_then(Value::as_u64).unwrap_or(0);
    let tags: Vec<String> = entry
        .get("tags")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|t| t.as_str().map(str::to_owned))
                .collect()
        })
        .unwrap_or_default();

    let safetensors = entry.get("safetensors");
    let by_dtype: BTreeMap<String, u64> = safetensors
        .and_then(|s| s.get("parameters"))
        .and_then(Value::as_object)
        .map(|m| {
            m.iter()
                .filter_map(|(k, v)| Some((k.clone(), v.as_u64()?)))
                .collect()
        })
        .unwrap_or_default();
    let safetensors_bytes = bytes_from_dtype_params(&by_dtype);
    let safetensors_total = safetensors
        .and_then(|s| s.get("total"))
        .and_then(Value::as_u64);
    let gguf = entry.get("gguf");
    let gguf_total = gguf.and_then(|g| g.get("total")).and_then(Value::as_u64);
    let parameters = safetensors_total.or(gguf_total);

    Some(HfSearchResult {
        downloads: number("downloads"),
        likes: number("likes"),
        pipeline_tag: entry
            .get("pipeline_tag")
            .and_then(Value::as_str)
            .map(str::to_owned),
        last_modified: entry
            .get("lastModified")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        safetensors_bytes,
        estimated_params_b: parameters
            .map(|p| p as f64 / 1e9)
            .or_else(|| estimate_params_from_name(&model_id)),
        parameters,
        library_name: entry
            .get("library_name")
            .and_then(Value::as_str)
            .map(str::to_owned),
        license: license_from_tags(&tags),
        gated: entry
            .get("gated")
            .is_some_and(|g| g.as_bool().unwrap_or_else(|| g.is_string())),
        gguf_architecture: gguf
            .and_then(|g| g.get("architecture"))
            .and_then(Value::as_str)
            .map(str::to_owned),
        gguf_context_length: gguf
            .and_then(|g| g.get("context_length"))
            .and_then(Value::as_u64),
        tags,
        model_id,
    })
}

/// Bytes of tensor data given parameter counts by element type, as the Hub
/// reports them for safetensors. `None` if there are none or one has a type
/// this crate does not know the width of.
fn bytes_from_dtype_params(by_dtype: &BTreeMap<String, u64>) -> Option<u64> {
    if by_dtype.is_empty() {
        return None;
    }
    by_dtype.iter().try_fold(0u64, |total, (dtype, count)| {
        let bits = dtype_bits(dtype)?;
        total.checked_add(count.checked_mul(bits)?.div_ceil(8))
    })
}

fn license_from_tags(tags: &[String]) -> Option<String> {
    tags.iter()
        .find_map(|t| t.strip_prefix("license:"))
        .map(str::to_owned)
}

/// Detailed model metadata fetched from HuggingFace Hub.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfModelMeta {
    /// Model ID.
    pub model_id: String,
    /// The commit these facts are from.
    #[serde(default)]
    pub commit: Option<String>,
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
    /// Size of the safetensors weights in bytes, from the tensors' element
    /// types (`None` for repositories with none).
    pub safetensors_bytes: Option<u64>,
    /// Detected quantization format.
    pub quantization: String,
    /// License id from the model's tags.
    #[serde(default)]
    pub license: Option<String>,
    /// Whether access requires accepting terms first.
    #[serde(default)]
    pub gated: bool,
    /// The library the model is for.
    #[serde(default)]
    pub library_name: Option<String>,
    /// Raw config.json as parsed Value (for building ModelSpec).
    pub config: Value,
}

impl HfModelMeta {
    /// A `ModelSpec` at `context_length` tokens from the real `config.json`
    /// (exact KV layout, experts) and the measured weight size.
    pub fn model_spec(&self, context_length: u32) -> ModelSpec {
        let mut spec = model_spec_from_config(&self.config, self.safetensors_bytes);
        if spec.params_b <= 0.0 {
            spec.params_b = self.estimated_params_b;
        }
        if self.config.as_object().is_none_or(|c| c.is_empty()) {
            spec.quantization = detect_quantization_from_id(&self.model_id);
        }
        spec.with_context(context_length)
    }
}

/// The largest `config.json` read.
const MAX_CONFIG_BYTES: usize = 4 * 1024 * 1024;
/// The largest README read.
const MAX_README_BYTES: usize = 2 * 1024 * 1024;

/// Fetch detailed model metadata including config.json.
pub async fn fetch_model_meta(model_id: &str, token: Option<&SecretString>) -> Result<HfModelMeta> {
    fetch_model_meta_with(model_id, &HubOptions::from_token(token)).await
}

/// [`fetch_model_meta`] with explicit connection settings.
///
/// Makes two requests: the repository info (tensor dtypes, tags, access) and
/// its `config.json` at the same commit. A repository with no `config.json`
/// (GGUF-only ones) yields empty config fields, not an error.
pub async fn fetch_model_meta_with(model_id: &str, options: &HubOptions) -> Result<HfModelMeta> {
    validate_repo_id(model_id)?;
    let client = options.client()?;
    let repo = RepoHandle::model(&client, model_id)?;
    let expand = [
        "sha",
        "safetensors",
        "gguf",
        "gated",
        "tags",
        "library_name",
    ];
    let body = get_json(
        options,
        &format!("/api/models/{model_id}"),
        &expand
            .iter()
            .map(|f| ("expand[]".to_owned(), (*f).to_owned()))
            .collect::<Vec<_>>(),
        MAX_SEARCH_BYTES,
        Some(model_id),
    )
    .await?;
    let info: Value = serde_json::from_slice(&body).map_err(|e| {
        PMetalError::Hub(format!(
            "the Hub's model info for {model_id} is unreadable: {e}"
        ))
    })?;
    let commit = info.get("sha").and_then(Value::as_str).map(str::to_owned);

    // The config at the commit the info describes; none is fine.
    let config = match &commit {
        Some(commit) => match repo
            .read_file(commit, "config.json", MAX_CONFIG_BYTES)
            .await
        {
            Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_else(|e| {
                tracing::warn!("{model_id}: config.json is not valid JSON: {e}");
                Value::Object(Default::default())
            }),
            Err(e) => {
                tracing::debug!("{model_id}: no readable config.json ({e})");
                Value::Object(Default::default())
            }
        },
        None => Value::Object(Default::default()),
    };

    let summary = parse_search_entry(&info);
    let spec = model_spec_from_config(&config, summary.as_ref().and_then(|s| s.safetensors_bytes));
    let quantization = if spec.quantization != "fp16" {
        spec.quantization.clone()
    } else {
        detect_quantization_from_id(model_id)
    };
    let params_b = if spec.params_b > 0.0 {
        spec.params_b
    } else {
        summary
            .as_ref()
            .and_then(|s| s.estimated_params_b)
            .unwrap_or(0.0)
    };
    let get = |key: &str| config.get(key).and_then(Value::as_u64);
    Ok(HfModelMeta {
        model_id: model_id.to_owned(),
        commit,
        hidden_size: get("hidden_size"),
        num_hidden_layers: get("num_hidden_layers"),
        intermediate_size: get("intermediate_size"),
        vocab_size: get("vocab_size"),
        num_attention_heads: get("num_attention_heads"),
        num_key_value_heads: get("num_key_value_heads"),
        head_dim: get("head_dim"),
        model_type: config
            .get("model_type")
            .and_then(Value::as_str)
            .map(str::to_owned),
        max_position_embeddings: get("max_position_embeddings"),
        num_local_experts: get("num_local_experts"),
        num_experts_per_tok: get("num_experts_per_tok"),
        estimated_params_b: params_b,
        safetensors_bytes: summary.as_ref().and_then(|s| s.safetensors_bytes),
        quantization,
        license: summary.as_ref().and_then(|s| s.license.clone()),
        gated: summary.as_ref().is_some_and(|s| s.gated),
        library_name: summary.and_then(|s| s.library_name),
        config,
    })
}

/// The GGUF quantizations a repository offers, smallest first, with their
/// sizes: what to choose from to pull one quant out of a multi-quant
/// repository. A split model counts once, with all its shards.
pub async fn list_gguf_variants(model_id: &str, options: &HubOptions) -> Result<Vec<GgufVariant>> {
    let client = options.client()?;
    let repo = RepoHandle::model(&client, model_id)?;
    let commit = repo
        .resolve_commit(None)
        .await
        .map_err(|e| repo.error(&e))?;
    let files = repo.list_files(&commit).await.map_err(|e| repo.error(&e))?;
    Ok(gguf_variants(&files))
}

/// A model's README: the front matter the Hub reads (license, languages,
/// base model, ...) and the text after it.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelCard {
    /// The YAML front matter as JSON (`null` when there is none).
    pub front_matter: Value,
    /// The README after the front matter.
    pub body: String,
    /// The license the card declares (`license:` in the front matter).
    pub license: Option<String>,
    /// Models this one was derived from (`base_model:`).
    pub base_models: Vec<String>,
}

impl ModelCard {
    /// Parse the text of a README.md.
    pub fn parse(readme: &str) -> Self {
        let (front, body) = split_front_matter(readme);
        let front_matter = front
            .and_then(|yaml| serde_saphyr::from_str::<Value>(yaml).ok())
            .unwrap_or(Value::Null);
        let strings = |key: &str| -> Vec<String> {
            match front_matter.get(key) {
                Some(Value::String(s)) => vec![s.clone()],
                Some(Value::Array(a)) => a
                    .iter()
                    .filter_map(|v| v.as_str().map(str::to_owned))
                    .collect(),
                _ => Vec::new(),
            }
        };
        Self {
            license: strings("license").into_iter().next(),
            base_models: strings("base_model"),
            front_matter,
            body: body.to_owned(),
        }
    }
}

/// Split a README into its YAML front matter and the rest.
fn split_front_matter(readme: &str) -> (Option<&str>, &str) {
    let text = readme.strip_prefix('\u{feff}').unwrap_or(readme);
    let Some(rest) = text
        .strip_prefix("---\n")
        .or_else(|| text.strip_prefix("---\r\n"))
    else {
        return (None, text);
    };
    // The closing fence is a line of its own.
    let mut offset = 0;
    for line in rest.split_inclusive('\n') {
        if line.trim_end() == "---" {
            return (Some(&rest[..offset]), &rest[offset + line.len()..]);
        }
        offset += line.len();
    }
    (None, text)
}

/// Fetch and parse a model's README.md at its default branch.
pub async fn fetch_model_card(model_id: &str, options: &HubOptions) -> Result<ModelCard> {
    let client = options.client()?;
    let repo = RepoHandle::model(&client, model_id)?;
    let commit = repo
        .resolve_commit(None)
        .await
        .map_err(|e| repo.error(&e))?;
    let bytes = repo
        .read_file(&commit, "README.md", MAX_README_BYTES)
        .await?;
    Ok(ModelCard::parse(&String::from_utf8_lossy(&bytes)))
}

/// Parameter counts from a name written `…-30B-A3B…` (30B total, 3B active).
pub(crate) fn moe_sizes_from_name(model_id: &str) -> Option<(f64, f64)> {
    let name = model_id
        .rsplit('/')
        .next()
        .unwrap_or(model_id)
        .to_ascii_lowercase();
    let tokens: Vec<&str> = name.split(['-', '_']).collect();
    tokens.windows(2).find_map(|pair| {
        let total = pair[0].strip_suffix('b')?.parse::<f64>().ok()?;
        let active = pair[1]
            .strip_prefix('a')?
            .strip_suffix('b')?
            .parse::<f64>()
            .ok()?;
        (active > 0.0 && active < total).then_some((total, active))
    })
}

/// Estimate parameter count from a model name/ID.
///
/// Parses common patterns like "Qwen3-0.6B", "Llama-3.2-1B", "Mistral-7B-Instruct".
pub(crate) fn estimate_params_from_name(model_id: &str) -> Option<f64> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_result_is_read_with_its_real_weight_size() {
        // Shapes taken from live Hub responses: `total` is a parameter count
        // (not bytes), `parameters` is keyed by element type.
        let body = json!([{
            "id": "Qwen/Qwen3-0.6B",
            "downloads": 30452693u64,
            "likes": 1000,
            "pipeline_tag": "text-generation",
            "lastModified": "2026-04-20T02:45:58.000Z",
            "library_name": "transformers",
            "gated": false,
            "tags": ["transformers", "safetensors", "license:apache-2.0", "text-generation"],
            "safetensors": {"parameters": {"BF16": 751632384u64}, "total": 751632384u64},
        }]);
        let results = parse_search_results(body.to_string().as_bytes()).unwrap();
        let r = &results[0];
        assert_eq!(r.model_id, "Qwen/Qwen3-0.6B");
        assert_eq!(r.parameters, Some(751_632_384));
        // 751,632,384 bf16 parameters are 1.5 GB, not 0.75.
        assert_eq!(r.safetensors_bytes, Some(1_503_264_768));
        assert!((r.estimated_params_b.unwrap() - 0.7516).abs() < 1e-3);
        assert_eq!(r.license.as_deref(), Some("apache-2.0"));
        assert!(!r.gated);
    }

    #[test]
    fn mixed_precision_and_packed_weights_are_summed_by_element_width() {
        // A 4-bit checkpoint: packed words (U32) plus bf16 scales and biases.
        let body = json!([{
            "id": "mlx-community/m-4bit",
            "safetensors": {"parameters": {"U32": 100u64, "BF16": 50u64, "F32": 10u64}, "total": 160u64},
        }]);
        let r = &parse_search_results(body.to_string().as_bytes()).unwrap()[0];
        assert_eq!(r.safetensors_bytes, Some(100 * 4 + 50 * 2 + 10 * 4));
    }

    #[test]
    fn a_gguf_repository_has_parameters_but_no_weight_size() {
        let body = json!([{
            "id": "bartowski/Qwen_Qwen3-0.6B-GGUF",
            "gated": "manual",
            "gguf": {"total": 596049920u64, "architecture": "qwen3", "context_length": 40960,
                     "chat_template": "{{ long template }}"},
        }]);
        let r = &parse_search_results(body.to_string().as_bytes()).unwrap()[0];
        assert_eq!(r.safetensors_bytes, None);
        assert_eq!(r.parameters, Some(596_049_920));
        assert_eq!(r.gguf_architecture.as_deref(), Some("qwen3"));
        assert_eq!(r.gguf_context_length, Some(40960));
        assert!(r.gated, "manual approval is gated");
    }

    #[test]
    fn odd_entries_are_skipped_or_defaulted_not_fatal() {
        let body = json!([
            {"id": "ok/model", "downloads": "not a number", "tags": "nope"},
            {"id": "../../etc/passwd"},
            {"id": "a/b?x=1"},
            {"no_id": true},
            {"id": "other/model-7B"},
        ]);
        let results = parse_search_results(body.to_string().as_bytes()).unwrap();
        let ids: Vec<_> = results.iter().map(|r| r.model_id.as_str()).collect();
        assert_eq!(ids, ["ok/model", "other/model-7B"]);
        assert_eq!(results[0].downloads, 0);
        assert!(results[0].tags.is_empty());
        // Without the Hub's count, the name says 7B.
        assert_eq!(results[1].estimated_params_b, Some(7.0));
    }

    #[test]
    fn an_error_body_is_an_error() {
        assert!(parse_search_results(br#"{"error":"nope"}"#).is_err());
        assert!(parse_search_results(b"<html>").is_err());
    }

    #[test]
    fn the_request_asks_for_fields_and_filters_by_tag() {
        let mut options = SearchOptions::new("qwen3");
        options.limit = 500;
        options.author = Some("Qwen".into());
        options.tags = vec!["gguf".into(), "license:apache-2.0".into()];
        let query = search_query(&options);
        let has = |k: &str, v: &str| query.iter().any(|(a, b)| a == k && b == v);
        assert!(has("search", "qwen3"));
        assert!(has("limit", "100"), "the limit is capped");
        assert!(has("pipeline_tag", "text-generation"));
        assert!(has("author", "Qwen"));
        assert!(has("filter", "gguf") && has("filter", "license:apache-2.0"));
        assert!(has("expand[]", "safetensors") && has("expand[]", "gguf"));
        assert!(
            !query.iter().any(|(k, _)| k == "full"),
            "full and expand are exclusive"
        );
    }

    #[test]
    fn a_result_is_sized_from_its_measured_weights() {
        let r = HfSearchResult {
            model_id: "o/Qwen3-30B-A3B-4bit".into(),
            downloads: 0,
            likes: 0,
            pipeline_tag: None,
            last_modified: String::new(),
            safetensors_bytes: Some(17_000_000_000),
            tags: vec![],
            estimated_params_b: Some(5.0), // packed words, not weights
            parameters: Some(5_000_000_000),
            library_name: None,
            license: None,
            gated: false,
            gguf_architecture: None,
            gguf_context_length: None,
        };
        let device = DeviceSpec::from_device(64_000_000_000, 400.0, true);
        let fit = r.fit(&device, 4096).unwrap();
        assert!(fit.weights_measured);
        assert!((fit.weights_gb - 17.0).abs() < 1e-9);
        // `A3B` names the active size: 3B of 30B.
        let spec = r.model_spec(4096);
        assert!(spec.is_moe);
        assert_eq!(spec.active_params_b, Some(3.0));
    }

    #[test]
    fn moe_sizes_come_from_names_that_state_them() {
        assert_eq!(
            moe_sizes_from_name("Qwen/Qwen3-30B-A3B-Instruct"),
            Some((30.0, 3.0))
        );
        assert_eq!(
            moe_sizes_from_name("zai/GLM-4.5-Air-106B-A12B"),
            Some((106.0, 12.0))
        );
        assert_eq!(moe_sizes_from_name("meta/Llama-3.1-8B"), None);
        assert_eq!(
            moe_sizes_from_name("x/y-7B-a9b"),
            None,
            "active above total is not a split"
        );
    }

    #[test]
    fn parameter_counts_are_read_off_names() {
        for (id, params) in [
            ("Qwen/Qwen3-0.6B", Some(0.6)),
            ("meta-llama/Llama-3.2-1B", Some(1.0)),
            ("mistralai/Mistral-7B-Instruct-v0.3", Some(7.0)),
            ("mistralai/Mixtral-8x7B-v0.1", Some(56.0)),
            ("bert-base-uncased", None),
        ] {
            assert_eq!(estimate_params_from_name(id), params, "{id}");
        }
    }

    #[test]
    fn download_counts_are_abbreviated() {
        assert_eq!(format_downloads(999), "999");
        assert_eq!(format_downloads(1_234), "1.2K");
        assert_eq!(format_downloads(30_452_693), "30.5M");
    }

    #[test]
    fn a_model_card_yields_license_and_base_model() {
        let readme = "---\nlicense: apache-2.0\nbase_model:\n- Qwen/Qwen3-0.6B-Base\ntags:\n- a\n---\n# Title\n\nbody text\n";
        let card = ModelCard::parse(readme);
        assert_eq!(card.license.as_deref(), Some("apache-2.0"));
        assert_eq!(card.base_models, ["Qwen/Qwen3-0.6B-Base"]);
        assert_eq!(card.front_matter["tags"][0], "a");
        assert_eq!(card.body, "# Title\n\nbody text\n");
    }

    #[test]
    fn a_readme_without_front_matter_is_all_body() {
        let card = ModelCard::parse("# Just text\n---\nnot front matter\n");
        assert!(card.front_matter.is_null());
        assert_eq!(card.license, None);
        assert_eq!(card.body, "# Just text\n---\nnot front matter\n");
    }

    #[test]
    fn front_matter_that_is_not_yaml_is_ignored() {
        let card = ModelCard::parse("---\n: : [unclosed\n---\ntext");
        assert!(card.front_matter.is_null());
        assert_eq!(card.body, "text");
    }
}
