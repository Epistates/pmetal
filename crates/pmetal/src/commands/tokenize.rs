use std::path::Path;

use indicatif::{ProgressBar, ProgressStyle};

/// Tokenize a corpus into binary shards for pretraining.
///
/// The corpus is anything `pmetal_data::rows` reads (JSONL, JSON, Parquet,
/// Arrow, CSV, or a directory of shards). Each output shard is a flat binary
/// file in the format expected by `pmetal_data::streaming`:
/// `[u32 doc_len][u32 token_0 .. u32 token_N-1]` repeated for every document,
/// little-endian byte order.
pub(crate) async fn run_tokenize(
    input: &str,
    output_dir: &str,
    tokenizer_id: &str,
    text_column: &str,
    docs_per_shard: usize,
) -> anyhow::Result<()> {
    // --- Tokenizer loading ---------------------------------------------------
    // The Rust `tokenizers` crate (0.22.x) only supports local files — it does
    // not include the HuggingFace Hub HTTP client that the Python bindings bundle.
    // Accepted forms for --tokenizer:
    //   - /path/to/tokenizer.json          (direct file)
    //   - /path/to/model-dir               (directory containing tokenizer.json)
    //   - Qwen/Qwen3-0.6B                  (HF model id — must already be downloaded
    //                                        via `pmetal download`; we resolve the
    //                                        local cache path automatically)
    let tokenizer = load_tokenizer(tokenizer_id)?;

    let bar = ProgressBar::new_spinner();
    bar.set_style(
        ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] {pos} rows ({per_sec})")
            .unwrap_or_else(|_| ProgressStyle::default_spinner()),
    );

    let stats = pmetal_data::streaming::tokenize_corpus(
        input,
        Path::new(output_dir),
        &tokenizer,
        text_column,
        docs_per_shard,
        |rows| bar.set_position(rows),
    );
    bar.finish_and_clear();
    let stats = stats?;

    println!(
        "Done. {} documents, {} shards, {} total tokens.",
        stats.documents, stats.shards, stats.tokens
    );
    if stats.skipped > 0 {
        println!(
            "  ({} rows skipped — missing or empty column '{text_column}', or unreadable)",
            stats.skipped
        );
    }

    Ok(())
}

/// Resolve and load a tokenizer from a local path or a cached HuggingFace model
/// directory.
///
/// Resolution order:
/// 1. If `spec` is an existing file — load it directly.
/// 2. If `spec` is an existing directory — look for `tokenizer.json` inside it.
/// 3. Otherwise treat `spec` as a HuggingFace model ID and search the local
///    HuggingFace cache (`~/.cache/huggingface/hub/models--*/snapshots/*/tokenizer.json`).
///    If not found, emit a helpful error directing the user to `pmetal download`.
fn load_tokenizer(spec: &str) -> anyhow::Result<tokenizers::Tokenizer> {
    let p = Path::new(spec);

    // Case 1: direct file path.
    if p.is_file() {
        return tokenizers::Tokenizer::from_file(p)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer from '{}': {e}", p.display()));
    }

    // Case 2: directory containing tokenizer.json.
    if p.is_dir() {
        let tok_file = p.join("tokenizer.json");
        if tok_file.exists() {
            return tokenizers::Tokenizer::from_file(&tok_file).map_err(|e| {
                anyhow::anyhow!(
                    "Failed to load tokenizer from '{}': {e}",
                    tok_file.display()
                )
            });
        }
        anyhow::bail!(
            "Directory '{}' does not contain a tokenizer.json file.",
            p.display()
        );
    }

    // Case 3: HuggingFace model ID — search local cache.
    // The cache layout is:
    //   ~/.cache/huggingface/hub/models--{org}--{repo}/snapshots/{hash}/tokenizer.json
    let hf_cache = dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".cache")
        .join("huggingface")
        .join("hub");

    // Convert "Qwen/Qwen3-0.6B" -> "models--Qwen--Qwen3-0.6B"
    let model_dir_name = format!("models--{}", spec.replace('/', "--"));
    let model_cache = hf_cache.join(&model_dir_name);

    if model_cache.is_dir() {
        // Walk snapshots to find the most recently modified tokenizer.json.
        let snapshots_dir = model_cache.join("snapshots");
        if snapshots_dir.is_dir() {
            let mut candidates: Vec<std::path::PathBuf> = std::fs::read_dir(&snapshots_dir)
                .map(|rd| {
                    rd.flatten()
                        .map(|e| e.path().join("tokenizer.json"))
                        .filter(|p| p.exists())
                        .collect()
                })
                .unwrap_or_default();

            // Sort by modification time (newest first) for determinism.
            candidates.sort_by_key(|p| {
                p.metadata()
                    .and_then(|m| m.modified())
                    .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
            });
            candidates.reverse();

            if let Some(tok_file) = candidates.into_iter().next() {
                tracing::info!("Resolved tokenizer from cache: {}", tok_file.display());
                return tokenizers::Tokenizer::from_file(&tok_file).map_err(|e| {
                    anyhow::anyhow!(
                        "Failed to load tokenizer from '{}': {e}",
                        tok_file.display()
                    )
                });
            }
        }
    }

    anyhow::bail!(
        "Tokenizer '{}' not found locally.\n\
         Run `pmetal download {}` to fetch it first, then retry.\n\
         Alternatively pass a direct path to a tokenizer.json file or model directory.",
        spec,
        spec
    )
}
