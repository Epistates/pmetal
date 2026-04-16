use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use indicatif::{ProgressBar, ProgressStyle};

/// Tokenize a JSONL corpus into binary shards for pretraining.
///
/// Each output shard is a flat binary file in the format expected by
/// `pmetal_data::streaming`: `[u32 doc_len][u32 token_0 .. u32 token_N-1]`
/// repeated for every document, little-endian byte order.
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

    // --- Output directory ----------------------------------------------------
    let out_dir = Path::new(output_dir);
    std::fs::create_dir_all(out_dir)?;

    // --- Count lines for progress bar ----------------------------------------
    let total_lines = {
        let f = File::open(input)
            .map_err(|e| anyhow::anyhow!("Cannot open input file '{}': {e}", input))?;
        BufReader::new(f).lines().count()
    };

    let bar = ProgressBar::new(total_lines as u64);
    bar.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} docs ({per_sec})",
        )
        .unwrap_or_else(|_| ProgressStyle::default_bar()),
    );

    // --- Main tokenization loop ----------------------------------------------
    let reader = {
        let f = File::open(input)
            .map_err(|e| anyhow::anyhow!("Cannot open input file '{}': {e}", input))?;
        BufReader::new(f)
    };

    let mut shard_index: usize = 0;
    let mut shard_docs: Vec<Vec<u32>> = Vec::with_capacity(docs_per_shard);
    let mut total_docs: u64 = 0;
    let mut total_tokens: u64 = 0;
    let mut skipped: u64 = 0;

    for (line_num, line_result) in reader.lines().enumerate() {
        let line =
            line_result.map_err(|e| anyhow::anyhow!("Read error at line {}: {e}", line_num + 1))?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Parse JSON object and extract text field.
        let value: serde_json::Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!("Skipping line {}: JSON parse error: {e}", line_num + 1);
                skipped += 1;
                bar.inc(1);
                continue;
            }
        };

        let text = match value.get(text_column).and_then(|v| v.as_str()) {
            Some(t) if !t.is_empty() => t.to_owned(),
            _ => {
                tracing::warn!(
                    "Skipping line {}: missing or empty column '{text_column}'",
                    line_num + 1
                );
                skipped += 1;
                bar.inc(1);
                continue;
            }
        };

        // Tokenize — encode without special tokens; pretraining loops handle
        // BOS/EOS insertion at the sequence packing stage.
        let encoding = tokenizer
            .encode(text.as_str(), false)
            .map_err(|e| anyhow::anyhow!("Tokenization error at line {}: {e}", line_num + 1))?;

        let ids: Vec<u32> = encoding.get_ids().to_vec();
        if ids.is_empty() {
            skipped += 1;
            bar.inc(1);
            continue;
        }

        total_tokens += ids.len() as u64;
        shard_docs.push(ids);
        total_docs += 1;
        bar.inc(1);

        // Flush shard when full.
        if shard_docs.len() >= docs_per_shard {
            flush_shard(out_dir, shard_index, &shard_docs)?;
            shard_index += 1;
            shard_docs.clear();
        }
    }

    // Flush any remaining documents.
    if !shard_docs.is_empty() {
        flush_shard(out_dir, shard_index, &shard_docs)?;
        shard_index += 1;
    }

    bar.finish_and_clear();

    let shards_written = shard_index;
    println!("Done. {total_docs} documents, {shards_written} shards, {total_tokens} total tokens.");
    if skipped > 0 {
        println!("  ({skipped} lines skipped — missing column or parse errors)");
    }

    Ok(())
}

/// Flush a batch of documents to a numbered shard file via `pmetal_data::streaming::write_shard`.
fn flush_shard(out_dir: &Path, index: usize, docs: &[Vec<u32>]) -> anyhow::Result<()> {
    let path = out_dir.join(format!("shard_{index:05}.bin"));
    let refs: Vec<&[u32]> = docs.iter().map(Vec::as_slice).collect();
    pmetal_data::streaming::write_shard(&path, &refs)
        .map_err(|e| anyhow::anyhow!("Failed to write shard {}: {e}", path.display()))
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
