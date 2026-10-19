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
    let tokenizer = load_tokenizer(tokenizer_id, &pmetal_hub::DownloadOptions::default()).await?;

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

/// Resolve and load a tokenizer from a local path or a Hub model.
///
/// Resolution order:
/// 1. If `spec` is an existing file, load it directly.
/// 2. If `spec` is an existing directory, load the `tokenizer.json` inside it.
/// 3. Otherwise treat `spec` as a Hub model ID and get its `tokenizer.json`
///    from the Hub cache, downloading just that file when it is not cached
///    (never the weights). `HF_HOME`, `HF_HUB_CACHE`, `HF_ENDPOINT` and
///    `HF_HUB_OFFLINE` apply as for every other command.
async fn load_tokenizer(
    spec: &str,
    options: &pmetal_hub::DownloadOptions,
) -> anyhow::Result<tokenizers::Tokenizer> {
    let p = Path::new(spec);
    let file = if p.is_file() {
        p.to_path_buf()
    } else if p.is_dir() {
        let file = p.join("tokenizer.json");
        anyhow::ensure!(
            file.is_file(),
            "directory '{}' does not contain a tokenizer.json",
            p.display()
        );
        file
    } else if pmetal_hub::is_hf_id(spec) {
        let file = pmetal_hub::download_file_with(spec, "tokenizer.json", options)
            .await
            .map_err(|e| anyhow::anyhow!("could not get tokenizer.json of '{spec}': {e}"))?;
        tracing::info!("Resolved tokenizer from the Hub cache: {}", file.display());
        file
    } else {
        anyhow::bail!(
            "tokenizer '{spec}' is not a file, a directory or a Hub model ID (org/name); pass \
             the path of a tokenizer.json or of a model directory"
        );
    };
    tokenizers::Tokenizer::from_file(&file)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer from '{}': {e}", file.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOKENIZER_JSON: &str = r#"{"version":"1.0","added_tokens":[],"normalizer":null,
        "pre_tokenizer":{"type":"WhitespaceSplit"},"post_processor":null,"decoder":null,
        "model":{"type":"WordLevel","vocab":{"[UNK]":0,"hello":1},"unk_token":"[UNK]"}}"#;

    /// A Hub cache holding `org/model`'s tokenizer.json, in a directory of
    /// its own: where the cache is must come from the options, not from a
    /// guess at `~/.cache`.
    fn cache_with_tokenizer(root: &Path) -> pmetal_hub::DownloadOptions {
        let repo = root.join("models--org--model");
        let snapshot = repo
            .join("snapshots")
            .join("0123456789abcdef0123456789abcdef01234567");
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::create_dir_all(repo.join("refs")).unwrap();
        std::fs::write(
            repo.join("refs").join("main"),
            "0123456789abcdef0123456789abcdef01234567",
        )
        .unwrap();
        std::fs::write(snapshot.join("tokenizer.json"), TOKENIZER_JSON).unwrap();
        pmetal_hub::DownloadOptions {
            hub: pmetal_hub::HubOptions {
                cache_dir: Some(root.to_path_buf()),
                offline: Some(true),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn a_hub_id_is_found_in_the_configured_cache_not_a_hardcoded_one() {
        let dir = tempfile::tempdir().unwrap();
        let options = cache_with_tokenizer(dir.path());
        let tokenizer = load_tokenizer("org/model", &options).await.unwrap();
        assert_eq!(tokenizer.token_to_id("hello"), Some(1));
    }

    #[tokio::test]
    async fn an_id_that_is_nowhere_says_what_to_do() {
        let dir = tempfile::tempdir().unwrap();
        let options = cache_with_tokenizer(dir.path());
        let err = load_tokenizer("org/other", &options)
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("org/other"), "{err}");
        let err = load_tokenizer("not-a-model", &options)
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("tokenizer.json") && err.contains("not-a-model"),
            "{err}"
        );
    }

    #[tokio::test]
    async fn a_file_and_a_directory_load_directly() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("tokenizer.json"), TOKENIZER_JSON).unwrap();
        let options = pmetal_hub::DownloadOptions::default();
        for spec in [
            dir.path().to_str().unwrap(),
            dir.path().join("tokenizer.json").to_str().unwrap(),
        ] {
            assert!(load_tokenizer(spec, &options).await.is_ok(), "{spec}");
        }
        let empty = tempfile::tempdir().unwrap();
        let err = load_tokenizer(empty.path().to_str().unwrap(), &options)
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("does not contain a tokenizer.json"), "{err}");
    }
}
