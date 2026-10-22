# pmetal-hub

HuggingFace Hub integration for PMetal: download, verify, search, size for this Mac, and publish.

## Overview

`pmetal-models`, `pmetal-trainer`, `pmetal-merge`, `pmetal-vocoder`, `pmetal-mcp`, the `pmetal` CLI and the Python bindings resolve model references through it, so a repository id works wherever a path does. It depends on `pmetal-core` and `pmetal-gguf`, and needs no GPU or runtime.

- **Downloads from one commit.** A revision is resolved to a commit, that commit's file list is read
  once, and exactly the files the model needs are fetched concurrently into the standard Hub cache
  (`models--owner--name/{blobs,refs,snapshots}`), shared with the Hub's own tools.
- **Only what the model needs.** Configs and tokenizer files plus one set of weights: safetensors
  (exactly the shards the index names), or one GGUF quantization (split models come whole), or, as
  a last resort, PyTorch pickles. Pickles are read with a restricted unpickler that cannot run
  code (`torch`) and written as safetensors next to them, so what comes back always loads.
- **Verified.** Every file is checked against the size the Hub lists (optionally its SHA-256), and
  the result must pass `verify_model_dir`, which reads headers only: safetensors files exactly as
  long as their headers say, GGUF sets complete.
- **Offline and flaky networks.** `HF_HUB_OFFLINE` (or `HubOptions::offline`) never touches the
  network; a named branch falls back to the copy last cached for it when the Hub is unreachable.
  Requests have connect and stall timeouts and a bounded retry count.
- **Search that knows your Mac.** Results carry the real weight size (from tensor dtypes) and can be
  sized against the device's recommended working set: weights, KV cache at a stated context, and a
  decode speed estimate from memory bandwidth.
- **Publishing.** `upload_model` creates the repository if needed and commits the directory
  atomically, with progress; credential files, `.git` and Finder droppings are never sent.

## Installation

```toml
[dependencies]
pmetal-hub = "0.7"
```

## Usage

### Download a Model

The plain functions take a repository id, an optional revision and an optional token. With no
revision, a complete cached copy is returned without contacting the Hub.

```rust,no_run
use pmetal_hub::download_model;

async fn fetch() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = download_model("meta-llama/Llama-3.2-1B", None, None).await?;
    println!("Model at: {}", model_path.display());
    Ok(())
}
```

### With Authentication

Pass a token to reach private or gated repos; without one, `HF_TOKEN`, `HF_TOKEN_PATH` and the token
stored by `hf auth login` are tried in that order. `SecretString` keeps it out of `Debug` output, and
error messages are scrubbed of anything that looks like a token.

```rust,no_run
use pmetal_core::SecretString;
use pmetal_hub::download_model;

async fn fetch_gated() -> Result<(), Box<dyn std::error::Error>> {
    let token = SecretString::new(std::env::var("HF_TOKEN")?);
    let model_path = download_model("meta-llama/Llama-3.2-1B", None, Some(&token)).await?;
    println!("Model at: {}", model_path.display());
    Ok(())
}
```

### Options, Quantizations and Progress

`DownloadOptions` chooses the weight format, a GGUF quantization, verification, and a progress
callback with rate and ETA. `org/repo:Q4_K_M` selects a quantization wherever a model reference is
accepted (`resolve_model_path`).

```rust,no_run
use std::sync::Arc;
use pmetal_hub::{DownloadOptions, WeightFormat, download_model_with, plan_model_download};

async fn pull_one_quant() -> Result<(), Box<dyn std::error::Error>> {
    let options = DownloadOptions {
        weights: WeightFormat::Gguf,
        gguf_quant: Some("Q4_K_M".into()),
        progress: Some(Arc::new(|p| {
            if let (Some(rate), Some(eta)) = (p.bytes_per_sec, p.eta) {
                eprintln!("{}/{} bytes, {rate:.0} B/s, {eta:?} left", p.bytes_done, p.total_bytes);
            }
        })),
        ..Default::default()
    };
    // What would be fetched, and how big is it?
    let plan = plan_model_download("Qwen/Qwen3-0.6B-GGUF", &options).await?;
    println!("{} files, {} bytes", plan.model.files.len(), plan.model.total_bytes());
    let dir = download_model_with("Qwen/Qwen3-0.6B-GGUF", &options).await?;
    println!("{}", dir.display());
    Ok(())
}
```

### Will It Fit?

Search results can be sized for a device. Ask the device for its recommended working set; do not
derive it from installed RAM.

```rust,no_run
use pmetal_hub::search::{SearchOptions, search_models_with};
use pmetal_hub::{DeviceSpec, FitLevel};

async fn search() -> Result<(), Box<dyn std::error::Error>> {
    // 48 GB working set, 546 GB/s memory bandwidth (from the Metal device).
    let device = DeviceSpec::from_device(48_000_000_000, 546.0, true);
    let mut options = SearchOptions::new("qwen3");
    options.tags = vec!["mlx".into()];
    for result in search_models_with(&options).await? {
        if let Some(fit) = result.fit(&device, 4096) {
            if fit.fit_level != FitLevel::TooLarge {
                println!("{}: {:.1} GB, ~{:.0} tok/s (estimate)", result.model_id, fit.total_required_gb, fit.estimated_tps);
            }
        }
    }
    Ok(())
}
```

The same estimate works without the network, from a parameter count and a quantization:

```rust
use pmetal_hub::{DeviceSpec, FitLevel, ModelSpec, estimate_fit, split_quant_selector};

// An 8B model at 4 bits on a Mac with a 48 GB working set and 546 GB/s.
let device = DeviceSpec::from_device(48_000_000_000, 546.0, true);
let fit = estimate_fit(&ModelSpec::new(8.0, "q4_k_m").with_context(8192), &device);
assert_ne!(fit.fit_level, FitLevel::TooLarge);

// `org/repo:QUANT` names one quantization of a GGUF repository.
assert_eq!(
    split_quant_selector("Qwen/Qwen3-8B-GGUF:Q4_K_M"),
    ("Qwen/Qwen3-8B-GGUF", Some("Q4_K_M")),
);
```

### Check a Local Model Directory

```rust
use pmetal_hub::verify_model_dir;

// A directory that is not there is a report with a problem, not a panic.
let report = verify_model_dir(std::path::Path::new("./no-such-model"));
assert!(!report.is_complete());
for problem in &report.problems {
    eprintln!("{problem}");
}
```

### Publish a Model

`upload_model` creates the repository if it is missing and writes the directory in one commit.
Large files go to the Hub's large-file storage; `.git`, `.DS_Store`, AppleDouble `._*` files and
`.env` files are never sent. With no token it uses `HF_TOKEN` or the token stored by `hf auth login`.

```rust,no_run
use pmetal_hub::{UploadOptions, upload_model};

async fn publish() -> Result<(), Box<dyn std::error::Error>> {
    let options = UploadOptions {
        private: true,
        ignore: vec!["checkpoints/**".into()],
        ..Default::default()
    };
    let report = upload_model("./output/fused", "me/my-model", None, &options).await?;
    println!("{}", report.repo_url);
    Ok(())
}
```

### Cache Management

```rust,no_run
use pmetal_hub::{cache_dir, cache_size, clear_cache, evict_model, find_cached_model};

fn manage() -> Result<(), Box<dyn std::error::Error>> {
    println!("cache at {}", cache_dir().display());
    println!("{} bytes on disk", cache_size()?);

    if let Some(path) = find_cached_model("meta-llama/Llama-3.2-1B") {
        println!("already downloaded to {}", path.display());
    }

    evict_model("meta-llama/Llama-3.2-1B")?; // one model
    clear_cache()?; // everything the cache holds, nothing else in that directory
    Ok(())
}
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | Access token |
| `HF_TOKEN_PATH` | File to read the token from |
| `HF_HUB_DISABLE_IMPLICIT_TOKEN` | Ignore every token source but an explicit one |
| `HF_ENDPOINT` | Hub base URL (a mirror, a private deployment) |
| `HF_HOME` | Hugging Face home (default: `~/.cache/huggingface`) |
| `HF_HUB_CACHE` | Hub cache directory (default: `$HF_HOME/hub`) |
| `HF_HUB_OFFLINE` | `1` to use only the local cache |

## Modules

| Module | Description |
|--------|-------------|
| `download` | Model downloads, plans, verification options |
| `dataset` | Dataset and Parquet-split downloads |
| `select` | Which files a model needs (weight formats, GGUF quantizations, shards) |
| `verify` | Checking a model directory is complete |
| `search` | Model search, metadata, model cards, GGUF variants |
| `fit` | Memory and decode-speed estimates for a device |
| `tensors` | Safetensors header summaries |
| `torch` | Memory-mapped, restricted reading of PyTorch checkpoints; conversion to safetensors |
| `cache` | Local cache management |
| `upload` | Publishing a model directory |
| `resolve` | `resolve_model_path`, `is_hf_id` and the `org/repo:QUANT` selector: a repository id or a local path to a directory on disk |

## Feature flags

| Feature | Default | Description |
|---------|---------|-------------|
| `testing` | no | Builders for PyTorch checkpoints, for this crate's own tests. Not part of the supported API |

## Requirements

- Rust 1.95 or later
- Network access to the Hub for downloads, search and uploads (or `HF_HUB_OFFLINE=1` and a populated cache)

## License

MIT OR Apache-2.0
