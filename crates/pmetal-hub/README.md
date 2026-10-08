# pmetal-hub

HuggingFace Hub integration for model management.

## Overview

This crate provides integration with the HuggingFace Hub: downloading, caching, and publishing models.

## Features

- **Model Downloading**: Download models from HuggingFace Hub
- **Publishing**: Upload a trained or fused model directory, creating the repository if needed
- **Local Caching**: Efficient cache management
- **Token Authentication**: Secure access to private models
- **Progress Tracking**: Download progress with ETA

## Usage

### Download a Model

The API is free functions, not a client object. `download_model` checks the cache first, so calling
it again on a downloaded model does no network I/O.

```rust,no_run
use pmetal_hub::download_model;

async fn fetch() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = download_model("meta-llama/Llama-3.2-1B", None, None).await?;
    println!("Model at: {}", model_path.display());
    Ok(())
}
```

### With Authentication

Pass a token to reach private or gated repos. `SecretString` keeps it out of `Debug` output.

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

### Publish a Model

`upload_model` creates the repository if it is missing and writes the directory in one commit.
Large files go to the Hub's large-file storage; `.git`, `.DS_Store` and AppleDouble `._*` files are
never sent. With no token it uses `HF_TOKEN` or the token stored by `hf auth login`.

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
    clear_cache()?; // everything
    Ok(())
}
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | HuggingFace API token |
| `HF_HOME` | Cache directory (default: `~/.cache/huggingface`) |

## Modules

| Module | Description |
|--------|-------------|
| `download` | Model downloading |
| `cache` | Local cache management |
| `upload` | Publishing a model directory |

## License

MIT OR Apache-2.0
