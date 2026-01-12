# pmetal-hub

HuggingFace Hub integration for model management.

## Overview

This crate provides seamless integration with the HuggingFace Hub, enabling model downloading, caching, and management.

## Features

- **Model Downloading**: Download models from HuggingFace Hub
- **Local Caching**: Efficient cache management
- **Token Authentication**: Secure access to private models
- **Progress Tracking**: Download progress with ETA

## Usage

### Download a Model

```rust
use pmetal_hub::Hub;

let hub = Hub::new()?;

// Download model to cache
let model_path = hub.download("unsloth/Llama-3.2-1B")?;

// Use cached path
println!("Model at: {}", model_path.display());
```

### With Authentication

```rust
use pmetal_hub::Hub;

// Use HF_TOKEN environment variable or provide explicitly
let hub = Hub::with_token(std::env::var("HF_TOKEN")?)?;

// Access private/gated models
let model_path = hub.download("meta-llama/Llama-3.2-1B")?;
```

### Cache Management

```rust
use pmetal_hub::Cache;

let cache = Cache::default();

// Check if model is cached
if cache.contains("unsloth/Llama-3.2-1B")? {
    let path = cache.get("unsloth/Llama-3.2-1B")?;
}

// Clear cache
cache.clear()?;
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
| `upload` | Model uploading (planned) |

## License

MIT OR Apache-2.0
