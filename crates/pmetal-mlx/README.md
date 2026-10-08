# pmetal-mlx

MLX backend integration with advanced training utilities.

## Overview

This crate provides the bridge between PMetal and Apple's MLX framework, along with custom implementations for training utilities not available in the base MLX library.

## Features

- **Gradient Checkpointing**: Memory-efficient training for large models
- **KV Cache**: Efficient key-value caching for inference
- **Mixture of Experts**: MoE layer implementations
- **Speculative Decoding**: Faster inference with draft models

## Usage

```rust,no_run
use pmetal_mlx::prelude::*;

fn setup() {
    // KV cache for inference: (layers, max_seq_len, kv_heads, head_dim)
    let _cache = KVCache::new(KVCacheConfig::new(28, 4096, 8, 128));
}
```

## Modules

| Module | Description |
|--------|-------------|
| `kernels` | Custom MLX kernels (fused attention, cut cross entropy, RMS norm, GDN, etc.) |
| `kernels/gated_delta` | Gated Delta Network (GDN) recurrence with fused Metal shader |
| `gradient_checkpoint` | Gradient-checkpointing config (the mechanism lives in `pmetal-bridge`) |
| `kv_cache` | Key-value cache for efficient inference |
| `moe` | Mixture of Experts support |
| `speculative` | Speculative decoding utilities |

## License

MIT OR Apache-2.0
