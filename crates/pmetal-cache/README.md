# pmetal-cache

KV caches, prefix reuse and speculative-decoding state for PMetal's decoder
models.

## Overview

`pmetal-runtime` owns the arrays, the ops and the fused kernels. This crate
holds the state a decoder carries from one step to the next: the key-value
caches of attention layers, the recurrent state of linear-attention and Mamba
layers, the reuse of a shared prompt prefix across requests, and what a target
model records for speculative decoding.

## Features

- **KV cache**: dense, rotating, quantized, TurboQuant and recurrent-state caches, and a fused batched cache for continuous-batching decode
- **Prefix cache**: the KV state of an already-prefilled prompt prefix, reused by the next request that shares it
- **Speculative decoding state**: the hidden states a draft model reads and the recurrent-layer inputs a rollback replays

## Usage

```rust,no_run
use pmetal_cache::{KVCache, KVCacheConfig};

fn setup() {
    // KV cache for inference: (layers, max_seq_len, kv_heads, head_dim)
    let _cache = KVCache::new(KVCacheConfig::new(28, 4096, 8, 128));
}
```

## Modules

| Module | Description |
|--------|-------------|
| `kv_cache` | Key-value caches for inference |
| `prefix_cache` | Prompt-prefix reuse |
| `speculative` | `SpecCapture`, the state a target's forward pass records for speculative decoding |
| `test_utils` | Parity and numerical-comparison helpers for tests (behind the `test-utils` feature) |

## License

MIT OR Apache-2.0
