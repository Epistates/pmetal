# pmetal-cache

KV caches, prefix reuse and speculative-decoding state for PMetal's decoder
models.

## Overview

`pmetal-runtime` owns the arrays, the ops and the fused kernels. This crate
holds the state a decoder carries from one step to the next: the key-value
caches of attention layers, the recurrent state of linear-attention and Mamba
layers, the reuse of a shared prompt prefix across requests, and what a target
model records for speculative decoding.

`pmetal-models`, `pmetal-lora`, `pmetal-trainer` and `pmetal-serve` build their
decode paths on it.

## Features

- **KV cache**: standard, sliding-window, rotating (with attention sinks), affine-quantized, asymmetric-quantized, TurboQuant and recurrent-state caches, and a fused batched cache for continuous-batching decode
- **Prefix cache**: the KV state of an already-prefilled prompt prefix, reused by the next request that shares it
- **Speculative decoding state**: the hidden states a draft model reads and the recurrent-layer inputs a rollback replays

## Installation

```toml
[dependencies]
pmetal-cache = "0.7"
```

## Usage

```rust
use pmetal_cache::{KVCache, KVCacheConfig};
use pmetal_runtime::compat::{Dtype, ops};

// Two layers, room for 128 tokens, 4 KV heads of 64 channels.
let mut cache = KVCache::new(KVCacheConfig::new(2, 128, 4, 64));

// Keys and values are [batch, kv_heads, seq, head_dim]; a layer appends its
// new tokens and gets back everything cached so far.
let keys = ops::zeros(&[1, 4, 10, 64], Dtype::Float32);
let values = ops::zeros(&[1, 4, 10, 64], Dtype::Float32);
let (k, v) = cache.update_and_fetch(0, &keys, &values).unwrap();
assert_eq!((k.dim(2), v.dim(2)), (10, 10));
assert_eq!(cache.seq_len(), 10);

// Speculative decoding gives back the tokens the target rejected, and a
// fork shares the arrays with an independent copy (prefix reuse).
let fork = cache.fork();
cache.rollback(4);
assert_eq!(cache.seq_len(), 6);
assert_eq!(fork.seq_len(), 10);
```

## Modules

| Module | Description |
|--------|-------------|
| `kv_cache` | Key-value caches for inference |
| `prefix_cache` | Prompt-prefix reuse |
| `speculative` | `SpecCapture`, the state a target's forward pass records for speculative decoding |
| `test_utils` | Parity and numerical-comparison helpers for tests (behind the `test-utils` feature) |

## Feature flags

| Feature | Default | Description |
|---------|---------|-------------|
| `test-utils` | no | The `test_utils` module, for parity tests in downstream crates |

## Requirements

- Rust 1.95 or later
- Apple Silicon Mac: the caches hold `pmetal-runtime` arrays

## License

MIT OR Apache-2.0
