# pmetal-mlx

MLX backend integration with advanced training utilities.

## Overview

This crate provides the bridge between PMetal and Apple's MLX framework, along with custom implementations for training utilities not available in the base MLX library.

## Features

- **Quantization**: NF4, FP4, Int8 implementations
- **Gradient Checkpointing**: Memory-efficient training for large models
- **KV Cache**: Efficient key-value caching for inference
- **Mixture of Experts**: MoE layer implementations
- **NEFTune**: Noise injection for improved fine-tuning
- **Sequence Packing**: Efficient batching for variable-length sequences
- **Speculative Decoding**: Faster inference with draft models

## Usage

```rust
use pmetal_mlx::prelude::*;

// Create a KV cache for inference
let cache = KVCache::new(num_layers, batch_size, max_seq_len, head_dim);

// Use sequence packing for training
let packed = SequencePacker::pack(&sequences, max_length)?;
```

## Modules

| Module | Description |
|--------|-------------|
| `kernels` | Custom MLX kernels (cross entropy, RMS norm, etc.) |
| `quantization` | Weight quantization implementations |
| `gradient_checkpoint` | Memory-efficient gradient computation |
| `kv_cache` | Key-value cache for efficient inference |
| `moe` | Mixture of Experts support |
| `neftune` | NEFTune noise injection |
| `sequence_packing` | Efficient sequence batching |
| `speculative` | Speculative decoding utilities |

## Quantization Formats

| Format | Bits | Memory Savings | Quality |
|--------|------|----------------|---------|
| NF4 | 4 | 75% | High |
| FP4 | 4 | 75% | Medium |
| Int8 | 8 | 50% | Very High |

## License

MIT OR Apache-2.0
