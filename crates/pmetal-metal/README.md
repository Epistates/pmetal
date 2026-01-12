# pmetal-metal

High-performance Metal GPU kernels for Apple Silicon.

## Overview

This crate provides custom Metal shaders that accelerate LLM training and inference on Apple Silicon. These kernels are the foundation of PMetal's performance advantages over Python-based frameworks.

## Features

- **FlashAttention**: O(n) memory attention with fused softmax (forward + backward)
- **Fused LoRA**: Combined base + adapter forward pass (~2x speedup)
- **Fused Cross-Entropy**: Chunked loss computation (Unsloth optimization)
- **Fused RoPE**: Rotary position embeddings computed in-kernel
- **Fused Sampler**: JIT-compiled token sampling
- **Fused SwiGLU**: MLP activation fusion
- **Fused Norm+LoRA**: Combined layer norm and adapter application

## Architecture

```
pmetal-metal/
├── src/
│   ├── context.rs      # Thread-safe Metal device management
│   ├── buffer.rs       # Type-safe GPU buffer abstraction
│   ├── bridge.rs       # MLX array ↔ Metal buffer conversion
│   ├── pipeline.rs     # Compute pipeline management
│   └── kernels/
│       ├── metal/      # .metal shader source files
│       └── *.rs        # Rust wrappers for each kernel
```

## Usage

```rust
use pmetal_metal::{MetalContext, FlashAttention};

// Initialize Metal context
let ctx = MetalContext::new()?;

// Use FlashAttention for memory-efficient attention
let attention = FlashAttention::new(&ctx, head_dim, num_heads)?;
let output = attention.forward(&query, &key, &value, mask)?;
```

## Kernels

| Kernel | Speedup | Memory | Description |
|--------|---------|--------|-------------|
| `flash_attention` | 1.5-2x | O(n) vs O(n²) | Memory-efficient attention |
| `fused_lora` | ~2x | Same | Combined base+adapter forward |
| `fused_cross_entropy` | 1.3x | O(1) per chunk | Chunked loss computation |
| `fused_rope` | 1.2x | Same | In-kernel position encoding |
| `fused_sampler` | 1.4x | Same | JIT-compiled sampling |

## Requirements

- macOS 13+ (Ventura or later)
- Apple Silicon (M1/M2/M3/M4)
- Metal Toolchain (via Xcode or `xcodebuild -downloadComponent MetalToolchain`)

## License

MIT OR Apache-2.0
