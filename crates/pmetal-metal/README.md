# pmetal-metal

High-performance Metal GPU kernels for Apple Silicon.

## Overview

This crate provides custom Metal shaders that accelerate LLM training and inference on Apple Silicon. These kernels are the foundation of PMetal's performance advantages over Python-based frameworks.

## Features

- **FlashAttention**: O(n) memory attention with fused softmax (forward + backward)
- **Fused LoRA**: Combined base + adapter forward pass (~2x speedup)
- **Fused Cross-Entropy**: Chunked vocabulary loss (avoids logits materialization)
- **Fused RoPE**: Rotary position embeddings computed in-kernel
- **Fused Sampler**: JIT-compiled token sampling
- **Fused SwiGLU**: MLP activation fusion
- **Fused Norm+LoRA**: Combined layer norm and adapter application
- **Fused GDN**: Gated Delta Network recurrence (Metal shader, 32-thread SIMD)
- **MoE Routing**: Expert selection and dispatch kernel

## Architecture

```text
pmetal-metal/
├── src/
│   ├── context.rs        # Thread-safe Metal device management
│   ├── buffer.rs         # Type-safe GPU buffer abstraction
│   ├── bridge.rs         # MLX array ↔ Metal buffer conversion
│   ├── pipeline.rs       # Compute pipeline management
│   ├── accelerate.rs     # vDSP/cblas wrappers (RMSNorm, Adam, GEMM, etc.)
│   ├── kernels/
│   │   ├── metal/        # .metal shader source files
│   │   └── *.rs          # Rust wrappers for each kernel
│   └── ane/              # Apple Neural Engine (feature-gated: `ane`)
│       ├── mod.rs            # Module root and architecture diagram
│       ├── runtime.rs        # Private API FFI (dlopen + objc2)
│       ├── iosurface.rs      # IOSurface zero-copy (fp16 + fp32)
│       ├── mil.rs            # MIL 1.3 program builder
│       ├── kernel.rs         # Weight blobs, int8 quantization, RoPE helpers
│       ├── dynamic_kernel.rs # Dynamic weight kernel generators (training)
│       ├── dynamic_trainer.rs# Compile-once training loop
│       ├── extend.rs         # Multi-layer prefill/decode/verify kernels over an IOSurface KV cache
│       ├── lm.rs             # ANE text generation from a Hugging Face checkpoint
│       └── inference_hybrid.rs # CPU decode engine for Qwen3.5 hybrid models
```

### ANE Dynamic Weight Pipeline

The ANE module provides a complete training and inference pipeline using Apple's private `AppleNeuralEngine.framework` APIs. The dynamic weight pipeline compiles its training kernels once at startup and packs weights alongside activations in the IOSurface spatial dimension, so training never recompiles.

**Inference** (`ane::lm::AneLm`) runs the model's layers as a few multi-layer extend programs over a KV cache held in IOSurfaces, with int8 weights by default. The same programs handle prefill, decode and speculative verification, and they compile once: Apple's ANE service caches them for later runs.

| Kernel | Purpose |
|--------|---------|
| `projection` | One linear layer, compiled once per weight shape: the FFN down projection, the QKV and output projection backward, and the decomposed forward pass when a fused kernel doesn't fit in ANE memory |
| `sdpa_fwd` | Self-attention forward (QKV projection + SDPA + output projection) |
| `ffn_w13` | FFN forward (W1 gate + W3 up + SiLU) |
| `ffn_bwd_w2t` | FFN backward through W2 |
| `ffn_bwd_w13t` | FFN backward through W1/W3 |
| `sdpa_bwd1` | Attention backward part 1 (dV, attention probs), fused with the output projection |
| `sdpa_bwd2` | Attention backward part 2 (dQ, dK) |

## Usage

`MetalContext::global()` returns the process-wide shared context; prefer it over `new()` so device,
queue and pipeline cache are shared.

```rust,no_run
use half::f16;
use pmetal_metal::{
    FlashAttention, FlashAttentionConfig, MetalBuffer, MetalContext, Result,
};

fn attend(
    queries: &MetalBuffer<f16>,
    keys: &MetalBuffer<f16>,
    values: &MetalBuffer<f16>,
) -> Result<()> {
    let ctx = MetalContext::global()?;

    let config = FlashAttentionConfig {
        batch_size: 1,
        num_heads: 32,
        num_kv_heads: 8, // GQA; set equal to num_heads for standard MHA
        query_seq_len: 512,
        kv_seq_len: 512,
        head_dim: 128,
        is_causal: true,
        ..Default::default()
    };

    let attention = FlashAttention::new(ctx, config)?;
    let _output = attention.forward(queries, keys, values)?;
    Ok(())
}
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
- ANE features require `--features ane` at build time

## License

MIT OR Apache-2.0
