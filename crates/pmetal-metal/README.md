# pmetal-metal

High-performance Metal GPU kernels for Apple Silicon.

## Overview

This crate provides custom Metal shaders for the operations the tensor runtime has no op for. Dense attention, RoPE, SwiGLU, LoRA and GEMM are deliberately not here: the runtime runs them in its lazy graph, with autodiff, and a hand-written kernel for any of them measured slower than the runtime op.

## Features

- **Fused Cross-Entropy**: Chunked vocabulary loss (avoids logits materialization)
- **Fused AdamW** and **FP8 training** kernels
- **Quantized and MoE GEMMs**: dequantizing matmuls, grouped expert GEMMs, fused expert MLPs
- **dW GEMM**: the f32 weight-gradient GEMM the ANE trainer batches into one command buffer
- **Model merging** and **TurboQuant** KV-cache transforms

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
use pmetal_metal::{
    FusedCrossEntropy, FusedCrossEntropyConfig, MetalBuffer, MetalContext, Result,
};

fn loss(logits: &MetalBuffer<f32>, targets: &MetalBuffer<i32>) -> Result<()> {
    let ctx = MetalContext::global()?;

    // 512 tokens over a 32k vocabulary.
    let config = FusedCrossEntropyConfig::new(512, 32_768);
    let cross_entropy = FusedCrossEntropy::new(ctx, config)?;
    let _output = cross_entropy.forward(logits, targets)?;
    Ok(())
}
```

## Kernels

| Kernel | Memory | Description |
|--------|--------|-------------|
| `fused_cross_entropy` | O(1) per chunk | Chunked loss computation |
| `dw_gemm` | Same | f32 weight-gradient GEMM for the ANE trainer |

## Requirements

- macOS 14+ (Sonoma or later), MLX's floor
- Apple Silicon (M1/M2/M3/M4/M5)
- Metal Toolchain (via Xcode or `xcodebuild -downloadComponent MetalToolchain`)
- ANE features require `--features ane` at build time

## License

MIT OR Apache-2.0
