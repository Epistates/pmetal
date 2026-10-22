# pmetal-metal

Metal GPU kernels and direct Apple Neural Engine programming for Apple Silicon.

## Overview

This crate provides custom Metal shaders for the operations the tensor runtime has no op for. Dense attention, RoPE, SwiGLU, LoRA and GEMM are deliberately not here: the runtime runs them in its lazy graph, with autodiff, and a hand-written kernel for any of them measured slower than the runtime op.

It sits under the training and inference crates: `pmetal-trainer`, `pmetal-models`, `pmetal-serve` and `pmetal-mcp` depend on it, and the `pmetal` CLI and `pmetal-py` reach it through them. It needs only `pmetal-core` from the workspace. The whole crate is `cfg(target_os = "macos")`.

## Features

- **Fused AdamW**: one-dispatch optimizer update over every parameter
- **dW GEMM**: the f32 weight-gradient GEMM the ANE trainer batches into one command buffer
- **Apple Neural Engine**: MIL program generation, IOSurface transfer, a compile-once training loop and the `AneLm` inference engine (prefill, decode and speculative verification on one set of kernels)
- **CPU helpers**: vDSP and cblas wrappers (`accelerate`) and NEON f16/f32 bulk conversion (`neon_convert`)
- **Device management**: a thread-safe `MetalContext` with a pipeline cache, typed `MetalBuffer`s over unified memory, and zero-copy views of runtime arrays

## Architecture

```text
pmetal-metal/
├── src/
│   ├── context.rs        # Thread-safe Metal device management
│   ├── buffer.rs         # Type-safe GPU buffer abstraction
│   ├── view.rs           # Zero-copy Metal views of runtime arrays
│   ├── pipeline.rs       # Compute pipeline management
│   ├── accelerate.rs     # vDSP/cblas wrappers (RMSNorm, Adam, GEMM, etc.)
│   ├── neon_convert.rs   # NEON f16 <-> f32 bulk conversion
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
│       ├── loss.rs           # Composable loss functions for the trainer
│       ├── scratch.rs        # Pre-allocated scratch pool for the backward pass
│       ├── extend.rs         # Multi-layer prefill/decode/verify kernels over an IOSurface KV cache
│       └── lm.rs             # ANE text generation from a Hugging Face checkpoint
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

## Installation

```toml
[dependencies]
pmetal-metal = "0.7"
```

## Usage

`MetalContext::global()` returns the process-wide shared context; prefer it over `new()` so device,
queue and pipeline cache are shared.

```rust,no_run
use pmetal_metal::kernels::{AdamWConfig, FusedAdamW};
use pmetal_metal::{BufferUsage, MetalBuffer, MetalContext, Result};

fn adamw_step() -> Result<()> {
    let ctx = MetalContext::global()?;

    // Two parameters, flattened into one buffer each for the weights, the
    // gradients and the two moments.
    let sizes = [4096, 1024];
    let total: usize = sizes.iter().sum();
    let buffer = || MetalBuffer::<f32>::zeros(&ctx, total, BufferUsage::Shared);
    let (params, grads, m, v) = (buffer()?, buffer()?, buffer()?, buffer()?);

    let adamw = FusedAdamW::new(ctx.clone(), &sizes)?;
    let info = MetalBuffer::from_slice(
        &ctx,
        &FusedAdamW::build_param_info(&sizes),
        BufferUsage::Shared,
    )?;
    // One set of scalars per parameter group; step 1 of a 1e-4 run.
    let groups = [AdamWConfig::default().scalars()?];
    adamw.execute_update(&params, &grads, &m, &v, &info, &groups)
}
```

## Kernels

| Kernel | Memory | Description |
|--------|--------|-------------|
| `fused_training` | Same | One-dispatch AdamW over every parameter |
| `dw_gemm` | Same | f32 weight-gradient GEMM for the ANE trainer |

## Requirements

- Rust 1.95 or later
- macOS 14+ (Sonoma or later), MLX's floor
- Apple Silicon (M1/M2/M3/M4/M5)
- Metal Toolchain (via Xcode or `xcodebuild -downloadComponent MetalToolchain`)

## Feature flags

| Feature | Default | Description |
|---------|---------|-------------|
| `ane` | yes | The `ane` module (also turns on `pmetal-core/ane`). It drives the Neural Engine through Apple's private `AppleNeuralEngine.framework`; build with `default-features = false` to leave it out |

## License

MIT OR Apache-2.0
