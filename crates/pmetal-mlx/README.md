# pmetal-mlx

Model-side operations, KV caches and decoding state over the PMetal runtime.

## Overview

`pmetal-runtime` owns the arrays, the ops and the fused kernels. This crate is
what the model forwards build on top of them: attention with masks, sinks and
logit softcapping, the gated delta rule, the cut cross-entropy loss, rotary
embeddings at explicit positions, and the caches a decoder carries.

## Features

- **Attention**: scaled dot-product attention (MLX's fused kernel, also under autograd), attention sinks on the runtime's fused op, Gemma 2 logit softcapping with f32 scores
- **Gated delta rule**: sequential, chunkwise parallel (differentiable, so training uses it) and the fused inference kernel, one dispatcher
- **Cut cross-entropy**: the LM-head loss without the logits, with label smoothing and a z-loss
- **KV cache**: dense, rotating, quantized, TurboQuant and recurrent-state caches, and prefix reuse
- **Mixture of Experts**: the expert MLP and the combine step
- **Speculative decoding state**: The hidden states a draft model reads and the recurrent-layer inputs a rollback replays

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
| `kernels::fused_attention` | `fused_sdpa`, `sink_sdpa`, the mask builders |
| `kernels::gated_delta` | The gated delta rule and its dispatcher |
| `kernels::cut_cross_entropy` | `CutCrossEntropy` over an LM head |
| `kernels::rope` | `RopePositions` and rotation at contiguous or explicit positions |
| `kernels::fused_moe` | The MoE combine |
| `kernels::training_context` | `with_training_mode`, `is_training` |
| `kv_cache`, `prefix_cache` | Key-value caches for inference |
| `moe` | The expert MLP |
| `speculative` | `SpecCapture`, the state a target's forward pass records for speculative decoding |
| `memory` | Allocator and system memory figures |
| `bridge` | Array to Metal buffer |

## License

MIT OR Apache-2.0
