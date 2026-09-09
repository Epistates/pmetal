# pmetal-mhc

Manifold-Constrained Hyper-Connections (mHC) for stable deep network training.

## Overview

This crate implements Manifold-Constrained Hyper-Connections from [DeepSeek-AI](https://arxiv.org/abs/2512.24880), a technique that replaces standard residual connections with learnable, doubly stochastic mixing matrices. By constraining residual mappings to the doubly stochastic manifold via Sinkhorn-Knopp projections, mHC keeps signal amplification bounded (Amax gain ~ 1) even in very deep networks (60+ layers).

## Architecture

```text
Input [batch, n, C]
    │
    ▼
┌──────────────────────────────────────────┐
│ compute_mappings()                       │
│  1. Flatten → RMSNorm                   │
│  2. Linear projection: α·(x·φ) + b     │
│  3. Constraints:                         │
│     H^pre  ← σ(·)           ∈ [0, 1]   │
│     H^post ← 2σ(·)          ∈ [0, 2]   │
│     H^res  ← Sinkhorn(·)    doubly stochastic │
└──────────────────────────────────────────┘
    │
    ▼
apply_pre_mapping(x, H^pre) → [batch, C]
    │
    ▼
┌──────────────────────────────────────────┐
│         Sublayer (Attention / FFN)        │
└──────────────────────────────────────────┘
    │
    ▼
apply_post_res_mapping(x, h_out, H^post, H^res)
    x_{l+1} = H^res @ x + H^post^T @ h_out
    │
    ▼
Output [batch, n, C]
```

## Features

- **Training Stability**: Doubly stochastic constraints guarantee bounded signal flow across arbitrary depth
- **Compositional Closure**: Products of doubly stochastic matrices remain doubly stochastic
- **Dynamic Mappings**: Input-dependent via learned projections, or static for reduced parameters
- **Metal GPU Acceleration**: Fused kernels for RMSNorm + projection, batched Sinkhorn, and residual merging
- **Mixed Precision**: BF16 activations with FP32 mappings
- **Complete Training Support**: Forward and backward passes with gradient accumulation

## Usage

```rust,no_run
use ndarray::{Array2, Array3};
use pmetal_mhc::{MhcConfig, MhcLayer, MhcPreset};

fn step(
    attention_fn: impl Fn(&Array2<f32>) -> Array2<f32>,
    grad_output: &Array3<f32>,
) {
    let config = MhcConfig::from_preset(MhcPreset::Medium);
    let layer = MhcLayer::new(config);

    // Forward
    let x: Array3<f32> = Array3::zeros((2, 4, 512));
    let (h_in, mappings) = layer.pre_layer(&x);
    let h_out = attention_fn(&h_in);
    let _output = layer.post_res_layer(&x, &h_out, &mappings);

    // Backward
    let (_grad_x, _grad_h_in) = layer.backward(&x, &h_out, &mappings, grad_output, None);

    layer.apply_gradients_sgd(1e-3);
}
```

### Transformer Block

```rust,no_run
use ndarray::{Array2, Array3};
use pmetal_mhc::{MhcConfig, MhcPreset, MhcTransformerBlock};

fn block(
    attn_fn: impl Fn(&Array2<f32>) -> Array2<f32>,
    ffn_fn: impl Fn(&Array2<f32>) -> Array2<f32>,
    x: &Array3<f32>,
) -> Array3<f32> {
    let config = MhcConfig::from_preset(MhcPreset::Medium);
    let block = MhcTransformerBlock::new(config, attn_fn, ffn_fn, 0);
    block.forward(x)
}
```

## Presets

| Preset | Hidden Dim | Target Scale |
|--------|-----------|-------------|
| **Small** | 1280 | ~3B parameters |
| **Medium** | 1920 | ~9B parameters |
| **Large** | 2560 | ~27B parameters |
| **Custom** | Configurable | Any |

## Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `expansion_rate` | Stream expansion factor | 3 |
| `sinkhorn_iterations` | Sinkhorn-Knopp iterations | 10 |
| `alpha_init` | Gating factor initialization | 0.01 |
| `hidden_dim` | Hidden dimension | 2560 |
| `dynamic_mappings` | Input-dependent mappings | true |
| `epsilon` | Sinkhorn numerical stability | 1e-6 |
| `mixed_precision` | BF16 activations | false |
| `fuse_kernels` | Enable kernel fusion | true |

## Sinkhorn-Knopp Algorithm

The doubly stochastic constraint is enforced via iterative row/column normalization:

1. Exponentiate input to ensure positivity
2. Alternately normalize rows and columns until convergence
3. Result: rows and columns each sum to 1

This guarantees the residual mixing matrix preserves signal magnitude, enabling stable training at depth.

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `metal` | Yes | Apple Silicon GPU acceleration via Metal compute shaders |
| `cuda` | No | Reserved for future NVIDIA GPU support |

## Benchmarks

```bash
cargo bench --bench sinkhorn       # Sinkhorn-Knopp at n={4,8,16,32}
cargo bench --bench mhc_kernels    # Mapping computation across presets
```

## Modules

| Module | Description |
|--------|-------------|
| `config` | Configuration and presets |
| `params` | Learnable parameters and gradients |
| `layer` | MhcLayer and MhcTransformerBlock |
| `mappings` | Mapping computation pipeline |
| `sinkhorn` | Sinkhorn-Knopp forward and backward |
| `kernels` | Metal GPU compute kernels |

## References

- [Hyper-Connections](https://arxiv.org/abs/2409.19606) — Original hyper-connection formulation
- [Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880) — Doubly stochastic constraint (DeepSeek-AI)

## License

MIT OR Apache-2.0
