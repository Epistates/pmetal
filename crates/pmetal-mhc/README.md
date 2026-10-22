# pmetal-mhc

Manifold-Constrained Hyper-Connections (mHC) for stable deep network training.

## Overview

This crate implements Manifold-Constrained Hyper-Connections from [DeepSeek-AI](https://arxiv.org/abs/2512.24880), a technique that replaces a transformer's single residual stream with `n` parallel streams mixed by learned, doubly stochastic matrices. Constraining the residual mixing matrix to the doubly stochastic manifold with a Sinkhorn-Knopp projection keeps signal amplification bounded (Amax gain ~ 1) even in very deep networks (60+ layers).

It is a self-contained CPU reference implementation on `ndarray`: `f32` throughout, with the long reductions (the RMSNorm's mean of squares, gradient norms) accumulated in `f64`. Forward and backward passes are written out by hand and verified against finite differences and against float64 transcriptions of the paper's equations (`tests/`). Accelerated implementations are tested against it.

It depends on no other workspace crate and runs on any platform. The `pmetal` umbrella crate re-exports it behind its `mhc` feature.

## Architecture

```text
Input x  [batch, n, C]
    │
    ▼
┌───────────────────────────────────────────────────────────┐
│ compute_mappings()                                         │
│  1. x' = RMSNorm(vec(x))                      (no weight)  │
│  2. proj = x' · φ                  φ: [nC, 2n + n²]        │
│  3. H̃ = α · proj + b           per mapping: pre, post, res │
│  4. Constraints:                                           │
│     H^pre  = σ(H̃^pre) + ε                    ∈ (0, 1]      │
│     H^post = 2σ(H̃^post)                      ∈ (0, 2)      │
│     H^res  = Sinkhorn(H̃^res)       approximately doubly stochastic │
└───────────────────────────────────────────────────────────┘
    │
    ▼
apply_pre_mapping(x, H^pre) → h_in [batch, C]
    │
    ▼
┌───────────────────────────────────────────────────────────┐
│               Sublayer F (attention / FFN)                 │
└───────────────────────────────────────────────────────────┘
    │
    ▼
apply_post_res_mapping(x, h_out, H^post, H^res)
    x_{l+1} = H^res · x + H^postᵀ · h_out
    │
    ▼
Output [batch, n, C]
```

## Parameters and initialization

`MhcParams` holds the three gates `α` (initialized to `alpha_init = 0.01`), the static biases `b_pre`, `b_post`, `b_res` and one fused projection `φ` of shape `[nC, 2n + n²]` whose columns feed `H^pre`, `H^post` and `H^res` in that order: one matrix product per token gives all three mappings, in the layout the released models store. The RMSNorm has no weight of its own; a per-channel scale would only duplicate `φ`'s rows.

A fresh layer is a plain residual connection when its streams are equal (the usual start: the embedding replicated `n` times, see `expand_to_streams`). `MhcParams::identity_init` sets

- `b_pre` so that `H^pre = 1/n` (the streams' mean),
- `b_post = 0`, so `H^post = 1`,
- `b_res` to 0 on the diagonal and -8 elsewhere, so `H^res` is the identity to within ~3e-4 per entry,
- `φ = 0`;

`MhcParams::new` and `MhcParams::with_rng` add a small random `φ ~ N(0, 0.02²)`. Seed it for reproducible runs.

## Sinkhorn-Knopp

`H^res` is the iterated normalization of `exp(H̃^res)`, written as in the released DeepSeek-V4 kernel: a row softmax (the stable form of `exp`, then the first row normalization), `+ ε`, then `iterations` rounds of column and row normalization with `+ ε` in every denominator, ending on a column step. The iteration count is fixed (20, as in the paper), so the map is smooth and its gradient is exact: `sinkhorn_knopp_backward` replays the forward and runs each normalization's adjoint in reverse, O(n²) per iteration. Columns of the result sum to 1; rows converge towards it (within 1e-3 for logits within ±1.5 at 20 iterations; the more skewed the matrix, the slower).

## Installation

```toml
[dependencies]
pmetal-mhc = "0.7"
```

## Usage

```rust
use ndarray::Array3;
use pmetal_mhc::{MhcConfig, MhcError, MhcLayer};

fn main() -> Result<(), MhcError> {
    // A small layer; `MhcConfig::from_preset(MhcPreset::Medium)` is the 9B shape.
    let config = MhcConfig { hidden_dim: 16, ..MhcConfig::default() };
    let mut layer = MhcLayer::new(config)?;

    // Forward: [batch, n streams, C] in, the same shape out. The sublayer
    // (attention, an FFN) sees the streams' aggregate, [batch, C].
    let x = Array3::from_elem((2, 4, 16), 0.1f32);
    let (h_in, mappings) = layer.pre_layer(&x)?;
    let h_out = h_in.mapv(|v| 2.0 * v);
    let output = layer.post_res_layer(&x, &h_out, &mappings)?;
    assert_eq!(output.shape(), x.shape());

    // Backward: the sublayer's own backward pass runs inside the layer's.
    let grad_output = Array3::from_elem((2, 4, 16), 1.0f32);
    let (grad_x, grads) = layer.backward(&x, &h_out, &mappings, &grad_output, |g| {
        g.mapv(|v| 2.0 * v)
    })?;
    assert_eq!(grad_x.shape(), x.shape());

    layer.apply_gradients_sgd(&grads, 1e-3);
    Ok(())
}
```

### Transformer Block

```rust
use ndarray::{Array2, Array3};
use pmetal_mhc::{MhcConfig, MhcError, MhcTransformerBlock};

fn main() -> Result<(), MhcError> {
    let config = MhcConfig { hidden_dim: 16, ..MhcConfig::default() };
    let attn = |h: &Array2<f32>| h.mapv(|v| 2.0 * v);
    let ffn = |h: &Array2<f32>| h.mapv(|v| 0.5 * v);
    let block = MhcTransformerBlock::new(config, attn, ffn)?;

    let x = Array3::from_elem((2, 4, 16), 0.1f32);
    assert_eq!(block.forward(&x)?.shape(), x.shape());
    Ok(())
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
| `expansion_rate` | Number of streams `n` | 4 |
| `sinkhorn_iterations` | Row/column normalization rounds | 20 |
| `alpha_init` | Gating factor initialization | 0.01 |
| `hidden_dim` | Per-stream dimension `C` | 2560 |
| `use_dynamic_mappings` | Input-dependent mappings (else static biases only) | true |
| `epsilon` | RMSNorm, `H^pre` and Sinkhorn epsilon | 1e-6 |

## Verification

- `tests/reference_gradients.rs`: float64 transcription of the paper's equations (independent of the crate) compared with the crate's forward, and every gradient (the input, `α`, `b`, `φ`, through the Sinkhorn projection, the RMSNorm, the sublayer) compared with finite differences of that reference.
- `tests/properties.rs`: Sinkhorn invariants for arbitrary logits (property tests), bf16-rounded streams, extreme magnitudes.
- Unit tests pin Sinkhorn and the mappings to hand-computed values.

## Benchmarks

```bash
cargo bench --bench sinkhorn       # Sinkhorn-Knopp at n={4,8,16,32}, forward and backward
cargo bench --bench mhc_kernels    # Mapping computation across presets
```

## Modules

| Module | Description |
|--------|-------------|
| `config` | Configuration and presets |
| `params` | Learnable parameters, initialization and gradients |
| `layer` | MhcLayer and MhcTransformerBlock |
| `mappings` | Mapping computation, application and their backward passes |
| `sinkhorn` | Sinkhorn-Knopp forward and backward |
| `error` | `MhcError` |

## Feature flags

None.

## Requirements

- Rust 1.95 or later

## References

- [Hyper-Connections](https://arxiv.org/abs/2409.19606): the original hyper-connection formulation
- [Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880): the doubly stochastic constraint (DeepSeek-AI)

## License

MIT OR Apache-2.0
