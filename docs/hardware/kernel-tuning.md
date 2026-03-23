# Kernel Tuning

Per-tier Metal kernel tuning — matrix tiles, FlashAttention blocks, threadgroup sizes, chunk sizes, and batch multipliers.

PMetal automatically selects kernel parameters based on your device tier and GPU family. For several hot paths, Tuna now persists the resolved specialization per device/problem shape and compiles the Metal shader with matching function constants rather than relying only on host-side heuristics.

## Matrix Tile Size (GEMM, LoRA forward)

| Tier | Apple7–9 | Apple10 (M5+, NAX) |
|------|----------|-------------------|
| Base | 32×32×32 | 64×32×32 |
| Pro | 64×32×32 | 64×64×32 |
| Max | 64×64×32 | 128×64×32 |
| Ultra | 64×64×32 | 128×64×32 |

These tier tables apply to the standard Metal GEMM/LoRA kernels. On Apple10/M5 hardware, the Metal 4 / MPP dispatcher now auto-tunes and persists among `32×32` / `1`-simdgroup, `64×32` / `2`-simdgroup, `32×64` / `2`-simdgroup, and `64×64` / `4`-simdgroup variants, plus Morton-vs-linear tile walk order. Aligned M/N tiles use static full-tile extents. Apple7-9 continue to use the standard Metal kernels. For 4-bit affine quantized linear inference, Apple10/M5 also benchmarks and persists MLX `quantized_matmul` versus the MPP path per device and problem shape. Standard Metal FlashAttention now benchmarks and persists among the known-valid per-head-dimension block pairs instead of relying only on the tier table below. For supported `head_dim = 64`, `96`, and `128` inference attention shapes, Apple10/M5 also benchmarks and persists MLX fast SDPA vs Metal FlashAttention vs MPP FlashAttention, rejecting MPP candidates that diverge numerically from the MLX reference.

## FlashAttention Block Sizes

Baseline block-size seed per head dimension before persisted tuning:

| Head Dim | Base | Pro | Max | Ultra |
|----------|------|-----|-----|-------|
| 64 | 64×32 | 64×64 | 64×64 | 64×64 |
| 80 | 32×32 | 64×32 | 64×32 | 64×32 |
| 96 | 32×32 | 64×32 | 64×32 | 64×32 |
| 128 | 32×32 | 32×32 | 64×32 | 64×32 |
| 256 | 16×16 | 16×16 | 32×16 | 32×16 |

## Fused Kernel Threadgroup Sizes

### Fused RMSNorm + LoRA

Tuna now benchmarks and persists `THREADS_PER_TOKEN` plus the tiled/non-tiled path choice for this kernel. The table below is the heuristic seed used to order the benchmark candidates.

| Tier | Threads / Token | Tiled Path |
|------|-----------------|-----------|
| Base | 128 | `out_features > 256` |
| Pro | 256 | `out_features > 256` |
| Max | 512 | `out_features > 256` |
| Ultra | 512 | `out_features > 256` |

### Fused SwiGLU / Fused MLP

Tuna now benchmarks and persists `THREADS_PER_TOKEN` and `SWIGLU_CHUNK_SIZE` for these standard-Metal kernels. The table below is the heuristic seed used to order the benchmark candidates.

| Tier | Threads / Token | Chunk Size |
|------|-----------------|-----------|
| Base | 128 or 256 | 1024 or 2048 |
| Pro | 256 | 2048 or 4096 |
| Max | 512 | 2048 or 4096 |
| Ultra | 512 | 2048 or 4096 |

### Fused Linear Cross-Entropy

Tuna now benchmarks and persists fused linear cross-entropy threadgroup size and default chunk size per device, dtype, and problem shape. The table below is the heuristic seed used to order the benchmark candidates.

| Tier | Threads / Token | Chunk Size |
|------|-----------------|-----------|
| Base | 128 / 256 / 512 | 1024 or 2048 |
| Pro | 256 / 512 / 1024 | 2048 or 4096 |
| Max | 256 / 512 / 1024 | 4096 or 8192 |
| Ultra | 256 / 512 / 1024 | 4096 or 8192 |

`CE_THREADS_PER_TOKEN` still scales with vocabulary size and clamps to the hardware maximum. Base-tier devices seed at `128 / 256 / 512` across `< 32k`, `32k..128k`, and `> 128k` vocabularies, while Pro / Max / Ultra seed at `256 / 512 / 1024`. Wider hidden states start from a smaller chunk-size seed before benchmarking.

## Batch Size Multiplier

| Tier | Multiplier |
|------|-----------|
| Base | 1× |
| Pro | 2× |
| Max | 4× |
| Ultra | 8× |

## Metal GPU Optimizations

| Kernel | Description |
|--------|-------------|
| FlashAttention | O(n) memory attention with fused softmax, tier-aware block sizes |
| Fused GDN | Gated Delta Network recurrence kernel — single-pass state update |
| Fused LoRA | Combined forward pass for adapter layers (~2× speedup) |
| Fused Cross-Entropy | Chunked vocabulary loss computation |
| Fused Linear Cross-Entropy | Skips logits materialization entirely, with tuned chunk/thread specialization |
| Fused RoPE | Rotary position embeddings in-kernel |
| Fused SwiGLU | Fused gate + activation with benchmarked-and-persisted Tuna thread/chunk specialization |
| Fused RMSNorm + LoRA | Combined normalization and adapter projection with benchmarked-and-persisted Tuna thread/tiled specialization |
| Fused Sampler | JIT-compiled token sampling |
| Fused MLP | Combined gate/up/down projections |
| Async Scheduler | Double/triple-buffered GPU command scheduling |

## See Also

- [Apple Silicon Support](/hardware/apple-silicon/) — Hardware detection matrix
- [pmetal bench](/cli/bench/) — Benchmark on your hardware
