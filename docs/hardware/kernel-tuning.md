# Kernel Tuning

Per-tier Metal kernel tuning — threadgroup sizes, chunk sizes, and batch multipliers.

PMetal automatically selects kernel parameters based on your device tier and GPU family. For several hot paths, Tuna now persists the resolved specialization per device/problem shape and compiles the Metal shader with matching function constants rather than relying only on host-side heuristics.

On Apple10/M5 hardware, the Metal 4 / MPP dispatcher benchmarks and persists MLX `quantized_matmul` versus the MPP path for 4-bit affine quantized linear inference, per device and problem shape.

Dense GEMM, attention, RoPE, SwiGLU and LoRA are not tuned here: they run as tensor-runtime ops, and the hand-written kernels that used to cover them measured slower than those ops.

## Fused Kernel Threadgroup Sizes

### Fused Linear Cross-Entropy

Tuna now benchmarks and persists fused linear cross-entropy threadgroup size and default chunk size per device, dtype, and problem shape. The table below is the heuristic seed used to order the benchmark candidates.

| Tier | Threads / Token | Chunk Size |
|------|-----------------|-----------|
| Base | 128 / 256 / 512 | 1024 or 2048 |
| Pro | 256 / 512 / 1024 | 2048 or 4096 |
| Max | 256 / 512 / 1024 | 4096 or 8192 |
| Ultra | 256 / 512 / 1024 | 4096 or 8192 |

`CE_THREADS_PER_TOKEN` still scales with vocabulary size and clamps to the hardware maximum. Base-tier devices seed at `128 / 256 / 512` across `< 32k`, `32k..128k`, and `> 128k` vocabularies, while Pro / Max / Ultra seed at `256 / 512 / 1024`. Wider hidden states start from a smaller chunk-size seed before benchmarking.

### Fused Merge

Tuna now benchmarks and persists `threads_per_group` and `elements_per_thread` for the standard-Metal fused-merge elementwise kernels. This path stays first-class on Apple7-Apple9 and is also reused on Apple10 for non-MPP merge workloads.

| Tier | Threads / Group | Elements / Thread |
|------|-----------------|-------------------|
| Base | 128 or 256 | 2, 4, or 8 |
| Pro | 128, 256, or 512 | 4 or 8 |
| Max | 256 or 512 | 4 or 8 |
| Ultra | 256 or 512 | 4 or 8 |

The candidate ordering is tier-aware, but the final result is benchmarked per device and problem shape and stored in `merge.json`. Persistent merge cache keys include device identity so results are safe to reuse across different Apple Silicon tiers and bins.

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
| Fused GDN | Gated Delta Network recurrence kernel — single-pass state update |
| Fused Cross-Entropy | Chunked vocabulary loss computation |
| Fused Linear Cross-Entropy | Skips logits materialization entirely, with tuned chunk/thread specialization |
| Fused Sampler | JIT-compiled token sampling |
| Async Scheduler | Double/triple-buffered GPU command scheduling |

## See Also

- [Apple Silicon Support](/hardware/apple-silicon/) — Hardware detection matrix
- [pmetal bench](/cli/bench/) — Benchmark on your hardware
