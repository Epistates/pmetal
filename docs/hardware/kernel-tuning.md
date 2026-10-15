# Kernel Tuning

Per-tier Metal kernel tuning — threadgroup sizes, chunk sizes, and batch multipliers.

PMetal automatically selects kernel parameters based on your device tier and GPU family.

On Apple10/M5 hardware, the Metal 4 / MPP dispatcher benchmarks and persists MLX `quantized_matmul` versus the MPP path for 4-bit affine quantized linear inference, per device and problem shape.

Dense GEMM, attention, RoPE, SwiGLU and LoRA are not tuned here: they run as tensor-runtime ops, and the hand-written kernels that used to cover them measured slower than those ops.

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
| Fused Sampler | JIT-compiled token sampling |
| Async Scheduler | Double/triple-buffered GPU command scheduling |

## See Also

- [Apple Silicon Support](/hardware/apple-silicon/) — Hardware detection matrix
- [pmetal bench](/cli/bench/) — Benchmark on your hardware
