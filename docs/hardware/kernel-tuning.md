# Kernel Tuning

What PMetal chooses per device, and which custom Metal kernels it carries.

Dense GEMM, attention, RoPE, SwiGLU, LoRA and cross-entropy run as tensor-runtime ops in a lazy graph (with autodiff). The hand-written kernels that once covered them measured slower than those ops and were removed, so there are no per-tier tile or threadgroup tables to tune.

## Batch Size Multiplier

`DeviceProperties::batch_size_multiplier` scales a recommended batch size by tier:

| Tier | Multiplier |
|------|-----------|
| Base | 1x |
| Pro | 2x |
| Max | 4x |
| Ultra | 8x |

## Custom Metal Kernels

| Kernel | Where | Description |
|--------|-------|-------------|
| Gated DeltaNet recurrence | `pmetal-runtime` (`nn::gated_delta`) | Fused Metal kernel for Qwen3.5-family linear-attention layers at inference; training uses the differentiable chunkwise algorithm |
| Fused AdamW | `pmetal-metal` | One-dispatch optimizer update over every parameter |
| Weight-gradient GEMM | `pmetal-metal` | The f32 dW GEMM the ANE trainer batches into one command buffer |
| Custom kernels from Rust | `pmetal_runtime::MetalKernel` | Runs a Metal kernel source through MLX's `metal_kernel` |

Per-device choices that are made at run time, not by a tuning table: the KV cache mode (`bench-workload` records which it picked), the memory bandwidth figure (measured once by a GPU copy benchmark and cached), and the NAX path on M5 with macOS 26.2 or later.

## See Also

- [Apple Silicon Support](/hardware/apple-silicon/): Hardware detection
- [pmetal bench](/cli/bench/): Benchmark on your hardware
