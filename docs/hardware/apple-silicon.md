# Apple Silicon Support

Hardware detection, per-chip capabilities, M1–M5 support matrix, and ANE integration.

PMetal automatically detects your Apple Silicon hardware at startup and tunes kernel parameters accordingly.

## Chip Support Matrix

| Chip | GPU Family | GPU Cores | BW (GB/s) | ANE TFLOPS | NAX | Notes |
|------|-----------|-----------|-----------|------------|-----|-------|
| M1 | Apple7 | 8 | 100 | ~11 | No | |
| M1 Pro | Apple7 | 14 | 200 | ~11 | No | |
| M1 Max | Apple7 | 24 | 400 | ~11 | No | |
| M1 Ultra | Apple7 | 48 | 800 | ~22 | No | 2-die UltraFusion |
| M2 | Apple8 | 8 | 100 | ~15 | No | |
| M2 Pro | Apple8 | 16 | 200 | ~15 | No | |
| M2 Max | Apple8 | 30 | 400 | ~15 | No | |
| M2 Ultra | Apple8 | 48 | 800 | ~30 | No | 2-die UltraFusion |
| M3 | Apple9 | 10 | 120 | 15.8 | No | Dynamic caching |
| M3 Pro | Apple9 | 18 | 273 | 15.8 | No | |
| M3 Max | Apple9 | 30 | 546 | 15.8 | No | |
| M3 Ultra | Apple9 | 60 | 800 | 31.6 | No | 32 NE cores |
| M4 | Apple9 | 10 | 120 | ~12.2 | No | Dynamic caching |
| M4 Pro | Apple9 | 20 | 273 | 12.57 | No | |
| M4 Max | Apple9 | 40 | 546 | 10.93 | No | |
| M4 Ultra | Apple9 | 80 | 800 | ~24 | No | 2-die UltraFusion |
| **M5** | **Apple10** | **10** | **120** | **~12.2** | **Yes** | NAX support |
| **M5 Pro** | **Apple10** | **20** | **273** | **~12.3** | **Yes** | |
| **M5 Max** | **Apple10** | **40** | **546** | **TBD** | **Yes** | |
| **M5 Ultra** | **Apple10** | **80** | **800** | **TBD** | **Yes** | 2-die UltraFusion |

ANE TFLOPS measured at FP16 via PMetal benchmarks.

## Auto-Detection

PMetal detects at startup:

- **GPU family** (Apple7–Apple10) and architecture generation
- **Device tier** (Base/Pro/Max/Ultra)
- **GPU core count**
- **ANE core count** and availability
- **Memory bandwidth** (persisted GPU copy benchmark with spec fallback)
- **NAX** (M5+, Apple10)
- **Metal features** (dynamic caching, mesh shaders)
- **UltraFusion topology** (via `sysctl hw.packages`)

Use `pmetal info` to display all detected hardware properties.

On UltraFusion machines, `pmetal info` now also reports the local same-process executor scaffold: per-die resource slices, the in-memory stage count, and the heuristic 32 TB/s interconnect used by PMetal's local UltraFusion planner.

Use `pmetal bench-corpus` to collect a comparable kernel report for the current machine. That corpus exercises standard-Metal hot paths on M1-M4, includes fused-merge plus initial MoE/model-family coverage across Apple7-Apple10, and adds MPP GEMM coverage on Apple10/M5 when NAX is available.

Use `pmetal bench-workload` when you want a fast end-to-end regression loop instead of only kernel microbenchmarks. The default workload uses the cached `Qwen/Qwen3-0.6B` model plus `TeichAI/gemini-3-pro-preview-high-reasoning-250x`, and the inference report now records the KV cache mode chosen on the current machine.

Across Apple7-10 GPUs, PMetal now benchmarks, persists, and reuses standard-Metal Tuna specializations for `fused_swiglu`, `fused_mlp`, `fused_norm_lora`, fused linear cross-entropy, and fused merge. Shared MLX MoE routers now use GPU-native `argpartition` selection instead of heavier full sorts or CPU top-k paths. M1-M4 remain first-class paths rather than “fallback” hardware. PMetal also now auto-selects the inference KV cache mode: on Apple7-Apple9 it prefers fp16 KV cache when the model plus context comfortably fit the device budget and only promotes to q8 when memory pressure justifies it.

## M5-Specific: NAX

M5 (Apple10, arch gen 17) introduces Neural Accelerator units within GPU cores for hardware-accelerated:

- GEMM (fused matrix multiply)
- Quantized inference (FP4/FP8)
- Scaled dot-product attention

Accessed via Metal 4.0 (`-std=metal4.0`) kernels. NAX availability is checked via `DeviceProperties::has_nax()`. PMetal currently ships the Metal 4 dispatcher, NAX-capable hardware detection, persisted Apple10/M5 MPP dispatch tuning across `32×32`, `64×32`, `32×64`, and `64×64` threadgroup variants, and persisted runtime selection between MLX fast SDPA, Metal FlashAttention, and MPP FlashAttention for supported no-custom-mask `head_dim = 64`, `80`, `96`, and `128` inference shapes, including softcapped configs. Upstream `mlx-rs` NAX passthrough is still in progress.

## ANE (Apple Neural Engine)

PMetal's ANE pipeline:

- **Dynamic Weight Pipeline**: 9 MIL kernels compiled once at startup
- **Hybrid Inference**: ANE prefill + CPU decode with KV cache
- **Power-of-2 bucketing**: Optimal kernel compilation for sequence lengths
- **CPU RMSNorm**: f32 computation on CPU to avoid fp16 ANE overflow
- **IOSurface Zero-Copy**: Shared memory surfaces for CPU-ANE transfer
- **Experimental RT Eval**: `infer` / `serve` support `--ane-real-time`, but PMetal still falls back to standard ANE if the private real-time path rejects the request on the current OS/framework; on the local M4 Max, both the tiny-kernel check and the generated SDPA forward probe still hit `ANEProgramProcessRequestDirect() ... Program Inference error` on 2026-03-23
- **M1–M5 Compatibility**: Per-matrix blobs for M1, single-blob for M3+

## See Also

- [Kernel Tuning](/hardware/kernel-tuning/) — Per-tier parameter tuning
- [pmetal info](/cli/info/) — View your hardware info
