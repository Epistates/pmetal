# Apple Silicon Support

Hardware detection, chip generations, and the Neural Engine.

PMetal detects your Apple Silicon hardware at startup and chooses its paths accordingly. macOS 14 or later is required.

## Chip Generations

| Chip family | GPU family | Arch gen | NAX | Notes |
|-------------|-----------|----------|-----|-------|
| M1 / Pro / Max / Ultra | Apple7 | 13 | No | |
| M2 / Pro / Max / Ultra | Apple8 | 14 | No | |
| M3 / Pro / Max / Ultra | Apple9 | 15 | No | Dynamic caching |
| M4 / Pro / Max / Ultra | Apple9 | 16 | No | Dynamic caching |
| **M5 / Pro / Max / Ultra** | **Apple10** | **17** | **Yes** | Needs macOS 26.2 or later for NAX |

An Ultra is two Max dies joined by UltraFusion, with 32 Neural Engine cores; every other chip reports 16. GPU core counts are read from the device, so binned parts are reported as they are. Published peak memory bandwidth per chip is in [Hardware Support](/hardware-support/).

## Auto-Detection

PMetal detects at startup:

- **GPU family** (Apple7 to Apple10) and architecture generation
- **Device tier** (Base, Pro, Max, Ultra)
- **GPU core count**
- **ANE core count** and availability
- **Memory bandwidth** (measured by a cached GPU copy benchmark, else Apple's published peak)
- **NAX** (generation 17 or later, macOS 26.2 or later)
- **Metal features** (dynamic caching, mesh shaders)
- **UltraFusion topology**

Use `pmetal info` to display all detected hardware properties. On UltraFusion machines it also reports the local same-process executor plan: per-die resource slices and the number of in-memory pipeline stages.

Use `pmetal bench-corpus` to collect a comparable kernel report for the current machine, and `pmetal bench-workload` for a fast end-to-end regression loop (inference plus a short LoRA run on a cached model and dataset; it records the KV cache mode chosen on the machine).

Across Apple7 to Apple10 GPUs, PMetal runs dense GEMM, attention, RoPE, SwiGLU and LoRA as tensor-runtime ops. M1 to M4 are first-class paths, not fallbacks.

## M5: NAX

M5-generation GPUs have neural accelerator units in each GPU core, reached through Metal 4 tensor operations, which the runtime uses (through MLX's NAX kernels) for matrix multiplication, quantized matmul and attention. Detection and the runtime apply the same gate, so both agree on which device takes the NAX path.

## ANE (Apple Neural Engine)

PMetal's ANE pipeline uses Apple's private framework APIs (experimental):

- **Training** (`train --ane`): training kernels compiled once, with weights packed alongside activations in IOSurfaces. Attention and FFN kernels too large for ANE memory fall back to per-projection kernels
- **Inference** (`infer --ane`, `serve --ane`): the whole model on the ANE, several layers per program over a KV cache held in IOSurfaces, int8 weights by default. Tokens are sampled from its logits as on the GPU: temperature, top-k, top-p, min-p, the seed, and the repetition, frequency and presence penalties
- **Power-of-2 bucketing**: kernels are compiled for context lengths from 512 up to `--ane-max-seq-len`
- **CPU RMSNorm**: f32 computation on the CPU in training to avoid fp16 ANE overflow
- **IOSurface zero-copy**: shared memory surfaces for CPU-ANE transfer

## See Also

- [Kernel Tuning](/hardware/kernel-tuning/): What is tuned per device
- [pmetal info](/cli/info/): View your hardware info
