# Apple Silicon Hardware Support

Status of hardware-specific optimizations in PMetal.

## Detection System

| Component | File | Status |
|-----------|------|--------|
| GPU family (`Apple7`–`Apple10`) | `pmetal-metal/src/context.rs` | Name-string based |
| Device tier (`Base`/`Pro`/`Max`/`Ultra`) | `pmetal-metal/src/context.rs` | Name-string based |
| Feature flags (dynamic caching, mesh shaders, NAX) | `pmetal-metal/src/context.rs` | Derived from family |
| Architecture generation (14–17) | `pmetal-metal/src/context.rs` | Mapped from GPU family |
| GPU core count | `pmetal-metal/src/context.rs` | Estimated from name + tier |
| ANE core count | `pmetal-metal/src/context.rs` | Tier-based (16/32) |
| Memory bandwidth | `pmetal-metal/src/context.rs` | Persisted GPU copy benchmark + spec fallback |
| NAX (Neural Accelerators in GPU) | `pmetal-metal/src/context.rs` | Apple10+ (M5) |
| ANE perf stats | `pmetal-metal/src/ane/runtime.rs` | `_ANEPerformanceStats` API |
| UltraFusion topology | `pmetal-metal/src/context.rs` | Detected via `sysctl hw.packages` |

## Per-Chip Support Matrix

| Chip | GPU Family | Arch Gen | Tier | GPU Cores | BW (GB/s) | ANE TFLOPS | NAX | Notes |
|------|-----------|----------|------|-----------|-----------|------------|-----|-------|
| M1 | Apple7 | 14 | Base | 8 | 100 | ~11 | No | |
| M1 Pro | Apple7 | 14 | Pro | 14 | 200 | ~11 | No | |
| M1 Max | Apple7 | 14 | Max | 24 | 400 | ~11 | No | |
| M1 Ultra | Apple7 | 14 | Ultra | 48 | 800 | ~22 | No | 2-die UltraFusion |
| M2 | Apple8 | 15 | Base | 8 | 100 | ~15 | No | |
| M2 Pro | Apple8 | 15 | Pro | 16 | 200 | ~15 | No | |
| M2 Max | Apple8 | 15 | Max | 30 | 400 | ~15 | No | |
| M2 Ultra | Apple8 | 15 | Ultra | 48 | 800 | ~30 | No | 2-die UltraFusion |
| M3 | Apple9 | 16 | Base | 10 | 120 | 15.8 | No | Dynamic caching |
| M3 Pro | Apple9 | 16 | Pro | 18 | 273 | 15.8 | No | |
| M3 Max | Apple9 | 16 | Max | 30 | 546 | 15.8 | No | |
| M3 Ultra | Apple9 | 16 | Ultra | 60 | 800 | 31.6 | No | 2-die, 32 NE cores |
| M4 | Apple9 | 16 | Base | 10 | 120 | ~12.2* | No | Dynamic caching |
| M4 Pro | Apple9 | 16 | Pro | 20 | 273 | 12.57* | No | |
| M4 Max | Apple9 | 16 | Max | 40 | 546 | 10.93* | No | 64 ms/step training |
| M4 Ultra | Apple9 | 16 | Ultra | 80 | 800 | ~24* | No | 2-die UltraFusion |
| **M5** | **Apple10** | **17** | **Base** | **10** | **120** | **~12.2*** | **Yes** | NAX, same H16 ANE |
| **M5 Pro** | **Apple10** | **17** | **Pro** | **20** | **273** | **12.17-12.44*** | **Yes** | |
| **M5 Max** | **Apple10** | **17** | **Max** | **40** | **546** | **TBD** | **Yes** | |
| **M5 Ultra** | **Apple10** | **17** | **Ultra** | **80** | **800** | **TBD** | **Yes** | 2-die UltraFusion |

*ANE TFLOPS measured at FP16 via pmetal benchmarks. Apple's rated "TOPS" for M4+ uses INT8/mixed precision (~38 TOPS for M4 Pro), not comparable to FP16 measurements.

## M5-Specific Features

### NAX (Neural Accelerators in GPU Cores)

M5 (Apple10, arch gen 17) introduces Neural Accelerator units within GPU cores, enabling hardware-accelerated:
- GEMM (fused matrix multiply)
- Quantized inference (FP4/FP8)
- Scaled dot-product attention

These are accessed via Metal 4.0 (`-std=metal4.0`) kernels. MLX upstream has production NAX kernel libraries (`steel_gemm_fused_nax.metal`, `quantized_nax.metal`, `steel_attention_nax.metal`). NAX availability: `architecture_gen >= 17` (checked via `DeviceProperties::has_nax()`).

### ANE (Apple Neural Engine) on M5

- **ANE Family**: H16 (same as M4 — no architectural upgrade)
- **NE Cores**: 16 (Pro/Max/Base), 32 (Ultra via UltraFusion)
- **Measured FP16 TFLOPS**: 12.17–12.44 (comparable to M4 Pro)
- **Training**: 101–120 ms/step (vs M4 Max at 64 ms/step)
- **Weight reload**: NOT supported (weights baked at compile time)
- **Chaining API**: `_ANEChainingRequest` exists, but submitting a loopback request aborted the process on an M4 Max, so PMetal doesn't use it. PMetal runs several layers per ANE program instead
- **Real-time eval**: `_ANEClient.evaluateRealTimeWithModel:` exists, but it failed with `Program Inference error` on an M4 Max and won't load a model on macOS 27, so PMetal uses standard evaluation
- **Perf stats**: `_ANEPerformanceStats.hwExecutionTime` provides ns-precision hardware timing

### ANE MIL Compatibility

| Feature | M1 | M3 | M4 | M5 |
|---------|-----|-----|-----|-----|
| `program(1.3) / ios18` | Partial | Yes | Yes | Yes |
| Single-blob weights | Fail | Yes | Yes | Yes |
| Per-matrix weight blobs | Yes | Yes | Yes | Yes |
| Channel flexibility | Unknown | ch=512 only | Flexible | Flexible |
| BLOBFILE offset refs | Fail | Yes | Yes | Yes |
| CPU RMSNorm workaround | N/A | Needed | Needed | Needed |

## Kernel Tuning by Tier

Dense GEMM, attention, RoPE, SwiGLU and LoRA run as tensor-runtime ops (the runtime uses MLX's own NAX kernels on Apple10/M5), so there are no per-tier tile tables for them: the hand-written kernels that used to cover them measured slower than those ops and were removed.

### Fused Linear Cross-Entropy (`fused_cross_entropy.rs`)

The fused linear CE path now benchmarks and persists both its `CE_THREADS_PER_TOKEN` specialization and default vocabulary chunk size per device, dtype, and problem shape. The MLX CutCrossEntropy fallback uses the same resolved default chunk size when the caller leaves the chunk size at the default, so M1-M4 and M5 share the measured chunk choice instead of drifting apart.

Heuristic seed values before benchmark candidate generation:

| Tier | Threads / Token | Chunk Size |
|------|-----------------|-----------|
| Base | 128 / 256 / 512 | 1024 or 2048 |
| Pro | 256 / 512 / 1024 | 2048 or 4096 |
| Max | 256 / 512 / 1024 | 4096 or 8192 |
| Ultra | 256 / 512 / 1024 | 4096 or 8192 |

Thread count seeds still scale with vocabulary size and clamp to the device maximum:
- Base-tier: `< 32k vocab` → `128`, `32k..128k` → `256`, `> 128k` → `512`
- Pro / Max / Ultra: `< 32k vocab` → `256`, `32k..128k` → `512`, `> 128k` → `1024`

Chunk-size seeds still narrow for wider hidden states:
- Base: `1024` once `hidden_size >= 4096`
- Pro: `2048` once `hidden_size >= 8192`
- Max / Ultra: `4096` once `hidden_size >= 8192`

### Batch Size Multiplier

| Tier | Multiplier |
|------|-----------|
| Base | 1x |
| Pro | 2x |
| Max | 4x |
| Ultra | 8x |

## Gaps & Future Work

### P0 — NAX kernel integration

MLX upstream has NAX-optimized kernels for M5. Integration path:

- [x] M5 detection (Apple10, arch gen 17)
- [x] NAX availability flag (`has_nax`)
- [x] Tier-aware baseline tile recommendations for standard Metal kernels
- [x] Async MPP command-buffer API in Rust dispatcher
- [x] Persisted MPP Morton walk-order auto-tuning on Apple10/M5
- [x] Static full-tile MPP extents for aligned M/N dispatches
- [x] Benchmark and persist MLX vs MPP backend choice for 4-bit affine quantized linear inference on Apple10/M5
- [x] Tier-aware MPP dispatcher tuning across `32×32`, `64×32`, `32×64`, and `64×64` MPP tile variants on Apple10/M5
- [ ] Track upstream MLX NAX API changes and update `pmetal-runtime` bindings as needed

### P1 — UltraFusion-aware distributed

Current distributed crate (`pmetal-distributed`) is multi-machine over TCP/mDNS. UltraFusion's 32 TB/s interconnect bandwidth could enable:

- [x] Add a local UltraFusion planner plus same-process in-memory stage transport for dual-die pipeline execution scaffolding
- [ ] Validate measurable gain for local UltraFusion pipeline execution on real Ultra hardware
- [ ] Intra-machine model parallelism across dies (pipeline or tensor parallel)
- [ ] Die-affine buffer placement for large models that exceed single-die cache
- [ ] Hybrid: UltraFusion tensor parallel + network data parallel across machines

### P2 — Dynamic auto-tuning

Replace hardcoded tier-based parameters with runtime optimization:

- [x] Persist hot-path inference backend benchmarks across launches (MPP vs MLX matmul for quantized linears)
- [x] Add a structured `pmetal bench-corpus` kernel benchmark command for comparable per-tier measurements on M1-M4 and M5, including initial MoE/model-family cases
- [x] Add a real `pmetal bench-workload` command for fast end-to-end Qwen3/dataset regression runs on local Apple Silicon, and record the selected inference KV cache mode in the report
- [x] Auto-select fp16 vs q8 KV cache for inference based on model/context fit instead of hardcoding q8 as the default on Apple7-Apple9
- [ ] Expand first-run autotuning only where checked-in corpus data still shows meaningful upside beyond the now-covered hot paths
- [x] Measure and persist approximate GPU unified-memory bandwidth via copy benchmark, with spec-table fallback when probing is unavailable
- [x] Record M4 Max benchmark-corpus reports (standard and `--quick` runs, 2026-03-23)
- [ ] Run `pmetal bench-corpus` on representative M1/M2/M3/M4/M5 hardware and check in the reports that justify default choices
- [ ] M5 Pro/Max/Ultra profiling once hardware is available
