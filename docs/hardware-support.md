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
- **Chaining API**: `_ANEChainingRequest` available (research — PMetal can prepare experimental loopback requests, but stable execution is not solved yet)
- **Real-time eval**: `_ANEClient.evaluateRealTimeWithModel:` is detected; PMetal exposes an experimental `--ane-real-time` opt-in with automatic fallback to standard ANE if the private path fails
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

### Matrix Tile Size (standard GEMM, LoRA forward)

| Tier | Apple7–9 | Apple10 (M5+, NAX) |
|------|----------|-------------------|
| Base | 32×32×32 | 64×32×32 |
| Pro | 64×32×32 | 64×64×32 |
| Max | 64×64×32 | 128×64×32 |
| Ultra | 64×64×32 | 128×64×32 |

These tier tables apply to the standard Metal GEMM/LoRA kernels. On Apple10/M5 hardware, the Metal 4 / MPP dispatcher now auto-tunes and persists among `32×32` / `1`-simdgroup, `64×32` / `2`-simdgroup, `32×64` / `2`-simdgroup, and `64×64` / `4`-simdgroup kernel variants, plus Morton-vs-linear tile walk order. Aligned full tiles use static extents, and the dispatcher exposes an async command-buffer API for overlap. Apple7-9 continue to use the standard Metal kernels.

### FlashAttention (`flash_attention.rs`)

Block size selection per head dimension:

| Head Dim | Base | Pro | Max | Ultra |
|----------|------|-----|-----|-------|
| 64 | 64×32 | 64×64 | 64×64 | 64×64 |
| 80 | 32×32 | 64×32 | 64×32 | 64×32 |
| 96 | 32×32 | 64×32 | 64×32 | 64×32 |
| 128 | 32×32 | 32×32 | 64×32 | 64×32 |
| 256 | 16×16 | 16×16 | 32×16 | 32×16 |

### Fused RMSNorm + LoRA (`fused_norm_lora.rs`)

Tuna now benchmarks and persists the effective specialization for this kernel per problem shape, and the Metal shader is compiled with the matching `THREADS_PER_TOKEN` constant instead of relying on dead host-side heuristics.

Heuristic seed values before benchmark candidate generation:

| Tier | Threads / Token | Tiled Path |
|------|-----------------|-----------|
| Base | 128 | `out_features > 256` |
| Pro | 256 | `out_features > 256` |
| Max | 512 | `out_features > 256` |
| Ultra | 512 | `out_features > 256` |

If the caller disables tiling explicitly, PMetal respects that and stays on the non-tiled path.

### Fused SwiGLU (`fused_swiglu.rs`)

Tuna now benchmarks and persists the effective specialization for standard-Metal `fused_swiglu` / `fused_mlp`, and the shader is compiled with matching `THREADS_PER_TOKEN` and `SWIGLU_CHUNK_SIZE` function constants.

Heuristic seed values before benchmark candidate generation:

| Tier | Threads / Token | Chunk Size |
|------|-----------------|-----------|
| Base | 128 or 256 | 1024 or 2048 |
| Pro | 256 | 2048 or 4096 |
| Max | 512 | 2048 or 4096 |
| Ultra | 512 | 2048 or 4096 |

The lower Base-tier values apply to smaller `intermediate_size` shapes; higher values apply to larger MLPs.

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
- [ ] Upstream mlx-rs NAX kernel passthrough (requires mlx-rs update to MLX with NAX)
- [x] Benchmark and persist Apple10/M5 MPP FlashAttention vs Metal FlashAttention vs MLX fast SDPA for supported no-custom-mask `head_dim = 64`, `80`, `96`, and `128` inference shapes, including softcapped configs

### P1 — ANE chaining API

`_ANEChainingRequest` with loopback could pipeline multiple layers as a single ANE program:

- [x] Class detection + telemetry
- [x] Experimental loopback request construction + `_ANEClient.prepareChainingWithModel:` submission API
- [ ] Stable single-chain execution on hardware without private-framework aborts (`cargo test -p pmetal-metal test_prepare_loopback_chain_smoke -- --ignored --nocapture` on the local M4 Max still aborted the child process on 2026-03-23)
- [ ] Benchmark chained vs sequential dispatch latency
- [ ] If viable: integrate into ANE inference engine for multi-layer dispatch

### P2 — ANE real-time evaluation path

`_ANEClient.evaluateRealTimeWithModel:` may provide lower/more predictable latency:

- [x] Runtime probe + `AneModel::evaluate_real_time*` wrapper
- [x] Experimental `--ane-real-time` opt-in for `infer` / `serve`
- [x] Automatic fallback to standard ANE dispatch if the private RT path fails
- [ ] Compare RT vs standard eval latency distribution (PMetal now has both a tiny-kernel check and a generated SDPA forward latency probe, but on the local M4 Max both ignored tests still hit `ANEProgramProcessRequestDirect() ... Program Inference error` on 2026-03-23)
- [ ] Promote beyond experimental only after measured latency wins and stable correctness

### P3 — UltraFusion-aware distributed

Current distributed crate (`pmetal-distributed`) is multi-machine over TCP/mDNS. UltraFusion's 32 TB/s interconnect bandwidth could enable:

- [x] Add a local UltraFusion planner plus same-process in-memory stage transport for dual-die pipeline execution scaffolding
- [ ] Validate measurable gain for local UltraFusion pipeline execution on real Ultra hardware
- [ ] Intra-machine model parallelism across dies (pipeline or tensor parallel)
- [ ] Die-affine buffer placement for large models that exceed single-die cache
- [ ] Hybrid: UltraFusion tensor parallel + network data parallel across machines

### P4 — Dynamic auto-tuning

Replace hardcoded tier-based parameters with runtime optimization:

- [x] Persist hot-path inference backend benchmarks across launches (FlashAttention vs MLX SDPA, MPP vs MLX matmul)
- [x] Benchmark and persist MLX CutCrossEntropy vs Metal fused linear cross-entropy backend choice for benchmarkable shapes
- [x] Benchmark and persist fused linear cross-entropy thread/chunk specialization per device, dtype, and problem shape
- [x] Benchmark and persist standard Metal FlashAttention block-size selection for supported head dimensions
- [x] Wire broader standard-Metal fused kernels through persisted Tuna specialization (`fused_swiglu`, `fused_mlp`, `fused_norm_lora`, fused linear cross-entropy)
- [x] Benchmark and persist standard-Metal `fused_swiglu`, `fused_mlp`, and `fused_norm_lora` kernel specialization choices
- [x] Benchmark and persist standard-Metal `fused_merge` specialization choices, with device-safe cache keys and `merge.json` disk persistence
- [x] Add a structured `pmetal bench-corpus` kernel benchmark command for comparable per-tier measurements on M1-M4 and M5, including initial MoE/model-family cases
- [x] Add a real `pmetal bench-workload` command for fast end-to-end Qwen3/dataset regression runs on local Apple Silicon, and record the selected inference KV cache mode in the report
- [x] Auto-select fp16 vs q8 KV cache for inference based on model/context fit instead of hardcoding q8 as the default on Apple7-Apple9
- [ ] Expand first-run autotuning only where checked-in corpus data still shows meaningful upside beyond the now-covered hot paths
- [x] Measure and persist approximate GPU unified-memory bandwidth via copy benchmark, with spec-table fallback when probing is unavailable
- [x] Record local M4 Max benchmark-corpus reports in `.strategy/bench_corpus_m4_max_2026_03_23.json` and `.strategy/bench_corpus_m4_max_quick_2026_03_23.json`
- [ ] Run `pmetal bench-corpus` on representative M1/M2/M3/M4/M5 hardware and check in the reports that justify default choices
- [ ] M5 Pro/Max/Ultra profiling once hardware is available
