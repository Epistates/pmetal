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
| Memory bandwidth | `pmetal-metal/src/context.rs` | Tier + family lookup table |
| NAX (Neural Accelerators in GPU) | `pmetal-metal/src/context.rs` | Apple10+ (M5) |
| ANE perf stats | `pmetal-metal/src/ane/runtime.rs` | `_ANEPerformanceStats` API |
| UltraFusion topology | — | Not detected |

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
- **Chaining API**: `_ANEChainingRequest` available (research — no working invocation yet)
- **Real-time eval**: `evaluateRealTimeWithModel:` available
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

### Matrix Tile Size (GEMM, LoRA forward)

| Tier | Apple7–9 | Apple10 (M5+, NAX) |
|------|----------|-------------------|
| Base | 32×32×32 | 64×32×32 |
| Pro | 64×32×32 | 64×64×32 |
| Max | 64×64×32 | 128×64×32 |
| Ultra | 64×64×32 | 128×64×32 |

### FlashAttention (`flash_attention.rs`)

Block size selection per head dimension:

| Head Dim | Base | Pro | Max | Ultra |
|----------|------|-----|-----|-------|
| 64 | 64×32 | 64×32 | 64×64 | 64×64 |
| 80 | 64×32 | 64×32 | 64×64 | 64×64 |
| 96 | 64×32 | 64×32 | 64×64 | 64×64 |
| 128 | 32×32 | 32×32 | 64×64 | 64×64 |
| 256 | 32×16 | 32×16 | 32×32 | 32×32 |

### Fused RMSNorm + LoRA (`fused_norm_lora.rs`)

| Tier | Threadgroup Size |
|------|-----------------|
| Base | 128 |
| Pro | 128 |
| Max | 256 |
| Ultra | 256 |

### Fused SwiGLU (`fused_swiglu.rs`)

| Tier | Threadgroup Size |
|------|-----------------|
| Base | 256 |
| Pro | 256 |
| Max | 512 |
| Ultra | 512 |

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
- [x] NAX-aware tile size tuning
- [ ] Upstream mlx-rs NAX kernel passthrough (requires mlx-rs update to MLX with NAX)
- [ ] Profile NAX vs standard kernels on M5 for quantized inference
- [ ] Benchmark NAX SDPA vs FlashAttention on M5

### P1 — ANE chaining API

`_ANEChainingRequest` with loopback could pipeline multiple layers as a single ANE program:

- [x] Class detection + telemetry
- [ ] Prototype single-chain invocation (2 layers, loopback input→output)
- [ ] Benchmark chained vs sequential dispatch latency
- [ ] If viable: integrate into ANE inference engine for multi-layer dispatch

### P2 — ANE real-time evaluation path

`_ANEClient.evaluateRealTimeWithModel:` may provide lower/more predictable latency:

- [ ] Prototype RT eval path
- [ ] Compare RT vs standard eval latency distribution
- [ ] If beneficial: add `--ane-realtime` CLI flag

### P3 — UltraFusion-aware distributed

Current distributed crate (`pmetal-distributed`) is multi-machine over TCP/mDNS. UltraFusion's 32 TB/s interconnect bandwidth could enable:

- [ ] Intra-machine model parallelism across dies (pipeline or tensor parallel)
- [ ] Die-affine buffer placement for large models that exceed single-die cache
- [ ] Hybrid: UltraFusion tensor parallel + network data parallel across machines

### P4 — Dynamic auto-tuning

Replace hardcoded tier-based parameters with runtime optimization:

- [ ] Auto-benchmark kernel configs on first run and cache optimal parameters
- [ ] Query actual memory bandwidth via IOKit/sysctl instead of tier lookup table
- [ ] M5 Pro/Max/Ultra profiling once hardware is available
