# Apple Silicon Hardware Support

Status of hardware-specific optimizations in PMetal.

## Detection System

| Component | File | Status |
|-----------|------|--------|
| GPU family (`Apple7`–`Apple10`) | `pmetal-metal/src/context.rs` | Name-string based |
| Device tier (`Base`/`Pro`/`Max`/`Ultra`) | `pmetal-metal/src/context.rs` | Name-string based |
| Feature flags (dynamic caching, mesh shaders) | `pmetal-metal/src/context.rs` | Derived from family |
| GPU core count | — | Not detected |
| Memory bandwidth | — | Hardcoded per tier |
| UltraFusion topology | — | Not detected |

## Per-Chip Support Matrix

| Chip | GPU Family | Tier | Kernel Tuning | Multi-Die | Notes |
|------|-----------|------|---------------|-----------|-------|
| M1 | Apple7 | Base | Baseline | N/A | |
| M1 Pro | Apple7 | Pro | Yes | N/A | |
| M1 Max | Apple7 | Max | Yes | N/A | |
| M1 Ultra | Apple7 | Ultra | Yes | No | UltraFusion, 2 dies |
| M2 | Apple8 | Base | Baseline | N/A | |
| M2 Pro | Apple8 | Pro | Yes | N/A | |
| M2 Max | Apple8 | Max | Yes | N/A | |
| M2 Ultra | Apple8 | Ultra | Yes | No | UltraFusion, 2 dies |
| M3 | Apple9 | Base | Baseline | N/A | Dynamic caching |
| M3 Pro | Apple9 | Pro | Yes | N/A | |
| M3 Max | Apple9 | Max | Yes | N/A | |
| M3 Ultra | Apple9 | Ultra | Yes | No | UltraFusion, 2 dies |
| M4 | Apple9 | Base | Baseline | N/A | Dynamic caching |
| M4 Pro | Apple9 | Pro | Yes | N/A | |
| M4 Max | Apple9 | Max | Yes | N/A | |
| M4 Ultra | Apple9 | Ultra | Yes | No | UltraFusion, 2 dies |
| M5 | Apple10 | Base | No (falls through to Apple9) | N/A | Not yet shipping |
| M5 Pro | Apple10 | Pro | No | N/A | |
| M5 Max | Apple10 | Max | No | N/A | |
| M5 Ultra | Apple10 | Ultra | No | No | |

## Kernel Tuning by Tier

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

### P0 — M4 Ultra multi-die awareness

Metal presents UltraFusion as a single unified GPU, so correctness is not affected. However, kernel dispatch is not topology-aware — threadgroup placement doesn't account for cross-die latency or NUMA-like memory affinity. Potential wins:

- [ ] Query actual GPU core count via Metal API (`maxThreadgroupMemoryLength`, recommended threadgroup size hints) instead of hardcoding per tier
- [ ] Profile whether large reductions (RMSNorm, softmax) benefit from die-aware partitioning
- [ ] Benchmark current FlashAttention block sizes on M4 Ultra — the 2-die topology may favor different tiling

### P1 — M5 kernel tuning

Apple10 family falls through to Apple9 paths. When M5 ships:

- [ ] Profile all fused kernels on M5 hardware
- [ ] Identify new Metal GPU family capabilities (Apple10-specific features)
- [ ] Tune block sizes, threadgroup sizes, and tile dimensions
- [ ] Check if `has_neural_accelerators_in_gpu` (speculated Apple10 feature) enables new kernel patterns

### P2 — Dynamic tuning

Replace hardcoded tier-based parameters with runtime-queried values:

- [ ] Use `MTLDevice.maxThreadsPerThreadgroup` for threadgroup sizing
- [ ] Use `MTLDevice.recommendedMaxWorkingSetSize` for buffer allocation strategy
- [ ] Query actual memory bandwidth via IOKit/sysctl instead of tier lookup table
- [ ] Auto-benchmark kernel configs on first run and cache optimal parameters

### P3 — UltraFusion-aware distributed

Current distributed crate (`pmetal-distributed`) is multi-machine over TCP/mDNS. UltraFusion's 32 TB/s interconnect bandwidth could enable:

- [ ] Intra-machine model parallelism across dies (pipeline or tensor parallel)
- [ ] Die-affine buffer placement for large models that exceed single-die cache
- [ ] Hybrid: UltraFusion tensor parallel + network data parallel across machines
