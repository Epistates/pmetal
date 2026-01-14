# Optimization Transfer Analysis: Inference/Training → Merging & Distillation

This document analyzes optimization patterns used in PMetal's inference and training code that can be applied to improve the performance of model merging and knowledge distillation operations.

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Source Optimizations](#source-optimizations)
   - [Fused Operations](#fused-operations)
   - [Online Softmax / Streaming](#online-softmax--streaming-computation)
   - [Batched Command Buffers](#batched-command-buffer-pattern)
   - [Zero-Copy Bridging](#zero-copy-bridging)
   - [Kernel Auto-Tuning](#kernel-auto-tuning)
3. [Training-Specific Optimizations](#training-specific-optimizations)
4. [Current Merge Implementation](#current-merge-implementation)
5. [Current Distillation Implementation](#current-distillation-implementation)
6. [Transfer Opportunities](#optimization-transfer-opportunities)
7. [Implementation Roadmap](#implementation-roadmap)
8. [Code References](#code-locations--references)

---

## Executive Summary

Analysis of the PMetal codebase reveals **14 concrete optimization opportunities** for merging and distillation, with 5 being directly applicable and high-impact:

| Optimization | Expected Gain | Complexity |
|--------------|---------------|------------|
| Fused Merge Kernels | 3-5x speedup | Medium |
| Batched Command Buffers | 20-40% speedup | Low |
| Online Magnitude Thresholding | 2x speedup | Medium |
| Zero-Copy Tensor Access | 30-50% memory reduction | Medium |
| Distillation Loss+Grad Fusion | ~40% speedup | Medium |

---

## Source Optimizations

### Fused Operations

**Location**: `pmetal-mlx/src/kernels/` and `pmetal-metal/src/kernels/`

The core pattern: **Combine multiple MLX operations → Single Metal dispatch → Eliminate intermediate allocations**

#### Fused Attention (`fused_attention.rs`)
- Combines scaled dot-product attention in a single Metal kernel
- Eliminates intermediate tensor materialization
- Avoids `expand_kv_heads` overhead for GQA/MQA
- **Performance**: 30-50% faster than manual SDPA for single-token inference

#### Fused LoRA (`fast_lora.rs`, `fused_lora.rs`)
- Pre-transposes matrices to avoid transpose overhead
- Scale fused into LoRA B to avoid separate multiply
- Reduces forward pass complexity:
  ```
  Before: y = x @ W.T + scale * (x @ A.T) @ B.T
  After:  y = x @ W_t + (x @ A_t) @ B_scaled_t
  ```
- **Performance**: ~2x speedup on forward+backward combined

#### Fused Cross-Entropy (`cross_entropy.rs`, `metal_cross_entropy.rs`)
- Combines softmax and loss computation in single kernel
- Uses online softmax to avoid materializing full probability tensors
- **Memory**: Reduces from O(vocab) to O(1) per token

#### Fused SwiGLU (`metal_swiglu.rs`)
- Combines gate and linear projections in single dispatch

---

### Online Softmax / Streaming Computation

**Location**: `pmetal-metal/src/kernels/fused_distill.rs`

Instead of materializing `softmax(logits)`, use single-pass online computation:

```rust
/// Uses online softmax to avoid O(vocab) memory per token
/// - KL Divergence, Jensen-Shannon, Soft Cross-Entropy
/// - Temperature scaling built into kernel
/// - SIMD parallelization for large vocabularies
```

**Key Features**:
- Caches logsumexp for efficient backward pass
- Already applied to distillation losses
- Transferable to merge sparsification

---

### Batched Command Buffer Pattern

**Location**: `pmetal-metal/src/kernels/fused_training.rs`

```rust
/// Single command buffer accumulates multiple kernel dispatches
/// Instead of: create_buffer → encode → commit → wait (per kernel)
/// Do:        create_buffer → [encode x100] → commit → wait (once)
```

**Performance Impact**:
- Eliminates GPU-CPU sync overhead (~0.1ms per sync)
- With 100+ kernel dispatches per step: 10-15ms saved per training step
- Current training: ~1740 tok/s → Target with full batching: ~2400 tok/s (40% gain)

---

### Zero-Copy Bridging

**Location**: `pmetal-trainer/src/mlx_metal_optimizer.rs`, `pmetal-distill/src/losses/kl_divergence.rs`

```rust
/// MLX and Metal share unified memory on Apple Silicon
/// mlx_sys::mlx_array_data_float32() → raw pointer
/// metal_buffer_from_ptr() → MetalBufferView (zero-copy)
/// Fused Metal Kernel → mlx_array_set() to update existing arrays
```

**Key Innovation**: Uses `mlx_array_set()` to update existing MLX arrays with Metal kernel results
- Preserves array identity
- Maintains computational graph connectivity
- Enables proper gradient flow during backpropagation

---

### Kernel Auto-Tuning

**Location**: `pmetal-metal/src/tuna.rs`

```rust
/// Tuner automatically finds optimal kernel parameters
/// - Check cache for known problem size
/// - Generate candidate tile configurations (32x32, 64x32, etc.)
/// - Benchmark each candidate
/// - Cache winner for future runs
```

**Use Case**: Different hardware (M1 vs M3 Max) requires different tile sizes for optimal performance.

---

## Training-Specific Optimizations

### Gradient Checkpointing

**Location**: `pmetal-mlx/src/gradient_checkpoint.rs`

**Memory Savings**:
- Standard training: O(L) memory for L layers
- Full checkpointing: O(1) memory, O(2L) compute
- Block checkpointing: O(k) memory, O(L + L/k) compute

**Pattern**: Trade memory for recomputation by not storing intermediate activations.

### KV Cache Optimization

**Location**: `pmetal-mlx/src/kv_cache.rs`

- Store keys/values in **attention format** `[B, heads, seq, head_dim]`
- Matches MLX implementation, eliminates transpose overhead
- Supports: Standard cache, Sliding window, Rotating (circular buffer)

### Grouped GEMM MoE

**Location**: `pmetal-mlx/src/grouped_gemm_moe.rs`

```
Naive:   for expert in 0..num_experts:
           tokens = gather(tokens_for_expert)
           output = expert(tokens)  // separate GEMM

Grouped: sort_by_expert()
         batched_gemm(sorted_tokens, all_expert_weights)  // single kernel
         scatter_back()
```

**Performance**: 1.75-1.8x speedup on MoE models

### Offloading Strategy

**Location**: `pmetal-mlx/src/offloading.rs`

**Targets**:
- Embedding offloading: 20-30% reduction for large vocab
- Activation offloading: 30-40% during training
- Combined: 60% reduction for extreme cases

**Modes**:
1. CPU offloading (unified memory, CPU-preferred)
2. Disk offloading (memory-mapped)
3. Lazy loading (on-demand)

### Memory Management

**Location**: `pmetal-mlx/src/memory.rs`

Monitor RSS via `getrusage`, control MLX's caching allocator.

---

## Current Merge Implementation

### Merge Methods

**Location**: `pmetal-merge/src/methods/`

| Method | Pattern | Complexity |
|--------|---------|-----------|
| **Linear** | Simple weighted average | O(1) per param |
| **SLERP** | Spherical interpolation | O(1) per param |
| **Task Arithmetic** | `W_new = W_base + λ * Σ w_i * (W_i - W_base)` | O(n_models) per param |
| **TIES** | Task arithmetic + magnitude sparsification + sign consensus | O(n_models + sort) per param |
| **DARE** | Random pruning + rescaling (alternative to TIES) | O(n_models + random) per param |
| **ModelStock** | Per-parameter ensemble learning | O(n_models) per param |

**Current Limitation**: All methods use **per-tensor** execution. No batching, no kernel fusion.

### Sparsification & Consensus

**Location**: `pmetal-merge/src/sparsify.rs`, `pmetal-merge/src/consensus.rs`

**Sparsification Patterns**:
```rust
/// sparsify_by_magnitude: Keep top density fraction by absolute value
/// Breadcrumbs: Keep middle density fraction (remove outliers)

// Current: Flatten → Sort → Apply threshold → Reshape
// Performance: O(n log n) due to sorting per tensor
```

**Consensus Patterns**:
```rust
/// sign_consensus: Weighted sum of signs
/// majority_sign: Return sign of weighted sum
/// unanimous_agreement: All tensors must agree

// Pattern: Convert to signs → Weight → Sum → Mask
```

### Lazy Tensor Loading

**Location**: `pmetal-merge/src/loader.rs`

- Keep file handles open, load tensors on-demand
- Critical for merging large models on memory-constrained macOS devices
- Maintains tensor name index for quick lookups

---

## Current Distillation Implementation

### Loss Functions

**Location**: `pmetal-distill/src/losses/`

| Loss | Implementation | GPU Acceleration |
|------|-----------------|------------------|
| **KL Divergence** | Forward/Reverse KL | Metal kernel with online softmax |
| **Soft Cross-Entropy** | CE(teacher_soft, student_logits) | Metal kernel |
| **Jensen-Shannon** | JS divergence | Metal kernel |
| **MSE** | Hidden state alignment | Metal kernel |

**Key Feature**: All use **online softmax** to avoid materializing vocab-sized tensors.

### Offline Distillation

**Location**: `pmetal-distill/src/offline.rs`

**Pattern**:
- Pre-compute teacher logits → Cache to disk
- Compression options: TopK, Int8, Int4
- TopK most effective (only top 128 of 32k vocab needed)

**Benefit**: Large teacher models can be discarded, only cache used during training.

### TAID (Temporally Adaptive Interpolated Distillation)

**Location**: `pmetal-distill/src/taid.rs`

```
Standard distillation:    Student learns P_T (teacher)
TAID:                     Student learns P_I = α*P_T + (1-α)*P_S

Where α adapts based on:
  - Training progress (cosine schedule)
  - Sample difficulty (adaptive per-sample)

Early training: α ≈ 0.9 (more teacher)
Late training:  α ≈ 0.5 (more student)
```

Prevents mode collapse, enables stable transfer.

---

## Optimization Transfer Opportunities

### HIGH PRIORITY - Direct Transfers

#### 1. Fused Merge Operations
- **What**: Implement Metal kernel for task vector computation
- **Current**: Per-tensor MLX operations
- **Target**: Single Metal dispatch combining subtraction + sparsification + consensus
- **Gain**: 3-5x speedup on merge operations
- **Reference**: `pmetal-metal/src/kernels/fused_lora.rs`

#### 2. Online Softmax for Merge Sparsification
- **What**: Replace magnitude-based threshold with streaming computation
- **Current**: Sort entire tensor to find kth element
- **Target**: Single pass to compute threshold + apply mask
- **Gain**: ~2x speedup for sparsification
- **Reference**: `fused_distill.rs` online softmax pattern

#### 3. Batched Merge Tensor Processing
- **What**: Process all tensors in single Metal batch
- **Current**: Per-tensor processing in MLX
- **Target**: BatchedCommandBuffer for 100+ tensor merges
- **Gain**: 20-40% from eliminating sync overhead
- **Reference**: `fused_training.rs` BatchedCommandBuffer

#### 4. Zero-Copy Merge Tensor Access
- **What**: Avoid copying tensors between MLX and Metal
- **Current**: Load entire tensors into MLX, process
- **Target**: Use `metal_buffer_from_ptr()` directly on safetensors data
- **Gain**: 30-50% memory reduction for large models
- **Reference**: `mlx_metal_optimizer.rs`, `kl_divergence.rs`

#### 5. Kernel Auto-Tuning for Merge
- **What**: Auto-tune merge kernel parameters per hardware
- **Current**: Fixed parameters
- **Target**: Use Tuna auto-tuner for TIES kernel tile sizes
- **Gain**: 10-15% from hardware-specific optimization
- **Reference**: `pmetal-metal/src/tuna.rs`

### MEDIUM PRIORITY - Enhanced Patterns

#### 6. Distillation Loss Fusion
- **What**: Combine loss computation with gradient computation
- **Current**: Separate forward + backward passes
- **Target**: Fused loss + gradient in single Metal kernel
- **Gain**: ~40% from eliminating intermediate materialization
- **Reference**: `fused_cross_entropy.rs` pattern

#### 7. Attention Format for Merge Caches
- **What**: Store intermediate merge results in optimized format
- **Current**: Standard tensor format
- **Target**: Use attention format `[B, heads, seq, head_dim]` for structured merges
- **Gain**: ~10% from better cache locality
- **Reference**: `kv_cache.rs` format

#### 8. Grouped GEMM for Multi-Model Merges
- **What**: Process multiple model tensors in single batch
- **Current**: Sequential per-model processing
- **Target**: Group by tensor shape, batched matrix ops
- **Gain**: 1.5-1.8x for models with many similar-sized tensors
- **Reference**: `grouped_gemm_moe.rs`

#### 9. Async Scheduling for Merge
- **What**: Pipeline CPU tensor loading with GPU merge computation
- **Current**: Sequential load → merge
- **Target**: AsyncScheduler with double-buffering
- **Gain**: 30-40% throughput improvement
- **Reference**: `async_scheduler.rs`

#### 10. Sparse Tensor Format for Merge Results
- **What**: Store sparse merge results efficiently
- **Current**: Dense output
- **Target**: Sparse encoding for sparsified merges
- **Gain**: 50-80% storage reduction for sparse merges
- **Reference**: TIES sparsification pattern

### LOWER PRIORITY - Advanced Patterns

#### 11. Gradient Checkpointing for Distillation
- **What**: Reduce memory during distillation training
- **Current**: Full forward pass stored
- **Target**: Checkpoint teacher outputs, recompute as needed
- **Gain**: 30-40% memory reduction
- **Reference**: `gradient_checkpoint.rs`

#### 12. FP8 Quantization for Merge Computation
- **What**: Use lower precision during merge
- **Current**: FP32 computation
- **Target**: FP8 with dynamic scaling
- **Gain**: 2x memory reduction, potential 30-40% speedup
- **Reference**: `fp8_quantization.rs`

#### 13. Offloading for Large Merge Operations
- **What**: Offload intermediate results to disk
- **Current**: Keep all in memory
- **Target**: CPU/disk offloading for constrained devices
- **Gain**: Enable merging 405B models on 24GB devices
- **Reference**: `offloading.rs`

#### 14. JIT Compilation for Complex Merges
- **What**: Compile entire merge kernel graph
- **Current**: Individual MLX ops
- **Target**: Use MLX compile_with_state for full graph fusion
- **Gain**: 50-100% speedup, requires mlx-rs improvements
- **Reference**: `training_loop.rs` notes on JIT limitations

---

## Implementation Roadmap

### Phase 1: Quick Wins (1-2 weeks)
1. **Batched command buffer for merge tensors**
   - Lowest complexity, good ROI
   - Pattern already exists in `fused_training.rs`

2. **Online magnitude thresholding**
   - Replace sort-based threshold with streaming
   - Adapt online softmax pattern

### Phase 2: Core Fusion (2-4 weeks)
3. **FusedTIES kernel**
   - Single Metal dispatch: `subtract → sparsify → consensus → scale`
   - Highest impact for TIES/DARE merges

4. **Zero-copy tensor loading**
   - Use `metal_buffer_from_ptr()` on mmap'd safetensors
   - Major memory improvement

5. **Fused distillation loss+gradient**
   - Combine forward and backward in single kernel
   - Adapt `fused_cross_entropy.rs` pattern

### Phase 3: Advanced Optimizations (4-8 weeks)
6. **Kernel auto-tuning for merge**
   - Integrate Tuna auto-tuner
   - Hardware-specific optimization

7. **Async scheduling**
   - Pipeline tensor loading with GPU computation
   - Double-buffering pattern

8. **FP8 merge computation**
   - Lower precision for speed
   - Dynamic scaling for accuracy

---

## Code Locations & References

| Pattern | Location | Applicable To |
|---------|----------|---------------|
| Fused Operations | `pmetal-metal/src/kernels/fused_*.rs` | Merge, Distill |
| Online Softmax | `pmetal-metal/src/kernels/fused_distill.rs` | Sparsification |
| Batched Buffers | `pmetal-metal/src/kernels/fused_training.rs` | Merge |
| Zero-Copy Bridge | `pmetal-trainer/src/mlx_metal_optimizer.rs` | Merge, Distill |
| Auto-Tuning | `pmetal-metal/src/tuna.rs` | Merge kernels |
| Lazy Loading | `pmetal-merge/src/loader.rs` | Already used |
| Gradient Checkpoint | `pmetal-mlx/src/gradient_checkpoint.rs` | Distill training |
| Offloading | `pmetal-mlx/src/offloading.rs` | Large merges |
| Async Scheduling | `pmetal-mlx/src/async_scheduler.rs` | Merge pipeline |
| Grouped GEMM | `pmetal-mlx/src/grouped_gemm_moe.rs` | Multi-model merge |
| FP8 Quantization | `pmetal-mlx/src/fp8_quantization.rs` | Merge compute |

---

## Example: Batched Merge Implementation

Current per-tensor approach:
```rust
// Current: GPU-CPU sync per tensor
for (name, tensors) in model_tensors {
    let merged = merge_method.merge(&tensors, base, params)?;
    merged.eval()?;  // Sync here
    results.insert(name, merged);
}
```

Target batched approach:
```rust
// Target: Single sync for all tensors
let ctx = MetalContext::new()?;
let mut batch = BatchedCommandBuffer::new(ctx)?;

for (name, tensors) in model_tensors {
    merge_method.queue_merge(&mut batch, &tensors, base, params)?;
}

batch.execute()?;  // Single sync
```

---

## Example: Fused TIES Kernel

Current MLX operations:
```rust
// Current: Multiple MLX ops, multiple intermediate tensors
let task_vector = tensor.subtract(&base)?;           // Intermediate 1
let abs_values = task_vector.abs()?;                  // Intermediate 2
let threshold = compute_threshold(&abs_values, density)?;  // O(n log n) sort
let mask = abs_values.greater(&threshold)?;          // Intermediate 3
let sparse = task_vector.multiply(&mask)?;           // Intermediate 4
let signs = sparse.sign()?;                          // Intermediate 5
let consensus = sign_consensus(&signs, weights)?;    // Intermediate 6
let result = sparse.multiply(&consensus)?;           // Final
```

Target fused kernel:
```metal
// Target: Single Metal kernel, no intermediates
kernel void fused_ties(
    device const float* tensor,
    device const float* base,
    device const float* weights,
    device float* output,
    constant TiesParams& params
) {
    // All operations fused in single dispatch
    float task_value = tensor[idx] - base[idx];
    float abs_value = abs(task_value);

    // Online threshold (pre-computed or streaming)
    bool keep = abs_value > params.threshold;

    // Fused consensus
    float sign = task_value > 0 ? 1.0 : -1.0;
    float weighted_sign = sign * weights[model_idx];
    // Atomic or reduction for consensus...

    output[idx] = keep ? (task_value * consensus) : 0.0;
}
```

---

*Last updated: January 2025*
*Based on analysis of pmetal codebase*
