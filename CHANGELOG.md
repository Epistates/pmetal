# Changelog

All notable changes to PMetal will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-03-06

### Added

- **Apple Neural Engine (ANE) integration** behind `ane` feature flag — MIL 1.3 program generation, private API FFI via dlopen, IOSurface zero-copy, compilation budget tracking, hybrid CPU/ANE trainer with async gradient accumulation
- **`AneInferenceEngine`** — forward-only ANE kernels (no concat taps, ~6x smaller IO vs training) with CPU-side embedding, RMSNorm, sampling (greedy/temperature/top-k), and autoregressive generation via Easy API `.device(Device::Ane)`
- **KV cache for autoregressive generation** — hybrid ANE prefill + CPU decode architecture eliminates O(n²×L) recomputation per token; ANE processes the full prompt, CPU handles single-token decode steps with cached KV pairs via `cblas_sgemv`
- **GQA/MQA support** — `n_kv_heads` config field enables grouped-query attention (Llama 3, Mistral, etc.); concat-based KV head expansion in ANE kernels
- **SafeTensors weight loading** — direct loading of HuggingFace safetensors format (single and multi-file) with automatic bf16/f16/f32 dtype conversion
- **LoRA adapter fusion** — merge adapter weights (`W += (alpha/rank) * B @ A`) before ANE kernel compilation; supports both `self_attn` and `mlp` target modules
- **Dynamic weight pipeline**: 9 MIL kernels compiled once at startup; weights packed alongside activations in IOSurface spatial dimension — zero recompilation during training
- **`DynamicAneTrainer`**: compile-once training loop replacing the static trainer that consumed ~76% of training time in recompilation
- **`DynamicKernelConfig`** and 12 dynamic kernel generators in `dynamic_kernel.rs`
- **MIL program fragment helpers**: `emit_rmsnorm_fuse` and `emit_dyn_matmul_with_act` for composable RMSNorm fusion and dynamic matmul in ANE kernel generation
- **`rmsnorm_fwd` dynamic kernel**: Fused RMSNorm forward pass on ANE
- **fp32 IOSurface support**: `IoSurface::new_f32()` with packed write/read for dynamic weight pipeline
- **MIL builder extensions**: `emit_cast`, `emit_slice_by_size`, `new_fp32_input` for dynamic kernel generation
- **Non-standard `head_dim` support**: Full forward and backward kernel support for models where `head_dim != dim/n_heads` (e.g., Qwen3 with `head_dim=128`, `dim/n_heads=64`)
- **Training dashboard (TUI)**: `pmetal dashboard` subcommand using ratatui for real-time loss curves, timing breakdown, and throughput monitoring
- **`MetricsJsonCallback`**: Emits full `StepMetrics` including ANE timing, Adam timing, and throughput to JSONL
- **GSPO trainer**: Group Sequence Policy Optimization (fixes GRPO length bias)
- **DAPO trainer**: Decoupled Clip and Dynamic Sampling Policy Optimization (all 4 ByteDance innovations)
- **Python bindings** (`pmetal-py`) via PyO3/maturin with type stubs
- **High-level Easy API** (`pmetal::easy`) — builder pattern for fine-tuning and inference
- **Version and device introspection** (`pmetal::version`)
- **Examples**: `device_info`, `finetune_easy`, `finetune_manual`, `inference_easy`
- **Python CI workflow** (`.github/workflows/python.yml`)
- `Device::Ane` variant with feature-gated support
- ANE-specific error types in `pmetal-core` and `pmetal-metal`
- ANE training loop integration in `pmetal-trainer`
- `silu_inplace` in Accelerate wrappers for CPU decode SwiGLU

### Fixed

- **Metal resource exhaustion on long training runs**: `eval_training_state()` now evaluates model params and optimizer states (momentum, velocity) alongside losses, preventing unbounded computation graph growth in deferred-eval mode
- **Gradient checkpointing default**: `CheckpointStrategy` default changed from `Smart` to `None` — MLX backend does not implement it yet; configs remain forward-compatible
- **Training defaults**: batch_size default 4→1, gradient_accumulation_steps default 1→4 (same effective batch size, lower per-step memory pressure)
- ANE inference gibberish output: added RoPE and per-head QK-norm to prefill kernel and CPU decode
- ANE inference missing `compile_kernels()` call in `generate_cached_ane`
- All backward kernels (static + dynamic) now use `q_dim()`/`kv_dim()` instead of hardcoded `dim` — fixes incorrect gradient shapes for non-standard architectures
- `sdpa_bwd1_input_ch`: `4*dim` → `q_dim + 2*kv_dim + dim`
- `sdpa_bwd1_output_ch`: `dim + 2*score_ch` → `kv_dim + 2*score_ch`
- `sdpa_bwd2_input_ch`: `2*score_ch + 2*dim` → `2*score_ch + q_dim + kv_dim`
- Dynamic backward kernels: `wo_bwd`, `sdpa_bwd1`, `sdpa_bwd2`, `qkv_bwd` all updated for q_dim/kv_dim
- SafeTensors dtype/alignment error handling
- Token ID bounds check in CPU decode
- Softmax numerical stability for zero-sum edge case
- ANE GQA inference failure (`status=0x1d`): replaced unreliable `tile` MIL op with concat-based KV head expansion in all 3 SDPA kernels
- Token ID truncation: `embed_lookup`/`embed_backward` changed from `u16` to `u32` (Qwen3 vocab=151936 exceeds u16 max)
- RMSNorm epsilon hardcoded to 1e-5: now configurable via `cfg.rms_norm_eps` (Qwen3 requires 1e-6)
- CI: Exclude `pmetal-py` from CI clippy/build/test (requires Python dev libs not available on runner)

### Improved

- NEON f16↔f32 conversion upgraded from 4-wide to 8-wide (`fcvtn2`/`fcvtl2`)
- Accelerate/vDSP wrappers expanded with 12 new functions: `rmsnorm`, `rmsnorm_backward`, `cross_entropy_loss`, `softmax_inplace`, `adam_update`, `embed_lookup`, `embed_backward`, `matrix_transpose`, `gemm`, `vadd`, `vmul` (with scalar fallbacks on non-macOS)
- `supports_neural_engine()` now performs real ANE detection via framework dlopen
- Easy API ANE path now auto-detects SafeTensors/flat weights, LoRA adapters, and GQA config; uses `generate_cached()` for KV-cached inference
- ANE config validation (`new()` returns `Result`)
- Kernel config validation (`TransformerKernelConfig::validate()`)
- LoRA safety: rank=0 guard and tensor shape validation
- Decode memory efficiency: pooled scores buffer
- 15 new tests for non-standard head_dim kernels (7 static + 8 dynamic)
- MIL debug dump on ANE compile failure (`/tmp/ane_debug_layer{N}_{attn|ffn}.mil`)
- Qwen3 GQA kernel test (n_heads=16, n_kv_heads=8, verifies no `tile` ops)
- Dynamic kernel documentation: All 12 kernels now document detailed input tensor names alongside dimension formulas

## [0.1.2] - 2026-03-02

### Fixed

- **GPU occupancy waste in gradient scaling**: `scale_gradients` grid dispatch was 4x over-provisioned after float4/half4 vectorization — each thread processes 4 elements but the grid still dispatched one thread per element; corrected from `div_ceil(32)` to `div_ceil(128)`
- **Threadgroup memory overallocation in fused LoRA**: Static `threadgroup float[128 * 256]` arrays in `fused_lora_forward` and `fused_lora_backward_x` allocated 128KB each, exceeding Apple Silicon's 32KB threadgroup memory limit; switched to dynamic threadgroup memory via `setThreadgroupMemoryLength` with host-side size calculation based on actual tile and rank dimensions
- **Silent loss of final async checkpoint**: When `TrainingLoop` was dropped, the pending background checkpoint thread was silently detached — if the process exited before the thread finished, the final safetensors file could be truncated or corrupt; added `Drop` impl that joins the pending handle and logs errors
- **LoRA rank validation**: Raised rank limit from 64 to 256 to match `MAX_LORA_RANK` now that dynamic threadgroup memory removes the static allocation constraint

### Improved

- **Checkpoint I/O deduplication**: Extracted shared file write logic (`write_checkpoint_to_dir`) from `save_checkpoint`, `save_checkpoint_owned`, and `save_best_checkpoint` — eliminated ~100 lines of duplicated directory creation, safetensors serialization, and metadata JSON writes
- **Edge case test coverage**: Added tests for NEON fp16↔fp32 conversion (NaN, Inf, -Inf, -0.0, subnormals, exact 4-element alignment, 1M+ element arrays) and Accelerate vDSP wrappers (negative values, single-element arrays, 1M+ element arrays)

## [0.1.1] - 2026-02-27

### Improved

- **Unified chat template detection**: New `detect_chat_template()` inspects `tokenizer_config.json` Jinja strings before falling back to model-name heuristics — training, inference, and distillation now detect templates consistently
- **Broader inference templates**: Added inference formatters for Llama-2, Gemma, Mistral, Phi-3, Phi-4, and GPT-OSS (previously only ChatML and Llama-3 were supported)
- **Template-aware stop tokens**: Inference now encodes the correct EOS token per template type (`<|eot_id|>` for Llama-3, `<end_of_turn>` for Gemma, etc.) instead of hardcoding `<|im_end|>`
- **Array chat_template support**: Handles HuggingFace models that store `chat_template` as an array of `{name, template}` objects (e.g., Command-R)
- **Distillation template detection**: Distillation now applies the student model's chat template during dataset formatting (was `None` before)
- **Distillation completion output**: Summary box with detected template and actionable next-steps command

### Fixed

- **Silent download failures**: Tokenizer and config file download errors now logged with `warn!`/`debug!` instead of silently swallowed with `let _ =`
- **Silent quantize fallback**: Invalid `--method` values now produce a clear error listing valid methods instead of silently falling back to Q4K
- **Dataset directory error**: Passing a directory to `--dataset` now auto-discovers `train.jsonl`/`data.jsonl`/`dataset.jsonl` or suggests `.jsonl` files found, instead of an opaque "Is a directory" error
- **Tokenizer-not-found guidance**: Error now explains that GGUF models don't bundle tokenizers and suggests `pmetal download <model-id>`
- **Memory stats NaN**: `pmetal memory` guards against division by zero when `total_gb()` is 0
- **EOS token stripping**: `extract_final_response` now strips all known EOS tokens (was only `<|im_end|>` and `<|endoftext|>`)
- **Qwen3 LoRA gradient checkpointing warning**: Now emitted once per run instead of per-layer per-step (via `std::sync::Once`)

## [0.1.0] - 2026-02-26

Initial public release.

### Core Framework

- **pmetal-core**: Foundation types, configuration system, and shared traits for the workspace
- **pmetal-cli**: Command-line interface with `train`, `infer`, and `bench` subcommands

### Model Support

- **pmetal-models**: Dynamic architecture loading with support for:
  - Llama (2, 3, 3.1, 3.2, 3.3, 4)
  - Qwen (2, 2.5, 3, 3-MoE)
  - DeepSeek (V3, V3.2, V3.2-Speciale)
  - Mistral (7B, 8x7B)
  - Gemma (2, 3), Phi (3, 4), Granite (3.0, 3.1), Cohere (Command R), GPT-OSS, Nemotron-H
  - Vision: Pixtral 12B, Qwen2-VL, MLlama 3.2-Vision

### Training

- **pmetal-trainer**: SFT, DPO, and GRPO training loops with learning rate schedulers and gradient checkpointing
- **pmetal-lora**: LoRA and QLoRA with configurable rank, alpha, and target modules
- **pmetal-data**: Dataset loading for ShareGPT, Alpaca, Messages, and raw text formats with sequence packing (99.7% efficiency)
- **pmetal-distill**: Knowledge distillation with KL divergence, Jensen-Shannon, soft cross-entropy, hidden state alignment, and offline logit caching

### GPU Acceleration

- **pmetal-metal**: Custom Metal compute kernels:
  - FlashAttention with O(n) memory
  - Fused LoRA forward pass
  - Fused cross-entropy (Unsloth-style chunked loss)
  - Fused RoPE
  - Fused sampler with JIT compilation
  - Fused DoRA kernels

### Model Operations

- **pmetal-merge**: Model merging via Linear, SLERP, TIES, DARE, DELLA, NearSwap, and Model Stock methods
- **pmetal-gguf**: GGUF format reading, writing, dequantization, and imatrix quantization
- **pmetal-hub**: HuggingFace Hub downloading, caching, and upload support

### Experimental

- **pmetal-mhc**: Manifold-Constrained Hyper-Connections (Sinkhorn-Knopp doubly stochastic projections) with Metal GPU acceleration
- **pmetal-distributed**: Peer-to-peer distributed training with mDNS auto-discovery, ring all-reduce, and gradient compression
- **pmetal-vocoder**: BigVGAN neural vocoder for text-to-speech synthesis
- **pmetal-mlx**: MLX backend integration with KV cache management, quantization, speculative decoding, and NEFTune

### Infrastructure

- Rust edition 2024, minimum supported Rust version 1.85
- Continuous fuzzing for GGUF reader via `cargo-fuzz`
- CI with clippy, fmt, test, and fuzz workflows
- Dual licensed under MIT and Apache-2.0
