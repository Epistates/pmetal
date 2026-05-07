# Changelog

All notable changes to PMetal will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] - 2026-05-07

### Added

#### Distributed inference & training

- **`pmetal-distributed` crate (Phases 1-4 + 7)**: Thunderbolt-fabric-aware multi-Mac cluster runtime with feature-gated tensor, expert, context, ZeRO, and pipeline parallelism modules
  - `pmetal cluster` CLI: per-node launch, ring/mesh topology discovery, fabric handshake
  - Pipeline harness with overlap of computation and Thunderbolt transfers
  - Canonical expert-rank mapping + per-architecture MoE/MLA tensor-parallel plans
  - Ring all-reduce / all-gather with corrected chunk indexing

#### TurboQuant KV cache (production-ready)

- **TurboQuant KV cache quantization**: Provably near-optimal KV cache compression based on random rotation + Lloyd-Max scalar quantization + QJL residual for unbiased inner products (arXiv:2504.19874). Achieves 4-6x KV cache compression with near-zero quality loss. Available via `--kv-turboquant` or presets `--kv-turboquant-preset q3_5` (near-lossless) / `q2_5` (6.4x compression)
  - Separate key/value runtimes with independent bit widths and outlier-aware mixed-precision
  - Direct attention path for single-token decode avoids full cache dequantization
  - Data-oblivious (no calibration data required) — quantizes KV entries online as generated
  - Precomputed codebooks via Lloyd-Max algorithm for Beta distribution (deterministic from seed)
  - Metal kernel backend with CPU fallback
  - **Phase 0**: split monolithic `mod.rs` (6101 → 222 LOC) into config/core/state/bits/math submodules
  - **Phase 3**: GPU-resident hot/cold pipeline + Mixed K/V storage; `mixed_score` as layout oracle
  - **Phase A/B**: QJL ablation harness (feature-gated) + per-row `key_slot_scale` codebook adaptation
  - **Phase C/C′**: Variant F drop-QJL opt-in path; d128/d256 `no_qjl_2pass` fast paths (4..=8 bits)
  - **Phase D**: `TurboQuantPackMode` config + Fullbyte dense-values kernel
  - **Phase E**: `TurboQuantOutlierMode` — encode-side top-K outlier storage, zero pre-quant + decode override, outlier-bias on d128/d256 fullbyte score kernel; CPU mirror in scalar encode/decode
  - **Phase F**: Hamming skip-list dispatch — `skiplist_threshold` config, GPU `sign_hash` buffer, Metal Hamming-distances kernel + FFI, GQA support
  - Mixed-precision attention parity baseline; defensive residual-norm clamp + NaN-safe encode

- **Asymmetric K/V head dimensions**: KV cache, TurboQuant, and fused attention now support models where key and value projections have different widths (e.g. DeepSeek MLA with `qk_head_dim != v_head_dim`)

- **`pmetal serve --kv-turboquant`**: TurboQuant KV cache in the serving engine with `--kv-turboquant-preset q3_5` for near-lossless 4.6x KV compression in production

#### Quantization & model formats

- **Optimized FP8 checkpoint loading**: Hugging Face FP8 `weight_scale_inv` sidecars are dequantized or repacked into MLX `mxfp8` weights for Qwen3-family native paths; mode-aware quantized matmul plumbing handles floating-point quantized weights without dense fallback
- **Expanded GGUF quantization/export**: `pmetal quantize` now writes standard GGUF metadata from Hugging Face configs, tokenizer/pre-tokenizer metadata, HF-to-GGUF tensor names, stacked MoE expert tensors, and method-specific file types
- **Broader GGUF format coverage**: quantization/dequantization support now includes K-quants, legacy Q4/Q5/Q8 variants, Q1_0, TQ1_0/TQ2_0, MXFP4, NVFP4, BF16, F16, and F32 round trips
- **MLX safetensors quantization path**: quality-based bit allocation with `--target-bpw`, GPU-resident weight loading, and tokenizer/config sidecar copy for MLX-format quantized exports

#### Inference server (OpenAI- + Anthropic-compatible)

- **Continuous batching with paged-KV-style admission + shared prefix cache** in `pmetal-serve`: per-request slot scheduling, KV-cache prefix sharing, concurrent decode for many simultaneous chats
  - Token-block admission budget (`--cb-block-size`, `--cb-max-blocks`) prevents over-admitting active contexts and skips head-of-line requests when a smaller queued request fits the remaining block budget
  - Continuous batching now reuses the shared prompt prefix cache, prefills only uncached suffix tokens, and saves extended prefixes after final prefill
  - Continuous batching derives the same cache mode as the single-request serving path, honoring `--kv-quant` and `--kv-turboquant`
  - Hybrid/recurrent models are rejected from continuous batching instead of silently running without recurrent state
- **Anthropic-compatible `/v1/messages` endpoint**: streaming `message_start` → `content_block_start` → `content_block_delta*` → `content_block_stop` → `message_delta` → `message_stop` events; non-streaming JSON path
- **`/v1/embeddings` endpoint**: 17 architectures supported via `forward_hidden` (Llama/Llama4/Qwen2/Qwen3/Qwen3MoE/Qwen3Next/Mistral/Gemma/Gemma4/Phi/Phi4/DeepSeek/Cohere/Granite/GptOss/NemotronH/BERT) — pooling via `pmetal_models::pooling`
- **Token logprobs**: `SamplingParams.logprobs_top_n` plumbed end-to-end through non-streaming and SSE streaming on both `/v1/chat/completions` and `/v1/completions`. New `pmetal_models::generation::token_logprobs` primitive; ANE/CPU paths emit `logprob: None`
- **Best-effort tool calling on `/v1/chat/completions`**: `try_parse_tool_calls` accepts `{name, arguments}` or `{tool_calls: [...]}`. `ChatCompletionRequest.tools` gates the attempt; chat templating threads tool defs into the rendered prompt
- **`IncrementalDecoder<Aux>` SSE buffer**: shared UTF-8 boundary buffer + per-token aux pipelining (used for logprobs alignment) across chat/completions/anthropic streams

#### Job orchestration substrate (TUI / GUI / MCP / CLI parity)

- **`JobSpec` substrate**: 16 canonical spec types in `pmetal-core` (Train, Distill, GRPO, Bench, Eval, Pretrain, Tokenize, Serve, Generate, RLKD, EmbedTrain, DFlash, Memory, Ollama, …) with `#[derive(JobSpec)]` proc-macro
- **`JobEvent` canonical streaming protocol**: progress / metric / log / artefact / complete / failed events emitted by all 4 surfaces (CLI, TUI, GUI, MCP)
- **CLI**: 8 specced `Commands` variants flattened — 613 LOC removed from `main.rs`; `cli/<sub>.rs` Args structs and JobSpec argv round-trip tests; `--log-events` flag stub
- **TUI**: 14 tabs with full CLI parity, `?`-key help overlay, `Ctrl+1..9` tab jump, active-job footer badge, descriptor-driven forms with shared `FormTabState` primitive; channel-based metrics streaming (`ChannelMetricsCallback`) for direct-path train/distill/grpo/bench/eval/pretrain
- **GUI (Tauri)**: complete 9-DTO frontend-lockstep migration to `*Spec` types; Serve, Bench, Eval, Jobs, Pretrain pages; embed-train + rlkd + ollama routes; channel-based metrics streaming
- **MCP**: 9 tools migrated + 11 generate flags + tokenize/memory/dflash; `train` tool migrated with JobEvent JSONL consumer

#### SOTA distillation (`pmetal-distill`)

- **Universal Logit Distillation (ULD)** — Wasserstein-1 over sorted logit distributions for cross-tokenizer KD (Boizard et al. 2024); optional `top_k` truncation; permutation-invariant by design
- **Generalized Knowledge Distillation (GKD)** — λ-weighted off-policy + on-policy KL blend (Agarwal et al. 2024); `OnPolicySampler` trait with `GreedySampler` reference impl; `compute_full(t_off, s_off, t_on, s_on, T)`
- **MiniLLM** — reverse-KL with optional teacher-mix `target = mix·T + (1-mix)·S` (Gu et al. 2024)
- **Skewed JSD (DistiLLM-2)** — `α·KL(T||M_α) + (1-α)·KL(S||M_α)` with `M_α = α·T + (1-α)·S`, log-sum-exp computation; α=0.5 reduces to standard symmetric JSD (Ko et al. 2024)
- **Attention-transfer loss + weighted Metal path** for hidden-state distillation
- **Offline teacher-logit caching**: `pmetal distill --offline-cache <path>` precomputes teacher logits to disk; new `Int8PerToken` compressed-block variant replaces NaN-sentinel scheme with explicit `per_token_meta` field (legacy `Int8` variant retained for read-back)
- **`DistillLossOutput.metrics: HashMap<&'static str, f32>`**: lazily-evaluated `teacher_entropy`, `student_entropy`, `kl_per_token`, `top1_agreement` exposed to trainer JSONL/TUI streaming
- **TAID difficulty-aware** observability: `alpha_var` surfaced for per-step monitoring
- **Configurable `ignore_index`**: PyTorch-standard `-100` default on `TrainingConfig`; safe label clamping before gather
- **Hidden-state shape assertions** before matmul (clear error vs. silent broadcast bug)

#### SOTA model merging (`pmetal-merge`)

- **Fisher merging** (Matena & Raffel 2022): diagonal-Fisher-weighted average `θ = Σ F_i⊙θ_i / (Σ F_i + ε)`; lazy-loaded Fisher safetensors; `fallback_to_mean` for tensors without Fisher entries
- **RegMean** (Jin et al. 2023): closed-form linear-layer merge `W = (Σ G_i)⁻¹ · (Σ G_i W_i)` via hand-rolled Gauss-Jordan `pseudo_inverse_2d` with Tikhonov ridge; falls back to mean for non-2D weights
- **MoE expert permutation alignment**: per-(model, layer) Hungarian solver (Jonker-Volgenant style, O(N³)) over L2-normalized cosine similarity of expert fingerprints; tensor-name remapping `experts.{i}.` → `experts.{π(i)}.` before merge; gated by `align_moe_experts`
- **Honor `config.dtype` in save path**: `MergeBuilder.dtype` builder, `TensorWriter::with_dtype` plumbing, per-dtype byte packing for F16/BF16/F32; previously hardcoded to F16
- **Cross-model dtype consistency check**: `verify_source_dtypes` errors on mismatch unless `allow_mixed_dtype` is set
- **Tied-embedding detection**: `lm_head.weight` and `embed_tokens.weight` aliasing detected and merged once under canonical name
- **Tokenizer + config sidecar copy**: `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `config.json`, `generation_config.json` copied on full-model merge; `config.json.torch_dtype` patched to match output dtype
- **Post-merge sanity sweep** (`SanityLevel::{Off,Quick,Full}`, default `Quick`): NaN/inf detection aborts save; full mode reports per-tensor `mean/std/abs_max/sparsity`
- **`MergeConfig.dry_run`**: short-circuits write phase, logs would-write summary

#### LoRA / QLoRA — full text-architecture coverage

- New LoRA adapters: **Granite, Llama4, DeepSeek, NemotronH, MLlama, Cohere, Phi, Gemma4, GPT-OSS, Qwen3-MoE, Qwen3-Next**
- New QLoRA adapters: **Granite, Llama4, DeepSeek, NemotronH, Cohere, Phi, Gemma4, GPT-OSS, Qwen3-MoE, Qwen3-Next**
- **LoRA+** wired into `run_compiled` training path; Gemma4 QLoRA KV-cache path
- **DeepSeek `merge_lora`** properly implemented; Phi4 dispatched to existing `PhiLoraForCausalLM`
- Interface-parity gradient-checkpointing hooks across 7 adapters

#### Bridge & native paths

- **`pmetal-bridge` crate**: Zero-allocation MLX C++ bridge replacing mlx-rs as the core runtime. Native inference at 201 tok/s (Qwen3.5 0.8B), 4-bit quantized inference (28 tok/s on 27B), compiled attention, KV cache trimming, and full training ops (autograd, optimizer, random, math, reduction, comparison) — all without mlx-rs overhead
- **Fused [T=1] decode kernels** for `gpt_oss` and `llama4` (Bridge Phase 4)
- **Fused [N,1] batched-decode** path for Tier-1/2 architectures
- **Cheap native KV cache fork support** for Qwen3, GPT-OSS, Llama4, DeepSeek, and generic `KVCache`, preserving dense, quantized, and TurboQuant cache state for serving prefix reuse
- **`BRIDGE_TRY_{DST,VOID}` error coverage**: thread-local exception slot replaces process-abort across most ops; `pmetal_bridge::check_last_error()?` surfaces `BridgeError::CxxException` after any op; `InlineArray::try_*` variants for matmul/softmax/reshape/sdpa/gather_mm/dequantize/etc.
- **Scalar dtype footgun fix**: `InlineArray::scalar_like(value, peer)` + `mul_scalar`/`add_scalar`/`sub_scalar`/`div_scalar` eliminate manual `.as_dtype(model_dtype)` calls
- **`async_eval` actually async**: prior implementation blocked the calling thread
- Bridge file splits: `bridge.h` and `bridge.cpp` carved into `cpp/bridge/` sub-headers + 6 source files; `bridge_turboquant.cpp` split by kernel family; `inline_array.rs`, `qwen3_native.rs`, `deepseek_native.rs`, `llama4_native.rs`, `gpt_oss_native.rs` split into submodule directories
- **`forward_hidden`** for 17 architectures (embeddings + retrieval support)

#### Preference & RL trainers (`pmetal-trainer`)

- **`PairedPreferenceTrainer<L>` trait + `DpoLoss` kernel**: `DpoTrainer::train` and `OnlineDpoTrainer::train_step` now delegate to the shared trainer; `ReferenceStrategy::{StopGradient, Zero, Precomputed}` covers the three reference-logp sources
- **Shared log-prob helpers** fanned out to KTO/ORPO/GRPO/OnlineDPO via `logprob_utils::{compute_log_probs, compute_log_probs_with_avg, shifted_selective_log_softmax}`

#### Model, training & data

- **Full-parameter pretraining**: End-to-end `pmetal pretrain` pipeline for training models from scratch
  - Model factory supporting llama, qwen, gemma, mistral, phi, and gpt-oss architectures
  - Gradient accumulation, cosine/linear/constant LR scheduling, gradient clipping
  - Full model + optimizer checkpoint save/restore for resumable runs
  - Memory-mapped streaming shard reader (`StreamingShardReader`) with zero-copy I/O via memmap2
  - `pmetal tokenize` command for converting JSONL corpora to binary shards
  - Pretrain tab in TUI and GUI with real-time loss/throughput/ETA monitoring

- **Gradient checkpointing**: `checkpoint_apply()` wraps forward functions via `mlx::core::checkpoint()` to recompute activations during backward, reducing peak training memory from O(layers) to O(1)

- **Optimizer checkpoint/resume**: `AdamW` gains `step_count()`, `set_lr()`, and `restore_state()` for saving and restoring optimizer state across training runs

- **Gemma 4 architecture**: Full Gemma 4 model support with sliding-window attention and per-layer KV head configuration

- **DFlash speculative decoding**: Native Rust port of the dflash-mlx speculative decoding pipeline for accelerated generation

- **Jinja chat templates**: Real upstream jinja rendering via minijinja with 16 parity-audited template types (ChatML, Llama2/3/4, Mistral, Gemma/Gemma4, Phi3/4, Qwen, DeepSeek, Cohere, Alpaca, Vicuna, Zephyr, GptOss)

- **Qwen3.5 MoE dispatch improvements**: Expert prefetch reset per generation, configurable GDN chunk size, chunked prefill, and generation helpers

- **Adaptive sequence packing**: `compute_pack_seq_len()` uses p99 of actual dataset sequence lengths instead of `max_position_embeddings` — up to 256x reduction in wasted compute for short-sequence datasets. `--pack-max-seq-len` for explicit override

#### Metal 4 / MPP backend

- **Metal 4 / MPP kernel backend** (Epistates/pmetal#14): Trait-based kernel dispatch with `Metal3Backend` and `Metal4Backend` for M5+ (Apple10/NAX) GPUs
  - `KernelBackend` trait with 16 methods covering GEMM, attention, fused linear, training, MoE, distillation
  - `KernelDispatch` router on `MetalContext` — selects Metal 4 for large GEMMs (M>1, K%32==0) on M5, Metal 3 for everything else
  - `Metal4CommandBuffer` with correct begin/end lifecycle, `CommandAllocatorPool`, `ResidencyManager`
  - Compile-time `#[cfg(has_metal4)]` gating + runtime `has_nax` check — zero overhead on M1-M4

- **15 MPP-optimized Metal 4 shaders**: All following Apple MPP best practices (single simdgroup execution, Morton-order threadgroup walk, K-dimension alignment to 32, accumulation-loop barriers at BK=128)
  - 8 existing shaders optimized: `mpp_gemm`, `mpp_flash_attention`, `mpp_quantized`, `mpp_fused_swiglu`, `mpp_fused_norm_lora`, `mpp_dw_gemm`, `mpp_grouped_gemm`, `mpp_fused_lora`
  - 5 new shaders: `mpp_fused_training` (AdamW), `mpp_fused_cross_entropy`, `mpp_fused_rope`, `mpp_fused_moe`, `mpp_fused_distill`
  - 2 additional: `mpp_fused_mlp` (gate+up+down combined), quantized MoE expert variants

#### Benchmarking & inference UX

- **Benchmark enhancements**: workload presets, custom dataset/expert-dir controls, inference session repeats, train sample/step/batch/sequence controls, warmup passes, GDN prefill stage profiling, TurboQuant flag for bench commands, and fused gate/up expert packing with auto-detected tensor layout

- **`--mode` sampling presets**: Per-model-family recommended sampling parameters (Qwen3/3.5 thinking/instruct modes). `--mode auto` selects based on `--no-thinking` flag

- **`--detect-repetition`**: Opt-in n-gram repetition loop detection (8-token pattern x 4 repeats), force-stops infinite loops

- **Chip name in decode stats**: Inference output now shows the Apple Silicon chip (e.g., `[M4 Max]`)

### Changed

- **mlx-rs removed**: All crates migrated from mlx-rs to the `pmetal-bridge` compat API. Entire model, training, serving, and GUI stacks now use the zero-allocation C++ bridge
- Thinking trace shown by default for thinking models; use `--hide-thinking` to suppress
- Migrated scattered `has_nax()` checks to `MetalContext::dispatch()` for centralized backend routing
- Split `compat.rs` (3620 lines) into 7-file `compat/` module directory
- Split `bridge.cpp` (6749 lines) into 6 C++ source files with shared `bridge_internal.h`
- `Array::id()` replaces `data_ptr()` for weight change detection — safe with lazy evaluation
- **Persistent MLX build cache** in CI and `build.rs`: skips redundant cmake compilation across CI runs
- **`pmetal_hub::resolve_model_path`** adopted across `core`, `cli`, and `gui` for consistent local-cache → Hub-ID resolution
- **Distillation orchestration stubs removed**: `Distiller::run_online`/`run_offline`/`run_progressive` deleted — orchestration now lives entirely in `pmetal-trainer`
- **MLX MoE routing audit**: confirmed no `argpartition(-scores, -k)` anti-top-k regressions remain; documented as a permanent footgun
- **`MergeMethod` trait** extended with `merge_named(name, …)` (default forwards to `merge`); Fisher and RegMean dispatch through name-aware path

### Fixed

- **AdamW bias correction**: step counter was advancing per-parameter instead of per-step
- **Cross-entropy loss masking**: ignored labels are masked before gather, using a selective `logsumexp - target_logit` path that avoids materializing full `log_softmax`
- **Gradient clipping** in compiled training path now uses `_clipped` step variants
- **FFI exception safety**: ~33 C++ bridge functions wrapped in try/catch
- **LoRA inference segfault**: put_along_axis crash during generation
- **UTF-8 char boundary panics** in inference/GUI output stream handling
- **Distributed ring reduce**: all-gather chunk indexing used wrong offset, corrupting gradient aggregation
- **Distributed transport/compression audit**: namespace PSK handshakes, TCP fallback hardening, bounded compressed-gradient deserialization, and out-of-range sparse-index guards
- **Serving parameter validation**: OpenAI-compatible routes validate sampling parameters before streaming and non-streaming generation dispatch
- **select_axis parameter order**: standardized (data, index, axis) across all call sites
- **Lazy array segfaults**: diffusion sampler now evals sigmas/timesteps before slice access
- **Sampling penalties**: correctly wired through native bridge decode path
- **Qwen3-Next MoE routing**: corrected anti-top-k expert selection bug (sign/slice pair)
- **Qwen3-Next hybrid cache flag** in LoRA + distillation paths
- **TurboQuant d128 pass-2** cross-simdgroup reduction; lazy-transpose footgun in mixed-precision attention
- **TurboQuant serving prefix-cache compatibility**: prefix cache now stores forked KV caches instead of dense snapshots, preserving compressed TurboQuant history without fp16 re-inflation
- **Native attention correctness/perf audit**: fail-fast TurboQuant dispatch, checked SDPA wrappers, true hot-ring cache behavior, centralized quantized tuple growth, and unsupported-cache rejection in tree verification
- **GKD `compute_weighted`** no longer scales by `(1-λ)` (silently zeroed training at λ=1.0)
- **`pmetal_serve::sse::IncrementalDecoder`** prevents UTF-8 boundary panics on partial codepoint emission
- Zero clippy warnings across entire workspace
- Stale `pmetal-mlx-sys` metallib path in release workflow (renamed to `pmetal-bridge`)

### Removed

- **mlx-rs dependency**: Fully replaced by `pmetal-bridge` — removes ~15K lines of Rust FFI bindings
- 1065 lines of dead code: `qwen3_train.rs`, unused LoRA functions in `qwen3_native.rs`
- 5 superseded LoRA training modules
- **Dropped model support for StarCoder2, FalconH1, RecurrentGemma, and Jamba**
- **`Distiller::run_*` orchestration stubs** (now lives in `pmetal-trainer`)

## [0.4.0] - 2026-03-23

### Added

- **`pmetal-mcp` crate**: Full MCP (Model Context Protocol) server exposing 45 tools for Claude Desktop and other MCP clients. Covers all pmetal functionality — training, inference, distillation, GRPO, RLKD, quantization, model merging, dataset operations, evaluation, benchmarking, model search, and Ollama export
  - **Device & models**: `device_info`, `search_models`, `download_model`, `list_local_models`, `model_fit`, `model_info`
  - **Inference**: `generate` (blocking), `chat` (via running serve instance), `start_serve`, `benchmark`, `bench_train`, `bench_gen`, `bench_corpus`
  - **Training**: `train`, `distill`, `grpo`, `rlkd`, `embed_train` — all as background jobs with full parameter coverage matching the CLI
  - **Runtime training control**: `job_set_lr`, `job_reduce_lr`, `job_reset_lr`, `job_save_checkpoint`, `job_graceful_stop` — LLM-driven adaptive training via the control file protocol
  - **Job management**: `list_jobs`, `job_status`, `job_logs`, `stop_job`
  - **Dataset ops**: `dataset_analyze`, `dataset_preview`, `dataset_validate`, `dataset_download`, `dataset_convert`, `dataset_filter`, `dataset_split`, `dataset_merge`, `dataset_sample`, `dataset_template`, `dataset_prepare`
  - **Quantization & conversion**: `quantize`, `fuse_lora`, `merge_models`, `pack_experts`, `ollama_create`, `ollama_modelfile`
  - **Evaluation**: `eval_perplexity`
  - All tools include rich `#[description]` annotations for parameter documentation in the MCP schema
  - Standalone binary (`pmetal-mcp`) for Claude Desktop + `pmetal mcp` subcommand (behind `mcp` feature flag)
  - Uses `turbomcp` v3.0.7 from crates.io

- **Runtime training control protocol**: Extended the control file protocol (`.lr_control.json`) with `SaveCheckpoint` and `GracefulStop` commands. The adaptive LR controller now polls the control file before checking its `enabled` flag, so external agents (MCP, TUI) can always send commands regardless of whether automatic detection is active

- **`--no-adaptive-lr` flag**: Disables automatic spike/plateau/divergence detection while keeping the control file protocol active. Enables fully LLM-driven learning rate control — the agent observes loss via `job_status` and manually adjusts LR via `job_set_lr`/`job_reduce_lr`

- **UltraFusion execution planner** (`pmetal-distributed`): Per-die stage planner for M-series Ultra Macs with in-memory channel transport backend for same-process links, avoiding TCP overhead on UltraFusion interconnect

- **MPP FlashAttention for head_dim 64/96**: Metal 4 MPP flash attention kernel now supports head_dim 64, 96, and 128 with stride-2/stride-3 SIMD lane packing and causal/non-causal variants

- **Tuna persistent disk cache**: The auto-tuner now persists benchmark results to disk, avoiding re-tuning on restart. Expanded search covers FlashAttention, FusedCrossEntropy, FusedNormLora, and FusedSwiGLU via function constants

- **MoE GPU top-k selection**: Expert top-k selection moved from CPU sort to GPU `argpartition_axis`, eliminating a sync point in the MoE forward path

- **`bench-workload` CLI command**: Benchmark a real cached workload for inference and short LoRA training with named presets (`--preset dense-qwen3`, `--preset hybrid-qwen3next`)

- **KV cache quantization auto-select**: `--kv-quant` is now optional — omitting it auto-selects the fastest quantization mode that fits the device memory budget

- **UltraFusion info display**: `pmetal info` shows UltraFusion topology, die count, and local executor plan on Ultra Macs

- **Qwen3 LoRA RoPE reset**: Qwen3 LoRA and QLoRA gain dense attention and RoPE reset support

- **ANE real-time evaluation**: Experimental `_ANEClient` real-time dispatch with automatic fallback to standard evaluation on failure. Propagated via `--ane-real-time` CLI flag

- **`bench-corpus` CLI command**: Structured kernel benchmarking with device-tier-aware test cases, JSON reporting, and `--quick`/`--output` flags

- **GPU memory bandwidth probing**: Real GPU copy benchmark replaces static spec-table lookup, with disk-cached results and spec-table fallback

- **Persistent runtime kernel backend selection**: Benchmark-and-persist infrastructure races MLX vs MPP backends on Apple10/M5, validates numerical agreement, and caches the winner to disk for 4-bit quantized linear, fused attention, and LoRA matmul

- **MPP kernel tile variants**: Metal 4 GEMM supports parameterized tile variants (32x32, 64x32, 32x64, 64x64) with Tuna auto-tuner selection per device and problem shape

- **Serve ANE/CPU-hybrid engine caching**: Serve engine auto-selects optimal backend (ANE, CPU-hybrid, GPU) at startup with permanent downgrade on failure. Compiled engines cached across requests

- **Rollback enabled by default for LoRA**: Best-loss checkpoint rollback now defaults to on with extended warmup grace period. Persistent snapshot to disk via atomic write. `for_lora()` factory for recommended defaults

- **Extended StepMetrics**: `gpu_fwd_bwd_ms`, `optimizer_ms`, `io_staging_ms`, `overhead_ms` fields for fine-grained training profiling

- **Zero-copy MoE expert dispatch**: `ExpertBufferPool` with `read_experts_aligned` + `encode_expert_aligned` for pread-to-Metal expert weight dispatch. Auto-enable KV-Q8 when memory-constrained

- **ANE dual-die support**: On UltraFusion chips, compile variant-B kernel set with distinct MIL hashes and alternate per step for dual-die thermal distribution. Auto-recompile on throughput degradation (>15% or >25K dispatches)

- **Batched parameter eval**: Model dispatcher evaluates parameters in batches of 128 tensors per sync instead of all-at-once, reducing peak memory during model loading

- **Architecture enhancements**: DeepSeek V3/V3.2, GPT-OSS, Jamba, Llama 4, Qwen3, and Qwen3-MoE model improvements and weight sanitization refinements

- **Third-party attribution**: Complete THIRD_PARTY_NOTICES with entries for mlx-lm, llama.cpp/GGML, Candle, and Burn

### Changed

- **ANE is now opt-in**: The `--no-ane` flag has been replaced with `--ane` across CLI, TUI, orchestrator, and MCP. ANE training is experimental and limited to small models, so it defaults to off. The orchestrator's `DispatchConfig` now sets `ane: false` by default
- **Gradient checkpointing support corrected**: Qwen3 and Qwen3Next no longer claim gradient checkpointing support (was incorrectly advertised)
- **Training loop refactored**: Gradient checkpointing helper extracted, step logging tracks step numbers correctly, training loop tests expanded

### Removed

- **Merge methods**: Removed merge methods with incompatible licenses. Cleaned up related references across documentation and configuration

### Fixed

- **MetalSampler use-after-free**: Retained source logits array until GPU completion in serve engine
- **Fused merge Tuna cache**: Now uses persistent disk cache instead of ephemeral per-session tuning

## [0.3.13] - 2026-03-22

### Added

- **Warmup-aware adaptive LR**: Grace period now automatically extends to cover the LR scheduler's warmup duration via `set_warmup_steps()`, preventing false divergence triggers during the normal LR ramp. Backed by ZClip/SPAM research — loss increasing during warmup is expected behavior
- **`WarmupCapped` LR event**: Optional early warmup monitoring (disabled by default) for pre-training runs where loss rise during warmup may indicate problems. Enable with `warmup_max_loss_increase: 0.03-0.05`

- **Metal 4 / MPP kernel suite**: 8 Metal Performance Primitives kernels for M5 (Apple10) NAX hardware acceleration, compiled as a separate `pmetal_kernels_metal4.metallib` with automatic runtime dispatch
  - `mpp_gemm.metal`: Core GEMM with Morton ordering, fp16/fp32 variants, and alpha/beta accumulation via cooperative tensor postfix fusion (BK=128 K-loop per MPP Guide Section 2.3.4)
  - `mpp_flash_attention.metal`: FlashAttention-2 with both QK and PV block GEMMs via matmul2d — QK uses 32x32 tiles, PV uses 4 chunks of 32x32 for D=128, P stored as half for PV matmul (30KB threadgroup memory budget)
  - `mpp_fused_swiglu.metal`: Fused SwiGLU MLP — gate and up projections via matmul2d into threadgroup tiles, SwiGLU activation as cooperative post-step
  - `mpp_fused_lora.metal`: Fully fused LoRA forward — base projection via matmul2d, xA computed cooperatively in threadgroup scratch (shared across output elements), LoRA overlay added per-element. Training variant saves xA for backward pass
  - `mpp_fused_norm_lora.metal`: Fused RMSNorm + Linear + LoRA — SIMD cooperative RMS reduction, vectorized norm+dot for base projection, xA computed once and shared via threadgroup scratch
  - `mpp_grouped_gemm.metal`: MoE grouped GEMM with per-expert Morton ordering for LLC cache locality, sequential expert-offset tile lookup
  - `mpp_dw_gemm.metal`: ANE training weight gradient GEMM — simple overwrite and alpha/beta accumulation paths with cooperative tensor postfix fusion
  - `mpp_quantized.metal`: NAX quantized inference — 4-bit (on-the-fly dequant + matmul2d, BK=32) and 8-bit (on-the-fly dequant with per-group scale, BK=64) variants
- **Dual metallib build system**: Conditional Metal 4 compilation (`-std=metal4.0 -target air64-apple-macos26.0`) when Metal compiler >= 400 and SDK >= 26.0, with `has_metal4` cfg flag for Rust-side conditional compilation
- **Metal 4 pipeline cache**: `PipelineCache::load_metal4_library()`, `get_or_create_metal4_pipeline()` with function constant support and `"metal4:"` key prefix
- **NAX detection**: `DeviceProperties::has_nax()` (Apple10+ / architecture gen >= 17), automatic Metal 4 library loading when NAX is available
- **MPP GEMM Rust dispatch** (`mpp_gemm.rs`): `MppGemm` with `is_available()` check, type-erased `execute(&dyn AsMetalBuffer)`, Morton ordering via function constants, linearized 1D grid
- **Benchmark infrastructure** (`mpp_bench.rs`): `bench_gpu_op()` with warmup/timed iterations, `bench_comparative()` for Metal 3 vs Metal 4 side-by-side, `GemmBenchConfig` with standard problem sizes from decode (M=1) to training (M=2048)
- **Generic FP8 quantization** (`fp8_utils.rs`): `quantize_model_linears()` via `ModuleParameters` trait traversal — works for all 18 architectures

### Changed

- **Crate consolidation**: `pmetal-cli` merged into `pmetal` crate — SDK facade + CLI binary in one crate. `cargo build -p pmetal` now builds the binary (feature-gated behind `cli`). Library-only usage: `--no-default-features --features core`
- **Adaptive LR defaults (conservative)**: `divergence_slope_threshold` 0.05 → 0.005, `warmup_fraction` 0.15 → 0.25, warmup monitoring disabled by default. Grace period tied to actual LR scheduler warmup_steps instead of a fixed fraction of total steps
- **Metal 3 fused_swiglu_forward_f16**: Rewritten with SIMD cooperative reduction (was per-thread independent dots)
- **Metal 3 fused_cross_entropy**: Label smoothing enabled in SIMD path, `use_simd` now defaults to true for all vocab sizes
- **Metal 3 grouped_gemm backward_dx**: Replaced untiled O(M*N*K) inner loop with BLOCK_K-sized N-reduction strips using threadgroup staging
- **FP8 dispatcher**: Changed catch-all error to 18 explicit match arms calling generic `quantize_model_linears()`

### Fixed

- **mpp_fused_norm_lora.metal 512KB threadgroup alloc**: `threadgroup half norm_tile[64*4096]` (524KB) exceeded 32KB limit — would crash pipeline creation. Rewrote to use 68-float scratch buffer
- **mpp_fused_norm_lora.metal LoRA O(R*H) per output**: LoRA recomputed `xA` from scratch for every output element. Now computes xA once cooperatively and shares via threadgroup scratch
- **mpp_quantized.metal 8-bit unscaled output**: Per-group scale was deferred to nonexistent "separate pass". Rewrote with accumulation loop applying scale during on-the-fly dequantization
- **mpp_gemm.rs buffer type mismatch**: `execute()` took `MetalBuffer<f32>` but dispatched to f16 kernels. Changed to `&dyn AsMetalBuffer` (type-erased)
- **mpp_gemm.rs accumulate buffer binding**: Accumulate kernel (A=0, B=1, C=2, D=3, params=4) was bound with params at index 3 and missing D buffer. Fixed index mapping
- **mpp_flash_attention.metal scalar PV GEMM**: O += P @ V used per-element scalar loops despite header claiming block GEMM. Implemented via matmul2d with 4 chunks of 32x32 for D=128
- **mpp_fused_lora.metal stub LoRA phases**: Only base projection was implemented. Added cooperative xA computation + per-element LoRA overlay for both training and inference variants
- **matmul2d_descriptor tile size mismatch** (prior diligence): 6 kernels had `desc(32,32)` producing 32x32 threadgroup tiles but Rust dispatch strided by 64 — 75% of output uncomputed. Fixed to `desc(64,64)` except FlashAttention (correctly 32x32 for Bq=Bk=32)
- **Adaptive LR divergence threshold too high**: `divergence_slope_threshold` of 0.05 required ~200% loss increase over 40 steps — effectively dead code. Reduced to 0.005 (20% over 40 steps)
- **Adaptive LR step counter in run_packed**: Accumulated losses were all fed to the adaptive controller with the same batch-end step number. Now retroactively sets the correct per-step value so the grace period and detection windows align properly
- **Clippy `needless_return`**: Removed bare `return` in `pmetal-metal/build.rs` match arm

## [0.3.12] - 2026-03-21

### Added

- **MLX memory management**: Wire real MLX Metal memory API via `mlx_rs::memory` (published in pmetal-mlx-rs 0.25.8). Exposes `clear_cache`, `get_active_memory`, `get_peak_memory`, `get_cache_memory`, `get/set_memory_limit`, `set_cache_limit`, `set_wired_limit`, `reset_peak_memory`. The previous `clear_cache()` was a complete no-op — MLX buffer cache was never freed
- **Memory diagnostics**: `log_memory_stats()` reports active/cache/peak/limit at model load, training start, and completion for visibility into Metal allocator state
- **LoRA**: Implemented dynamic QLoRA and fused metal kernels
- **KV cache quantization** (SOTA inference): q8_0 KV cache is now the default for inference and serving — community benchmarks confirm <0.4% PPL degradation with 12-38% throughput gain
  - Symmetric quantization: `--kv-quant 8` (default), `--kv-quant 4` for aggressive savings
  - Asymmetric K/V quantization: `--kv-k-bits 8 --kv-v-bits 4` — K is more sensitive than V, asymmetric gives near-q4 memory savings with near-q8 quality
  - `--kv-group-size` (default 64), `--no-kv-quant` to disable
  - `CacheMode::Quantized` and `CacheMode::AsymmetricQuantized` variants wired into `KVCache::update_and_fetch()` via per-layer `QuantizedKVCache` delegation
  - `CacheMode::describe()` for human-readable display in logs
  - `DynamicModel::create_cache_with_mode()` with automatic group_size adjustment for non-standard head dimensions (Phi-3 mini head_dim=96, NemotronH head_dim=32)
  - Serve engine defaults to q8_0 KV cache for all requests
- **Context-aware fit estimation**: Efficiency factor is now context-dependent (0.60 dense / 0.50 MoE base, log-linear penalty above 8k context) instead of flat 0.55. KV cache memory calculation accounts for quantization bits. Fit notes recommend q8_0 when memory is tight

### Changed

- **Trainer**: Modularized training loop and configured experimental trainers
- **MLX**: Moved `kv_cache` to a module and updated gated delta
- **Style**: Formatted CLI, Hub, Models, and Serve crates
- **Cleanup**: Removed `easy_reference.rs`

### Fixed

- **Training memory explosion (72 GB peak for 0.6B model → 23 GB)**: Three root causes identified and fixed:
  1. Computation graphs accumulated across deferred evaluation steps. Packed and compiled training paths deferred eval for 10 steps, keeping ~10 full forward+backward graphs in memory simultaneously. Now evaluates each step immediately
  2. Gradient accumulation kept backward graphs alive. Each micro-batch's gradient arrays held references to the entire backward computation graph until gradients were applied. Now evaluates accumulated gradients after each micro-batch
  3. `eval_params(model.parameters())` evaluated all 600M+ frozen base model params every step. Changed to `eval_params(model.trainable_parameters())` — only LoRA adapters (~1-10M params)
- **ANE training attempt doubled memory for LoRA/QLoRA**: The orchestrator always attempted ANE training first (loading full model weights), failed (ANE is incompatible with LoRA adapters), then loaded the GPU LoRA model — keeping both in memory. Now skips ANE entirely for LoRA/QLoRA
- **GUI training defaults misaligned with CLI**: GUI used `batch_size=4` (CLI: 1) and `gradient_checkpointing=false` (CLI: true), causing 5-10x more memory usage for identical models. Defaults now match CLI
- **GUI training logs silently dropped**: Tracing filter only showed `pmetal_gui` crate logs. Training progress from orchestrator, trainer, and model crates was invisible. Now shows all pmetal crate logs
- **MLX buffer cache not freed after training**: `clear_cache()` now runs after training ends (success, error, or cancel) in both orchestrator and GUI
- **KV cache `Quantized` mode was a no-op**: `CacheMode::Quantized` existed in the enum but `KVCache::update_and_fetch()` treated it identically to `Standard`. Now properly delegates to per-layer `QuantizedKVCache` instances with quantize-on-write / dequantize-on-read
- **KV cache quantization crash on non-standard head dimensions**: Models with head_dim not divisible by 64 (Phi-3 mini=96, NemotronH=32, FalconH1=16) would fail in MLX's `quantize` op. `create_cache_with_mode()` now auto-adjusts group_size to the largest compatible power-of-2

## [0.3.11] - 2026-03-20

### Added

- **Production serving engine** (`pmetal-serve`): Complete rewrite from greedy-only PoC to production-quality inference server
  - SOTA GPU-native sampling via `pmetal_models::Sampler`: temperature, top-k, top-p, min-p, repetition/frequency/presence penalties, seeded RNG
  - True token-by-token streaming via `tokio::sync::mpsc` channels — first byte reaches client as GPU produces it (no collect-then-emit)
  - `spawn_blocking` generation — HTTP event loop stays responsive during prefill/decode
  - Client disconnect detection — generation thread stops immediately when receiver drops
  - Input validation: bounds checking on all sampling params (rejects NaN, Inf, out-of-range), max_tokens clamped to max_seq_len
  - OpenAI API: `stop` field accepts both `"string"` and `["array"]` formats, `system_fingerprint` field in responses, request-time `created` timestamps
  - Stop token collection via `collect_all_stop_tokens()` (merges generation_config.json + chat template + tokenizer + well-known probes)
  - Multi-token stop strings filtered with warning (only single-token stop strings supported)
- **qdot MoE Metal kernels** (`fused_moe.metal`): Rewrote all 5 kernels with pre-scaled activation technique — ~30-40% compute reduction, thread-local x caching, register-only design (no threadgroup shared memory overflow), 64-thread threadgroups with function constant specialization
- **Deferred GPU dispatch**: Encode all K experts in single command buffer with one submit instead of K separate GPU flushes
- **Persistent IO thread pool** (`expert_io.rs`): Persistent workers via mpsc channels with zero-copy aligned pread into `AlignedBuffer` (2MB posix_memalign + `newBufferWithBytesNoCopy`)
- **Async expert prefetcher**: Background IO thread with ownership transfer via `Option::take` (no clone)
- **Real benchmark harness**: `--benchmark` / `--benchmark-iters` on `pmetal infer` runs real per-token forward passes with GPU sync, reports mean/min/p50/p99 decode latency
- **forward_offloaded wired end-to-end**: Full pipeline: route → pread K experts → parse → GPU dequant (single CMD buffer) → combine

### Fixed

- **Reasoning dataset training producing no `<think>` tags**: When GUI/CLI set custom text columns (`thinking`, `solution`) with prompt column `problem`, the Custom format path bypassed the Reasoning format's `<think>`/`</think>` tag injection. Auto-detection now routes reasoning-pattern columns to the Reasoning format. Also fixed prompt-column loss masking when prompt isn't in text columns (prompt is now prepended to training text)
- **Adaptive LR controller too aggressive**: Divergence detection was firing on normal training noise, crushing LR from 2e-4 to 1e-7 within 10% of training. SOTA-aligned overhaul:
  - Divergence confirmation: requires 2 consecutive positive-slope windows before triggering (was: single window)
  - Divergence cooldown: 80 steps after reduction before re-checking (was: immediate re-check after 40-step window refill)
  - Max divergence reductions: capped at 4 (was: unlimited cascading)
  - ZClip-style spike exclusion from EMA: detected spikes no longer inflate the EMA threshold
  - Gentler reduction factor: 0.7 per reduction (was: 0.5)
  - Increased grace period: 15% of training steps (was: 10%)
- **GUI LoRA inference producing garbage** (`_framework` token repeated): GUI was not reading `target_modules` from adapter_config.json (all modules got rank=16 instead of only attention) and was not merging LoRA weights before inference. Now reads `target_modules`/`use_rslora`, calls `merge_lora()` + `eval_all()` matching the working CLI path
- **GUI fuse with cached models failing**: "Fuse with remote base models is not supported" error removed — now calls `resolve_model_path()` to download/resolve, matching CLI behavior
- **Fused model not recognized by LM Studio**: Three fixes:
  - Safetensors metadata now uses `format: mlx` (required by LM Studio on macOS)
  - Generates `model.safetensors.index.json` with weight map (required for model discovery)
  - Fused weights preserve base model dtype (bf16/f16) instead of upcasting to f32 — halves output file size
- **`adapter_config.json` missing `base_model`**: Training now saves `base_model` field in adapter config (distillation, GRPO, RLKD paths). Enables auto-detection of base model from LoRA adapter
- **Serve security hardening**: Default bind `127.0.0.1` (was `0.0.0.0`), sanitized error responses (no internal MLX paths leaked), 2MB request body limit, removed unnecessary `unsafe impl Sync for ModelState`
- **Serve extra forward pass**: Decode loop restructured to not run a wasted forward pass after the final token
- **Serve SSE error handling**: `[DONE]` no longer emitted after `TokenEvent::Error` — stream ends with error event only
- **Serve UTF-8 streaming**: Token buffer accumulates and decodes together to prevent garbled multi-byte characters at BPE boundaries

### Changed

- **GUI fuse modal UX**: "Cancel" becomes "Done" after successful fuse with "Fuse Another" option. Base model field is read-only, auto-detected from LoRA adapter's `adapter_config.json`
- **GUI inference auto-select**: Selecting a LoRA adapter in inference automatically selects the matching base model (mirrors training page behavior)
- **Deleted MoE combine bridge**: Removed `FusedMoeCombine` Metal kernel — the 6 MLX ops are already async on GPU; the Metal side-channel added sync barriers making it 5-20x slower
- **GUI streaming inference**: Token-by-token streaming with full stop-token and sampling-config support
- **CLI `--loss-scale` flag**: Gradient scaling for ANE training at >350M params
- **Comprehensive documentation site** (`docs/`): Getting-started, installation, hardware, models, training, CLI reference (21 commands), configuration, SDK, Python, and contributing guides
- **Metal GPU backward kernels** (`dw_gemm.metal`): Tiled fp32 SGEMM for weight gradient GEMMs in ANE training — single `BatchedCommandBuffer` per step
- **GUI adapter discovery**: Scans `~/pmetal-output/` for trained LoRA adapters with rank, alpha, base model metadata
- **GUI adapter dropdowns**: Fuse modal and inference page replace manual path entry with adapter select dropdowns
- **GUI chat template support**: Model-specific chat templates via `detect_chat_template()`
- **`save_adapter_config_with_base()`**: Adapter config now includes `base_model` field

### Removed

- **`pmetal::easy` module** (breaking): Removed in favor of direct `pmetal-models` / `pmetal-hub` / `pmetal-data` usage

## [0.3.10] - 2026-03-18

### Added

- **Training orchestrator** (`pmetal-trainer::orchestrator`): Single `run_training()` entry point replaces four separate training pipeline implementations (CLI ~1000 lines, GUI ~190 lines, easy API ~200 lines, TUI bridge ~70 lines). All consumers now share one canonical pipeline with: ANE training with GPU fallback, QLoRA and standard LoRA, all dispatch modes (packed, compiled, metal-fused, standard), adaptive LR, checkpointing, metrics callbacks, and phase status reporting. Net -1300 lines of duplicated pipeline code
- **`TrainingJobConfig` struct**: Replaces 38 positional parameters with a typed config struct. Includes `DispatchConfig` (optimization flags), `QLoraOrchConfig` (quantization), `TrainingPhase` enum (status reporting), and `PhaseCallback` trait (GUI/TUI status wiring)
- **ANE large-vocab support** (`VocabMap::from_token_ids`, `VocabMap::remap_u32`): ANE training now correctly handles models with vocab > 65536 (e.g. Qwen3 @ 151936). Token IDs are processed as u32 through VocabMap compaction before converting to the u16 format required by ANE IOSurface operations. Previously, u32→u16 casting silently truncated IDs above 65535, corrupting embeddings and gradients
- **ANE first-class metrics**: ANE training path now wires `MetricsJsonCallback` with per-step metrics (loss, tok/s, ANE timing breakdowns), config JSON, and user-provided callbacks (cancel support). Previously ANE produced no metrics output, making GUI/TUI appear stuck during ANE training

### Fixed

- **GUI metrics not updating during training**: Metrics file watcher now detects file truncation (from ANE→GPU fallback) and resets read position. Previously `last_pos` exceeded the new file length after truncation, causing the watcher to skip all new data indefinitely
- **GUI output path relative to process cwd**: Training output now resolves to `~/pmetal-output/` instead of relative to the GUI's working directory (`crates/pmetal-gui/src-tauri/`). Absolute paths from the frontend are preserved as-is
- **GPU metrics callback truncates ANE metrics**: GPU `MetricsJsonCallback` creation moved to after ANE attempt completes, so ANE metrics aren't wiped on fallback
- **GUI drops warmup/lr_schedule/save_steps/logging_steps**: All four fields from the GUI training config DTO are now properly mapped to `TrainingConfig` instead of falling through to defaults
- **Phase status not visible in GUI**: Added `tokio::task::yield_now()` after each phase emit in pre-MLX orchestrator phases and ANE path, allowing the tokio runtime to deliver status events between blocking operations
- **TUI missing `embedding_lr` and `lr_schedule` parsing**: Direct training path now parses `--embedding-lr` and `--lr-schedule` args that were previously ignored
- **Easy API drops `embedding_lr`**: `FinetuneBuilder` now maps `embedding_lr` into `TrainingConfig.embedding_learning_rate`

- **GUI live training dashboard**: Full-screen live view replaces the config form when training is active. Includes real-time loss curve (SVG), metric cards (loss, best loss, tok/s, LR, grad norm, progress %), run details panel with hyperparameters, and progress bar. Config form returns when training stops
- **GUI cached dataset dropdown**: Dataset selector uses a `<select>` dropdown (matching the model selector style) with cached HuggingFace datasets, plus a text input for custom paths or HF dataset IDs
- **GUI dataset column picker**: When a dataset is selected, columns are auto-detected and shown in dropdowns for text, prompt (loss masking), and format selection. Falls back to manual text input when columns can't be detected
- **Multi-column dataset support** (`--text-columns col1,col2`): Concatenate multiple JSONL columns as training text. CLI: `--text-columns thinking,solution --column-separator "\n\n"`. GUI: ordered pill builder with add/remove/reorder. All training paths (Train, Distill, GRPO, RLKD) support column flags uniformly via shared `build_column_config` helper
- **Custom dataset columns** (`--text-column`, `--prompt-column`, `--response-column`): CLI, GUI, TUI, and easy API support arbitrary JSONL column names via `DatasetFormat::Custom`. Prompt column enables loss masking; prompt+response columns concatenate with masking. Distill, GRPO, and RLKD commands now also accept column flags
- **Unified `from_jsonl_tokenized`**: Merged `from_jsonl_tokenized` and `from_jsonl_tokenized_with_columns` into a single method with `columns: Option<&DatasetColumnConfig>`. All 18 call sites updated. DRY across CLI, TUI, GUI, easy API, and Python bindings
- **Dataset statistics and seq len validation**: `DatasetStatistics` with min/max/mean/median/p95/p99 lengths, truncation count/percentage, and suggested `max_seq_len`. `validate_seq_len()` warns when >10% truncated or mean length much shorter than max_seq_len. Logged for all training paths: Train, Distill, GRPO, RLKD, and easy API
- **`peek_dataset_columns`**: Tauri command + API function — reads first JSONL record and returns field names for the GUI column picker
- **GUI training status phases**: Live status messages during setup ("Loading model...", "Loading dataset and tokenising...", "Training...") so users see what's happening before metrics arrive
- **GUI training config summary**: Hyperparameters (LR, batch, seq len, LoRA rank, packing, flash attention) displayed in the active training banner and run detail panel
- **GUI failed-run alerts**: Failed training runs immediately surface with error message in a red banner, no longer hidden until the user clicks stop
- **GUI auto-updater**: Tauri updater plugin with signed update artifacts and `latest.json` manifest in GitHub releases
- **TUI setup phase indicator**: Dashboard shows "Loading model and preparing dataset..." in loss chart and stats panel while model loads, before any metrics arrive
- **TUI `JobPhase` event**: New `AppMsg::JobPhase` message propagates setup status from the command runner to the dashboard
- **TUI dataset peek**: Shows detected columns, estimated token lengths, and seq len warnings when a dataset is selected in the training form
- **GUI seq len warnings**: Contextual warnings under the max seq len input — red (most samples truncated), amber (some truncated), blue (wasteful). Shows "Based on first N rows" with a "check all rows" button that scans the full dataset on the backend
- **GUI retry button**: Completed/cancelled/failed runs show "Retry with these settings" which loads the run's config back into the form for adjustment and re-launch
- **Easy API `on_status()` callback**: Reports granular setup phases (resolving model, resolving dataset, loading tokenizer, tokenizing dataset, loading LoRA adapters, training) — wired to GUI for real-time phase display
- **`find_cached_model` / `find_cached_dataset`** (`pmetal-hub`): Fast local cache lookup for HF repos without network calls

### Fixed

- **Cached models re-downloaded on every training start** (`pmetal-hub`): `download_model` and `download_dataset` now check the local HF cache (`~/.cache/huggingface/hub/`) before making any network calls. If a valid snapshot exists, the cached path is returned instantly. Eliminates ~10s startup latency for cached models across all consumers (CLI, TUI, GUI, easy API, Python SDK)
- **Seq len suggestions use next-multiple-of-64** instead of next-power-of-2, producing practical values (7168 instead of 8192) for GPU-aligned training
- **Non-string dataset columns crash** (`parse_custom_line`): Selecting a column containing an array (e.g. OpenAI `messages` chat format) or number crashed with "not a string". Now handles all JSON types: arrays of message objects auto-extract role+content, numbers/booleans convert to string, other types serialize to JSON. Relates to #2

- **GUI/API trending models and datasets stale**: Changed HuggingFace API sort from `sort=downloads` (all-time) to `sort=trending` for default browse views. Search queries still sort by downloads. Fixed hardcoded User-Agent version string to use `CARGO_PKG_VERSION`
- **HF dataset ID resolution** (`pmetal-data`, `easy.rs`, `commands.rs`, `main.rs`): HuggingFace dataset IDs (e.g., `nohurry/Opus-4.6-Reasoning-3000x-filtered`) and local HF cache directories are now resolved to the actual data file within. Traverses `snapshots/{hash}/` structure, follows symlinks, finds `.jsonl`/`.json`/`.parquet`/`.csv`/`.arrow` in priority order
- **Dataset directory passed as file path**: All three resolution sites (`easy.rs`, GUI `commands.rs`, CLI `main.rs`) now call `resolve_dataset_path_pub` for `DatasetSource::Local` directories instead of passing them as-is to `from_jsonl_tokenized`
- **Metrics not appearing in GUI/TUI**: `log_every` changed from 10 to 1 in `easy.rs` so metrics appear after the first training step. `MetricsJsonCallback` now flushes every step for the first 20 steps (then every 5), ensuring watchers see data promptly
- **`train_start` event handling in GUI**: `apply_metrics_to_training` now recognizes the `train_start` event, sets status message, and reads `total_epochs` from step metrics
- **Watcher task leak on training completion** (GUI + TUI): `finalize_training_run` (and distillation/GRPO variants) now sets `cancel_flag = true` so the 500ms metrics-polling task exits. TUI `CommandRunner::remove()` now calls `job.cancel.cancel()` before dropping
- **QLoRA re-resolves dataset**: `run_qlora_training_in_process` now receives the pre-resolved `PathBuf` instead of re-downloading from HuggingFace on every run
- **Stale status message on failure**: `finalize_training_run` clears `status_message` to `None` so "Loading model..." doesn't overlay the error message
- **README.md failure aborts dataset download** (`pmetal-hub`): README failures are now non-fatal warnings; only data file download failures abort
- **MLX mutex crash on GUI exit**: Added `on_window_event(Destroyed)` handler that calls `std::process::exit(0)` to skip C++ destructor crashes from MLX Arrays dropped on the wrong thread
- **TUI log corruption**: Tracing subscriber suppressed in TUI mode to prevent stderr writes from corrupting the raw terminal. Optional `PMETAL_LOG_FILE` env var for file-based debug logging (with graceful fallback on bad paths)
- **`log_lines` dead code removed**: Removed unused `log_lines` field from `TrainingRun`, `DistillationRun`, and `GrpoRun` GUI state structs

### Changed

- **Release workflow**: Added Tauri signing keys, updater artifacts (`.tar.gz` + `.sig`), and `latest.json` manifest generation for auto-updates
- **GUI Cargo.toml**: Added `tauri-plugin-updater` and `tauri-plugin-process` dependencies

## [0.3.9] - 2026-03-17

### Added

- **RLKD CLI command** (`pmetal rlkd`): Reinforcement Learning with Knowledge Distillation — combines GRPO policy gradient optimization with distillation from a frozen teacher model. CLI exposes `--alpha`, `--final-alpha`, `--anneal-alpha`, `--top-k-distill`, all SFT/LoRA arguments, and `MetricsJsonCallback` integration
- **Embedding training CLI command** (`pmetal embed-train`): Sentence-transformer fine-tuning for BERT/encoder models with contrastive losses (InfoNCE, Triplet, CoSENT). Supports pair and triplet datasets, configurable pooling (CLS, Mean, LastToken), L2 normalization toggle, and automatic tokenizer/config copying to output
- **GRPO VLM mode** (`--vlm`): Vision-Language Model support for GRPO training with image inputs. Loads images from dataset `images` field, passes to reward functions, uses `forward_with_images` for multimodal forward passes. Configurable `--max-image-size`
- **GRPO ML reward model** (`--reward-model`): Pretrained reward model scoring during GRPO. Loads from local path or HuggingFace ID, runs inference-only alongside heuristic rewards. Configurable `--reward-model-weight`, `--reward-model-max-length`, and `--reward-model-template`
- **GRPO speculative decoding** (`--speculative`): Draft/verify rollout generation with 2-4x throughput improvement. Configurable `--speculative-draft-tokens` (default 3). Greedy verification for correctness guarantees
- **GRPO async reward pipelining** (`--async-rewards`): Background reward scoring concurrent with GPU training for ML reward models
- **Cut Cross-Entropy CLI flag** (`--cut-cross-entropy`): Memory-efficient loss computation for SFT training, avoiding full [batch, seq, vocab] logit materialization
- **KL-calibrated GGUF quantization** (`--kl-calibrate`): Per-tensor quantization type selection via NRMSE + cosine distance calibration. `--target-bpw` for budget-constrained quantization, `--kl-threshold` for quality control
- **GRPO TUI form fields**: VLM toggle, speculative decoding, async rewards, ML reward model path, and draft tokens exposed in the interactive TUI
- **Training TUI**: Cut Cross-Entropy toggle added to training tab form

### Fixed

- **Cut Cross-Entropy ignore index panic** (`pmetal-mlx`): `take_axis` with -100 (ignore index) targets caused out-of-bounds gather. Targets are now clamped to valid range before gather; loss masking handles ignored positions
- **Cut Cross-Entropy division by zero** (`pmetal-mlx`): `n_valid=0` (all tokens ignored) caused NaN loss. Guarded with `n_valid.max(1)`
- **Llama LoRA position IDs dropped** (`pmetal-lora`): `forward_hidden_with_positions` silently discarded position IDs, breaking packed-sequence training with non-contiguous positions. Added full position-aware path through attention, decoder layer, and model stack using `apply_rope_with_positions`
- **lm_head weight computed twice per CCE step** (`pmetal-trainer`): Training loop called `lm_head_weight()` for probe and again inside the gradient closure. Weight is now computed once and captured into the closure
- **GRPO VLM pixel_values not replicated per-completion** (`pmetal-trainer`): Images were stacked per-group instead of replicated per-completion, causing batch dimension mismatch. Images now repeat `n_completions` times per group
- **GRPO `run_async` flush skips adaptive LR** (`pmetal-trainer`): Final flush step bypassed adaptive LR, rollback logic, and callbacks. Now applies the same post-step processing as the main loop
- **CoSENT loss overflow with no positive pairs** (`pmetal-trainer`): All-zero labels caused `logsumexp(-1e9)` overflow to `+inf` and `NaN` gradients. Returns `0.0` when no positive pairs exist in the batch
- **LastToken pooling O(batch) GPU syncs** (`pmetal-models`): Per-element `.item()` loop forced one GPU-to-CPU synchronization per batch element. Replaced with vectorized `take_along_axis` + `broadcast_to` for a single gather operation
- **EmbeddingDataset silent empty strings** (`pmetal-data`): Missing text keys (`text_a`/`text_b`) silently produced empty-string training pairs. Now returns an explicit parse error with line number and expected key names
- **BERT `hidden_act` always GELU** (`pmetal-models`): `BertIntermediate::forward` ignored the `hidden_act` config field. Now dispatches to `relu`, `silu`/`swish`, `tanh`, or `gelu` (default) based on config
- **GGUF BPW budget silent non-convergence** (`pmetal-gguf`): `apply_bpw_budget` loop exhausted without warning when all tensors were at minimum quality. Emits `tracing::warn!` when target BPW is unreachable
- **Speculative decode cross-sequence early exit** (`pmetal-models`): Outer generation loop exited when any single sequence hit `max_new_tokens`, truncating other in-progress sequences. Removed `max_generated` check; per-sequence `finished` tracking now controls termination
- **Speculative decode O(seq_len) draft warm-up** (`pmetal-models`): Draft cache was rebuilt from full sequence prefix every step, making total cost O(seq_len^2). Draft caches are now persisted and incrementally advanced with only newly accepted tokens
- **Fused LoRA backward threadgroup memory** (`pmetal-metal`): `fused_lora_backward_a` kernel missing threadgroup memory size check. Added allocation guard with fallback to MLX for large `out_features`
- **LoRA+ double scaling** (`pmetal-lora`): Fused kernel and `AdamWGroups` optimizer could both apply the LoRA+ differential learning rate. Added `kernel_loraplus` flag to prevent double scaling
- **Clippy compliance**: Fixed `field_reassign_with_default` in GGUF calibration summary, `doc_overindented_list_items` in speculative decode docs

### Changed

- **RLKD stats**: Documented that `grpo_component` and `distill_component` in training stats are proportional approximations (`total_loss * (1-alpha)` and `total_loss * alpha`), not true decomposed values
- **Speculative decode bonus token**: Documented that greedy argmax for the bonus token is by design (required for speculative decoding correctness), not a sampling oversight
- **GGUF prefix subsampling**: Expanded documentation warning that prefix subsample assumes i.i.d. weight distribution, which may not hold for structured tensors
- **EmbeddingTrainer**: Added doc warnings that `encode` requires models returning hidden states (not logits) — causal LMs produce `[batch, vocab]` after pooling, which is nonsensical as an embedding

## [0.3.8] - 2026-03-17

### Added

- **Distributed training** (`pmetal-trainer`): Data-parallel gradient synchronization across Apple Silicon clusters via `DistributedGradientSync`. Flatten/all-reduce(Mean)/scatter pipeline with optional gradient compression (fp16, top-k sparsity). Integrated at all 4 training loop sites (run, run_metal_fused, run_jit_compiled, run_packed) with loss sync, epoch barriers, and rank-0-only checkpointing. Feature-gated behind `distributed`
- **Pipeline-parallel inference** (`pmetal-distributed`): Layer-range pipeline parallelism enabling models larger than single-device memory. `ShardableModel` trait decomposes forward pass into embed/apply_layer/normalize/lm_head stages. `PipelineGenerationLoop` for end-to-end autoregressive generation with `StreamMultiplexer` for concurrent request routing
- **Activation transport**: Length-prefixed wire format for hidden state transfer between pipeline stages with fp16 compression codec. `TransportReceiver::recv_vec` for dynamic-size message reception
- **Topology-aware layer assignment**: Proportional (RAM-based) and bandwidth-aware (exhaustive search for 2-3 nodes) solvers with automatic strategy selection based on cluster topology
- **Weight cache**: LRU eviction with reference counting to prevent in-use eviction, per-layer loading, and prefetch support for pipeline stages
- **OpenAI-compatible inference server** (`pmetal-serve`): Drop-in local inference backend with `POST /v1/chat/completions` (streaming SSE and non-streaming), `POST /v1/completions`, `GET /v1/models`, `GET /v1/metrics`, `GET /health`. Chat template auto-detection, stop token collection, and greedy sampling
- **Serving metrics**: Per-request timing (`RequestMetrics`) with first-token latency, total latency, and tok/s. `ServingMetrics` atomic aggregation exposed via `/v1/metrics` endpoint
- **SSE streaming**: Token-by-token Server-Sent Events with role announcement, per-token content deltas, finish_reason, and `[DONE]` sentinel per OpenAI spec
- **Speculative decoding** (`pmetal-models`): Layer-split draft+verify decoder via `SpeculativeDecoder<M: ShardableModel>`. Draft phase uses early layers (default: num_layers/3) for N-token proposals, verify phase runs full model with accept/reject on consecutive matches. `SpeculativeStats` tracks acceptance rate and tokens-per-step
- **f64-accurate LoRA merge** (`pmetal-merge`): Streaming f64 matmul via ndarray for bit-accurate delta computation. Row-by-row fused base+delta+downcast, tiled low-memory path (512-row chunks), bias merging, fan_in_fan_out transpose, overflow clamping before dtype downcast
- **RAM/RAM+ merge method**: Reinforced Agent Merging with unique/shared parameter classification and adaptive tensor-local lambda rescaling
- **Multi-SLERP merge method**: Barycentric spherical interpolation for 3+ models with iterative pairwise SLERP and weight renormalization
- **Frankenmerging config**: `OutputSlice`/`InputSlice` layer-range-based merging with per-slice merge methods, base models, and parameters. `run_merge_sliced()` execution engine with tensor name remapping
- **`ParameterSetting`**: Scalar or conditional (tensor-name filtered) merge parameters enabling per-tensor-type weight variation (attention vs mlp layers)
- **TVD distillation loss**: Total Variation Distance (`0.5 * Σ|P_teacher - P_student|`), bounded [0,1], symmetric proper distance metric
- **Hinge ranking distillation loss**: Pairwise margin-based ranking preservation over top-k teacher tokens with configurable margin
- **Logistic ranking distillation loss**: Softplus-based smooth ranking loss with better gradient flow than hinge, operates on logits for numerical stability
- **CLI `--distributed-peers`, `--distributed-auto`, `--compression-strategy`**: Distributed training flags behind `distributed` feature
- **CLI `pmetal serve --model <path> --port 8080`**: Inference server command behind `serve` feature
- **CLI `--accurate` and `--low-memory`**: Flags for f64 LoRA merge path
- **New merge methods in CLI**: `ram`, `ram_plus`, `multislerp` registered as merge method options

### Fixed

- **Alignment violation in distributed gradient sync**: `sync_gradients` and `sync_loss` previously created `Vec<u8>` buffers with align-1, but the ring backend requires align-4 for f32 operations. Fixed by reinterpreting the `Vec<f32>` buffer directly via aligned pointer cast
- **Double-framing deadlock in activation transport**: `serialize()` embedded its own length prefix AND `TransportSender::send()` added another, causing `recv_activation` to misparse messages. Removed embedded prefix; transport layer handles all framing
- **Double EMA on `running_loss` in distributed mode**: Distributed sync block re-applied EMA that `train_step` already applied, causing doubly-decayed loss values for the adaptive LR controller. Removed manual EMA update in distributed block
- **Zero weights in bandwidth-aware layer assignment**: 3+-node fallback computed `(ram / 1M) * (bw / 1M)` which produced zero for small values, causing NaN proportions. Added `.max(1)` guards
- **`argpartition` panic on ranking losses**: `k.min(vocab - 1)` could underflow when vocab=0. Added `.max(0)` guard

### Changed

- **DataLoader sharding**: `rank` and `world_size` fields for modular-arithmetic data partitioning across distributed nodes
- **Merge config system**: `ParameterSetting` type propagated to CLI merge parameter construction, supporting both scalar and conditional forms

## [0.3.7] - 2026-03-16

### Added

- **`pmetal merge` CLI command**: Model merging exposed as a first-class CLI command supporting all merge methods (Linear, SLERP, TIES, DARE, DELLA, NearSwap, Model Stock) with `--method`, `--base`, `--t`, `--weight-a`, `--weight-b`, `--density`, and `--dtype` flags
- **`pmetal eval` CLI command**: Dataset evaluation command — measures loss/perplexity over a validation set with optional LoRA adapter, `--num-samples` cap, and `--json` output
- **`pmetal info` CLI command**: Prints device and runtime information; `--json` flag emits structured JSON for scripting
- **`pmetal search --json` output**: Structured JSON output mode for search results including fit estimates, download counts, parameter estimates, and tags — enables scripting and GUI integration
- **`QuantizeMethod` enum**: Replaces the string `--method` argument for `pmetal quantize` with a typed enum (`dynamic`, `q8_0`, `q4_k_m`, etc.) — invalid methods now fail at argument parsing rather than deep inside the quantizer
- **GRPO CLI arguments**: `--epochs`, `--lora-r`, `--lora-alpha`, `--max-completion-length`, and `--seed` exposed as CLI arguments, replacing previous hardcoded defaults
- **`loraplus_lr_ratio` and `neftune_noise_alpha`**: New fields on training loop configurations — enables LoRA+ differential learning rates and NEFTune noise injection directly from config
- **`trainable_params()` helper**: New utility in `pmetal-lora` for counting total vs. trainable parameter counts, useful for logging and memory estimation
- **`lora_alpha: f32`**: Distillation CLI and `run_distillation_cli` now accept `lora_alpha` as `f32` instead of `usize` for finer-grained scaling control
- **`seed` parameter in distillation and GRPO CLI**: Reproducible runs via explicit `--seed` flag in all training entry points
- **Gemma3 sliding window auto-detection**: `DynamicModel` loader now reads `model_type == "gemma3"` and sets `is_gemma3 = true` on the config, enabling the correct every-6th-layer global attention pattern without manual config overrides
- **KV cache support for more architectures**: `DynamicModel::forward_with_cache` now routes DeepSeek, Cohere, StarCoder2, and Llama4 to their native caching paths; RecurrentGemma and Jamba now get clear error messages that they require `forward()` directly; hybrid models (NemotronH, Qwen3Next) get a descriptive error directing to `forward_with_hybrid_cache`
- **Speculative decoding greedy path**: `SpeculativeDecoder::verify_greedy()` — exact-correct verification for temperature=0 decoding using argmax equality; avoids the numerically unstable rejection-sampling limit as temperature→0
- **Hub cache management** (`pmetal-hub`): New `cache.rs` module with cache inspection, eviction, and size-reporting helpers
- **Shared model utilities** (`pmetal-models/utils.rs`): Common helpers extracted from per-architecture modules to reduce duplication

### Fixed

- **Scale factor broadcasting in distillation**: `squeeze` applied to the scale factor dimension so it broadcasts correctly across batch and sequence axes — previously caused shape mismatches on non-unit batch sizes
- **TAID `mean_alpha` forcing GPU sync**: `TaidLossOutput::mean_alpha` changed from `f32` to a lazy `Array` — the `.eval()` call is deferred until callers explicitly call `.item::<f32>()`, removing a forced GPU-CPU sync before the backward pass
- **SLERP numerical stability**: Added epsilon clamping in the SLERP merge path to prevent NaN when interpolation parameter is at the boundary values (0.0 or 1.0)
- **Llama LoRA `trainable_params` / gradient application**: Replaced 100+ lines of repeated field accesses with an `insert_adapter!` macro and loop over projection names, fixing DoRA `magnitude` parameter that was silently dropped from gradient maps
- **GaLore improvements**: Corrected projection matrix update schedule and subspace dimensionality handling
- **Distillation hidden-state loss**: Refactored alignment computation to correctly handle variable-rank teacher/student hidden state tensors
- **Jensen-Shannon / KL divergence loss**: Numerical stability improvements — log-sum-exp stabilization applied consistently across all reduction paths
- **Offline distillation**: Fixed logit cache loading to handle both single-file and sharded cache layouts

### Changed

- **`lm_groups.rs` / LoRA+ optimizer groups**: `build_lora_param_groups` significantly reworked — LoRA+ differential LR ratio (`loraplus_lr_ratio`) applied to `lora_b` parameters, NEFTune noise injection integrated into group construction
- **GRPO trainer**: `epochs`, `lora_r`, `lora_alpha`, `max_completion_length`, and `seed` plumbed through from CLI args; previously these were hardcoded to `1`, `16`, `32`, `512`, and a fixed seed
- **Training loop**: `loraplus_lr_ratio` and `neftune_noise_alpha` read from config and forwarded to optimizer group construction
- **`pmetal-core` config / scheduler / traits**: Config structs gained `loraplus_lr_ratio` and `neftune_noise_alpha` fields; scheduler types and learning rate trait bounds refined; `TrainingCallback` trait extended with blanket impls for boxed callbacks
- **Data pipeline**: Tokenizer, packing, `vocab_compact`, dataset, and chat template modules updated — minor correctness and efficiency fixes accumulated across the release cycle
- **GGUF reader / writer / quantize**: Reader handles additional tensor metadata fields; writer improves alignment padding; quantize module uses `QuantizeMethod` enum instead of string matching
- **Hub search**: `search_models` returns richer result structs used by both the human-readable table and the new `--json` output path; upload path fixes for large model shards
- **Metal kernels**: GDN, LoRA, grouped GEMM, and fused SwiGLU Metal shaders updated — improved numerical correctness and register pressure
- **GUI app icons and Tauri config**: Updated icons (32×32, 128×128, 128×128@2x, icns, ico) and `tauri.conf.json` for the 0.3.7 release build; Python vocoder `easy` API additions and mel spectrogram fix

## [0.3.6] - 2026-03-15

### Added

- **Desktop GUI (Tauri + Svelte)**: Full desktop application for model management, training, distillation, GRPO, inference, merging, and quantization. 10 pages: Dashboard, Models, Datasets, Training, Distillation, GRPO, Inference, Merging, Quantize, Settings. Real-time training metrics with live loss charts via broadcast events. Model download with HuggingFace Hub integration, dataset browser, and inference chat interface with streaming token display
- **GUI in-process execution**: Training, distillation, GRPO, inference, model merging, LoRA fuse, and quantization run as direct library calls instead of shelling out to the `pmetal` binary. Eliminates binary discovery issues, reduces process overhead, and enables richer progress reporting. Device info and model metadata also read from library APIs
- **`easy::dpo()` / `easy::simpo()` / `easy::orpo()` / `easy::kto()` builders**: `PreferenceTuneBuilder` in `easy.rs` for preference optimization methods. Full pipeline: model download → tokenizer → dataset loading → LoRA setup → training loop → weight saving. Supports method-specific config (DPO beta/loss type, SimPO gamma/CPO, ORPO beta, KTO desirable/undesirable weights)
- **`easy::infer().generate_streaming()`**: Streaming inference API with per-delta callback. Supports both base models and LoRA adapters. Returns `false` from callback to cancel early. ANE fallback emits full result as single delta
- **Preference trainer `train()` methods**: DPO, KTO, ORPO, and SimPO trainers now have self-contained `train()` methods with optimizer integration, batching, epoch loops, callback lifecycle, and metrics collection. Previously only exposed per-step primitives
- **`TrainingCallback::should_stop()`**: Clean cancellation mechanism — callbacks return `true` to request training loop to finish the current step and exit with `Cancelled` error. Checked after every step in all 5 `TrainingLoop::run*` methods, all 4 preference trainer `train()` loops, and `GrpoTrainer::run()`
- **`PMetalError::Cancelled`**: New error variant for clean training cancellation. Corresponding `Cancelled` variants added to `SftError`, `DpoError`, `KtoError`, `OrpoError`, `SimpoError`, and `GrpoError`
- **Preference batch padding utilities**: `pad_u32_sequences`, `pad_i64_sequences`, `pad_f32_sequences` in `preference_batch.rs` for batching variable-length preference pairs
- **NemotronH runtime FP8 quantization**: `quantize_fp8()` converts float weights to FP8 (E4M3) at runtime for all four block types (Mamba, attention, MLP, MoE). Shared helpers `materialize_linear_weight` and `linear_forward_with_optional_fp8` consolidate FP8 dequantization across the model. MoE weights are restacked after quantization for batched dispatch
- **FluxPipeline::from_pretrained**: Load Flux diffusion pipelines from HuggingFace-style model directories. Discovers components via `model_index.json`, parses both native and diffusers-style config keys for CLIP, T5, FluxDiT, and VAE
- **Python training callbacks**: `Trainer.add_callback()` now wires callbacks into the training loop. Built-in `ProgressCallback`, `LoggingCallback`, and `MetricsJsonCallback` map to native Rust implementations; arbitrary Python objects bridge through `PythonCallbackBridge`

### Fixed

- **Training cancellation via `panic_any` replaced**: GUI and TUI previously used `std::panic::panic_any(CancelledRun)` + `catch_unwind` to abort training — fragile, UB-prone through FFI, and could be swallowed by intermediate catch_unwind. Replaced with `TrainingCallback::should_stop()` returning a clean `Err(Cancelled)` from the training loop
- **GUI QLoRA silently failed on non-Llama models**: `run_qlora_training_in_process` hardcoded `LlamaConfig` deserialization, causing confusing errors or silent misconfiguration for Gemma/Qwen/Phi models. Now detects `model_type` from config.json and returns a clear error for unsupported architectures
- **GUI `resume_from` silently ignored**: Training config accepted `resume_from` but discarded it (`let _ = eval`). Now returns an error directing users to the CLI
- **GUI GRPO with no reward function produced noise**: `DummyReward` returning constant 0.1 for all completions made GRPO training meaningless when reasoning rewards were disabled. Now requires explicit reward configuration
- **Preference trainers doubled compute per step**: DPO, KTO, ORPO, and SimPO `train()` methods ran a second full forward pass after the gradient step solely for logging metrics. Replaced with `RefCell` side-channels that capture metric arrays from within the autograd closure — same metrics, zero extra compute
- **Base model thinking mode**: Auto-detect base vs instruct models and disable `<think>` tag prefill for base models. Base models don't understand thinking tags, causing infinite generation without a closing tag
- **Fused model 5x slower than LoRA**: Skip ANE-hybrid path for models under 2B parameters where GPU KV-cache decode is significantly faster (115 vs 20 tok/s). ANE-hybrid benefits larger models where prefill dominates
- **DataLoader panics on bad images**: Replace `panic!()` in VLM batch construction with proper `DataLoaderError` enum and `try_next_batch()` method. Image preprocessing failures and missing-image errors now propagate as `Result` instead of crashing
- **Division by zero with log_every=0**: Clamp `log_every` and `save_every` to minimum 1 across `TrainingLoop`, `LoggingCallback`, `CheckpointCallback`, and CLI
- **LoRA scaling with rank 0**: `LoraConfig::scaling()` returns 0.0 when rank is 0 instead of dividing by zero
- **BF16 LoRA weights**: `sanitize_loaded_weights()` converts BF16 tensors to FP16 since MLX doesn't natively support BF16 on Apple Silicon
- **Qwen3Next silent weight mismatch**: Weight loading now returns errors for unmatched or missing parameters instead of logging a warning and continuing with a partially loaded model
- **Dataset download only fetched README**: `download_dataset()` now enumerates repo files and downloads actual data files (parquet, json, jsonl, csv, arrow, etc.) with split-aware filtering
- **Model download silent failures**: `download_model()` tracks per-file failures and reports them instead of silently skipping failed downloads
- **Flux loading via DynamicModel**: `DynamicModel::load()` for Flux now returns an error directing to `FluxPipeline` instead of incorrectly loading a diffusion model as a causal LM

### Changed

- **GUI architecture: library calls replace subprocess spawning**: Training, distillation, GRPO, inference, merge, fuse, and quantize commands now call `pmetal` library functions directly instead of spawning `pmetal` CLI as a child process. System info reads from `MetalContext::global()` instead of parsing `pmetal memory` stdout. Removes `which` and `futures-util` dependencies
- **TUI direct training execution**: `command_runner.rs` dispatches `train`, `distill`, and `grpo` commands as in-process library calls via `run_direct_command()`, falling back to subprocess for other commands. Training parameters parsed from `CommandSpec` args with `parse_arg`/`required_arg`/`optional_arg` helpers
- **ORPO loss computation refactored**: `compute_orpo_loss_static` now contains the full computation directly instead of creating a throwaway `OrpoTrainer` instance. The instance method `compute_orpo_loss` delegates to it
- **SimPO gradient-safe loss path**: New `compute_loss_with_cpo_for_grad` static method keeps the computation graph lazy (no `.eval()`/`.item()` calls) for correct autograd. The existing `compute_loss_with_cpo` remains for non-grad contexts
- **`FinetuneBuilder` expanded**: New builder methods — `lora_dropout()`, `use_rslora()`, `use_dora()`, `gradient_checkpointing_layers()`, `callback()`, `metrics_path()`. LoRA config now forwards dropout, RSLoRA, and DoRA settings
- **GRPO CLI gains new parameters**: `epochs`, `lora_r`, `lora_alpha`, `max_completion_length` exposed as CLI arguments and TUI form fields. GRPO now saves `adapter_config.json` alongside LoRA weights
- **CLI `emit_console_output` flag**: Training, distillation, and GRPO CLI functions accept `emit_console_output: bool` and `extra_callbacks: Vec<Box<dyn TrainingCallback>>` to suppress terminal output when called from GUI/TUI
- **DataLoader error handling**: New `DataLoaderError` enum with `Mlx`, `ImagePreprocess`, and `MissingImages` variants. All 7 training loop entry points migrated from `next_batch()` to `try_next_batch()`
- **AdapterManager validation**: `load()` now validates path existence, checks for adapter artifacts in directories, and rejects unsupported file types
- **Metal shader build isolation**: Shader compiler cache redirected to build output directory, preventing pollution of user's home directory
- **unsafe_code lint scoping**: Moved blanket `#![allow(unsafe_code)]` from crate-level `lib.rs` into individual modules that contain unsafe blocks across pmetal-metal, pmetal-mlx, pmetal-models, pmetal-trainer, pmetal-distill, and pmetal-distributed

## [0.3.5] - 2026-03-15

### Added

- **Tool/function calling support**: Chat templates now support tool definitions and tool call formatting for models that natively support function calling:
  - **Qwen/ChatML**: `<tools>` schema injection, `<tool_call>`/`<tool_response>` tags, consecutive tool message merging
  - **Llama 3.1+/4**: `Environment: ipython` header, JSON function calls, `ipython` role for tool responses
  - **Mistral v3+**: `[AVAILABLE_TOOLS]`/`[TOOL_CALLS]`/`[TOOL_RESULTS]` bracketed format
  - **DeepSeek**: Qwen-style tool tags with DeepSeek's unicode tokens
  - CLI: `pmetal infer --tools tools.json -p "What's the weather?"` accepts OpenAI-format tool definitions
- **Tool calling types**: `ToolDefinition`, `ToolCall`, `FunctionCall`, `FunctionDefinition` — OpenAI-compatible structs with serde support for JSON parsing
- **`Message` tool fields**: `tool_calls: Option<Vec<ToolCall>>` for assistant messages, `tool_call_id: Option<String>` for tool response messages, `Message::tool()` and `Message::assistant_tool_calls()` constructors
- **`ChatTemplate::apply_with_tools()`**: New method accepting optional `&[ToolDefinition]` — injects tools into system prompts using model-native format

### Fixed

- **Premature early stop during LoRA training**: The adaptive LR controller was falsely detecting "divergence" from the normal LoRA initialization loss rise (LoRA B starts at zero → first 5-10% of steps naturally increase loss). This triggered rollback cycles that exhausted `max_rollbacks` and killed training at ~5% progress. Fixed with three changes:
  - **Grace period** (`warmup_fraction: 0.1`): No spike/plateau/divergence detection fires during the first 10% of training steps. EMA and loss window still accumulate during this period so detection is primed when it activates
  - **Rollback disabled by default** (`rollback_enabled: false`): Weight rollback undoes valid LoRA weight updates and causes the same initialization pattern to repeat. Now opt-in for long pre-training runs
  - **Less sensitive thresholds**: `divergence_slope_threshold` 0.01 → 0.05, `divergence_window` 20 → 40, `plateau_patience` 50 → 100, `spike_threshold` 3.0 → 3.5
- **Adaptive LR grace period not applied**: `set_total_steps()` is now called by all 7 training entry points (5 in `TrainingLoop`, 1 in `GrpoTrainer`, 1 in `DistillationTrainer`) to compute the grace period from total steps

### Changed

- **Adaptive LR defaults**: Retuned for LoRA fine-tuning rather than pre-training. The controller now acts as a safety net (catches NaN, true catastrophic divergence) rather than an aggressive optimizer
- **Distillation adaptive LR**: `for_distillation()` config uses shorter 5% grace period (distillation has smoother early loss) and tighter divergence thresholds

## [0.3.4] - 2026-03-14

### Added

- **Mixture-of-Depths (MoD)** for Llama 4: Proper implementation per Raposo et al. (2024) — lightweight router with `argpartition_axis` top-k, gather-before-compute on sub-batch, scatter-after, BCE auxiliary loss. Configurable capacity factor and per-layer selection
- **Llama 4 RoPE**: Real RoPE implementation via `pmetal_mlx::kernels::rope::apply_rope` (Metal-accelerated), replacing the placeholder stub. Correctly wired into iRoPE layer dispatch — RoPE layers get rotary embeddings, NoPE layers skip them
- **Llama 4 temperature scaling**: Per Meta's formula `log(floor((pos+1)/floor_scale) + 1) * attn_scale + 1.0`, applied to Q states in NoPE layers before QK matmul for long-context attention stabilization
- **Llama 4 GQA**: KV-head broadcast expansion for grouped-query attention — enables Scout (40 Q / 8 KV) and Maverick configs
- **MoE top-k > 1**: `Llama4Router` uses `argpartition_axis` for O(n) expert selection with L1-normalized weights and per-slot dispatch loop, replacing hardcoded argmax
- **ANE fused kernels**: `gen_dynamic_sdpa_fwd` (single-kernel attention: RMSNorm + QKV + SDPA + Wo) and `gen_dynamic_ffn_w13` (single-kernel FFN: RMSNorm + W1 + W3 + SiLU), replacing 6+ separate ANE evaluations per layer
- **ANE fused backward**: `gen_dynamic_ffn_bwd_w2t` and `gen_dynamic_ffn_bwd_w13t` for fused FFN backward pass
- **Metal dequantization kernels**: Q4_0 and IQ4_XS Metal compute shaders, verified correct per GGML spec. Bridge methods in `MlxMetalBridge` for GPU-accelerated dequantization
- **Cancellation safety infrastructure**: `CompletionToken::Drop` guard in `AsyncScheduler` waits for in-flight GPU commands; `retain_resource()` / `as_retained()` for Metal buffer lifetime extension
- **IoSurface helpers**: `write_f32_strided_at`, `write_f32_at_col_offset`, `zero_channel_range_f32` for fused backward kernel IO
- **CloudBridge**: Complete training state export (weights, optimizer state, RNG, dataloader position, metadata) with working Python bootstrap scripts for FSDP/DeepSpeed cluster resumption and Rust-side loader functions
- **Formal verification**: `cargo-kani` proofs for ring all-reduce chunk arithmetic (95 checks) and k-ary tree topology consistency (607 checks), with justfile recipes
- **Reasoning templates**: `MathReasoningTemplate` (GRPO + accuracy/format rewards) and `CodeReasoningTemplate` (structural code fence + test case matching)
- **Reasoning dataset auto-detection**: `pmetal dataset prepare` automatically detects `problem`/`thinking`/`solution` columns and formats them as `<think>` tagged ChatML conversations
- **`--columns` flag**: General column remapping for `dataset prepare` (e.g., `--columns "instruction=question,output=answer"`)
- **`adapter_config.json`**: Saved alongside LoRA weights during training (r, alpha, target_modules, use_rslora). Loaded automatically at inference and fuse time — eliminates config guesswork
- **Supply chain**: `cargo-vet` initialized with Mozilla, Google, and Bytecode Alliance audit imports; 17 workspace crates covered; 5 transitive dependency exemptions with exact lockfile versions
- **Tracing spans**: 6 `info_span!` markers in Python trainer for phase-level observability (model_resolve, load_tokenizer, load_dataset, load_model, training_loop, save_weights)

### Fixed

- **LoRA inference garbage output**: Merged LoRA weights into base model at inference time (`W += scale*B@A`), matching mlx-lm's pattern. The separate-forward path had dtype mismatch issues (BF16 base × F32 LoRA)
- **Auto-chat mode regression**: Removed heuristic that forced chat template on base models just because their tokenizer has `<|im_end|>`. Chat mode now requires explicit `--chat` or an instruction-tuned model
- **Missing EOS in training data**: Training sequences now end with the model's actual EOS token (e.g., `<|endoftext|>` for Qwen). Previously only had turn delimiter (`<|im_end|>`) — model never learned to stop generating
- **Fuse command wrong alpha/rank**: `pmetal fuse` now reads `adapter_config.json` for correct alpha and rank instead of defaulting to `scale=1.0`. Also filters MLP LoRA weights (rank=0) when auto-detecting rank from shapes
- **ANE `x2norm` backward bug**: FFN weight gradients (`dW1`, `dW3`) were computed against the wrong pre-norm tensor (`xnorm` from attention block instead of `x2norm` from FFN block). Restored `x2norm` field and CPU RMSNorm recomputation for gradient correctness
- **ANE `sdpa_bwd` surface dtype**: Backward SDPA output surfaces were allocated as fp32 but ANE kernels produce fp16 — stride mismatch corrupted dV/dQ/dK gradients. Fixed to `IoSurface::for_tensor()` (fp16)
- **MoD argpartition sign**: Router negated weights before `argpartition_axis`, selecting bottom-k (least important) tokens instead of top-k. Removed negation
- **MLX bridge `copy_as_f32` regression**: Renamed methods dropped auto dtype conversion — callers passing wrong dtype would panic. Restored `copy_as_f32` / `copy_as_f16` with auto-conversion
- **MLX bridge `view_f32` eval**: Removed `.eval()` call before accessing data pointer — unevaluated arrays returned null. Restored defensive eval
- **Python API surface**: Restored `ProgressCallback`, `LoggingCallback(log_every=10)`, `__version__`, and `PythonCallbackBridge` that were deleted during PyO3 migration
- **TUI training completion**: Reads final metrics from JSONL file on disk (immune to polling lag). Shows actual loss and step count instead of `0.0000` / sample count
- **TUI Steps/min overflow**: Guards against divide-by-zero when `total_ms=0` — shows `—` instead of `60000`
- **Dataset prepare panic**: Empty results no longer crash with index-out-of-bounds. Shows diagnostic message with format hints

### Changed

- **LoRA inference uses merge**: `merge_lora()` is called before generation, producing a single merged weight matrix per layer. This is equivalent to the fuse command but happens in-memory without saving
- **PyO3 0.23 → 0.28**: `allow_threads` → `detach`, `with_gil` → `attach`, `from_py_object` on all pyclass types, `Bound<'py, PyDict>` return types
- **tokio 1.49 → 1.50**
- **`unsafe_code` lint**: Escalated from `warn` to `deny` workspace-wide

## [0.3.3] - 2026-03-12

### Added

- **Self-contained binary**: `mlx.metallib` is now gzip-compressed and embedded into the `pmetal` binary at build time via `build.rs` + `include_bytes!`. On first run it extracts to `~/.cache/pmetal/lib/` if not already present. `cargo install pmetal-cli` now produces a fully self-contained binary with no external metallib dependency (~31MB added to binary, 70% smaller than the raw 102MB metallib)
- **Adaptive LR rollback**: When divergence is detected and `rollback_enabled = true`, the adaptive LR controller emits `LrEvent::RollbackTriggered` — the training loop restores LoRA weights from the best in-memory EMA snapshot, resets optimizer momentum, and continues with a halved LR multiplier
- **Early-stop on repeated divergence**: After `max_rollbacks` exhausted rollbacks, the controller emits `LrEvent::EarlyStop` — the training loop saves a final checkpoint and exits cleanly instead of spiraling deeper into loss divergence
- **In-memory LoRA snapshot**: `TrainingLoop` holds the best LoRA weight snapshot in RAM via `snapshot_best_weights()` / `restore_best_weights()`. LoRA params are typically 1–20 MB, making this negligible overhead vs checkpoint I/O
- **`AdaptiveAction` enum**: `apply_adaptive_lr()` now returns `AdaptiveAction::Continue | Rollback | EarlyStop` so training loops can react to controller decisions without re-parsing event strings

### Fixed

- **`apply_adaptive_lr` return type**: Previously returned `()`, discarding rollback/early-stop events — callers had no way to react. Now returns `AdaptiveAction`
- **Divergence rollback vs plain reduction ambiguity**: Divergence path now checks `rollback_enabled` and `has_best_snapshot` before deciding between rollback and plain LR reduction — prevents silent rollback when no snapshot exists
- **EMA state reset on rollback**: Spike EMA and variance are reset alongside LR multiplier on rollback so z-score anomaly detection re-stabilizes correctly after weight restoration
- **`total_steps` in metrics**: `run_standard()` and `run_jit_compiled()` computed `total_steps: max_steps.unwrap_or(0)` — now estimates from `dataset.len() / batch_size * epochs` when `max_steps` is `None`, giving accurate progress in the TUI
- **`stats_summary` missing rollback count**: `AdaptiveLrController::stats_summary()` now includes `rollbacks=N` in its output string

### Improved

- **Rollback tests**: Four new unit tests — `test_rollback_triggered_on_divergence`, `test_early_stop_after_max_rollbacks`, `test_rollback_disabled_falls_through_to_divergence`, `test_should_snapshot_best_tracks_ema_improvement`

## [0.3.2] - 2026-03-11

### Added

- **Adaptive learning rate controller**: EMA-based z-score spike detection, patience-based plateau detection, and linear regression divergence detection — automatically adjusts LR multiplier during training to recover from loss spikes, reduce LR on plateaus, and halt on divergence
- **Manual LR override via TUI**: Press `L` in Training, Distillation, or GRPO tabs to set a custom learning rate mid-run; uses atomic control file protocol (`{output_dir}/.lr_control.json`) for safe subprocess communication
- **WSD (Warmup-Stable-Decay) scheduler**: New `LrSchedulerType::Wsd` with configurable `stable_ratio` — holds peak LR for a plateau phase before linear decay, popular for large-scale pretraining
- **GRPO adaptive LR + callbacks**: `GrpoTrainer` now supports adaptive LR, `TrainingCallback` lifecycle events, and `StepMetrics` emission for live TUI monitoring
- **HuggingFace Hub search** (`pmetal search`): CLI command and TUI integration (press `S` in Models tab) to search HF Hub for text-generation models with download counts, parameter estimates, and memory fit assessment
- **Memory fit estimation**: New `pmetal-hub` module estimates inference/training memory requirements, tok/s throughput, and color-coded fit levels (green/yellow/red) based on device specs and model architecture
- **Model detail panel**: Models tab shows memory breakdown — weights, KV cache, overhead, training estimate, and recommended batch size
- **Distillation metrics callbacks**: `DistillationTrainer` now emits step-by-step metrics via `TrainingCallback`, enabling live TUI dashboard during distillation runs
- **Command logging in Jobs tab**: Spawned commands are logged with the full CLI invocation for easier debugging

### Fixed

- **NaN/Inf loss guard**: Adaptive LR skips EMA updates on non-finite losses to prevent EMA poisoning — returns scheduled LR unchanged
- **EMA variance bias correction**: Early-training z-scores now use bias-corrected variance (`raw_var / (1 - alpha^n)`), matching Adam's moment correction — prevents false spike detection in first ~20 steps
- **Zero-variance z-score fallback**: When loss variance is near zero (std_dev < 1e-8), uses absolute deviation threshold instead of division-by-zero; returns z=10 for >50% deviation, z=0 otherwise
- **Atomic control file protocol**: LR control file is renamed to `.lr_control.claimed` before reading and deleted after — prevents race conditions between TUI writer and training subprocess reader
- **Distillation metrics LR**: Distillation step metrics now report post-adaptive LR instead of pre-adjustment scheduled LR
- **Adaptive LR in all training paths**: `apply_adaptive_lr()` now called in `run_metal_fused()`, `run_compiled()`, `run_jit_compiled()`, and `run_packed()` paths (was only in `run_standard()`)
- **TUI LR override validation**: LR range check now accepts 1.0 (was exclusive upper bound); shows error modal on invalid input instead of silent log warning
- **Distillation/GRPO job routing**: Status updates were always routed to the Training tab regardless of job type. Added `active_job_type` tracking to route metrics, completion, and failure to the correct tab (Distill, GRPO, or Training)
- **Distillation CLI args**: TUI sent `--lora-alpha` and `--log-metrics` flags that the CLI didn't accept, causing immediate exit code 2. Added both args to the `Distill` command and `--log-metrics` to `Grpo`
- **Parquet dataset support in distill/GRPO**: Distillation and GRPO commands only supported JSONL datasets. Now auto-detect `.parquet` files and route to the parquet loader, matching the training command's behavior
- **Tab click targeting**: Mouse clicks on Monitor, Inference, and Jobs tabs selected the wrong tab due to hardcoded fixed-width hit-testing. Now computes actual tab widths from rendered text
- **Error diagnostics**: Failed jobs now show the last 5 stderr lines in the tab status panel instead of just "Process exited with code N", with a hint to check the Jobs tab for full output
- **UTF-8 safe string truncation**: `truncate_str` used byte indexing which panics on multi-byte characters; switched to `chars()` iterator
- **Leaked channel in HF search**: `search_hf()` created a sender/receiver pair even without a CommandRunner, silently dropping results
- **Integer overflow in fit estimation**: `estimate_params_from_config` used plain multiplication; switched to `saturating_mul`/`saturating_add`
- **Context length truncation**: u64→u32 cast could wrap for extreme values; capped at 1M before cast

### Improved

- **TUI tab ordering**: System (formerly Device) is now the default first tab; Dashboard renamed to Monitor
- **Empty state messaging**: Monitor tab shows actionable guidance ("Start a run from Training, Distill, or GRPO tab") instead of "Waiting for training data..."
- **Idle state hint**: Tabs show "Press S to start" instead of "Press S to start training" (generic across all job types)

### Security

- **Bounded API responses**: `bounded_json()` caps HF API response bodies at 4MB to prevent heap exhaustion
- **Model ID validation**: `is_valid_model_id()` rejects path traversal, URL injection, and malformed values in HF API paths

## [0.3.1] - 2026-03-11

### Added

- **M5 / Apple10 device detection**: GPU family `Apple10` with architecture generation 17, NAX (Neural Accelerators in GPU) availability flag, and NAX-aware tile size tuning (M5 Max/Ultra get 128×64×32)
- **UltraFusion topology detection**: `sysctl hw.packages` detects multi-die Ultra chips; `is_ultra_fusion` and `die_count` fields on `DeviceProperties`
- **GPU and ANE core count estimation**: Per-chip core counts derived from device name and tier, with UltraFusion die multiplication
- **Memory bandwidth estimation**: Tier + GPU family lookup table for estimated bandwidth (GB/s)
- **ANE performance stats API**: `evaluate_with_stats()` on `AneModel` uses `_ANEPerformanceStats` with `hwExecutionTime` for nanosecond-precision hardware timing
- **TUI device tab enhancements**: GPU core counts (with per-die breakdown for Ultra), ANE core counts, memory bandwidth, architecture generation, NAX and UltraFusion feature flags
- **`crates/pmetal/README.md`**: Crate-level README with feature flags table, quick start examples, hardware support summary, and re-export reference

### Fixed

- **`AppleGPUFamily::Unknown` ordering bug**: `Unknown` was declared last in the enum, causing derived `Ord` to rank it above `Apple10` — unknown GPUs incorrectly got `has_dynamic_caching`, `has_nax`, etc. set to `true`. Fixed by moving `Unknown` to first position
- **Future chip name collision**: `name.contains("M1")` matched "M10"; replaced with `has_chip_id()` that checks the character after the match isn't a digit
- **Dead `sysctl` subprocess in `query_memory_bandwidth`**: Spawned `sysctl` whose result was discarded; removed and renamed to `estimate_memory_bandwidth()` using tier-based lookup

### Improved

- **README updates**: Root README now documents hardware support matrix (M1–M5), 9 TUI tabs (was 7), 16 crates (was 15), all fused Metal kernels (GDN, SwiGLU, RMSNorm+LoRA), ANE perf stats and M1–M5 compatibility
- **Hardware support docs**: Complete M1–M5 chip matrix with arch gen, core counts, bandwidth, ANE TFLOPS measurements; NAX kernel integration roadmap; UltraFusion distributed roadmap

## [0.3.0] - 2026-03-10

### Added

- **TUI Control Center** (`pmetal tui`): Full terminal interface with 9 tabs — Dashboard, Device, Models, Datasets, Training, Distillation, GRPO, Inference, Jobs. Async event loop with crossterm/ratatui, modal system (confirm, text input, model picker, dataset picker, error, progress), and reusable form field widgets
- **Live job integration**: Training, distillation, and GRPO tabs spawn pmetal subprocesses and stream metrics in real time via `CommandRunner` + JSONL polling
- **LoRA fuse command** (`pmetal fuse`): Merge LoRA adapter weights into base model, with optional fuse-then-quantize pipeline
- **Chat template support for Llama 4, DeepSeek, and Cohere**: Full template formatting, Jinja detection, model name heuristics, stop tokens, and inference formatting for all three model families
- **Llama 4 template**: `<|header_start|>`/`<|header_end|>`/`<|eot|>` tokens (distinct from Llama 3's `<|start_header_id|>`/`<|end_header_id|>`/`<|eot_id|>`)
- **DeepSeek template**: Full-width unicode tokens (`<｜begin▁of▁sentence｜>`, `<｜User｜>`, `<｜Assistant｜>`) with thinking mode support (`<think>`/`</think>` prefill)
- **Cohere Command R template**: `<|START_OF_TURN_TOKEN|>`, `<|USER_TOKEN|>`, `<|CHATBOT_TOKEN|>`, `<|END_OF_TURN_TOKEN|>` tokens
- **Comprehensive stop token collection**: `collect_all_stop_tokens()` now probes 11 well-known special tokens across all model families (added `<|eot|>`, `<|end|>`, `<|return|>`, `<|END_OF_TURN_TOKEN|>`, `<｜end▁of▁sentence｜>`)
- **LoRA inference auto-chat detection**: Probes vocabulary for `<|im_end|>`/`<|eot_id|>` to auto-enable chat mode on base models fine-tuned with LoRA
- **Streaming generation support**: `GenerationConfig` streaming extensions in `pmetal-models`
- **Epoch/total_steps in StepMetrics**: Training progress now flows through entire pipeline (training loop → JSONL callback → TUI) showing step X/Y and epoch M/N
- **Hardware support documentation**: Apple Silicon hardware matrix and tuning reference (`docs/hardware-support.md`)

### Fixed

- **TUI inference word wrap**: Model output now wraps correctly within the terminal width instead of clipping off-screen; `normalize_code_fences()` preprocessor ensures ``` markers always appear on their own line even when the model emits text without newlines
- **TUI inference code block rendering**: Fenced code blocks (```python, etc.) now render properly with distinct styling even when the token stream lacks explicit newline characters
- **TUI UTF-8 safe text handling**: Word wrap and code block truncation now use char-count width instead of byte length, preventing panics on multi-byte characters
- **GRPO accuracy reward — last-occurrence extraction**: `AccuracyReward` now uses `rfind()` for `<answer>` tags and `\boxed{}`, correctly grabbing the final answer when the model retries within chain-of-thought
- **GRPO accuracy reward — broken fallback**: Old code compared the entire completion (including reasoning) against the answer when no `<answer>` tags were found; now falls back to last non-empty line
- **GRPO accuracy reward — whitespace normalization**: Answer comparison now collapses internal whitespace runs to single space, preventing false negatives from formatting differences
- **LoRA inference stop tokens**: `run_inference_with_lora` now uses full chat template + comprehensive stop token collection instead of just tokenizer EOS — fixes infinite generation on chat-finetuned models
- **LoRA inference missing parameters**: All sampling parameters (top_k, top_p, min_p, penalties, seed) now passed through to LoRA inference path
- **Llama 4 misdetection**: Model name heuristic now correctly routes `llama-4`/`llama4` to Llama 4 template (was incorrectly using Llama 3 tokens)

### Added

- **GRPO `\boxed{}` answer extraction**: `AccuracyReward` now extracts answers from LaTeX `\boxed{...}` expressions with brace-depth tracking, standard for math GRPO (DeepSeek-R1 style)

### Improved

- **TUI replaces legacy dashboard**: `pmetal tui` provides full control center; legacy `pmetal dashboard` retained for simple metrics monitoring
- **Chat template Jinja detection**: Ordered detection ensures DeepSeek (full-width unicode), Cohere, Llama 4 are matched before generic patterns
- **EOS token stripping**: `strip_eos_tokens()` now handles all model-family EOS tokens

## [0.2.1] - 2026-03-09

### Added

- **Cross-vocabulary distillation**: Sparse top-k alignment (k=128) enables teacher/student with different vocab sizes; implemented in KL divergence, soft cross-entropy, and Jensen-Shannon losses
- **Fused GDN Metal kernel**: Gated Delta Network forward pass for Qwen 3.5 hybrid layers (`fused_gdn.metal` + `fused_gdn.rs`)
- **Gated delta MLX kernel**: Forward and backward passes for GDN in `pmetal-mlx`
- **CPU RMSNorm for ANE inference**: Compute RMSNorm on CPU in f32 to avoid fp16 overflow/saturation on ANE; per-head QK-norm stays on ANE where values are safe
- **`cpu_rmsnorm` flag in kernel generators**: `gen_sdpa_fwd_kv()` and `gen_ffn_fwd()` accept `cpu_rmsnorm: bool` — when true, emits identity instead of RMSNorm and omits weight blobs
- **Test serialization config**: `.cargo/config.toml` sets `RUST_TEST_THREADS=1` to prevent MLX GPU memory races

### Fixed

- **ANE inference garbage output**: fp16 `reduce_sum(x², axis=channel)` overflows for residual values > 256 due to ANE saturation arithmetic; CPU RMSNorm in f32 eliminates the corruption
- **Cross-vocab distillation crash**: Mismatched teacher/student vocab sizes (e.g., Qwen3-4B 151,936 → Qwen3.5-0.8B 152,080) no longer panic; `align_vocab()` handles alignment transparently
- **3D tensor indexing in `align_vocab`**: Use `(Ellipsis, ..k)` for correct last-axis slicing of rank-3+ tensors
- **Qwen 3.5 (1+w) RMSNorm**: Weight sanitization adds 1.0 to RMSNorm weights during loading
- **Clippy lints**: Unnecessary parentheses in `fused_gdn.rs`, too-many-arguments on `rmsnorm_backward`, let-and-return in `next_power_of_2`

### Improved

- **ANE inference cleanup**: Removed ~80 lines of diagnostic logging from hot path
- **Metal GPU path gating**: Cross-vocab losses gate Metal GPU path on matching vocabs, fall back to CPU for mismatched
- **Documentation**: Updated all crate READMEs to reflect current architecture support, training methods, and features

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
  - Fused cross-entropy (chunked vocabulary loss)
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
