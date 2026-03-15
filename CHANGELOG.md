# Changelog

All notable changes to PMetal will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
