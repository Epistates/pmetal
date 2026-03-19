[![Crates.io](https://img.shields.io/crates/v/pmetal.svg)](https://crates.io/crates/pmetal)
[![Rust](https://img.shields.io/badge/rust-1.86+-orange.svg)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-macOS-lightgrey.svg)](https://www.apple.com/macos)

# PMetal

**Powdered Metal** — An ML SDK, framework, and application suite for Apple Silicon, written in Rust.

PMetal is a complete machine learning platform for Apple Silicon — from low-level Metal GPU kernels and Apple Neural Engine integration to high-level training APIs, a terminal TUI, and a full desktop GUI. Ship fine-tuned models without leaving the Apple ecosystem.

## Use PMetal Your Way

### Desktop GUI

<img src="public/pmetal_gui.png" alt="pmetal screenshot showing GUI" style="width: 100%; max-width: 100%; margin: 20px 0;"/>

A full Tauri + Svelte desktop application for visual model management, training, and inference.

```bash
cd crates/pmetal-gui
bun install && bun tauri dev
```

10 pages: Dashboard, Models, Datasets, Training, Distillation, GRPO, Inference, Merging, Quantize, and Settings. Download models from HuggingFace, configure LoRA training with live loss metrics, chat with models, merge weights, and quantize — all from the GUI. Training runs in-process with real-time progress updates.

### Terminal TUI

<img src="public/pmetal_tui.png" alt="pmetal screenshot showing TUI" style="width: 100%; max-width: 100%; margin: 20px 0;"/>

A full-featured terminal control center with 9 tabs.

```bash
pmetal tui
```

| Tab | Description |
|-----|-------------|
| **Dashboard** | Live loss curves (braille), LR schedule, throughput sparklines, timing breakdown gauges |
| **Device** | GPU/ANE info, Metal feature detection, memory gauge, kernel tuning, UltraFusion topology |
| **Models** | Browse cached models, HuggingFace Hub search (`S`), memory fit estimation, download |
| **Datasets** | Scan and preview local datasets (JSONL, Parquet, CSV) with line counts |
| **Training** | Configure and launch SFT/LoRA/QLoRA training runs with sectioned parameter forms |
| **Distillation** | Configure knowledge distillation (online, offline, progressive, cross-vocab) |
| **GRPO** | Configure GRPO/DAPO reasoning training with reward functions and sampling params |
| **Inference** | Interactive chat interface with markdown rendering and generation settings sidebar |
| **Jobs** | Training run history with log viewer, status tracking, and metadata |

Keybindings: `Tab`/`Shift+Tab` to switch tabs, `Alt+1-9` for direct access, `L` to adjust learning rate mid-run, `q` to quit.

### CLI

```bash
# LoRA fine-tuning with sequence packing (default)
pmetal train \
  --model Qwen/Qwen3-0.6B \
  --dataset train.jsonl \
  --output ./output \
  --lora-r 16 --batch-size 4 --learning-rate 2e-4

# Inference with LoRA adapter
pmetal infer \
  --model Qwen/Qwen3-0.6B \
  --lora ./output/lora_weights.safetensors \
  --prompt "Explain quantum entanglement" \
  --chat --show-thinking

# Knowledge distillation
pmetal distill \
  --teacher Qwen/Qwen3-4B \
  --student unsloth/Qwen3.5-0.8B-Base \
  --dataset train.jsonl

# GRPO reasoning training
pmetal grpo \
  --model Qwen/Qwen3-0.6B \
  --dataset reasoning.jsonl \
  --reasoning-rewards

# HuggingFace model search with memory fit
pmetal search "qwen 0.6b" --detailed

# Merge models with SLERP
pmetal merge \
  --models model-a model-b \
  --method slerp --t 0.5

# Quantize to GGUF
pmetal quantize \
  --model ./output \
  --output model.gguf --type q4km

# Fuse LoRA into base model
pmetal fuse \
  --model Qwen/Qwen3-0.6B \
  --lora ./output/lora_weights.safetensors

# Evaluate perplexity
pmetal eval \
  --model Qwen/Qwen3-0.6B \
  --dataset eval.jsonl

# Start OpenAI-compatible server (requires --features serve)
pmetal serve --model Qwen/Qwen3-0.6B --port 8080
```

#### All CLI Commands

| Command | Description |
|---------|-------------|
| `train` | Fine-tune with LoRA/QLoRA/DoRA (SFT) |
| `infer` | Interactive inference with chat, tool use, and thinking mode |
| `distill` | Knowledge distillation (online, offline, progressive) |
| `grpo` | GRPO/DAPO reasoning training (VLM, speculative, async rewards) |
| `rlkd` | Reinforcement Learning with Knowledge Distillation |
| `embed-train` | Sentence-transformer fine-tuning (InfoNCE, Triplet, CoSENT) |
| `search` | Search HuggingFace Hub with memory fit estimation |
| `download` | Download a model from HuggingFace Hub |
| `merge` | Merge two or more models (12 strategies) |
| `quantize` | GGUF quantization (13 format options) |
| `fuse` | Fuse LoRA adapter weights into base model |
| `eval` | Evaluate model perplexity on a dataset |
| `serve` | OpenAI-compatible inference server (feature-gated) |
| `tui` | Full TUI control center (9 tabs) |
| `dashboard` | Real-time training metrics visualization |
| `dataset` | Dataset utilities: `analyze`, `download`, `convert` |
| `ollama` | Ollama integration: `modelfile`, `create`, `templates` |
| `info` | Show device info (GPU, ANE, bandwidth, NAX) |
| `memory` | Show memory usage and available capacity |
| `init` | Generate a sample configuration file |
| `bench` | Benchmark training performance |
| `bench-gen` | Benchmark generation loop timing |
| `bench-ffi` | Benchmark FFI overhead |

## SDK

PMetal is an embeddable SDK — integrate training, inference, and model operations into your own Rust applications. The `easy` module provides high-level builders, while the underlying crates (`pmetal-trainer`, `pmetal-models`, `pmetal-lora`, etc.) offer full control over every pipeline stage.

```rust
use pmetal::easy;

// Fine-tune with LoRA
let result = easy::finetune("Qwen/Qwen3-0.6B", "train.jsonl")
    .lora(16, 32.0)
    .learning_rate(2e-4)
    .epochs(3)
    .output("./output")
    .run()
    .await?;

// DPO preference optimization
let result = easy::dpo("Qwen/Qwen3-0.6B", "preferences.jsonl")
    .dpo_beta(0.1)
    .reference_model("Qwen/Qwen3-0.6B")
    .run()
    .await?;

// Inference
let output = easy::infer("Qwen/Qwen3-0.6B")
    .temperature(0.7)
    .lora("./output/lora_weights.safetensors")
    .generate("What is 2+2?")
    .await?;

// Streaming inference
easy::infer("Qwen/Qwen3-0.6B")
    .generate_streaming("Tell me a story", |delta| {
        print!("{delta}");
        true // return false to stop early
    })
    .await?;
```

Available builders: `easy::finetune()`, `easy::dpo()`, `easy::simpo()`, `easy::orpo()`, `easy::kto()`, `easy::infer()`.

For lower-level control, use the crates directly — `pmetal-trainer::TrainingLoop`, `pmetal-models::DynamicModel`, `pmetal-lora::DynamicLoraModel`, `pmetal-distill::Distiller`, etc. See the [`examples/`](crates/pmetal/examples/) directory for complete working examples including manual training loop orchestration and ANE-specific workflows.

## Python SDK

PMetal exposes a Python extension module via PyO3. Install with `maturin develop` from `crates/pmetal-py`.

### Quick Start (Easy API)

```python
import pmetal

# Fine-tune with sensible defaults
result = pmetal.finetune(
    "Qwen/Qwen3-0.6B",
    "train.jsonl",
    lora_r=16,
    learning_rate=2e-4,
    epochs=3,
)
print(f"Loss: {result['final_loss']}, Steps: {result['total_steps']}")

# Inference
text = pmetal.infer("Qwen/Qwen3-0.6B", "What is 2+2?")
print(text)

# Inference with LoRA adapter
text = pmetal.infer(
    "Qwen/Qwen3-0.6B",
    "Explain quantum entanglement",
    lora="./output/lora_weights.safetensors",
)
```

### Full Control

```python
import pmetal

# Configure training components
lora_config = pmetal.LoraConfig(r=16, alpha=32.0)
training_config = pmetal.TrainingConfig(
    learning_rate=2e-4,
    num_epochs=3,
    batch_size=4,
    max_seq_len=2048,
)

# Create trainer
trainer = pmetal.Trainer(
    model_id="Qwen/Qwen3-0.6B",
    lora_config=lora_config,
    training_config=training_config,
    dataset_path="train.jsonl",
)
trainer.add_callback(pmetal.ProgressCallback())
result = trainer.train()

# Load model for inference
model = pmetal.Model.load("Qwen/Qwen3-0.6B")
print(model.generate("Hello world", temperature=0.7))
```

## Installation

Prebuilt signed binaries are available on the [Releases](https://github.com/Epistates/pmetal/releases) page.

Crates are available on [crates.io](https://crates.io/crates/pmetal).

Build from source:

```bash
git clone https://github.com/epistates/pmetal.git && cd pmetal
cargo build --release          # CLI + TUI
cd crates/pmetal-gui && bun install && bun tauri build  # GUI (optional)
```

## Hardware Support

PMetal automatically detects Apple Silicon capabilities at startup and tunes kernel parameters accordingly.

| Chip Family | GPU Family | NAX | ANE | UltraFusion | Status |
|-------------|-----------|-----|-----|-------------|--------|
| M1 / Pro / Max / Ultra | Apple7 | - | 16 cores | Ultra: 2-die | Fully supported |
| M2 / Pro / Max / Ultra | Apple8 | - | 16 cores | Ultra: 2-die | Fully supported |
| M3 / Pro / Max / Ultra | Apple9 | - | 16 cores | Ultra: 2-die | Fully supported |
| M4 / Pro / Max / Ultra | Apple9 | - | 16 cores | Ultra: 2-die | Fully supported |
| **M5 / Pro / Max / Ultra** | **Apple10** | **Yes** | **16 cores** | **Ultra: 2-die** | **Fully supported** |

**Auto-detected features**: GPU family, device tier, core counts, memory bandwidth, dynamic caching, mesh shaders, NAX (M5+), UltraFusion topology (via `sysctl hw.packages`), ANE availability.

**Tier-based kernel tuning**: Matrix tile sizes, FlashAttention block sizes, fused kernel threadgroup sizes, and batch multipliers are automatically selected based on device tier (Base/Pro/Max/Ultra) and GPU family. See [`docs/hardware-support.md`](docs/hardware-support.md) for the full tuning matrix.

## Architecture

PMetal is organized as a Rust workspace with 18 specialized crates:

```
pmetal/
├── pmetal-core         # Foundation: configs, traits, types, error handling
├── pmetal-metal        # Custom Metal GPU kernels + ANE runtime
├── pmetal-mlx          # MLX backend integration (KV cache, RoPE, etc.)
├── pmetal-models       # LLM architectures (Llama, Qwen, DeepSeek, etc.)
├── pmetal-lora         # LoRA/QLoRA training implementations
├── pmetal-trainer      # Training loops (SFT, DPO, SimPO, ORPO, KTO, GRPO, etc.)
├── pmetal-data         # Dataset loading, chat templates, tokenization
├── pmetal-hub          # HuggingFace Hub integration + model fit estimation
├── pmetal-distill      # Knowledge distillation (online, offline, cross-vocab, TAID)
├── pmetal-merge        # Model merging (14 strategies)
├── pmetal-gguf         # GGUF format with imatrix quantization
├── pmetal-mhc          # Manifold-Constrained Hyper-Connections
├── pmetal-distributed  # Distributed training (mDNS, Ring All-Reduce)
├── pmetal-vocoder      # BigVGAN neural vocoder
├── pmetal-serve        # OpenAI-compatible inference server
├── pmetal-py           # Python bindings (maturin/PyO3)
├── pmetal-cli          # Command-line interface + TUI control center
└── pmetal-gui          # Desktop GUI (Tauri + Svelte + TailwindCSS)
```

The `pmetal` facade crate re-exports all modules with feature flags and provides the `easy` API for quick-start usage.

## Supported Models

### Inference (via `DynamicModel` dispatcher)

All models below can be loaded from HuggingFace Hub or local safetensors and used for inference via the CLI, TUI, GUI, or SDK.

| Family | Architecture | Variants | `model_type` values |
|--------|-------------|----------|-------------------|
| Llama | `Llama` | 2, 3, 3.1, 3.2, 3.3 | `llama`, `llama3` |
| Llama 4 | `Llama4` | Scout, Maverick | `llama4` |
| Qwen 2 | `Qwen2` | 2, 2.5 | `qwen2`, `qwen2_5` |
| Qwen 3 | `Qwen3` | 3 | `qwen3` |
| Qwen 3 MoE | `Qwen3MoE` | 3-MoE | `qwen3_moe` |
| Qwen 3.5 | `Qwen3Next` | 3.5 (Next) | `qwen3_next`, `qwen3_5` |
| DeepSeek | `DeepSeek` | V3, V3.2, V3.2-Speciale | `deepseek`, `deepseek_v3` |
| Mistral | `Mistral` | 7B, Mixtral 8x7B | `mistral`, `mixtral` |
| Gemma | `Gemma` | 2, 3 | `gemma`, `gemma2`, `gemma3` |
| Phi 3 | `Phi` | 3, 3.5 | `phi`, `phi3` |
| Phi 4 | `Phi4` | 4 | `phi4` |
| Cohere | `Cohere` | Command R | `cohere`, `command_r` |
| Granite | `Granite` | 3.0, 3.1, Hybrid MoE | `granite`, `granitehybrid` |
| NemotronH | `NemotronH` | Hybrid (Mamba+Attention) | `nemotron_h` |
| StarCoder2 | `StarCoder2` | 3B, 7B, 15B | `starcoder2` |
| RecurrentGemma | `RecurrentGemma` | Griffin | `recurrentgemma`, `griffin` |
| Jamba | `Jamba` | 1.5 | `jamba` |
| Flux | `Flux` | 1-dev, 1-schnell | `flux` |

### LoRA/QLoRA Training Support

LoRA training is supported for models that have implementations in `DynamicLoraModel`. Architecture detection is automatic — just point `pmetal train` at a model directory or HuggingFace ID.

| Architecture | LoRA | QLoRA | Notes |
|-------------|------|-------|-------|
| Llama | Yes | Yes | Covers Llama 2, 3, 3.1, 3.2, 3.3. Gradient checkpointing supported. |
| Qwen 2 | Yes | — | Uses Qwen3 LoRA implementation internally. |
| Qwen 3 | Yes | Yes | Gradient checkpointing supported. |
| Qwen 3.5 (Next) | Yes | — | Hybrid architecture with nested `text_config` handling. |
| Gemma | Yes | Yes | GeGLU activation, special RMSNorm. |
| Mistral | Yes | Yes | Sliding window attention support. |
| Phi 3 | Yes | — | Partial RoPE, fused gate_up projection. |

Architectures not listed above (Llama 4, Qwen 3 MoE, DeepSeek, Cohere, Granite, NemotronH, Phi 4, StarCoder2, RecurrentGemma, Jamba) support inference but do not yet have LoRA training integration via `DynamicLoraModel`. Contributions welcome.

### Architecture Modules (Not Yet in Dispatcher)

The following architectures have implementations in `pmetal-models` but are not wired into the `DynamicModel` dispatcher and cannot be loaded via the CLI or `DynamicModel::load()`:

| Family | Module | Notes |
|--------|--------|-------|
| GPT-OSS | `gpt_oss` | MoE with Top-4 sigmoid routing, 20B/120B variants |
| Pixtral | `pixtral` | 12B vision-language model |
| Qwen2-VL | `qwen2_vl` | 2B, 7B vision-language model |
| MLlama | `mllama` | Llama 3.2-Vision |
| CLIP | `clip` | ViT-L/14 vision encoder |
| Whisper | `whisper` | Base, Small, Medium, Large speech models |
| T5 | `t5` | Encoder-decoder architecture |

These modules can be used directly via their Rust types (e.g., `pmetal_models::architectures::gpt_oss::GptOssForCausalLM`) but require manual weight loading.

### Diffusion Models

| Family | Variants | Status |
|--------|----------|--------|
| Flux | 1-dev, 1-schnell | Dispatcher + pipeline implemented |

## Training Methods

All training methods support callback-based cancellation (`should_stop()`), metrics JSONL logging, and adaptive learning rate control.

| Method | CLI | GUI | TUI | Library |
|--------|-----|-----|-----|---------|
| SFT (Supervised Fine-Tuning) | `train` | Yes | Yes | `easy::finetune()` |
| LoRA | `train` | Yes | Yes | `easy::finetune()` |
| QLoRA (4-bit) | `train --quantization nf4` | Yes | Yes | `easy::finetune()` |
| DoRA | `train --dora` | Yes | Yes | `easy::finetune()` |
| DPO (Direct Preference) | — | — | — | `easy::dpo()` |
| SimPO (Simple Preference) | — | — | — | `easy::simpo()` |
| ORPO (Odds-Ratio Preference) | — | — | — | `easy::orpo()` |
| KTO (Kahneman-Tversky) | — | — | — | `easy::kto()` |
| GRPO (Reasoning) | `grpo` | Yes | Yes | `GrpoTrainer` |
| DAPO (Decoupled GRPO) | `grpo --dapo` | Yes | Yes | `DapoTrainer` |
| Knowledge Distillation | `distill` | Yes | Yes | `Distiller` |
| TAID (Temporally Adaptive) | — | — | — | `TaidDistiller` |
| ANE Training | `train` (auto) | — | Yes | `AneTrainingLoop` |

| RLKD (RL + Distillation) | `rlkd` | — | — | `RlkdTrainer` |
| Embedding Training | `embed-train` | — | — | `EmbeddingTrainer` |

Additional methods available via the library only: GSPO (`GspoTrainer`), PPO (`PpoTrainer`), Online DPO (`OnlineDpoTrainer`), Diffusion Training (`DiffusionTrainer`).

## Key Features

### Metal GPU Optimizations

Custom Metal shaders provide significant speedups:

- **FlashAttention**: O(n) memory attention with fused softmax, tier-aware block sizes
- **Fused GDN**: Gated Delta Network recurrence kernel (ported from FLA Triton) — single-pass state update with SIMD reductions
- **Fused LoRA**: Combined forward pass for adapter layers (~2x speedup with `lora-metal-fused` feature)
- **Fused Cross-Entropy**: Unsloth-style chunked loss computation
- **Fused Linear Cross-Entropy**: Skips logits materialization entirely
- **Fused RoPE**: Rotary position embeddings in-kernel
- **Fused SwiGLU**: Fused gate + activation with tier-tuned threadgroups
- **Fused RMSNorm + LoRA**: Combined normalization and adapter projection
- **Fused Sampler**: JIT-compiled token sampling
- **Fused MLP**: Combined gate/up/down projections
- **Async Scheduler**: Double/triple-buffered GPU command scheduling

### ANE (Neural Engine) Pipeline

Native ANE integration for power-efficient training and inference:

- **Dynamic Weight Pipeline**: 9 MIL kernels compiled once at startup; weights packed alongside activations in IOSurface spatial dimension
- **Hybrid Inference**: ANE prefill + CPU decode with KV cache. Power-of-2 sequence bucketing for optimal kernel compilation
- **CPU RMSNorm**: RMSNorm computed in f32 on CPU to avoid fp16 overflow on ANE (saturation arithmetic)
- **IOSurface Zero-Copy**: fp32 shared memory surfaces for CPU-ANE data transfer with no serialization overhead
- **M1-M5 Compatibility**: Per-matrix weight blobs for M1, single-blob for M3+. CPU FFN fallback for 4B+ models

### Training Infrastructure

- **Sequence Packing**: Efficiently pack multiple sequences into single batches for 2-5x throughput. Enabled by default
- **Gradient Checkpointing**: Trade compute for memory on large models with configurable layer grouping
- **Adaptive LR**: EMA-based anomaly detection with spike recovery, plateau reduction, and divergence detection
- **Callback System**: `TrainingCallback` trait with lifecycle hooks (`on_step_start`, `on_step_end`, `should_stop`) for metrics logging, progress reporting, and clean cancellation
- **Checkpoint Management**: Save and resume training from checkpoints with best-loss rollback
- **Tool/Function Calling**: Chat templates with native tool definitions for Qwen, Llama 3.1+, Mistral v3+, and DeepSeek
- **Schedule-Free Optimizer**: Memory-efficient optimizer without learning rate schedules
- **Metal Fused Optimizer**: GPU-accelerated AdamW parameter updates
- **8-bit Adam**: Memory-efficient optimizer for large models
- **LoRA+**: Differentiated learning rates for LoRA A and B matrices
- **NEFTune**: Noise-augmented fine-tuning for improved generation quality
- **Distributed Training**: mDNS auto-discovery, Ring All-Reduce with gradient compression

### Dataset Formats

Auto-detected training data formats:

- **ShareGPT**: `{"conversations": [{"from": "human", "value": "..."}, ...]}`
- **Alpaca**: `{"instruction": "...", "input": "...", "output": "..."}`
- **OpenAI/Messages**: `{"messages": [{"role": "user", "content": "..."}, ...]}`
- **Reasoning**: `{"problem": "...", "thinking": "...", "solution": "..."}`
- **Simple**: `{"text": "..."}`
- **Parquet**: Supports both standard text columns and reasoning formats

**Custom columns**: Use `--text-column` for arbitrary field names, `--text-columns col1,col2` to concatenate multiple columns, and `--prompt-column`/`--response-column` for SFT loss masking. All training commands (train, distill, grpo, rlkd) support column flags uniformly.

The `pmetal dataset` subcommand provides utilities for analysis, download from HuggingFace, and format conversion (Parquet, JSON, JSONL, CSV, ShareGPT, Alpaca).

### Model Operations

- **HuggingFace Hub Search**: `pmetal search` with memory fit estimation and download
- **Model Merging** (16 strategies via library, 12 via CLI):

  | CLI | Library | Description |
  |-----|---------|-------------|
  | `linear` | `LinearMerge` | Simple weighted averaging |
  | `slerp` | `SlerpMerge` | Spherical linear interpolation |
  | `ties` | `TiesMerge` | Task arithmetic with sparsification and sign consensus |
  | `dare_ties` | `DareMerge` | Random pruning with rescaling (TIES variant) |
  | `dare_linear` | `DareMerge` | Random pruning with rescaling (linear variant) |
  | `task_arithmetic` | `TaskArithmeticMerge` | Task vector arithmetic |
  | `della` | `DellaMerge` | Adaptive magnitude-based pruning |
  | `della_linear` | `DellaMerge` | Adaptive magnitude pruning (linear variant) |
  | `breadcrumbs` | `BreadcrumbsMerge` | Breadcrumbs merge strategy |
  | `model_stock` | `ModelStockMerge` | Geometric interpolation based on task vector similarity |
  | `nearswap` | `NearswapMerge` | Near-swap merge strategy |
  | `passthrough` | `PassthroughMerge` | Layer passthrough composition |
  | — | `RamMerge` | RAM merge strategy |
  | — | `SouperMerge` | Souper merge strategy |
  | — | `MultiSlerpMerge` | Multi-model SLERP |

- **GPU-Accelerated Merging**: Metal-based merge operations for large models
- **FP8-Aware Merging**: Merge with FP8 quantization for memory efficiency
- **Async Merge Pipeline**: Double-buffered streaming merge for large models
- **LoRA Fusing**: Merge LoRA adapters into base weights (standard and accurate modes)
- **GGUF Quantization** (13 format options):

  | Format | Description |
  |--------|-------------|
  | `dynamic` | Auto-select per layer |
  | `q8_0` | 8-bit quantization |
  | `q6k` | 6-bit k-quant |
  | `q5km` | 5-bit k-quant (medium) |
  | `q5ks` | 5-bit k-quant (small) |
  | `q4km` | 4-bit k-quant (medium) |
  | `q4ks` | 4-bit k-quant (small) |
  | `q3km` | 3-bit k-quant (medium) |
  | `q3ks` | 3-bit k-quant (small) |
  | `q3kl` | 3-bit k-quant (large) |
  | `q2k` | 2-bit k-quant |
  | `f16` | Float16 |
  | `f32` | Float32 |

  Supports importance matrix (`--imatrix`) for improved quantization quality. KL-calibrated quantization (`--kl-calibrate`) selects per-tensor quantization types via NRMSE + cosine distance, with optional `--target-bpw` for budget-constrained quantization.

- **FP8 Runtime Quantization**: Convert to FP8 (E4M3) at inference time for ~2x memory reduction

### Knowledge Distillation

Multiple distillation methods and loss functions:

- **Methods**: Online (live teacher inference), Offline (cached logits with compression), Progressive
- **TAID**: Temporally Adaptive Interpolated Distillation (ICLR 2025 SOTA) — `TaidDistiller`
- **Token-Level Losses**: KL Divergence, Jensen-Shannon, Soft Cross-Entropy, TVD, Hinge Ranking, Logistic Ranking
- **Hidden State Losses**: MSE, Cosine similarity, L1
- **Reasoning-Aware**: Rationale distillation for reasoning models
- **Cross-Vocabulary**: Distill between models with different tokenizers
- **Offline Logit Caching**: Compressed logit storage for memory-efficient offline distillation

## Configuration

### `pmetal train` Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--lora-r` | 16 | LoRA rank |
| `--lora-alpha` | 32.0 | LoRA scaling factor (2x rank) |
| `--batch-size` | 1 | Micro-batch size |
| `--learning-rate` | 2e-4 | Learning rate |
| `--max-seq-len` | 0 | Max seq len (0 = auto-detect) |
| `--epochs` | 1 | Number of training epochs |
| `--max-grad-norm` | 1.0 | Gradient clipping |
| `--quantization` | none | QLoRA method (nf4, fp4, int8) |
| `--gradient-accumulation-steps` | 4 | Gradient accumulation steps |
| `--no-ane` | false | Disable ANE training |
| `--embedding-lr` | None | Separate LR for embeddings |
| `--no-metal-fused-optimizer` | false | Disable Metal fused optimizer |
| `--lr-schedule` | cosine | Schedule type (constant, linear, cosine, cosine_with_restarts, polynomial, wsd) |
| `--no-gradient-checkpointing` | false | Disable gradient checkpointing (enabled by default) |
| `--gradient-checkpointing-layers` | 4 | Number of layers per checkpoint block |
| `--warmup-steps` | 100 | Learning rate warmup steps |
| `--weight-decay` | 0.01 | AdamW weight decay coefficient |
| `--no-sequence-packing` | false | Disable sequence packing |
| `--cut-cross-entropy` | false | Memory-efficient loss (avoids full logit materialization) |
| `--text-column` | — | Custom JSONL column name for training text |
| `--text-columns` | — | Multi-column concat (comma-separated, e.g. `thinking,solution`) |
| `--prompt-column` | — | Column for prompt (enables SFT loss masking) |
| `--response-column` | — | Column for response (with prompt masking) |
| `--column-separator` | `\n\n` | Separator for `--text-columns` |
| `--config` | — | Path to YAML configuration file |

### `pmetal infer` Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--temperature` | Model default | Sampling temperature |
| `--top-k` | Model default | Top-k sampling |
| `--top-p` | Model default | Nucleus sampling |
| `--min-p` | Model default | Min-p dynamic sampling |
| `--max-tokens` | 256 | Maximum generation length |
| `--repetition-penalty`| 1.0 | Repetition penalty |
| `--frequency-penalty` | 0.0 | Frequency penalty |
| `--presence-penalty` | 0.0 | Presence penalty |
| `--chat` | false | Apply chat template |
| `--show-thinking` | false | Show reasoning content |
| `--fp8` | false | Use FP8 weights (~2x mem reduction) |
| `--compiled` | false | Use JIT-compiled sampling |
| `--no-ane` | false | Disable ANE inference |
| `--ane-max-seq-len` | 1024 | Max ANE kernel sequence length |
| `--tools` | — | Tool/function definitions file (OpenAI format) |
| `--system` | — | System message |

### Feature Flags

| Feature | Default | Crate | Description |
|---------|---------|-------|-------------|
| `core` | Yes | `pmetal-core` | Foundation types, configs, traits |
| `gguf` | Yes | `pmetal-gguf` | GGUF format support |
| `metal` | Yes | `pmetal-metal` | Metal GPU kernels |
| `hub` | Yes | `pmetal-hub` | HuggingFace Hub integration |
| `mlx` | Yes | `pmetal-mlx` | MLX backend |
| `models` | Yes | `pmetal-models` | LLM architectures |
| `lora` | Yes | `pmetal-lora` | LoRA/QLoRA |
| `trainer` | Yes | `pmetal-trainer` | Training loops (pulls in `data`, `distill`) |
| `easy` | Yes | — | High-level builders (pulls in `trainer`, `hub`, `data`) |
| `ane` | Yes | — | Apple Neural Engine |
| `data` | Yes* | `pmetal-data` | Dataset loading (*default via `easy`) |
| `distill` | Yes* | `pmetal-distill` | Knowledge distillation (*default via `trainer`) |
| `lora-metal-fused` | No | — | ~2x LoRA training speedup via fused Metal kernels |
| `merge` | No | `pmetal-merge` | Model merging strategies |
| `vocoder` | No | `pmetal-vocoder` | BigVGAN neural vocoder |
| `distributed` | No | `pmetal-distributed` | Distributed training |
| `mhc` | No | `pmetal-mhc` | Manifold-Constrained Hyper-Connections |
| `full` | No | — | All features |

## Development

### Building

```bash
# Release build (default features: ANE + Dashboard)
cargo build --release

# Build without ANE
cargo build --release --no-default-features --features dashboard

# Run tests (single-threaded for Metal compatibility)
just test

# Build GUI
cd crates/pmetal-gui && bun install && bun tauri build
```

### Formal Verification

```bash
# cargo-kani proofs for ring all-reduce and topology
just kani-verify
```

## License

Licensed under either of MIT or Apache-2.0.

## Acknowledgments

- [MLX](https://github.com/ml-explore/mlx) - Apple's machine learning framework
- [mlx-rs](https://github.com/oxideai/mlx-rs) - Rust bindings for MLX
- [Unsloth](https://github.com/unslothai/unsloth) - Inspiration for fused kernels
- [Tauri](https://tauri.app) - Desktop application framework
