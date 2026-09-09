# Training Overview

All training methods available in PMetal — SFT, LoRA, DPO, GRPO, distillation, and more.

PMetal supports 12+ training methods across CLI, GUI, TUI, and the Rust/Python SDK. All methods support callback-based cancellation, JSONL metrics logging, and adaptive learning rate control.

## Method Matrix

| Method | CLI | GUI | TUI | Library |
|--------|-----|-----|-----|---------|
| SFT (Supervised Fine-Tuning) | `train` | Yes | Yes | `orchestrator::run_training()` |
| LoRA | `train` | Yes | Yes | `orchestrator::run_training()` |
| QLoRA (4-bit) | `train --quantization nf4` | Yes | Yes | `orchestrator::run_training()` |
| DoRA | — | — | — | `LoraConfig { use_dora: true }` |
| DPO (Direct Preference) | — | — | — | `DpoTrainer` |
| SimPO | — | — | — | `SimpoTrainer` |
| ORPO | — | — | — | `OrpoTrainer` |
| KTO | — | — | — | `KtoTrainer` |
| GRPO (Reasoning) | `grpo` | Yes | Yes | `GrpoTrainer` |
| DAPO (Decoupled GRPO) | `grpo --dapo` | Yes | Yes | `GrpoTrainer` DAPO mode |
| Knowledge Distillation | `distill` | Yes | Yes | `Distiller` |
| TAID | — | — | — | `TaidDistiller` |
| ANE Training | `train` (auto) | — | Yes | `AneTrainingLoop` |

Library-only methods: GSPO, PPO, Online DPO, Diffusion Training.

## Training Infrastructure

### Sequence Packing
Packs multiple sequences into single batches for 2–5× throughput. Enabled by default with proper attention masking.

### Gradient Checkpointing
Trade compute for memory on large models. Configurable layer grouping (default: 4 layers per block).

### Adaptive Learning Rate
EMA-based anomaly detection with automatic spike recovery, plateau reduction, and divergence detection.

### Optimizers

| Optimizer | Description |
|-----------|-------------|
| AdamW | Standard with configurable weight decay |
| Metal Fused AdamW | GPU-accelerated parameter updates |
| Schedule-Free | Memory-efficient, no LR schedule needed |
| 8-bit Adam | Memory-efficient for large models |
| LoRA+ | Differentiated LR for A and B matrices |

### LR Schedules
`constant`, `linear`, `cosine`, `cosine_with_restarts`, `polynomial`, `wsd`

### Additional Features
- **NEFTune** — Noise-augmented fine-tuning for improved generation quality
- **Checkpoint Management** — Save/resume with best-loss rollback
- **Tool/Function Calling** — Chat templates with native tool definitions
- **Distributed Training** — mDNS auto-discovery, Ring All-Reduce

## Dataset Formats

Auto-detected:

| Format | Structure |
|--------|-----------|
| ShareGPT | `{"conversations": [{"from": "human", "value": "..."}]}` |
| Alpaca | `{"instruction": "...", "input": "...", "output": "..."}` |
| OpenAI/Messages | `{"messages": [{"role": "user", "content": "..."}]}` |
| Reasoning | `{"problem": "...", "thinking": "...", "solution": "..."}` |
| Simple | `{"text": "..."}` |
| Parquet | Standard text columns or reasoning formats |

## See Also

- [Training Methods](/training/methods/) — Detailed method descriptions
- [Distillation](/training/distillation/) — Knowledge distillation deep dive
- [pmetal train](/cli/train/) — CLI training parameters
