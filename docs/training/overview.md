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
| DPO / Robust DPO, IPO, hinge | `preference --loss dpo\|ipo\|hinge` | Yes | Yes | `PreferenceTrainer` |
| SimPO | `preference --loss simpo` | Yes | Yes | `PreferenceTrainer` |
| ORPO | `preference --loss orpo` | Yes | Yes | `PreferenceTrainer` |
| KTO | `preference --loss kto` | Yes | Yes | `KtoTrainer` |
| GRPO (Reasoning) | `grpo` | Yes | Yes | `GrpoTrainer` |
| DAPO | `grpo --dapo` | Yes | Yes | `GrpoConfig::for_dapo` |
| Dr. GRPO | `grpo --loss-type dr_grpo` | Yes | Yes | `GrpoConfig::for_dr_grpo` |
| GSPO | `grpo --loss-type gspo --num-iterations 2` | Yes | Yes | `GrpoConfig::for_gspo` |
| Knowledge Distillation | `distill` | Yes | Yes | `Distiller` |
| TAID | — | — | — | `TaidDistiller` |
| ANE Training | `train` (auto) | — | Yes | `AneTrainingLoop` |

## Training Infrastructure

### Sequence Packing
Packs multiple sequences into single batches for 2–5× throughput. Enabled by default with proper attention masking.

### Gradient Checkpointing
Trade compute for memory on large models. Configurable layer grouping (default: 4 layers per block).

### Adaptive Learning Rate
EMA-based anomaly detection with automatic spike recovery, plateau reduction, and divergence detection.

### Optimizers

Every training command takes `--optimizer` (and YAML configs take `training.optimizer`):

| Optimizer | Description |
|-----------|-------------|
| `adamw` (default) | Decoupled weight decay, betas (0.9, 0.999), eps 1e-8 |
| `sgd` | Heavy-ball momentum 0.9, L2 weight decay |
| `lion` | Sign of an interpolated momentum, betas (0.9, 0.99). Use a learning rate 3-10x smaller and a weight decay 3-10x larger than for AdamW (Chen et al., arXiv 2302.06675) |
| `adafactor` | Factored second moments, update clipping at RMS 1, decay 1 - t^-0.8, no first moment, learning rate from the schedule (Shazeer and Stern, arXiv 1804.04235). Usually run around 1e-3 |
| Metal Fused AdamW | GPU-accelerated AdamW updates, used for `adamw` unless `--no-metal-fused-optimizer` |
| LoRA+ | Differentiated LR for A and B matrices, with any optimizer |

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
