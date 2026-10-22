# Training Overview

All training methods available in PMetal: SFT, LoRA, DPO, GRPO, distillation, and more.

PMetal supports the training methods below from the CLI, TUI, GUI, MCP server, Python and Rust. All support cancellation, JSONL metrics logging (`--log-metrics`) and, where the loop has one, adaptive learning rate control.

## Method Matrix

Every command is a job spec shared by the CLI, TUI, GUI, MCP server and Python bindings, so a method available as a command has a form in the GUI and a tool in the MCP server.

| Method | Command | Library |
|--------|---------|---------|
| SFT (Supervised Fine-Tuning) | `train` | `orchestrator::run_training()` |
| LoRA | `train` | `orchestrator::run_training()` |
| QLoRA (NF4, NVFP4 or int8 base) | `train --quantization nf4\|fp4\|int8` | `orchestrator::run_training()` |
| DoRA, rsLoRA | `train --dora`, `train --rslora` | `LoraConfig { use_dora, use_rslora }` |
| Full-parameter fine-tuning | `train --full` | `orchestrator::run_training()` (`TrainingJobConfig::full`) |
| DPO / Robust DPO, IPO, hinge | `preference --loss dpo\|ipo\|hinge` | `PreferenceTrainer` |
| SimPO | `preference --loss simpo` | `PreferenceTrainer` |
| ORPO | `preference --loss orpo` | `PreferenceTrainer` |
| KTO | `preference --loss kto` | `KtoTrainer` |
| GRPO (Reasoning) | `grpo --loss-type grpo` (rewards: `--reward-cmd`, `--reward-regex`, `--reward-json-schema`, Python callables) | `GrpoTrainer`, `grpo::run_grpo_job` |
| DAPO | `grpo --dapo` (the default `--loss-type`) | `GrpoConfig::for_dapo` |
| Dr. GRPO | `grpo --loss-type dr_grpo` | `GrpoConfig::for_dr_grpo` |
| GSPO | `grpo --loss-type gspo --num-iterations 2` | `GrpoConfig::for_gspo` |
| Knowledge Distillation | `distill` | `Distiller` |
| TAID | not wired to a command | `TaidDistiller` |
| RLKD (RL + Distillation) | `rlkd` | `RlkdTrainer` |
| Embedding training | `embed-train` | `EmbeddingTrainer` |
| Pretraining | `pretrain` | `pmetal_trainer` |
| Block diffusion (DiffusionGemma) | `train-diffusion` | `DiffusionGemmaTrainer` |
| MTP predictor and assistant training | `train-mtp` | `pmetal_trainer::mtp_training` |
| DFlash draft training | `train-draft` | `pmetal_trainer::mtp_training` |
| ANE training (experimental) | `train --ane` | `AneTrainingLoop` |

## Training Infrastructure

### Sequence Packing
Packs multiple sequences into single rows so short samples do not waste compute on padding. On by default with block-diagonal attention masking, and for models with recurrent layers (Nemotron-H, Granite 4.0-H) the recurrent state is reset at sequence boundaries; `--no-sequence-packing` turns it off.

### Gradient Checkpointing
Trade compute for memory: each decoder layer is recomputed in the backward pass instead of stored. On by default; `--no-gradient-checkpointing` turns it off.

### Adaptive Learning Rate
EMA-based anomaly detection with automatic spike recovery, plateau reduction, and divergence detection.

### Optimizers

Every training command takes `--optimizer` (and YAML configs take `training.optimizer`):

| Optimizer | Description |
|-----------|-------------|
| `adamw` (default) | Updates as `torch.optim.AdamW` does: bias-corrected moments, decoupled weight decay, betas (0.9, 0.999), eps 1e-8 |
| `sgd` | Heavy-ball momentum 0.9, L2 weight decay |
| `lion` | Sign of an interpolated momentum, betas (0.9, 0.99). Use a learning rate 3-10x smaller and a weight decay 3-10x larger than for AdamW (Chen et al., arXiv 2302.06675) |
| `adafactor` | Factored second moments, update clipping at RMS 1, decay 1 - t^-0.8, no first moment, learning rate from the schedule (Shazeer and Stern, arXiv 1804.04235). Usually run around 1e-3 |
| `muon` | Momentum orthogonalized by five Newton-Schulz steps on the hidden weight matrices (Jordan et al., 2024), with the update scaled to AdamW's RMS (`0.2·sqrt(max(rows, cols))`, Liu et al., arXiv 2502.16982) so an AdamW learning rate and weight decay carry over. Embeddings, the output head, norms and biases train with AdamW |
| `adamw_schedule_free` | Schedule-free AdamW (Defazio et al., arXiv 2405.15682): iterate averaging in place of a decaying schedule. Run at a constant learning rate; the saved and evaluated weights are the average, which the loop swaps to for validation and checkpoints. Takes the learning rate AdamW would, without the decay |
| `adamw_8bit` | AdamW with both moments held in 8 bits, quantized blockwise (blocks of 256) against a dynamic code (Dettmers et al., arXiv 2110.02861); a quarter of AdamW's optimizer memory. Tensors under 4096 elements keep f32 moments |
| LoRA+ | Differentiated LR for A and B matrices, with any optimizer |

Every optimizer trains the same parameter groups. Biases and
normalization weights train without weight decay (the split
`transformers`' `Trainer` makes), `--embedding-lr` sets the rate of the
token embedding and the output head (and of an adapter on the head), and
LoRA+ trains each B matrix
at `loraplus_lr_ratio` times its A matrix's rate. The scheduler moves every
group's rate together, keeping the ratios.

### LR Schedules
`constant`, `linear`, `cosine`, `cosine_with_restarts`, `polynomial`, `wsd`

### Additional Features
- **NEFTune** (`--neftune-alpha`), **label smoothing** (`--label-smoothing`) and **z-loss** (`--z-loss`)
- **Cut cross-entropy** (`--cut-cross-entropy`) computes the loss without materializing the full logits
- **Checkpoint Management**: Save/resume with best-loss rollback
- **Tool/Function Calling**: Chat templates with native tool definitions
- **Distributed Training**: mDNS auto-discovery, Ring All-Reduce

## Dataset Formats

Auto-detected from a row's columns, structured layouts first:

| Format | Structure |
|--------|-----------|
| OpenAI/Messages | `{"messages": [{"role": "user", "content": "..."}]}` (content parts, `tool_calls`, `tools`, per-message `weight`) |
| ShareGPT | `{"conversations": [{"from": "human", "value": "..."}]}` |
| Alpaca | `{"instruction": "...", "input": "...", "output": "..."}` |
| Prompt/completion | `{"prompt": "...", "completion": "..."}` (strings or message lists; only the completion trains) |
| Reasoning | `{"problem": "...", "thinking": "...", "solution": "..."}` |
| Simple | `{"text": "..."}` |

Any of these can be in JSONL, JSON, Parquet, Arrow or CSV files, or in a
directory (a Hugging Face dataset's `train` split, every shard). Every
assistant turn of a conversation trains; plain-text completions end in an EOS.

## See Also

- [Training Methods](/training/methods/): Detailed method descriptions
- [Distillation](/training/distillation/): Knowledge distillation deep dive
- [pmetal train](/cli/train/): CLI training parameters
