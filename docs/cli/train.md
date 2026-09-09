# pmetal train

Fine-tune models with LoRA, QLoRA, or DoRA using supervised fine-tuning (SFT).

Fine-tune a model with LoRA/QLoRA/DoRA. Supports SFT on any supported architecture with automatic hardware detection and kernel tuning.

## Usage

```bash
pmetal train \
  --model <MODEL> \
  --dataset <DATASET> \
  --output <OUTPUT_DIR> \
  [OPTIONS]
```

## Examples

```bash
# Basic LoRA fine-tuning
pmetal train \
  --model Qwen/Qwen3-0.6B \
  --dataset train.jsonl \
  --output ./output \
  --lora-r 16 --batch-size 4 --learning-rate 2e-4

# QLoRA with 4-bit quantization
pmetal train \
  --model meta-llama/Llama-3.2-1B \
  --dataset train.jsonl \
  --output ./output \
  --quantization nf4 --lora-r 16

# DoRA with custom schedule
pmetal train \
  --model Qwen/Qwen3-0.6B \
  --dataset train.jsonl \
  --lr-schedule cosine_with_restarts

# From a config file
pmetal train --config training.yaml
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | *required* | HuggingFace model ID or local path |
| `--dataset` | *required* | Path to training dataset (JSONL, Parquet, CSV) |
| `--output` | `./output` | Output directory for weights and logs |
| `--lora-r` | `16` | LoRA rank |
| `--lora-alpha` | `32.0` | LoRA scaling factor (2× rank) |
| `--batch-size` | `1` | Micro-batch size |
| `--learning-rate` | `2e-4` | Learning rate |
| `--max-seq-len` | `0` | Max sequence length (0 = auto-detect) |
| `--epochs` | `1` | Number of training epochs |
| `--max-grad-norm` | `1.0` | Gradient clipping |
| `--quantization` | none | QLoRA method: `nf4`, `fp4`, `int8` |
| `--gradient-accumulation-steps` | `4` | Gradient accumulation steps |
| `--embedding-lr` | None | Separate LR for embeddings |
| `--no-metal-fused-optimizer` | `false` | Disable Metal fused optimizer |
| `--lr-schedule` | `cosine` | `constant`, `linear`, `cosine`, `cosine_with_restarts`, `polynomial`, `wsd` |
| `--no-gradient-checkpointing` | `false` | Disable gradient checkpointing |
| `--gradient-checkpointing-layers` | `4` | Layers per checkpoint block |
| `--warmup-steps` | `100` | Learning rate warmup steps |
| `--weight-decay` | `0.01` | AdamW weight decay |
| `--no-sequence-packing` | `false` | Disable sequence packing |
| `--cut-cross-entropy` | `false` | Memory-efficient loss (avoids full logit materialization) |
| `--text-column` | — | Custom JSONL column name for training text |
| `--text-columns` | — | Multi-column concat (comma-separated, e.g. `thinking,solution`) |
| `--prompt-column` | — | Column for prompt (enables SFT loss masking) |
| `--response-column` | — | Column for response (with prompt masking) |
| `--column-separator` | `\n\n` | Separator for `--text-columns` |
| `--config` | — | Path to YAML configuration file |

## Dataset Formats

Training data is auto-detected:

- **ShareGPT**: `{"conversations": [{"from": "human", "value": "..."}, ...]}`
- **Alpaca**: `{"instruction": "...", "input": "...", "output": "..."}`
- **OpenAI/Messages**: `{"messages": [{"role": "user", "content": "..."}, ...]}`
- **Reasoning**: `{"problem": "...", "thinking": "...", "solution": "..."}`
- **Simple**: `{"text": "..."}`
- **Parquet**: Standard text columns or reasoning formats

### Custom Columns

Use `--text-column` for arbitrary field names, or `--text-columns` to concatenate multiple columns:

```bash
# Single custom column
pmetal train --model ... --dataset data.jsonl --text-column response

# Concatenate thinking + solution columns
pmetal train --model ... --dataset data.jsonl \
  --text-columns thinking,solution --column-separator "\n\n"

# SFT loss masking (only train on response, mask prompt)
pmetal train --model ... --dataset data.jsonl \
  --prompt-column instruction --response-column output
```

## Output

Training produces:

- `lora_weights.safetensors` — LoRA adapter weights
- `training_metrics.jsonl` — Per-step metrics log
- `checkpoint/` — Resumable checkpoints (if training is interrupted)

## See Also

- [Configuration File](/configuration/config-file/) — YAML config format
- [Training Overview](/training/overview/) — All training methods
- [Supported Models](/models/supported/) — LoRA-compatible architectures
