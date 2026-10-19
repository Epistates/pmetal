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
| `--quantization` | none | QLoRA base format: `nf4`, `fp4` (NVFP4), `int8` |
| `--quant-block-size` | `64` | Weights per scale for `nf4` and `int8` (32, 64 or 128) |
| `--double-quant` | `false` | Pack `nf4`'s absmax values to 8 bits as well |
| `--gradient-accumulation-steps` | `4` | Gradient accumulation steps |
| `--embedding-lr` | None | Separate LR for embeddings |
| `--lr-schedule` | `cosine` | `constant`, `linear`, `cosine`, `cosine_with_restarts`, `polynomial`, `wsd` |
| `--no-gradient-checkpointing` | `false` | Disable gradient checkpointing |
| `--gradient-checkpointing-layers` | `4` | Layers per checkpoint block |
| `--warmup-steps` | `0` | Learning rate warmup steps |
| `--weight-decay` | `0.01` | Weight decay |
| `--optimizer` | `adamw` | `adamw`, `sgd` (momentum 0.9), `lion`, `adafactor`, `muon`, `adamw_schedule_free`, `adamw_8bit`. Lion wants a 3-10x smaller learning rate and 3-10x larger weight decay than AdamW; schedule-free AdamW wants a constant rate |
| `--no-sequence-packing` | `false` | Disable sequence packing |
| `--cut-cross-entropy` | `false` | Loss from fused Metal kernels that never hold the logits; gradient as precise as the full logits' |
| `--text-column` | — | Custom JSONL column name for training text |
| `--text-columns` | — | Multi-column concat (comma-separated, e.g. `thinking,solution`) |
| `--prompt-column` | — | Column for prompt (enables SFT loss masking) |
| `--response-column` | — | Column for response (with prompt masking) |
| `--column-separator` | `\n\n` | Separator for `--text-columns` |
| `--config` | — | Path to YAML configuration file |

## Dataset Formats

Training data is auto-detected from a row's columns (structured layouts before `text`):

- **OpenAI/Messages**: `{"messages": [{"role": "user", "content": "..."}, ...]}`
- **ShareGPT**: `{"conversations": [{"from": "human", "value": "..."}, ...]}`
- **Alpaca**: `{"instruction": "...", "input": "...", "output": "..."}`
- **Prompt/completion**: `{"prompt": "...", "completion": "..."}`, strings or message lists
- **Reasoning**: `{"problem": "...", "thinking": "...", "solution": "..."}`
- **Simple**: `{"text": "..."}`

The file can be JSONL, JSON, Parquet, Arrow or CSV, or a directory holding a
dataset's shards (its `train` split is read, every shard in order).

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

- `adapter_model.safetensors` and `adapter_config.json` — the LoRA adapter, in the layout `peft` reads (`PeftModel.from_pretrained(base, "./output")` loads it; adapters from earlier releases, `lora_weights.safetensors`, still load here)
- `training_metrics.jsonl` — Per-step metrics log
- `checkpoint/` — Resumable checkpoints (if training is interrupted)

## See Also

- [Configuration File](/configuration/config-file/) — YAML config format
- [Training Overview](/training/overview/) — All training methods
- [Supported Models](/models/supported/) — LoRA-compatible architectures
