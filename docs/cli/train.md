# pmetal train

Fine-tune a model with LoRA, QLoRA or DoRA using supervised fine-tuning (SFT).

Supports SFT on any supported architecture with automatic hardware detection and kernel tuning. The adapter is written to the output directory in the layout `peft` reads.

Ctrl-C (SIGINT) or SIGTERM finishes the step in flight, saves the adapter and exits with status 130; `--resume` continues from the latest checkpoint. `--log-events <file>` (a global option, given before or after the command) writes the run's events as JSONL, which is how the TUI, the GUI and the MCP server follow a run.

## Usage

```bash
pmetal train \
  --model <MODEL> \
  --dataset <DATASET> \
  [--output <OUTPUT_DIR>] \
  [OPTIONS]
```

`--model` and `--dataset` are required unless `--config` supplies them.

## Options

### Model and data

| Option | Default | Description |
|--------|---------|-------------|
| `-c`, `--config` | none | Path to a YAML configuration file (see [Configuration File](/configuration/config-file/)) |
| `-m`, `--model` | required | HuggingFace model ID or local path |
| `-d`, `--dataset` | required | Training dataset: a JSONL, JSON, Parquet, Arrow or CSV file, or a HuggingFace dataset ID |
| `--eval-dataset` | none | Evaluation dataset (same forms as `--dataset`) |
| `-o`, `--output` | `./output` | Output directory for the adapter and checkpoints |
| `--text-column` | none | Custom JSONL column holding the training text |
| `--text-columns` | none | Multiple columns to concatenate as the training text (comma-separated, e.g. `thinking,solution`) |
| `--column-separator` | blank line | Separator for `--text-columns` |
| `--prompt-column` | none | Column holding the prompt (enables SFT loss masking) |
| `--response-column` | none | Column holding the response (with prompt masking) |

### LoRA

| Option | Default | Description |
|--------|---------|-------------|
| `--lora-r` | `16` | LoRA rank |
| `--lora-alpha` | `32.0` | LoRA scaling factor (2x rank is the usual choice) |
| `--lora-dropout` | `0.0` | Dropout applied to the LoRA branch during training |
| `--lora-targets` | all linear layers | Projections to adapt, comma-separated (`q_proj,v_proj`); MoE routers never are |
| `--dora` | `false` | Train DoRA (weight-decomposed LoRA) adapters: slower per step, usually better at low rank |
| `--rslora` | `false` | Rank-stabilized scaling (`alpha/sqrt(r)` instead of `alpha/r`), which keeps the update magnitude steady as rank grows |
| `--quantization` | `none` | QLoRA base format: `none`, `nf4`, `fp4` (NVFP4) or `int8` |
| `--quant-block-size` | `64` | Weights per scale for `nf4` and `int8` (32, 64 or 128; `fp4`'s block is 16) |
| `--double-quant` | `false` | Pack `nf4`'s absmax values to 8 bits as well |

### Training loop

| Option | Default | Description |
|--------|---------|-------------|
| `--batch-size` | `1` | Micro-batch size |
| `--gradient-accumulation-steps` | `4` | Gradient accumulation steps |
| `--epochs` | `1` | Number of training epochs |
| `--max-steps` | none | Stop after this many optimizer steps, whatever `--epochs` says |
| `--max-seq-len` | `0` | Maximum sequence length; `0` detects it from the model config |
| `--max-grad-norm` | `1.0` | Gradient clipping norm; `0` disables |
| `--no-sequence-packing` | `false` | Disable sequence packing |
| `--pack-max-seq-len` | none | Override the maximum sequence length used for packing |
| `--no-gradient-checkpointing` | `false` | Disable gradient checkpointing |
| `--cut-cross-entropy` | `false` | Loss from fused Metal kernels that never hold the logits; gradient as precise as the full logits' |
| `--label-smoothing` | `0.0` | Label smoothing of the training loss, in [0, 1); evaluation reports the plain loss |
| `--z-loss` | `0.0` | Weight of the z-loss `weight * lse^2` added to each token's loss (PaLM; 1e-4 is the paper's value) |
| `--neftune-alpha` | none | NEFTune noise alpha on the token embeddings of every training forward pass (5 for small models, 15 for larger) |
| `--seed` | `42` | Random seed |
| `--resume` | `false` | Resume from the latest checkpoint in the output directory |

### Optimizer and schedule

| Option | Default | Description |
|--------|---------|-------------|
| `--learning-rate` | `2e-4` | Learning rate |
| `--embedding-lr` | none | Separate learning rate for embeddings |
| `--optimizer` | `adamw` | `adamw`, `sgd` (momentum 0.9), `lion`, `adafactor`, `muon`, `adamw_schedule_free` or `adamw_8bit`. Lion wants a 3-10x smaller learning rate and 3-10x larger weight decay than AdamW; schedule-free AdamW wants a constant rate |
| `--weight-decay` | `0.01` | Weight decay |
| `--lr-schedule` | `cosine` | `constant`, `linear`, `cosine`, `cosine_with_restarts`, `polynomial` or `wsd` |
| `--warmup-steps` | `0` | Linear warmup steps |
| `--warmup-ratio` | none | Share of the optimizer steps spent warming up (0 to 1); replaces `--warmup-steps` when set |
| `--min-lr` | `0` | Floor the decaying schedules stop at |
| `--wsd-stable-ratio` | `0.7` | Share of the steps the `wsd` schedule holds the peak rate for |
| `--wsd-decay-style` | `linear` | Shape of the `wsd` decay: `linear`, `cosine` or `one_minus_sqrt` |
| `--cosine-num-restarts` | `1` | Cycles of `cosine_with_restarts` |
| `--polynomial-power` | `1` | Exponent of the `polynomial` schedule |
| `--no-adaptive-lr` | `false` | Turn off the automatic adaptive learning rate (reductions on loss spikes, plateaus and divergence; it can only lower the scheduled rate) |

### Logging and checkpoints

| Option | Default | Description |
|--------|---------|-------------|
| `--logging-steps` | `10` | Log the loss every this many optimizer steps |
| `--eval-steps` | `100` | Evaluate every this many optimizer steps (needs `--eval-dataset`) |
| `--save-steps` | `500` | Write a checkpoint every this many optimizer steps |
| `--log-metrics` | none | Write per-step metrics as JSONL to this path; a bare file name goes inside the output directory |

### Apple Neural Engine and distributed training

| Option | Default | Description |
|--------|---------|-------------|
| `--ane` | `false` | Train on the Apple Neural Engine (experimental) |
| `--loss-scale` | `1.0` | Loss scaling factor for ANE training |
| `--distributed-peers` | none | Distributed training: comma-separated peer addresses (`ip:port`) |
| `--distributed-auto` | `false` | Distributed training: discover peers automatically over mDNS |
| `--compression-strategy` | `none` | Gradient compression for distributed training: `none`, `topk`, `fp16` or `random` |

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

# DoRA with a custom schedule
pmetal train \
  --model Qwen/Qwen3-0.6B \
  --dataset train.jsonl \
  --dora --lr-schedule cosine_with_restarts

# Resume an interrupted run
pmetal train --model Qwen/Qwen3-0.6B --dataset train.jsonl --output ./output --resume

# From a config file
pmetal train --config training.yaml
```

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

Training produces, in the output directory:

- `adapter_model.safetensors` and `adapter_config.json`: the LoRA adapter, in the layout `peft` reads (`PeftModel.from_pretrained(base, "./output")` loads it; adapters from earlier releases, `lora_weights.safetensors`, still load here)
- `checkpoints/`: resumable checkpoints, one every `--save-steps` steps (`--resume` continues from the latest)
- the file named by `--log-metrics`, when given: per-step metrics

## See Also

- [Configuration File](/configuration/config-file/): YAML config format
- [Training Overview](/training/overview/): All training methods
- [Supported Models](/models/supported/): LoRA-compatible architectures
- [pmetal fuse](/cli/fuse/): Fold the adapter into the base model
- [pmetal infer](/cli/infer/): Run the adapter (`--lora`)
