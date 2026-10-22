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

# Full-parameter fine-tuning: every weight trains, and the result is a whole model
pmetal train \
  --model Qwen/Qwen3-0.6B \
  --dataset train.jsonl \
  --output ./tuned \
  --full --learning-rate 2e-5 --optimizer adamw_8bit

# From a config file
pmetal train --config training.yaml
```

## Full-parameter fine-tuning

`--full` trains every weight of the model, not adapters, through the same training loop, optimizers,
gradient checkpointing and sequence packing as LoRA. The model is the one `pmetal infer` and `serve`
run, with nothing attached, so what trains is what is served.

- **Result.** `--output` holds a complete Hugging Face-layout model: sharded `model-0000N-of-0000M.safetensors`
  with `model.safetensors.index.json` (one `model.safetensors` when it fits in one file) under the
  checkpoint's own tensor names, and the checkpoint's `config.json`, tokenizer and chat template.
  `pmetal infer --model ./tuned` and `pmetal serve --model ./tuned` load it, as does any other tool
  that reads the format. The weights are written in the dtype the model trained in. The output
  directory must not be the model's own.
- **Memory.** It holds the weights, an f32 gradient per weight and the optimizer's state, several
  times the model's size where LoRA needs little more than the model. Before anything loads, the
  run's estimate (weights, gradients, optimizer moments and f32 master copies of half-precision
  weights, activations) is checked against the device's recommended working set, and the run is
  refused when it will not fit, with the breakdown and what would shrink it:
  `--optimizer adamw_8bit` (one byte per moment), `--freeze-embeddings`, `--cut-cross-entropy`,
  a smaller `--batch-size` or `--max-seq-len`. `--no-memory-check` runs it anyway.
- **Learning rate.** LoRA's default `2e-4` diverges a full fine-tune, so `--full` refuses a
  `--learning-rate` above `1e-4`. Use `1e-5` to `5e-5`.
- **Optimizers.** AdamW keeps f32 master weights for a bf16 model; `adamw_8bit` and `muon` work as in
  LoRA runs (Muon trains the embedding, head and norms with AdamW).
- **Frozen embedding.** `--freeze-embeddings` keeps the input token embedding as loaded, saving its
  gradient and optimizer state. A model that ties its output head to the embedding (Qwen3-0.6B does)
  keeps the head too; an untied head still trains.
- **Checkpoints and resume.** `--save-steps` and `--resume` work as in LoRA runs; a checkpoint is the
  whole model and its optimizer's state, so only the latest is kept. The adaptive learning-rate
  rollback, which holds a second copy of the weights, is off.
- **What it does not do.** `--quantization` (a packed base cannot take gradients), `--ane`
  (the Neural Engine path already trains every weight), `--dora`, `--rslora` and `--lora-targets` are
  refused with `--full`, and so is a quantized or GGUF checkpoint. A checkpoint whose tensors the
  model does not read under their own names (fused or stacked layouts) is refused before training,
  since the result could not be saved.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | *required* | HuggingFace model ID or local path |
| `--dataset` | *required* | Path to training dataset (JSONL, Parquet, CSV) |
| `--output` | `./output` | Output directory for weights and logs |
| `--full` | `false` | Train every weight and save a whole model (see [Full-parameter fine-tuning](#full-parameter-fine-tuning)) |
| `--freeze-embeddings` | `false` | With `--full`, keep the input token embedding as loaded |
| `--no-memory-check` | `false` | With `--full`, skip the check of the memory estimate against the device |
| `--lora-r` | `16` | LoRA rank |
| `--lora-alpha` | `32.0` | LoRA scaling factor (2× rank) |
| `--batch-size` | `1` | Micro-batch size |
| `--learning-rate` | `2e-4` | Learning rate |
| `--max-seq-len` | `0` | Max sequence length (0 = auto-detect) |
| `--epochs` | `1` | Number of training epochs |
| `--max-steps` | None | Stop after this many optimizer steps, whatever `--epochs` says |
| `--lora-targets` | all linear layers | Projections to adapt, comma-separated (`q_proj,v_proj`); MoE routers never are |
| `--logging-steps` | `10` | Log the loss every this many optimizer steps |
| `--eval-steps` | `100` | Evaluate every this many optimizer steps (needs `--eval-dataset`) |
| `--save-steps` | `500` | Write a checkpoint every this many optimizer steps |
| `--eval-dataset` | None | Evaluation dataset (same formats as `--dataset`) |
| `--log-metrics` | None | Write per-step metrics as JSONL to this path |
| `--max-grad-norm` | `1.0` | Gradient clipping |
| `--quantization` | none | QLoRA base format: `nf4`, `fp4` (NVFP4), `int8` |
| `--quant-block-size` | `64` | Weights per scale for `nf4` and `int8` (32, 64 or 128) |
| `--double-quant` | `false` | Pack `nf4`'s absmax values to 8 bits as well |
| `--gradient-accumulation-steps` | `4` | Gradient accumulation steps |
| `--embedding-lr` | None | Separate LR for embeddings |
| `--label-smoothing` | `0.0` | Label smoothing of the training loss, in [0, 1); evaluation reports the plain loss |
| `--z-loss` | `0.0` | Weight of the z-loss `weight * lse^2` added to each token's loss (PaLM; 1e-4 is the paper's value) |
| `--neftune-alpha` | None | NEFTune noise alpha on the token embeddings of every training forward pass (5 for small models, 15 for larger) |
| `--lr-schedule` | `cosine` | `constant`, `linear`, `cosine`, `cosine_with_restarts`, `polynomial`, `wsd` |
| `--warmup-ratio` | None | Share of the optimizer steps spent warming up (0 to 1); replaces `--warmup-steps` when set |
| `--min-lr` | `0` | Floor the decaying schedules stop at |
| `--wsd-stable-ratio` | `0.7` | Share of the steps the `wsd` schedule holds the peak rate for |
| `--wsd-decay-style` | `linear` | Shape of the `wsd` decay: `linear`, `cosine`, `one_minus_sqrt` |
| `--cosine-num-restarts` | `1` | Cycles of `cosine_with_restarts` |
| `--polynomial-power` | `1` | Exponent of the `polynomial` schedule |
| `--no-gradient-checkpointing` | `false` | Disable gradient checkpointing |
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

- `adapter_model.safetensors` and `adapter_config.json` — the LoRA adapter, in the layout `peft` reads (`PeftModel.from_pretrained(base, "./output")` loads it; adapters from earlier releases, `lora_weights.safetensors`, still load here). With `--full`, a whole model instead: see [Full-parameter fine-tuning](#full-parameter-fine-tuning)
- `training_metrics.jsonl` — Per-step metrics log
- `checkpoint/` — Resumable checkpoints (if training is interrupted)

## See Also

- [Configuration File](/configuration/config-file/) — YAML config format
- [Training Overview](/training/overview/) — All training methods
- [Supported Models](/models/supported/) — LoRA-compatible architectures
