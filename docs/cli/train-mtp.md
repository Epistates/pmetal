# pmetal train-mtp

Train MTP checkpoints for exact speculative decoding.

`train-mtp` supports two checkpoint families:

- Gemma 4 assistant checkpoints (`model_type = "gemma4_assistant"`) for `pmetal infer --draft-model`.
- Qwen3Next/Qwen3.6 `mtp.*` predictor checkpoints for `pmetal infer --mtp --mtp-model`.

## Usage

```bash
pmetal train-mtp \
  --model <TARGET_MODEL> \
  --family auto \
  --shards <TOKEN_SHARD>[,<TOKEN_SHARD>...] \
  --output <OUTPUT_DIR>
```

## Examples

```bash
# Prepare token shards from JSONL rows with a "text" column.
pmetal tokenize \
  --input train.jsonl \
  --output ./tok \
  --tokenizer /path/to/target-model

# Train a Qwen3Next/Qwen3.6 MTP predictor from the frozen target.
pmetal train-mtp \
  --model /path/to/qwen3next-target \
  --family qwen3-next \
  --shards ./tok/shard_00000.bin \
  --output ./qwen-mtp \
  --seq-len 512 --batch-size 1 --steps 1000

pmetal infer \
  --model /path/to/qwen3next-target \
  --mtp --mtp-model ./qwen-mtp \
  --prompt "Summarize monotonic queues."

# Train a Gemma 4 assistant checkpoint.
pmetal train-mtp \
  --model google/gemma-4-31B-it \
  --family gemma4 \
  --shards ./tok/shard_00000.bin \
  --output ./gemma4-assistant \
  --assistant-layers 2 --num-assistant-tokens 6

pmetal infer \
  --model google/gemma-4-31B-it \
  --draft-model ./gemma4-assistant \
  --prompt "Summarize monotonic queues."
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | required | Frozen target model ID or local path |
| `--family` | `auto` | `auto`, `qwen3-next`, or `gemma4` |
| `--shards` | required | Tokenized shard files from `pmetal tokenize` |
| `--output` | `./mtp-output` | Output checkpoint directory |
| `--init-mtp` | — | Existing assistant/MTP checkpoint to continue training |
| `--seq-len` | `512` | Sequence length per sample |
| `--batch-size` | `1` | Training batch size |
| `--steps` | `1000` | Optimizer steps |
| `--learning-rate` | `2e-4` | Peak AdamW learning rate |
| `--min-lr` | `1e-5` | Minimum LR for cosine schedule |
| `--warmup-steps` | `100` | Linear warmup steps |
| `--lr-schedule` | `cosine` | `constant`, `linear`, or `cosine` |
| `--weight-decay` | `0.01` | AdamW weight decay |
| `--max-grad-norm` | `1.0` | Gradient clipping norm; `0` disables clipping |
| `--checkpoint-every` | `500` | Save periodic checkpoints; `0` disables |
| `--log-every` | `10` | Log step loss every N steps |
| `--mtp-layers` | `1` | Qwen predictor layers when creating from target |
| `--assistant-layers` | `2` | Gemma assistant layers when creating from target |
| `--num-assistant-tokens` | `6` | Gemma `generation_config.json` assistant token count |

## Outputs

The output directory contains `config.json`, `generation_config.json` where applicable,
`model.safetensors`, `optimizer.safetensors`, and `metadata.json`. Periodic checkpoints
are written under `checkpoints/step_<N>` when `--checkpoint-every` is enabled.
