# pmetal train-mtp

Train MTP checkpoints for exact speculative decoding.

`train-mtp` supports two checkpoint families:

- Gemma 4 assistant checkpoints (`model_type = "gemma4_assistant"`) for `pmetal infer --draft-model`.
- Qwen3Next/Qwen3.6 `mtp.*` predictor checkpoints for `pmetal infer --mtp --mtp-model`.

The target model stays frozen; only the predictor is trained, on token shards from [`pmetal tokenize`](/cli/tokenize/).

Ctrl-C (SIGINT) or SIGTERM finishes the step in flight, saves the checkpoint and exits with status 130.

## Usage

```bash
pmetal train-mtp \
  --model <TARGET_MODEL> \
  --shards <TOKEN_SHARD>[,<TOKEN_SHARD>...] \
  [OPTIONS]
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `-m`, `--model` | required | Frozen target model ID or local path |
| `--family` | `auto` | `auto` (detect from the target's config), `qwen3-next` or `gemma4` |
| `-s`, `--shards` | required | Tokenized shard files from `pmetal tokenize` (comma-separated or repeated) |
| `-o`, `--output` | `./mtp-output` | Output checkpoint directory |
| `--init-mtp` | none | Existing assistant/MTP checkpoint to continue from |
| `--resume` | none | Continue an interrupted run: a checkpoint directory (`<output>/checkpoints/step_N`) or the output directory (its furthest checkpoint). Restores the predictor's weights, the optimizer state (moments and f32 master copies) and the step, and reads the shards from that step on. Not with `--init-mtp` |
| `--seq-len` | `512` | Sequence length per sample |
| `--batch-size` | `1` | Training batch size |
| `--steps` | `1000` | Optimizer steps |
| `--learning-rate` | `2e-4` | Peak AdamW learning rate |
| `--min-lr` | `1e-5` | Minimum LR for the decaying schedules |
| `--warmup-steps` | `100` | Linear warmup steps |
| `--lr-schedule` | `cosine` | `cosine`, `linear`, `constant`, `cosine_with_restarts`, `polynomial` or `wsd` |
| `--weight-decay` | `0.01` | AdamW weight decay |
| `--max-grad-norm` | `1.0` | Gradient clipping norm; `0` disables clipping |
| `--checkpoint-every` | `500` | Save periodic checkpoints; `0` disables |
| `--log-every` | `10` | Log step loss every N steps; `0` disables step logs |
| `--eos-token-id` | `0` | EOS token ID the shard reader inserts between documents |
| `--mtp-layers` | `1` | Qwen predictor layers to create when `--init-mtp` is omitted |
| `--assistant-layers` | `2` | Gemma assistant layers to create when `--init-mtp` is omitted |
| `--num-assistant-tokens` | `6` | Gemma `generation_config.json` assistant token count |
| `--seed` | `42` | Random seed |

## Examples

```bash
# Prepare token shards from JSONL rows with a "text" column.
pmetal tokenize \
  --input train.jsonl \
  --output ./tok \
  --tokenizer /path/to/qwen3next-target

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

Tokenize with the tokenizer of the target model: the shards hold token IDs, and a different vocabulary trains a predictor for the wrong one.

## Outputs

The output directory contains `config.json`, `generation_config.json` where applicable,
`model.safetensors`, `optimizer.safetensors`, and `metadata.json`. Periodic checkpoints
are written under `checkpoints/step_<N>` when `--checkpoint-every` is enabled.

## See Also

- [pmetal tokenize](/cli/tokenize/): Build the token shards
- [pmetal infer](/cli/infer/): Run the trained predictor (`--mtp`, `--draft-model`)
- [pmetal train-draft](/cli/train-draft/): DFlash draft checkpoints
