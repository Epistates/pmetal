# pmetal train-draft

Train a DFlash block-diffusion draft checkpoint for the dedicated `pmetal dflash`
runtime.

The current DFlash training objective uses a frozen Qwen3 target and trains the
draft to reconstruct masked blocks from captured target hidden states.

## Usage

```bash
pmetal train-draft \
  --target <QWEN3_TARGET> \
  --draft-config <DFLASH_CONFIG_JSON> \
  --shards <TOKEN_SHARD>[,<TOKEN_SHARD>...] \
  --output <OUTPUT_DIR>
```

## Examples

```bash
pmetal tokenize \
  --input train.jsonl \
  --output ./tok \
  --tokenizer Qwen/Qwen3-0.6B

pmetal train-draft \
  --target Qwen/Qwen3-0.6B \
  --draft-config dflash_config.json \
  --shards ./tok/shard_00000.bin \
  --output ./dflash-draft \
  --seq-len 512 --batch-size 1 --steps 1000

pmetal dflash \
  --target Qwen/Qwen3-0.6B \
  --draft ./dflash-draft \
  --prompt "Summarize monotonic queues."
```

Use `--draft <checkpoint>` instead of `--draft-config` to continue from an existing
DFlash checkpoint.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--target` | required | Frozen Qwen3 target model ID or local path |
| `--draft` | — | Existing DFlash checkpoint to continue training |
| `--draft-config` | — | DFlash config JSON used when `--draft` is omitted |
| `--shards` | required | Tokenized shard files from `pmetal tokenize` |
| `--output` | `./draft-output` | Output checkpoint directory |
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

## Outputs

The output directory contains `config.json`, `model.safetensors`,
`optimizer.safetensors`, and `metadata.json`. Periodic checkpoints are written under
`checkpoints/step_<N>` when `--checkpoint-every` is enabled.
