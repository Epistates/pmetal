# pmetal pretrain

Pretrain a model from scratch: full-parameter training with no LoRA, on tokenized shards.

The model is built from an architecture name (and an optional config JSON) with fresh weights, then trained with AdamW on packed batches streamed from the shards. Build the shards with [`pmetal tokenize`](/cli/tokenize/).

Ctrl-C (SIGINT) or SIGTERM finishes the step in flight, writes the final checkpoint and exits with status 130. `--resume` continues from a checkpoint, including the optimizer state and the position in the shard stream.

## Usage

```bash
pmetal pretrain \
  --arch <ARCH> \
  --shards <SHARD>[,<SHARD>...] \
  [OPTIONS]
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `-a`, `--arch` | required | Architecture: `llama`, `qwen2`, `qwen3`, `qwen3.5` (also `qwen3.6`), `qwen3_moe`, `gemma`, `gemma4`, `mistral`, `phi3` (also `phi4`) or `gpt-oss` |
| `--model-config` | the architecture's defaults | Model config JSON that overrides the defaults (required for `gemma4`) |
| `-s`, `--shards` | required | Tokenized shard files (`.bin`), comma-separated or repeated; a shell glob such as `./tok/shard_*.bin` is expanded before the command runs |
| `--eval-shards` | none | Held-out shard files for evaluation |
| `--seq-len` | `2048` | Sequence length of the packed training batches |
| `--batch-size` | `4` | Batch size |
| `--gradient-accumulation-steps` | `1` | Gradient accumulation (effective batch is `batch_size` times this) |
| `--steps` | `10000` | Training steps |
| `--learning-rate` | `3e-4` | Peak learning rate |
| `--min-lr` | `1e-5` | Minimum learning rate (the cosine floor) |
| `--warmup-steps` | `1000` | Linear warmup steps |
| `--lr-schedule` | `cosine` | `constant`, `linear` or `cosine` |
| `--weight-decay` | `0.1` | AdamW weight decay |
| `--max-grad-norm` | `1.0` | Gradient clipping norm; `0` disables |
| `--z-loss` | `0.0` | MoE router z-loss coefficient; `0` disables |
| `--document-mask` | `false` | Mask attention to each document of a packed row, so a document never attends to the one before it (the architecture must take an attention mask; the dense mask costs more than the causal fast path) |
| `--eos-token-id` | `0` | EOS token ID inserted between documents in packed sequences |
| `-o`, `--output` | `./pretrain-output` | Output and checkpoint directory |
| `--checkpoint-every` | `1000` | Save a checkpoint every N steps; `0` disables |
| `--resume` | none | Resume from a checkpoint directory |
| `--log-every` | `10` | Log step, loss and learning rate every N steps; `0` disables |
| `--eval-every` | `0` | Evaluate on `--eval-shards` every N steps; `0` disables |
| `--eval-batches` | `10` | Batches per evaluation round |
| `--seed` | `42` | Random seed |

## Examples

```bash
# Tokenize a corpus, then pretrain a small Llama on it
pmetal tokenize --input corpus.jsonl --output ./tok --tokenizer Qwen/Qwen3-0.6B

pmetal pretrain \
  --arch llama \
  --model-config llama-small.json \
  --shards ./tok/shard_00000.bin \
  --eos-token-id 151643 \
  --seq-len 1024 --batch-size 8 --steps 20000 \
  --output ./pretrain-output

# Continue from a checkpoint
pmetal pretrain \
  --arch llama \
  --model-config llama-small.json \
  --shards ./tok/shard_00000.bin \
  --resume ./pretrain-output/step_10000
```

## Output

Each checkpoint is a directory under the output directory: `step_<N>/` every `--checkpoint-every` steps and `final/` when the run ends (or is stopped). It holds `model.safetensors`, `optimizer.safetensors` (AdamW moments and the f32 master copy of every half-precision parameter) and `metadata.json` (step, loss, learning rate, the position in the shard stream and the optimizer's counters).

## See Also

- [pmetal tokenize](/cli/tokenize/): Build the token shards
- [Training Overview](/training/overview/): All training methods
