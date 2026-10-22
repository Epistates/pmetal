# pmetal train-diffusion

LoRA fine-tune a DiffusionGemma block-diffusion model.

Trains LoRA adapters on the encoder and the decoder of a DiffusionGemma model (dense or mixture-of-experts) with the denoising loss. Every linear layer but the output head is adapted unless `--lora-targets` names a subset; MoE routers and experts are never adapted. With `--quantization` the frozen base is packed (QLoRA), which cuts the memory of the 26B MoE model a great deal. The adapter is written in the PEFT layout.

Ctrl-C (SIGINT) or SIGTERM finishes the step in flight, saves the adapters and exits with status 130.

## Usage

```bash
pmetal train-diffusion \
  --model <MODEL> \
  --dataset <DATASET> \
  [OPTIONS]
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | required | DiffusionGemma model ID or local path |
| `--dataset` | required | JSONL dataset; each line has `prompt` and `response` (aliases `context`/`target` and `input`/`output`) |
| `-o`, `--output` | `./diffusion-lora-output` | Output directory for the trained LoRA adapters |
| `--lora-r` | `16` | LoRA rank |
| `--lora-alpha` | `32.0` | LoRA alpha (scaling is alpha / r) |
| `--lora-targets` | every linear layer but the output head | Comma-separated projections to adapt, by name (`q_proj`, `gate_proj`, ...), in the encoder and the decoder |
| `--steps` | `1000` | Total optimizer steps (the dataset is cycled to reach this count) |
| `--learning-rate` | `2e-4` | AdamW learning rate |
| `--weight-decay` | `0.0` | AdamW weight decay |
| `--max-grad-norm` | `1.0` | Gradient clipping norm; `0` disables |
| `--corrupted-only` | `true` | Restrict the denoising loss to corrupted canvas positions (the ELBO reconstruction term); `false` averages over the whole canvas. Takes the value: `--corrupted-only false` |
| `--quantization` | `none` | QLoRA base format: `none`, `nf4`, `fp4` or `int8`, as for `pmetal train`. Packs every projection but the routers, and the MoE experts (4-bit affine for `nf4` and `fp4`, 8-bit for `int8`), so only the f32 LoRA adapters train |
| `--quant-block-size` | `64` | Weights per scale for `nf4` and `int8` (32, 64 or 128; `fp4`'s block is 16) |
| `--double-quant` | `false` | Pack `nf4`'s absmax values to 8 bits as well |
| `--max-context-len` | `512` | Truncate each tokenized prompt to at most this many context tokens |
| `--checkpoint-every` | `0` | Save an intermediate adapter checkpoint every N steps; `0` disables |
| `--log-every` | `10` | Log every N steps; `0` disables step logs |
| `--seed` | `42` | Random seed |

## Examples

```bash
# LoRA on the 26B MoE model, with a 4-bit base
pmetal train-diffusion \
  --model google/diffusiongemma-26B-A4B-it \
  --dataset train.jsonl \
  --quantization nf4 \
  --output ./diffusion-lora \
  --steps 500
```

A dataset row:

```json
{"prompt": "What is the capital of France?", "response": "Paris."}
```

## See Also

- [Training Overview](/training/overview/): All training methods
- [pmetal train](/cli/train/): LoRA and QLoRA for autoregressive models
