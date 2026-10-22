# pmetal fuse

Fuse LoRA adapter weights into the base model.

Merges LoRA adapter weights into the base model weights, producing a standalone model without adapter overhead. The output directory is a complete model that loads like any other (and can go to [`pmetal quantize`](/cli/quantize/) or [`pmetal upload`](/cli/upload/)).

## Usage

```bash
pmetal fuse \
  --model <BASE_MODEL> \
  --lora <LORA_ADAPTER> \
  --output <OUTPUT_DIR> \
  [OPTIONS]
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `-m`, `--model` | required | Base model ID or local path |
| `-l`, `--lora` | required | LoRA adapter: a directory containing `adapter_model.safetensors`, or the file itself |
| `-o`, `--output` | required | Output directory for the fused model |
| `--alpha` | from the adapter | LoRA scaling alpha |
| `--rank` | from the adapter | LoRA rank |
| `--low-memory` | `false` | Tile the fold (512 rows at a time) to bound memory on very large tensors; the result is the same |

## Examples

```bash
# Fuse LoRA weights
pmetal fuse \
  --model Qwen/Qwen3-0.6B \
  --lora ./output \
  --output ./fused

# Bound memory on very large tensors
pmetal fuse \
  --model Qwen/Qwen3-0.6B \
  --lora ./output/adapter_model.safetensors \
  --output ./fused \
  --low-memory
```

## What is folded

The base model is memory-mapped and each adapted weight becomes `W + s·B·A` in f64, with
`s = alpha / r` (`alpha / sqrt(r)` for rsLoRA adapters). DoRA adapters renormalize each output row
to their magnitude vector, MLX-quantized bases are dequantized, folded and quantized again, and
adapters from `pmetal train` and from PEFT are both read. The scale comes from the adapter's
`adapter_config.json`; `--alpha` and `--rank` override it. An adapter whose modules do not match the
base model is an error. Every output file is written atomically, and tensors the adapter does not
touch are copied byte for byte.

A mixture-of-experts adapter holds one factor pair per expert (`lora_expert_A` / `lora_expert_B`).
Each pair is folded into its expert's weights, whether the base stores the experts one tensor each
(`mlp.experts.{e}.{proj}.weight`), fused (`mlp.experts.gate_up_proj`, `mlp.experts.down_proj`) or
stacked (`mlp.switch_mlp.{proj}.weight`). Quantized experts cannot take a fold and are refused.

## See Also

- [pmetal train](/cli/train/): Generate LoRA weights
- [pmetal quantize](/cli/quantize/): Quantize the fused model
- [pmetal upload](/cli/upload/): Publish the fused model
