# pmetal fuse

Fuse LoRA adapter weights into the base model.

Merge LoRA adapter weights into the base model weights, producing a standalone model without adapter overhead.

## Usage

```bash
pmetal fuse \
  --model <BASE_MODEL> \
  --lora <LORA_WEIGHTS> \
  [OPTIONS]
```

## Examples

```bash
# Fuse LoRA weights
pmetal fuse \
  --model Qwen/Qwen3-0.6B \
  --lora ./output/adapter_model.safetensors

# Accurate mode (higher precision)
pmetal fuse \
  --model Qwen/Qwen3-0.6B \
  --lora ./output/adapter_model.safetensors \
  --accurate
```

## What is folded

Both modes use the same streaming fold: the base model is memory-mapped and each adapted weight
becomes `W + s·B·A` in f64, with `s = alpha / r` (`alpha / sqrt(r)` for rsLoRA adapters). DoRA
adapters renormalize each output row to their magnitude vector, MLX-quantized bases are
dequantized, folded and quantized again, and adapters from `pmetal train` and from PEFT are both
read. The scale comes from the adapter's `adapter_config.json`; `--alpha` and `--rank` override it.
An adapter whose modules do not match the base model is an error.

| Mode | Description |
|------|-------------|
| Standard | Whole matrices folded at once |
| Accurate (`--accurate`, `--low-memory`) | The same fold; `--low-memory` tiles it to bound memory |

## See Also

- [pmetal train](/cli/train/) — Generate LoRA weights
- [pmetal quantize](/cli/quantize/) — Quantize the fused model
