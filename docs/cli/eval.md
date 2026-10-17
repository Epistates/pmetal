# pmetal eval

Evaluate model perplexity on a dataset.

Evaluate a model's perplexity on a dataset to measure generation quality.

## Usage

```bash
pmetal eval \
  --model <MODEL> \
  --dataset <DATASET> \
  [OPTIONS]
```

## Examples

```bash
# Evaluate perplexity
pmetal eval \
  --model Qwen/Qwen3-0.6B \
  --dataset eval.jsonl

# Evaluate with LoRA adapter
pmetal eval \
  --model Qwen/Qwen3-0.6B \
  --dataset eval.jsonl \
  --lora ./output/adapter_model.safetensors
```

## See Also

- [pmetal train](/cli/train/) — Train a model
- [pmetal infer](/cli/infer/) — Run inference
