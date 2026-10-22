# pmetal eval

Evaluate a model's perplexity on a dataset.

Perplexity is `exp` of the average negative log-likelihood per token of the dataset's text, rendered with the model's chat template and cut at `--max-seq-len`. Lower is better. The dataset is read in the formats `pmetal train` reads; `--lora` scores a model with an adapter on top, to compare it with the base model on the same data.

## Usage

```bash
pmetal eval \
  --model <MODEL> \
  --dataset <DATASET> \
  [OPTIONS]
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `-m`, `--model` | required | Model ID or local path |
| `-d`, `--dataset` | required | Dataset: a file or a Hub dataset ID |
| `--lora` | none | LoRA adapter: a directory written by `pmetal train`, or its weights file |
| `--max-seq-len` | `1024` | Maximum sequence length |
| `--num-samples` | `0` | Samples to evaluate; `0` evaluates all |
| `--json` | `false` | Print the result as JSON (`model`, `dataset`, `num_samples`, `total_tokens`, `avg_nll`, `perplexity`) |

## Examples

```bash
# Evaluate perplexity
pmetal eval \
  --model Qwen/Qwen3-0.6B \
  --dataset eval.jsonl

# Evaluate with a LoRA adapter
pmetal eval \
  --model Qwen/Qwen3-0.6B \
  --dataset eval.jsonl \
  --lora ./output
```

## See Also

- [pmetal train](/cli/train/): Train a model
- [pmetal infer](/cli/infer/): Run inference
