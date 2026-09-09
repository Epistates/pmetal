# pmetal search

Search HuggingFace Hub with memory fit estimation for your hardware.

Search the HuggingFace Hub for models. Shows memory fit estimation based on your Apple Silicon hardware.

## Usage

```bash
pmetal search <QUERY> [OPTIONS]
```

## Examples

```bash
# Basic search
pmetal search "qwen 0.6b"

# Detailed view with memory estimates
pmetal search "qwen 0.6b" --detailed

# Filter by model type
pmetal search "llama 3" --limit 20
```

## Output

The `--detailed` flag shows memory requirements and whether the model fits in your device's unified memory, accounting for GPU/ANE allocation and training overhead.

## See Also

- [pmetal download](/cli/download/) — Download a model
- [Supported Models](/models/supported/) — Compatible architectures
