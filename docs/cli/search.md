# pmetal search

Search the HuggingFace Hub for models, with a memory-fit estimate for your hardware.

Each result shows its size, downloads, likes and whether it fits in your device's unified memory.

## Usage

```bash
pmetal search <QUERY> [OPTIONS]
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `<QUERY>` | required | Search query, e.g. `"qwen3 0.6B"` or `"llama 8b"` |
| `-l`, `--limit` | `15` | Maximum number of results |
| `--detailed` | `false` | Detailed fit analysis: memory breakdown, accounting for GPU/ANE allocation, and training estimates |
| `--download` | `false` | Download the first result that fits on your device |
| `--json` | `false` | Print the results as JSON |
| `--tag` | none | Only models with this tag (repeatable): `gguf`, `mlx`, `moe`, `license:apache-2.0`, ... |
| `--author` | none | Only models from this user or organization |
| `--pipeline-tag` | `text-generation` | Pipeline tag to search; `any` searches every kind of model |
| `--sort` | `downloads` | Result order: `downloads` (last 30 days), `likes`, `trending` or `updated` |

## Examples

```bash
# Basic search
pmetal search "qwen 0.6b"

# Detailed view with memory estimates
pmetal search "qwen 0.6b" --detailed

# More results
pmetal search "llama 3" --limit 20

# One organization's GGUF models, newest first
pmetal search "qwen3" --author Qwen --tag gguf --sort updated

# Fetch the first model that fits this machine
pmetal search "qwen3 0.6b" --download
```

## See Also

- [pmetal download](/cli/download/): Download a model
- [Supported Models](/models/supported/): Compatible architectures
