# pmetal download

Download a model from HuggingFace Hub to the local cache.

The model goes to the Hugging Face hub cache, the same one every other command reads: `$HF_HUB_CACHE` when set, otherwise `$HF_HOME/hub`, otherwise `~/.cache/huggingface/hub`. A model that is already cached is not downloaded again. Safetensors weights, configs and tokenizer files are fetched; other weight formats (`.bin`, `.gguf`, ONNX, TensorFlow, Flax) and archives are skipped. The command prints the snapshot directory it downloaded to.

## Usage

```bash
pmetal download <MODEL> [--revision <REVISION>]
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `<MODEL>` | *required* | Model ID on the Hub, e.g. `Qwen/Qwen3-0.6B` |
| `--revision` | default branch | Branch, tag or commit to download |

## Examples

```bash
# Download a model
pmetal download Qwen/Qwen3-0.6B

# Download a specific revision
pmetal download Qwen/Qwen3-0.6B --revision main

# Download into a cache on another volume
HF_HUB_CACHE=/Volumes/Models/huggingface pmetal download Qwen/Qwen3-0.6B
```

## See Also

- [pmetal search](/cli/search/) — Search for models
