# pmetal download

Download a model from the HuggingFace Hub to the local cache.

The model goes to the Hugging Face hub cache, the same one every other command reads: `$HF_HUB_CACHE` when set, otherwise `$HF_HOME/hub`, otherwise `~/.cache/huggingface/hub`. A command that is given a Hub ID downloads the model itself, so running `download` first is only needed to fetch ahead of time, to pick a weight format, or to get the snapshot directory for a command that takes a local path.

The command fetches the configs, the tokenizer and one set of weights from a single commit, then checks them. Other frameworks' weights (ONNX, TensorFlow, Flax) and archives are skipped. A model that is already cached is not downloaded again. The command prints the snapshot directory it downloaded to.

Authentication uses `HF_TOKEN`, or the token stored by `hf auth login`. `HF_ENDPOINT` selects a mirror and `HF_HUB_OFFLINE=1` is the same as `--offline`.

## Usage

```bash
pmetal download <MODEL> [OPTIONS]
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `<MODEL>` | required | Model ID on the Hub, e.g. `Qwen/Qwen3-0.6B`. A GGUF repository takes a quantization suffix: `org/model-GGUF:Q4_K_M` |
| `--revision` | the cached copy, else the default branch | Branch, tag or commit to download |
| `--format` | `auto` | Weight format to fetch: `auto` (safetensors, else GGUF), `safetensors`, `gguf` (one quantization) or `pytorch` (`pytorch_model*.bin` pickles, read with a restricted unpickler and written as safetensors next to them) |
| `--quant` | none | GGUF quantization to fetch (e.g. `Q4_K_M`, `Q8_0`); implies `--format gguf` |
| `--verify-sha256` | `false` | Also hash large files against the Hub's SHA-256 digests (reads them back from disk) |
| `--dry-run` | `false` | Show the commit, the files and their total size, then stop |
| `--offline` | `false` | Use only the local cache; never touch the network |

## Examples

```bash
# Download a model
pmetal download Qwen/Qwen3-0.6B

# See what would be fetched, and how large it is
pmetal download Qwen/Qwen3-0.6B --dry-run

# Download a specific revision
pmetal download Qwen/Qwen3-0.6B --revision main

# One GGUF quantization of a GGUF repository
pmetal download Qwen/Qwen3-0.6B-GGUF --quant Q4_K_M

# Download into a cache on another volume
HF_HUB_CACHE=/path/to/huggingface/hub pmetal download Qwen/Qwen3-0.6B
```

## See Also

- [pmetal search](/cli/search/): Search for models
- [pmetal verify](/cli/verify/): Check a downloaded model directory
- [pmetal upload](/cli/upload/): Publish a model directory
