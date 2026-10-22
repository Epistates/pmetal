# pmetal upload

Publish a model directory to the HuggingFace Hub.

The repository is created if it does not exist, and the whole directory lands in one commit.
Large files (weights, GGUF exports) go to the Hub's large-file storage. `.git`, `.DS_Store`,
AppleDouble `._*` files and `.cache` directories are never sent.

Authentication uses `HF_TOKEN` if set, otherwise the token stored by `hf auth login`. The token needs write
access to the target namespace.

## Usage

```bash
pmetal upload <PATH> <OWNER/NAME> [OPTIONS]
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `<PATH>` | required | Local directory to publish (a fused model, an adapter, a quantized export) |
| `<OWNER/NAME>` | required | Target repository |
| `--private` | `false` | Create the repository as private (no effect if it already exists) |
| `--revision` | the default branch | Branch to commit to |
| `-m`, `--message` | none | Commit message |
| `--create-pr` | `false` | Open a pull request instead of committing to the branch |
| `--exclude` | none | Leave out files matching a glob relative to `PATH`; repeatable |

## Examples

```bash
# Publish a fused model
pmetal upload ./output/fused me/qwen3-0.6b-support

# Publish an adapter privately, without its intermediate checkpoints
pmetal upload ./output me/qwen3-lora --private --exclude "checkpoints/**"

# Propose an update to an existing repository
pmetal upload ./output/fused me/qwen3-0.6b-support --create-pr -m "Retrain on v2 data"
```

## See Also

- [pmetal download](/cli/download/): Download a model
- [pmetal fuse](/cli/fuse/): Fuse LoRA adapter weights into the base model
- [pmetal verify](/cli/verify/): Check a model directory before publishing
