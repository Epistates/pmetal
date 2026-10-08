# pmetal upload

Publish a model directory to HuggingFace Hub.

The repository is created if it does not exist, and the whole directory lands in one commit.
Large files (weights, GGUF exports) go to the Hub's large-file storage. `.git`, `.DS_Store`,
AppleDouble `._*` files and `.cache` directories are never sent.

## Usage

```bash
pmetal upload <PATH> <OWNER/NAME> [OPTIONS]
```

## Authentication

Uses `HF_TOKEN` if set, otherwise the token stored by `hf auth login`. The token needs write
access to the target namespace.

## Options

| Option | Description |
|--------|-------------|
| `--private` | Create the repository as private (no effect if it already exists) |
| `--revision <BRANCH>` | Branch to commit to (default: the repository's default branch) |
| `-m, --message <MSG>` | Commit message |
| `--create-pr` | Open a pull request instead of committing to the branch |
| `--exclude <GLOB>` | Leave out files matching a glob relative to `PATH`; repeatable |

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

- [pmetal download](/cli/download/) — Download a model
- [pmetal fuse](/cli/fuse/) — Fuse LoRA adapter weights into the base model
