# pmetal ollama

Generate Modelfiles for registering a model with Ollama.

The commands write a Modelfile (and print the `ollama create` command that registers it); they do not run Ollama themselves. The base is a GGUF file (see [`pmetal quantize`](/cli/quantize/)) or the name of a model Ollama already has. An output path may not contain `..`.

## Usage

```bash
pmetal ollama modelfile --base <BASE> [OPTIONS]
pmetal ollama create --name <NAME> --base <BASE> [OPTIONS]
pmetal ollama templates
```

## pmetal ollama modelfile

Generate a Modelfile for a trained model.

| Option | Default | Description |
|--------|---------|-------------|
| `-b`, `--base` | required | Base model: a GGUF path or an Ollama model name |
| `--lora` | none | LoRA adapter path |
| `-o`, `--output` | `Modelfile` | Output Modelfile path |
| `--system` | none | System prompt |
| `--temperature` | none | Temperature (0.0 to 2.0) |
| `--num-ctx` | none | Context window size |
| `--top-k` | none | Top-k sampling |
| `--top-p` | none | Top-p nucleus sampling |
| `--template` | detected from the base's name | Chat template: `llama3`, `qwen3`, `gemma`, `mistral`, `phi3` or `deepseek` |
| `--license` | none | License text for the model |

## pmetal ollama create

Generate a Modelfile for a named model, and print the command that registers it with Ollama (`ollama create <name> -f <Modelfile>`).

| Option | Default | Description |
|--------|---------|-------------|
| `-n`, `--name` | required | Model name for Ollama (letters, digits, `-`, `_`, `.`, `/` and `:` only) |
| `-b`, `--base` | required | Base model: a GGUF path or an Ollama model name |
| `--lora` | none | LoRA adapter path |
| `--system` | none | System prompt |
| `--temperature` | none | Temperature (0.0 to 2.0) |
| `--num-ctx` | none | Context window size |
| `-o`, `--output` | `Modelfile` | Output Modelfile path |
| `--template` | detected from the base's name | Chat template, as above |

## pmetal ollama templates

List the available chat templates and the model families each is for.

## Examples

```bash
# Write a Modelfile for a quantized model
pmetal ollama modelfile --base ./model.gguf --output Modelfile --system "You are a helpful assistant."

# The same for a named model, then register it
pmetal ollama create --name my-model --base ./model.gguf
ollama create my-model -f Modelfile

# List the templates
pmetal ollama templates
```

## See Also

- [pmetal quantize](/cli/quantize/): Quantize to GGUF
- [pmetal fuse](/cli/fuse/): Fuse LoRA before export
