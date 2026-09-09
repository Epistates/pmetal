# pmetal ollama

Ollama integration — generate Modelfiles and create Ollama models.

Integration with [Ollama](https://ollama.ai) for model deployment.

## Subcommands

### modelfile

Generate an Ollama Modelfile from a PMetal model.

```bash
pmetal ollama modelfile --base ./output --output Modelfile
```

### create

Create an Ollama model directly.

```bash
pmetal ollama create --name my-model --base ./output
```

### templates

List available chat templates.

```bash
pmetal ollama templates
```

## See Also

- [pmetal quantize](/cli/quantize/) — Quantize to GGUF for Ollama
- [pmetal fuse](/cli/fuse/) — Fuse LoRA before export
