# pmetal verify

Check that a model directory is complete and undamaged.

A download that stopped before the last shard leaves a directory that looks like a model and fails
when a command loads it. `pmetal verify` reads only the file headers (a few kilobytes per file,
however large the weights) and says what is missing.

## Usage

```bash
pmetal verify <MODEL> [--json]
```

`MODEL` is a model directory, or a Hub model ID with a copy in the Hub cache.

## What is checked

- Weights of one format (safetensors, GGUF or PyTorch) are present, and every shard the weight
  index names exists.
- Each safetensors file is exactly as long as its header says; each GGUF set is complete, with every
  tensor inside its file.
- `config.json` is readable JSON (GGUF carries its own configuration, so it needs none).

## Output

A summary of the format, the files, their size and the tensor count, then each problem and warning,
and a verdict. With `--json`, the same as a JSON object (`complete`, `loadable`, `format`,
`weight_files`, `weight_bytes`, `tensors`, `problems`, `warnings`).

The exit status is `0` for a complete model and `1` for an incomplete or damaged one, so
`pmetal verify ./model && pmetal infer ...` works in a script.

## See Also

- [pmetal download](/cli/download/): fetch a model (it verifies what it downloads)
- [pmetal info](/cli/info/): the device and the build
