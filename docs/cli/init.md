# pmetal init

Generate a sample YAML training configuration file.

Writes a configuration with a model, LoRA settings, training settings and a dataset path, to start from. Edit it and train with `pmetal train --config <file>`.

## Usage

```bash
pmetal init [OPTIONS]
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `-o`, `--output` | `config.yaml` | Output path for the config file |

## Examples

```bash
# Generate the default config
pmetal init

# Specify the output path, then train from it
pmetal init --output my-config.yaml
pmetal train --config my-config.yaml
```

## See Also

- [Configuration File](/configuration/config-file/): Full config documentation
- [pmetal train](/cli/train/): Train from a config file
