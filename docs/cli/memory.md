# pmetal memory

Show memory usage and available capacity.

Displays the memory of your Apple Silicon device as the runtime sees it: the total, what is in use, what is available for model loading and training, and the peak use. With `--json` these are `total_gb`, `used_gb`, `available_gb` and `peak_gb`.

## Usage

```bash
pmetal memory [--json]
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--json` | `false` | Print the report as JSON |

## Examples

```bash
pmetal memory
pmetal memory --json
```

## See Also

- [pmetal info](/cli/info/): Full device info
- [pmetal search](/cli/search/): Memory fit estimation for models
