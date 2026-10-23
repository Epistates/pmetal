# pmetal dashboard

Real-time training metrics visualization.

Opens the [TUI](/cli/tui/) on its Monitor tab, following a training run's metrics file. The other tabs are available from there, so a run can be started, steered and watched in one place. `pmetal dashboard` is the same program as `pmetal tui`, only started on a different tab.

## Usage

```bash
pmetal dashboard [OPTIONS]
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `-m`, `--metrics-file` | none | Training metrics JSONL file to follow (written by a trainer's `--log-metrics`) |

## Features

- Live loss curves rendered with braille characters
- Learning rate schedule visualization
- Throughput sparklines (tokens/second)
- Timing breakdown gauges (forward pass, backward pass, optimizer step)
- Memory usage monitoring
- Follows a run started from the Training tab (or any `--log-metrics` file) and reports when the loss diverges

## Examples

```bash
# Follow a run's metrics
pmetal train --model Qwen/Qwen3-0.6B --dataset train.jsonl --log-metrics metrics.jsonl
pmetal dashboard --metrics-file metrics.jsonl
```

## See Also

- [pmetal tui](/cli/tui/): The full TUI
- [pmetal train](/cli/train/): Training commands
