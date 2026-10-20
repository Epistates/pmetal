# pmetal dashboard

Real-time training metrics visualization.

Opens the [TUI](/cli/tui/) on its Monitor tab, following a training run's metrics file. Shows live loss curves, learning rate schedules, throughput sparklines, and timing breakdowns. The other tabs are available from there, so a run can be started, steered and watched in one place.

## Usage

```bash
pmetal dashboard [OPTIONS]
```

## Features

- Live loss curves rendered with braille characters
- Learning rate schedule visualization
- Throughput sparklines (tokens/second)
- Timing breakdown gauges (forward pass, backward pass, optimizer step)
- Memory usage monitoring
- Follows a run started from the Training tab (or any `--log-metrics` file) and reports when the loss diverges

## See Also

- [pmetal tui](/cli/tui/): The full TUI
- [pmetal train](/cli/train/): Training commands
