# Getting Started

Get up and running with PMetal in minutes — install, train your first model, and run inference.

PMetal is a complete machine learning platform for Apple Silicon — from low-level Metal GPU kernels and Apple Neural Engine integration to high-level training APIs, a terminal TUI, and a full desktop GUI.

## Prerequisites

- **macOS** on Apple Silicon (M1 or later)
- **Xcode Command Line Tools**: `xcode-select --install`

For building from source, also install:
- **Rust 1.86+** via [rustup](https://rustup.rs/)
- **Metal Toolchain**: `xcodebuild -downloadComponent MetalToolchain`
- **CMake**: `brew install cmake`

## Quick Install

```bash
# Option 1: Prebuilt binary
curl -fsSL https://github.com/Epistates/pmetal/releases/latest/download/pmetal-aarch64-apple-darwin.tar.gz | tar xz
sudo mv pmetal /usr/local/bin/

# Option 2: Install from crates.io
cargo install pmetal
```

See [Installation](/installation/) for all options including building from source and GUI setup.

## Your First Training Run

Fine-tune a model with LoRA in one command:

```bash
pmetal train \
  --model Qwen/Qwen3-0.6B \
  --dataset train.jsonl \
  --output ./output \
  --lora-r 16 --batch-size 4 --learning-rate 2e-4
```

PMetal automatically downloads the model from HuggingFace Hub, detects your hardware capabilities, and tunes kernel parameters for your specific chip.

## Run Inference

Chat with your fine-tuned model:

```bash
pmetal infer \
  --model Qwen/Qwen3-0.6B \
  --lora ./output/lora_weights.safetensors \
  --prompt "Explain quantum entanglement" \
  --chat --show-thinking
```

## Use the SDK

Integrate PMetal into your own Rust applications:

```rust
use pmetal::easy;

let result = easy::finetune("Qwen/Qwen3-0.6B", "train.jsonl")
    .lora(16, 32.0)
    .learning_rate(2e-4)
    .epochs(3)
    .output("./output")
    .run()
    .await?;
```

Or from Python:

```python
import pmetal

result = pmetal.finetune(
    "Qwen/Qwen3-0.6B",
    "train.jsonl",
    lora_r=16,
    learning_rate=2e-4,
    epochs=3,
)
```

## Explore Further

- **[CLI Reference](/cli/train/)** — All 21 commands
- **[Rust SDK](/sdk/easy-api/)** — Builder API reference
- **[Python SDK](/python/quick-start/)** — PyO3 bindings
- **[Training Methods](/training/overview/)** — SFT, DPO, GRPO, distillation, and more
- **[Supported Models](/models/supported/)** — All supported architectures
- **[Hardware](/hardware/apple-silicon/)** — Chip detection and kernel tuning
