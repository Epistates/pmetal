# Getting Started

Install PMetal, fine-tune a model, run it and serve it.

PMetal trains and runs language models on the GPU and Neural Engine of an Apple Silicon Mac. The same job specifications drive a command line, a terminal UI, a desktop app, an MCP server, Python bindings and a Rust SDK.

## Prerequisites

- **macOS 14 (Sonoma) or later** on Apple Silicon (M1 or later)

For building from source (`cargo install`, `brew install` and the repository), also:
- **Xcode Command Line Tools**: `xcode-select --install`
- **Rust 1.95+** via [rustup](https://rustup.rs/)
- **Metal Toolchain**: `xcodebuild -downloadComponent MetalToolchain`
- **CMake**: `brew install cmake`

## Quick Install

```bash
# Option 1: Homebrew
brew install epistates/tap/pmetal

# Option 2: crates.io (serve and mcp are opt-in features)
cargo install pmetal --features serve,mcp
```

A prebuilt binary and the desktop app are on the [Releases](https://github.com/Epistates/pmetal/releases) page. See [Installation](/installation/) for every option, including building from source and GUI setup.

## Your First Training Run

Fine-tune a model with LoRA in one command:

```bash
pmetal train \
  --model Qwen/Qwen3-0.6B \
  --dataset train.jsonl \
  --output ./output \
  --lora-r 16 --batch-size 4 --learning-rate 2e-4
```

PMetal downloads the model from the Hugging Face Hub on first use and detects your chip. The adapter is written to `./output` as `adapter_model.safetensors` and `adapter_config.json`, the layout PEFT reads. Training data is JSONL (or JSON, Parquet, Arrow, CSV) in any of the [formats](/training/overview/#dataset-formats) PMetal detects from the columns.

## Run Inference

Chat with your fine-tuned model:

```bash
pmetal infer \
  --model Qwen/Qwen3-0.6B \
  --lora ./output/adapter_model.safetensors \
  --prompt "Explain quantum entanglement" \
  --chat
```

Thinking models show their reasoning; `--hide-thinking` hides it and `--no-thinking` turns it off.

## Serve a model

```bash
pmetal serve --model Qwen/Qwen3-0.6B --port 8080
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "Qwen/Qwen3-0.6B", "messages": [{"role": "user", "content": "Hello"}]}'
```

The server speaks the OpenAI Chat Completions, Completions, Responses and Embeddings APIs and the Anthropic Messages API, and listens on loopback unless you pass `--host`.

## Use the SDK

Integrate PMetal into your own Rust applications:

```rust
use pmetal::trainer::orchestrator::{TrainingJobConfig, run_training};

let config = TrainingJobConfig {
    model_id: "Qwen/Qwen3-0.6B".to_string(),
    dataset: "train.jsonl".to_string(),
    output_dir: "./output".to_string(),
    ..Default::default()
};

let result = run_training(config, None, Vec::new()).await?;
println!("final loss {:.4}", result.final_loss);
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

- **[CLI Reference](/cli/train/)**: Command reference
- **[Rust SDK](/sdk/advanced/)**: Crate-level API reference
- **[Python SDK](/python/quick-start/)**: PyO3 bindings
- **[Training Methods](/training/overview/)**: SFT, DPO, GRPO, distillation, and more
- **[Supported Models](/models/supported/)**: All supported architectures
- **[Hardware](/hardware/apple-silicon/)**: Chip detection and kernel tuning
