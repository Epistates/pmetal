# pmetal-py

Python bindings for PMetal via PyO3.

## Overview

This crate provides a `cdylib` Python extension module exposing PMetal's training, inference, and model management APIs to Python. Built with PyO3 and installable via maturin.

## Installation

```bash
cd crates/pmetal-py
pip install maturin
maturin develop --release
```

## Quick Start

### Easy API

```python
import pmetal

# Fine-tune with sensible defaults
result = pmetal.finetune(
    "Qwen/Qwen3-0.6B",
    "train.jsonl",
    lora_r=16,
    learning_rate=2e-4,
    epochs=3,
)
print(f"Loss: {result['final_loss']}, Steps: {result['total_steps']}")

# Inference
text = pmetal.infer("Qwen/Qwen3-0.6B", "What is 2+2?")
print(text)
```

### Full Control

```python
import pmetal

# Configure training
lora_config = pmetal.LoraConfig(r=16, alpha=32.0)
training_config = pmetal.TrainingConfig(
    learning_rate=2e-4,
    num_epochs=3,
    batch_size=4,
    max_seq_len=2048,
)

# Create and run trainer
trainer = pmetal.Trainer(
    model_id="Qwen/Qwen3-0.6B",
    lora_config=lora_config,
    training_config=training_config,
    dataset_path="train.jsonl",
)
trainer.add_callback(pmetal.ProgressCallback())
result = trainer.train()

# Load model for inference
model = pmetal.Model.load("Qwen/Qwen3-0.6B")
print(model.generate("Hello world", temperature=0.7))
```

## API Reference

### Module-Level Functions

| Function | Description |
|----------|-------------|
| `finetune(model_id, dataset_path, ...)` | Fine-tune with sensible defaults |
| `infer(model_id, prompt, ...)` | Run inference |
| `download_model(model_id, ...)` | Download from HuggingFace Hub |
| `download_file(url, ...)` | Download a file |

### Classes

| Class | Description |
|-------|-------------|
| `Model` | Model loading and inference (`Model.load()`, `model.generate()`) |
| `Trainer` | Training orchestration (`Trainer(...)`, `trainer.train()`) |
| `Tokenizer` | Tokenization (`Tokenizer.from_file()`) |
| `LoraConfig` | LoRA configuration (r, alpha, dropout, use_rslora, use_dora) |
| `TrainingConfig` | Training hyperparameters (learning_rate, batch_size, num_epochs, ...) |
| `GenerationConfig` | Generation parameters (max_tokens, temperature, top_k, top_p, ...) |
| `DataLoaderConfig` | Data loading parameters (batch_size, max_seq_len, shuffle, ...) |
| `ProgressCallback` | Progress bar callback |
| `LoggingCallback` | Logging callback |
| `MetricsJsonCallback` | JSONL metrics callback |

### Enums

| Enum | Values |
|------|--------|
| `Dtype` | Float32, Float16, BFloat16 |
| `Quantization` | None, NF4, FP4, Int8 |
| `LoraBias` | None, All, LoraOnly |
| `LrSchedulerType` | Constant, Linear, Cosine, CosineWithRestarts, Polynomial |
| `OptimizerType` | AdamW, Adam8bit, ScheduleFree |
| `DatasetFormat` | Auto, Simple, Alpaca, ShareGpt, OpenAi, Reasoning |
| `ModelArchitecture` | Llama, Qwen2, Qwen3, Gemma, Mistral, Phi, ... |

## Modules

| Module | Description |
|--------|-------------|
| `easy` | Top-level `finetune()` and `infer()` functions |
| `model` | `Model` class with `load()` and `generate()` |
| `trainer` | `Trainer` class with `train()` and callbacks |
| `config` | Configuration classes and enums |
| `callbacks` | Training callback classes |
| `tokenizer` | `Tokenizer` class |
| `hub` | HuggingFace Hub download functions |
| `array_bridge` | MLX array to Python conversion |
| `error` | Error handling and conversion |

## License

MIT OR Apache-2.0
