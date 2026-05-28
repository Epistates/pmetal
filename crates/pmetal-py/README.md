# pmetal-py

Python bindings for PMetal via PyO3.

## Overview

This crate provides the `pmetal` Python extension module for model loading,
inference, LoRA fine-tuning, MTP assistant training, DFlash draft training, and
HuggingFace Hub model management on Apple Silicon.

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

result = pmetal.finetune(
    "Qwen/Qwen3-0.6B",
    "train.jsonl",
    lora_r=16,
    learning_rate=2e-4,
    epochs=3,
)
print(result["final_loss"], result["total_steps"])

text = pmetal.infer(
    "Qwen/Qwen3-0.6B",
    "What is 2+2?",
    max_tokens=64,
    temperature=0.0,
)
print(text)

run = pmetal.infer_with_metrics(
    "Qwen/Qwen3.6-Coder-30B-A3B-Instruct",
    "Write a short Rust function.",
    mtp=True,
    mtp_draft_tokens=3,
)
print(run["text"])
print(run["speculative"])
```

### Full Control

```python
import pmetal

lora_config = pmetal.LoraConfig(r=16, alpha=32.0)
training_config = pmetal.TrainingConfig(
    learning_rate=2e-4,
    num_epochs=3,
    batch_size=4,
    max_seq_len=2048,
)

trainer = pmetal.Trainer(
    model_id="Qwen/Qwen3-0.6B",
    lora_config=lora_config,
    training_config=training_config,
    dataset_path="train.jsonl",
)
trainer.add_callback(pmetal.ProgressCallback(100))
trainer.add_callback(pmetal.LoggingCallback(log_every=10))
result = trainer.train()

model = pmetal.Model.load("Qwen/Qwen3-0.6B")
print(model.generate("Hello world", temperature=0.7))
```

## API Reference

### Module-Level Functions

| Function | Description |
|----------|-------------|
| `finetune(model_id, dataset_path, ...)` | LoRA fine-tuning with sensible defaults |
| `infer(model_id, prompt, ...)` | Shared-runner inference with LoRA, FP8, MTP, KV cache, chat, and sampling options |
| `infer_with_metrics(model_id, prompt, ...)` | Inference plus generation and speculative decoding metrics |
| `train_mtp(model, shards, ...)` | Train Qwen3Next/Qwen3.6 MTP predictors or Gemma 4 assistants from tokenized shards |
| `train_draft(target, shards, ...)` | Train a DFlash draft model for a Qwen3 target |
| `download_model(model_id, revision=None)` | Download a HuggingFace model repository |
| `download_file(model_id, filename, revision=None)` | Download one file from a HuggingFace model repository |

### Classes

| Class | Description |
|-------|-------------|
| `Model` | Model loading and inference (`Model.load()`, `model.generate()`) |
| `DFlashGenerator` | DFlash block-diffusion speculative decoder |
| `Trainer` | Training orchestration (`Trainer(...)`, `trainer.train()`) |
| `Tokenizer` | Tokenization (`Tokenizer.from_file()`, `Tokenizer.from_pretrained()`) |
| `LoraConfig` | LoRA configuration (`r`, `alpha`, `dropout`, `use_rslora`, `use_dora`) |
| `TrainingConfig` | Training hyperparameters (`learning_rate`, `batch_size`, `num_epochs`, ...) |
| `GenerationConfig` | Generation parameters (`max_tokens`, `temperature`, `top_k`, `top_p`, `min_p`, ...) |
| `DataLoaderConfig` | Data loading parameters (`batch_size`, `max_seq_len`, `shuffle`, ...) |
| `ProgressCallback` | Progress bar callback requiring `total_steps` |
| `LoggingCallback` | Periodic logging callback |
| `MetricsJsonCallback` | JSONL metrics callback |

### Enums

| Enum | Values |
|------|--------|
| `Dtype` | Float32, Float16, BFloat16, Float8E4M3, Float8E5M2, Int32, Int64, UInt8, Bool |
| `Quantization` | None, NF4, FP4, Int8, FP8 |
| `LoraBias` | None, All, LoraOnly |
| `LrSchedulerType` | Constant, Linear, Cosine, CosineWithRestarts, Polynomial, Wsd |
| `OptimizerType` | AdamW, Sgd, Adafactor, Lion |
| `DatasetFormat` | Simple, Alpaca, ShareGpt, OpenAi, Auto |
| `ModelArchitecture` | Llama, Llama4, Qwen2, Qwen3, Qwen3MoE, Gemma, Mistral, Phi, Phi4, DeepSeek, Cohere, Granite, NemotronH, Qwen3Next, GptOss, Gemma4, Bert, Flux |

## Typing

The package ships `py.typed` and `__init__.pyi`, so type checkers can see the
public API exposed by the compiled extension.

## License

MIT OR Apache-2.0
