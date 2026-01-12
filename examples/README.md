# PMetal Examples

This directory contains example configurations and scripts for common PMetal workflows.

## Examples

### Training

- `lora_finetune.sh` - Basic LoRA fine-tuning workflow
- `qlora_finetune.sh` - QLoRA fine-tuning with 4-bit quantization

### Inference

- `inference.sh` - Text generation with base model
- `inference_lora.sh` - Text generation with LoRA adapter

### Data Preparation

- `sample_dataset.jsonl` - Example training data format

## Quick Start

```bash
# 1. Build PMetal
cargo build --release

# 2. Fine-tune a model
./examples/lora_finetune.sh

# 3. Run inference
./examples/inference_lora.sh
```

## Dataset Formats

PMetal supports multiple dataset formats. See `sample_dataset.jsonl` for examples of:

- **ShareGPT**: Multi-turn conversations
- **Alpaca**: Instruction/input/output format
- **Messages**: OpenAI-style chat format

## Configuration

Most examples can be customized by editing the shell scripts. Key parameters:

| Parameter | Description |
|-----------|-------------|
| `--model` | HuggingFace model ID or local path |
| `--dataset` | Path to training JSONL file |
| `--lora-r` | LoRA rank (4, 8, 16, 32) |
| `--batch-size` | Training batch size |
| `--learning-rate` | Optimizer learning rate |
