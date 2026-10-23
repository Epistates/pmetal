# PMetal Examples

This directory contains runnable shell examples and small sample datasets for common PMetal workflows. All scripts use `./target/release/pmetal` by default; override `PMETAL_BIN` to point at another binary (an installed `pmetal` works too). Each is a thin wrapper around one `pmetal` command: see the [CLI reference](../docs/cli/train.md) for every option.

The scripts use small models (`Qwen/Qwen3-0.6B`), which are downloaded to the Hugging Face cache on first use.

## Examples

### Training

- `lora_finetune.sh` - Basic LoRA fine-tuning workflow
- `qlora_finetune.sh` - QLoRA fine-tuning with a 4-bit quantized base

### Inference

- `inference.sh` - Text generation with a base model
- `inference_lora.sh` - Text generation with a LoRA adapter
- `serve_openai.sh` - OpenAI-compatible local server with continuous batching

The prebuilt release binary and the Homebrew formula include `serve`. A binary built from source needs it enabled: `cargo build -p pmetal --release --features serve,mcp`.

### Quantization and Benchmarking

- `quantize_gguf.sh` - GGUF quantization with a configurable method
- `bench_workload.sh` - End-to-end inference and LoRA workload benchmark

### Data Preparation

- `sample_dataset.jsonl` - Example training data in three row layouts
- `sample_corpus.jsonl` - Example text corpus for tokenizer sharding
- `tokenize_corpus.sh` - Tokenize a JSONL corpus into PMetal shards

## Quick Start

```bash
# 1. Build PMetal
cargo build --release

# 2. Run inference
./examples/inference.sh

# 3. Fine-tune a model
./examples/lora_finetune.sh

# 4. Run inference with the adapter
./examples/inference_lora.sh
```

Check a dataset before training on it (this reads rows only, and loads no model):

```bash
./target/release/pmetal dataset validate --path examples/sample_dataset.jsonl
```

## Dataset Formats

PMetal reads rows in several layouts and detects each row's layout from its columns; one file can mix them. `sample_dataset.jsonl` has all three of these:

- **ShareGPT**: Multi-turn conversations (`conversations` with `from` and `value`)
- **Alpaca**: Instruction/input/output format
- **Messages**: OpenAI-style chat format (`messages` with `role` and `content`)

Rows can also be a prompt and its completion, a problem with its thinking and solution, or plain `text`. `pmetal dataset formats` lists them all.

## Configuration

Most examples can be customized with environment variables:

| Parameter | Description |
|-----------|-------------|
| `PMETAL_BIN` | Path to the `pmetal` binary |
| `MODEL` | HuggingFace model ID or local path |
| `OUTPUT` | Output file or directory |
| `PROMPT` | Inference prompt |
| `MAX_TOKENS` | Generated token budget |
| `LORA_PATH` | Adapter directory for `inference_lora.sh` (the `OUTPUT` of a training script) |
| `HOST` | Server bind host |
| `PORT` | Server port |
| `DATASET` | Training JSONL file or workload dataset |
| `TOKENIZER` | Tokenizer model ID or local path |
| `METHOD` | GGUF quantization method |
| `PRESET` | Workload benchmark preset |

Example:

```bash
MODEL=Qwen/Qwen3-4B MAX_TOKENS=128 ./examples/inference.sh
```
