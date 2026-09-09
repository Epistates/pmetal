# Quick Start

Get started with PMetal's Python SDK — fine-tune and run inference in a few lines.

PMetal exposes a Python extension module via PyO3. Install with maturin from `crates/pmetal-py`.

## Installation

```bash
cd crates/pmetal-py
pip install maturin
maturin develop --release
```

## Fine-Tuning

```python
import pmetal

result = pmetal.finetune(
    "Qwen/Qwen3-0.6B",
    "train.jsonl",
    lora_r=16,
    learning_rate=2e-4,
    epochs=3,
)
print(f"Loss: {result['final_loss']}, Steps: {result['total_steps']}")
```

## Inference

```python
# Simple generation
text = pmetal.infer("Qwen/Qwen3-0.6B", "What is 2+2?")
print(text)

# With LoRA adapter
text = pmetal.infer(
    "Qwen/Qwen3-0.6B",
    "Explain quantum entanglement",
    lora="./output/lora_weights.safetensors",
)
print(text)

# With generation parameters
text = pmetal.infer(
    "Qwen/Qwen3-0.6B",
    "Tell me a story",
    temperature=0.8,
    max_tokens=512,
)
```

## See Also

- [Full Control](/python/full-control/) — Custom training loops and model loading
- [Rust SDK](/sdk/advanced/) — Rust equivalent API
