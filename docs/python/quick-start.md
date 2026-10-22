# Quick Start

Get started with PMetal's Python SDK: fine-tune and run inference in a few lines.

PMetal exposes a typed Python extension module (PyO3, one abi3 wheel for CPython 3.11 and later).
It needs macOS 14 or later on Apple Silicon.

## Installation

The wheel is not on PyPI yet, so build it from a checkout with [maturin](https://www.maturin.rs):

```bash
cd crates/pmetal-py
pip install maturin
maturin develop --profile wheel
```

`just build-wheel` builds the wheel a release ships, with the `pmetal` command line and the Metal library inside the package.

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
print(f"Loss: {result.final_loss}, Steps: {result.total_steps}, adapter: {result.weights_path}")
```

`finetune` takes the fields of [`TrainSpec`](/python/full-control/#the-job-specs): the same options,
defaults and checks as `pmetal train`. Ctrl-C stops the run after the current step and keeps the
weights.

## Inference

```python
# One prompt (the model is loaded for the call)
text = pmetal.infer("Qwen/Qwen3-0.6B", "What is 2+2?", chat=True, no_thinking=True)
print(text)

# With a LoRA adapter
text = pmetal.infer(
    "Qwen/Qwen3-0.6B",
    "Explain quantum entanglement",
    lora="./output/adapter_model.safetensors",
)

# Sampling options are InferSpec's; a sampling option you leave out takes the model's own value
text = pmetal.infer("Qwen/Qwen3-0.6B", "Tell me a story", temperature=0.8, max_tokens=512, seed=1)
```

For many prompts against one model, keep a `Model` loaded:

```python
model = pmetal.Model.load("Qwen/Qwen3-0.6B")
print(model.generate("The capital of France is", max_tokens=16, temperature=0.0))
for piece in model.stream("Once upon a time", max_tokens=64):
    print(piece, end="", flush=True)
```

## See Also

- [Full Control](/python/full-control/): specs, callbacks, running any command, the Hub
- [Rust SDK](/sdk/advanced/): Rust equivalent API
