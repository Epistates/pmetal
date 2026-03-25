# pmetal infer

Run interactive inference with chat, tool use, thinking mode, and LoRA adapters.

Run inference on a loaded model. Supports interactive chat, tool/function calling, thinking mode, FP8 quantization, LoRA adapter loading, and opt-in per-layer profiling for supported hybrid models.

## Usage

```bash
pmetal infer \
  --model <MODEL> \
  [--prompt <PROMPT>] \
  [OPTIONS]
```

## Examples

```bash
# Simple generation
pmetal infer --model Qwen/Qwen3-0.6B --prompt "What is 2+2?"

# Interactive chat with LoRA
pmetal infer \
  --model Qwen/Qwen3-0.6B \
  --lora ./output/lora_weights.safetensors \
  --chat --show-thinking

# FP8 quantized inference (2× memory reduction)
pmetal infer --model Qwen/Qwen3-4B --fp8 --chat

# With tool definitions
pmetal infer \
  --model Qwen/Qwen3-0.6B \
  --tools tools.json --chat

# ANE-optimized inference
pmetal infer --model Qwen/Qwen3-0.6B --ane-max-seq-len 2048

# JIT-compiled sampling
pmetal infer --model Qwen/Qwen3-0.6B --compiled --chat

# Profile Qwen 3.5 hybrid prefill + cached decode layers and write JSON
pmetal infer \
  --model unsloth/Qwen3.5-0.8B \
  --prompt "write a fizzbuzz program in python" \
  --chat --no-thinking --temperature 0 \
  --profile-layers \
  --profile-output .strategy/qwen35_layer_profile.json
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | *required* | HuggingFace model ID or local path |
| `--prompt` | — | Input prompt (omit for stdin) |
| `--lora` | — | Path to LoRA adapter weights |
| `--temperature` | model default | Sampling temperature |
| `--top-k` | model default | Top-k sampling |
| `--top-p` | model default | Nucleus sampling |
| `--min-p` | model default | Min-p dynamic sampling |
| `--max-tokens` | `256` | Maximum generation length |
| `--repetition-penalty` | `1.0` | Repetition penalty |
| `--frequency-penalty` | `0.0` | Frequency penalty |
| `--presence-penalty` | `0.0` | Presence penalty |
| `--chat` | `false` | Apply chat template |
| `--show-thinking` | `false` | Show reasoning content |
| `--fp8` | `false` | FP8 weights (~2× mem reduction) |
| `--compiled` | `false` | JIT-compiled sampling |
| `--profile-layers` | `false` | Run an opt-in per-layer forward profile for supported hybrid models |
| `--profile-output` | — | Write the layer profile report as pretty JSON |
| `--no-ane` | `false` | Disable ANE inference |
| `--ane-max-seq-len` | `1024` | Max ANE kernel sequence length |
| `--tools` | — | Tool definitions file (OpenAI format) |
| `--system` | — | System message |

## Layer Profiling

`--profile-layers` is currently implemented for standard `Qwen 3.5 / qwen3_next` inference. It runs one real prefill pass and one real cached decode pass using the shared inference runner, forcing MLX evaluation at each measured section so the report reflects actual wall time instead of only op scheduling overhead.

Use `--profile-output <PATH>` to capture the full JSON report. The CLI summary now prints:
- total layer time vs non-layer overhead
- aggregated time by layer kind (`linear_attention` vs `full_attention`)
- top section buckets within each kind
- the slowest individual layers and their main sections

That makes long-prompt hybrid profiles much easier to read when you are deciding whether the next prompt-heavy optimization should target GDN prefill, full-attention preparation/SDPA, sparse MoE combine, or decode-only paths.

## Chat Mode

With `--chat`, PMetal applies the model's chat template and starts an interactive session:

```
> What is quantum entanglement?
Quantum entanglement is a phenomenon where two particles...

> Can you explain it more simply?
Think of it like two coins that always land on opposite sides...
```

## Tool Use

Pass OpenAI-format tool definitions with `--tools`:

```json
[
  {
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get current weather",
      "parameters": {
        "type": "object",
        "properties": {
          "location": { "type": "string" }
        }
      }
    }
  }
]
```

Supported for Qwen, Llama 3.1+, Mistral v3+, and DeepSeek models.

## See Also

- [pmetal serve](/cli/serve/) — OpenAI-compatible inference server
- [Rust SDK](/sdk/easy-api/) — Programmatic inference
- [Python SDK](/python/quick-start/) — Python inference
