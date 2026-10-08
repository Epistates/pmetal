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
  --chat

# FP8 quantized inference (2× memory reduction)
pmetal infer --model Qwen/Qwen3-4B --fp8 --chat

# With tool definitions
pmetal infer \
  --model Qwen/Qwen3-0.6B \
  --tools tools.json --chat

# Images and videos (Qwen3.5-family vision models)
pmetal infer --model Qwen/Qwen3.5-0.8B --image photo.jpg \
  --prompt "What is in this picture?" --chat

# A video is a directory of its frames; extract them first at the rate the
# model samples (2 per second), `ffmpeg -i clip.mp4 -vf fps=2 frames/%04d.png`,
# and give that rate
pmetal infer --model Qwen/Qwen3.5-0.8B --video frames/ --video-fps 2 \
  --prompt "Describe what happens in this video." --chat

# Inference on the Apple Neural Engine
pmetal infer --model Qwen/Qwen3-4B --backend ane --ane-max-seq-len 2048 --chat

# JIT-compiled sampling
pmetal infer --model Qwen/Qwen3-0.6B --backend compiled --chat

# Gemma 4 MTP assistant (exact speculative decode)
pmetal infer \
  --model google/gemma-4-31B-it \
  --draft-model google/gemma-4-31B-it-assistant \
  --prompt "Summarize monotonic queues." \
  --temperature 1.0 --top-k 64 --top-p 0.95

# Qwen3Next / Qwen3.6 bundled MTP (exact speculative decode)
pmetal infer \
  --model /path/to/qwen3next-with-mtp \
  --mtp --mtp-draft-tokens 3 \
  --prompt "Summarize monotonic queues." \
  --temperature 0.7 --top-k 20 --top-p 0.8

# Qwen MTP checkpoint trained with `pmetal train-mtp`
pmetal infer \
  --model /path/to/qwen3next-target \
  --mtp --mtp-model ./qwen-mtp \
  --prompt "Summarize monotonic queues." \
  --temperature 0.7

# Qwen bundled MTP with FP8 target/MTP weights and packed expert offload
pmetal infer \
  --model /path/to/qwen3next-with-mtp \
  --mtp --fp8 --experts-dir /path/to/packed_experts \
  --prompt "Summarize monotonic queues."

# Profile Qwen 3.5 hybrid prefill + cached decode layers and write JSON
pmetal infer \
  --model Qwen/Qwen3.5-0.8B \
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
| `--image` | — | Image file to show a Qwen3.5-family vision model, before the prompt (repeatable) |
| `--video` | — | Directory of a video's frames to show a Qwen3.5-family vision model, after the images (repeatable). The frames are its image files in natural file-name order (`frame2.png` before `frame10.png`); video files are not decoded |
| `--video-fps` | `24` | Frame rate of the `--video` frames. They are sampled to the checkpoint's rate (2 per second in the released configs, at least 4 and at most 768 frames) and time-stamped in the prompt. Without it the frames are taken to be 24 per second, as the reference does, with a warning: frames extracted at 2 per second would then mostly be dropped and their timestamps would be 12 times too small |
| `--draft-model` | — | Gemma 4 MTP assistant model for exact speculative decoding |
| `--mtp` | off | Enable bundled Qwen3Next/Qwen3.6 MTP exact speculative decoding |
| `--mtp-model` | bundled `mtp.*` in `--model` | Optional external Qwen MTP checkpoint directory |
| `--mtp-draft-tokens` | `3` | Number of Qwen MTP draft tokens per verify step |
| `--temperature` | model default | Sampling temperature |
| `--top-k` | model default | Top-k sampling |
| `--top-p` | model default | Nucleus sampling |
| `--min-p` | model default | Min-p dynamic sampling |
| `--max-tokens` | model default | Maximum generation length (see [Output Length](#output-length)) |
| `--repetition-penalty` | `1.0` | Repetition penalty |
| `--frequency-penalty` | `0.0` | Frequency penalty |
| `--presence-penalty` | `0.0` | Presence penalty |
| `--chat` | `false` | Apply chat template |
| `--fp8` | `false` | FP8 weights (~2× mem reduction) |
| `--backend` | `auto` | Generation path: `auto`, `standard`, `compiled` (JIT-compiled sampling), `metal-sampler` (fused Metal sampling kernel), `ane` (Apple Neural Engine) or `minimal` (debug loop) |
| `--profile-layers` | `false` | Run an opt-in per-layer forward profile for supported hybrid models |
| `--profile-output` | — | Write the layer profile report as pretty JSON |
| `--ane-max-seq-len` | `4096` | Largest context (prompt plus output) the ANE compiles a model for |
| `--tools` | — | Tool definitions file (OpenAI format) |
| `--system` | — | System message |
| `--no-thinking` | `false` | Turn thinking off (the chat template's `enable_thinking`) |
| `--reasoning-effort` | template default | How long the model thinks, for chat templates with a `reasoning_effort` control |
| `--no-preserve-thinking` | `false` | Keep only the current turn's thinking in the prompt, for chat templates with a `preserve_thinking` control |
| `--chat-template-kwargs` | — | Extra chat template keyword arguments as a JSON object |

## Output Length

Without `--max-tokens`, the output budget is the model's own:

1. `max_new_tokens` from the model's `generation_config.json`;
2. else its `max_length`, which counts the prompt, as in transformers;
3. else the length the model's card recommends, for the families that give
   one (see [Sampling Defaults](#sampling-defaults));
4. else 32,768 tokens when the model thinks (its chat template has a
   thinking control or writes a thinking block, and thinking is on, by
   `--no-thinking` or by the template's default). That is the length the Qwen3,
   Qwen3.5 and Qwen3.6 cards recommend for most queries; a thinking model
   spends most of its budget reasoning, so a few hundred tokens stop it
   mid-thought;
5. else 256 tokens.

The budget never runs past the context window (`max_position_embeddings`)
once the prompt is in. An explicit `--max-tokens` always wins; the Qwen3.8 card
recommends up to 262,144 reasoning tokens plus 131,072 answer tokens for
long agentic tasks. `pmetal serve` applies the same rule to a request without
`max_tokens`.

## Sampling Defaults

Any sampling parameter left unset comes from the model maker's published
settings for the model's family and mode (thinking or not, or `--mode`), else
the model's `generation_config.json` read as transformers reads it, with
transformers' `GenerationConfig` defaults for the fields it leaves out: greedy
unless it sets `do_sample: true`, and temperature 1.0, top-p 1.0 and top-k 50
when sampling. `pmetal serve` fills a request's unset parameters the
same way. The family is read from `config.json` and the model's name.

| Family | Thinking | Non-thinking | Output budget |
|--------|----------|--------------|---------------|
| Qwen3 (hybrid) | 0.6 / 0.95 / 20 | 0.7 / 0.8 / 20 | 32,768 |
| Qwen3-2507 Instruct, Qwen3-Next Instruct | — | 0.7 / 0.8 / 20 | 16,384 |
| Qwen3-2507 Thinking, Qwen3-Next Thinking | 0.6 / 0.95 / 20 | — | 32,768 |
| Qwen3.5 | 1.0 / 0.95 / 20 (coding 0.6) | 0.7 / 0.8 / 20, presence 1.5 (reasoning 1.0 / 1.0 / 40, presence 2.0) | 32,768 |
| Qwen3.6 | 1.0 / 0.95 / 20 (coding 0.6) | 0.7 / 0.8 / 20, presence 1.5 | 32,768 |
| Qwen3.8, Qwen3.8-Flash-Next | 1.0 / 0.95 / 20 | 0.7 / 0.8 / 20, presence 1.5 | 32,768 |
| Gemma 4 | 1.0 / 0.95 / 64 | 1.0 / 0.95 / 64 | — |
| gpt-oss | 1.0 / 1.0 | — | — |
| Mistral Small 3.x | — | 0.15 | — |
| Magistral | 0.7 / 0.95 | — | 40,960 |
| Phi-4-reasoning | 0.8 / 0.95 / 50 | — | 32,768 |
| DeepSeek-R1 and distillations | 0.6 / 0.95 | — | — |
| DeepSeek-V3-0324 | — | 0.3 | — |
| Nemotron Nano 2 | 0.6 / 0.95 | greedy | — |

Values are temperature / top-p / top-k. Families not listed (Llama 3.x and 4,
Gemma 2/3, Phi-3/4, Mistral 7B, Mixtral, Cohere, Granite, SmolLM2, Qwen2.5)
publish no recommendation beyond their `generation_config.json`.

## Long Context (YaRN)

Qwen3.5, 3.6 and 3.8 run 262,144 tokens natively. For longer inputs their
cards recommend static YaRN, set in the model's `config.json`: change
`rope_parameters` in `text_config` to

```json
{
  "mrope_interleaved": true,
  "mrope_section": [11, 11, 10],
  "rope_type": "yarn",
  "rope_theta": 10000000,
  "partial_rotary_factor": 0.25,
  "factor": 4.0,
  "original_max_position_embeddings": 262144
}
```

Both of PMetal's engines then compute the rotary embedding as transformers
does (the blended frequencies, and the attention factor `0.1 * ln(factor) + 1`
on the rotated channels), and the context window, which bounds the default
output budget, becomes `original_max_position_embeddings * factor`. Static
YaRN applies the same scaling to every input, which the cards note can hurt
short texts, so enable it only for long ones, and size `factor` to the
context you need (2.0 for 524,288 tokens).

Dense Qwen3 works the same way from 32,768 tokens: its card adds

```json
"rope_scaling": {
  "rope_type": "yarn",
  "factor": 4.0,
  "original_max_position_embeddings": 32768
}
```

to `config.json`. Every architecture reads `rope_scaling` and
`rope_parameters` with the same code, which implements each `rope_type`
transformers defines (`linear`, `dynamic`, `yarn`, `longrope`, `llama3`,
`proportional`). A type an architecture cannot run is refused by name when
the model loads.

## Thinking Controls

Chat templates take keyword arguments the way Hugging Face transformers'
`apply_chat_template(messages, **kwargs)` passes them, and PMetal renders them
byte for byte the same. The flags above set the ones model makers document:

| Model | Control | Values |
|-------|---------|--------|
| Qwen3.8 | `reasoning_effort` | `xhigh` (default), `medium`, `low` |
| Qwen3.8 | `preserve_thinking` | on by default; `--no-preserve-thinking` turns it off |
| gpt-oss | `reasoning_effort` | `low`, `medium` (default), `high` |
| Qwen3 onwards, Gemma 4 | `enable_thinking` | `--no-thinking` turns it off |

Unset, each control keeps the template's own default: Qwen3 and Qwen3.8 think,
while Gemma 4 and Qwen3.5-0.8B answer directly unless
given `--chat-template-kwargs '{"enable_thinking": true}'`.

```bash
pmetal infer --model Qwen/Qwen3.8-27B --chat --reasoning-effort low \
  --prompt "Is 2^31 - 1 prime?"
```

`--reasoning-effort` or `--no-preserve-thinking` on a model whose template has
no such control is an error rather than a silent no-op, and so is a level the
template refuses. Anything else a template
reads goes through `--chat-template-kwargs`, e.g.
`--chat-template-kwargs '{"model_identity": "You are a test model."}'` on
gpt-oss.

## Layer Profiling

`--profile-layers` is currently implemented for standard `Qwen 3.5 / qwen3_next` inference. It runs one real prefill pass and one real cached decode pass using the shared inference runner, forcing MLX evaluation at each measured section so the report reflects actual wall time instead of only op scheduling overhead.

`--mtp` is its own verifier/drafter generation backend. It supports bundled Qwen `mtp.*` weights or an external `--mtp-model` checkpoint, FP8 target/MTP weights, packed expert offload, LoRA-merged Qwen3Next targets, and checkpoints with more than one MTP predictor layer. `--backend` is ignored while MTP is active because exact speculative verification owns the decode loop. LoRA and `--experts-dir` are not combined; fuse the adapter first if you need packed expert offload.

When Gemma 4 or Qwen MTP is active, the CLI prints a `Speculative:` summary with draft
acceptance rate, accepted/attempted draft tokens, average accepted draft tokens per verify
step, and the number of target bonus/correction tokens.

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
- [Rust SDK](/sdk/advanced/) — Programmatic inference
- [Python SDK](/python/quick-start/) — Python inference
