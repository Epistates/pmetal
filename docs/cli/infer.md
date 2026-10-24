# pmetal infer

Generate text from a model: one prompt in, one completion out, with the model's chat template, tool definitions, thinking controls, LoRA adapters, speculative decoding, and opt-in per-layer profiling for supported hybrid models.

Give `--prompt`, or `--messages-file` to continue a conversation (see [Continuing a Conversation](#continuing-a-conversation)). For an interactive conversation use the Inference tab of [`pmetal tui`](/cli/tui/) or the Chat page of the desktop app; to serve a model to other programs use [`pmetal serve`](/cli/serve/).

`--model` may also be a `.gguf` file, or a directory holding one: the configuration and the tokenizer are read from the file. The dense Llama, Qwen 2, Qwen 3, Mistral, Gemma 1 to 4 and Phi 3 / Phi 4 decoders load this way (see [GGUF Checkpoints](/models/supported/#gguf-checkpoints)).

Ctrl-C, or SIGTERM (how a supervisor such as the desktop app stops it), ends the generation after the token in flight; the text so far is kept and the statistics are printed (with `--json`, `stopped_by` is `interrupted`), and the command exits with status 130.

## Usage

```bash
pmetal infer \
  --model <MODEL> \
  --prompt <PROMPT> \
  [OPTIONS]
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `-m`, `--model` | required | HuggingFace model ID or local path |
| `-p`, `--prompt` | required unless `--messages-file` | Input prompt |
| `--messages-file` | none | A JSON array of chat messages to continue instead of a prompt (see [Continuing a Conversation](#continuing-a-conversation)); not with `--prompt` |
| `--lora` | none | LoRA adapter: a directory written by `pmetal train` (or any PEFT adapter directory) or its weights file |
| `--chat` | `false` | Apply the model's chat template (instruct models and `--tools` use it regardless) |
| `--system` | none | System message |
| `--image` | none | Image file to show a Qwen3.5-family vision model, before the prompt (repeatable, in order) |
| `--video` | none | Directory of a video's frames to show a Qwen3.5-family vision model, after the images (repeatable). The frames are its image files in natural file-name order (`frame2.png` before `frame10.png`); video files are not decoded |
| `--video-fps` | `24` | Frame rate of the `--video` frames. They are sampled to the checkpoint's rate (2 per second in the released configs, at least 4 and at most 768 frames) and time-stamped in the prompt. Without it the frames are taken to be 24 per second with a warning: frames extracted at 2 per second would then mostly be dropped and their timestamps would be 12 times too small |
| `--max-tokens` | model default | Maximum generation length (see [Output Length](#output-length)) |
| `--temperature` | model default | Sampling temperature; `0` is greedy |
| `--top-k` | model default | Top-k sampling; `0` disables |
| `--top-p` | model default | Nucleus sampling (0.0 to 1.0) |
| `--min-p` | model default | Min-p dynamic sampling; `0` disables |
| `--repetition-penalty` | `1.0` | Penalty on tokens of the prompt and output; `1.0` disables |
| `--frequency-penalty` | `0.0` | Penalty proportional to a token's count; `0` disables |
| `--presence-penalty` | `0.0` | Penalty for any token that has appeared; `0` disables |
| `--seed` | random | Random seed for reproducible generation |
| `--mode` | `auto` | Sampling preset with the model card's parameters: `auto`, `thinking-general`, `thinking-coding`, `instruct-general` or `instruct-reasoning` (see [Sampling Defaults](#sampling-defaults)) |
| `--no-thinking` | `false` | Turn thinking off (the chat template's `enable_thinking`) |
| `--hide-thinking` | `false` | Leave the thinking trace out of the output |
| `--reasoning-effort` | template default | How long the model thinks, for chat templates with a `reasoning_effort` control |
| `--no-preserve-thinking` | `false` | Keep only the current turn's thinking in the prompt, for chat templates with a `preserve_thinking` control |
| `--chat-template-kwargs` | none | Extra chat template keyword arguments as a JSON object; `--no-thinking`, `--reasoning-effort` and `--no-preserve-thinking` win over the same keys |
| `--tools` | none | JSON file of tool definitions (OpenAI format) |
| `--json` | `false` | Print one JSON object on stdout instead of streaming the text: `model`, `text`, `prompt_tokens`, `generated_tokens`, `stopped_by` (`eos`, `length` or `interrupted`), `seconds`, `time_to_first_token_seconds`, `decode_tokens_per_second` and `speculative_acceptance`. The banner and statistics go to stderr |
| `--fp8` | `false` | FP8 weights (about 2x memory reduction) |
| `--experts-dir` | none | Packed expert directory for SSD-offloaded MoE inference (see [`pmetal pack-experts`](/cli/pack-experts/)) |
| `--backend` | `auto` | Generation path: `auto` (the GPU, or the ANE with `--ane`), `standard` (the GPU, whatever `--ane` says) or `ane` (Apple Neural Engine) |
| `--ane` | `false` | Run on the Apple Neural Engine (experimental) |
| `--ane-max-seq-len` | `4096` | Largest context (prompt plus output) the ANE compiles a model for. Each request gets the smallest power of two from 512 that fits it, up to this; a smaller context runs faster |
| `--draft-model` | none | Draft model for exact speculative decoding: a Gemma 4 MTP assistant, or with `--ane` a DFlash draft model (which runs on the GPU, drafting for the model on the ANE) |
| `--mtp` | `false` | Bundled Qwen3Next/Qwen3.6 MTP exact speculative decoding |
| `--mtp-model` | bundled `mtp.*` in `--model` | External Qwen MTP checkpoint directory (needs `--mtp`) |
| `--mtp-draft-tokens` | `3` | Qwen MTP draft tokens per verify step |
| `--kv-quant` | engine default | KV cache quantization bits: `8` (q8_0), `4` (q4_0) or `0` (fp16) |
| `--kv-k-bits`, `--kv-v-bits` | none | Key and value bits on their own, for an asymmetric cache (override `--kv-quant`) |
| `--kv-group-size` | `64` | KV cache quantization group size |
| `--kv-turboquant` | `false` | TurboQuant KV cache compression |
| `--kv-turboquant-preset` | none | Mixed-bit TurboQuant preset: `q2_5` or `q3_5` |
| `--kv-quant-preset` | none | Mixed-bit affine preset: `q2_5` or `q3_5` |
| `--kv-qjl` | `false` | QJL residual correction for 2- and 3-bit key quantization |
| `--no-kv-quant` | `false` | Use an fp16 KV cache |
| `--detect-repetition` | `false` | Stop on n-gram repetition loops |
| `--benchmark` | `false` | Time prompt processing and generation instead of answering (needs `--benchmark-prompt-tokens`) |
| `--benchmark-iters` | `5` | Measured trials for `--benchmark` |
| `--benchmark-prompt-tokens` | none | Synthetic prompt length for `--benchmark` |
| `--profile-layers` | `false` | Per-layer forward profile for supported hybrid models |
| `--profile-output` | none | Write the layer profile report as pretty JSON |

Images and videos run on the native engine only: they cannot be combined with `--lora`, `--fp8`, `--experts-dir`, `--ane`, `--mtp`, `--draft-model`, `--benchmark` or `--profile-layers`.

## Examples

```bash
# Simple generation
pmetal infer --model Qwen/Qwen3-0.6B --prompt "What is 2+2?"

# Chat template, a LoRA adapter trained with `pmetal train`
pmetal infer \
  --model Qwen/Qwen3-0.6B \
  --lora ./output \
  --chat --prompt "Summarize monotonic queues."

# FP8 quantized inference (2x memory reduction)
pmetal infer --model Qwen/Qwen3-4B --fp8 --chat --prompt "Explain LoRA in two sentences."

# With tool definitions
pmetal infer \
  --model Qwen/Qwen3-0.6B \
  --tools tools.json --prompt "What is the weather in Paris?"

# Images and videos (Qwen3.5-family vision models)
pmetal infer --model Qwen/Qwen3.5-0.8B --image photo.jpg \
  --prompt "What is in this picture?" --chat

# A video is a directory of its frames; extract them first at the rate the
# model samples (2 per second), `ffmpeg -i clip.mp4 -vf fps=2 frames/%04d.png`,
# and give that rate
pmetal infer --model Qwen/Qwen3.5-0.8B --video frames/ --video-fps 2 \
  --prompt "Describe what happens in this video." --chat

# Inference on the Apple Neural Engine
pmetal infer --model Qwen/Qwen3-4B --backend ane --ane-max-seq-len 2048 \
  --chat --prompt "Explain monotonic queues."

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

# Throughput benchmark with a 512-token synthetic prompt
pmetal infer --model Qwen/Qwen3-0.6B --benchmark --benchmark-prompt-tokens 512 \
  --prompt "unused by --benchmark"

# Profile Qwen 3.5 hybrid prefill + cached decode layers and write JSON
pmetal infer \
  --model Qwen/Qwen3.5-0.8B \
  --prompt "write a fizzbuzz program in python" \
  --chat --no-thinking --temperature 0 \
  --profile-layers \
  --profile-output qwen35_layer_profile.json
```

## Which Engine Decodes

A model is decoded by one of two implementations, and `pmetal infer` logs which one and why at startup (the `Engine selected` line).

- **The architecture's own engine.** Qwen3 (dense and MoE), Qwen3.5/3.6/3.8, Gemma 4 (dense, MoE and the QAT releases), Llama 4, DeepSeek V3 and gpt-oss each have one, built for that architecture's layers and cache. It is the default for these.
- **The shared model**, the one `pmetal serve` runs, for every other architecture, and for a run that asks for something only it does: `--lora`, `--fp8`, `--experts-dir`, a Gemma 4 `--draft-model`, or a KV cache mode the architecture's engine does not have (for example `--kv-quant` on Gemma 4, whose cache is not quantized).

An architecture's engine is used only while it is no slower than the shared model. On an M4 Max, decoding the same prompt on both from the same rest, the engines measure 1.0x to 1.6x the shared model's speed (Qwen3 1.0x, Gemma 4 1.0x to 1.2x, Qwen3.5 1.6x), and their greedy outputs agree except where two tokens are within bf16 rounding of each other. `tests/engine_ab_real_weights.rs` repeats that measurement on a checkpoint you point it at.

Setting `PMETAL_DISABLE_NATIVE_BRIDGE` runs the shared model whatever the checkpoint, to compare the two on the same files.

Decode speed on a laptop is not steady: on the M4 Max this was measured on, a GPU that has just been busy for a couple of seconds decodes at about half the speed of one that has rested, with either engine. Compare engines, or two builds, by running each from rest, one after the other, in both orders.

## Quantized Checkpoints

An MLX-quantized checkpoint (an `mlx-community` 4-bit model) is run from its packed weights,
not unpacked to bf16: the native engines read the packed words, and so does the shared model
the other architectures run on (projections, embedding tables and routed experts alike). A
decode step reads a quarter of the bytes a bf16 copy would, which is most of the time it takes.
`--fp8` quantizes dense weights, so it unpacks a quantized checkpoint first. An architecture
that cannot keep a quantized checkpoint packed (Mllama, Qwen4-Exp) logs what it unpacked.

## Continuing a Conversation

`--messages-file` takes the conversation so far as a JSON array of messages, and the model answers the last one:

```json
[
  {"role": "system", "content": "Be brief."},
  {"role": "user", "content": "Capital of France?"},
  {"role": "assistant", "content": "Paris."},
  {"role": "user", "content": "And of Italy?"}
]
```

```bash
pmetal infer --model Qwen/Qwen3-0.6B --messages-file chat.json
```

The roles are `system` (only first), `user`, `assistant` and `tool`; the last message must be a user's or a tool's result, and nothing but an assistant's tool call may be empty. An assistant message may carry `reasoning_content` and `tool_calls`, and a tool message a `tool_call_id`. The conversation goes through the model's chat template as it would for `--prompt` with `--chat`, and `--tools` and `--image` apply as they do there. `--system` is the same as a first system message, so give one or the other. A prompt is a conversation of one message, so the two forms give the same text. The file may be up to 32 MiB, and a problem in it is reported with the number of the message.

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
| Qwen3-2507 Instruct, Qwen3-Next Instruct | none | 0.7 / 0.8 / 20 | 16,384 |
| Qwen3-2507 Thinking, Qwen3-Next Thinking | 0.6 / 0.95 / 20 | none | 32,768 |
| Qwen3.5 | 1.0 / 0.95 / 20 (coding 0.6) | 0.7 / 0.8 / 20, presence 1.5 (reasoning 1.0 / 1.0 / 40, presence 2.0) | 32,768 |
| Qwen3.6 | 1.0 / 0.95 / 20 (coding 0.6) | 0.7 / 0.8 / 20, presence 1.5 | 32,768 |
| Qwen3.8, Qwen3.8-Flash-Next | 1.0 / 0.95 / 20 | 0.7 / 0.8 / 20, presence 1.5 | 32,768 |
| Gemma 4 | 1.0 / 0.95 / 64 | 1.0 / 0.95 / 64 | none |
| gpt-oss | 1.0 / 1.0 | none | none |
| Mistral Small 3.x | none | 0.15 | none |
| Magistral | 0.7 / 0.95 | none | 40,960 |
| Phi-4-reasoning | 0.8 / 0.95 / 50 | none | 32,768 |
| DeepSeek-R1 and distillations | 0.6 / 0.95 | none | none |
| DeepSeek-V3-0324 | none | 0.3 | none |
| Nemotron Nano 2 | 0.6 / 0.95 | greedy | none |

Values are temperature / top-p / top-k. Families not listed (Llama 3.x and 4,
Gemma 2/3, Phi-3/4, Mistral 7B, Cohere, Granite, SmolLM2, Qwen2.5)
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

`--profile-layers` is currently implemented for standard `Qwen 3.5 / qwen3_next` inference. It runs one real prefill pass and one real cached decode pass using the shared inference runner, forcing MLX evaluation at each measured section so the report reflects actual wall time instead of only op scheduling overhead. It is its own mode: it cannot be combined with `--benchmark` or `--lora`.

`--mtp` is its own verifier/drafter generation backend. It supports bundled Qwen `mtp.*` weights or an external `--mtp-model` checkpoint, FP8 target/MTP weights, packed expert offload, LoRA-merged Qwen3Next targets, and checkpoints with more than one MTP predictor layer. `--backend` is ignored while MTP is active because exact speculative verification owns the decode loop. LoRA and `--experts-dir` are not combined; fuse the adapter first if you need packed expert offload.

When Gemma 4 or Qwen MTP is active, the CLI prints a `Speculative:` summary with draft
acceptance rate, accepted/attempted draft tokens, average accepted draft tokens per verify
step, and the number of target bonus/correction tokens.

Use `--profile-output <PATH>` to capture the full JSON report. The CLI summary prints:
- total layer time vs non-layer overhead
- aggregated time by layer kind (`linear_attention` vs `full_attention`)
- top section buckets within each kind
- the slowest individual layers and their main sections

That makes long-prompt hybrid profiles much easier to read when you are deciding whether the next prompt-heavy optimization should target GDN prefill, full-attention preparation/SDPA, sparse MoE combine, or decode-only paths.

## Chat Template

With `--chat` (or an instruct model, or `--tools`), PMetal wraps the prompt in the model's chat template, with `--system` as the system message, and generates the assistant's reply to it. The command answers one prompt and exits; for a multi-turn conversation use the Inference tab of [`pmetal tui`](/cli/tui/).

```bash
pmetal infer --model Qwen/Qwen3-0.6B --chat --system "Answer in one sentence." \
  --prompt "What is quantum entanglement?"
```

## Tool Use

Pass OpenAI-format tool definitions with `--tools`; the model's reply carries its tool calls:

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

- [pmetal serve](/cli/serve/): OpenAI-compatible inference server
- [pmetal dflash](/cli/dflash/): Block-diffusion speculative decoding on the GPU
- [pmetal train-mtp](/cli/train-mtp/): Train an MTP predictor or assistant
- [Rust SDK](/sdk/advanced/): Programmatic inference
- [Python SDK](/python/quick-start/): Python inference
