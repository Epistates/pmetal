# pmetal serve

Start an OpenAI-compatible inference server.

Start an HTTP inference server with OpenAI- and Anthropic-compatible APIs. The prebuilt release binary and the Homebrew formula include this command; a build from source needs the `serve` feature (see the note at the end of this page).

## Usage

```bash
pmetal serve --model <MODEL> [OPTIONS]
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `-m`, `--model` | required | Model ID or local path |
| `-p`, `--port` | `8080` | Port to listen on |
| `--host` | `127.0.0.1` | Address to bind. The default is this machine only; binding another address without `--api-key` logs a warning |
| `--api-key` | none | API key clients must send (`Authorization: Bearer <key>` or `x-api-key: <key>`); repeatable. The `PMETAL_API_KEY` environment variable (comma-separated) sets the same without the key appearing in the process list. With none, the server is open: keep it on loopback |
| `--cors-origin` | none | Browser origin whose pages may call the server (CORS), or `*` for any; repeatable. Without one, no CORS headers are sent |
| `--request-timeout` | `1800` | Seconds one request may run before the server ends it (a 504, or an error frame in a stream); `0` for no limit |
| `--max-seq-len` | the model's window, within memory | Context length held per sequence (prompt plus output); see [Context length](#context-length) |
| `--experts-dir` | none | Packed expert directory for SSD-offloaded MoE inference (see [`pmetal pack-experts`](/cli/pack-experts/)) |
| `--fp8` | `false` | Quantize weights to FP8 (E4M3) at load time, about 2x memory savings |
| `--kv-quant` | engine default | KV cache quantization bits: `8` (q8_0), `4` (q4_0) or `0` (fp16) |
| `--no-kv-quant` | `false` | Use an fp16 KV cache |
| `--kv-group-size` | `64` | KV cache quantization group size |
| `--kv-turboquant` | `false` | TurboQuant KV cache compression |
| `--kv-turboquant-preset` | none | TurboQuant quality preset: `q2_5` or `q3_5` |
| `--ane` | `false` | Generate on the Apple Neural Engine (experimental). A model the ANE cannot run is an error, not a quiet GPU run |
| `--ane-max-seq-len` | `4096` | Largest context (prompt plus output) the ANE compiles a model for |
| `--draft-model` | none | DFlash draft model drafting for the model on the ANE: it runs on the GPU and the ANE verifies its guesses. Needs `--ane` |
| `--continuous-batch` | `false` | Serve requests together (see [Continuous batching](#continuous-batching)) |
| `--cb-max-slots` | `8` | Requests running at once |
| `--cb-max-queue-depth` | `256` | Requests that may wait for a slot |
| `--cb-block-size` | `32` | Tokens per block that admission counts KV memory in |
| `--cb-max-blocks` | `0` | KV blocks that may be reserved at once (`0` is what the device can hold beside the model) |
| `--cb-exact` | `false` | One forward per request, bit for bit the single-request reply |

## Examples

```bash
# Start server
pmetal serve --model Qwen/Qwen3-0.6B --port 8080

# Require an API key
PMETAL_API_KEY=change-me pmetal serve --model Qwen/Qwen3-0.6B

# Serve several requests at once
pmetal serve --model Qwen/Qwen3-0.6B --continuous-batch --cb-max-slots 16

# Serve a pre-fused adapter model
pmetal fuse \
  --model Qwen/Qwen3-0.6B \
  --lora ./output \
  --output ./output/fused
pmetal serve --model ./output/fused --port 8080
```

## API Compatibility

The server speaks both the OpenAI and the Anthropic wire formats:

- `POST /v1/chat/completions`: chat completions (streaming and non-streaming, tool calling, token logprobs, images and videos)
- `POST /v1/responses`: the OpenAI Responses API (text and image input, function tools, structured output, streaming events); server-side state such as `previous_response_id` is refused by name
- `POST /v1/completions`: text completions; `echo` with `logprobs` scores the prompt too
- `POST /v1/embeddings`: embeddings, 17 architectures via `forward_hidden`
- `POST /v1/messages`: Anthropic-compatible messages, with SSE streaming and image blocks
- `POST /v1/messages/count_tokens`: the input tokens a messages request would use
- `GET /v1/models` and `GET /v1/models/{id}`: list loaded models, or describe one
- `GET /v1/metrics`: serving metrics as JSON
- `GET /metrics`: Prometheus / OpenMetrics metrics
- `GET /health`: liveness check

Request bodies are capped at 64 MiB, room for base64-encoded images and video frames.

## Images and Videos

Two families read media in chat messages, on both chat endpoints, streaming or not:

| Family | Reads | Checked against |
|--------|-------|-----------------|
| Qwen3.5, 3.6, 3.8 (checkpoints that ship a vision tower) | images and videos | [`pmetal infer --image`](/cli/infer/): the same prompt tokens and the same greedy reply, token for token |
| Llama 3.2 Vision (Mllama) | images | transformers' processor and `generate`: the same prompt tokens and the same greedy reply, token for token |

On `/v1/chat/completions`, a message's `content` is a list of parts:

| Part | Shape |
|------|-------|
| Text | `{"type": "text", "text": "..."}` |
| Image | `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}` |
| Video | `{"type": "video", "video": ["data:image/png;base64,...", ...]}` or `{"type": "video", "video": {"frames": [...], "fps": 30}}` |

An image is any format the `image` crate decodes (PNG, JPEG, WebP, GIF, BMP, TIFF), sent inline
as a `data:image/...;base64,` URI; `image_url` may also be the URI string itself, and the
`input_image` part type is read the same way. `detail` is accepted and has no effect: the
checkpoint's own processor sizes every image from the pixel budget in its
`preprocessor_config.json`, as the reference processor does, so a request can't ask for more or
fewer tokens per image. A video is its decoded frames, each one such a URI; `fps` is the rate of
the frames you send, which the processor samples down to its own rate (2 per second in the
released configs) and uses to time-stamp them in the prompt (24 is assumed when it is absent, as
the reference processor assumes). Video files are not decoded: extract the frames and send those.

A message may hold any number of images and videos, in any order among its text, and later turns
may add more. The Qwen and Llama 3.2 Vision model cards show the image before the question, which
is what `pmetal infer --image` sends.

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.5-0.8B",
    "messages": [{"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,'"$(base64 -i photo.png)"'"}},
      {"type": "text", "text": "Describe this image in one sentence."}
    ]}]
  }'
```

On `/v1/messages`, an image is a block
`{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}}`.

Each image and video goes to the model's own chat template as an item of its message, where you
put it among the text, so its placeholder lands exactly where the reference processor's
`apply_chat_template` puts it. The tokens count toward `usage.prompt_tokens` and must fit in
`--max-seq-len` with room to generate.

- **Qwen3.5 family:** the checkpoint's processor expands each placeholder to the media's tokens,
  one per 32×32 pixels after resizing. The released budget keeps up to 16.7 megapixels (16,384
  tokens), so a 640×480 picture is 300 tokens and a 12-megapixel photo about 11,700: the default
  context (see below) holds a couple of those; raise `--max-seq-len` for more. The vision
  tower loads with the first request that carries media and stays loaded.
- **Llama 3.2 Vision:** each image is one `<|image|>` token; the processor fits it into up to four
  560×560 tiles, and the text model's cross-attention layers read them. Text after an image
  attends to that image, up to the next one. The model's template refuses a system message
  alongside images, and the 400 says so. Videos are refused.

Refused with a 400 that says why:

- a remote `http(s)` image URL, a file path, any other URL scheme, or an uploaded `file_id`: the
  server fetches and reads nothing on a client's behalf, so send the bytes inline
- a `video_url` part, audio, and any other part type
- `mm_processor_kwargs`: media are preprocessed with the checkpoint's settings, and an override
  ignored silently would change what the model sees
- images or videos sent to a model that cannot read them; the error names the model. Gemma 4
  checkpoints are among these for now: their vision tower is ported, but the Gemma 4 text model
  does not load or run it

A prompt with media runs on the single-request path even with `--continuous-batch`, and never
touches the prefix cache: both key a prompt by its token ids, and every image's tokens are the
same placeholder id whatever the image shows. Text-only requests are unaffected.

Serving a decision model (a Clef release, recognized by its `joint_head_config.json`) starts a
different router: `POST /v1/systemone` answers typed questions about a state, next to
`GET /v1/models` and `GET /health`. A decision model never generates, so the generation flags do
not apply. See [pmetal decide](/cli/decide/) for the request and response bodies.

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-0.6B", "messages": [{"role": "user", "content": "Hello"}]}'
```

### Continuous batching

`--continuous-batch` serves requests together instead of one after another. The
scheduler re-forms the batch after every step, so a request starts as soon as a
slot is free and leaves the step it finishes; each step decodes one token for
every running request and runs one chunk of the prompt that is prefilling, so a
long prompt delays the others by a chunk and no more.

A model with a batched forward decodes all running requests in one forward,
reading the weights once however many there are. Those are Llama, Mistral, Qwen 2
and 3, Qwen3-MoE, Gemma 1 and 2, Phi and Phi-4, and Cohere (not Cohere 2 or
Command R+), with a plain dense KV cache, and only when the model has no sliding
window and a plain RoPE: a scaled RoPE (Llama 3 bands, YaRN, LongRoPE, dynamic
NTK) takes the one-forward-per-request path, and so does every Granite release
(a Granite's residual multiplier has no place in the batched forward). Gemma 3 and
4, gpt-oss, the hybrid models (Qwen 3.5 and later, Nemotron-H, Granite 4.0-H),
DeepSeek, Llama 4, quantized, TurboQuant and sliding-window caches, and the other
architectures decode one forward per request. A request that decodes alone gets
what the single-request path gives it, bit for bit; with company its logits can
differ in the last bit of a bf16 value, so a greedy near-tie may fall the other
way. `--cb-exact` decodes every request on its own cache, so a reply is exactly
the single-request one however requests join and leave, at the cost of the
batched speed.

A request that begins with tokens an earlier request already ran reuses their
K/V. For a hybrid model the cached prefix holds the recurrent state beside the
K/V, so it serves a request that repeats a whole earlier prompt (a chat's next
turn) but not one that diverges inside it.

The scheduler's options are listed in the table above (`--cb-max-slots`, `--cb-max-queue-depth`,
`--cb-block-size`, `--cb-max-blocks` and `--cb-exact`).

A request reserves its worst case (prompt plus `max_tokens`) while it runs, so
none is ever preempted; one that does not fit waits, and a small request may pass
it only so many times before it holds the line. A client that hangs up frees its
slot at once.

### Context length

Without `--max-seq-len`, each sequence gets the model's context window
(`max_position_embeddings`, or the YaRN-stretched length), capped so the
weights plus a full-length fp16 KV cache per sequence (per continuous-batching
slot) fit in 70% of the device's working set; never under 4096. A request's
cache grows with its sequence, so a long window costs memory only when a
request fills it. A long prompt prefills in chunks that shrink as the context
deepens, so no chunk's attention runs long enough for the GPU watchdog to
stop it. Pass `--max-seq-len` for a shorter (or longer) context.

### Output length

A chat or text completion without `max_tokens` (or `max_completion_tokens`)
gets the model's own budget, by the rule `pmetal infer` uses (see
[Output Length](/cli/infer/#output-length)): `generation_config.json`, else
the model card's length, else 32,768 tokens for a thinking model, else 256,
inside the context window and `--max-seq-len`.

### Sampling defaults

Sampling parameters a request leaves out (temperature, top-p, top-k, min-p,
the penalties) take the model maker's settings for the model's family and for
whether it is thinking, else its `generation_config.json` with transformers'
defaults for what it leaves out (greedy unless `do_sample` is true), as in [`pmetal infer`](/cli/infer/#sampling-defaults). Pass
`"temperature": 0` for plain greedy decoding (no default penalties then).

### Thinking controls

Chat completions accept OpenAI's top-level `reasoning_effort` and a
`chat_template_kwargs` object, both handed to the model's chat template as
keyword arguments, exactly as `apply_chat_template(messages, **kwargs)` takes
them (a value inside `chat_template_kwargs` wins). Qwen3.8 reads
`reasoning_effort` (`xhigh`, `medium`, `low`), `enable_thinking` and
`preserve_thinking`; gpt-oss reads `reasoning_effort` (`low`, `medium`,
`high`). A level the model refuses is a 400.

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3.8-27B", "reasoning_effort": "low",
       "chat_template_kwargs": {"preserve_thinking": false},
       "messages": [{"role": "user", "content": "Hello"}]}'
```

For multi-turn conversations, send an assistant turn's thinking back as
`reasoning_content` next to its `content`; a `<think>…</think>` left at the
head of `content` is split out the same way. `/v1/messages` maps
`"thinking": {"type": "disabled"}` (or `"enabled"`) to `enable_thinking`.

### Reasoning in replies

A model that reasons before it answers gets its reasoning returned apart from
the answer, and `content` holds the answer only. Chat completions put it in
`message.reasoning_content`, and when streaming in `delta.reasoning_content`
chunks ahead of the `delta.content` ones, as other OpenAI-compatible servers
do. `/v1/messages` answers with a `thinking` block ahead of the `text` block
(streamed as `thinking_delta`s); its `signature` is empty, since a local
model's thinking is not signed. Qwen3-family `<think>…</think>` (including a
block the chat template opens itself), DeepSeek-R1, Phi-4-reasoning,
Magistral's `[THINK]`, Gemma 4's thought channel and gpt-oss's `analysis`
channel are recognised; the markers themselves appear in neither field.
Logprobs cover the answer's tokens.

### Quantized checkpoints

An MLX-quantized checkpoint (an `mlx-community` 4-bit model, say) keeps its
projections and its embedding table packed in memory: MLX's quantized matmul reads
the packed words, so a decode step reads a quarter of the bytes a bf16 copy would
and runs that much faster. This holds for the architectures that load through the
generic loader (Llama, Qwen 2 and 3, Mistral, Cohere); the others unpack to bf16
as they load. `--fp8` quantizes dense weights, so it unpacks them first.

### Speculative decoding

A drafter speeds a single-request decode up without changing what it says: it
proposes a few tokens, the model checks all of them in one forward, and the
longest agreeing run is kept. A reply is what the model alone gives (greedy
decoding gives its exact tokens) or is drawn from the same distribution
(sampling), through the same sampler: temperature, top-k, top-p, min-p, the
penalties, `logit_bias`, a seed and a JSON-schema grammar all behave as they do
without a drafter.

| Flag | Drafter | Target |
|------|---------|--------|
| `--mtp` | the model's own multi-token-prediction head (the `mtp.*` weights bundled in the checkpoint) | Qwen3.5, 3.6, 3.8 |
| `--mtp-model <dir>` | the same head, read from a checkpoint of its own | with `--mtp` |
| `--draft-model <id or path>` | a Gemma 4 assistant, or a DFlash draft model | Gemma 4; Qwen, Llama, Mistral, Gemma, Phi, DeepSeek and GPT-OSS for DFlash |
| `--draft-tokens <n>` | tokens a round proposes (default 3 for MTP, the assistant's own setting, the DFlash block size) | |

```bash
pmetal serve --model Qwen/Qwen3.5-0.8B --mtp
pmetal serve --model Qwen/Qwen3-4B --draft-model z-lab/Qwen3-4B-DFlash-b16
```

The drafter is loaded and checked against the model at startup: one for another
model family, or of another hidden or vocabulary size, is an error then, not on
the first request. With `--ane`, `--draft-model` still means a DFlash draft model
that drafts on the GPU for the model on the ANE.

What it does not do:

- Log-probabilities (the reply's or the prompt's) and images decode on the plain
  path: the verifier's rows are read for checking, not kept. The reply is the
  same either way; the request just does not speed up.
- A drafter reads no prefix cache, so `usage.prompt_tokens_details.cached_tokens`
  is 0, and the target runs on a float KV cache (`--kv-quant` and `--kv-turboquant`
  are refused with a drafter).
- Continuous batching serves many requests at once and a drafter one at a time:
  `--mtp` or `--draft-model` with `--continuous-batch` is an error.
- Tokens stream a round at a time (up to the drafted length together) instead of
  one by one.

How many drafted tokens the model accepted is in `/v1/metrics`
(`speculative_acceptance_rate`) and `/metrics`
(`pmetal_speculative_drafted_tokens_total`, `pmetal_speculative_accepted_tokens_total`).

### Counting tokens

`POST /v1/messages/count_tokens` takes the body of a messages request that shapes
the prompt (`model`, `messages`, `system`, `tools`, `tool_choice`, `thinking`)
and answers `{"input_tokens": N}`. The count is of the prompt `/v1/messages`
would run: the same chat template, tools and thinking setting, images expanded to
the tokens the model reads them as, tokenized by the model's tokenizer. Nothing is
generated.

### Scoring a prompt

`/v1/completions` with `echo: true` and `logprobs: N` returns the prompt's own
log-probabilities ahead of the completion's, as loglikelihood evaluation reads
them: `logprobs.tokens`, `token_logprobs`, `top_logprobs` and `text_offset` cover
the prompt's tokens and then the generated ones, with `null` for the first
token (nothing precedes it). `max_tokens: 0` scores the prompt and generates
nothing. `top_logprobs` holds the `N` likeliest tokens at each position, the
chosen one among them when it is one of them. The prompt is prefilled in full (the
prefix cache is not read), and streams as the first chunk, with its text. It
needs the GPU engine: `--ane` refuses it, and `--continuous-batch` runs such a
request on the single-request path.

### Metrics

`GET /metrics` serves Prometheus text; a scraper that asks for
`application/openmetrics-text` gets OpenMetrics, anything else text exposition
format 0.0.4. Like every route but `/health` it needs an API key when the server
has one (Prometheus sends it as `Authorization: Bearer <key>`).

| Series | Meaning |
|--------|---------|
| `pmetal_http_requests_total{route,method,status}` | requests answered, by the route's pattern |
| `pmetal_http_request_duration_seconds{route,method}` | seconds to the response headers (a streamed reply's end is in the generation series) |
| `pmetal_http_requests_in_flight` | requests being handled, scrapes included |
| `pmetal_generation_requests_total`, `pmetal_tokens_total{kind}` | generations finished; prompt, cached-prompt and completion tokens |
| `pmetal_time_to_first_token_seconds`, `pmetal_generation_duration_seconds`, `pmetal_generation_tokens_per_second` | histograms per generation |
| `pmetal_speculative_drafted_tokens_total`, `pmetal_speculative_accepted_tokens_total` | the drafter's work; accepted over drafted is the acceptance rate |
| `pmetal_requests_running`, `pmetal_requests_queued`, `pmetal_request_capacity` | requests holding a slot, waiting for the model, and the limit before a 429 |
| `pmetal_continuous_batching_active_slots`, `pmetal_continuous_batching_queued_requests` | the batching scheduler |
| `pmetal_prefix_cache_entries`, `_bytes`, `pmetal_prefix_cache_hits_total`, `_misses_total` | the prefix cache |
| `pmetal_memory_active_bytes`, `_peak_bytes`, `_cache_bytes` | GPU memory MLX holds |
| `pmetal_build_info{version,model}`, `pmetal_start_time_seconds` | what runs, and since when |

`/v1/metrics` keeps its JSON: totals, mean latencies, `rolling_tokens_per_second`
and the speculative counts.

:::note
The prebuilt release binary and the Homebrew formula both ship this command (they are built with
`--features serve,mcp`). If you build PMetal yourself, `serve` is opt-in:
`cargo install pmetal --features serve,mcp`.
:::

## See Also

- [pmetal infer](/cli/infer/): Single-prompt inference
- [pmetal decide](/cli/decide/): Decision-model requests, also served on `/v1/systemone`
- [pmetal mcp](/cli/mcp/): MCP server for Claude Desktop and Claude Code
- [Feature Flags](/configuration/feature-flags/): Enable the serve feature
