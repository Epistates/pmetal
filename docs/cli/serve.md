# pmetal serve

Start an OpenAI-compatible inference server.

Start an HTTP inference server with an OpenAI-compatible API. Requires the `serve` feature flag.

## Usage

```bash
pmetal serve --model <MODEL> [OPTIONS]
```

## Examples

```bash
# Start server
pmetal serve --model Qwen/Qwen3-0.6B --port 8080

# Serve a pre-fused adapter model
pmetal fuse \
  --model Qwen/Qwen3-0.6B \
  --lora ./output/lora_weights.safetensors \
  --output ./output/fused
pmetal serve --model ./output/fused --port 8080
```

## API Compatibility

The server speaks both the OpenAI and the Anthropic wire formats:

- `POST /v1/chat/completions` — chat completions (streaming and non-streaming, tool calling, token logprobs, images and videos)
- `POST /v1/completions` — text completions
- `POST /v1/embeddings` — embeddings, 17 architectures via `forward_hidden`
- `POST /v1/messages` — Anthropic-compatible messages, with SSE streaming and image blocks
- `GET /v1/models` — list loaded models
- `GET /v1/metrics` — serving metrics
- `GET /health` — liveness check

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

### Context length

Without `--max-seq-len`, each sequence gets the model's context window
(`max_position_embeddings`, or the YaRN-stretched length), capped so the
weights plus a full-length fp16 KV cache per sequence (per continuous-batching
slot) fit in 70% of the device's working set, and at 32,768 tokens; never
under 4096. Pass `--max-seq-len` for longer contexts.

### Output length

A chat or text completion without `max_tokens` (or `max_completion_tokens`)
gets the model's own budget, by the rule `pmetal infer` uses (see
[Output Length](/cli/infer/#output-length)): `generation_config.json`, else
32,768 tokens for a thinking model, else 256, inside the context window and
`--max-seq-len`.

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

:::note
The prebuilt release binary and the Homebrew formula both ship this command. If you build PMetal
yourself, `serve` is opt-in: `cargo install pmetal --features serve`.
:::

## See Also

- [pmetal infer](/cli/infer/) — Interactive inference
- [pmetal mcp](/cli/mcp/) — MCP server for Claude Desktop and Claude Code
- [Feature Flags](/configuration/feature-flags/) — Enable the serve feature
