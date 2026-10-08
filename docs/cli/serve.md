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
  tokens), so a 640×480 picture is 300 tokens and a 12-megapixel photo about 11,700: raise
  `--max-seq-len` from its default of 4096 for large photos, or downscale them first. The vision
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

:::note
The prebuilt release binary and the Homebrew formula both ship this command. If you build PMetal
yourself, `serve` is opt-in: `cargo install pmetal --features serve`.
:::

## See Also

- [pmetal infer](/cli/infer/) — Interactive inference
- [pmetal mcp](/cli/mcp/) — MCP server for Claude Desktop and Claude Code
- [Feature Flags](/configuration/feature-flags/) — Enable the serve feature
