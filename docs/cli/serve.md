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

- `POST /v1/chat/completions` — chat completions (streaming and non-streaming, tool calling, token logprobs)
- `POST /v1/completions` — text completions
- `POST /v1/embeddings` — embeddings, 17 architectures via `forward_hidden`
- `POST /v1/messages` — Anthropic-compatible messages, with SSE streaming
- `GET /v1/models` — list loaded models
- `GET /v1/metrics` — serving metrics
- `GET /health` — liveness check

Request bodies are capped at 2 MiB.

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
