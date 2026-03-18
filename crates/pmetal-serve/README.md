# pmetal-serve

OpenAI-compatible inference server for PMetal.

## Overview

This crate provides a drop-in local inference backend compatible with the OpenAI API. Built on Axum, it serves PMetal models via standard OpenAI-format HTTP endpoints with streaming SSE support.

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completions (non-streaming and SSE streaming) |
| `/v1/completions` | POST | Raw text completions |
| `/v1/models` | GET | List loaded models |
| `/v1/metrics` | GET | Rolling serving metrics (tok/s, latencies, request counts) |
| `/health` | GET | Liveness check |

## Usage

### Via CLI

```bash
# Start server (requires pmetal-cli built with --features serve)
pmetal serve --model Qwen/Qwen3-0.6B --port 8080

# Query with curl
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-0.6B", "messages": [{"role": "user", "content": "Hello"}]}'
```

### As a Library

```rust
use pmetal_serve::{InferenceEngine, ServeConfig, server::run_server};

let engine = InferenceEngine::new(model, tokenizer)?;
let config = ServeConfig {
    port: 8080,
    host: "0.0.0.0".to_string(),
    max_concurrent: 16,
};

run_server(engine, config).await?;
```

## Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `port` | Port to listen on | 8080 |
| `host` | Host to bind to | `0.0.0.0` |
| `max_concurrent` | Maximum concurrent requests | 16 |

## Modules

| Module | Description |
|--------|-------------|
| `engine` | `InferenceEngine` — model loading and generation |
| `routes` | Axum route handlers and `ServingMetrics` |
| `server` | `ServeConfig` and server startup |
| `types` | OpenAI-compatible request/response types |
| `error` | Error handling |

## Dependencies

- **axum** 0.8 — HTTP framework
- **tower-http** — CORS and tracing middleware
- **tokio** — async runtime
- **tokio-stream** — SSE streaming

## License

MIT OR Apache-2.0
