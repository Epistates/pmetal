# pmetal-serve

OpenAI-compatible inference server for PMetal.

## Overview

This crate provides a drop-in local inference backend compatible with the OpenAI API. Built on Axum, it serves PMetal models via standard OpenAI-format HTTP endpoints with streaming SSE support.

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completions (non-streaming and SSE streaming), with images and videos |
| `/v1/completions` | POST | Raw text completions |
| `/v1/embeddings` | POST | Embeddings: the model's own pooling, unit length, `dimensions`, `base64` |
| `/v1/responses` | POST | OpenAI Responses API: output items, function tools, streaming events |
| `/v1/messages` | POST | Anthropic-compatible messages, with image blocks |
| `/v1/models`, `/v1/models/{id}` | GET | List loaded models; fetch one |
| `/v1/metrics` | GET | Rolling serving metrics (tok/s, latencies, request counts) |
| `/health` | GET | Liveness check |

Qwen3.5-family vision models read images (`image_url` parts holding a `data:image/...;base64,`
URI) and videos (a `video` part holding frames) in chat messages, and Llama 3.2 Vision reads
images; see `media` for the accepted forms and what is refused. Any other model answers such a
request with a 400 that names it. Images are bounded (50 megapixels each, 200 in all, 512 per
request) from their headers, before anything is decoded.

## What the API does

* **Chat completions and completions**: `n` choices, `stop` (up to four sequences, matched in the
  text however the tokens fall), `seed`, `max_tokens` / `max_completion_tokens` (cut to the
  context left; a prompt that fills the context is `context_length_exceeded`), `logit_bias`,
  `logprobs` / `top_logprobs` (with `bytes`), `stream_options.include_usage`, usage with
  `cached_tokens` and `reasoning_tokens`, and `finish_reason` of `stop`, `length` or
  `tool_calls`. A thinking model's reasoning comes back as `reasoning_content`. A documented
  field the server cannot honour (`audio`, `prediction`, `web_search_options`, `suffix`, ...) is a
  400 `unsupported_parameter` naming it, never ignored.
* **Tools**: calls are read out of the reply per model family (Hermes/Qwen JSON, Qwen 3.5 XML,
  Llama 3, Mistral, Gemma 4, gpt-oss) and streamed as `tool_calls` deltas; `tool_choice`
  `required` or a named function forces a call by constrained decoding (families whose syntax is
  not JSON answer 400 for it).
* **Structured output**: `response_format` `json_object` and `json_schema` are enforced token by
  token (llguidance): the reply always parses and validates.
* **Messages API**: `system` text or blocks, `tool_use` / `tool_result` / `thinking` blocks,
  `tool_choice`, `stop_sequences` with the matched `stop_sequence`, and the streaming events
  (`message_start`, `content_block_*`, `message_delta`, `message_stop`, `ping`, `error`).
* **Embeddings**: input as a string, strings, token ids or lists of them; pooling and
  normalization from the model's sentence-transformers files; `dimensions` truncates and
  renormalizes; `encoding_format: "base64"`.
* **Safety**: loopback by default; optional API keys; CORS only for configured origins; a body
  limit and a request time limit; a client that leaves cancels its generation.

A decision model (a Clef release) is served by its own router instead: `POST /v1/systemone`
answers typed questions about a state, beside `GET /v1/models` and `GET /health`.

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

```rust,no_run
use std::path::Path;

use pmetal_data::Tokenizer;
use pmetal_models::DynamicModel;
use pmetal_serve::{InferenceEngine, ServeConfig, server::run_server};

async fn serve(model_dir: &str) -> anyhow::Result<()> {
    let tokenizer = Tokenizer::from_model_dir(model_dir)?;

    // The model loads on the engine's own thread and stays there; every
    // request runs on it.
    let dir = model_dir.to_string();
    let engine = InferenceEngine::new(
        move || Ok(DynamicModel::load(&dir)?),
        tokenizer,
        "my-model".to_string(),
        Path::new(model_dir),
        4096, // max_seq_len
    )?;

    // `continuous_batching: Some(BatcherConfig { .. })` opts into the
    // continuous-batching engine instead of one-request-at-a-time decode.
    let config = ServeConfig {
        port: 8080,
        host: "127.0.0.1".to_string(),
        max_concurrent: 16,
        // Required as `Authorization: Bearer <key>` or `x-api-key: <key>`
        // when not empty.
        api_keys: vec![],
        ..ServeConfig::default()
    };

    run_server(engine, config).await
}
```

## Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `port` | Port to listen on | 8080 |
| `host` | Host to bind to | `127.0.0.1` |
| `max_concurrent` | Maximum concurrent requests | 16 |
| `api_keys` | Keys accepted as `Authorization: Bearer` or `x-api-key`; empty leaves the server open (`/health` is always open) | none |
| `cors_allowed_origins` | Browser origins allowed to call the server, or `*`; empty sends no CORS headers | none |
| `max_body_bytes` | Largest request body | 64 MiB |
| `request_timeout` | Longest one request may run; a request past it ends with a timeout error | 30 minutes |
| `max_choices` | Largest `n` a request may ask for | 16 |

Errors are the APIs' own: OpenAI's `{"error": {"message", "type", "param", "code"}}` on every
route but `/v1/messages`, whose errors are Anthropic's `{"type": "error", "error": {"type",
"message"}}`. A request naming a model the server does not serve is a 404 `model_not_found`.

## Modules

| Module | Description |
|--------|-------------|
| `engine` | `InferenceEngine` — model loading and generation |
| `routes` | Axum route handlers and `ServingMetrics` |
| `server` | `ServeConfig` and server startup |
| `types` | OpenAI-compatible request/response types |
| `media` | Image and video content parts: parsing, validation, decoding |
| `anthropic` | The `/v1/messages` endpoint |
| `responses` | The `/v1/responses` endpoint |
| `request` | Parameter validation and the engine's sampling parameters |
| `assemble` | From generated tokens to a reply: text, reasoning, stop sequences, tool calls |
| `stop` | Text-level stop sequences |
| `tools` | Tool-call parsers per model family, `tool_choice` plans |
| `structured` | Grammar-constrained decoding (`response_format`, forced tool calls) |
| `vocab` | The bytes each token stands for |
| `extract` | Request bodies that fail with the APIs' own errors |
| `error` | Error handling |

## Dependencies

- **axum** 0.8 — HTTP framework
- **tower-http** — CORS and tracing middleware
- **tokio** — async runtime
- **tokio-stream** — SSE streaming

## License

MIT OR Apache-2.0
