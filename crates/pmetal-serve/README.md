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
| `/v1/messages/count_tokens` | POST | The input tokens a messages request would use |
| `/v1/models`, `/v1/models/{id}` | GET | List loaded models; fetch one |
| `/v1/metrics` | GET | Rolling serving metrics as JSON (tok/s, latencies, request counts, speculative acceptance) |
| `/metrics` | GET | Prometheus / OpenMetrics metrics |
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
  `logprobs` / `top_logprobs` (with `bytes`; on completions, `echo` with `logprobs` scores the
  prompt too, `max_tokens: 0` only that), `stream_options.include_usage`, usage with
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
* **Speculative decoding**: `InferenceEngine::enable_speculation` loads a drafter (a Qwen3.5-family
  model's MTP head, a Gemma 4 assistant or a DFlash draft model) beside the model; requests then
  decode on it, through their own sampler, to the tokens plain decoding gives. See
  `pmetal_models::speculator`.
* **Metrics**: `/metrics` is Prometheus text (OpenMetrics when the scraper asks for it): requests by
  route, latency and time-to-first-token histograms, tokens, queue and slot gauges, the prefix
  cache, speculative acceptance and GPU memory (`telemetry`).
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

## Continuous batching

With `continuous_batching` set, requests share the model instead of taking
turns. The scheduler re-forms the batch after every step, so a request joins
as soon as a slot is free and leaves the step it finishes (iteration-level
scheduling, Yu et al., OSDI 2022). Each step decodes one token for every
decoding request and runs one chunk of the prompt that is prefilling, so a
long prompt delays the others by a chunk and no more (chunked prefill,
Agrawal et al., OSDI 2024); a request's chunks are the same whatever else is
running, so its reply does not depend on its neighbours' prompts.

A model with a fused `[N, 1]` forward (Llama, Mistral, Qwen 2 and 3,
Qwen3-MoE, Gemma 1 and 2, Phi, Cohere and Granite with a plain RoPE) and a
plain dense KV cache decodes every running request in one forward over a
shared KV cache, reading the weights once however many requests there are.
Everything else (hybrid models with recurrent state; quantized, TurboQuant
and sliding-window caches; architectures without the fused forward) decodes
one forward per request on the request's own cache.

A request that begins with tokens an earlier request already ran reuses their
K/V instead of running them again. A hybrid model's cache entries hold its
recurrent state beside the K/V, so they serve a request that repeats the whole
of an earlier prompt (a chat's next turn), not one that diverges inside it.

A request that decodes alone is bit for bit what the single-request path gives
it, and so is every request in `BatcherConfig::per_slot_decode` mode (the
CLI's `--cb-exact`), whatever joins and leaves. In a fused batch with company
the matmuls are `[N, hidden]` products where a lone request's are vector
products, which add in another order: a bf16 logit can differ in its last bit
and a greedy near-tie can fall the other way, and the rows of one batch are
independent of one another.

Admission counts KV memory in token blocks against what the device can hold
beside the model (the device's recommended working set, not a fraction of its
RAM), reserves each request's worst case so nothing is ever preempted, lets a
small request pass a large one that does not fit yet only so many times, and
frees a request's slot the moment its client hangs up.

## Modules

| Module | Description |
|--------|-------------|
| `engine` | `InferenceEngine` — model loading and generation |
| `continuous_batch` | The scheduler: slots, fair admission by KV memory, the iteration plan |
| `continuous_driver` | Per-slot and fused KV caches, batched sampling, the decode step |
| `continuous_pump` | The pump that runs the scheduler against the model, per request events |
| `routes` | Axum route handlers and `ServingMetrics` |
| `telemetry` | The Prometheus registry, the request-counting middleware and `GET /metrics` |
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
- **prometheus-client** — the metrics registry and OpenMetrics encoding

## License

MIT OR Apache-2.0
