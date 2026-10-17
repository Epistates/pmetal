# pmetal-serve

OpenAI-compatible inference server for PMetal.

## Overview

This crate provides a drop-in local inference backend compatible with the OpenAI API. Built on Axum, it serves PMetal models via standard OpenAI-format HTTP endpoints with streaming SSE support.

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completions (non-streaming and SSE streaming), with images and videos |
| `/v1/completions` | POST | Raw text completions |
| `/v1/embeddings` | POST | Pooled sentence embeddings |
| `/v1/messages` | POST | Anthropic-compatible messages, with image blocks |
| `/v1/models` | GET | List loaded models |
| `/v1/metrics` | GET | Rolling serving metrics (tok/s, latencies, request counts) |
| `/health` | GET | Liveness check |

Qwen3.5-family vision models read images (`image_url` parts holding a `data:image/...;base64,`
URI) and videos (a `video` part holding frames) in chat messages, and Llama 3.2 Vision reads
images; see `media` for the accepted forms and what is refused. Any other model answers such a
request with a 400 that names it.

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
        continuous_batching: None,
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
| `server` | `ServeConfig` and server startup |
| `types` | OpenAI-compatible request/response types |
| `media` | Image and video content parts: parsing, validation, decoding |
| `anthropic` | The `/v1/messages` endpoint |
| `error` | Error handling |

## Dependencies

- **axum** 0.8 — HTTP framework
- **tower-http** — CORS and tracing middleware
- **tokio** — async runtime
- **tokio-stream** — SSE streaming

## License

MIT OR Apache-2.0
