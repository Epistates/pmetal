//! Speculative decoding behind the server: a drafter changes how fast the
//! reply comes, never what it says.
//!
//! Two servers over the same seeded tiny model, one decoding plainly and one
//! with a drafter, answer the same requests; the replies must be equal, over
//! every feature a reply has: streaming, stop sequences, a seed, a logit bias,
//! penalties, a grammar, several choices, log-probabilities (which a drafter
//! cannot report, so the request decodes plainly). The drafter is a DFlash
//! draft model with random weights, so most of its drafts are wrong and every
//! round exercises the rewind.

mod support;

use axum::Router;
use axum::http::StatusCode;
use pmetal_models::architectures::dflash_draft::{
    DFlashDraftConfig, DFlashDraftModel, DFlashExtras,
};
use pmetal_models::dflash_drafts::DFlashDraft;
use pmetal_models::speculator::Speculator;
use pmetal_runtime::compat::random;
use pmetal_serve::{BatcherConfig, ServeConfig};
use serde_json::{Value, json};
use support::*;

/// A random DFlash draft for the tiny llama of `tests/support` (hidden size
/// 32, two layers, a 288-token vocabulary).
fn draft() -> DFlashDraft {
    random::seed(99);
    DFlashDraft::V1(
        DFlashDraftModel::new(DFlashDraftConfig {
            model_type: "dflash_qwen3".to_string(),
            hidden_size: 32,
            num_hidden_layers: 2,
            intermediate_size: 64,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            rms_norm_eps: 1e-6,
            vocab_size: 288,
            max_position_embeddings: 256,
            rope_theta: Some(10_000.0),
            head_dim: 16,
            tie_word_embeddings: true,
            block_size: 4,
            dflash_config: DFlashExtras {
                target_layer_ids: vec![0, 1],
                mask_token_id: 5,
                ..Default::default()
            },
            ..Default::default()
        })
        .unwrap(),
    )
}

fn speculating_tiny(max_seq_len: usize) -> Tiny {
    let tiny = tiny(max_seq_len);
    let info = tiny
        .engine
        .enable_speculation_with(|model, max| Speculator::from_dflash(model, draft(), None, max))
        .expect("the draft fits the tiny model");
    assert_eq!((info.kind, info.draft_tokens), ("dflash", 4));
    tiny
}

/// A plain server and a speculating one over the same model.
fn pair() -> (Router, Router, tempfile::TempDir, tempfile::TempDir) {
    let (plain, a) = app(512);
    let (speculating, b) = speculating_tiny(512).router(&ServeConfig::default());
    (plain, speculating, a, b)
}

/// A reply without what differs between any two requests.
fn essence(mut reply: Value) -> Value {
    for key in ["id", "created", "system_fingerprint"] {
        reply.as_object_mut().unwrap().remove(key);
    }
    // A drafter prefills every prompt in full: the prefix cache is not read.
    if let Some(details) = reply["usage"]["prompt_tokens_details"].as_object_mut() {
        details.remove("cached_tokens");
    }
    reply
}

async fn same_reply(plain: &Router, speculating: &Router, path: &str, body: Value) -> Value {
    let a = post(plain, path, &body).await;
    let b = post(speculating, path, &body).await;
    assert_eq!(a.status, StatusCode::OK, "{}", a.text());
    assert_eq!(b.status, StatusCode::OK, "{}", b.text());
    let (a, b) = (essence(a.json()), essence(b.json()));
    assert_eq!(a, b, "{body}");
    a
}

fn chat(extra: Value) -> Value {
    let mut body = json!({
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "say something"}],
        "max_tokens": 24,
        "temperature": 0,
    });
    body.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    body
}

fn scraped(body: &str, series: &str) -> f64 {
    body.lines()
        .filter(|l| !l.starts_with('#'))
        .find_map(|l| l.strip_prefix(series)?.strip_prefix(' ')?.trim().parse().ok())
        .unwrap_or_else(|| panic!("{series} in\n{body}"))
}

#[tokio::test(flavor = "multi_thread")]
async fn a_drafter_changes_the_speed_and_not_the_reply() {
    let (plain, speculating, _a, _b) = pair();
    let chat_path = "/v1/chat/completions";

    // Greedy, and with the penalties and a bias that shape every draw.
    same_reply(&plain, &speculating, chat_path, chat(json!({}))).await;
    same_reply(
        &plain,
        &speculating,
        chat_path,
        chat(json!({"frequency_penalty": 0.7, "presence_penalty": 0.4, "logit_bias": {"104": 5.0}})),
    )
    .await;
    // Sampled, seeded: the target's own draws, however many drafts were kept.
    for seed in [1, 2, 3] {
        same_reply(
            &plain,
            &speculating,
            chat_path,
            chat(json!({"temperature": 0.9, "top_p": 0.9, "seed": seed, "max_tokens": 40})),
        )
        .await;
    }
    // Several choices, each its own seeded draw.
    same_reply(
        &plain,
        &speculating,
        chat_path,
        chat(json!({"temperature": 1.0, "seed": 7, "n": 3})),
    )
    .await;
    // A stop sequence cuts the reply where plain decoding cuts it.
    let whole = same_reply(&plain, &speculating, chat_path, chat(json!({}))).await;
    let text = whole["choices"][0]["message"]["content"].as_str().unwrap();
    let stop: String = text.chars().skip(5).take(2).collect();
    let cut = same_reply(
        &plain,
        &speculating,
        chat_path,
        chat(json!({"stop": [stop.clone()]})),
    )
    .await;
    assert_eq!(cut["choices"][0]["finish_reason"], "stop");
    assert!(
        !cut["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .contains(&stop)
    );
    // The reply's length, usage included.
    same_reply(&plain, &speculating, chat_path, chat(json!({"max_tokens": 3}))).await;
    // Raw completions, a prompt as token ids.
    same_reply(
        &plain,
        &speculating,
        "/v1/completions",
        json!({"model": MODEL_ID, "prompt": [104, 105, 32, 116], "max_tokens": 20, "temperature": 0}),
    )
    .await;
    assert_no_unobserved_error();
}

#[tokio::test(flavor = "multi_thread")]
async fn a_streamed_reply_is_the_same_frames_of_text() {
    let (plain, speculating, _a, _b) = pair();
    let body = chat(json!({"stream": true, "stream_options": {"include_usage": true}}));
    let text = |reply: &Reply| -> String {
        reply
            .sse_json()
            .iter()
            .filter_map(|f| f["choices"][0]["delta"]["content"].as_str().map(str::to_owned))
            .collect()
    };
    let (a, b) = (
        post(&plain, "/v1/chat/completions", &body).await,
        post(&speculating, "/v1/chat/completions", &body).await,
    );
    assert_eq!(b.status, StatusCode::OK, "{}", b.text());
    assert_eq!(text(&a), text(&b));
    assert!(!text(&b).is_empty());
    let usage = |reply: &Reply| reply.sse_json().last().unwrap()["usage"].clone();
    assert_eq!(usage(&a), usage(&b));
    assert_eq!(b.sse().last().unwrap().1, "[DONE]");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_grammar_is_followed_under_speculation_as_without() {
    let (plain, speculating, _a, _b) = pair();
    let schema = json!({
        "type": "object",
        "properties": {
            "name": {"type": "string", "maxLength": 6},
            "kind": {"enum": ["cat", "dog"]}
        },
        "required": ["name", "kind"],
        "additionalProperties": false
    });
    for seed in [3, 4] {
        let reply = same_reply(
            &plain,
            &speculating,
            "/v1/chat/completions",
            chat(json!({
                "temperature": 1.0, "seed": seed, "max_tokens": 80,
                "response_format": {"type": "json_schema",
                    "json_schema": {"name": "pet", "schema": schema, "strict": true}}
            })),
        )
        .await;
        // The grammar's doing, not the model's: noise that parses.
        let parsed: Value =
            serde_json::from_str(reply["choices"][0]["message"]["content"].as_str().unwrap())
                .expect("a reply that follows the schema parses");
        assert!(["cat", "dog"].contains(&parsed["kind"].as_str().unwrap()));
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn log_probabilities_come_from_the_target_by_decoding_plainly() {
    let (plain, speculating, _a, _b) = pair();
    let metrics = |router: &Router| {
        let router = router.clone();
        async move { get(&router, "/v1/metrics").await.json() }
    };
    let before = metrics(&speculating).await;
    let reply = same_reply(
        &plain,
        &speculating,
        "/v1/chat/completions",
        chat(json!({"logprobs": true, "top_logprobs": 2})),
    )
    .await;
    assert_eq!(reply["choices"][0]["logprobs"]["content"].as_array().unwrap().len(), 24);
    let after = metrics(&speculating).await;
    assert_eq!(
        after["speculative_drafted_tokens"], before["speculative_drafted_tokens"],
        "a request for log-probabilities decodes without the drafter"
    );
    // Prompt log-probabilities too.
    same_reply(
        &plain,
        &speculating,
        "/v1/completions",
        json!({"model": MODEL_ID, "prompt": "hi there", "max_tokens": 0,
               "echo": true, "logprobs": 1, "temperature": 0}),
    )
    .await;
}

#[tokio::test(flavor = "multi_thread")]
async fn the_acceptance_rate_is_in_both_metrics() {
    let (_plain, speculating, _a, _b) = pair();
    let reply = post(&speculating, "/v1/chat/completions", &chat(json!({"max_tokens": 30}))).await;
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.text());

    let json = get(&speculating, "/v1/metrics").await.json();
    let drafted = json["speculative_drafted_tokens"].as_u64().unwrap();
    let accepted = json["speculative_accepted_tokens"].as_u64().unwrap();
    assert!(drafted > 0, "{json}");
    assert!(accepted <= drafted);
    assert_eq!(
        json["speculative_acceptance_rate"].as_f64().unwrap(),
        accepted as f64 / drafted as f64
    );

    let scrape = get_with_headers(
        &speculating,
        "/metrics",
        &[("accept", "application/openmetrics-text")],
    )
    .await
    .text();
    assert_eq!(scraped(&scrape, "pmetal_speculative_drafted_tokens_total"), drafted as f64);
    assert_eq!(scraped(&scrape, "pmetal_speculative_accepted_tokens_total"), accepted as f64);
    assert_no_unobserved_error();
}

#[tokio::test(flavor = "multi_thread")]
async fn speculation_and_continuous_batching_exclude_each_other() {
    // Batching first, then a drafter.
    let batching = tiny(256);
    batching
        .engine
        .enable_continuous_batching_auto(BatcherConfig {
            max_slots: 2,
            ..Default::default()
        })
        .unwrap();
    let err = batching
        .engine
        .enable_speculation_with(|model, max| Speculator::from_dflash(model, draft(), None, max))
        .expect_err("a drafter cannot join a batching server");
    assert!(err.to_string().contains("continuous batching"), "{err}");

    // A drafter first, then batching.
    let speculating = speculating_tiny(256);
    let err = speculating
        .engine
        .enable_continuous_batching_auto(BatcherConfig::default())
        .expect_err("batching cannot join a drafting server");
    assert!(err.to_string().contains("speculative decoding"), "{err}");

    // Twice is an error too.
    let err = speculating
        .engine
        .enable_speculation_with(|model, max| Speculator::from_dflash(model, draft(), None, max))
        .expect_err("one drafter at a time");
    assert!(err.to_string().contains("already"), "{err}");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_drafter_that_does_not_fit_is_refused_when_enabled() {
    let tiny = tiny(256);
    // The tiny llama has hidden size 32; this draft has 48.
    random::seed(5);
    let wide = DFlashDraft::V1(
        DFlashDraftModel::new(DFlashDraftConfig {
            model_type: "dflash_qwen3".to_string(),
            hidden_size: 48,
            num_hidden_layers: 1,
            intermediate_size: 64,
            num_attention_heads: 3,
            num_key_value_heads: 1,
            vocab_size: 288,
            max_position_embeddings: 256,
            rope_theta: Some(10_000.0),
            head_dim: 16,
            tie_word_embeddings: true,
            block_size: 4,
            dflash_config: DFlashExtras {
                target_layer_ids: vec![0, 1],
                mask_token_id: 5,
                ..Default::default()
            },
            ..Default::default()
        })
        .unwrap(),
    );
    let err = tiny
        .engine
        .enable_speculation_with(move |model, max| Speculator::from_dflash(model, wide, None, max))
        .expect_err("a draft of another size is refused");
    assert!(err.to_string().contains("hidden size"), "{err}");
    assert!(tiny.engine.speculation().is_none());
}
