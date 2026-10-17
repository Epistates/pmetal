//! `/v1/embeddings` against OpenAI's documented behavior and the model's own
//! sentence-transformers files: input forms, unit-length vectors, `dimensions`
//! (Matryoshka truncation, renormalized), `encoding_format: "base64"`, the
//! pooling the model was trained with, the tokenizer's special tokens, and
//! batching that does not change a vector.

mod support;

use axum::http::StatusCode;
use base64::Engine as _;
use pmetal_serve::ServeConfig;
use serde_json::{Value, json};
use support::*;

async fn embed(router: &axum::Router, extra: Value) -> Value {
    let mut body = json!({"model": MODEL_ID, "input": "hi there"});
    body.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    let reply = post(router, "/v1/embeddings", &body).await;
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.text());
    reply.json()
}

fn vector(reply: &Value, i: usize) -> Vec<f32> {
    reply["data"][i]["embedding"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect()
}

fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

#[tokio::test(flavor = "multi_thread")]
async fn vectors_are_unit_length_and_in_input_order() {
    let (router, _dir) = app(256);
    let reply = embed(&router, json!({"input": ["hi there", "the", "hi"]})).await;
    assert_eq!(reply["object"], "list");
    assert_eq!(reply["model"], MODEL_ID);
    let data = reply["data"].as_array().unwrap();
    assert_eq!(data.len(), 3);
    for (i, item) in data.iter().enumerate() {
        assert_eq!(item["object"], "embedding");
        assert_eq!(item["index"], i);
        let v = vector(&reply, i);
        assert_eq!(v.len(), 32);
        assert!((norm(&v) - 1.0).abs() < 1e-4, "norm {}", norm(&v));
    }
    assert_ne!(vector(&reply, 0), vector(&reply, 1));
    let tokens = tokenizer().encode("hi there").unwrap().len()
        + tokenizer().encode("the").unwrap().len()
        + tokenizer().encode("hi").unwrap().len();
    assert_eq!(reply["usage"]["prompt_tokens"], tokens);
    assert_eq!(reply["usage"]["total_tokens"], tokens);
    assert_no_unobserved_error();
}

#[tokio::test(flavor = "multi_thread")]
async fn token_ids_embed_as_the_text_does() {
    let (router, _dir) = app(256);
    let ids = tokenizer().encode("hi there").unwrap();
    let by_text = embed(&router, json!({"input": "hi there"})).await;
    let by_ids = embed(&router, json!({"input": ids})).await;
    let by_lists = embed(&router, json!({"input": [ids.clone(), ids.clone()]})).await;
    assert_eq!(vector(&by_text, 0), vector(&by_ids, 0));
    assert_eq!(vector(&by_lists, 1), vector(&by_text, 0));
    assert_eq!(by_ids["usage"]["prompt_tokens"], ids.len());
}

#[tokio::test(flavor = "multi_thread")]
async fn a_batch_gives_each_input_the_vector_it_has_alone() {
    let (router, _dir) = app(512);
    let texts: Vec<String> = (1..=40)
        .map(|n| format!("hi there {}", "the ".repeat(n)))
        .collect();
    let batched = embed(&router, json!({"input": texts})).await;
    for i in [0usize, 7, 19, 39] {
        let alone = embed(&router, json!({"input": texts[i]})).await;
        let (a, b) = (vector(&alone, 0), vector(&batched, i));
        let worst = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max);
        assert!(worst < 1e-4, "input {i}: differs by {worst}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn dimensions_truncate_and_renormalize() {
    let (router, _dir) = app(256);
    let full = vector(&embed(&router, json!({})).await, 0);
    let short = vector(&embed(&router, json!({"dimensions": 8})).await, 0);
    assert_eq!(short.len(), 8);
    assert!((norm(&short) - 1.0).abs() < 1e-4);
    // The leading components, scaled to unit length.
    let lead = &full[..8];
    let scale = norm(lead);
    for (got, want) in short.iter().zip(lead) {
        assert!((got - want / scale).abs() < 1e-4);
    }
    for (extra, param) in [
        (json!({"dimensions": 0}), "dimensions"),
        (json!({"dimensions": 33}), "dimensions"),
        (json!({"encoding_format": "hex"}), "encoding_format"),
        (json!({"input": []}), "input"),
        (json!({"input": ""}), "input.0"),
        (json!({"input": ["hi", ""]}), "input.1"),
    ] {
        let mut body = json!({"model": MODEL_ID, "input": "hi"});
        body.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        let reply = post(&router, "/v1/embeddings", &body).await;
        assert_eq!(
            reply.status,
            StatusCode::BAD_REQUEST,
            "{extra}: {}",
            reply.text()
        );
        assert_eq!(reply.json()["error"]["param"], param, "{extra}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn base64_is_the_same_vector_as_little_endian_f32_bytes() {
    let (router, _dir) = app(256);
    let floats = vector(&embed(&router, json!({})).await, 0);
    let reply = embed(&router, json!({"encoding_format": "base64"})).await;
    let encoded = reply["data"][0]["embedding"].as_str().unwrap();
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .unwrap();
    assert_eq!(bytes.len(), 32 * 4);
    let (words, rest) = bytes.as_chunks::<4>();
    assert!(rest.is_empty());
    let decoded: Vec<f32> = words.iter().map(|b| f32::from_le_bytes(*b)).collect();
    assert_eq!(decoded, floats);
}

#[tokio::test(flavor = "multi_thread")]
async fn the_models_sentence_transformers_files_choose_pooling() {
    // A last-token embedder: modules.json, its pooling module, no Normalize.
    let tiny = tiny_with(256, |dir| {
        std::fs::write(
            dir.join("modules.json"),
            json!([
                {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
                {"idx": 1, "name": "1", "path": "1_Pooling",
                 "type": "sentence_transformers.models.Pooling"}
            ])
            .to_string(),
        )
        .unwrap();
        std::fs::create_dir(dir.join("1_Pooling")).unwrap();
        std::fs::write(
            dir.join("1_Pooling/config.json"),
            json!({"word_embedding_dimension": 32, "pooling_mode_cls_token": false,
                   "pooling_mode_mean_tokens": false, "pooling_mode_lasttoken": true})
            .to_string(),
        )
        .unwrap();
    });
    let (router, _dir) = tiny.router(&ServeConfig::default());

    let default = embed(&router, json!({})).await;
    let last = embed(&router, json!({"pooling": "last_token"})).await;
    let mean = embed(&router, json!({"pooling": "mean"})).await;
    assert_eq!(
        vector(&default, 0),
        vector(&last, 0),
        "the model's own pooling is the default"
    );
    assert_ne!(vector(&default, 0), vector(&mean, 0));
    // No Normalize module: the vector is the pooler's, as the model defines it.
    assert!((norm(&vector(&default, 0)) - 1.0).abs() > 1e-3);
}

#[tokio::test(flavor = "multi_thread")]
async fn text_is_tokenized_with_the_tokenizers_special_tokens() {
    // A tokenizer that puts a BOS in front of every encoding, as a model that
    // is trained on one does.
    let dir = tempfile::tempdir().unwrap();
    write_model_dir_for_tests(dir.path());
    let tokenizer =
        pmetal_data::tokenizer::testing::byte_level_bpe(MERGES, SPECIALS, Some("<|im_start|>"));
    let with_bos = tokenizer.encode_with_special_tokens("hi there").unwrap();
    let without = tokenizer.encode("hi there").unwrap();
    assert_eq!(with_bos.len(), without.len() + 1);
    let engine = pmetal_serve::InferenceEngine::new_with_backend(
        tiny_llama,
        tokenizer,
        MODEL_ID.into(),
        dir.path(),
        256,
        false,
        1024,
    )
    .unwrap();
    let router = pmetal_serve::server::build_router_with(engine, &ServeConfig::default());
    let reply = embed(&router, json!({})).await;
    assert_eq!(reply["usage"]["prompt_tokens"], with_bos.len());
}

#[tokio::test(flavor = "multi_thread")]
async fn an_input_longer_than_the_context_is_a_400() {
    let (router, _dir) = app(32);
    let long = "the ".repeat(100);
    let reply = post(
        &router,
        "/v1/embeddings",
        &json!({"model": MODEL_ID, "input": ["hi", long]}),
    )
    .await;
    assert_eq!(reply.status, StatusCode::BAD_REQUEST, "{}", reply.text());
    assert_eq!(reply.json()["error"]["param"], "input.1");
    assert_eq!(reply.json()["error"]["code"], "context_length_exceeded");
}
