//! `pmetal serve` returns a thinking model's reasoning apart from its answer.
//! Opt-in: needs a Qwen3.5-family checkpoint (Qwen3.5-0.8B is enough).
//!
//! ```text
//! PMETAL_QWEN3_5_DIR=<snapshot> cargo test --release -p pmetal --features serve \
//!     --test serve_reasoning_real_weights -- --nocapture
//! ```
//!
//! With thinking on, Qwen3.5's chat template opens the `<think>` block in the
//! prompt and the model closes it. Checked, greedy, against the engine's own
//! tokens split by `pmetal_data::stream_format::SplitOutput`:
//!
//! * `/v1/chat/completions` puts the reasoning in `message.reasoning_content`
//!   and only the answer in `content`; streamed, `delta.reasoning_content`
//!   and `delta.content` add up to the same two;
//! * `/v1/messages` answers with a `thinking` block ahead of the `text`
//!   block, streamed as `thinking_delta`s then `text_delta`s;
//! * with thinking off there is no reasoning at all.

#![cfg(feature = "serve")]

mod common;

use std::path::PathBuf;

use common::{post, sse_data, wait_for};
use pmetal_data::stream_format::SplitOutput;
use pmetal_models::DynamicModel;
use pmetal_serve::engine::SamplingParams;
use pmetal_serve::{InferenceEngine, ServeConfig};
use serde_json::{Value, json};

const QUESTION: &str = "What is 17 + 25? Answer with the number.";
const MAX_TOKENS: usize = 1500;
const PORT: u16 = 18732;

fn model_dir() -> Option<PathBuf> {
    std::env::var_os("PMETAL_QWEN3_5_DIR").map(PathBuf::from)
}

fn greedy() -> SamplingParams {
    SamplingParams {
        max_tokens: MAX_TOKENS,
        temperature: 0.0,
        top_k: None,
        top_p: None,
        min_p: None,
        repetition_penalty: None,
        frequency_penalty: None,
        presence_penalty: None,
        seed: None,
        extra_stop_token_ids: vec![],
        stop_sequences: vec![],
        logprobs_top_n: None,
    }
}

/// The concatenation of `field` over a chat stream's deltas.
fn streamed(chunks: &[Value], field: &str) -> String {
    chunks
        .iter()
        .filter_map(|chunk| chunk["choices"][0]["delta"][field].as_str())
        .collect()
}

#[test]
fn serve_returns_reasoning_apart_from_the_answer() {
    let Some(dir) = model_dir() else {
        eprintln!("PMETAL_QWEN3_5_DIR is not set; skipping");
        return;
    };
    let tokenizer = pmetal_data::Tokenizer::from_model_dir(&dir).unwrap();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let load_dir = dir.clone();
        let engine = InferenceEngine::new_with_backend(
            move || Ok(DynamicModel::load(&load_dir)?),
            pmetal_data::Tokenizer::from_model_dir(&dir).unwrap(),
            "qwen3_5".into(),
            &dir,
            8192,
            false,
            4096,
        )
        .unwrap();

        // What the engine generates, split.
        let messages: Vec<pmetal_serve::types::ChatMessage> =
            serde_json::from_value(json!([{"role": "user", "content": QUESTION}])).unwrap();
        let thinking: pmetal_data::chat_templates::ChatTemplateKwargs =
            serde_json::from_value(json!({"enable_thinking": true})).unwrap();
        let prompt = engine
            .prepare_chat(&messages, None, &thinking)
            .await
            .unwrap();
        let (tokens, _, finish, _) = engine.generate_prompt(&prompt, greedy()).await.unwrap();
        pmetal_bridge::check_last_error().unwrap();
        assert_eq!(finish, "stop", "the model finished its answer");
        let split = SplitOutput::split(&tokenizer, &prompt.input_ids, &tokens);
        let reasoning = split
            .reasoning_text(&tokenizer)
            .expect("the model reasoned");
        let answer = tokenizer.decode(&split.answer).unwrap().trim().to_string();
        println!("reasoning: {} chars; answer {answer:?}", reasoning.len());
        assert!(answer.contains("42"), "{answer:?}");
        assert!(!reasoning.contains("think>") && !answer.contains("think>"));

        tokio::spawn(pmetal_serve::server::run_server(
            engine,
            ServeConfig {
                port: PORT,
                ..Default::default()
            },
        ));
        wait_for(PORT).await;

        let chat = |stream: bool, enable_thinking: bool| {
            json!({
                "model": "qwen3_5", "max_tokens": MAX_TOKENS, "temperature": 0, "stream": stream,
                "chat_template_kwargs": {"enable_thinking": enable_thinking},
                "messages": [{"role": "user", "content": QUESTION}]
            })
        };

        let (status, body) = post(PORT, "/v1/chat/completions", &chat(false, true)).await;
        assert_eq!(status, 200, "{body}");
        let reply: Value = serde_json::from_str(&body).unwrap();
        let message = &reply["choices"][0]["message"];
        assert_eq!(message["reasoning_content"], json!(reasoning));
        assert_eq!(message["content"].as_str().unwrap().trim(), answer);

        let (status, body) = post(PORT, "/v1/chat/completions", &chat(true, true)).await;
        assert_eq!(status, 200, "{body}");
        let chunks = sse_data(&body);
        assert_eq!(streamed(&chunks, "reasoning_content").trim(), reasoning);
        assert_eq!(streamed(&chunks, "content").trim(), answer);
        // Reasoning comes first, then the answer, never in one delta.
        let first_content = chunks
            .iter()
            .position(|c| c["choices"][0]["delta"]["content"].is_string())
            .unwrap();
        assert!(
            chunks[first_content..]
                .iter()
                .all(|c| c["choices"][0]["delta"]["reasoning_content"].is_null())
        );

        let (status, body) = post(PORT, "/v1/chat/completions", &chat(false, false)).await;
        assert_eq!(status, 200, "{body}");
        let reply: Value = serde_json::from_str(&body).unwrap();
        assert!(reply["choices"][0]["message"]["reasoning_content"].is_null());
        assert!(
            reply["choices"][0]["message"]["content"]
                .as_str()
                .unwrap()
                .contains("42")
        );

        let messages = |stream: bool| {
            json!({
                "model": "qwen3_5", "max_tokens": MAX_TOKENS, "temperature": 0, "stream": stream,
                "thinking": {"type": "enabled", "budget_tokens": 1024},
                "messages": [{"role": "user", "content": QUESTION}]
            })
        };
        let (status, body) = post(PORT, "/v1/messages", &messages(false)).await;
        assert_eq!(status, 200, "{body}");
        let reply: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(reply["content"][0]["type"], "thinking");
        assert_eq!(reply["content"][0]["thinking"], json!(reasoning));
        assert_eq!(reply["content"][1]["type"], "text");
        assert_eq!(reply["content"][1]["text"].as_str().unwrap().trim(), answer);

        let (status, body) = post(PORT, "/v1/messages", &messages(true)).await;
        assert_eq!(status, 200, "{body}");
        let events = sse_data(&body);
        let deltas = |kind: &str, field: &str| -> String {
            events
                .iter()
                .filter(|e| e["delta"]["type"] == kind)
                .filter_map(|e| e["delta"][field].as_str())
                .collect()
        };
        assert_eq!(deltas("thinking_delta", "thinking").trim(), reasoning);
        assert_eq!(deltas("text_delta", "text").trim(), answer);
        let starts: Vec<&Value> = events
            .iter()
            .filter(|e| e["type"] == "content_block_start")
            .map(|e| &e["content_block"]["type"])
            .collect();
        assert_eq!(starts, [&json!("thinking"), &json!("text")]);
    });
}
