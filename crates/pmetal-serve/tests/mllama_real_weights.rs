//! Llama 3.2 Vision through the server's engine against transformers, on a
//! real checkpoint. Opt-in: needs the checkpoint and a reference.
//!
//! ```text
//! python mllama_ref.py <checkpoint> <dir> mps [float32]   # writes shapes.png + reference.json
//! PMETAL_MLLAMA_CHECKPOINT=<checkpoint> PMETAL_MLLAMA_SERVE_REFERENCE=<dir> \
//!     cargo test --release -p pmetal-serve --test mllama_real_weights -- --nocapture
//! ```
//!
//! The reference is `MllamaProcessor.apply_chat_template` plus
//! `processor(images, text, add_special_tokens=False)` plus greedy
//! `generate` on one image and one question; `reference.json` holds its
//! prompt ids, its tokens, and the margin between the top two logits at each
//! step. Checked:
//!
//! * the server's prompt for the same message (image then text) is the
//!   reference's, id for id, and its tile count;
//! * the server's greedy reply is the reference's, token for token, up to the
//!   first step whose top-two margin is within bf16 noise of a tie; the test
//!   prints where that is.

use std::path::PathBuf;

use base64::Engine as _;
use pmetal_models::DynamicModel;
use pmetal_serve::InferenceEngine;
use pmetal_serve::engine::SamplingParams;
use serde_json::{Value, json};

/// bf16 logits of the size these reach are good to about this much.
const TIE: f64 = 0.125;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn serve_replies_to_an_image_as_transformers_does() {
    let (Some(dir), Some(reference_dir)) = (
        std::env::var_os("PMETAL_MLLAMA_CHECKPOINT").map(PathBuf::from),
        std::env::var_os("PMETAL_MLLAMA_SERVE_REFERENCE").map(PathBuf::from),
    ) else {
        eprintln!("PMETAL_MLLAMA_CHECKPOINT / PMETAL_MLLAMA_SERVE_REFERENCE not set; skipping");
        return;
    };
    let reference: Value = serde_json::from_str(
        &std::fs::read_to_string(reference_dir.join("reference.json")).unwrap(),
    )
    .unwrap();
    let ids = |key: &str| -> Vec<u32> {
        reference[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect()
    };
    let png = std::fs::read(reference_dir.join("shapes.png")).unwrap();
    let data_uri = format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(png)
    );

    let load_dir = dir.clone();
    let engine = InferenceEngine::new_with_backend(
        move || Ok(DynamicModel::load(&load_dir)?),
        pmetal_data::Tokenizer::from_model_dir(&dir).unwrap(),
        "llama-3.2-vision".into(),
        &dir,
        4096,
        false,
        4096,
    )
    .unwrap();
    assert!(engine.accepts_media());
    let messages: Vec<pmetal_serve::types::ChatMessage> = serde_json::from_value(json!([
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_uri}},
            {"type": "text", "text": reference["question"]}
        ]}
    ]))
    .unwrap();
    let prompt = engine.prepare_chat(&messages, None).await.unwrap();
    assert_eq!(
        prompt.input_ids,
        ids("input_ids"),
        "the server's prompt is the reference's"
    );

    let want = ids("tokens");
    let (got, ..) = engine
        .generate_prompt(
            &prompt,
            SamplingParams {
                max_tokens: want.len(),
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
            },
        )
        .await
        .unwrap();
    pmetal_bridge::check_last_error().unwrap();
    let tokenizer = pmetal_data::Tokenizer::from_model_dir(&dir).unwrap();
    println!("reference: {:?}", reference["reply"]);
    println!("server:    {:?}", tokenizer.decode(&got).unwrap());

    let margins: Vec<f64> = reference["margins"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let agree = got.iter().zip(&want).take_while(|(g, w)| g == w).count();
    // The engine stops at an end-of-turn token the reference also emits.
    let complete = got.len() == agree && (agree == want.len() || agree + 1 == want.len());
    let first_tie = margins.iter().position(|&m| m < TIE).unwrap_or(want.len());
    println!(
        "{agree} of {} reference tokens agree; the first near-tie is at step {first_tie}",
        want.len()
    );
    assert!(
        complete || agree >= first_tie,
        "the reply left the reference at step {agree}, before any near-tie ({first_tie}): \
         got {got:?}, want {want:?}"
    );
}
