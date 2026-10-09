//! `pmetal serve` against `pmetal infer --image` on a real Qwen3.5-family
//! vision checkpoint. Opt-in: needs the checkpoint.
//!
//! ```text
//! PMETAL_QWEN3_5_VL_DIR=<snapshot> cargo test --release -p pmetal --features serve \
//!     --test serve_vision_real_weights -- --nocapture
//! ```
//!
//! The same image and question go to both: to `pmetal infer`'s runner as an
//! image file with `--chat`, and to the server's engine as a chat message
//! whose content holds the question's text and the image as a `data:` URI.
//! Checked, greedy:
//!
//! * the server's prompt is the runner's, id for id: the chat template placed
//!   the image where `infer` puts it and the processor expanded it the same;
//! * the server's reply is the runner's, token for token. The runner decodes
//!   on the native engine and the server on `DynamicModel`, so this holds
//!   the two engines' multimodal prefill and positioned decode to the same
//!   argmax at every step;
//! * the reply over HTTP, non-streaming and streamed, through
//!   `/v1/chat/completions` (continuous batching on) and `/v1/messages`, is
//!   that reply's text.

#![cfg(feature = "serve")]

mod common;

use std::path::{Path, PathBuf};

use base64::Engine as _;
use common::sse_data;
use pmetal::inference_runner::{InferenceRunner, InferenceRunnerConfig};
use pmetal_models::DynamicModel;
use pmetal_serve::engine::SamplingParams;
use pmetal_serve::{BatcherConfig, InferenceEngine, ServeConfig};
use serde_json::{Value, json};

const QUESTION: &str = "Describe this image in one sentence.";
const MAX_TOKENS: usize = 96;
const PORT: u16 = 18731;

fn model_dir() -> Option<PathBuf> {
    std::env::var_os("PMETAL_QWEN3_5_VL_DIR").map(PathBuf::from)
}

/// 640×480, white, a red circle on the left and a green triangle on the right.
fn picture() -> image::RgbImage {
    image::RgbImage::from_fn(640, 480, |x, y| {
        let (fx, fy) = (x as f32, y as f32);
        let in_circle = (fx - 200.0).powi(2) + (fy - 240.0).powi(2) <= 110.0f32.powi(2);
        // Apex (460, 140), base from (370, 340) to (550, 340).
        let in_triangle =
            (140.0..=340.0).contains(&fy) && (fx - 460.0).abs() <= (fy - 140.0) * 90.0 / 200.0;
        if in_circle {
            image::Rgb([220, 30, 30])
        } else if in_triangle {
            image::Rgb([30, 180, 60])
        } else {
            image::Rgb([255, 255, 255])
        }
    })
}

fn png_bytes(image: &image::RgbImage) -> Vec<u8> {
    let mut bytes = Vec::new();
    image::DynamicImage::ImageRgb8(image.clone())
        .write_to(
            &mut std::io::Cursor::new(&mut bytes),
            image::ImageFormat::Png,
        )
        .unwrap();
    bytes
}

fn drain(op: &str) {
    pmetal_bridge::check_last_error().unwrap_or_else(|e| panic!("{op}: bridge error: {e}"));
}

/// What `pmetal infer --chat --image <file> --temperature 0` generates.
fn infer(dir: &Path, image: &Path) -> (Vec<u32>, Vec<u32>) {
    let mut runner = InferenceRunner::prepare(InferenceRunnerConfig {
        model_path: dir.to_path_buf(),
        prompt: QUESTION.into(),
        chat: true,
        images: vec![image.to_path_buf()],
        temperature: Some(0.0),
        max_tokens: Some(MAX_TOKENS),
        ..Default::default()
    })
    .unwrap();
    let prompt = runner.state.input_ids().to_vec();
    let mut tokens = Vec::new();
    runner
        .state
        .generate_streaming(|token| {
            tokens.push(token);
            true
        })
        .unwrap();
    drain("infer");
    (prompt, tokens)
}

async fn post(path: &str, body: &Value) -> (u16, String) {
    common::post(PORT, path, body).await
}

#[test]
fn serve_replies_to_an_image_as_infer_does() {
    let Some(dir) = model_dir() else {
        eprintln!("PMETAL_QWEN3_5_VL_DIR is not set; skipping");
        return;
    };
    let scratch = tempfile::tempdir().unwrap();
    let image_path = scratch.path().join("shapes.png");
    let png = png_bytes(&picture());
    std::fs::write(&image_path, &png).unwrap();
    let data_uri = format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(&png)
    );

    let (infer_prompt, infer_tokens) = infer(&dir, &image_path);
    let tokenizer = pmetal_data::Tokenizer::from_model_dir(&dir).unwrap();
    println!(
        "infer: {} prompt tokens, reply {:?}",
        infer_prompt.len(),
        tokenizer.decode(&infer_tokens).unwrap()
    );

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
            "qwen3_5-vl".into(),
            &dir,
            4096,
            false,
            4096,
        )
        .unwrap();
        assert!(engine.accepts_media());
        let content = json!([
            {"type": "image_url", "image_url": {"url": data_uri}},
            {"type": "text", "text": QUESTION}
        ]);
        let messages: Vec<pmetal_serve::types::ChatMessage> =
            serde_json::from_value(json!([{"role": "user", "content": content}])).unwrap();
        let prompt = engine.prepare_chat(&messages, None, &Default::default()).await.unwrap();
        assert_eq!(prompt.input_ids, infer_prompt, "the server's prompt is infer's");
        let (tokens, ..) = engine
            .generate_prompt(
                &prompt,
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
                },
            )
            .await
            .unwrap();
        drain("serve engine");
        let text = tokenizer.decode(&tokens).unwrap();
        println!("serve: reply {text:?}");
        assert_eq!(tokens, infer_tokens, "the server's reply is infer's, token for token");
        // A chat reply's content starts at its first visible character.
        let text = text.trim_start().to_string();

        // Over HTTP, with batching on.
        engine
            .enable_continuous_batching_auto(BatcherConfig {
                max_slots: 2,
                ..Default::default()
            })
            .unwrap();
        tokio::spawn(pmetal_serve::server::run_server(
            engine,
            ServeConfig {
                port: PORT,
                ..Default::default()
            },
        ));
        common::wait_for(PORT).await;
        let chat = |stream: bool| {
            json!({
                "model": "qwen3_5-vl", "max_tokens": MAX_TOKENS, "temperature": 0,
                "stream": stream, "messages": [{"role": "user", "content": content}]
            })
        };
        let (status, body) = post("/v1/chat/completions", &chat(false)).await;
        assert_eq!(status, 200, "{body}");
        let reply: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(reply["choices"][0]["message"]["content"], json!(text));
        assert_eq!(reply["usage"]["prompt_tokens"], json!(infer_prompt.len()));
        let (status, body) = post("/v1/chat/completions", &chat(true)).await;
        assert_eq!(status, 200, "{body}");
        let streamed: String = sse_data(&body)
            .iter()
            .filter_map(|chunk| chunk["choices"][0]["delta"]["content"].as_str())
            .collect();
        assert_eq!(streamed, text, "streamed");

        let payload = data_uri.split_once(',').unwrap().1;
        let (status, body) = post(
            "/v1/messages",
            &json!({
                "model": "qwen3_5-vl", "max_tokens": MAX_TOKENS, "temperature": 0, "stream": true,
                "messages": [{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": payload}},
                    {"type": "text", "text": QUESTION}
                ]}]
            }),
        )
        .await;
        assert_eq!(status, 200, "{body}");
        let streamed: String = sse_data(&body)
            .iter()
            .filter_map(|event| event["delta"]["text"].as_str())
            .collect();
        assert_eq!(streamed, text, "/v1/messages streamed");
    });
}
