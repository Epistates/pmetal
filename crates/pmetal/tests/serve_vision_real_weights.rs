//! `pmetal serve` against `pmetal infer --image` / `--video` on a real
//! Qwen3.5-family vision checkpoint. Opt-in: needs the checkpoint.
//!
//! ```text
//! PMETAL_QWEN3_5_VL_DIR=<snapshot> cargo test --release -p pmetal --features serve \
//!     --test serve_vision_real_weights -- --nocapture
//! ```
//!
//! The same image (then the same video) and question go to both: to
//! `pmetal infer`'s runner as files with `--chat`, and to the server's engine
//! as a chat message whose content holds the media as `data:` URIs and the
//! question's text. Checked, greedy:
//!
//! * the server's prompt is the runner's, id for id: both render the media as
//!   items of the message through the model's chat template, and the
//!   processor expands each the same;
//! * the server's reply is the runner's, step for step, up to bf16 ties. The
//!   runner decodes on the native engine and the server on `DynamicModel`;
//!   both round the logits to bf16, so where the reference's top two are
//!   closer than a bf16 step (this image's first token: `" A"` over `" The"`
//!   by 0.066 in f32, an exact tie in transformers' own bf16) the engines
//!   may break the tie differently. Every step where infer's token is not
//!   the server's argmax must be within [`TIE`] of it in the server's own
//!   log-probabilities; the server then continues from infer's choice, so
//!   the rest of infer's reply is held to the same standard;
//! * the reply over HTTP, non-streaming and streamed, through
//!   `/v1/chat/completions` (continuous batching on) and `/v1/messages`, is
//!   the server engine's own reply text.

#![cfg(feature = "serve")]

mod common;

use std::path::{Path, PathBuf};

use base64::Engine as _;
use common::sse_data;
use pmetal::inference_runner::{InferenceRunner, InferenceRunnerConfig, VideoInput};
use pmetal_models::DynamicModel;
use pmetal_serve::engine::SamplingParams;
use pmetal_serve::{BatcherConfig, InferenceEngine, PreparedPrompt, ServeConfig};
use serde_json::{Value, json};

const QUESTION: &str = "Describe this image in one sentence.";
const VIDEO_QUESTION: &str = "Which way does the ball move?";
const MAX_TOKENS: usize = 96;
const PORT: u16 = 18631;
/// The video's frames and their rate.
const FRAMES: u32 = 8;
const FPS: f64 = 4.0;

/// How far apart, in the server's log-probabilities, infer's token may be
/// from the server's argmax: a little over two bf16 steps of a logit between
/// 16 and 32 (0.125 each), the rounding two engines can each be off by.
const TIE: f32 = 0.3;

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

/// Frame `frame` of a red ball crossing a sky-and-grass scene left to right.
fn ball_frame(frame: u32) -> image::RgbImage {
    let (width, height, radius) = (320u32, 240u32, 25.0f64);
    let cx = 30.0 + frame as f64 * (width as f64 - 60.0) / (FRAMES - 1) as f64;
    image::RgbImage::from_fn(width, height, |x, y| {
        let (dx, dy) = (x as f64 - cx, y as f64 - 150.0);
        if dx * dx + dy * dy <= radius * radius {
            image::Rgb([220, 20, 20])
        } else if y > 175 {
            image::Rgb([40, 160, 40])
        } else {
            image::Rgb([150, 200, 250])
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

fn data_uri(png: &[u8]) -> String {
    format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(png)
    )
}

fn drain(op: &str) {
    pmetal_bridge::check_last_error().unwrap_or_else(|e| panic!("{op}: bridge error: {e}"));
}

/// What `pmetal infer --chat --temperature 0` generates for `prompt` with
/// `images` and `videos`: the prompt's ids and the reply.
fn infer(dir: &Path, prompt: &str, images: Vec<PathBuf>, videos: Vec<VideoInput>) -> Reply {
    let mut runner = InferenceRunner::prepare(InferenceRunnerConfig {
        model_path: dir.to_path_buf(),
        prompt: prompt.into(),
        chat: true,
        images,
        videos,
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
    Reply { prompt, tokens }
}

struct Reply {
    prompt: Vec<u32>,
    tokens: Vec<u32>,
}

fn greedy(max_tokens: usize, logprobs_top_n: Option<usize>) -> SamplingParams {
    SamplingParams {
        max_tokens,
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
        logprobs_top_n,
    }
}

/// Hold infer's reply to the server's engine: the same prompt, and at every
/// step infer's token is the server's greedy choice or within [`TIE`] of it.
/// Returns the server's own greedy reply.
async fn assert_same_reply(
    engine: &InferenceEngine,
    content: &Value,
    infer: &Reply,
    what: &str,
) -> Vec<u32> {
    let messages: Vec<pmetal_serve::types::ChatMessage> =
        serde_json::from_value(json!([{"role": "user", "content": content}])).unwrap();
    let prompt = engine
        .prepare_chat(&messages, None, &Default::default())
        .await
        .unwrap();
    assert_eq!(
        prompt.input_ids, infer.prompt,
        "{what}: the server's prompt is infer's"
    );

    let mut own = None;
    let mut agreed = 0;
    let mut ties = Vec::new();
    while agreed < MAX_TOKENS {
        // The server continues from the steps already held, infer's.
        let mut forced: PreparedPrompt = prompt.clone();
        forced.input_ids.extend_from_slice(&infer.tokens[..agreed]);
        let (tokens, logprobs, ..) = engine
            .generate_prompt(&forced, greedy(MAX_TOKENS - agreed, Some(5)))
            .await
            .unwrap();
        drain(what);
        let logprobs = logprobs.expect("log-probabilities were asked for");
        assert_eq!(logprobs.len(), tokens.len(), "{what}: one entry per token");
        let own = own.get_or_insert_with(|| tokens.clone());
        let rest = &infer.tokens[agreed..];
        let same = tokens.iter().zip(rest).take_while(|(a, b)| a == b).count();
        if same == tokens.len() && same == rest.len() {
            break;
        }
        let step = agreed + same;
        assert!(
            same < tokens.len() && same < rest.len(),
            "{what}: at step {step} one engine stopped and the other went on\n\
             infer: {:?}\nserve: {own:?}",
            infer.tokens
        );
        let chosen = &logprobs[same];
        let want = rest[same];
        let gap = chosen
            .top_logprobs
            .iter()
            .find(|(token, _)| *token == want)
            .map(|&(_, logprob)| chosen.logprob - logprob);
        assert!(
            gap.is_some_and(|gap| gap <= TIE),
            "{what}: at step {step} the server picks {} and infer {want}, {gap:?} apart in the \
             server's log-probabilities: more than a bf16 tie\ninfer: {:?}\nserve: {own:?}",
            chosen.token,
            infer.tokens
        );
        ties.push((step, chosen.token, want, gap.unwrap()));
        agreed = step + 1;
    }
    println!(
        "{what}: {} tokens, broken differently at (step, serve, infer, gap) {ties:?}",
        infer.tokens.len()
    );
    own.take().unwrap()
}

#[test]
fn serve_replies_to_media_as_infer_does() {
    let Some(dir) = model_dir() else {
        eprintln!("PMETAL_QWEN3_5_VL_DIR is not set; skipping");
        return;
    };
    let scratch = tempfile::tempdir().unwrap();
    let tokenizer = pmetal_data::Tokenizer::from_model_dir(&dir).unwrap();

    let image_path = scratch.path().join("shapes.png");
    let png = png_bytes(&picture());
    std::fs::write(&image_path, &png).unwrap();
    let image_uri = data_uri(&png);
    let infer_image = infer(&dir, QUESTION, vec![image_path], vec![]);
    println!(
        "infer, image: {} prompt tokens, reply {:?}",
        infer_image.prompt.len(),
        tokenizer.decode(&infer_image.tokens).unwrap()
    );

    let frames_dir = scratch.path().join("video");
    std::fs::create_dir(&frames_dir).unwrap();
    let mut frame_uris = Vec::new();
    for frame in 0..FRAMES {
        let png = png_bytes(&ball_frame(frame));
        std::fs::write(frames_dir.join(format!("frame{}.png", frame + 1)), &png).unwrap();
        frame_uris.push(data_uri(&png));
    }
    let infer_video = infer(
        &dir,
        VIDEO_QUESTION,
        vec![],
        vec![VideoInput {
            frames_dir,
            fps: Some(FPS),
        }],
    );
    println!(
        "infer, video: {} prompt tokens, reply {:?}",
        infer_video.prompt.len(),
        tokenizer.decode(&infer_video.tokens).unwrap()
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
            {"type": "image_url", "image_url": {"url": image_uri}},
            {"type": "text", "text": QUESTION}
        ]);
        let tokens = assert_same_reply(&engine, &content, &infer_image, "image").await;
        let text = tokenizer.decode(&tokens).unwrap();
        println!("serve, image: reply {text:?}");
        // A chat reply's content starts at its first visible character.
        let text = text.trim_start().to_string();

        let video_content = json!([
            {"type": "video", "video": {"frames": frame_uris, "fps": FPS}},
            {"type": "text", "text": VIDEO_QUESTION}
        ]);
        let tokens = assert_same_reply(&engine, &video_content, &infer_video, "video").await;
        println!(
            "serve, video: reply {:?}",
            tokenizer.decode(&tokens).unwrap()
        );

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
        let (status, body) = common::post(PORT, "/v1/chat/completions", &chat(false)).await;
        assert_eq!(status, 200, "{body}");
        let reply: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(reply["choices"][0]["message"]["content"], json!(text));
        assert_eq!(
            reply["usage"]["prompt_tokens"],
            json!(infer_image.prompt.len())
        );
        let (status, body) = common::post(PORT, "/v1/chat/completions", &chat(true)).await;
        assert_eq!(status, 200, "{body}");
        let streamed: String = sse_data(&body)
            .iter()
            .filter_map(|chunk| chunk["choices"][0]["delta"]["content"].as_str())
            .collect();
        assert_eq!(streamed, text, "streamed");

        let payload = image_uri.split_once(',').unwrap().1;
        let (status, body) = common::post(
            PORT,
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
