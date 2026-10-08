//! Images and videos through the server, on the tiny seeded Qwen 3.5-family
//! vision model of `pmetal-models`' parity fixture (4 text layers, a 2-block
//! vision tower, patch 4, merge 2), with a word-level tokenizer and a chat
//! template that renders image and video items the way the released one does.
//!
//! Checked:
//!
//! * the prompt: the chat template places each image where the client put it
//!   among the text, and the processor expands it to its token run;
//! * the reply: greedy decoding with the KV and recurrent caches, after a
//!   prefill from merged embeddings, equals recomputing the whole sequence
//!   from scratch at every step with `get_rope_index` positions, so the
//!   cached decode continues at the right 3-D position;
//! * the same reply streamed, through `/v1/chat/completions` (both modes, with
//!   continuous batching on) and through `/v1/messages`;
//! * a video's frames, and two different images in the same text giving
//!   different replies;
//! * the 400s: remote URLs, file paths, a text-only model.
//!
//! Every engine-side forward drains the bridge's error channel.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use base64::Engine as _;
use pmetal_bridge::compat::{Array, Module};
use pmetal_data::qwen_vl_processing::{QwenVlProcessor, RgbImage, VideoFrames};
use pmetal_models::DynamicModel;
use pmetal_models::architectures::qwen3_5_vision::{
    Qwen3_5MultimodalConfig, Qwen3_5VisionModel, merge_media_features, rope_index,
};
use pmetal_serve::engine::{SamplingParams, TokenEvent};
use pmetal_serve::{BatcherConfig, InferenceEngine, PreparedPrompt};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

const MODEL_ID: &str = "tiny-qwen3_5-vl";
const STEPS: usize = 6;

/// The released template's handling of content items, in a few lines.
const TEMPLATE: &str = "{%- for m in messages %}<s_turn>{{ m.role }} \
{%- if m.content is string %} {{ m.content }}{% else %}\
{%- for item in m.content %}\
{%- if item.type == 'image' %}<|vision_start|><|image_pad|><|vision_end|>\
{%- elif item.type == 'video' %}<|vision_start|><|video_pad|><|vision_end|>\
{%- else %} {{ item.text }} {% endif %}\
{%- endfor %}{% endif %}<e_turn>{% endfor %}\
{%- if add_generation_prompt %}<s_turn>assistant {% endif %}";

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../pmetal-models/tests/fixtures/{name}"))
}

/// A checkpoint directory: the fixture's config and weights, processor
/// configs for its patch size, and the template.
fn model_dir() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    std::fs::copy(
        fixture("qwen3_5_vision_config.json"),
        dir.path().join("config.json"),
    )
    .unwrap();
    std::fs::copy(
        fixture("qwen3_5_vision_weights.safetensors"),
        dir.path().join("model.safetensors"),
    )
    .unwrap();
    let processor = |size: Value, extra: Value| {
        let mut config = json!({
            "size": size, "patch_size": 4, "temporal_patch_size": 2, "merge_size": 2,
            "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5]
        });
        config
            .as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        config.to_string()
    };
    std::fs::write(
        dir.path().join("preprocessor_config.json"),
        processor(
            json!({"shortest_edge": 64, "longest_edge": 4096}),
            json!({}),
        ),
    )
    .unwrap();
    std::fs::write(
        dir.path().join("video_preprocessor_config.json"),
        processor(
            json!({"shortest_edge": 64, "longest_edge": 8192}),
            json!({"fps": 2, "min_frames": 2, "max_frames": 8}),
        ),
    )
    .unwrap();
    std::fs::write(
        dir.path().join("tokenizer_config.json"),
        json!({"chat_template": TEMPLATE}).to_string(),
    )
    .unwrap();
    dir
}

/// Word-level, one id per fixture vocab entry; the media tokens at the ids
/// the fixture's config names.
fn tokenizer() -> pmetal_data::Tokenizer {
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;

    let named = [
        ("<unk>", 0),
        ("user", 1),
        ("assistant", 2),
        ("system", 3),
        ("describe", 4),
        ("this", 5),
        ("and", 6),
        ("compare", 7),
        ("<|vision_start|>", 58),
        ("<|vision_end|>", 59),
        ("<|image_pad|>", 62),
        ("<|video_pad|>", 63),
        ("<s_turn>", 64),
        ("<e_turn>", 65),
    ];
    let mut vocab: HashMap<String, u32> = (0..80).map(|i| (format!("t{i}"), i)).collect();
    for (word, id) in named {
        vocab.remove(&format!("t{id}"));
        vocab.insert(word.to_string(), id);
    }
    let model = WordLevel::builder()
        .vocab(vocab.into_iter().collect())
        .unk_token("<unk>".to_string())
        .build()
        .unwrap();
    let mut tokenizer = tokenizers::Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(Whitespace));
    let specials: Vec<tokenizers::AddedToken> = named
        .iter()
        .filter(|(word, _)| word.starts_with('<') && *word != "<unk>")
        .map(|(word, _)| tokenizers::AddedToken::from(word.to_string(), true))
        .collect();
    tokenizer.add_special_tokens(specials).unwrap();
    pmetal_data::Tokenizer::from_bytes(tokenizer.to_string(false).unwrap().as_bytes()).unwrap()
}

fn vision_engine(dir: &Path) -> InferenceEngine {
    let load_dir = dir.to_path_buf();
    InferenceEngine::new_with_backend(
        move || Ok(DynamicModel::load(&load_dir)?),
        tokenizer(),
        MODEL_ID.into(),
        dir,
        512,
        false,
        1024,
    )
    .unwrap()
}

/// A `width × height` image whose pixels depend on `seed`.
fn image(width: u32, height: u32, seed: u8) -> RgbImage {
    RgbImage::from_fn(width, height, |x, y| {
        image::Rgb([
            (x as u8).wrapping_mul(6).wrapping_add(seed),
            (y as u8).wrapping_mul(10),
            ((x + y) as u8).wrapping_mul(3).wrapping_sub(seed),
        ])
    })
}

fn data_uri(image: &RgbImage) -> String {
    let mut bytes = Vec::new();
    image::DynamicImage::ImageRgb8(image.clone())
        .write_to(
            &mut std::io::Cursor::new(&mut bytes),
            image::ImageFormat::Png,
        )
        .unwrap();
    format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(bytes)
    )
}

fn greedy(max_tokens: usize) -> SamplingParams {
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
        logprobs_top_n: None,
    }
}

fn drain(op: &str) {
    pmetal_bridge::check_last_error().unwrap_or_else(|e| panic!("{op}: bridge error: {e}"));
}

/// Greedy decoding with no cache at all: at every step the whole sequence,
/// media merged in, at the positions `get_rope_index` gives it.
fn recompute_greedy(
    dir: &Path,
    prompt: &[u32],
    images: &[RgbImage],
    videos: &[VideoFrames],
    steps: usize,
) -> Reference {
    let processor = QwenVlProcessor::from_model_dir(dir).unwrap();
    let config = Qwen3_5MultimodalConfig::from_model_dir(dir).unwrap();
    let images: Vec<_> = images
        .iter()
        .map(|i| processor.preprocess_image(i).unwrap())
        .collect();
    let videos: Vec<_> = videos
        .iter()
        .map(|v| processor.preprocess_video(v).unwrap())
        .collect();
    let tower = Qwen3_5VisionModel::load(dir).unwrap();
    let image_features = tower.encode(&images).unwrap();
    let video_features = tower.encode(&videos).unwrap();
    drain("vision tower");
    let grids = |media: &[pmetal_data::qwen_vl_processing::ProcessedMedia]| {
        media.iter().map(|m| m.grid_thw).collect::<Vec<_>>()
    };
    let mut model = DynamicModel::load(dir).unwrap();
    let qwen = model.as_qwen3_next_mut().unwrap();
    let mut ids = prompt.to_vec();
    let mut logprobs = Vec::with_capacity(steps);
    for _ in 0..steps {
        let positions = rope_index(
            &ids,
            &grids(&images),
            &grids(&videos),
            config.vision.spatial_merge_size,
            config.image_token_id,
            config.video_token_id,
        )
        .unwrap();
        let id_array = Array::from_i32_slice_shaped(
            &ids.iter().map(|&i| i as i32).collect::<Vec<_>>(),
            &[1, ids.len() as i32],
        );
        let text = Module::forward(&mut qwen.model.embed_tokens, &id_array).unwrap();
        let embeddings = merge_media_features(
            &text,
            &ids,
            image_features.as_ref(),
            video_features.as_ref(),
            config.image_token_id,
            config.video_token_id,
        )
        .unwrap();
        let (_, logits) = qwen
            .forward_embeddings(&embeddings, &positions.array(), None, None)
            .unwrap();
        let last = pmetal_bridge::compat::ops::slice_axis(
            &logits,
            1,
            ids.len() as i32 - 1,
            ids.len() as i32,
        );
        let mut last = last;
        last.eval();
        let row = last.to_f32_vec(last.size()).unwrap();
        drain("recompute");
        let (next, best) = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap();
        let log_sum_exp = row
            .iter()
            .map(|v| ((v - best) as f64).exp())
            .sum::<f64>()
            .ln();
        logprobs.push(-log_sum_exp as f32);
        ids.push(next as u32);
    }
    Reference {
        tokens: ids[prompt.len()..].to_vec(),
        logprobs,
    }
}

/// A greedy continuation and each token's log-probability.
struct Reference {
    tokens: Vec<u32>,
    logprobs: Vec<f32>,
}

fn chat_messages(content: Value) -> Vec<pmetal_serve::types::ChatMessage> {
    serde_json::from_value(json!([{"role": "user", "content": content}])).unwrap()
}

/// A streamed reply's tokens and log-probabilities.
async fn collect(mut rx: tokio::sync::mpsc::Receiver<TokenEvent>) -> (Vec<u32>, Vec<f32>) {
    let (mut tokens, mut logprobs) = (Vec::new(), Vec::new());
    while let Some(event) = rx.recv().await {
        match event {
            TokenEvent::Token { id, logprob } => {
                tokens.push(id);
                logprobs.push(logprob.expect("logprobs requested").logprob);
            }
            TokenEvent::Done { .. } => return (tokens, logprobs),
            TokenEvent::Error(e) => panic!("stream failed: {e}"),
        }
    }
    panic!("stream closed without Done");
}

/// Greedy, with each token's log-probability.
fn greedy_logprobs() -> SamplingParams {
    SamplingParams {
        logprobs_top_n: Some(0),
        ..greedy(STEPS)
    }
}

/// The engine's reply, which may stop early on one of the engine's stop
/// tokens, is a prefix of the recomputed one, token for token and with each
/// token's log-probability: a tiny random model often picks the same tokens
/// from slightly wrong logits, but not with the same probabilities.
fn assert_reply_matches(what: &str, (tokens, logprobs): (&[u32], &[f32]), want: &Reference) {
    assert!(!tokens.is_empty(), "{what}: empty reply");
    assert_eq!(tokens, &want.tokens[..tokens.len()], "{what}");
    for (step, (got, want)) in logprobs.iter().zip(&want.logprobs).enumerate() {
        assert!(
            (got - want).abs() < 1e-4,
            "{what}: token {step} has log-probability {got}, recomputed {want}"
        );
    }
}

/// [`InferenceEngine::generate_prompt`]'s tokens and log-probabilities.
async fn generate(engine: &InferenceEngine, prompt: &PreparedPrompt) -> (Vec<u32>, Vec<f32>) {
    let (tokens, logprobs, ..) = engine
        .generate_prompt(prompt, greedy_logprobs())
        .await
        .unwrap();
    let logprobs = logprobs.unwrap().iter().map(|e| e.logprob).collect();
    (tokens, logprobs)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn media_prompts_decode_like_an_uncached_recompute() {
    let dir = model_dir();
    let engine = vision_engine(dir.path());
    assert!(engine.accepts_media());

    // One image between two runs of text: the template puts its placeholder
    // there, and the 24×40 image becomes a 6×10 patch grid, 15 tokens.
    let picture = image(40, 24, 0);
    let messages = chat_messages(json!([
        {"type": "text", "text": "describe"},
        {"type": "image_url", "image_url": {"url": data_uri(&picture)}},
        {"type": "text", "text": "this"}
    ]));
    let prompt = engine.prepare_chat(&messages, None).await.unwrap();
    assert!(prompt.has_media());
    let expected = tokenizer()
        .encode(&format!(
            "<s_turn>user describe <|vision_start|>{}<|vision_end|> this <e_turn><s_turn>assistant ",
            "<|image_pad|>".repeat(15)
        ))
        .unwrap();
    assert_eq!(prompt.input_ids, expected, "prompt with its image expanded");

    let want = recompute_greedy(dir.path(), &prompt.input_ids, &[picture], &[], STEPS);
    let (_, _, _, metrics) = engine
        .generate_prompt(&prompt, greedy(STEPS))
        .await
        .unwrap();
    assert_eq!(metrics.prompt_tokens, expected.len());
    let got = generate(&engine, &prompt).await;
    assert_reply_matches("image, non-streaming", (&got.0, &got.1), &want);
    let streamed = collect(engine.generate_streaming_prompt(&prompt, greedy_logprobs())).await;
    assert_reply_matches("image, streaming", (&streamed.0, &streamed.1), &want);

    // The same prompt ids sent as text alone are a different prompt: nothing
    // about the image may come from a cache.
    let (_, text_only) = generate(
        &engine,
        &PreparedPrompt::from_tokens(prompt.input_ids.clone()),
    )
    .await;
    assert!(
        (text_only[0] - got.1[0]).abs() > 1e-3,
        "the image's features must reach the model"
    );

    // Another image of the same size, in the same place: same prompt ids,
    // its own reply.
    let other = image(40, 24, 97);
    let other_prompt = engine
        .prepare_chat(
            &chat_messages(json!([
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": data_uri(&other)}},
                {"type": "text", "text": "this"}
            ])),
            None,
        )
        .await
        .unwrap();
    assert_eq!(other_prompt.input_ids, prompt.input_ids);
    let want_other = recompute_greedy(dir.path(), &other_prompt.input_ids, &[other], &[], STEPS);
    let got_other = generate(&engine, &other_prompt).await;
    assert_reply_matches("second image", (&got_other.0, &got_other.1), &want_other);
    assert!(
        (got_other.1[0] - got.1[0]).abs() > 1e-3,
        "two images, one reply"
    );

    // A video, as two frames at 2 fps: one temporal group of 15 tokens
    // after its timestamp.
    let frames = [image(40, 24, 5), image(40, 24, 60)];
    let video_prompt = engine
        .prepare_chat(
            &chat_messages(json!([
                {"type": "video", "video": {"frames": frames.iter().map(data_uri).collect::<Vec<_>>(), "fps": 2}},
                {"type": "text", "text": "describe"}
            ])),
            None,
        )
        .await
        .unwrap();
    assert_eq!(
        video_prompt
            .input_ids
            .iter()
            .filter(|&&id| id == 63)
            .count(),
        15
    );
    let want_video = recompute_greedy(
        dir.path(),
        &video_prompt.input_ids,
        &[],
        &[VideoFrames {
            frames: frames.to_vec(),
            fps: Some(2.0),
        }],
        STEPS,
    );
    let got_video = generate(&engine, &video_prompt).await;
    assert_reply_matches("video", (&got_video.0, &got_video.1), &want_video);

    let (entries, _, hits, misses, _) = engine.prefix_cache_stats();
    assert_eq!(
        (entries, hits, misses),
        (0, 0, 0),
        "media prompts never touch the prefix cache"
    );
}

// ────────────────────────────────────────────────────────────────────────────
// Over HTTP
// ────────────────────────────────────────────────────────────────────────────

/// Serve `engine` on the first free port of the test range.
async fn serve(engine: InferenceEngine) -> u16 {
    for port in 18760..18800 {
        if let Ok(listener) = tokio::net::TcpListener::bind(("127.0.0.1", port)).await {
            let router = pmetal_serve::server::build_router(engine, 4);
            tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
            return port;
        }
    }
    panic!("no free port in 18760..18800");
}

/// POST `body` to `path`; the status and the (de-chunked) body.
async fn post(port: u16, path: &str, body: &Value) -> (u16, String) {
    let body = body.to_string();
    let mut stream = tokio::net::TcpStream::connect(("127.0.0.1", port))
        .await
        .unwrap();
    stream
        .write_all(
            format!(
                "POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n\
                 Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .as_bytes(),
        )
        .await
        .unwrap();
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).await.unwrap();
    let raw = String::from_utf8(raw).unwrap();
    let (head, body) = raw.split_once("\r\n\r\n").unwrap();
    let status = head.split(' ').nth(1).unwrap().parse().unwrap();
    let chunked = head
        .to_ascii_lowercase()
        .contains("transfer-encoding: chunked");
    (status, if chunked { dechunk(body) } else { body.into() })
}

fn dechunk(mut body: &str) -> String {
    let mut out = String::new();
    loop {
        let (size, rest) = body.split_once("\r\n").unwrap();
        let size = usize::from_str_radix(size.trim(), 16).unwrap();
        if size == 0 {
            return out;
        }
        out.push_str(&rest[..size]);
        body = &rest[size + 2..];
    }
}

/// The `data:` payloads of an SSE body.
fn sse_data(body: &str) -> Vec<String> {
    body.lines()
        .filter_map(|line| line.strip_prefix("data: ").or(line.strip_prefix("data:")))
        .map(str::to_owned)
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn images_over_http() {
    let dir = model_dir();
    let engine = vision_engine(dir.path());
    // Batching on: a media request must not go through the pump, which only
    // knows token ids.
    engine
        .enable_continuous_batching_auto(BatcherConfig {
            max_slots: 2,
            ..Default::default()
        })
        .unwrap();
    let tokenizer = Arc::new(tokenizer());
    let picture = image(40, 24, 0);
    let content = json!([
        {"type": "text", "text": "describe"},
        {"type": "image_url", "image_url": {"url": data_uri(&picture)}},
        {"type": "text", "text": "this"}
    ]);
    let prompt = engine
        .prepare_chat(&chat_messages(content.clone()), None)
        .await
        .unwrap();
    let want = recompute_greedy(dir.path(), &prompt.input_ids, &[picture], &[], STEPS);
    let port = serve(engine).await;

    let request = |stream: bool| {
        json!({
            "model": MODEL_ID, "max_tokens": STEPS, "temperature": 0, "stream": stream,
            "messages": [{"role": "user", "content": content}]
        })
    };
    let (status, body) = post(port, "/v1/chat/completions", &request(false)).await;
    assert_eq!(status, 200, "{body}");
    let reply: Value = serde_json::from_str(&body).unwrap();
    let text = reply["choices"][0]["message"]["content"]
        .as_str()
        .unwrap()
        .to_owned();
    let completion_tokens = reply["usage"]["completion_tokens"].as_u64().unwrap() as usize;
    assert_eq!(
        reply["usage"]["prompt_tokens"].as_u64().unwrap() as usize,
        prompt.input_ids.len()
    );
    assert!((1..=STEPS).contains(&completion_tokens), "{body}");
    assert_eq!(
        text,
        tokenizer.decode(&want.tokens[..completion_tokens]).unwrap(),
        "non-streaming reply"
    );

    let (status, body) = post(port, "/v1/chat/completions", &request(true)).await;
    assert_eq!(status, 200, "{body}");
    let events = sse_data(&body);
    assert_eq!(events.last().map(String::as_str), Some("[DONE]"), "{body}");
    let streamed: String = events
        .iter()
        .filter(|data| data.as_str() != "[DONE]")
        .filter_map(|data| {
            let chunk: Value = serde_json::from_str(data).unwrap();
            chunk["choices"][0]["delta"]["content"]
                .as_str()
                .map(str::to_owned)
        })
        .collect();
    assert_eq!(streamed, text, "streamed reply");

    // The same image as a /v1/messages image block.
    let payload = data_uri(&image(40, 24, 0))
        .split_once(',')
        .unwrap()
        .1
        .to_owned();
    let messages_request = |stream: bool| {
        json!({
            "model": MODEL_ID, "max_tokens": STEPS, "temperature": 0, "stream": stream,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "describe"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": payload}},
                {"type": "text", "text": "this"}
            ]}]
        })
    };
    let (status, body) = post(port, "/v1/messages", &messages_request(false)).await;
    assert_eq!(status, 200, "{body}");
    let reply: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(reply["content"][0]["text"].as_str().unwrap(), text);
    assert_eq!(
        reply["usage"]["input_tokens"].as_u64().unwrap() as usize,
        prompt.input_ids.len()
    );
    let (status, body) = post(port, "/v1/messages", &messages_request(true)).await;
    assert_eq!(status, 200, "{body}");
    let streamed: String = sse_data(&body)
        .iter()
        .filter_map(|data| {
            let event: Value = serde_json::from_str(data).ok()?;
            event["delta"]["text"].as_str().map(str::to_owned)
        })
        .collect();
    assert_eq!(streamed, text, "/v1/messages streamed");

    // A photo-sized body, past axum's 2 MiB default for `Json`: noise
    // compresses badly, so this PNG alone is over 3 MiB.
    let mut state = 0x9E37_79B9u32;
    let noise = RgbImage::from_fn(1100, 1000, |_, _| {
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state as u8
        };
        image::Rgb([next(), next(), next()])
    });
    let noise = data_uri(&noise);
    assert!(noise.len() > 4 * 1024 * 1024, "{}", noise.len());
    let (status, body) = post(
        port,
        "/v1/chat/completions",
        &json!({"model": MODEL_ID, "max_tokens": 2, "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": noise}}
        ]}]}),
    )
    .await;
    assert_eq!(status, 200, "{body}");

    // Refusals, each a 400 that says why.
    for (content, expected) in [
        (
            json!([{"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}}]),
            "remote image URLs are not fetched",
        ),
        (
            json!([{"type": "image_url", "image_url": {"url": dir.path().join("config.json").to_str().unwrap()}}]),
            "does not read file paths",
        ),
        (
            json!([{"type": "image_url", "image_url": {"url": "data:image/png;base64,bm90IGFuIGltYWdl"}}]),
            "could not decode image",
        ),
    ] {
        let (status, body) = post(
            port,
            "/v1/chat/completions",
            &json!({"model": MODEL_ID, "messages": [{"role": "user", "content": content}]}),
        )
        .await;
        assert_eq!(status, 400, "{body}");
        assert!(body.contains(expected), "{body}");
    }
    let (status, body) = post(
        port,
        "/v1/messages",
        &json!({"model": MODEL_ID, "max_tokens": 4, "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "url", "url": "https://example.com/cat.png"}}
        ]}]}),
    )
    .await;
    assert_eq!(status, 400, "{body}");
    assert!(body.contains("remote image URLs are not fetched"), "{body}");
    let (status, body) = post(
        port,
        "/v1/chat/completions",
        &json!({"model": MODEL_ID, "mm_processor_kwargs": {"fps": 1},
                "messages": [{"role": "user", "content": content}]}),
    )
    .await;
    assert_eq!(status, 400, "{body}");
    assert!(
        body.contains("mm_processor_kwargs is not supported"),
        "{body}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_text_only_model_refuses_images_by_name() {
    use pmetal_models::architectures::llama::{LlamaConfig, LlamaForCausalLM};

    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("config.json"),
        json!({"model_type": "llama"}).to_string(),
    )
    .unwrap();
    let engine = InferenceEngine::new_with_backend(
        || {
            Ok(DynamicModel::Llama(LlamaForCausalLM::new(LlamaConfig {
                vocab_size: 80,
                hidden_size: 32,
                intermediate_size: 64,
                num_hidden_layers: 1,
                num_attention_heads: 4,
                num_key_value_heads: Some(2),
                head_dim: Some(8),
                max_position_embeddings: 64,
                tie_word_embeddings: true,
                ..Default::default()
            })?))
        },
        tokenizer(),
        "tiny-llama".into(),
        dir.path(),
        64,
        false,
        1024,
    )
    .unwrap();
    assert!(!engine.accepts_media());
    let port = serve(engine).await;
    let png = data_uri(&image(8, 8, 1));
    let (status, body) = post(
        port,
        "/v1/chat/completions",
        &json!({"model": "tiny-llama", "messages": [{"role": "user", "content": [
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": {"url": png}}
        ]}]}),
    )
    .await;
    assert_eq!(status, 400, "{body}");
    assert!(
        body.contains(
            "model 'tiny-llama' does not accept images or videos: it is a text-only model"
        ),
        "{body}"
    );
    // Text in the list form still works.
    let (status, body) = post(
        port,
        "/v1/chat/completions",
        &json!({"model": "tiny-llama", "max_tokens": 2, "messages": [{"role": "user", "content": [
            {"type": "text", "text": "describe"}, {"type": "text", "text": " this"}
        ]}]}),
    )
    .await;
    assert_eq!(status, 200, "{body}");
}
