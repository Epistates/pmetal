//! Continuous batching on a real checkpoint: every request served in a batch
//! gets what it gets alone, and what batching buys in throughput.
//!
//! Opt-in: needs a small checkpoint.
//!
//! ```text
//! PMETAL_BATCH_CHECKPOINT=<model dir> \
//!     cargo test --release -p pmetal-serve --test batching_real_weights -- --nocapture
//! ```
//!
//! The exactness tests run each request alone on the single-request path,
//! then all of them together through the batching pump, with different
//! prompt lengths, token budgets and stop tokens, joining and leaving the
//! batch while others decode, and compare the replies token for token (and
//! the chosen tokens' log-probabilities to a tolerance). The throughput test
//! is `#[ignore]`d; run it with `--ignored` and nothing else on the GPU.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use pmetal_models::DynamicModel;
use pmetal_serve::InferenceEngine;
use pmetal_serve::continuous_batch::BatcherConfig;
use pmetal_serve::engine::{SamplingParams, TokenEvent};
use tokio::sync::mpsc::Receiver;

fn checkpoint() -> Option<PathBuf> {
    let dir = std::env::var_os("PMETAL_BATCH_CHECKPOINT").map(PathBuf::from);
    if dir.is_none() {
        eprintln!("PMETAL_BATCH_CHECKPOINT not set; skipping");
    }
    dir
}

fn load_engine(dir: &PathBuf, max_seq_len: usize) -> InferenceEngine {
    let load_dir = dir.clone();
    InferenceEngine::new_with_backend(
        move || Ok(DynamicModel::load(&load_dir)?),
        pmetal_data::Tokenizer::from_model_dir(dir).unwrap(),
        "batching-test".into(),
        dir,
        max_seq_len,
        false,
        4096,
    )
    .unwrap()
}

fn params(max_tokens: usize) -> SamplingParams {
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
        logprobs_top_n: Some(0),
        logit_bias: Vec::new(),
    }
}

/// Prompts of very different lengths that share no prefix, so the prefix
/// cache plays no part.
fn prompts(dir: &PathBuf) -> Vec<Vec<u32>> {
    let tokenizer = pmetal_data::Tokenizer::from_model_dir(dir).unwrap();
    let texts = [
        "The capital of France is",
        "Write a short poem about the sea at night, with waves and stars and a lonely lighthouse.",
        "1, 1, 2, 3, 5, 8, 13, 21,",
        "Explain in simple terms why the sky is blue during the day but turns red at sunset, \
         and mention what happens to the light as it passes through the atmosphere on its way \
         to the eye of an observer standing on the ground.",
        "def fibonacci(n):",
        "Once upon a time, in a village at the edge of a great forest, there lived",
    ];
    texts.iter().map(|t| tokenizer.encode(t).unwrap()).collect()
}

struct Reply {
    tokens: Vec<u32>,
    logprobs: Vec<f32>,
    first_token: Option<Duration>,
    finish: String,
}

async fn collect(mut rx: Receiver<TokenEvent>, start: Instant) -> Reply {
    let mut reply = Reply {
        tokens: Vec::new(),
        logprobs: Vec::new(),
        first_token: None,
        finish: String::new(),
    };
    while let Some(event) = rx.recv().await {
        match event {
            TokenEvent::Token { id, logprob } => {
                reply.first_token.get_or_insert_with(|| start.elapsed());
                reply.tokens.push(id);
                if let Some(entry) = logprob {
                    reply.logprobs.push(entry.logprob);
                }
            }
            TokenEvent::Done { finish_reason, .. } => {
                reply.finish = finish_reason;
                return reply;
            }
            TokenEvent::Error(e) => panic!("stream failed: {e}"),
        }
    }
    panic!("stream closed without Done");
}

async fn alone(engine: &InferenceEngine, prompt: &[u32], params: SamplingParams) -> Reply {
    let start = Instant::now();
    let rx = engine.generate_streaming(prompt, params);
    collect(rx, start).await
}

/// Compare `got` with `want`: the same tokens, and log-probabilities that
/// agree to `tolerance` (batched matmuls do not reduce in the order a
/// one-row matmul does, so the last bits of a bf16 logit may differ).
fn assert_same(label: &str, got: &Reply, want: &Reply, tolerance: f32) {
    let first_difference = got
        .tokens
        .iter()
        .zip(&want.tokens)
        .position(|(g, w)| g != w);
    assert!(
        first_difference.is_none() && got.tokens.len() == want.tokens.len(),
        "{label}: batched and alone differ at token {first_difference:?} \
         (lengths {} and {})\n batched: {:?}\n   alone: {:?}",
        got.tokens.len(),
        want.tokens.len(),
        got.tokens,
        want.tokens,
    );
    assert_eq!(got.finish, want.finish, "{label}: finish reason");
    let worst = got
        .logprobs
        .iter()
        .zip(&want.logprobs)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0f32, f32::max);
    assert!(
        !worst.is_nan() && worst <= tolerance,
        "{label}: log-probabilities differ by {worst}"
    );
    println!(
        "{label}: {} tokens identical, worst logprob delta {worst:.2e}",
        got.tokens.len()
    );
}

/// Run `requests` through the batching pump, the `i`-th submitted `i * gap`
/// after the first, and return their replies in order.
async fn batched(
    engine: &Arc<InferenceEngine>,
    requests: &[(Vec<u32>, SamplingParams)],
    gap: Duration,
) -> Vec<Reply> {
    let mut tasks = Vec::new();
    let begin = Instant::now();
    for (i, (prompt, params)) in requests.iter().enumerate() {
        tokio::time::sleep_until((begin + gap * i as u32).into()).await;
        let start = Instant::now();
        let rx = engine.generate_batched(prompt, params.clone()).unwrap();
        tasks.push(tokio::spawn(collect(rx, start)));
    }
    let mut replies = Vec::new();
    for task in tasks {
        replies.push(task.await.unwrap());
    }
    replies
}

fn enable_batching(engine: &InferenceEngine, max_slots: usize) {
    engine
        .enable_continuous_batching_auto(BatcherConfig {
            max_slots,
            ..BatcherConfig::default()
        })
        .unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn greedy_replies_match_single_requests_through_joins_and_leaves() {
    let Some(dir) = checkpoint() else { return };
    let engine = Arc::new(load_engine(&dir, 2048));
    engine.set_prefix_cache_limits(0, 0);
    let prompts = prompts(&dir);
    // Different budgets, so requests leave while others go on decoding, and a
    // stop token one of them is sure to meet.
    let budgets = [40usize, 24, 56, 16, 48, 32];
    let requests: Vec<(Vec<u32>, SamplingParams)> = prompts
        .iter()
        .zip(budgets)
        .map(|(prompt, n)| (prompt.clone(), params(n)))
        .collect();

    let mut expected = Vec::new();
    for (prompt, params) in &requests {
        expected.push(alone(&engine, prompt, params.clone()).await);
    }
    // A stop token taken from one of the replies, so a request ends on it.
    let stop = expected[2].tokens[10];
    let mut requests = requests;
    requests[2].1.extra_stop_token_ids = vec![stop];
    let want_stopped = alone(&engine, &requests[2].0, requests[2].1.clone()).await;
    assert_eq!(want_stopped.finish, "stop");

    enable_batching(&engine, 4);
    let got = batched(&engine, &requests, Duration::from_millis(40)).await;
    for (i, got) in got.iter().enumerate() {
        let want = if i == 2 { &want_stopped } else { &expected[i] };
        assert_same(&format!("request {i}"), got, want, 0.05);
    }
    pmetal_runtime::check_unobserved_error().unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn seeded_sampling_penalties_and_bias_match_single_requests() {
    let Some(dir) = checkpoint() else { return };
    let engine = Arc::new(load_engine(&dir, 2048));
    engine.set_prefix_cache_limits(0, 0);
    let prompts = prompts(&dir);
    let mut requests: Vec<(Vec<u32>, SamplingParams)> = Vec::new();
    for (i, prompt) in prompts.iter().enumerate() {
        let mut p = params(40);
        p.temperature = 0.8;
        p.top_p = Some(0.9);
        p.top_k = Some(40);
        p.seed = Some(100 + i as u64);
        match i % 3 {
            0 => p.repetition_penalty = Some(1.2),
            1 => {
                p.frequency_penalty = Some(0.4);
                p.presence_penalty = Some(0.3);
            }
            _ => p.logit_bias = vec![(prompt[0], -5.0), (prompt[prompt.len() - 1], 2.0)],
        }
        requests.push((prompt.clone(), p));
    }
    let mut expected = Vec::new();
    for (prompt, params) in &requests {
        expected.push(alone(&engine, prompt, params.clone()).await);
    }
    enable_batching(&engine, 3);
    let got = batched(&engine, &requests, Duration::from_millis(25)).await;
    for (i, (got, want)) in got.iter().zip(&expected).enumerate() {
        assert_same(&format!("request {i}"), got, want, 0.05);
    }
    pmetal_runtime::check_unobserved_error().unwrap();
}

/// Tokens per second and time to first token at 1, 4 and 16 concurrent
/// requests, through the pump and, for reference, one after another on the
/// single-request path. Run alone on the machine.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "benchmark: run with --ignored on an idle machine"]
async fn throughput_by_concurrency() {
    let Some(dir) = checkpoint() else { return };
    let engine = Arc::new(load_engine(&dir, 2048));
    engine.set_prefix_cache_limits(0, 0);
    let tokenizer = pmetal_data::Tokenizer::from_model_dir(&dir).unwrap();
    let new_tokens = 128usize;
    // Distinct prompts of about 100 tokens each.
    let prompt_for = |i: usize| -> Vec<u32> {
        let text = format!(
            "Request {i}. Summarize the history of the number {} and its role in mathematics, \
             music, and architecture, giving three examples from different centuries and \
             explaining why each one mattered to the people who used it at the time. \
             Begin your answer with the word number and keep it brief.",
            i * 7 + 3
        );
        tokenizer.encode(&text).unwrap()
    };

    // Warm up the kernels.
    let warm = alone(&engine, &prompt_for(99), params(16)).await;
    assert!(!warm.tokens.is_empty());

    println!("\nconcurrency | path        | tokens/s | mean TTFT ms | wall s");
    for concurrency in [1usize, 4, 16] {
        let requests: Vec<(Vec<u32>, SamplingParams)> = (0..concurrency)
            .map(|i| {
                let mut p = params(new_tokens);
                p.logprobs_top_n = None;
                (prompt_for(i), p)
            })
            .collect();

        // One after another on the single-request path.
        let begin = Instant::now();
        let mut replies = Vec::new();
        let mut handles = Vec::new();
        for (prompt, p) in &requests {
            let start = Instant::now();
            let rx = engine.generate_streaming(prompt, p.clone());
            handles.push(tokio::spawn(collect(rx, start)));
        }
        for h in handles {
            replies.push(h.await.unwrap());
        }
        report("single", concurrency, &replies, begin.elapsed());

        enable_batching_once(&engine, 16);
        let begin = Instant::now();
        let replies = batched(&engine, &requests, Duration::ZERO).await;
        report("batched", concurrency, &replies, begin.elapsed());
    }
    pmetal_runtime::check_unobserved_error().unwrap();
}

fn enable_batching_once(engine: &InferenceEngine, max_slots: usize) {
    if !engine.continuous_batching_enabled() {
        enable_batching(engine, max_slots);
    }
}

fn report(path: &str, concurrency: usize, replies: &[Reply], wall: Duration) {
    let tokens: usize = replies.iter().map(|r| r.tokens.len()).sum();
    let ttft: f64 = replies
        .iter()
        .filter_map(|r| r.first_token)
        .map(|d| d.as_secs_f64() * 1000.0)
        .sum::<f64>()
        / replies.len() as f64;
    println!(
        "{concurrency:>11} | {path:<11} | {:>8.1} | {ttft:>12.0} | {:>6.2}",
        tokens as f64 / wall.as_secs_f64(),
        wall.as_secs_f64()
    );
}
