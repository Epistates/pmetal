//! Continuous batching on a real checkpoint: what a request gets in a batch
//! against what it gets alone, and what batching buys in throughput.
//!
//! Opt-in: needs a small dense checkpoint.
//!
//! ```text
//! PMETAL_BATCH_CHECKPOINT=<model dir> \
//!     cargo test --release -p pmetal-serve --test batching_real_weights -- --nocapture
//! ```
//!
//! Four properties are checked, each against the reply the same request gets
//! on the single-request path:
//!
//! * decoding every request on its own cache (`per_slot_decode`) is **bit for
//!   bit** the single-request path's, however requests join and leave;
//! * a request that decodes **alone** in a fused pump is bit for bit the
//!   single-request path's too: a one-row fused forward is the serial one;
//! * the rows of a fused batch are **independent**: with the number of rows
//!   fixed, a row's logits do not change with what the other rows hold;
//! * a fused batch with company is within **bf16 rounding** of the
//!   single-request path: its matmuls are `[N, hidden]` products where a lone
//!   request's are vector products, which add in another order, so the last
//!   bit of a bf16 logit (about 0.1 at the size these reach) can differ and a
//!   greedy near-tie can fall the other way. (They differ between batch sizes
//!   too, since MLX picks another matmul kernel by the number of rows.)
//!
//! The throughput test and the reports are `#[ignore]`d; run them with
//! `--ignored` and nothing else on the GPU.

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

/// An engine over the checkpoint with batching on, decoding fused or, with
/// `per_slot`, one request at a time.
fn engine(dir: &PathBuf, max_slots: usize, per_slot: Option<bool>) -> Arc<InferenceEngine> {
    let load_dir = dir.clone();
    let engine = InferenceEngine::new_with_backend(
        move || Ok(DynamicModel::load(&load_dir)?),
        pmetal_data::Tokenizer::from_model_dir(dir).unwrap(),
        "batching-test".into(),
        dir,
        2048,
        false,
        4096,
    )
    .unwrap();
    // No request shares a prefix with another, so the prefix cache plays no
    // part.
    engine.set_prefix_cache_limits(0, 0);
    if let Some(per_slot_decode) = per_slot {
        engine
            .enable_continuous_batching_auto(BatcherConfig {
                max_slots,
                per_slot_decode,
                ..BatcherConfig::default()
            })
            .unwrap();
    }
    Arc::new(engine)
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

/// Prompts of very different lengths that share no prefix.
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

type Request = (Vec<u32>, SamplingParams);

/// Greedy requests with different budgets, so they leave while others go on
/// decoding.
fn greedy_requests(dir: &PathBuf) -> Vec<Request> {
    let budgets = [40usize, 24, 56, 16, 48, 32];
    prompts(dir)
        .into_iter()
        .zip(budgets)
        .map(|(prompt, n)| (prompt, params(n)))
        .collect()
}

/// Seeded requests with a sampling filter and, by turns, a repetition
/// penalty, frequency and presence penalties, and a logit bias.
fn sampled_requests(dir: &PathBuf) -> Vec<Request> {
    prompts(dir)
        .into_iter()
        .enumerate()
        .map(|(i, prompt)| {
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
            (prompt, p)
        })
        .collect()
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

/// Each request alone, on the single-request path, one after another.
async fn alone(engine: &InferenceEngine, requests: &[Request]) -> Vec<Reply> {
    let mut replies = Vec::new();
    for (prompt, p) in requests {
        let start = Instant::now();
        replies.push(collect(engine.generate_streaming(prompt, p.clone()), start).await);
    }
    replies
}

/// Run `requests` through the pump, the `i`-th submitted `i * gap` after the
/// first, and return their replies in order.
async fn batched(engine: &Arc<InferenceEngine>, requests: &[Request], gap: Duration) -> Vec<Reply> {
    let mut tasks = Vec::new();
    let begin = Instant::now();
    for (i, (prompt, p)) in requests.iter().enumerate() {
        tokio::time::sleep_until((begin + gap * i as u32).into()).await;
        let start = Instant::now();
        let rx = engine.generate_batched(prompt, p.clone()).unwrap();
        tasks.push(tokio::spawn(collect(rx, start)));
    }
    let mut replies = Vec::new();
    for task in tasks {
        replies.push(task.await.unwrap());
    }
    replies
}

/// The requests one at a time through the pump: each decodes without company.
async fn batched_one_by_one(engine: &Arc<InferenceEngine>, requests: &[Request]) -> Vec<Reply> {
    let mut replies = Vec::new();
    for request in requests {
        replies.extend(batched(engine, std::slice::from_ref(request), Duration::ZERO).await);
    }
    replies
}

/// First index where two replies' tokens differ (or the shorter length, when
/// one is a prefix of the other), and the worst difference in chosen-token
/// log-probability before it.
fn compare(a: &Reply, b: &Reply) -> (Option<usize>, f32) {
    let n = a.tokens.len().min(b.tokens.len());
    let first = (0..n)
        .find(|&i| a.tokens[i] != b.tokens[i])
        .or_else(|| (a.tokens.len() != b.tokens.len()).then_some(n));
    let upto = first.unwrap_or(n);
    let worst = (0..upto.min(a.logprobs.len()).min(b.logprobs.len()))
        .map(|i| (a.logprobs[i] - b.logprobs[i]).abs())
        .fold(0.0f32, f32::max);
    (first, worst)
}

/// The replies are the same token for token, with the same finish reason and
/// log-probabilities no further apart than `tolerance`.
fn assert_same(label: &str, got: &Reply, want: &Reply, tolerance: f32) {
    let (first, worst) = compare(got, want);
    assert!(
        first.is_none(),
        "{label}: the replies differ at token {first:?} (lengths {} and {})\n  got: {:?}\n want: {:?}",
        got.tokens.len(),
        want.tokens.len(),
        got.tokens,
        want.tokens,
    );
    assert_eq!(got.finish, want.finish, "{label}: finish reason");
    assert!(
        !worst.is_nan() && worst <= tolerance,
        "{label}: log-probabilities differ by {worst}"
    );
    println!(
        "{label}: {} tokens identical, worst log-probability difference {worst:.3e}",
        got.tokens.len()
    );
}

/// Decoding every request on its own cache gives exactly what the
/// single-request path gives, through requests joining and leaving a pump
/// with fewer slots than requests, a stop token, seeded sampling, penalties
/// and a logit bias.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn per_slot_decode_is_bit_identical_to_the_single_request_path() {
    let Some(dir) = checkpoint() else { return };
    let engine = engine(&dir, 3, Some(true));
    let mut requests = greedy_requests(&dir);
    requests.extend(sampled_requests(&dir));
    let mut want = alone(&engine, &requests).await;
    // A stop token taken from one of the replies, so a request ends on it.
    requests[2].1.extra_stop_token_ids = vec![want[2].tokens[10]];
    want[2] = alone(&engine, &requests[2..3]).await.remove(0);
    assert_eq!(want[2].finish, "stop");

    let got = batched(&engine, &requests, Duration::from_millis(30)).await;
    for (i, (got, want)) in got.iter().zip(&want).enumerate() {
        assert_same(&format!("request {i}"), got, want, 0.0);
    }
    pmetal_runtime::check_unobserved_error().unwrap();
}

/// A request that decodes alone in a fused pump gets exactly what the
/// single-request path gives it, seeded sampling, penalties and bias included.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_lone_request_in_a_fused_pump_matches_the_single_request_path() {
    let Some(dir) = checkpoint() else { return };
    let engine = engine(&dir, 8, Some(false));
    for (what, requests) in [
        ("greedy", greedy_requests(&dir)),
        ("sampled", sampled_requests(&dir)),
    ] {
        let want = alone(&engine, &requests).await;
        let got = batched_one_by_one(&engine, &requests).await;
        for (i, (got, want)) in got.iter().zip(&want).enumerate() {
            assert_same(&format!("{what} {i}"), got, want, 0.0);
        }
    }
    pmetal_runtime::check_unobserved_error().unwrap();
}

/// A fused batch with company is within bf16 rounding of the single-request
/// path: the same first token (the prefill is the same), log-probabilities
/// within the size of a bf16 step at these logits, and agreement token for
/// token until a greedy near-tie falls the other way, which the test reports.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fused_replies_track_the_single_request_path_within_bf16() {
    let Some(dir) = checkpoint() else { return };
    let engine = engine(&dir, 8, Some(false));
    let requests = greedy_requests(&dir);
    let want = alone(&engine, &requests).await;
    let got = batched(&engine, &requests, Duration::from_millis(30)).await;
    let mut agreeing = 0;
    for (i, (got, want)) in got.iter().zip(&want).enumerate() {
        let (first, worst) = compare(got, want);
        println!(
            "request {i}: {}, worst log-probability difference {worst:.3}",
            match first {
                None => format!("all {} tokens identical", got.tokens.len()),
                Some(at) => format!("tokens differ from {at} of {}", want.tokens.len()),
            }
        );
        assert_eq!(got.tokens[0], want.tokens[0], "request {i}: first token");
        assert!(
            worst <= 0.3,
            "request {i}: log-probabilities differ by {worst}"
        );
        agreeing += usize::from(first.is_none());
    }
    // Most replies of a few dozen tokens agree entirely.
    assert!(
        agreeing * 2 >= requests.len(),
        "only {agreeing} replies agree"
    );
    pmetal_runtime::check_unobserved_error().unwrap();
}

/// A row's logits do not depend on the contents of the other rows of its
/// batch: with the number of rows fixed, the first row's decode step is bit
/// for bit the same beside rows of any other lengths. (The attention kernel
/// MLX picks depends on how many keys the batch reaches, so a row beside one
/// of 1100 keys or more is computed another way; this test reports that
/// boundary rather than pinning it.)
#[test]
fn fused_rows_do_not_depend_on_the_contents_of_the_others() {
    use pmetal_cache::kv_cache::{FusedBatchKVCache, KVCache};
    use pmetal_runtime::compat::{Array, Dtype};

    let Some(dir) = checkpoint() else { return };
    let mut model = DynamicModel::load(&dir).unwrap();
    let base = prompts(&dir).remove(3);
    let prefill = |model: &mut DynamicModel, len: usize| -> KVCache {
        let mut tokens = Vec::new();
        while tokens.len() < len {
            tokens.extend_from_slice(&base);
        }
        tokens.truncate(len);
        let mut cache = model.create_cache(4096);
        let ids = Array::from_u32_slice(&tokens, &[1, len as i32]);
        model
            .forward_with_cache_last(&ids, None, Some(&mut cache))
            .unwrap()
            .eval();
        cache
    };
    let flat = |a: &Array| -> Vec<f32> {
        a.as_dtype(Dtype::Float32.as_i32())
            .as_slice::<f32>()
            .to_vec()
    };
    let target = prefill(&mut model, 40);
    let config = target.config().clone();
    let others: Vec<(usize, KVCache)> = [10usize, 40, 100, 400, 1100]
        .iter()
        .map(|&n| (n, prefill(&mut model, n)))
        .collect();

    // The first row's logits with rows of the given context lengths beside it.
    let mut first_row = |beside: &[usize]| -> Vec<f32> {
        let mut fused = FusedBatchKVCache::new(config.clone(), 16).unwrap();
        fused.admit(0).unwrap();
        fused.load_slot(0, &target).unwrap();
        for (i, n) in beside.iter().enumerate() {
            fused.admit(i + 1).unwrap();
            let (_, cache) = others.iter().find(|(len, _)| len == n).unwrap();
            fused.load_slot(i + 1, cache).unwrap();
        }
        let rows = 1 + beside.len();
        let ids = Array::from_i32_slice(&vec![1234; rows]).reshape(&[rows as i32, 1]);
        let active: Vec<usize> = (0..rows).collect();
        let out = flat(&model.forward_batched(&ids, &active, &mut fused).unwrap());
        out[..out.len() / rows].to_vec()
    };

    for rows in [2usize, 3, 4, 8] {
        let reference = first_row(&vec![40; rows - 1]);
        for lengths in [10usize, 100, 400] {
            let mut beside = vec![40; rows - 1];
            beside[rows - 2] = lengths;
            assert!(
                first_row(&beside) == reference,
                "with {rows} rows the first row differs beside {beside:?}"
            );
        }
        let reaching = first_row(&{
            let mut beside = vec![40; rows - 1];
            beside[0] = 1100;
            beside
        });
        println!(
            "{rows} rows, one reaching 1100 keys: the first row {}",
            if reaching == reference {
                "is unchanged"
            } else {
                "differs in the last bits (attention switches kernels at 1024 keys)"
            }
        );
    }
    pmetal_runtime::check_unobserved_error().unwrap();
}

/// Tokens per second and time to first token at 1, 4 and 16 concurrent
/// requests: one after another on the single-request path, through a pump
/// that decodes a request per forward, and through one that decodes a batch
/// per forward. Run alone on the machine.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "benchmark: run with --ignored on an idle machine"]
async fn throughput_by_concurrency() {
    let Some(dir) = checkpoint() else { return };
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
    let per_slot = engine(&dir, 16, Some(true));
    let fused = engine(&dir, 16, Some(false));

    // Warm up the kernels.
    for e in [&per_slot, &fused] {
        let warm = batched(e, &[(prompt_for(99), params(16))], Duration::ZERO).await;
        assert!(!warm[0].tokens.is_empty());
        let _ = batched(
            e,
            &(0..4)
                .map(|i| (prompt_for(90 + i), params(8)))
                .collect::<Vec<_>>(),
            Duration::ZERO,
        )
        .await;
    }

    println!("\nconcurrency | path        | tokens/s | mean TTFT ms | wall s");
    for concurrency in [1usize, 4, 16] {
        let requests: Vec<Request> = (0..concurrency)
            .map(|i| {
                let mut p = params(new_tokens);
                p.logprobs_top_n = None;
                (prompt_for(i), p)
            })
            .collect();

        // All at once on the single-request path: they run one after another
        // on the model thread.
        let begin = Instant::now();
        let mut handles = Vec::new();
        for (prompt, p) in &requests {
            let start = Instant::now();
            let rx = per_slot.generate_streaming(prompt, p.clone());
            handles.push(tokio::spawn(collect(rx, start)));
        }
        let mut replies = Vec::new();
        for h in handles {
            replies.push(h.await.unwrap());
        }
        report("single", concurrency, &replies, begin.elapsed());

        for (label, e) in [("per-slot", &per_slot), ("fused", &fused)] {
            let begin = Instant::now();
            let replies = batched(e, &requests, Duration::ZERO).await;
            report(label, concurrency, &replies, begin.elapsed());
        }
    }
    pmetal_runtime::check_unobserved_error().unwrap();
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

/// A lone row in the fused cache, step after step, against the serial
/// forward: logits bit for bit.
#[test]
fn a_lone_fused_row_matches_the_serial_forward_step_by_step() {
    use pmetal_cache::kv_cache::FusedBatchKVCache;
    use pmetal_runtime::compat::{Array, Dtype};

    let Some(dir) = checkpoint() else { return };
    let mut model = DynamicModel::load(&dir).unwrap();
    let prompt = prompts(&dir).remove(3);
    let mut serial = model.create_cache(2048);
    let ids = Array::from_u32_slice(&prompt, &[1, prompt.len() as i32]);
    model
        .forward_with_cache_last(&ids, None, Some(&mut serial))
        .unwrap()
        .eval();
    let mut fused = FusedBatchKVCache::new(serial.config().clone(), 8).unwrap();
    fused.admit(0).unwrap();
    fused.load_slot(0, &serial).unwrap();
    let flat = |a: &Array| -> Vec<f32> {
        a.as_dtype(Dtype::Float32.as_i32())
            .as_slice::<f32>()
            .to_vec()
    };
    for step in 0..40 {
        let token = Array::from_i32_slice(&[1000 + step * 7]).reshape(&[1, 1]);
        let want = flat(
            &model
                .forward_with_cache_last(&token, None, Some(&mut serial))
                .unwrap(),
        );
        let got = flat(&model.forward_batched(&token, &[0], &mut fused).unwrap());
        let worst = want
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(want == got, "step {step}: logits differ by {worst}");
    }
    pmetal_runtime::check_unobserved_error().unwrap();
}
