//! `POST /v1/completions` with `echo` and `logprobs`: the prompt's own
//! log-probabilities, as loglikelihood evaluation reads them.
//!
//! The oracle is the model itself, run over the prompt in one uncached
//! forward and reduced with a log-softmax on the host.

mod support;

use axum::http::StatusCode;
use pmetal_runtime::compat::Array;
use serde_json::{Value, json};
use support::*;

/// `ids` under the tiny model: for every position `i >= 1` the log-probability
/// of `ids[i]` given `ids[..i]`, and the full log-softmax row that predicted it.
fn oracle(ids: &[u32]) -> Vec<(f32, Vec<f32>)> {
    let mut model = tiny_llama().unwrap();
    let input = Array::from_slice(
        &ids.iter().map(|&t| t as i32).collect::<Vec<_>>(),
        &[1, ids.len() as i32],
    );
    let logits = model.forward(&input, None).unwrap();
    let logits = logits.as_dtype(pmetal_runtime::compat::Dtype::Float32.as_i32());
    logits.eval();
    let vocab = logits.dim(2) as usize;
    let flat = logits.as_slice::<f32>();
    (0..ids.len() - 1)
        .map(|i| {
            let row = &flat[i * vocab..(i + 1) * vocab];
            let max = row.iter().copied().fold(f32::MIN, f32::max);
            let lse = max + row.iter().map(|x| (x - max).exp()).sum::<f32>().ln();
            let log_probs: Vec<f32> = row.iter().map(|x| x - lse).collect();
            (log_probs[ids[i + 1] as usize], log_probs)
        })
        .collect()
}

fn completion(extra: Value) -> Value {
    let mut body = json!({"model": MODEL_ID, "temperature": 0});
    body.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    body
}

fn floats(v: &Value) -> Vec<Option<f64>> {
    v.as_array().unwrap().iter().map(Value::as_f64).collect()
}

const PROMPT: &str = "the thin hit";

#[tokio::test(flavor = "multi_thread")]
async fn an_echoed_prompt_comes_with_its_own_log_probabilities() {
    let (router, _dir) = app(256);
    let ids = tokenizer().encode(PROMPT).unwrap();
    assert!(ids.len() >= 6, "{ids:?}");
    let want = oracle(&ids);

    // Scoring alone: no output tokens.
    let reply = post(
        &router,
        "/v1/completions",
        &completion(json!({"prompt": PROMPT, "max_tokens": 0, "echo": true, "logprobs": 2})),
    )
    .await;
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.text());
    let reply = reply.json();
    let choice = &reply["choices"][0];
    assert_eq!(choice["text"], PROMPT);
    assert_eq!(choice["finish_reason"], "length");
    assert_eq!(reply["usage"]["completion_tokens"], 0);
    assert_eq!(reply["usage"]["prompt_tokens"], ids.len());

    let lp = &choice["logprobs"];
    let n = ids.len();
    assert_eq!(lp["tokens"].as_array().unwrap().len(), n);
    let token_logprobs = floats(&lp["token_logprobs"]);
    assert_eq!(token_logprobs.len(), n);
    assert_eq!(token_logprobs[0], None, "nothing predicts the first token");
    assert!(lp["top_logprobs"][0].is_null());
    for i in 1..n {
        let got = token_logprobs[i].unwrap() as f32;
        assert!(
            (got - want[i - 1].0).abs() < 2e-4,
            "token {i}: {got} against {}",
            want[i - 1].0
        );
        // The two likeliest tokens, by the oracle's own ranking.
        let mut ranked: Vec<(usize, f32)> = want[i - 1].1.iter().copied().enumerate().collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        let top = lp["top_logprobs"][i].as_object().unwrap();
        assert!(top.len() <= 2 && !top.is_empty(), "{top:?}");
        let best = top.values().map(|v| v.as_f64().unwrap()).fold(f64::MIN, f64::max);
        assert!(
            (best as f32 - ranked[0].1).abs() < 2e-4,
            "token {i}: best {best} against {}",
            ranked[0].1
        );
    }

    // The offsets walk the echoed text token by token.
    let tokens: Vec<&str> = lp["tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t.as_str().unwrap())
        .collect();
    let offsets: Vec<usize> = lp["text_offset"]
        .as_array()
        .unwrap()
        .iter()
        .map(|o| o.as_u64().unwrap() as usize)
        .collect();
    assert_eq!(offsets[0], 0);
    for i in 0..n - 1 {
        assert_eq!(offsets[i + 1], offsets[i] + tokens[i].len(), "offset {i}");
    }
    assert_eq!(tokens.concat(), PROMPT);
    assert_no_unobserved_error();
}

#[tokio::test(flavor = "multi_thread")]
async fn generated_tokens_follow_the_prompt_in_the_arrays() {
    let (router, _dir) = app(256);
    let ids = tokenizer().encode(PROMPT).unwrap();
    let plain = post(
        &router,
        "/v1/completions",
        &completion(json!({"prompt": PROMPT, "max_tokens": 3, "logprobs": 2})),
    )
    .await
    .json();
    let echoed = post(
        &router,
        "/v1/completions",
        &completion(json!({"prompt": PROMPT, "max_tokens": 3, "logprobs": 2, "echo": true})),
    )
    .await
    .json();
    let (plain_lp, echoed_lp) = (
        &plain["choices"][0]["logprobs"],
        &echoed["choices"][0]["logprobs"],
    );
    let n = ids.len();
    assert_eq!(echoed_lp["tokens"].as_array().unwrap().len(), n + 3);
    assert_eq!(
        echoed["choices"][0]["text"].as_str().unwrap(),
        format!("{PROMPT}{}", plain["choices"][0]["text"].as_str().unwrap())
    );
    // The generated part is what the same request without echo reports.
    assert_eq!(
        echoed_lp["tokens"].as_array().unwrap()[n..],
        plain_lp["tokens"].as_array().unwrap()[..]
    );
    let (a, b) = (
        floats(&echoed_lp["token_logprobs"]),
        floats(&plain_lp["token_logprobs"]),
    );
    for i in 0..3 {
        assert!((a[n + i].unwrap() - b[i].unwrap()).abs() < 1e-4, "token {i}");
    }
    // A greedy token is the likeliest, so it is in its own top map, which is
    // the `logprobs` likeliest tokens of the distribution, the chosen one
    // included (not its alternatives).
    for i in n..n + 3 {
        let top = echoed_lp["top_logprobs"][i].as_object().unwrap();
        assert_eq!(top.len(), 2, "{top:?}");
        let token = echoed_lp["tokens"][i].as_str().unwrap();
        let own = top.get(token).and_then(Value::as_f64).expect(token);
        assert!((own - a[i].unwrap()).abs() < 1e-5, "token {i}");
    }
    // Offsets keep counting through the generated text.
    let offsets: Vec<u64> = echoed_lp["text_offset"]
        .as_array()
        .unwrap()
        .iter()
        .map(|o| o.as_u64().unwrap())
        .collect();
    assert_eq!(offsets[n], PROMPT.len() as u64);
    // Echoing changed nothing about the tokens.
    assert_eq!(
        echoed["usage"]["completion_tokens"],
        plain["usage"]["completion_tokens"]
    );
}

/// What a loglikelihood harness sends: token-id prompts, one output token,
/// greedy, echo. Its `is_greedy` reads the top entry of each prompt position.
#[tokio::test(flavor = "multi_thread")]
async fn a_loglikelihood_request_scores_each_prompt_in_the_batch() {
    let (router, _dir) = app(256);
    let a = tokenizer().encode("the thin hit").unwrap();
    let b = tokenizer().encode("hi there").unwrap();
    let reply = post(
        &router,
        "/v1/completions",
        &completion(json!({"prompt": [a, b], "max_tokens": 1, "logprobs": 1, "echo": true})),
    )
    .await;
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.text());
    let reply = reply.json();
    let choices = reply["choices"].as_array().unwrap();
    assert_eq!(choices.len(), 2);
    for (choice, ids) in choices.iter().zip([&a, &b]) {
        let lp = &choice["logprobs"];
        assert_eq!(lp["tokens"].as_array().unwrap().len(), ids.len() + 1);
        let want = oracle(ids);
        let got = floats(&lp["token_logprobs"]);
        for i in 1..ids.len() {
            assert!((got[i].unwrap() as f32 - want[i - 1].0).abs() < 2e-4, "{i}");
            // is_greedy: the prompt token is the top-1 exactly when its
            // log-probability is the best the top map holds.
            let top = lp["top_logprobs"][i].as_object().unwrap();
            let best = top.values().map(|v| v.as_f64().unwrap()).fold(f64::MIN, f64::max);
            let greedy = want[i - 1]
                .1
                .iter()
                .copied()
                .enumerate()
                .max_by(|x, y| x.1.total_cmp(&y.1))
                .map(|(token, _)| token as u32 == ids[i])
                .unwrap();
            assert_eq!(
                (got[i].unwrap() - best).abs() < 1e-6,
                greedy,
                "is_greedy at {i}"
            );
        }
    }
    assert_eq!(
        reply["usage"]["prompt_tokens"].as_u64().unwrap() as usize,
        a.len() + b.len()
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_streamed_echo_carries_the_same_arrays() {
    let (router, _dir) = app(256);
    let body = completion(json!({"prompt": PROMPT, "max_tokens": 3, "logprobs": 1, "echo": true}));
    let whole = post(&router, "/v1/completions", &body).await.json();
    let mut streaming = body.clone();
    streaming["stream"] = json!(true);
    let reply = post(&router, "/v1/completions", &streaming).await;
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.text());
    let frames = reply.sse_json();
    let text: String = frames
        .iter()
        .filter_map(|f| f["choices"][0]["text"].as_str())
        .collect();
    assert_eq!(text, whole["choices"][0]["text"].as_str().unwrap());
    // The first frame is the prompt with its scores, ahead of any generated
    // token; the arrays of all the frames join into the whole reply's.
    assert_eq!(frames[0]["choices"][0]["text"], PROMPT);
    let mut tokens = Vec::new();
    let mut token_logprobs = Vec::new();
    let mut offsets = Vec::new();
    for frame in &frames {
        let lp = &frame["choices"][0]["logprobs"];
        if lp.is_null() {
            continue;
        }
        tokens.extend(lp["tokens"].as_array().unwrap().clone());
        token_logprobs.extend(lp["token_logprobs"].as_array().unwrap().clone());
        offsets.extend(lp["text_offset"].as_array().unwrap().clone());
    }
    let lp = &whole["choices"][0]["logprobs"];
    assert_eq!(Value::Array(tokens), lp["tokens"]);
    assert_eq!(Value::Array(offsets), lp["text_offset"]);
    for (got, want) in floats(&Value::Array(token_logprobs))
        .iter()
        .zip(floats(&lp["token_logprobs"]))
    {
        match (got, want) {
            (None, None) => {}
            (Some(g), Some(w)) => assert!((g - w).abs() < 1e-4),
            other => panic!("{other:?}"),
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn scoring_works_with_continuous_batching_on() {
    let tiny = tiny(256);
    tiny.engine
        .enable_continuous_batching_auto(pmetal_serve::BatcherConfig {
            max_slots: 2,
            ..Default::default()
        })
        .unwrap();
    let (router, _dir) = tiny.router(&pmetal_serve::ServeConfig::default());
    let body = completion(json!({"prompt": PROMPT, "max_tokens": 0, "logprobs": 1, "echo": true}));
    let reply = post(&router, "/v1/completions", &body).await;
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.text());
    let ids = tokenizer().encode(PROMPT).unwrap();
    let want = oracle(&ids);
    let got = floats(&reply.json()["choices"][0]["logprobs"]["token_logprobs"]);
    assert!((got[1].unwrap() as f32 - want[0].0).abs() < 2e-4);
    // And a request without echo still batches.
    let plain = post(
        &router,
        "/v1/completions",
        &completion(json!({"prompt": PROMPT, "max_tokens": 2})),
    )
    .await;
    assert_eq!(plain.status, StatusCode::OK);
}

#[tokio::test(flavor = "multi_thread")]
async fn what_scoring_cannot_do_is_refused() {
    let (router, _dir) = app(256);
    for (extra, param) in [
        // Nothing to return: no output and no scores.
        (json!({"max_tokens": 0}), "max_tokens"),
        (json!({"max_tokens": 0, "echo": true}), "max_tokens"),
        (json!({"max_tokens": 0, "logprobs": 1}), "max_tokens"),
        (json!({"echo": true, "logprobs": 6}), "logprobs"),
    ] {
        let mut body = completion(json!({"prompt": PROMPT}));
        body.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        let reply = post(&router, "/v1/completions", &body).await;
        assert_eq!(reply.status, StatusCode::BAD_REQUEST, "{extra}: {}", reply.text());
        assert_eq!(reply.json()["error"]["param"], param, "{extra}");
    }
}
