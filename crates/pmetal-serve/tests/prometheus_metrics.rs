//! `GET /metrics`: the Prometheus and OpenMetrics exposition.

mod support;

use axum::http::StatusCode;
use pmetal_serve::ServeConfig;
use serde_json::json;
use support::*;

const OPENMETRICS: &str = "application/openmetrics-text; version=1.0.0";

/// The value of the series whose line starts with `series` (a metric name,
/// with its labels when it has them), in the body of a scrape.
fn sample(body: &str, series: &str) -> Option<f64> {
    body.lines()
        .filter(|l| !l.starts_with('#'))
        .find_map(|line| {
            let rest = line.strip_prefix(series)?;
            // The series ends at a space, or the match was a longer name.
            rest.strip_prefix(' ')?.trim().parse().ok()
        })
}

async fn scrape(router: &axum::Router, accept: Option<&str>) -> Reply {
    match accept {
        Some(accept) => get_with_headers(router, "/metrics", &[("accept", accept)]).await,
        None => get(router, "/metrics").await,
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_scrape_reports_what_the_server_did() {
    let (router, _dir) = app(256);
    let reply = post(
        &router,
        "/v1/chat/completions",
        &json!({"model": MODEL_ID, "max_tokens": 5, "temperature": 0,
                "messages": [{"role": "user", "content": "hi"}]}),
    )
    .await;
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.text());
    let usage = reply.json()["usage"].clone();
    let (prompt, completion) = (
        usage["prompt_tokens"].as_f64().unwrap(),
        usage["completion_tokens"].as_f64().unwrap(),
    );
    assert_eq!(get(&router, "/no/such/route").await.status, 404);

    let scraped = scrape(&router, Some(OPENMETRICS)).await;
    assert_eq!(scraped.status, StatusCode::OK);
    assert!(
        scraped.headers["content-type"]
            .to_str()
            .unwrap()
            .starts_with("application/openmetrics-text; version=1.0.0"),
        "{:?}",
        scraped.headers
    );
    let body = scraped.text();
    assert!(body.ends_with("# EOF\n"), "{body}");

    // HTTP: by route pattern, and what no route matched.
    assert_eq!(
        sample(
            &body,
            r#"pmetal_http_requests_total{route="/v1/chat/completions",method="POST",status="200"}"#
        ),
        Some(1.0),
        "{body}"
    );
    assert_eq!(
        sample(
            &body,
            r#"pmetal_http_requests_total{route="unmatched",method="GET",status="404"}"#
        ),
        Some(1.0),
        "a path no route matched is one series, not one per path\n{body}"
    );
    assert!(!body.contains("/no/such/route"));
    assert_eq!(
        sample(
            &body,
            r#"pmetal_http_request_duration_seconds_count{route="/v1/chat/completions",method="POST"}"#
        ),
        Some(1.0)
    );

    // Generation: the tokens the reply's usage reported.
    assert_eq!(sample(&body, "pmetal_generation_requests_total"), Some(1.0));
    assert_eq!(
        sample(&body, r#"pmetal_tokens_total{kind="prompt"}"#),
        Some(prompt)
    );
    assert_eq!(
        sample(&body, r#"pmetal_tokens_total{kind="completion"}"#),
        Some(completion)
    );
    assert_eq!(
        sample(&body, "pmetal_time_to_first_token_seconds_count"),
        Some(1.0)
    );
    assert_eq!(
        sample(&body, "pmetal_generation_tokens_per_second_count"),
        Some(1.0)
    );
    assert!(sample(&body, "pmetal_generation_duration_seconds_sum").unwrap() > 0.0);

    // State: nothing is running between requests, the limit is the
    // configured one, and MLX holds the model's weights.
    assert_eq!(sample(&body, "pmetal_requests_running"), Some(0.0));
    assert_eq!(sample(&body, "pmetal_requests_queued"), Some(0.0));
    assert_eq!(
        sample(&body, "pmetal_request_capacity"),
        Some(ServeConfig::default().max_concurrent as f64)
    );
    assert!(sample(&body, "pmetal_memory_active_bytes").unwrap() > 0.0);
    assert!(sample(&body, "pmetal_prefix_cache_hits_total").is_some());
    assert!(sample(&body, "pmetal_prefix_cache_misses_total").is_some());
    assert!(sample(&body, "pmetal_speculative_drafted_tokens_total").is_some());
    assert!(
        body.contains(&format!(
            "pmetal_build_info{{version=\"{}\",model=\"{MODEL_ID}\"}} 1",
            env!("CARGO_PKG_VERSION")
        )),
        "{body}"
    );
    assert_no_unobserved_error();
}

#[tokio::test(flavor = "multi_thread")]
async fn without_openmetrics_in_accept_the_classic_format_is_served() {
    let (router, _dir) = app(256);
    // A family with no series yet is not exposed; give one a series.
    assert_eq!(get(&router, "/v1/models").await.status, StatusCode::OK);
    for accept in [None, Some("text/plain"), Some("*/*")] {
        let scraped = scrape(&router, accept).await;
        assert_eq!(scraped.status, StatusCode::OK);
        assert_eq!(
            scraped.headers["content-type"].to_str().unwrap(),
            "text/plain; version=0.0.4; charset=utf-8",
            "Accept: {accept:?}"
        );
        let body = scraped.text();
        assert!(!body.contains("# EOF"), "{body}");
        assert!(
            body.contains("# TYPE pmetal_http_requests_total counter"),
            "a counter family is named by its series: {body}"
        );
        assert!(!body.contains("# TYPE pmetal_http_requests counter"));
        // Every line is a comment or a sample the old parser reads.
        for line in body.lines().filter(|l| !l.starts_with('#')) {
            let (_, value) = line.rsplit_once(' ').expect(line);
            value.parse::<f64>().unwrap_or_else(|_| panic!("{line}"));
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn the_prefix_cache_shows_in_the_scrape() {
    let (router, _dir) = app(512);
    let ask = || async {
        let reply = post(
            &router,
            "/v1/chat/completions",
            &json!({"model": MODEL_ID, "max_tokens": 2, "temperature": 0,
                    "messages": [{"role": "user", "content": "a prompt long enough to cache, twice"}]}),
        )
        .await;
        assert_eq!(reply.status, StatusCode::OK, "{}", reply.text());
    };
    ask().await;
    ask().await;
    let body = scrape(&router, Some(OPENMETRICS)).await.text();
    assert!(
        sample(&body, "pmetal_prefix_cache_entries").unwrap() >= 1.0,
        "{body}"
    );
    assert!(sample(&body, "pmetal_prefix_cache_bytes").unwrap() > 0.0);
    assert!(sample(&body, "pmetal_prefix_cache_hits_total").unwrap() >= 1.0);
    assert!(sample(&body, r#"pmetal_tokens_total{kind="cached"}"#).unwrap() > 0.0);
}

#[tokio::test(flavor = "multi_thread")]
async fn metrics_need_the_api_key_like_every_route_but_health() {
    let keyed = ServeConfig {
        api_keys: vec!["sk-test".into()],
        ..ServeConfig::default()
    };
    let (router, _dir) = tiny(256).router(&keyed);
    let refused = get(&router, "/metrics").await;
    assert_eq!(refused.status, StatusCode::UNAUTHORIZED);
    let allowed = get_with_headers(
        &router,
        "/metrics",
        &[("authorization", "Bearer sk-test")],
    )
    .await;
    assert_eq!(allowed.status, StatusCode::OK);
    // The refusal was counted, under the route it was aimed at.
    let body = allowed.text();
    assert_eq!(
        sample(
            &get_with_headers(&router, "/metrics", &[("x-api-key", "sk-test")])
                .await
                .text(),
            r#"pmetal_http_requests_total{route="/metrics",method="GET",status="401"}"#
        ),
        Some(1.0),
        "{body}"
    );
    // /health stays open.
    assert_eq!(get(&router, "/health").await.status, StatusCode::OK);
}

#[tokio::test(flavor = "multi_thread")]
async fn the_json_metrics_stay() {
    let (router, _dir) = app(256);
    let reply = get(&router, "/v1/metrics").await;
    assert_eq!(reply.status, StatusCode::OK);
    let json = reply.json();
    for key in [
        "total_requests",
        "avg_first_token_latency_ms",
        "rolling_tokens_per_second",
        "speculative_drafted_tokens",
        "speculative_accepted_tokens",
        "speculative_acceptance_rate",
    ] {
        assert!(json.get(key).is_some(), "{key} in {json}");
    }
}
