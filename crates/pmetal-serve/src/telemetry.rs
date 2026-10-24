//! Prometheus and OpenMetrics instrumentation, served at `GET /metrics`.
//!
//! The registry is an [`prometheus_client`] one, local to the server (no
//! process-wide recorder), so two servers in one process, and the tests, keep
//! separate numbers. Three kinds of series live in it:
//!
//! * HTTP, from the `track_http` middleware: requests by route, method and status, a
//!   latency histogram per route (for a streamed reply, the time to its
//!   headers), and the requests in flight. Routes are the router's patterns,
//!   never the raw path, so a client cannot mint series.
//! * Generation, from [`Telemetry::record_request`], once per finished
//!   generation: prompt, cached-prompt and completion tokens, time to first
//!   token, duration, decode speed, and the speculative decoder's drafted and
//!   accepted tokens (the acceptance rate is `accepted / drafted`).
//! * State, read off the engine at each scrape ([`Telemetry::render`]): the
//!   requests running and waiting, the batching slots, the prefix cache, and
//!   the GPU memory MLX holds.
//!
//! Names follow the Prometheus conventions: a `pmetal_` prefix, base units
//! (seconds, bytes), `_total` on counters.
//!
//! # Formats
//!
//! The registry encodes OpenMetrics. A scraper that accepts
//! `application/openmetrics-text` gets that; every other client gets the
//! classic text exposition format 0.0.4, which is the OpenMetrics text with
//! its counter families named by their `_total` series and without the
//! `# EOF` terminator ([`classic_text`]).

use std::sync::Mutex;
use std::sync::atomic::Ordering;
use std::time::Instant;

use axum::extract::{MatchedPath, Request, State};
use axum::http::{HeaderMap, HeaderValue, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::encoding::text::encode;
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::metrics::histogram::{Histogram, exponential_buckets};
use prometheus_client::registry::Registry;
use std::sync::Arc;

use crate::engine::{InferenceEngine, RequestMetrics};
use crate::routes::AppState;

/// `Content-Type` of an OpenMetrics 1.0 text body.
pub const OPENMETRICS_CONTENT_TYPE: &str =
    "application/openmetrics-text; version=1.0.0; charset=utf-8";

/// `Content-Type` of a classic Prometheus text body, exposition format 0.0.4.
pub const CLASSIC_CONTENT_TYPE: &str = "text/plain; version=0.0.4; charset=utf-8";

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct HttpLabels {
    route: String,
    method: String,
    status: String,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct RouteLabels {
    route: String,
    method: String,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct BuildLabels {
    version: String,
    model: String,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct TokenLabels {
    kind: &'static str,
}

/// The wire format a scraper asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// OpenMetrics 1.0 text.
    OpenMetrics,
    /// Prometheus text exposition format 0.0.4.
    Classic,
}

impl Format {
    /// The format an `Accept` header asks for: OpenMetrics when it ranks
    /// `application/openmetrics-text` at least as high as `text/plain`, the
    /// classic format otherwise, and for a client that states no preference
    /// (no header, `*/*`).
    pub fn negotiate(accept: Option<&str>) -> Self {
        let Some(accept) = accept else {
            return Self::Classic;
        };
        let (mut openmetrics, mut plain, mut any) = (None::<f32>, None::<f32>, None::<f32>);
        for range in accept.split(',') {
            let mut parts = range.split(';');
            let media = parts.next().unwrap_or_default().trim().to_ascii_lowercase();
            let q = parts
                .filter_map(|p| p.trim().strip_prefix("q="))
                .find_map(|q| q.trim().parse::<f32>().ok())
                .unwrap_or(1.0);
            let slot = match media.as_str() {
                "application/openmetrics-text" => &mut openmetrics,
                "text/plain" | "text/*" => &mut plain,
                "*/*" => &mut any,
                _ => continue,
            };
            *slot = Some(slot.map_or(q, |old| old.max(q)));
        }
        // A wildcard stands in for whatever the client named nothing for.
        let openmetrics = openmetrics.or(any).unwrap_or(0.0);
        let plain = plain.or(any).unwrap_or(0.0);
        if openmetrics > plain {
            Self::OpenMetrics
        } else {
            Self::Classic
        }
    }

    /// The `Content-Type` of a body in this format.
    pub fn content_type(self) -> &'static str {
        match self {
            Self::OpenMetrics => OPENMETRICS_CONTENT_TYPE,
            Self::Classic => CLASSIC_CONTENT_TYPE,
        }
    }
}

/// What the engine and the server's limits look like at one moment.
struct Snapshot {
    running: usize,
    queued: usize,
    capacity: usize,
    batching_slots: usize,
    batching_queued: usize,
    prefix_entries: usize,
    prefix_bytes: usize,
    prefix_hits: u64,
    prefix_misses: u64,
    memory_active: usize,
    memory_peak: usize,
    memory_cache: usize,
}

impl Snapshot {
    fn read(engine: &InferenceEngine, permits_in_use: usize, capacity: usize) -> Self {
        let (batching_slots, batching_queued) = engine.continuous_batching_depth();
        let (prefix_entries, prefix_bytes, prefix_hits, prefix_misses, _) =
            engine.prefix_cache_stats();
        Self {
            running: permits_in_use,
            queued: engine.queue_depth(),
            capacity,
            batching_slots,
            batching_queued,
            prefix_entries,
            prefix_bytes,
            prefix_hits,
            prefix_misses,
            memory_active: pmetal_runtime::inline_array::get_active_memory(),
            memory_peak: pmetal_runtime::inline_array::get_peak_memory(),
            memory_cache: pmetal_runtime::inline_array::get_cache_memory(),
        }
    }
}

/// The server's metrics: the registry and the handles that feed it.
pub struct Telemetry {
    registry: Mutex<Registry>,
    http_requests: Family<HttpLabels, Counter>,
    http_duration: Family<RouteLabels, Histogram>,
    http_in_flight: Gauge,
    generations: Counter,
    tokens: Family<TokenLabels, Counter>,
    time_to_first_token: Histogram,
    generation_duration: Histogram,
    tokens_per_second: Histogram,
    spec_drafted: Counter,
    spec_accepted: Counter,
    running: Gauge,
    queued: Gauge,
    capacity: Gauge,
    batching_slots: Gauge,
    batching_queued: Gauge,
    prefix_entries: Gauge,
    prefix_bytes: Gauge,
    prefix_hits: Counter,
    prefix_misses: Counter,
    memory_active: Gauge,
    memory_peak: Gauge,
    memory_cache: Gauge,
}

impl Telemetry {
    /// A registry for a server of `model`.
    pub fn new(model: &str) -> Self {
        let mut registry = Registry::default();
        let http_requests = Family::<HttpLabels, Counter>::default();
        let http_duration = Family::<RouteLabels, Histogram>::new_with_constructor(|| {
            // 5 ms to about 5 minutes: a health check to a long generation.
            Histogram::new(exponential_buckets(0.005, 2.5, 14))
        });
        let http_in_flight = Gauge::default();
        let generations = Counter::default();
        let tokens = Family::<TokenLabels, Counter>::default();
        let time_to_first_token = Histogram::new(exponential_buckets(0.01, 2.0, 14));
        let generation_duration = Histogram::new(exponential_buckets(0.05, 2.0, 14));
        let tokens_per_second = Histogram::new(exponential_buckets(1.0, 2.0, 12));
        let spec_drafted = Counter::default();
        let spec_accepted = Counter::default();
        let gauge = Gauge::default;
        let (running, queued, capacity) = (gauge(), gauge(), gauge());
        let (batching_slots, batching_queued) = (gauge(), gauge());
        let (prefix_entries, prefix_bytes) = (gauge(), gauge());
        let (prefix_hits, prefix_misses) = (Counter::default(), Counter::default());
        let (memory_active, memory_peak, memory_cache) = (gauge(), gauge(), gauge());

        registry.register(
            "pmetal_http_requests",
            "HTTP requests answered, by route pattern, method and status",
            http_requests.clone(),
        );
        registry.register(
            "pmetal_http_request_duration_seconds",
            "Seconds from a request's arrival to its response headers, by route pattern",
            http_duration.clone(),
        );
        registry.register(
            "pmetal_http_requests_in_flight",
            "HTTP requests being handled, scrapes included",
            http_in_flight.clone(),
        );
        registry.register(
            "pmetal_generation_requests",
            "Generations finished",
            generations.clone(),
        );
        registry.register(
            "pmetal_tokens",
            "Tokens processed: prompt tokens, prompt tokens the prefix cache supplied, and completion tokens",
            tokens.clone(),
        );
        registry.register(
            "pmetal_time_to_first_token_seconds",
            "Seconds from a generation's start to its first token",
            time_to_first_token.clone(),
        );
        registry.register(
            "pmetal_generation_duration_seconds",
            "Seconds a generation took, start to last token",
            generation_duration.clone(),
        );
        registry.register(
            "pmetal_generation_tokens_per_second",
            "Completion tokens per second of each generation",
            tokens_per_second.clone(),
        );
        registry.register(
            "pmetal_speculative_drafted_tokens",
            "Tokens a draft model or MTP head proposed to the target model",
            spec_drafted.clone(),
        );
        registry.register(
            "pmetal_speculative_accepted_tokens",
            "Drafted tokens the target model accepted; divide by drafted for the acceptance rate",
            spec_accepted.clone(),
        );
        registry.register(
            "pmetal_requests_running",
            "Generation requests holding a slot",
            running.clone(),
        );
        registry.register(
            "pmetal_requests_queued",
            "Generation requests waiting for the model",
            queued.clone(),
        );
        registry.register(
            "pmetal_request_capacity",
            "Generation requests the server runs at once before it answers 429",
            capacity.clone(),
        );
        registry.register(
            "pmetal_continuous_batching_active_slots",
            "Sequences decoding together under continuous batching",
            batching_slots.clone(),
        );
        registry.register(
            "pmetal_continuous_batching_queued_requests",
            "Requests waiting for a continuous-batching slot",
            batching_queued.clone(),
        );
        registry.register(
            "pmetal_prefix_cache_entries",
            "Prompt prefixes held in the prefix cache",
            prefix_entries.clone(),
        );
        registry.register(
            "pmetal_prefix_cache_bytes",
            "Bytes the prefix cache holds",
            prefix_bytes.clone(),
        );
        registry.register(
            "pmetal_prefix_cache_hits",
            "Prompts that found a cached prefix",
            prefix_hits.clone(),
        );
        registry.register(
            "pmetal_prefix_cache_misses",
            "Prompts that found no cached prefix",
            prefix_misses.clone(),
        );
        registry.register(
            "pmetal_memory_active_bytes",
            "GPU memory in live arrays",
            memory_active.clone(),
        );
        registry.register(
            "pmetal_memory_peak_bytes",
            "Most GPU memory in live arrays since the server started",
            memory_peak.clone(),
        );
        registry.register(
            "pmetal_memory_cache_bytes",
            "GPU memory the allocator holds for reuse",
            memory_cache.clone(),
        );
        let build = Family::<BuildLabels, Gauge>::default();
        build
            .get_or_create(&BuildLabels {
                version: env!("CARGO_PKG_VERSION").to_string(),
                model: model.to_string(),
            })
            .set(1);
        registry.register(
            "pmetal_build_info",
            "The server's version and the model it serves; always 1",
            build,
        );
        let started = Gauge::<i64, std::sync::atomic::AtomicI64>::default();
        started.set(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| i64::try_from(d.as_secs()).unwrap_or(i64::MAX)),
        );
        registry.register(
            "pmetal_start_time_seconds",
            "Unix time the server started",
            started,
        );

        Self {
            registry: Mutex::new(registry),
            http_requests,
            http_duration,
            http_in_flight,
            generations,
            tokens,
            time_to_first_token,
            generation_duration,
            tokens_per_second,
            spec_drafted,
            spec_accepted,
            running,
            queued,
            capacity,
            batching_slots,
            batching_queued,
            prefix_entries,
            prefix_bytes,
            prefix_hits,
            prefix_misses,
            memory_active,
            memory_peak,
            memory_cache,
        }
    }

    /// Count one finished generation.
    pub fn record_request(&self, metrics: &RequestMetrics) {
        self.generations.inc();
        let tokens = |kind, n: usize| {
            self.tokens
                .get_or_create(&TokenLabels { kind })
                .inc_by(n as u64);
        };
        tokens("prompt", metrics.prompt_tokens);
        tokens("cached", metrics.cached_prompt_tokens);
        tokens("completion", metrics.completion_tokens);
        self.time_to_first_token
            .observe(metrics.first_token_latency_ms / 1000.0);
        self.generation_duration
            .observe(metrics.total_latency_ms / 1000.0);
        if metrics.completion_tokens > 0 && metrics.tokens_per_second.is_finite() {
            self.tokens_per_second.observe(metrics.tokens_per_second);
        }
        if let Some(spec) = &metrics.speculative {
            self.spec_drafted.inc_by(spec.drafted_tokens as u64);
            self.spec_accepted.inc_by(spec.accepted_tokens as u64);
        }
    }

    /// Count an HTTP request that has been answered.
    fn record_http(&self, route: &str, method: &str, status: u16, seconds: f64) {
        self.http_requests
            .get_or_create(&HttpLabels {
                route: route.to_string(),
                method: method.to_string(),
                status: status.to_string(),
            })
            .inc();
        self.http_duration
            .get_or_create(&RouteLabels {
                route: route.to_string(),
                method: method.to_string(),
            })
            .observe(seconds);
    }

    /// The exposition: the engine's present state, then every series, in the
    /// format asked for. Returns its `Content-Type` and body.
    pub fn render(
        &self,
        engine: &InferenceEngine,
        permits_in_use: usize,
        capacity: usize,
        format: Format,
    ) -> (&'static str, String) {
        let snapshot = Snapshot::read(engine, permits_in_use, capacity);
        // Scrapes take turns, so a body never mixes two snapshots.
        let registry = self.registry.lock().unwrap_or_else(|e| e.into_inner());
        self.apply(&snapshot);
        let mut body = String::new();
        encode(&mut body, &registry).expect("writing to a String cannot fail");
        let body = match format {
            Format::OpenMetrics => body,
            Format::Classic => classic_text(&body),
        };
        (format.content_type(), body)
    }

    fn apply(&self, s: &Snapshot) {
        let gauge = |g: &Gauge, v: usize| {
            g.set(i64::try_from(v).unwrap_or(i64::MAX));
        };
        gauge(&self.running, s.running);
        gauge(&self.queued, s.queued);
        gauge(&self.capacity, s.capacity);
        gauge(&self.batching_slots, s.batching_slots);
        gauge(&self.batching_queued, s.batching_queued);
        gauge(&self.prefix_entries, s.prefix_entries);
        gauge(&self.prefix_bytes, s.prefix_bytes);
        gauge(&self.memory_active, s.memory_active);
        gauge(&self.memory_peak, s.memory_peak);
        gauge(&self.memory_cache, s.memory_cache);
        // The cache keeps its own totals; a counter takes them as they are.
        self.prefix_hits
            .inner()
            .store(s.prefix_hits, Ordering::Relaxed);
        self.prefix_misses
            .inner()
            .store(s.prefix_misses, Ordering::Relaxed);
    }
}

/// An OpenMetrics text body as the classic text exposition format 0.0.4.
///
/// The two differ in three ways that matter to a parser of the older format:
/// OpenMetrics names a counter family without `_total` (`# TYPE x counter`
/// over `x_total` samples) where the classic format names the family by the
/// series; OpenMetrics ends with `# EOF`; and it has `# UNIT` lines and the
/// `info` and `stateset` types, which the classic format lacks. Samples and
/// histograms are the same in both.
pub fn classic_text(openmetrics: &str) -> String {
    let counters: Vec<&str> = openmetrics
        .lines()
        .filter_map(|line| {
            let rest = line.strip_prefix("# TYPE ")?;
            let (name, kind) = rest.split_once(' ')?;
            (kind == "counter").then_some(name)
        })
        .collect();
    let mut out = String::with_capacity(openmetrics.len());
    for line in openmetrics.lines() {
        if line == "# EOF" || line.starts_with("# UNIT ") {
            continue;
        }
        let mut renamed = None;
        if let Some(rest) = line.strip_prefix("# TYPE ") {
            if let Some((name, kind)) = rest.split_once(' ') {
                let kind = match kind {
                    "info" | "stateset" => "gauge",
                    other => other,
                };
                if counters.contains(&name) {
                    renamed = Some(format!("# TYPE {name}_total counter"));
                } else {
                    renamed = Some(format!("# TYPE {name} {kind}"));
                }
            }
        } else if let Some(rest) = line.strip_prefix("# HELP ") {
            if let Some((name, help)) = rest.split_once(' ')
                && counters.contains(&name)
            {
                renamed = Some(format!("# HELP {name}_total {help}"));
            }
        }
        match renamed {
            Some(line) => out.push_str(&line),
            None => out.push_str(line),
        }
        out.push('\n');
    }
    out
}

/// The route label for a request: the router's pattern, so a client cannot
/// make a series by inventing paths. A request no route matched is
/// `unmatched`.
fn route_label(request: &Request) -> String {
    request
        .extensions()
        .get::<MatchedPath>()
        .map_or_else(|| "unmatched".to_string(), |p| p.as_str().to_string())
}

/// Middleware counting every request: by route, method and status, with its
/// latency, and while it is in flight.
pub(crate) async fn track_http(
    State(state): State<Arc<AppState>>,
    request: Request,
    next: Next,
) -> Response {
    let route = route_label(&request);
    let method = request.method().as_str().to_string();
    let started = Instant::now();
    state.telemetry.http_in_flight.inc();
    let response = next.run(request).await;
    state.telemetry.http_in_flight.dec();
    state.telemetry.record_http(
        &route,
        &method,
        response.status().as_u16(),
        started.elapsed().as_secs_f64(),
    );
    response
}

/// `GET /metrics` — the Prometheus scrape endpoint.
///
/// Like every route but `/health`, it needs an API key when the server has
/// one: a scraper sends it as `Authorization: Bearer <key>`.
pub async fn metrics(State(state): State<Arc<AppState>>, headers: HeaderMap) -> impl IntoResponse {
    let accept = headers.get(header::ACCEPT).and_then(|v| v.to_str().ok());
    let format = Format::negotiate(accept);
    let in_use = state
        .configured_permits
        .saturating_sub(state.request_permits.available_permits());
    let (content_type, body) =
        state
            .telemetry
            .render(&state.engine, in_use, state.configured_permits, format);
    (
        [(header::CONTENT_TYPE, HeaderValue::from_static(content_type))],
        body,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn negotiation_follows_the_accept_header() {
        use Format::{Classic, OpenMetrics};
        let cases = [
            (None, Classic),
            (Some("*/*"), Classic),
            (Some("text/plain"), Classic),
            (Some("text/plain; version=0.0.4"), Classic),
            (Some("application/openmetrics-text"), OpenMetrics),
            (
                Some("application/openmetrics-text; version=1.0.0; charset=utf-8"),
                OpenMetrics,
            ),
            // What a Prometheus server sends.
            (
                Some(
                    "application/openmetrics-text;version=1.0.0,application/openmetrics-text;version=0.0.1;q=0.75,text/plain;version=0.0.4;q=0.5,*/*;q=0.1",
                ),
                OpenMetrics,
            ),
            // The classic format ranked higher.
            (
                Some("application/openmetrics-text;q=0.5, text/plain;q=0.9"),
                Classic,
            ),
            (Some("application/openmetrics-text;q=0"), Classic),
            // Something else entirely.
            (Some("application/json"), Classic),
            (Some("text/html, application/xhtml+xml"), Classic),
        ];
        for (accept, want) in cases {
            assert_eq!(Format::negotiate(accept), want, "Accept: {accept:?}");
        }
    }

    const OPENMETRICS: &str = "\
# HELP requests Requests answered.
# TYPE requests counter
requests_total{route=\"/v1/models\"} 3
# HELP latency_seconds How long.
# TYPE latency_seconds histogram
latency_seconds_sum 1.5
latency_seconds_count 3
latency_seconds_bucket{le=\"+Inf\"} 3
# HELP up Whether it is up.
# TYPE up gauge
up 1
# HELP build_info Build.
# TYPE build_info info
build_info{version=\"1\"} 1
# HELP size_bytes Size.
# TYPE size_bytes gauge
# UNIT size_bytes bytes
size_bytes 7
# EOF
";

    #[test]
    fn classic_text_names_counters_by_their_series() {
        let classic = classic_text(OPENMETRICS);
        assert!(classic.contains("# HELP requests_total Requests answered.\n"));
        assert!(classic.contains("# TYPE requests_total counter\n"));
        assert!(classic.contains("requests_total{route=\"/v1/models\"} 3\n"));
        assert!(!classic.contains("# TYPE requests counter"));
        // Everything else passes through, the info metric as a gauge.
        assert!(classic.contains("# TYPE latency_seconds histogram\n"));
        assert!(classic.contains("latency_seconds_bucket{le=\"+Inf\"} 3\n"));
        assert!(classic.contains("# TYPE up gauge\nup 1\n"));
        assert!(classic.contains("# TYPE build_info gauge\n"));
        // And the OpenMetrics-only lines are gone.
        assert!(!classic.contains("# EOF"));
        assert!(!classic.contains("# UNIT"));
        assert!(classic.ends_with('\n'));
    }

    /// What the registry really emits, through the converter: every line of
    /// the classic body is a comment or `name{labels} value`.
    #[test]
    fn the_registry_encodes_to_both_formats() {
        let telemetry = Telemetry::new("m");
        telemetry.record_http("/v1/models", "GET", 200, 0.02);
        let mut raw = String::new();
        encode(&mut raw, &telemetry.registry.lock().unwrap()).unwrap();
        assert!(raw.ends_with("# EOF\n"), "{raw}");
        let classic = classic_text(&raw);
        assert!(classic.contains(
            "pmetal_http_requests_total{route=\"/v1/models\",method=\"GET\",status=\"200\"} 1\n"
        ));
        assert!(classic.contains("# TYPE pmetal_http_requests_total counter\n"));
        for line in classic.lines().filter(|l| !l.starts_with('#')) {
            let (series, value) = line.rsplit_once(' ').expect("a sample line");
            assert!(!series.is_empty());
            value.parse::<f64>().unwrap_or_else(|_| panic!("{line}"));
        }
    }
}
