//! Plain JSON requests to the Hub's HTTP API, for the endpoints the download
//! client has no call for (searching with field selection).
//!
//! Bounded in every direction: a retry count, a wait between retries, and a
//! cap on the body, which is read in chunks so an oversized or endless
//! response is cut off instead of held in memory.

use std::time::Duration;

use pmetal_core::{PMetalError, Result};

use crate::auth::{redact_secrets, resolve_token};
use crate::client::{HubOptions, status_message};

/// The longest wait honored from a `Retry-After`, and the longest backoff.
const MAX_WAIT: Duration = Duration::from_secs(10);
/// The first backoff; each retry doubles it.
const BASE_WAIT: Duration = Duration::from_millis(200);

/// GET `path` (starting with `/`) on the Hub with `query` and return the body,
/// at most `max_bytes` of it.
///
/// Transient failures (a dropped connection, 429, 5xx) are retried up to
/// `options.max_retries` times. `subject` names what is being fetched, for
/// error messages.
pub(crate) async fn get_json(
    options: &HubOptions,
    path: &str,
    query: &[(String, String)],
    max_bytes: usize,
    subject: Option<&str>,
) -> Result<Vec<u8>> {
    if options.is_offline() {
        return Err(PMetalError::Hub(
            "the network is off (offline mode); this needs the Hub".into(),
        ));
    }
    let client = options.http_client()?;
    let url = format!("{}{path}", options.resolved_endpoint());
    let token = resolve_token(options.token.as_ref());

    let mut attempt = 0usize;
    loop {
        let mut request = client.get(&url).query(query);
        if let Some(token) = &token {
            request = request.bearer_auth(token.expose_secret());
        }
        let outcome = request.send().await;
        let last = attempt >= options.retries();
        match outcome {
            Ok(response) if response.status().is_success() => {
                return read_bounded(response, max_bytes).await;
            }
            Ok(response) => {
                let status = response.status();
                let retry_after = response
                    .headers()
                    .get(reqwest::header::RETRY_AFTER)
                    .and_then(|v| v.to_str().ok())
                    .and_then(|v| v.trim().parse::<u64>().ok())
                    .map(Duration::from_secs);
                let transient = status.as_u16() == 429 || status.is_server_error();
                if transient && !last {
                    tokio::time::sleep(wait_for(attempt, retry_after)).await;
                    attempt += 1;
                    continue;
                }
                let detail = server_message(response).await;
                return Err(status_error(status.as_u16(), subject, retry_after, &detail));
            }
            Err(e) => {
                let transient = e.is_timeout() || e.is_connect();
                if transient && !last {
                    tokio::time::sleep(wait_for(attempt, None)).await;
                    attempt += 1;
                    continue;
                }
                // The URL carries the query, never a credential, but scrub anyway.
                return Err(PMetalError::Hub(redact_secrets(&format!(
                    "request to the Hub failed: {e}"
                ))));
            }
        }
    }
}

fn wait_for(attempt: usize, retry_after: Option<Duration>) -> Duration {
    let backoff = BASE_WAIT.saturating_mul(1u32 << attempt.min(8));
    retry_after.unwrap_or(backoff).min(MAX_WAIT)
}

/// The body of `response`, refusing more than `max_bytes`.
async fn read_bounded(mut response: reqwest::Response, max_bytes: usize) -> Result<Vec<u8>> {
    let too_big = || {
        PMetalError::Hub(format!(
            "the Hub's response is larger than {} bytes; refusing to read it",
            max_bytes
        ))
    };
    if response
        .content_length()
        .is_some_and(|n| n > max_bytes as u64)
    {
        return Err(too_big());
    }
    let mut body = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|e| PMetalError::Hub(format!("reading the Hub's response failed: {e}")))?
    {
        if body.len() + chunk.len() > max_bytes {
            return Err(too_big());
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

/// The Hub's own explanation of a failure: its `error` field, else the start
/// of the body.
async fn server_message(response: reqwest::Response) -> String {
    let code = response
        .headers()
        .get("x-error-message")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    let body = read_bounded(response, 16 * 1024).await.unwrap_or_default();
    let text = String::from_utf8_lossy(&body);
    let from_json = serde_json::from_str::<serde_json::Value>(&text)
        .ok()
        .and_then(|v| v.get("error").and_then(|e| e.as_str()).map(str::to_owned));
    code.or(from_json)
        .unwrap_or_else(|| text.chars().take(200).collect())
}

fn status_error(
    status: u16,
    subject: Option<&str>,
    retry_after: Option<Duration>,
    detail: &str,
) -> PMetalError {
    let message =
        status_message(status, subject, retry_after, detail).unwrap_or_else(|| match status {
            404 => format!(
                "not found{}{}",
                subject.map(|s| format!(": {s}")).unwrap_or_default(),
                if detail.is_empty() {
                    String::new()
                } else {
                    format!(" ({detail})")
                }
            ),
            _ => format!(
                "the Hub answered HTTP {status}{}",
                if detail.is_empty() {
                    String::new()
                } else {
                    format!(": {detail}")
                }
            ),
        });
    PMetalError::Hub(redact_secrets(&message))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn waits_double_and_are_capped() {
        assert_eq!(wait_for(0, None), Duration::from_millis(200));
        assert_eq!(wait_for(1, None), Duration::from_millis(400));
        assert_eq!(wait_for(20, None), MAX_WAIT);
        // The server's Retry-After wins over the backoff, within the cap.
        assert_eq!(
            wait_for(0, Some(Duration::from_secs(3))),
            Duration::from_secs(3)
        );
        assert_eq!(wait_for(0, Some(Duration::from_secs(600))), MAX_WAIT);
    }

    #[test]
    fn statuses_become_actionable_messages() {
        let e = status_error(401, Some("search"), None, "bad token").to_string();
        assert!(e.contains("HF_TOKEN"), "{e}");
        let e = status_error(404, Some("o/m"), None, "").to_string();
        assert!(e.contains("not found: o/m"), "{e}");
        let e = status_error(502, None, None, "bad gateway").to_string();
        assert!(e.contains("502") && e.contains("bad gateway"), "{e}");
    }
}
