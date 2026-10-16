//! Requests to a `pmetal serve` instance on this machine.
//!
//! The target is always `127.0.0.1` on a port the caller names, at a path this
//! module's callers fix: a tool cannot be pointed at another host or another
//! URL. The client follows no redirects and uses no proxy, so nothing can send a
//! request somewhere the address did not say. A call has a timeout, a reply a
//! size limit, and a cancelled request stops waiting.

use std::time::Duration;

use serde_json::Value;
use turbomcp::prelude::*;

/// Longest a request may take (a long generation is slow), unless
/// `PMETAL_MCP_CHAT_TIMEOUT_SECS` says otherwise.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(600);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
/// Largest reply read.
const MAX_REPLY_BYTES: usize = 16 * 1024 * 1024;
/// Ports below this belong to the system's own services.
const MIN_PORT: u32 = 1024;

fn timeout() -> Duration {
    std::env::var("PMETAL_MCP_CHAT_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|&s| s > 0)
        .map_or(DEFAULT_TIMEOUT, Duration::from_secs)
}

/// The port as a `u16`, if it is one a server may be listening on.
pub fn checked_port(port: u32) -> McpResult<u16> {
    u16::try_from(port)
        .ok()
        .filter(|_| port >= MIN_PORT)
        .ok_or_else(|| {
            McpError::invalid_params(format!(
                "{port} is not a port a server runs on ({MIN_PORT}-65535)"
            ))
        })
}

/// POST `body` to `path` on the server at `port`, and return the reply body.
/// A status other than success is an error carrying the server's message: the
/// caller's (invalid params) for a 4xx, the server's for anything else.
pub async fn post_json(
    port: u32,
    path: &str,
    body: &Value,
    ctx: &RequestContext,
) -> McpResult<String> {
    let port = checked_port(port)?;
    let url = format!("http://127.0.0.1:{port}{path}");
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .no_proxy()
        .connect_timeout(CONNECT_TIMEOUT)
        .timeout(timeout())
        .build()
        .map_err(|e| McpError::internal(format!("cannot create the HTTP client: {e}")))?;

    let exchange = async {
        let mut response =
            client.post(&url).json(body).send().await.map_err(|e| {
                McpError::unavailable(format!("failed to reach serve at {url}: {e}"))
            })?;
        let status = response.status();
        let mut reply: Vec<u8> = Vec::new();
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|e| McpError::internal(format!("serve at {url} stopped replying: {e}")))?
        {
            if reply.len() + chunk.len() > MAX_REPLY_BYTES {
                return Err(McpError::internal(format!(
                    "the reply from serve is larger than {} MiB",
                    MAX_REPLY_BYTES / (1024 * 1024)
                )));
            }
            reply.extend_from_slice(&chunk);
        }
        let text = String::from_utf8_lossy(&reply).into_owned();
        if status.is_success() {
            return Ok(text);
        }
        let message = format!("serve returned {status}: {text}");
        Err(if status.is_client_error() {
            McpError::invalid_params(message)
        } else {
            McpError::internal(message)
        })
    };
    tokio::pin!(exchange);
    let mut tick = tokio::time::interval(Duration::from_millis(250));
    loop {
        tokio::select! {
            result = &mut exchange => return result,
            _ = tick.tick() => {
                if ctx.is_cancelled() {
                    return Err(McpError::cancelled("the request was cancelled"));
                }
            }
        }
    }
}
