//! The Hub connection every operation in this crate shares: credentials,
//! endpoint, cache location, offline mode, and how Hub errors reach a user.

use std::path::PathBuf;
use std::time::Duration;

use hf_hub::{HFClient, HFError};
use pmetal_core::{PMetalError, Result, SecretString};

use crate::auth::{redact_secrets, resolve_token};

/// How to reach the Hub.
///
/// Every field is optional; the default is what the Hub's own tools do:
/// the `HF_TOKEN` / token-file credentials, `HF_ENDPOINT` (or
/// `https://huggingface.co`), the standard cache directory, and `HF_HUB_OFFLINE`.
#[derive(Debug, Clone, Default)]
pub struct HubOptions {
    /// Access token. `None` falls back to [`resolve_token`]'s implicit
    /// sources. Needed for private and gated repositories.
    pub token: Option<SecretString>,
    /// Hub base URL, for a mirror or a private deployment; the `HF_ENDPOINT`
    /// environment variable when `None`.
    pub endpoint: Option<String>,
    /// Cache directory; [`crate::cache_dir`] when `None`.
    pub cache_dir: Option<PathBuf>,
    /// Never touch the network: resolve everything from the cache. `None`
    /// follows `HF_HUB_OFFLINE`.
    pub offline: Option<bool>,
    /// Retries after a failed request that is worth repeating (a dropped
    /// connection, a 5xx, a 429); four when `None`. Each wait doubles, so the
    /// total time is bounded.
    pub max_retries: Option<usize>,
    /// How long a connection may go without a single byte arriving before the
    /// request fails (it is not a limit on the whole transfer); 60 seconds
    /// when `None`.
    pub stall_timeout: Option<Duration>,
}

/// Retries when [`HubOptions::max_retries`] is unset.
const DEFAULT_RETRIES: usize = 4;
/// Silence tolerated when [`HubOptions::stall_timeout`] is unset.
const DEFAULT_STALL_TIMEOUT: Duration = Duration::from_secs(60);
/// How long to wait for a connection to be established.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(20);
/// The Hub's address when nothing says otherwise.
const DEFAULT_ENDPOINT: &str = "https://huggingface.co";

impl HubOptions {
    /// Options carrying only the token a free function was handed.
    pub(crate) fn from_token(token: Option<&SecretString>) -> Self {
        Self {
            token: token.cloned(),
            ..Self::default()
        }
    }

    /// The cache directory these options resolve to.
    pub fn resolved_cache_dir(&self) -> PathBuf {
        self.cache_dir
            .clone()
            .unwrap_or_else(crate::cache::cache_dir)
    }

    /// Whether these options forbid network access.
    pub fn is_offline(&self) -> bool {
        self.offline
            .unwrap_or_else(|| offline_from(std::env::var("HF_HUB_OFFLINE").ok().as_deref()))
    }

    /// The Hub's base URL without a trailing slash: the explicit endpoint,
    /// else `HF_ENDPOINT`, else `https://huggingface.co`.
    pub fn resolved_endpoint(&self) -> String {
        self.endpoint
            .clone()
            .filter(|e| !e.trim().is_empty())
            .or_else(|| std::env::var("HF_ENDPOINT").ok())
            .filter(|e| !e.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_ENDPOINT.to_owned())
            .trim()
            .trim_end_matches('/')
            .to_owned()
    }

    /// The Hub client for these options.
    ///
    /// The token is always resolved here (see [`resolve_token`]) and handed to
    /// the client explicitly, so the client and this crate's own requests can
    /// never disagree about who is calling.
    pub(crate) fn client(&self) -> Result<HFClient> {
        let mut builder = HFClient::builder()
            .cache_dir(self.resolved_cache_dir())
            .retry_max_attempts(self.max_retries.unwrap_or(DEFAULT_RETRIES))
            .endpoint(self.resolved_endpoint())
            .client(self.http_client()?);
        if let Some(token) = resolve_token(self.token.as_ref()) {
            builder = builder.token(token.expose_secret());
        }
        builder.build().map_err(|e| hub_error(&e))
    }

    /// The HTTP client every request goes through. Without timeouts a stalled
    /// connection would hang a download forever; this one gives up on a
    /// connection that goes silent, however large the file.
    pub(crate) fn http_client(&self) -> Result<reqwest::Client> {
        let mut agent = format!("pmetal/{}", env!("CARGO_PKG_VERSION"));
        if let Ok(origin) = std::env::var("HF_HUB_USER_AGENT_ORIGIN") {
            agent.push_str("; ");
            agent.push_str(&origin);
        }
        reqwest::Client::builder()
            .user_agent(agent)
            .connect_timeout(CONNECT_TIMEOUT)
            .read_timeout(self.stall_timeout.unwrap_or(DEFAULT_STALL_TIMEOUT))
            .build()
            .map_err(|e| PMetalError::Hub(format!("could not create the HTTP client: {e}")))
    }

    /// Retries for this connection.
    pub(crate) fn retries(&self) -> usize {
        self.max_retries.unwrap_or(DEFAULT_RETRIES)
    }
}

/// Whether an `HF_HUB_OFFLINE` value turns offline mode on (the Hub tools'
/// truthy set: `1`, `on`, `yes`, `true`, any case).
fn offline_from(value: Option<&str>) -> bool {
    value.is_some_and(|v| {
        matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "on" | "yes" | "true"
        )
    })
}

/// A Hub failure as a user-facing error: what happened, and for the failures
/// users hit constantly (no access, gated model, rate limit, offline) what to
/// do about it. Never contains a credential.
pub(crate) fn hub_error(e: &HFError) -> PMetalError {
    hub_error_for(None, e)
}

/// [`hub_error`] for a failure while working on `repo` (`owner/name`).
pub(crate) fn hub_error_in(repo: &str, e: &HFError) -> PMetalError {
    hub_error_for(Some(repo), e)
}

fn hub_error_for(repo: Option<&str>, e: &HFError) -> PMetalError {
    // Listing endpoints report every failure as a bare HTTP status; the
    // statuses that mean "who are you", "you may not" and "slow down" read the
    // same as their dedicated variants.
    let (status, retry_after) = match e {
        HFError::Http { context } => (Some(context.status.as_u16()), None),
        HFError::AuthRequired { .. } => (Some(401), None),
        HFError::Forbidden { .. } => (Some(403), None),
        HFError::RateLimited { retry_after, .. } => (Some(429), *retry_after),
        _ => (None, None),
    };
    let message = status
        .and_then(|s| status_message(s, repo, retry_after, &e.to_string()))
        .unwrap_or_else(|| match e {
            HFError::RepoNotFound { repo_id, .. } => format!(
                "repository '{repo_id}' was not found: check the name, or supply a token if it \
                 is private (the Hub answers 404 for private repositories without access)"
            ),
            HFError::LocalEntryNotFound { path } => format!(
                "'{path}' is not in the local cache and the network is off (offline mode); \
                 download it once online first"
            ),
            other => other.to_string(),
        });
    PMetalError::Hub(redact_secrets(&message))
}

/// What a 401, 403 or 429 from the Hub means and what to do about it; `None`
/// for any other status. `detail` is the lower layer's own description.
pub(crate) fn status_message(
    status: u16,
    repo: Option<&str>,
    retry_after: Option<Duration>,
    detail: &str,
) -> Option<String> {
    let of = repo.map(|r| format!(" for '{r}'")).unwrap_or_default();
    Some(match status {
        401 => format!(
            "authentication required{of}, or the token was rejected: set HF_TOKEN, run \
             `hf auth login`, or pass a valid access token ({detail})"
        ),
        403 => {
            let gated = repo
                .map(|r| {
                    format!(
                        "; if '{r}' is gated, accept its terms at https://huggingface.co/{r} \
                         with the account the token belongs to"
                    )
                })
                .unwrap_or_default();
            format!(
                "access denied{of}: the token is valid but not allowed to do this{gated} \
                 (a fine-grained token also needs read access to the repository) ({detail})"
            )
        }
        429 => {
            let wait = retry_after
                .map(|d| format!(" Retry in {}s.", d.as_secs().max(1)))
                .unwrap_or_default();
            format!(
                "the Hub is rate limiting requests{of}.{wait} A token raises the limits \
                 (HF_TOKEN) ({detail})"
            )
        }
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offline_values_follow_the_hub_tools() {
        for on in ["1", "ON", "yes", "True", " true "] {
            assert!(offline_from(Some(on)), "{on:?}");
        }
        for off in ["", "0", "off", "no", "false", "2"] {
            assert!(!offline_from(Some(off)), "{off:?}");
        }
        assert!(!offline_from(None));
    }

    #[test]
    fn an_explicit_offline_choice_beats_the_environment() {
        let on = HubOptions {
            offline: Some(true),
            ..Default::default()
        };
        let off = HubOptions {
            offline: Some(false),
            ..Default::default()
        };
        assert!(on.is_offline());
        assert!(!off.is_offline());
    }

    #[test]
    fn a_bad_endpoint_is_an_error_not_a_panic() {
        let options = HubOptions {
            endpoint: Some("not a url".into()),
            ..Default::default()
        };
        assert!(options.client().is_err());
    }

    #[test]
    fn the_endpoint_is_explicit_first_and_has_no_trailing_slash() {
        let options = HubOptions {
            endpoint: Some(" http://mirror.example/ ".into()),
            ..Default::default()
        };
        assert_eq!(options.resolved_endpoint(), "http://mirror.example");
    }

    #[test]
    fn errors_for_a_missing_local_entry_say_the_network_is_off() {
        let err = hub_error_in(
            "o/m",
            &HFError::LocalEntryNotFound {
                path: "config.json".into(),
            },
        );
        assert!(err.to_string().contains("offline"), "{err}");
    }

    #[test]
    fn auth_statuses_explain_themselves_and_others_do_not() {
        let gated = status_message(403, Some("o/m"), None, "x").unwrap();
        assert!(gated.contains("https://huggingface.co/o/m"), "{gated}");
        assert!(
            status_message(401, None, None, "x")
                .unwrap()
                .contains("HF_TOKEN")
        );
        let limited = status_message(429, None, Some(Duration::from_secs(9)), "x").unwrap();
        assert!(limited.contains("Retry in 9s"), "{limited}");
        assert!(status_message(500, None, None, "x").is_none());
    }
}
