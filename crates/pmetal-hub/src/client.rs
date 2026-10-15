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

    /// The Hub client for these options.
    ///
    /// The token is always resolved here (see [`resolve_token`]) and handed to
    /// the client explicitly, so the client and this crate's own requests can
    /// never disagree about who is calling.
    pub(crate) fn client(&self) -> Result<HFClient> {
        let mut builder = HFClient::builder()
            .cache_dir(self.resolved_cache_dir())
            .retry_max_attempts(self.max_retries.unwrap_or(DEFAULT_RETRIES))
            .client(self.http_client()?);
        if let Some(token) = resolve_token(self.token.as_ref()) {
            builder = builder.token(token.expose_secret());
        }
        if let Some(endpoint) = self.endpoint.as_deref().filter(|e| !e.trim().is_empty()) {
            builder = builder.endpoint(endpoint.trim());
        }
        builder.build().map_err(|e| hub_error(&e))
    }
}

impl HubOptions {
    /// The HTTP client every request goes through. Without timeouts a stalled
    /// connection would hang a download forever; this one gives up on a
    /// connection that goes silent, however large the file.
    fn http_client(&self) -> Result<reqwest::Client> {
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
    let of = repo.map(|r| format!(" for '{r}'")).unwrap_or_default();
    // Listing endpoints report every failure as a bare HTTP status; the
    // statuses that mean "who are you" and "you may not" read the same as
    // their dedicated variants.
    let status = match e {
        HFError::Http { context } => Some(context.status.as_u16()),
        HFError::AuthRequired { .. } => Some(401),
        HFError::Forbidden { .. } => Some(403),
        _ => None,
    };
    let message = match (e, status) {
        (_, Some(401)) => format!(
            "authentication required{of}, or the token was rejected: set HF_TOKEN, run \
             `hf auth login`, or pass a valid access token ({e})"
        ),
        (_, Some(403)) => {
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
                 (a fine-grained token also needs read access to the repository) ({e})"
            )
        }
        (HFError::RepoNotFound { repo_id, .. }, _) => format!(
            "repository '{repo_id}' was not found: check the name, or supply a token if it is \
             private (the Hub answers 404 for private repositories without access)"
        ),
        (HFError::RateLimited { .. }, _) | (_, Some(429)) => {
            let wait = match e {
                HFError::RateLimited {
                    retry_after: Some(d),
                    ..
                } => format!(" Retry in {}s.", d.as_secs().max(1)),
                _ => String::new(),
            };
            format!(
                "the Hub is rate limiting requests{of}.{wait} A token raises the limits \
                 (HF_TOKEN) ({e})"
            )
        }
        (HFError::LocalEntryNotFound { path }, _) => format!(
            "'{path}' is not in the local cache and the network is off (offline mode); \
             download it once online first"
        ),
        (other, _) => other.to_string(),
    };
    PMetalError::Hub(redact_secrets(&message))
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
    fn errors_for_a_missing_local_entry_say_the_network_is_off() {
        let err = hub_error_in(
            "o/m",
            &HFError::LocalEntryNotFound {
                path: "config.json".into(),
            },
        );
        assert!(err.to_string().contains("offline"), "{err}");
    }
}
