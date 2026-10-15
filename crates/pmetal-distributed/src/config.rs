use crate::error::DistributedError;
use crate::secret::ClusterSecret;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::net::SocketAddr;

/// Configuration for distributed training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedConfig {
    /// Primary address per node (one per rank). The order must be consistent
    /// across the cluster. Entry `i` is the preferred fabric for rank `i`
    /// (typically Thunderbolt, populated by [`crate::auto::AutoDiscoveryBackend`]).
    pub nodes: Vec<SocketAddr>,

    /// Optional fallback addresses, indexed by rank. `fallback_addrs[i]`
    /// is the list of *additional* socket addrs to try for rank `i` if
    /// `nodes[i]` is unreachable (e.g. TB cable disconnected ⇒ try Ethernet).
    /// Entries are tried in order. Empty / absent ⇒ no fallback.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fallback_addrs: Vec<Vec<SocketAddr>>,

    /// Rank of this node (index into nodes list).
    pub rank: usize,

    /// Connection timeout in milliseconds (default: 30000).
    #[serde(default = "default_connection_timeout_ms")]
    pub connection_timeout_ms: u64,

    /// Maximum connection retry attempts (default: 50).
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,

    /// The cluster secret that authenticates and encrypts every link. Left
    /// unset, it is looked up as [`ClusterSecret::resolve`] does (the
    /// `PMETAL_CLUSTER_SECRET` variable, then `~/.pmetal/cluster_secret`).
    /// Never serialized.
    #[serde(skip)]
    pub secret: Option<ClusterSecret>,

    /// Run the links unauthenticated and unencrypted when no secret is
    /// available. For tests and throwaway local runs only: anyone on the
    /// network can then read gradients and inject their own.
    #[serde(default)]
    pub allow_plaintext: bool,
}

fn default_connection_timeout_ms() -> u64 {
    30000
}

fn default_max_retries() -> u32 {
    50
}

impl DistributedConfig {
    /// Create a new configuration with one address per node.
    pub fn new(nodes: Vec<SocketAddr>, rank: usize) -> Self {
        Self {
            nodes,
            fallback_addrs: Vec::new(),
            rank,
            connection_timeout_ms: default_connection_timeout_ms(),
            max_retries: default_max_retries(),
            secret: None,
            allow_plaintext: false,
        }
    }

    /// Create a configuration with primary + fallback addresses per node.
    /// `endpoints[i]` is rank `i`'s ranked list (best first); `endpoints[i][0]`
    /// becomes the primary, the remainder become fallbacks.
    pub fn with_endpoints(endpoints: Vec<Vec<SocketAddr>>, rank: usize) -> Self {
        let mut nodes = Vec::with_capacity(endpoints.len());
        let mut fallback_addrs = Vec::with_capacity(endpoints.len());
        for endpoint_list in endpoints {
            let mut iter = endpoint_list.into_iter();
            // Empty endpoint lists become a placeholder; validate() will reject.
            nodes
                .push(iter.next().unwrap_or_else(|| {
                    "0.0.0.0:0".parse().expect("placeholder addr always valid")
                }));
            fallback_addrs.push(iter.collect());
        }
        Self {
            nodes,
            fallback_addrs,
            rank,
            connection_timeout_ms: default_connection_timeout_ms(),
            max_retries: default_max_retries(),
            secret: None,
            allow_plaintext: false,
        }
    }

    /// Authenticate and encrypt the links with `secret` rather than the
    /// machine's own.
    pub fn with_secret(mut self, secret: ClusterSecret) -> Self {
        self.secret = Some(secret);
        self
    }

    /// Permit unauthenticated, unencrypted links when no secret is
    /// available. See [`Self::allow_plaintext`].
    pub fn with_plaintext_allowed(mut self) -> Self {
        self.allow_plaintext = true;
        self
    }

    /// All addresses to try for rank `r`, in priority order
    /// (primary first, then any fallbacks).
    pub fn addrs_for(&self, r: usize) -> Vec<SocketAddr> {
        let mut out = Vec::with_capacity(1 + self.fallback_addrs.get(r).map_or(0, Vec::len));
        if let Some(primary) = self.nodes.get(r) {
            out.push(*primary);
        }
        if let Some(extras) = self.fallback_addrs.get(r) {
            out.extend(extras.iter().copied());
        }
        out
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.nodes.is_empty() {
            return Err(DistributedError::Config("nodes list cannot be empty".to_string()).into());
        }

        if self.rank >= self.nodes.len() {
            return Err(DistributedError::Config(format!(
                "rank {} is out of bounds for {} nodes",
                self.rank,
                self.nodes.len()
            ))
            .into());
        }

        // Check for duplicate addresses
        let unique: HashSet<_> = self.nodes.iter().collect();
        if unique.len() != self.nodes.len() {
            return Err(DistributedError::Config(
                "nodes list contains duplicate addresses".to_string(),
            )
            .into());
        }

        Ok(())
    }

    /// Get the world size (number of nodes).
    pub fn world_size(&self) -> usize {
        self.nodes.len()
    }
}
