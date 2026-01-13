//! Network namespace isolation via pre-shared keys.
//!
//! Provides cryptographic isolation between different clusters or versions
//! to prevent cross-talk. Uses SHA3-256 hash of the namespace string to
//! generate a consistent PSK across all nodes in the same namespace.
//!
//! # Usage
//!
//! ```ignore
//! let namespace = NetworkNamespace::new("pmetal/0.1.0", Some("my-cluster"));
//! let psk = namespace.psk();
//! ```
//!
//! # Environment Variable
//!
//! The namespace can be overridden via `PMETAL_NAMESPACE` environment variable.

use sha3::{Digest, Sha3_256};
use std::env;
use tracing::info;

/// Default version string.
const DEFAULT_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Environment variable for namespace override.
const NAMESPACE_ENV_VAR: &str = "PMETAL_NAMESPACE";

/// Network namespace for cluster isolation.
#[derive(Debug, Clone)]
pub struct NetworkNamespace {
    /// Version string (e.g., "pmetal/0.1.0").
    version: String,
    /// Optional cluster name for additional isolation.
    cluster: Option<String>,
    /// Pre-computed PSK.
    psk: [u8; 32],
    /// Human-readable namespace string.
    namespace_str: String,
}

impl NetworkNamespace {
    /// Create a new network namespace.
    ///
    /// # Arguments
    ///
    /// * `version` - Version string (e.g., "pmetal/0.1.0")
    /// * `cluster` - Optional cluster name for additional isolation
    pub fn new(version: &str, cluster: Option<&str>) -> Self {
        // Check for environment override
        let namespace_override = env::var(NAMESPACE_ENV_VAR).ok();

        let namespace_str = match &namespace_override {
            Some(override_ns) => override_ns.clone(),
            None => match cluster {
                Some(c) => format!("{}/{}", version, c),
                None => version.to_string(),
            },
        };

        let psk = Self::compute_psk(&namespace_str);

        if namespace_override.is_some() {
            info!(
                "Using namespace override from {}: {}",
                NAMESPACE_ENV_VAR, namespace_str
            );
        } else {
            info!("Network namespace: {}", namespace_str);
        }

        Self {
            version: version.to_string(),
            cluster: cluster.map(|s| s.to_string()),
            psk,
            namespace_str,
        }
    }

    /// Create a namespace with the default version.
    pub fn default_version(cluster: Option<&str>) -> Self {
        Self::new(&format!("pmetal/{}", DEFAULT_VERSION), cluster)
    }

    /// Compute the PSK from the namespace string.
    fn compute_psk(namespace: &str) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(namespace.as_bytes());
        let result = hasher.finalize();

        let mut psk = [0u8; 32];
        psk.copy_from_slice(&result);
        psk
    }

    /// Get the pre-shared key.
    pub fn psk(&self) -> &[u8; 32] {
        &self.psk
    }

    /// Get the namespace string.
    pub fn namespace_str(&self) -> &str {
        &self.namespace_str
    }

    /// Get the version.
    pub fn version(&self) -> &str {
        &self.version
    }

    /// Get the cluster name.
    pub fn cluster(&self) -> Option<&str> {
        self.cluster.as_deref()
    }

    /// Check if another namespace is compatible.
    pub fn is_compatible(&self, other: &NetworkNamespace) -> bool {
        self.psk == other.psk
    }

    /// Verify a received PSK matches ours.
    pub fn verify_psk(&self, received: &[u8; 32]) -> bool {
        // Constant-time comparison
        let mut result = 0u8;
        for (a, b) in self.psk.iter().zip(received.iter()) {
            result |= a ^ b;
        }
        result == 0
    }

    /// Create a gossipsub topic for this namespace.
    pub fn gossipsub_topic(&self, suffix: &str) -> String {
        format!("{}/{}", self.namespace_str, suffix)
    }

    /// Create the protocol ID for libp2p.
    pub fn protocol_id(&self) -> String {
        format!("/{}", self.namespace_str.replace('/', "-"))
    }
}

impl Default for NetworkNamespace {
    fn default() -> Self {
        Self::default_version(None)
    }
}

/// Validate that a peer belongs to our namespace.
///
/// This should be called during the identify protocol exchange.
pub fn validate_peer_namespace(local: &NetworkNamespace, remote_protocol: &str) -> bool {
    let expected = local.protocol_id();
    remote_protocol.starts_with(&expected)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_psk_computation() {
        let ns1 = NetworkNamespace::new("pmetal/0.1.0", None);
        let ns2 = NetworkNamespace::new("pmetal/0.1.0", None);

        assert_eq!(ns1.psk(), ns2.psk());
    }

    #[test]
    fn test_different_versions() {
        let ns1 = NetworkNamespace::new("pmetal/0.1.0", None);
        let ns2 = NetworkNamespace::new("pmetal/0.2.0", None);

        assert_ne!(ns1.psk(), ns2.psk());
        assert!(!ns1.is_compatible(&ns2));
    }

    #[test]
    fn test_cluster_isolation() {
        let ns1 = NetworkNamespace::new("pmetal/0.1.0", Some("cluster-a"));
        let ns2 = NetworkNamespace::new("pmetal/0.1.0", Some("cluster-b"));

        assert_ne!(ns1.psk(), ns2.psk());
        assert!(!ns1.is_compatible(&ns2));
    }

    #[test]
    fn test_psk_verification() {
        let ns = NetworkNamespace::new("test", None);
        let valid_psk = *ns.psk();
        let mut invalid_psk = valid_psk;
        invalid_psk[0] ^= 0xFF;

        assert!(ns.verify_psk(&valid_psk));
        assert!(!ns.verify_psk(&invalid_psk));
    }

    #[test]
    fn test_gossipsub_topic() {
        let ns = NetworkNamespace::new("pmetal/0.1.0", Some("test"));
        let topic = ns.gossipsub_topic("gradients");

        assert_eq!(topic, "pmetal/0.1.0/test/gradients");
    }

    #[test]
    fn test_protocol_id() {
        let ns = NetworkNamespace::new("pmetal/0.1.0", None);
        let protocol = ns.protocol_id();

        assert!(protocol.starts_with("/pmetal-"));
    }
}
