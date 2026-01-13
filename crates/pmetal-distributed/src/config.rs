use crate::error::DistributedError;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::net::SocketAddr;

/// Configuration for distributed training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedConfig {
    /// List of all nodes in the cluster (IP:Port).
    /// The order must be consistent across all nodes.
    pub nodes: Vec<SocketAddr>,

    /// Rank of this node (index into nodes list).
    pub rank: usize,

    /// Connection timeout in milliseconds (default: 30000).
    #[serde(default = "default_connection_timeout_ms")]
    pub connection_timeout_ms: u64,

    /// Maximum connection retry attempts (default: 50).
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
}

fn default_connection_timeout_ms() -> u64 {
    30000
}

fn default_max_retries() -> u32 {
    50
}

impl DistributedConfig {
    /// Create a new configuration.
    pub fn new(nodes: Vec<SocketAddr>, rank: usize) -> Self {
        Self {
            nodes,
            rank,
            connection_timeout_ms: default_connection_timeout_ms(),
            max_retries: default_max_retries(),
        }
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
