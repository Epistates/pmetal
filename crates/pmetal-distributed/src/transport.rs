//! TCP transport layer for distributed training.
//!
//! Provides reliable, ordered delivery over TCP with optimizations
//! for gradient synchronization workloads.

use crate::config::DistributedConfig;
use crate::error::DistributedError;
use anyhow::Result;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::tcp::{OwnedReadHalf, OwnedWriteHalf};
use tokio::net::{TcpListener, TcpStream};
use tokio::time::timeout;
use tracing::{debug, info, warn};

/// Maximum backoff delay between connection retries.
const MAX_BACKOFF_MS: u64 = 5000;

/// Initial backoff delay between connection retries.
const INITIAL_BACKOFF_MS: u64 = 100;

/// Sender half of the transport.
pub struct TransportSender {
    stream: OwnedWriteHalf,
}

/// Receiver half of the transport.
pub struct TransportReceiver {
    stream: OwnedReadHalf,
}

impl TransportSender {
    /// Send data with length prefix.
    pub async fn send(&mut self, data: &[u8]) -> Result<()> {
        let len = (data.len() as u32).to_le_bytes();
        self.stream.write_all(&len).await?;
        self.stream.write_all(data).await?;
        Ok(())
    }
}

impl TransportReceiver {
    /// Receive data into buffer (must match expected size).
    pub async fn recv(&mut self, buffer: &mut [u8]) -> Result<()> {
        let mut len_buf = [0u8; 4];
        self.stream.read_exact(&mut len_buf).await?;
        let len = u32::from_le_bytes(len_buf) as usize;

        if len != buffer.len() {
            return Err(DistributedError::Protocol(format!(
                "Expected {} bytes, got {}",
                buffer.len(),
                len
            ))
            .into());
        }

        self.stream.read_exact(buffer).await?;
        Ok(())
    }
}

/// TCP transport for ring communication.
pub struct TcpTransport;

impl TcpTransport {
    /// Connect to peers in a ring topology.
    ///
    /// Each node connects to its next peer and accepts a connection
    /// from its previous peer, forming a bidirectional ring.
    pub async fn connect(config: &DistributedConfig) -> Result<(TransportSender, TransportReceiver)> {
        let world_size = config.nodes.len();
        let rank = config.rank;
        let my_addr = config.nodes[rank];
        let connection_timeout = Duration::from_millis(config.connection_timeout_ms);
        let max_retries = config.max_retries;

        info!("Node {} listening on {}", rank, my_addr);
        let listener = TcpListener::bind(my_addr).await?;

        let next_rank = (rank + 1) % world_size;
        let next_addr = config.nodes[next_rank];

        // Connect to next peer with exponential backoff
        let connect_fut = async {
            let mut retries = 0u32;
            loop {
                if retries >= max_retries {
                    return Err(DistributedError::MaxRetriesExceeded {
                        addr: next_addr,
                        max_retries,
                    });
                }

                match TcpStream::connect(next_addr).await {
                    Ok(stream) => {
                        // Enable TCP_NODELAY for low-latency gradient exchange
                        if let Err(e) = stream.set_nodelay(true) {
                            warn!("Failed to set TCP_NODELAY: {}", e);
                        }
                        info!("Connected to next peer {} at {}", next_rank, next_addr);
                        return Ok(stream);
                    }
                    Err(e) => {
                        if retries == 0 {
                            debug!("Waiting for peer {} at {} ({})", next_rank, next_addr, e);
                        }
                        retries += 1;

                        // Exponential backoff with jitter
                        let backoff = (INITIAL_BACKOFF_MS * 2u64.pow(retries.min(6)))
                            .min(MAX_BACKOFF_MS);
                        tokio::time::sleep(Duration::from_millis(backoff)).await;
                    }
                }
            }
        };

        // Accept connection from previous peer
        let accept_fut = async {
            match timeout(connection_timeout, listener.accept()).await {
                Ok(Ok((stream, addr))) => {
                    // Enable TCP_NODELAY
                    if let Err(e) = stream.set_nodelay(true) {
                        warn!("Failed to set TCP_NODELAY on incoming: {}", e);
                    }
                    info!("Accepted connection from {}", addr);
                    Ok(stream)
                }
                Ok(Err(e)) => {
                    Err(DistributedError::Io(e))
                }
                Err(_) => {
                    Err(DistributedError::Protocol(format!(
                        "Timeout waiting for incoming connection ({}ms)",
                        config.connection_timeout_ms
                    )))
                }
            }
        };

        // Run both concurrently - order doesn't matter since it's symmetric
        let (next_peer, prev_peer) = tokio::try_join!(
            async { connect_fut.await.map_err(|e: DistributedError| anyhow::anyhow!(e)) },
            async { accept_fut.await.map_err(|e: DistributedError| anyhow::anyhow!(e)) }
        )?;

        // We send to next_peer, receive from prev_peer
        let (_, write_next) = next_peer.into_split();
        let (read_prev, _) = prev_peer.into_split();

        Ok((
            TransportSender { stream: write_next },
            TransportReceiver { stream: read_prev },
        ))
    }
}

#[cfg(test)]
mod tests {
    #[tokio::test]
    async fn test_transport_sender_receiver() {
        // This would require setting up actual TCP connections
        // For unit testing, we'd use mock streams
    }
}
