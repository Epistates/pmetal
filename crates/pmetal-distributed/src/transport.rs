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
use tokio::sync::mpsc;
use tokio::time::timeout;
use tracing::{debug, info, warn};

/// Maximum backoff delay between connection retries.
const MAX_BACKOFF_MS: u64 = 5000;

/// Initial backoff delay between connection retries.
const INITIAL_BACKOFF_MS: u64 = 100;

/// Sender half of the transport.
pub struct TransportSender {
    inner: TransportSenderInner,
}

/// Receiver half of the transport.
pub struct TransportReceiver {
    inner: TransportReceiverInner,
}

enum TransportSenderInner {
    Tcp(OwnedWriteHalf),
    Memory(mpsc::Sender<Vec<u8>>),
}

enum TransportReceiverInner {
    Tcp(OwnedReadHalf),
    Memory(mpsc::Receiver<Vec<u8>>),
}

impl TransportSender {
    /// Send data with length prefix.
    pub async fn send(&mut self, data: &[u8]) -> Result<()> {
        match &mut self.inner {
            TransportSenderInner::Tcp(stream) => {
                let len = (data.len() as u32).to_le_bytes();
                stream.write_all(&len).await?;
                stream.write_all(data).await?;
            }
            TransportSenderInner::Memory(sender) => {
                sender.send(data.to_vec()).await.map_err(|e| {
                    DistributedError::Protocol(format!("in-memory send failed: {e}"))
                })?;
            }
        }
        Ok(())
    }
}

impl TransportReceiver {
    /// Receive data into buffer (must match expected size).
    ///
    /// Enforces a maximum message size of 512 MiB to prevent resource
    /// exhaustion from malicious or corrupted length prefixes.
    ///
    /// Each `read_exact` call is wrapped in a 60-second timeout so that a
    /// peer crash or network partition does not cause indefinite blocking.
    pub async fn recv(&mut self, buffer: &mut [u8]) -> Result<()> {
        const MAX_MSG_BYTES: usize = 512 * 1024 * 1024; // 512 MiB
        const READ_TIMEOUT: Duration = Duration::from_secs(60);

        match &mut self.inner {
            TransportReceiverInner::Tcp(stream) => {
                // Read the 4-byte length prefix with timeout.
                let mut len_buf = [0u8; 4];
                timeout(READ_TIMEOUT, stream.read_exact(&mut len_buf))
                    .await
                    .map_err(|_| {
                        DistributedError::Protocol(
                            "Timed out waiting for message length prefix (60s) — peer may have crashed"
                                .to_string(),
                        )
                    })??;
                let len = u32::from_le_bytes(len_buf) as usize;

                if len > MAX_MSG_BYTES {
                    return Err(DistributedError::Protocol(format!(
                        "Message size {} bytes exceeds maximum {} bytes",
                        len, MAX_MSG_BYTES
                    ))
                    .into());
                }

                if len != buffer.len() {
                    return Err(DistributedError::Protocol(format!(
                        "Expected {} bytes, got {}",
                        buffer.len(),
                        len
                    ))
                    .into());
                }

                timeout(READ_TIMEOUT, stream.read_exact(buffer))
                    .await
                    .map_err(|_| {
                        DistributedError::Protocol(format!(
                            "Timed out waiting for {} bytes of payload (60s) — peer may have crashed",
                            buffer.len()
                        ))
                    })??;
            }
            TransportReceiverInner::Memory(receiver) => {
                let data = timeout(READ_TIMEOUT, receiver.recv())
                    .await
                    .map_err(|_| {
                        DistributedError::Protocol(
                            "Timed out waiting for in-memory payload (60s)".to_string(),
                        )
                    })?
                    .ok_or_else(|| {
                        DistributedError::Protocol("in-memory channel closed".to_string())
                    })?;

                if data.len() > MAX_MSG_BYTES {
                    return Err(DistributedError::Protocol(format!(
                        "Message size {} bytes exceeds maximum {} bytes",
                        data.len(),
                        MAX_MSG_BYTES
                    ))
                    .into());
                }

                if data.len() != buffer.len() {
                    return Err(DistributedError::Protocol(format!(
                        "Expected {} bytes, got {}",
                        buffer.len(),
                        data.len()
                    ))
                    .into());
                }

                buffer.copy_from_slice(&data);
            }
        }

        Ok(())
    }

    /// Receive a dynamically-sized message, returning the data as a `Vec<u8>`.
    ///
    /// Reads the 4-byte length prefix, allocates a buffer, and reads the payload.
    /// This is useful when the caller does not know the message size in advance
    /// (e.g. activation messages with variable tensor shapes).
    pub async fn recv_vec(&mut self) -> Result<Vec<u8>> {
        const MAX_MSG_BYTES: usize = 512 * 1024 * 1024;
        const READ_TIMEOUT: Duration = Duration::from_secs(60);

        match &mut self.inner {
            TransportReceiverInner::Tcp(stream) => {
                let mut len_buf = [0u8; 4];
                timeout(READ_TIMEOUT, stream.read_exact(&mut len_buf))
                    .await
                    .map_err(|_| {
                        DistributedError::Protocol(
                            "Timed out waiting for message length prefix (60s)".to_string(),
                        )
                    })??;
                let len = u32::from_le_bytes(len_buf) as usize;

                if len > MAX_MSG_BYTES {
                    return Err(DistributedError::Protocol(format!(
                        "Message size {} bytes exceeds maximum {} bytes",
                        len, MAX_MSG_BYTES
                    ))
                    .into());
                }

                let mut buf = vec![0u8; len];
                timeout(READ_TIMEOUT, stream.read_exact(&mut buf))
                    .await
                    .map_err(|_| {
                        DistributedError::Protocol(format!(
                            "Timed out waiting for {} bytes of payload (60s)",
                            len
                        ))
                    })??;

                Ok(buf)
            }
            TransportReceiverInner::Memory(receiver) => {
                let data = timeout(READ_TIMEOUT, receiver.recv())
                    .await
                    .map_err(|_| {
                        DistributedError::Protocol(
                            "Timed out waiting for in-memory payload (60s)".to_string(),
                        )
                    })?
                    .ok_or_else(|| {
                        DistributedError::Protocol("in-memory channel closed".to_string())
                    })?;

                if data.len() > MAX_MSG_BYTES {
                    return Err(DistributedError::Protocol(format!(
                        "Message size {} bytes exceeds maximum {} bytes",
                        data.len(),
                        MAX_MSG_BYTES
                    ))
                    .into());
                }

                Ok(data)
            }
        }
    }
}

impl TransportSender {
    fn from_tcp(stream: OwnedWriteHalf) -> Self {
        Self {
            inner: TransportSenderInner::Tcp(stream),
        }
    }

    fn from_memory(sender: mpsc::Sender<Vec<u8>>) -> Self {
        Self {
            inner: TransportSenderInner::Memory(sender),
        }
    }
}

impl TransportReceiver {
    fn from_tcp(stream: OwnedReadHalf) -> Self {
        Self {
            inner: TransportReceiverInner::Tcp(stream),
        }
    }

    fn from_memory(receiver: mpsc::Receiver<Vec<u8>>) -> Self {
        Self {
            inner: TransportReceiverInner::Memory(receiver),
        }
    }
}

/// Create a unidirectional in-memory transport channel.
///
/// This is used for local same-process pipeline stages, such as UltraFusion
/// multi-die execution planning, where TCP framing would add unnecessary cost.
pub fn in_memory_channel(capacity: usize) -> (TransportSender, TransportReceiver) {
    let (tx, rx) = mpsc::channel(capacity.max(1));
    (
        TransportSender::from_memory(tx),
        TransportReceiver::from_memory(rx),
    )
}

/// TCP transport for ring communication.
pub struct TcpTransport;

impl TcpTransport {
    /// Connect to peers in a ring topology.
    ///
    /// Each node connects to its next peer and accepts a connection
    /// from its previous peer, forming a bidirectional ring.
    pub async fn connect(
        config: &DistributedConfig,
    ) -> Result<(TransportSender, TransportReceiver)> {
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
                        let backoff =
                            (INITIAL_BACKOFF_MS * 2u64.pow(retries.min(6))).min(MAX_BACKOFF_MS);
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
                Ok(Err(e)) => Err(DistributedError::Io(e)),
                Err(_) => Err(DistributedError::Protocol(format!(
                    "Timeout waiting for incoming connection ({}ms)",
                    config.connection_timeout_ms
                ))),
            }
        };

        // Run both concurrently - order doesn't matter since it's symmetric
        let (next_peer, prev_peer) = tokio::try_join!(
            async {
                connect_fut
                    .await
                    .map_err(|e: DistributedError| anyhow::anyhow!(e))
            },
            async {
                accept_fut
                    .await
                    .map_err(|e: DistributedError| anyhow::anyhow!(e))
            }
        )?;

        // We send to next_peer, receive from prev_peer
        let (_, write_next) = next_peer.into_split();
        let (read_prev, _) = prev_peer.into_split();

        Ok((
            TransportSender::from_tcp(write_next),
            TransportReceiver::from_tcp(read_prev),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn in_memory_transport_roundtrip() {
        let (mut sender, mut receiver) = in_memory_channel(4);
        sender.send(b"hello").await.expect("send");

        let mut buf = [0u8; 5];
        receiver.recv(&mut buf).await.expect("recv");
        assert_eq!(&buf, b"hello");
    }

    #[tokio::test]
    async fn in_memory_transport_recv_vec_roundtrip() {
        let (mut sender, mut receiver) = in_memory_channel(4);
        sender.send(b"payload").await.expect("send");

        let data = receiver.recv_vec().await.expect("recv_vec");
        assert_eq!(data, b"payload");
    }

    #[tokio::test]
    async fn in_memory_transport_len_mismatch_errors() {
        let (mut sender, mut receiver) = in_memory_channel(4);
        sender.send(b"toolong").await.expect("send");

        let mut buf = [0u8; 3];
        let err = receiver.recv(&mut buf).await.expect_err("length mismatch");
        assert!(err.to_string().contains("Expected 3 bytes, got 7"));
    }
}
