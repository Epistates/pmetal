use crate::{
    DistributedBackend, ReduceOp,
    config::DistributedConfig,
    error::DistributedError,
    transport::{TcpTransport, TransportReceiver, TransportSender},
};
use anyhow::Result;
use async_trait::async_trait;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::Mutex;
use zerocopy::{FromBytes, IntoBytes};

pub struct RingBackend {
    rank: usize,
    world_size: usize,
    sender: Mutex<TransportSender>,
    receiver: Mutex<TransportReceiver>,
    /// Monotonically increasing counter used to assign unique sequence numbers
    /// to barrier rounds, preventing stale tokens from a previous barrier from
    /// being mistaken for tokens from the current one.
    barrier_counter: AtomicU64,
}

impl RingBackend {
    pub async fn new(config: DistributedConfig) -> Result<Self> {
        config.validate()?;
        let (sender, receiver) = TcpTransport::connect(&config).await?;
        Ok(Self {
            rank: config.rank,
            world_size: config.nodes.len(),
            sender: Mutex::new(sender),
            receiver: Mutex::new(receiver),
            barrier_counter: AtomicU64::new(0),
        })
    }
}

#[async_trait]
impl DistributedBackend for RingBackend {
    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.world_size
    }

    async fn all_reduce(&self, buffer: &mut [u8], op: ReduceOp) -> Result<()> {
        // Validate buffer alignment and size for f32 operations
        if !buffer.len().is_multiple_of(4) {
            return Err(DistributedError::Protocol(format!(
                "Buffer length {} is not a multiple of 4 (f32 size)",
                buffer.len()
            ))
            .into());
        }

        if !(buffer.as_ptr() as usize).is_multiple_of(std::mem::align_of::<f32>()) {
            return Err(DistributedError::Protocol(
                "Buffer is not properly aligned for f32 operations".to_string(),
            )
            .into());
        }

        let floats: &mut [f32] = <[f32]>::mut_from_bytes(buffer)
            .map_err(|e| DistributedError::Protocol(format!("Buffer cast failed: {e}")))?;
        let len = floats.len();

        let chunk_size = len / self.world_size;
        let remainder = len % self.world_size;

        let get_chunk_range = |idx: usize| -> (usize, usize) {
            let start = idx * chunk_size + idx.min(remainder);
            let end = start + chunk_size + (if idx < remainder { 1 } else { 0 });
            (start, end)
        };

        // Lock both transport halves
        let mut sender = self.sender.lock().await;
        let mut receiver = self.receiver.lock().await;

        // 1. Scatter-Reduce
        let mut send_chunk_idx = self.rank;
        let mut recv_chunk_idx = (self.rank + self.world_size - 1) % self.world_size;

        // Recv buffer
        let max_chunk_size = chunk_size + 1;
        let mut recv_buf = vec![0u8; max_chunk_size * 4];

        for _ in 0..self.world_size - 1 {
            let (s_start, s_end) = get_chunk_range(send_chunk_idx);
            let (r_start, r_end) = get_chunk_range(recv_chunk_idx);

            // Prepare send data
            // We need to copy to a temp buffer because floats is borrowed by recv logic?
            // Actually, we can just send. send takes &[u8].
            // BUT: We need to use &mut sender and &mut receiver concurrently.
            // Since we locked them, we own them. We can pass them to concurrent futures.
            // But MutexGuard is not Send if we hold it across await? No, it is.
            // Wait, we can't borrow `sender` and `receiver` into `try_join` if they are MutexGuards?
            // Yes we can, they are distinct.

            let send_buf = floats[s_start..s_end].as_bytes().to_vec();

            let recv_bytes_len = (r_end - r_start) * 4;
            let recv_slice = &mut recv_buf[..recv_bytes_len];

            // Concurrent Send and Recv with timeout to prevent deadlock
            // if a peer crashes mid-transfer.
            tokio::time::timeout(std::time::Duration::from_secs(30), async {
                let send_fut = sender.send(&send_buf);
                let recv_fut = receiver.recv(recv_slice);
                tokio::try_join!(send_fut, recv_fut)
            })
            .await
            .map_err(|_| {
                anyhow::anyhow!(
                    "Ring all-reduce scatter-reduce timed out after 30s — peer may have crashed"
                )
            })??;

            // Reduce
            let recv_floats =
                <[f32]>::ref_from_bytes(recv_slice).expect("recv buffer aligned for f32");

            for i in 0..recv_floats.len() {
                floats[r_start + i] += recv_floats[i];
            }

            send_chunk_idx = recv_chunk_idx;
            recv_chunk_idx = (recv_chunk_idx + self.world_size - 1) % self.world_size;
        }

        // 2. All-Gather — each node sends its own fully-reduced chunk rightward.
        // After scatter-reduce, node r holds fully-reduced chunk r.
        // Step 0: send chunk r (own), receive chunk (r-1) from left neighbor.
        // Step k: send what was received in step k-1.
        send_chunk_idx = self.rank;
        recv_chunk_idx = (self.rank + self.world_size - 1) % self.world_size;

        for _ in 0..self.world_size - 1 {
            let (s_start, s_end) = get_chunk_range(send_chunk_idx);
            let (r_start, r_end) = get_chunk_range(recv_chunk_idx);

            let send_buf = floats[s_start..s_end].as_bytes().to_vec();

            let recv_bytes_len = (r_end - r_start) * 4;
            let recv_slice = &mut recv_buf[..recv_bytes_len];

            // Timeout mirrors the scatter-reduce phase.
            tokio::time::timeout(std::time::Duration::from_secs(30), async {
                let send_fut = sender.send(&send_buf);
                let recv_fut = receiver.recv(recv_slice);
                tokio::try_join!(send_fut, recv_fut)
            })
            .await
            .map_err(|_| {
                anyhow::anyhow!(
                    "Ring all-reduce all-gather timed out after 30s — peer may have crashed"
                )
            })??;

            // Copy (Gather)
            let recv_floats =
                <[f32]>::ref_from_bytes(recv_slice).expect("recv buffer aligned for f32");
            floats[r_start..r_end].copy_from_slice(recv_floats);

            send_chunk_idx = recv_chunk_idx;
            recv_chunk_idx = (recv_chunk_idx + self.world_size - 1) % self.world_size;
        }

        // Apply mean reduction: divide by world_size after the ring has summed.
        if op == ReduceOp::Mean {
            let divisor = self.world_size as f32;
            for f in floats.iter_mut() {
                *f /= divisor;
            }
        }

        Ok(())
    }

    /// Two-phase barrier using a monotonic sequence number.
    ///
    /// Phase 1 (propagate): send the barrier token with a unique sequence
    /// number around the ring; each node forwards it after receiving from its
    /// predecessor.  When the token reaches the initiator after `world_size - 1`
    /// hops, every node has observed it.
    ///
    /// Phase 2 (acknowledge): send the sequence number back around the ring
    /// in the same direction to signal completion.  When a node receives the
    /// acknowledgement it knows all nodes have finished Phase 1 and may proceed.
    ///
    /// The monotonic counter prevents tokens from a crashed/slow previous round
    /// from being mistaken for tokens from the current round.
    async fn barrier(&self) -> Result<()> {
        let world_size = self.world_size;
        if world_size < 2 {
            return Ok(());
        }

        // Allocate a fresh sequence number for this barrier invocation.
        let seq = self.barrier_counter.fetch_add(1, Ordering::SeqCst);

        let mut sender = self.sender.lock().await;
        let mut receiver = self.receiver.lock().await;

        // Each barrier token is 8 bytes: the little-endian u64 sequence number.
        let token: [u8; 8] = seq.to_le_bytes();

        // Phase 1: propagate the token around the ring.
        // Each node forwards after receiving (all world_size - 1 hops).
        for _ in 0..world_size - 1 {
            let mut recv_buf = [0u8; 8];
            tokio::time::timeout(std::time::Duration::from_secs(30), async {
                tokio::try_join!(sender.send(&token), receiver.recv(&mut recv_buf))
            })
            .await
            .map_err(|_| {
                anyhow::anyhow!("Barrier phase-1 timed out after 30s — peer may have crashed")
            })??;

            // Verify the token sequence number to detect stale messages.
            let recv_seq = u64::from_le_bytes(recv_buf);
            if recv_seq != seq {
                return Err(DistributedError::Protocol(format!(
                    "Barrier sequence mismatch: expected {seq}, got {recv_seq}"
                ))
                .into());
            }
        }

        // Phase 2: acknowledge completion.
        let ack_seq = seq.wrapping_add(u64::MAX / 2); // distinct from seq
        let ack_token: [u8; 8] = ack_seq.to_le_bytes();

        for _ in 0..world_size - 1 {
            let mut recv_buf = [0u8; 8];
            tokio::time::timeout(std::time::Duration::from_secs(30), async {
                tokio::try_join!(sender.send(&ack_token), receiver.recv(&mut recv_buf))
            })
            .await
            .map_err(|_| {
                anyhow::anyhow!("Barrier phase-2 timed out after 30s — peer may have crashed")
            })??;
        }

        Ok(())
    }
}

#[cfg(kani)]
mod verification {
    use super::*;

    #[kani::proof]
    #[kani::unwind(17)] // Sufficient for world_size up to 16 for testing
    fn verify_get_chunk_range() {
        let len: usize = kani::any();
        let world_size: usize = kani::any();

        // Preconditions
        kani::assume(world_size > 0 && world_size <= 16);
        kani::assume(len >= world_size && len < 1024);

        let chunk_size = len / world_size;
        let remainder = len % world_size;

        let get_chunk_range = |idx: usize| -> (usize, usize) {
            let start = idx * chunk_size + idx.min(remainder);
            let end = start + chunk_size + (if idx < remainder { 1 } else { 0 });
            (start, end)
        };

        let mut total_elements = 0;
        let mut last_end = 0;

        for i in 0..world_size {
            let (start, end) = get_chunk_range(i);

            // Chunks must be valid ranges
            assert!(start <= end);
            // Chunks must be contiguous
            assert!(start == last_end);
            // Chunks must be within bounds
            assert!(end <= len);

            total_elements += end - start;
            last_end = end;
        }

        // Total elements must match original length
        assert!(total_elements == len);
        assert!(last_end == len);
    }
}
