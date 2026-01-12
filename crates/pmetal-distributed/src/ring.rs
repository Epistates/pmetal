use crate::{
    config::DistributedConfig,
    error::DistributedError,
    transport::{TcpTransport, TransportReceiver, TransportSender},
    DistributedBackend,
};
use anyhow::Result;
use async_trait::async_trait;
use bytemuck::cast_slice_mut;
use tokio::sync::Mutex;

pub struct RingBackend {
    rank: usize,
    world_size: usize,
    sender: Mutex<TransportSender>,
    receiver: Mutex<TransportReceiver>,
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

    async fn all_reduce(&self, buffer: &mut [u8]) -> Result<()> {
        // Validate buffer alignment and size for f32 operations
        if !buffer.len().is_multiple_of(4) {
            return Err(DistributedError::Protocol(format!(
                "Buffer length {} is not a multiple of 4 (f32 size)",
                buffer.len()
            ))
            .into());
        }

        if buffer.as_ptr() as usize % std::mem::align_of::<f32>() != 0 {
            return Err(DistributedError::Protocol(
                "Buffer is not properly aligned for f32 operations".to_string(),
            )
            .into());
        }

        // Safe cast using bytemuck (validates alignment at compile time for Pod types)
        let floats: &mut [f32] = cast_slice_mut(buffer);
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
            
            let send_bytes_len = (s_end - s_start) * 4;
            // Create a temporary send buffer to avoid borrow checker issues with `floats`
            // (One future reads floats, the other writes floats).
            // Rust borrow checker will complain if we access `floats` in join! blocks if one is mutable.
            
            let mut send_buf = vec![0u8; send_bytes_len];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    floats[s_start..s_end].as_ptr() as *const u8,
                    send_buf.as_mut_ptr(),
                    send_bytes_len
                );
            }
            
            let recv_bytes_len = (r_end - r_start) * 4;
            let recv_slice = &mut recv_buf[..recv_bytes_len];
            
            // Concurrent Send and Recv
            let send_fut = sender.send(&send_buf);
            let recv_fut = receiver.recv(recv_slice);
            
            tokio::try_join!(send_fut, recv_fut)?;
            
            // Reduce
            let recv_floats = unsafe {
                std::slice::from_raw_parts(recv_slice.as_ptr() as *const f32, r_end - r_start)
            };
            
            for i in 0..recv_floats.len() {
                floats[r_start + i] += recv_floats[i];
            }
            
            send_chunk_idx = recv_chunk_idx;
            recv_chunk_idx = (recv_chunk_idx + self.world_size - 1) % self.world_size;
        }
        
        // 2. All-Gather
        send_chunk_idx = (self.rank + 1) % self.world_size;
        recv_chunk_idx = self.rank;
        
        for _ in 0..self.world_size - 1 {
            let (s_start, s_end) = get_chunk_range(send_chunk_idx);
            let (r_start, r_end) = get_chunk_range(recv_chunk_idx);
            
            let send_bytes_len = (s_end - s_start) * 4;
            let mut send_buf = vec![0u8; send_bytes_len];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    floats[s_start..s_end].as_ptr() as *const u8,
                    send_buf.as_mut_ptr(),
                    send_bytes_len
                );
            }
            
            let recv_bytes_len = (r_end - r_start) * 4;
            let recv_slice = &mut recv_buf[..recv_bytes_len];
            
            let send_fut = sender.send(&send_buf);
            let recv_fut = receiver.recv(recv_slice);
            
            tokio::try_join!(send_fut, recv_fut)?;
            
            // Copy (Gather)
            unsafe {
                std::ptr::copy_nonoverlapping(
                    recv_slice.as_ptr(),
                    floats[r_start..r_end].as_mut_ptr() as *mut u8,
                    recv_bytes_len
                );
            }
            
            send_chunk_idx = recv_chunk_idx;
            recv_chunk_idx = (recv_chunk_idx + self.world_size - 1) % self.world_size;
        }
        
        Ok(())
    }
    
    async fn barrier(&self) -> Result<()> {
        let mut buf = [0u8; 4]; // Minimum 1 float
        self.all_reduce(&mut buf).await
    }
}