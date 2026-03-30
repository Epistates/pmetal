//! Bridge between `pmetal-distributed` and the training loop.
//!
//! Provides gradient synchronization, loss reduction, and barrier operations
//! that integrate with the existing `FlattenedModuleParam` gradient format.

use pmetal_bridge::compat::{Array, Dtype, Exception, module::FlattenedModuleParam};
use std::rc::Rc;
use std::sync::Arc;

use pmetal_distributed::{CompressionStrategy, DistributedContext, GradientCompressor, ReduceOp};

use crate::{Result, SftError};

/// Maps `DistributedCompression` config enum to the distributed crate's `CompressionStrategy`.
fn map_compression(
    compression: pmetal_core::DistributedCompression,
    topk_ratio: f32,
) -> CompressionStrategy {
    match compression {
        pmetal_core::DistributedCompression::None => CompressionStrategy::None,
        pmetal_core::DistributedCompression::TopK => {
            CompressionStrategy::TopK { ratio: topk_ratio }
        }
        pmetal_core::DistributedCompression::Fp16 => {
            CompressionStrategy::Quantize(pmetal_distributed::QuantizationType::FP16)
        }
        pmetal_core::DistributedCompression::Random => CompressionStrategy::Random {
            probability: topk_ratio,
        },
    }
}

/// Gradient synchronization bridge for distributed training.
///
/// Handles the conversion between `FlattenedModuleParam` (MLX gradient format) and
/// the flat `f32` buffers expected by `DistributedContext::all_reduce`.
pub struct DistributedGradientSync {
    ctx: Arc<DistributedContext>,
    compressor: Option<GradientCompressor>,
    /// Reusable scratch buffer for flatten/scatter operations.
    buffer: Vec<f32>,
    /// Parameter layout: (name, shape, element_count) for deterministic ordering.
    param_layout: Vec<(Rc<str>, Vec<i32>, usize)>,
    /// Whether layout has been initialized from the first gradient set.
    layout_initialized: bool,
}

impl DistributedGradientSync {
    /// Create a new gradient sync bridge.
    pub fn new(
        ctx: Arc<DistributedContext>,
        config: &pmetal_core::DistributedTrainingConfig,
    ) -> Self {
        let strategy = map_compression(config.compression, config.topk_ratio);
        let compressor = if matches!(strategy, CompressionStrategy::None) {
            None
        } else {
            Some(GradientCompressor::new(strategy, config.error_feedback))
        };

        Self {
            ctx,
            compressor,
            buffer: Vec::new(),
            param_layout: Vec::new(),
            layout_initialized: false,
        }
    }

    /// Get the rank of this node.
    pub fn rank(&self) -> usize {
        self.ctx.rank()
    }

    /// Get the total number of nodes.
    pub fn world_size(&self) -> usize {
        self.ctx.world_size()
    }

    /// Whether this is the master node (rank 0).
    pub fn is_master(&self) -> bool {
        self.ctx.is_master()
    }

    /// Initialize the parameter layout from a gradient set.
    ///
    /// This must be called once before `sync_gradients`. The layout determines
    /// the order in which parameters are flattened into the buffer. All nodes
    /// must have the same layout for all-reduce to produce correct results.
    fn init_layout(&mut self, grads: &FlattenedModuleParam) {
        // Sort by parameter name for deterministic ordering across nodes
        let mut entries: Vec<_> = grads
            .iter()
            .map(|(name, arr)| {
                let shape: Vec<i32> = arr.shape().to_vec();
                let count: usize = shape.iter().map(|&d| d as usize).product();
                (name.clone(), shape, count)
            })
            .collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));

        let total_elements: usize = entries.iter().map(|e| e.2).sum();
        self.buffer.resize(total_elements, 0.0);
        self.param_layout = entries;
        self.layout_initialized = true;

        tracing::info!(
            "Distributed gradient sync initialized: {} params, {} elements ({:.1} MB), world_size={}",
            self.param_layout.len(),
            total_elements,
            (total_elements * 4) as f64 / 1_048_576.0,
            self.ctx.world_size(),
        );
    }

    /// Flatten gradients from `FlattenedModuleParam` into a contiguous f32 buffer.
    fn flatten_grads(&mut self, grads: &FlattenedModuleParam) -> Result<()> {
        let mut offset = 0;
        for (name, _shape, count) in &self.param_layout {
            if let Some(arr) = grads.get(name) {
                // Evaluate the gradient array to materialized f32 values
                let mut arr_eval = arr.clone();
                arr_eval.eval();
                let mut arr_f32 = arr_eval.as_dtype(Dtype::Float32.as_i32());
                arr_f32.eval();

                // Copy data from MLX Array into our flat buffer
                let data = arr_f32.as_slice::<f32>();
                let n = *count;
                self.buffer[offset..offset + n].copy_from_slice(&data[..n]);
                offset += n;
            } else {
                // Parameter not in this gradient set — fill with zeros
                let n = *count;
                self.buffer[offset..offset + n].fill(0.0);
                offset += n;
            }
        }
        Ok(())
    }

    /// Scatter the synchronized buffer back into `FlattenedModuleParam`.
    fn scatter_grads(&self, grads: &mut FlattenedModuleParam) -> Result<()> {
        let mut offset = 0;
        for (name, shape, count) in &self.param_layout {
            let n = *count;
            let slice = &self.buffer[offset..offset + n];
            let arr = Array::from_f32_slice(slice, shape.as_slice());
            grads.insert(name.clone(), arr);
            offset += n;
        }
        Ok(())
    }

    /// Synchronize gradients across all nodes using all-reduce (mean).
    ///
    /// This is the main entry point called from the training loop after
    /// gradient clipping and before optimizer update.
    ///
    /// Flow: eval grads → flatten to f32 → all_reduce(Mean) → scatter back
    pub async fn sync_gradients(&mut self, grads: &mut FlattenedModuleParam) -> Result<()> {
        if !self.layout_initialized {
            self.init_layout(grads);
        }

        // Flatten MLX Arrays into contiguous f32 buffer
        self.flatten_grads(grads)?;

        // All-reduce with optional compression
        if let Some(ref mut compressor) = self.compressor {
            // Compressed path: compress → serialize → all_reduce → deserialize → decompress
            let compressed = compressor.compress(&self.buffer);
            let mut serialized = pmetal_distributed::compression::serialize_compressed(&compressed);
            self.ctx
                .all_reduce(&mut serialized, ReduceOp::Mean)
                .await
                .map_err(|e| {
                    SftError::Mlx(Exception::custom(format!(
                        "distributed all_reduce failed: {e}"
                    )))
                })?;
            if let Some(decompressed) =
                pmetal_distributed::compression::deserialize_compressed(&serialized)
            {
                let result = compressor.decompress(&decompressed);
                self.buffer[..result.len()].copy_from_slice(&result);
            }
        } else {
            // Uncompressed path: reinterpret the f32 Vec as a &mut [u8] slice.
            // Vec<f32> is guaranteed to have alignment >= 4, satisfying the
            // ring backend's alignment check for f32 operations.
            let len = self.buffer.len();
            let byte_len = len * 4;
            let byte_ptr = self.buffer.as_mut_ptr().cast::<u8>();
            // SAFETY: Vec<f32> guarantees 4-byte alignment and contiguous layout.
            // The slice borrows self.buffer mutably for the duration of all_reduce.
            // No other access to self.buffer occurs during this call.
            #[allow(unsafe_code)]
            let byte_buf = unsafe { std::slice::from_raw_parts_mut(byte_ptr, byte_len) };
            self.ctx
                .all_reduce(byte_buf, ReduceOp::Mean)
                .await
                .map_err(|e| {
                    SftError::Mlx(Exception::custom(format!(
                        "distributed all_reduce failed: {e}"
                    )))
                })?;
        }

        // Scatter back into FlattenedModuleParam
        self.scatter_grads(grads)?;

        Ok(())
    }

    /// Reduce a scalar loss across all nodes (mean).
    pub async fn sync_loss(&self, loss: f32) -> Result<f32> {
        // Use a Vec<f32> for guaranteed 4-byte alignment (ring backend requires it).
        let mut aligned = vec![loss];
        let byte_len = 4;
        let byte_ptr = aligned.as_mut_ptr().cast::<u8>();
        #[allow(unsafe_code)]
        let buf = unsafe { std::slice::from_raw_parts_mut(byte_ptr, byte_len) };
        self.ctx
            .all_reduce(buf, ReduceOp::Mean)
            .await
            .map_err(|e| {
                SftError::Mlx(Exception::custom(format!(
                    "distributed loss sync failed: {e}"
                )))
            })?;
        Ok(aligned[0])
    }

    /// Barrier synchronization — all nodes must reach this point before any proceed.
    pub async fn barrier(&self) -> Result<()> {
        self.ctx.barrier().await.map_err(|e| {
            SftError::Mlx(Exception::custom(format!(
                "distributed barrier failed: {e}"
            )))
        })?;
        Ok(())
    }
}

/// Create a `DistributedContext` from a `DistributedTrainingConfig`.
///
/// Returns `None` if no distributed config is provided or if world_size would be 1.
pub async fn create_distributed_context(
    config: &pmetal_core::DistributedTrainingConfig,
) -> Result<Arc<DistributedContext>> {
    use pmetal_distributed::metrics::new_shared_metrics;

    let metrics = new_shared_metrics();

    if config.auto_discover {
        // Zero-config mDNS discovery
        let mut auto_config = pmetal_distributed::AutoDiscoveryConfig::default();
        auto_config.gradient_port = config.gradient_port;

        let backend = pmetal_distributed::AutoDiscoveryBackend::with_config(auto_config)
            .await
            .map_err(|e| SftError::Mlx(Exception::custom(format!("auto-discovery failed: {e}"))))?;

        tracing::info!("Waiting for peers via mDNS auto-discovery...");
        let peer_count = backend
            .wait_for_peers(1, std::time::Duration::from_secs(60))
            .await
            .map_err(|e| {
                SftError::Mlx(Exception::custom(format!("peer discovery timeout: {e}")))
            })?;
        tracing::info!("Found {} peers, establishing ring...", peer_count);

        backend.establish_ring().await.map_err(|e| {
            SftError::Mlx(Exception::custom(format!("ring establishment failed: {e}")))
        })?;

        Ok(Arc::new(DistributedContext::with_metrics(
            Box::new(backend),
            metrics,
        )))
    } else if !config.peers.is_empty() {
        // Manual peer list
        let nodes: Vec<std::net::SocketAddr> = config
            .peers
            .iter()
            .map(|s| {
                s.parse().map_err(|e| {
                    SftError::Mlx(Exception::custom(format!(
                        "invalid peer address '{s}': {e}"
                    )))
                })
            })
            .collect::<Result<Vec<_>>>()?;

        // Determine rank from environment or position
        let rank = std::env::var("PMETAL_RANK")
            .ok()
            .and_then(|r| r.parse().ok())
            .unwrap_or(0);

        let dist_config = pmetal_distributed::DistributedConfig::new(nodes, rank);
        let backend = pmetal_distributed::RingBackend::new(dist_config)
            .await
            .map_err(|e| SftError::Mlx(Exception::custom(format!("ring backend failed: {e}"))))?;

        Ok(Arc::new(DistributedContext::with_metrics(
            Box::new(backend),
            metrics,
        )))
    } else {
        Err(SftError::Mlx(Exception::custom(
            "distributed config has no peers and auto_discover is false",
        )))
    }
}
