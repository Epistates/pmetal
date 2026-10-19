//! Bridge between `pmetal-distributed` and the training loop.
//!
//! Provides gradient synchronization, loss reduction, and barrier operations
//! that integrate with the existing `FlattenedModuleParam` gradient format.

use pmetal_runtime::compat::{Array, Dtype, Exception, module::FlattenedModuleParam};
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

/// Wait for `future` from synchronous training code, whether that code runs on
/// a plain thread inside a runtime's context or on one of the runtime's own
/// workers (the orchestrator is an `async fn` that trains without yielding).
/// `Handle::block_on` alone panics on a worker ("cannot start a runtime from
/// within a runtime"), which is where `pmetal train` calls it from.
pub(crate) fn block_on<F: std::future::Future>(future: F) -> F::Output {
    let handle = tokio::runtime::Handle::current();
    match handle.runtime_flavor() {
        tokio::runtime::RuntimeFlavor::MultiThread => {
            tokio::task::block_in_place(|| handle.block_on(future))
        }
        _ => handle.block_on(future),
    }
}

/// How the ranks exchange gradients.
enum Transport {
    /// The TCP ring of `pmetal-distributed`, over host `f32` buffers: what
    /// peers and mDNS discovery set up, and the only one that compresses.
    Ring(Arc<DistributedContext>),
    /// MLX's own collectives (ring over TCP, or JACCL over Thunderbolt RDMA),
    /// for a process launched into an MLX distributed group. The gradients
    /// stay on the device and the collective is part of the step's graph.
    Mlx(pmetal_runtime::distributed::Group),
}

/// Gradient synchronization bridge for distributed training.
///
/// The ring transport converts between `FlattenedModuleParam` (MLX gradient
/// format) and the flat `f32` buffers `DistributedContext::all_reduce` takes;
/// the MLX transport averages the gradients on the device
/// ([`pmetal_runtime::distributed::average_gradients`]).
pub struct DistributedGradientSync {
    transport: Transport,
    compressor: Option<GradientCompressor>,
    /// Reusable scratch buffer for flatten/scatter operations.
    buffer: Vec<f32>,
    /// Parameter layout: (name, shape, element_count) for deterministic ordering.
    param_layout: Vec<(String, Vec<i32>, usize)>,
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
            // Each rank draws its own random masks; ranks sharing one would
            // all send the same coordinates.
            Some(GradientCompressor::with_seed(
                strategy,
                config.error_feedback,
                ctx.rank() as u64,
            ))
        };

        Self {
            transport: Transport::Ring(ctx),
            compressor,
            buffer: Vec::new(),
            param_layout: Vec::new(),
            layout_initialized: false,
        }
    }

    /// The bridge for a process MLX launched into a distributed group of more
    /// than one rank (`mlx.launch`, or `MLX_RANK` and `MLX_HOSTFILE` set by
    /// hand), or `None` when there is no such group.
    ///
    /// Gradients are averaged on the device. Compression needs the ring
    /// transport's host buffers, so a `config` that asks for it is told it is
    /// not applied.
    pub fn from_mlx_group(config: &pmetal_core::DistributedTrainingConfig) -> Result<Option<Self>> {
        if !pmetal_runtime::distributed::is_available() {
            return Ok(None);
        }
        let group = pmetal_runtime::distributed::Group::init(false)
            .map_err(|e| SftError::Mlx(Exception::custom(e.to_string())))?;
        if group.size() <= 1 {
            return Ok(None);
        }
        if config.compression != pmetal_core::DistributedCompression::None {
            tracing::warn!(
                "Gradient compression is not applied on MLX's collectives; gradients are \
                 averaged uncompressed"
            );
        }
        Ok(Some(Self {
            transport: Transport::Mlx(group),
            compressor: None,
            buffer: Vec::new(),
            param_layout: Vec::new(),
            layout_initialized: false,
        }))
    }

    /// Get the rank of this node.
    pub fn rank(&self) -> usize {
        match &self.transport {
            Transport::Ring(ctx) => ctx.rank(),
            Transport::Mlx(group) => group.rank() as usize,
        }
    }

    /// Get the total number of nodes.
    pub fn world_size(&self) -> usize {
        match &self.transport {
            Transport::Ring(ctx) => ctx.world_size(),
            Transport::Mlx(group) => group.size() as usize,
        }
    }

    /// Whether this is the master node (rank 0).
    pub fn is_master(&self) -> bool {
        self.rank() == 0
    }

    fn ring(&self) -> &Arc<DistributedContext> {
        match &self.transport {
            Transport::Ring(ctx) => ctx,
            Transport::Mlx(_) => unreachable!("the ring path is only taken on the ring transport"),
        }
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
                (name.to_string(), shape, count)
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
            self.world_size(),
        );
    }

    /// Flatten gradients from `FlattenedModuleParam` into a contiguous f32 buffer.
    fn flatten_grads(&mut self, grads: &FlattenedModuleParam) -> Result<()> {
        let mut offset = 0;
        for (name, _shape, count) in &self.param_layout {
            if let Some(arr) = grads.get(name.as_str()) {
                // Evaluate the gradient array to materialized f32 values
                let arr_eval = arr.clone();
                arr_eval
                    .try_eval()
                    .map_err(|e| SftError::Mlx(Exception::custom(e.to_string())))?;
                let arr_f32 = arr_eval.as_dtype(Dtype::Float32.as_i32());
                arr_f32
                    .try_eval()
                    .map_err(|e| SftError::Mlx(Exception::custom(e.to_string())))?;

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
            grads.insert(std::rc::Rc::from(name.as_str()), arr);
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
        if let Transport::Mlx(group) = &self.transport {
            // On the device, lazily: the collective runs when the step is
            // evaluated, with the gradients still where they were computed.
            return pmetal_runtime::distributed::average_gradients(
                grads,
                Some(group),
                pmetal_runtime::distributed::DEFAULT_BUCKET_BYTES,
            )
            .map_err(|e| {
                SftError::Mlx(Exception::custom(format!("gradient average failed: {e}")))
            });
        }
        if !self.layout_initialized {
            self.init_layout(grads);
        }

        // Flatten MLX Arrays into contiguous f32 buffer
        self.flatten_grads(grads)?;

        // All-reduce (mean), compressed when a strategy is configured. The
        // compressor's error feedback is for exactly this buffer.
        let ctx = Arc::clone(self.ring());
        let reduced = match self.compressor {
            Some(ref mut compressor) => {
                ctx.all_reduce_compressed(compressor, &mut self.buffer, ReduceOp::Mean)
                    .await
            }
            None => ctx.all_reduce_f32(&mut self.buffer, ReduceOp::Mean).await,
        };
        reduced.map_err(|e| {
            SftError::Mlx(Exception::custom(format!(
                "distributed all_reduce failed: {e}"
            )))
        })?;

        // Scatter back into FlattenedModuleParam
        self.scatter_grads(grads)?;

        Ok(())
    }

    /// Reduce a scalar loss across all nodes (mean).
    pub async fn sync_loss(&self, loss: f32) -> Result<f32> {
        if let Transport::Mlx(group) = &self.transport {
            let total = pmetal_runtime::distributed::all_sum(&Array::from_f32(loss), Some(group))
                .map_err(|e| {
                SftError::Mlx(Exception::custom(format!("loss sync failed: {e}")))
            })?;
            let mean = total.mul_scalar(1.0 / group.size() as f32);
            mean.try_eval()
                .map_err(|e| SftError::Mlx(Exception::custom(e.to_string())))?;
            return Ok(mean.item_f32());
        }
        let mut value = [loss];
        self.ring()
            .all_reduce_f32(&mut value, ReduceOp::Mean)
            .await
            .map_err(|e| {
                SftError::Mlx(Exception::custom(format!(
                    "distributed loss sync failed: {e}"
                )))
            })?;
        Ok(value[0])
    }

    /// Barrier synchronization — all nodes must reach this point before any proceed.
    pub async fn barrier(&self) -> Result<()> {
        if let Transport::Mlx(group) = &self.transport {
            let token = pmetal_runtime::distributed::all_sum(&Array::from_f32(0.0), Some(group))
                .map_err(|e| SftError::Mlx(Exception::custom(format!("barrier failed: {e}"))))?;
            return token
                .try_eval()
                .map_err(|e| SftError::Mlx(Exception::custom(e.to_string())));
        }
        self.ring().barrier().await.map_err(|e| {
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
        let auto_config = pmetal_distributed::AutoDiscoveryConfig {
            gradient_port: config.gradient_port,
            ..Default::default()
        };

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

#[cfg(test)]
mod tests {
    use super::block_on;

    /// Training code that runs on a runtime's own worker (as the orchestrator's
    /// does, being an `async fn` that never yields) can still wait on a future.
    #[test]
    fn a_runtime_worker_can_wait_on_a_future() {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        let answer = runtime.block_on(async { block_on(async { 6 * 7 }) });
        assert_eq!(answer, 42);
        // What the loop called before: `Handle::block_on` on a worker panics.
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            runtime.block_on(async {
                tokio::spawn(async { tokio::runtime::Handle::current().block_on(async { 1 }) })
                    .await
            })
        }));
        assert!(
            matches!(panicked, Ok(Err(_))),
            "Handle::block_on on a worker should panic the task"
        );
        // And so can a plain thread inside the runtime's context.
        let _inside = runtime.enter();
        assert_eq!(block_on(async { 1 + 1 }), 2);
    }
}
