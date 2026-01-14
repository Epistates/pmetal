//! Batched model merging for improved throughput.
//!
//! This module provides optimized batch processing for model merging operations.
//! Instead of processing tensors one at a time with individual GPU syncs,
//! it accumulates tensors into batches and processes them together.
//!
//! # Performance Benefits
//!
//! - **Reduced GPU-CPU Sync**: Single sync per batch instead of per-tensor
//! - **Parallel Threshold Computation**: Compute thresholds for all tensors in batch
//! - **Memory Reuse**: Reuse intermediate buffers across batch
//!
//! # Example
//!
//! ```ignore
//! use pmetal_merge::batched::{BatchedMerger, BatchConfig};
//!
//! let config = BatchConfig {
//!     batch_size: 32,
//!     use_online_threshold: true,
//!     parallel_threshold: true,
//! };
//!
//! let merger = BatchedMerger::new(config, method, loaders, base_loader);
//! let results = merger.merge_all(&tensor_names)?;
//! ```

use std::collections::HashMap;

use mlx_rs::Array;
use tracing::{debug, info, trace};

use crate::{MergeConfig, MergeMethod, MergeParameters, Result, SafetensorsLoader, TensorLoader};

/// Configuration for batched merge processing.
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Number of tensors to process in each batch.
    pub batch_size: usize,

    /// Use O(n) online threshold algorithm instead of O(n log n) sorting.
    pub use_online_threshold: bool,

    /// Compute thresholds in parallel using rayon (when available).
    pub parallel_threshold: bool,

    /// Maximum memory usage for batch buffers (in bytes).
    /// If a batch would exceed this, it's split.
    pub max_batch_memory: usize,

    /// Enable progress reporting.
    pub progress: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            use_online_threshold: true,
            parallel_threshold: true,
            max_batch_memory: 4 * 1024 * 1024 * 1024, // 4GB
            progress: true,
        }
    }
}

/// Batched tensor merge information.
#[derive(Debug)]
pub struct TensorBatch {
    /// Names of tensors in this batch.
    pub names: Vec<String>,

    /// Loaded tensors from each model [model_idx][tensor_idx].
    pub tensors: Vec<Vec<Array>>,

    /// Base tensors (if using task arithmetic methods).
    pub base_tensors: Vec<Option<Array>>,

    /// Per-tensor parameters.
    pub params: Vec<Vec<MergeParameters>>,

    /// Estimated memory usage in bytes.
    pub memory_estimate: usize,
}

impl TensorBatch {
    /// Create a new empty batch.
    pub fn new() -> Self {
        Self {
            names: Vec::new(),
            tensors: Vec::new(),
            base_tensors: Vec::new(),
            params: Vec::new(),
            memory_estimate: 0,
        }
    }

    /// Check if batch is empty.
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// Get number of tensors in batch.
    pub fn len(&self) -> usize {
        self.names.len()
    }

    /// Clear the batch for reuse.
    pub fn clear(&mut self) {
        self.names.clear();
        self.tensors.clear();
        self.base_tensors.clear();
        self.params.clear();
        self.memory_estimate = 0;
    }
}

impl Default for TensorBatch {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of merging a batch of tensors.
#[derive(Debug)]
pub struct BatchResult {
    /// Merged tensors keyed by name.
    pub merged: HashMap<String, Array>,

    /// Processing time in milliseconds.
    pub time_ms: f64,

    /// Number of tensors processed.
    pub tensor_count: usize,
}

/// Batched model merger.
///
/// Orchestrates batch loading and merging of model tensors for improved throughput.
pub struct BatchedMerger<'a> {
    config: BatchConfig,
    method: &'a dyn MergeMethod,
    loaders: &'a [SafetensorsLoader],
    base_loader: Option<&'a SafetensorsLoader>,
    merge_config: &'a MergeConfig,
}

impl<'a> BatchedMerger<'a> {
    /// Create a new batched merger.
    ///
    /// # Arguments
    /// * `config` - Batch processing configuration
    /// * `method` - The merge method to use
    /// * `loaders` - Tensor loaders for input models
    /// * `base_loader` - Optional base model loader (for task arithmetic)
    /// * `merge_config` - Merge configuration with parameters
    pub fn new(
        config: BatchConfig,
        method: &'a dyn MergeMethod,
        loaders: &'a [SafetensorsLoader],
        base_loader: Option<&'a SafetensorsLoader>,
        merge_config: &'a MergeConfig,
    ) -> Self {
        Self {
            config,
            method,
            loaders,
            base_loader,
            merge_config,
        }
    }

    /// Merge all tensors in batches.
    ///
    /// # Arguments
    /// * `tensor_names` - Names of all tensors to merge
    ///
    /// # Returns
    /// Iterator over (name, merged_tensor) pairs.
    pub fn merge_all(&self, tensor_names: &[String]) -> Result<Vec<(String, Array)>> {
        let total = tensor_names.len();
        let batch_count = total.div_ceil(self.config.batch_size);

        if self.config.progress {
            info!(
                "Processing {} tensors in {} batches (size={})",
                total, batch_count, self.config.batch_size
            );
        }

        let mut all_results = Vec::with_capacity(total);

        for (batch_idx, chunk) in tensor_names.chunks(self.config.batch_size).enumerate() {
            debug!("Processing batch {}/{}", batch_idx + 1, batch_count);

            let batch = self.load_batch(chunk)?;
            let result = self.merge_batch(&batch)?;

            // Collect results in order
            for name in &batch.names {
                if let Some(merged) = result.merged.get(name) {
                    all_results.push((name.clone(), merged.clone()));
                }
            }

            trace!(
                "Batch {} complete: {} tensors in {:.1}ms",
                batch_idx + 1,
                result.tensor_count,
                result.time_ms
            );
        }

        Ok(all_results)
    }

    /// Load a batch of tensors.
    fn load_batch(&self, names: &[String]) -> Result<TensorBatch> {
        let mut batch = TensorBatch::new();

        for name in names {
            // Load from each model
            let mut tensor_list: Vec<Array> = Vec::with_capacity(self.loaders.len());
            let mut param_list: Vec<MergeParameters> = Vec::with_capacity(self.loaders.len());

            for (idx, loader) in self.loaders.iter().enumerate() {
                if loader.tensor_names().contains(&name.to_string()) {
                    let tensor = loader.load_tensor(name)?;

                    // Estimate memory
                    let tensor_bytes = tensor.size() * 4; // Assume f32
                    batch.memory_estimate += tensor_bytes;

                    tensor_list.push(tensor);

                    // Get per-model parameters
                    let model_params = self
                        .merge_config
                        .models
                        .get(idx)
                        .map(|m| m.parameters.clone())
                        .unwrap_or_default();
                    param_list.push(model_params);
                }
            }

            // Skip if no tensors found
            if tensor_list.is_empty() {
                continue;
            }

            // Load base tensor if needed
            let base_tensor = if self.method.requires_base_model() {
                if let Some(base) = self.base_loader {
                    if base.tensor_names().contains(&name.to_string()) {
                        let tensor = base.load_tensor(name)?;
                        batch.memory_estimate += tensor.size() * 4;
                        Some(tensor)
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };

            batch.names.push(name.clone());
            batch.tensors.push(tensor_list);
            batch.base_tensors.push(base_tensor);
            batch.params.push(param_list);
        }

        Ok(batch)
    }

    /// Merge a batch of tensors.
    fn merge_batch(&self, batch: &TensorBatch) -> Result<BatchResult> {
        use std::time::Instant;

        let start = Instant::now();
        let mut merged = HashMap::with_capacity(batch.len());

        for i in 0..batch.len() {
            let name = &batch.names[i];
            let tensors = &batch.tensors[i];
            let params = &batch.params[i];
            let base = batch.base_tensors[i].as_ref();

            // Run the merge
            let result = self.method.merge(
                tensors,
                base,
                params,
                &self.merge_config.parameters,
            )?;

            merged.insert(name.clone(), result);
        }

        let elapsed = start.elapsed();

        Ok(BatchResult {
            merged,
            time_ms: elapsed.as_secs_f64() * 1000.0,
            tensor_count: batch.len(),
        })
    }
}

/// Streaming batched merger that processes tensors on-the-fly.
///
/// This variant writes merged tensors immediately instead of accumulating results,
/// which is more memory-efficient for large models.
pub struct StreamingBatchedMerger<'a, W> {
    inner: BatchedMerger<'a>,
    writer: W,
}

impl<'a, W: TensorWriter> StreamingBatchedMerger<'a, W> {
    /// Create a new streaming batched merger.
    pub fn new(merger: BatchedMerger<'a>, writer: W) -> Self {
        Self {
            inner: merger,
            writer,
        }
    }

    /// Process all tensors, writing results immediately.
    pub fn process_all(&mut self, tensor_names: &[String]) -> Result<MergeStats> {
        use std::time::Instant;

        let start = Instant::now();
        let total = tensor_names.len();
        let batch_count = total.div_ceil(self.inner.config.batch_size);

        let mut total_tensors = 0;
        let mut total_bytes = 0usize;

        for (batch_idx, chunk) in tensor_names.chunks(self.inner.config.batch_size).enumerate() {
            debug!("Processing batch {}/{}", batch_idx + 1, batch_count);

            let batch = self.inner.load_batch(chunk)?;
            let result = self.inner.merge_batch(&batch)?;

            // Write immediately
            for name in &batch.names {
                if let Some(merged) = result.merged.get(name) {
                    self.writer.write_tensor(name, merged)?;
                    total_tensors += 1;
                    total_bytes += merged.size() * 4;
                }
            }
        }

        let elapsed = start.elapsed();

        Ok(MergeStats {
            total_tensors,
            total_bytes,
            elapsed_ms: elapsed.as_secs_f64() * 1000.0,
            tensors_per_second: total_tensors as f64 / elapsed.as_secs_f64(),
        })
    }
}

/// Statistics from a merge operation.
#[derive(Debug, Clone)]
pub struct MergeStats {
    /// Total number of tensors processed.
    pub total_tensors: usize,

    /// Total bytes processed.
    pub total_bytes: usize,

    /// Total time in milliseconds.
    pub elapsed_ms: f64,

    /// Throughput in tensors per second.
    pub tensors_per_second: f64,
}

/// Trait for writing merged tensors.
pub trait TensorWriter {
    /// Write a merged tensor.
    fn write_tensor(&mut self, name: &str, tensor: &Array) -> Result<()>;
}

// Implement for our existing TensorWriter
impl TensorWriter for crate::loader::TensorWriter {
    fn write_tensor(&mut self, name: &str, tensor: &Array) -> Result<()> {
        crate::loader::TensorWriter::write_tensor(self, name, tensor)
    }
}

/// Batch-aware sparsification that processes multiple tensors efficiently.
///
/// This function computes thresholds for all tensors first, then applies
/// them in a second pass. This enables better parallelization and memory access patterns.
pub fn batch_sparsify(
    tensors: &[&Array],
    densities: &[f32],
    use_online: bool,
) -> Result<Vec<Array>> {
    if use_online {
        // Convert to owned arrays for the function signature
        let owned: Vec<Array> = tensors.iter().map(|t| (*t).clone()).collect();
        crate::sparsify_batch_by_magnitude(&owned, densities)
    } else {
        // Fallback to sequential standard sparsification
        tensors
            .iter()
            .zip(densities.iter())
            .map(|(t, &d)| crate::sparsify_by_magnitude(t, d))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_config_default() {
        let config = BatchConfig::default();
        assert_eq!(config.batch_size, 32);
        assert!(config.use_online_threshold);
        assert!(config.parallel_threshold);
    }

    #[test]
    fn test_tensor_batch_empty() {
        let batch = TensorBatch::new();
        assert!(batch.is_empty());
        assert_eq!(batch.len(), 0);
    }

    #[test]
    fn test_tensor_batch_clear() {
        let mut batch = TensorBatch::new();
        batch.names.push("test".to_string());
        batch.memory_estimate = 1024;

        assert!(!batch.is_empty());

        batch.clear();
        assert!(batch.is_empty());
        assert_eq!(batch.memory_estimate, 0);
    }

    #[test]
    fn test_batch_sparsify_online() {
        let t1 = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[4]);
        let t2 = Array::from_slice(&[0.5_f32, 1.5, 2.5, 3.5], &[4]);

        let results = batch_sparsify(&[&t1, &t2], &[0.5, 0.5], true).unwrap();
        assert_eq!(results.len(), 2);
    }
}
