//! Runtime backend selection for 4-bit affine quantized linear layers.
//!
//! Benchmarks MLX `quantized_matmul` against the M5-only Metal 4 / MPP path
//! and persists the winning backend per device and problem shape.

use std::{sync::OnceLock, time::Instant};

use crate::ArrayDtypeExt;
use pmetal_bridge::compat::{Array, Dtype, Exception};
use pmetal_metal::{
    BufferUsage, MetalBuffer, MetalContext, MppQuantizedGemm, MppQuantizedGemmConfig,
    context::{DeviceProperties, DeviceTier},
};
use serde::{Deserialize, Serialize};

use crate::bridge::MlxMetalBridge;

use super::persistent_cache::PersistentChoiceCache;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum QuantizedLinearBackend {
    Mlx,
    Mpp4Bit,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct QuantizedLinearDispatchKey {
    device_name: String,
    device_tier: &'static str,
    m: usize,
    n: usize,
    k: usize,
    group_size: i32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct QuantizedLinearProblem {
    m: usize,
    n: usize,
    k: usize,
    group_size: i32,
    output_shape: Vec<i32>,
}

static QUANTIZED_LINEAR_BACKEND_CACHE: OnceLock<PersistentChoiceCache<QuantizedLinearBackend>> =
    OnceLock::new();

fn quantized_linear_backend_cache() -> &'static PersistentChoiceCache<QuantizedLinearBackend> {
    QUANTIZED_LINEAR_BACKEND_CACHE
        .get_or_init(|| PersistentChoiceCache::new("quantized_linear_backends.json"))
}

fn device_tier_key(tier: DeviceTier) -> &'static str {
    match tier {
        DeviceTier::Base => "base",
        DeviceTier::Pro => "pro",
        DeviceTier::Max => "max",
        DeviceTier::Ultra => "ultra",
    }
}

impl QuantizedLinearDispatchKey {
    fn new(props: &DeviceProperties, problem: &QuantizedLinearProblem) -> Self {
        Self {
            device_name: props.name.clone(),
            device_tier: device_tier_key(props.device_tier),
            m: problem.m,
            n: problem.n,
            k: problem.k,
            group_size: problem.group_size,
        }
    }

    fn cache_key(&self) -> String {
        format!(
            "{}:{}:{}:{}:{}:{}",
            self.device_name, self.device_tier, self.m, self.n, self.k, self.group_size
        )
    }
}

fn cached_quantized_linear_backend(
    key: &QuantizedLinearDispatchKey,
) -> Option<QuantizedLinearBackend> {
    quantized_linear_backend_cache().get(&key.cache_key())
}

fn cache_quantized_linear_backend(
    key: QuantizedLinearDispatchKey,
    backend: QuantizedLinearBackend,
) {
    quantized_linear_backend_cache().insert(key.cache_key(), backend);
}

#[cfg(test)]
fn clear_cached_quantized_linear_backends() {
    quantized_linear_backend_cache().clear();
}

fn quantized_rhs_transposed_problem(
    x: &Array,
    w_q: &Array,
    scales: &Array,
    biases: &Array,
    group_size: i32,
) -> Option<QuantizedLinearProblem> {
    if group_size <= 0
        || x.dtype() != Dtype::Float16
        || w_q.dtype() != Dtype::Uint32
        || w_q.shape().len() != 2
        || scales.shape().len() != 2
        || biases.shape() != scales.shape()
    {
        return None;
    }

    let x_shape = x.shape();
    if x_shape.len() < 2 {
        return None;
    }

    let n = w_q.dim(0) as usize;
    let packed_dim = w_q.dim(1) as usize;
    let num_groups = scales.dim(1) as usize;
    let group_size = group_size as usize;
    let k_from_weight = packed_dim.checked_mul(8)?;
    let k_from_scales = num_groups.checked_mul(group_size)?;
    if k_from_weight != k_from_scales || scales.dim(0) as usize != n {
        return None;
    }

    let k = *x_shape.last()? as usize;
    if k != k_from_weight {
        return None;
    }

    let m = x_shape[..x_shape.len() - 1]
        .iter()
        .map(|dim| *dim as usize)
        .product::<usize>();

    let mut output_shape = x_shape.to_vec();
    *output_shape.last_mut()? = n as i32;

    Some(QuantizedLinearProblem {
        m,
        n,
        k,
        group_size: group_size as i32,
        output_shape,
    })
}

fn should_consider_mpp_quantized_linear(
    props: &DeviceProperties,
    problem: &QuantizedLinearProblem,
) -> bool {
    if !props.has_nax() || problem.m == 0 || problem.n < 64 || problem.k < 64 {
        return false;
    }

    let work = (problem.m as u128) * (problem.n as u128) * (problem.k as u128);
    let base_threshold = match props.device_tier {
        DeviceTier::Ultra | DeviceTier::Max => 262_144_u128,
        DeviceTier::Pro => 524_288_u128,
        DeviceTier::Base => 1_048_576_u128,
    };
    let threshold = if problem.n % 64 == 0 && problem.k % 128 == 0 {
        base_threshold / 2
    } else {
        base_threshold
    };

    work >= threshold
}

fn max_abs_diff(lhs: &Array, rhs: &Array) -> Result<f32, Exception> {
    let lhs = if lhs.dtype() == Dtype::Float32 {
        lhs.clone()
    } else {
        lhs.as_dtype(Dtype::Float32.as_i32())
    };
    let rhs = if rhs.dtype() == Dtype::Float32 {
        rhs.clone()
    } else {
        rhs.as_dtype(Dtype::Float32.as_i32())
    };

    let diff = lhs.subtract(&rhs).abs_val().max(None);
    let diff_owned = diff.clone();
    diff_owned.eval();
    Ok(diff_owned.item::<f32>())
}

fn run_mlx_quantized_rhs_transposed(
    x: &Array,
    w_q: &Array,
    scales: &Array,
    biases: &Array,
    group_size: i32,
) -> Result<Array, Exception> {
    Ok(x.quantized_matmul(w_q, scales, Some(biases), true, group_size, 4))
}

fn run_mpp_quantized_rhs_transposed(
    x: &Array,
    w_q: &Array,
    scales: &Array,
    biases: &Array,
    ctx: &std::sync::Arc<MetalContext>,
    problem: &QuantizedLinearProblem,
) -> Result<Array, Exception> {
    let x_2d = if x.shape().len() == 2 {
        x.clone()
    } else {
        x.reshape(&[problem.m as i32, problem.k as i32])
    };

    let x_view =
        MlxMetalBridge::view_f16(ctx, &x_2d).map_err(|e| Exception::custom(e.to_string()))?;
    let w_view =
        MlxMetalBridge::view_u32(ctx, w_q).map_err(|e| Exception::custom(e.to_string()))?;
    let scales_buffer =
        MlxMetalBridge::copy_as_f16(ctx, scales).map_err(|e| Exception::custom(e.to_string()))?;
    let bias_buffer =
        MlxMetalBridge::copy_as_f16(ctx, biases).map_err(|e| Exception::custom(e.to_string()))?;
    let output_buffer = MetalBuffer::new(ctx, problem.m * problem.n, BufferUsage::Shared)
        .map_err(|e| Exception::custom(e.to_string()))?;

    let mut config = MppQuantizedGemmConfig::new(problem.m, problem.n, problem.k);
    config.group_size = problem.group_size as usize;
    config.bits = 4;

    let gemm = MppQuantizedGemm::new(ctx.clone(), config);
    if !gemm.is_available() {
        return Err(Exception::custom(
            "MPP quantized GEMM unavailable on current device".to_string(),
        ));
    }

    gemm.execute(
        &x_view,
        &w_view,
        &scales_buffer,
        Some(&bias_buffer),
        &output_buffer,
    )
    .map_err(|e| Exception::custom(e.to_string()))?;

    MlxMetalBridge::buffer_into_array_f16(output_buffer, &problem.output_shape)
        .map_err(|e| Exception::custom(e.to_string()))
}

fn execute_quantized_linear_backend(
    backend: QuantizedLinearBackend,
    x: &Array,
    w_q: &Array,
    scales: &Array,
    biases: &Array,
    ctx: &std::sync::Arc<MetalContext>,
    problem: &QuantizedLinearProblem,
) -> Result<Array, Exception> {
    match backend {
        QuantizedLinearBackend::Mlx => {
            run_mlx_quantized_rhs_transposed(x, w_q, scales, biases, problem.group_size)
        }
        QuantizedLinearBackend::Mpp4Bit => {
            run_mpp_quantized_rhs_transposed(x, w_q, scales, biases, ctx, problem)
        }
    }
}

fn benchmark_quantized_linear_backends(
    x: &Array,
    w_q: &Array,
    scales: &Array,
    biases: &Array,
    ctx: &std::sync::Arc<MetalContext>,
    problem: &QuantizedLinearProblem,
) -> Result<(QuantizedLinearBackend, Array), Exception> {
    let mlx_start = Instant::now();
    let mlx_output = run_mlx_quantized_rhs_transposed(x, w_q, scales, biases, problem.group_size)?;
    let mlx_evaled = mlx_output.clone();
    mlx_evaled.eval();
    let mlx_elapsed = mlx_start.elapsed();

    let mpp_start = Instant::now();
    let mpp_output = match run_mpp_quantized_rhs_transposed(x, w_q, scales, biases, ctx, problem) {
        Ok(output) => {
            output.eval();
            Some(output)
        }
        Err(error) => {
            tracing::debug!(
                "MPP 4-bit quantized GEMM benchmark failed for [{}x{}] x [{}x{}]^T, using MLX: {error}",
                problem.m,
                problem.k,
                problem.n,
                problem.k
            );
            None
        }
    };
    let mpp_elapsed = mpp_start.elapsed();

    if let Some(mpp_output) = mpp_output {
        let max_diff = max_abs_diff(&mlx_output, &mpp_output)?;
        if max_diff <= 0.1 && mpp_elapsed < mlx_elapsed {
            tracing::debug!(
                "Selected MPP 4-bit quantized GEMM for [{}x{}] x [{}x{}]^T ({:?} vs {:?}, max_diff={:.5})",
                problem.m,
                problem.k,
                problem.n,
                problem.k,
                mpp_elapsed,
                mlx_elapsed,
                max_diff
            );
            return Ok((QuantizedLinearBackend::Mpp4Bit, mpp_output));
        }

        tracing::debug!(
            "Keeping MLX quantized_matmul for [{}x{}] x [{}x{}]^T ({:?} vs {:?}, max_diff={:.5})",
            problem.m,
            problem.k,
            problem.n,
            problem.k,
            mlx_elapsed,
            mpp_elapsed,
            max_diff
        );
    }

    Ok((QuantizedLinearBackend::Mlx, mlx_output))
}

/// Best-effort backend selection for 4-bit affine quantized linear layers.
///
/// This path is intended for weights stored in the MLX/affine quantized format
/// used by `mlx_rs::ops::quantize`, with logical weight shape `[out_features,
/// in_features]` and `transpose=true` semantics.
pub fn quantized_linear_rhs_transposed_best_effort(
    x: &Array,
    w_q: &Array,
    scales: &Array,
    biases: &Array,
    group_size: i32,
) -> Result<Array, Exception> {
    let Some(problem) = quantized_rhs_transposed_problem(x, w_q, scales, biases, group_size) else {
        return run_mlx_quantized_rhs_transposed(x, w_q, scales, biases, group_size);
    };

    let ctx = match MetalContext::global() {
        Ok(ctx) => ctx,
        Err(error) => {
            tracing::debug!("MPP quantized GEMM unavailable, falling back to MLX: {error}");
            return run_mlx_quantized_rhs_transposed(x, w_q, scales, biases, group_size);
        }
    };

    if !should_consider_mpp_quantized_linear(ctx.properties(), &problem) {
        return run_mlx_quantized_rhs_transposed(x, w_q, scales, biases, group_size);
    }

    let dispatch_key = QuantizedLinearDispatchKey::new(ctx.properties(), &problem);
    if let Some(backend) = cached_quantized_linear_backend(&dispatch_key) {
        return execute_quantized_linear_backend(backend, x, w_q, scales, biases, &ctx, &problem)
            .or_else(|error| {
                tracing::debug!(
                    "Cached {:?} quantized linear path failed, falling back to MLX: {error}",
                    backend
                );
                cache_quantized_linear_backend(dispatch_key.clone(), QuantizedLinearBackend::Mlx);
                run_mlx_quantized_rhs_transposed(x, w_q, scales, biases, group_size)
            });
    }

    let (backend, output) =
        benchmark_quantized_linear_backends(x, w_q, scales, biases, &ctx, &problem)?;
    cache_quantized_linear_backend(dispatch_key, backend);
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_bridge::compat::{Dtype, random};

    #[test]
    #[serial_test::serial]
    fn test_quantized_linear_backend_cache_roundtrip() {
        clear_cached_quantized_linear_backends();

        let key = QuantizedLinearDispatchKey {
            device_name: "Apple M5 Max".to_string(),
            device_tier: "max",
            m: 8,
            n: 256,
            k: 128,
            group_size: 64,
        };

        assert_eq!(cached_quantized_linear_backend(&key), None);
        cache_quantized_linear_backend(key.clone(), QuantizedLinearBackend::Mpp4Bit);
        assert_eq!(
            cached_quantized_linear_backend(&key),
            Some(QuantizedLinearBackend::Mpp4Bit)
        );

        clear_cached_quantized_linear_backends();
    }

    #[test]
    fn test_quantized_rhs_transposed_problem_infers_output_shape() {
        let x = pmetal_bridge::compat::ops::zeros(&[2, 4, 128], Dtype::Float16);
        let w_q = pmetal_bridge::compat::ops::zeros(&[256, 16], Dtype::Uint32);
        let scales = pmetal_bridge::compat::ops::zeros(&[256, 2], Dtype::Float32);
        let biases = pmetal_bridge::compat::ops::zeros(&[256, 2], Dtype::Float32);

        let problem = quantized_rhs_transposed_problem(&x, &w_q, &scales, &biases, 64).unwrap();
        assert_eq!(
            problem,
            QuantizedLinearProblem {
                m: 8,
                n: 256,
                k: 128,
                group_size: 64,
                output_shape: vec![2, 4, 256],
            }
        );
    }

    #[test]
    fn test_quantized_rhs_transposed_problem_rejects_mismatched_metadata() {
        let x = pmetal_bridge::compat::ops::zeros(&[2, 4, 128], Dtype::Float16);
        let w_q = pmetal_bridge::compat::ops::zeros(&[256, 16], Dtype::Uint32);
        let scales = pmetal_bridge::compat::ops::zeros(&[255, 2], Dtype::Float32);
        let biases = pmetal_bridge::compat::ops::zeros(&[256, 2], Dtype::Float32);

        assert!(quantized_rhs_transposed_problem(&x, &w_q, &scales, &biases, 64).is_none());
    }

    #[test]
    #[serial_test::serial]
    fn test_quantized_linear_rhs_transposed_best_effort_matches_mlx() {
        clear_cached_quantized_linear_backends();

        let x = random::normal(&[2, 4, 128], Dtype::Float32).as_dtype(Dtype::Float16.as_i32());
        let weight = random::normal(&[256, 128], Dtype::Float32).as_dtype(Dtype::Float16.as_i32());
        let (w_q, scales, biases) = weight.quantize_weights(64, 4);

        let output =
            quantized_linear_rhs_transposed_best_effort(&x, &w_q, &scales, &biases, 64).unwrap();
        let reference = run_mlx_quantized_rhs_transposed(&x, &w_q, &scales, &biases, 64).unwrap();

        let max_diff = max_abs_diff(&output, &reference).unwrap();
        assert!(max_diff < 0.1, "max_diff: {}", max_diff);

        clear_cached_quantized_linear_backends();
    }
}
