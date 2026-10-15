//! Post-merge sanity checking.
//!
//! Detects pathological merged tensors *before* they hit disk. The primary
//! guarantee is "no NaN, no inf" — a merged model with even one such value
//! is unloadable in practice and silently corrupts every downstream
//! consumer. The optional `Full` level additionally collects per-tensor
//! aggregate statistics (mean / std / abs_max / sparsity) for observability.
//!
//! The default level is [`SanityLevel::Quick`], which adds essentially no
//! cost beyond the f32 materialization that the writer already performs.

use crate::{MergeError, Result};
use pmetal_runtime::compat::Array;
use serde::{Deserialize, Serialize};

/// How aggressively to validate each merged tensor before writing it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SanityLevel {
    /// No checks. Fastest path; never fails.
    Off,
    /// Reject any tensor containing a NaN or inf. Default.
    #[default]
    Quick,
    /// Quick + collect descriptive aggregates (mean, std, abs_max, sparsity).
    /// Fires `tracing::warn!` for outliers (|mean| > 10, std > 100).
    Full,
}

/// Aggregate report for one merged tensor under [`SanityLevel::Full`].
#[derive(Debug, Clone)]
pub struct MergedTensorReport {
    /// Tensor name.
    pub name: String,
    /// Tensor shape.
    pub shape: Vec<i32>,
    /// Number of elements.
    pub n_elements: usize,
    /// Number of NaN values found.
    pub n_nan: usize,
    /// Number of +inf or -inf values found.
    pub n_inf: usize,
    /// Mean (None when level is `Quick`).
    pub mean: Option<f32>,
    /// Population standard deviation (None when level is `Quick`).
    pub std: Option<f32>,
    /// Maximum absolute value (None when level is `Quick`).
    pub abs_max: Option<f32>,
    /// Fraction of exact-zero elements (None when level is `Quick`).
    pub sparsity: Option<f32>,
}

impl MergedTensorReport {
    /// Quick variant: only NaN/inf counts populated.
    pub fn quick(name: String, shape: Vec<i32>, data: &[f32]) -> Self {
        let mut n_nan = 0_usize;
        let mut n_inf = 0_usize;
        for &v in data {
            if v.is_nan() {
                n_nan += 1;
            } else if v.is_infinite() {
                n_inf += 1;
            }
        }
        Self {
            name,
            shape,
            n_elements: data.len(),
            n_nan,
            n_inf,
            mean: None,
            std: None,
            abs_max: None,
            sparsity: None,
        }
    }

    /// Full variant: all stats populated. NaN/inf are excluded from the
    /// mean/std calculation but still counted in `n_nan` / `n_inf`.
    pub fn full(name: String, shape: Vec<i32>, data: &[f32]) -> Self {
        let mut n_nan = 0_usize;
        let mut n_inf = 0_usize;
        let mut n_zero = 0_usize;
        let mut sum = 0.0_f64;
        let mut sumsq = 0.0_f64;
        let mut count = 0_usize;
        let mut abs_max = 0.0_f32;
        for &v in data {
            if v.is_nan() {
                n_nan += 1;
                continue;
            }
            if v.is_infinite() {
                n_inf += 1;
                continue;
            }
            if v == 0.0 {
                n_zero += 1;
            }
            sum += v as f64;
            sumsq += (v as f64) * (v as f64);
            count += 1;
            let av = v.abs();
            if av > abs_max {
                abs_max = av;
            }
        }
        let n = data.len().max(1) as f32;
        let (mean, std) = if count > 0 {
            let m = sum / count as f64;
            let var = (sumsq / count as f64) - m * m;
            (Some(m as f32), Some(var.max(0.0).sqrt() as f32))
        } else {
            (Some(f32::NAN), Some(f32::NAN))
        };
        Self {
            name,
            shape,
            n_elements: data.len(),
            n_nan,
            n_inf,
            mean,
            std,
            abs_max: Some(abs_max),
            sparsity: Some(n_zero as f32 / n),
        }
    }

    /// True iff the tensor contains any NaN or inf — the load-blocking case.
    pub fn is_corrupt(&self) -> bool {
        self.n_nan > 0 || self.n_inf > 0
    }
}

/// Run the configured sanity level over a freshly-merged tensor. Returns
/// `Ok(Some(report))` when stats were collected, `Ok(None)` when level is
/// `Off`, and `Err(MergeError)` when the tensor contains NaN or inf so the
/// caller fails fast before writing a poisoned shard to disk.
///
/// The tensor is materialized to f32 for the inspection — the same buffer
/// layout the writer produces — so this function does not change the
/// in-graph dtype trajectory.
pub fn check_tensor(
    name: &str,
    tensor: &Array,
    level: SanityLevel,
) -> Result<Option<MergedTensorReport>> {
    if matches!(level, SanityLevel::Off) {
        return Ok(None);
    }
    let mut copy = tensor.as_dtype(pmetal_runtime::compat::Dtype::Float32.as_i32());
    let shape = copy.shape().to_vec();
    let n: usize = shape.iter().map(|&s| s as usize).product();
    let data: Vec<f32> = copy.to_f32_vec(n).unwrap_or_default();

    let report = match level {
        SanityLevel::Off => unreachable!(),
        SanityLevel::Quick => MergedTensorReport::quick(name.to_string(), shape, &data),
        SanityLevel::Full => {
            let r = MergedTensorReport::full(name.to_string(), shape, &data);
            if let (Some(m), Some(s)) = (r.mean, r.std) {
                if m.abs() > 10.0 || s > 100.0 {
                    tracing::warn!(
                        tensor = name,
                        mean = m,
                        std = s,
                        abs_max = r.abs_max.unwrap_or(0.0),
                        "merged tensor is an outlier — verify upstream weights"
                    );
                }
            }
            r
        }
    };

    if report.is_corrupt() {
        return Err(MergeError::InvalidConfig(format!(
            "merged tensor '{}' contains {} NaN and {} inf values; \
             refusing to write a corrupt shard. Verify input dtypes and \
             merge parameters",
            report.name, report.n_nan, report.n_inf
        )));
    }

    Ok(Some(report))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quick_report_counts_nan_and_inf() {
        let data = vec![
            1.0_f32,
            f32::NAN,
            2.0,
            f32::INFINITY,
            3.0,
            f32::NEG_INFINITY,
        ];
        let r = MergedTensorReport::quick("t".into(), vec![6], &data);
        assert_eq!(r.n_nan, 1);
        assert_eq!(r.n_inf, 2);
        assert_eq!(r.n_elements, 6);
        assert!(r.is_corrupt());
        assert!(r.mean.is_none());
    }

    #[test]
    fn full_report_collects_stats() {
        let data = vec![0.0_f32, 1.0, 2.0, 3.0, 0.0]; // zeros produce sparsity 0.4
        let r = MergedTensorReport::full("t".into(), vec![5], &data);
        assert_eq!(r.n_nan, 0);
        assert_eq!(r.n_inf, 0);
        assert!(!r.is_corrupt());
        assert!((r.mean.unwrap() - 1.2).abs() < 1e-6);
        assert!(r.std.unwrap() > 0.0);
        assert!((r.sparsity.unwrap() - 0.4).abs() < 1e-6);
        assert!((r.abs_max.unwrap() - 3.0).abs() < 1e-6);
    }

    #[test]
    fn check_tensor_off_skips_work() {
        let arr = Array::from_f32_slice(&[1.0_f32, 2.0, 3.0], &[3]);
        let r = check_tensor("t", &arr, SanityLevel::Off).unwrap();
        assert!(r.is_none());
    }

    #[test]
    fn check_tensor_clean_quick_passes() {
        let arr = Array::from_f32_slice(&[1.0_f32, 2.0, 3.0], &[3]);
        let r = check_tensor("t", &arr, SanityLevel::Quick).unwrap();
        let r = r.expect("should report");
        assert!(!r.is_corrupt());
        assert_eq!(r.n_elements, 3);
    }

    #[test]
    fn check_tensor_nan_quick_errors() {
        let arr = Array::from_f32_slice(&[1.0_f32, f32::NAN, 3.0], &[3]);
        let err = check_tensor("t", &arr, SanityLevel::Quick).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("NaN"));
        assert!(msg.contains("merged tensor 't'"));
    }
}
