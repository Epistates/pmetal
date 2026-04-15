//! Shared parity / numerical-comparison helpers for tests.
//!
//! This module is gated on either `cfg(test)` (when building the crate's
//! own tests) or the `test-utils` cargo feature (so cross-crate
//! integration tests in downstream crates such as `pmetal-models` can
//! reach the helpers). Production builds never enable the feature.
//!
//! The helpers are deliberately scoped to "compare two arrays element-wise
//! and report what differs", which is the bedrock for any future
//! architecture parity test (Gemma 4, Qwen3, Llama, …). When you find
//! yourself reaching for a bigger primitive, prefer adding it here over
//! duplicating it in a per-test module.

use pmetal_bridge::compat::{Array, ops};

/// Materialise an `Array` to a flat `Vec<f32>` regardless of its dtype.
///
/// Clones the input to avoid mutating the caller, evals the clone, then
/// reads it as f32 via the bridge's `to_f32_vec` (which handles the
/// fp16/bf16 → f32 conversion). Returns an empty `Vec` on materialise
/// failure rather than panicking — callers that care should check the
/// length explicitly.
pub fn to_f32_vec_eval(arr: &Array) -> Vec<f32> {
    let mut evaled = arr.clone();
    evaled.eval();
    let n = evaled.size();
    evaled.to_f32_vec(n).unwrap_or_default()
}

/// Maximum absolute element-wise difference between two arrays.
///
/// Both arrays are flattened to `Vec<f32>` before comparison, so
/// element-order matters but axis layout does not (the function does NOT
/// transpose). Returns `0.0` for empty inputs.
pub fn max_abs_diff(lhs: &Array, rhs: &Array) -> f32 {
    let lhs_v = to_f32_vec_eval(lhs);
    let rhs_v = to_f32_vec_eval(rhs);
    lhs_v
        .iter()
        .zip(rhs_v.iter())
        .map(|(l, r)| (l - r).abs())
        .fold(0.0f32, f32::max)
}

/// Mean absolute element-wise difference. Returns `0.0` for empty inputs.
pub fn mean_abs_diff(lhs: &Array, rhs: &Array) -> f32 {
    let lhs_v = to_f32_vec_eval(lhs);
    let rhs_v = to_f32_vec_eval(rhs);
    if lhs_v.is_empty() || rhs_v.is_empty() {
        return 0.0;
    }
    let n = lhs_v.len().min(rhs_v.len());
    let sum: f32 = lhs_v
        .iter()
        .zip(rhs_v.iter())
        .map(|(l, r)| (l - r).abs())
        .sum();
    sum / (n as f32)
}

/// Cosine similarity between two flattened arrays, in `[-1, 1]`.
/// Returns `1.0` when both inputs are zero (treats as identical) and `0.0`
/// when only one input is zero (treats as orthogonal).
pub fn cosine_similarity_flat(lhs: &Array, rhs: &Array) -> f32 {
    let lhs_v = to_f32_vec_eval(lhs);
    let rhs_v = to_f32_vec_eval(rhs);
    let n = lhs_v.len().min(rhs_v.len());
    let mut dot = 0.0f64;
    let mut lhs_sq = 0.0f64;
    let mut rhs_sq = 0.0f64;
    for i in 0..n {
        let l = lhs_v[i] as f64;
        let r = rhs_v[i] as f64;
        dot += l * r;
        lhs_sq += l * l;
        rhs_sq += r * r;
    }
    if lhs_sq == 0.0 && rhs_sq == 0.0 {
        return 1.0;
    }
    if lhs_sq == 0.0 || rhs_sq == 0.0 {
        return 0.0;
    }
    (dot / (lhs_sq.sqrt() * rhs_sq.sqrt())) as f32
}

/// Maximum absolute value of an array — useful as the magnitude denominator
/// for relative-tolerance pass/fail logic.
pub fn max_abs_value(arr: &Array) -> f32 {
    to_f32_vec_eval(arr)
        .iter()
        .map(|v| v.abs())
        .fold(0.0f32, f32::max)
}

/// Argmax along the last axis, materialised to a flat `Vec<i32>`.
pub fn argmax_last_axis(arr: &Array) -> Vec<i32> {
    let argmax = ops::argmax_axis(arr, -1);
    let evaled = argmax.clone();
    let _ = evaled.eval();
    evaled
        .as_slice::<u32>()
        .iter()
        .map(|&u| u as i32)
        .collect()
}

/// Counts the positions where the argmax along the last axis matches.
/// Returns the count, not a fraction — divide by `argmax_last_axis(lhs).len()`
/// for a ratio.
pub fn argmax_last_axis_match_count(lhs: &Array, rhs: &Array) -> usize {
    let l = argmax_last_axis(lhs);
    let r = argmax_last_axis(rhs);
    l.iter().zip(r.iter()).filter(|(a, b)| a == b).count()
}

/// True when both arrays have exactly the same shape vector.
pub fn shapes_equal(lhs: &Array, rhs: &Array) -> bool {
    lhs.shape() == rhs.shape()
}

/// Per-position max abs diff along axis 1 (the seq axis for `[B, T, …]`
/// tensors). Returns a `Vec<f32>` of length `seq_len = arr.dim(1)`.
///
/// Used to localise drift to a specific sequence position — for example
/// to verify that partial-RoPE bugs (which only affect positions ≥ 1)
/// don't show up at position 0.
pub fn per_position_max_abs_diff(lhs: &Array, rhs: &Array) -> Vec<f32> {
    let shape = lhs.shape();
    if shape.len() < 2 {
        return Vec::new();
    }
    let t = shape[1] as usize;
    let mut out = Vec::with_capacity(t);
    for pos in 0..t {
        let pos_i = pos as i32;
        let mut start = vec![0i32; shape.len()];
        let mut stop: Vec<i32> = shape.to_vec();
        start[1] = pos_i;
        stop[1] = pos_i + 1;
        let lhs_slice = lhs.slice(&start, &stop);
        let rhs_slice = rhs.slice(&start, &stop);
        out.push(max_abs_diff(&lhs_slice, &rhs_slice));
    }
    out
}

/// Per-position cosine similarity along axis 1.
pub fn per_position_cosine_similarity(lhs: &Array, rhs: &Array) -> Vec<f32> {
    let shape = lhs.shape();
    if shape.len() < 2 {
        return Vec::new();
    }
    let t = shape[1] as usize;
    let mut out = Vec::with_capacity(t);
    for pos in 0..t {
        let pos_i = pos as i32;
        let mut start = vec![0i32; shape.len()];
        let mut stop: Vec<i32> = shape.to_vec();
        start[1] = pos_i;
        stop[1] = pos_i + 1;
        let lhs_slice = lhs.slice(&start, &stop);
        let rhs_slice = rhs.slice(&start, &stop);
        out.push(cosine_similarity_flat(&lhs_slice, &rhs_slice));
    }
    out
}

/// Pass tolerance for one parity checkpoint.
///
/// Pass condition is `max_abs_diff <= atol OR max_abs_diff <= rtol * max_abs_ref`.
/// The OR (not AND) is intentional: small-magnitude tensors should not
/// also have to satisfy a strict relative tolerance.
#[derive(Debug, Clone, Copy)]
pub struct Tolerance {
    /// Absolute tolerance.
    pub atol: f32,
    /// Relative tolerance (multiplied by `max_abs(reference)`).
    pub rtol: f32,
}

impl Tolerance {
    /// Construct a tolerance with the given absolute and relative parts.
    pub const fn new(atol: f32, rtol: f32) -> Self {
        Self { atol, rtol }
    }
}

/// One row of the parity report — describes how a single Rust tensor
/// compared to its reference counterpart.
#[derive(Debug, Clone)]
pub struct ParityReport {
    /// Human-readable checkpoint name (e.g. `"layer_5_hidden"`).
    pub name: String,
    /// Shape of the Rust tensor.
    pub shape_rust: Vec<i32>,
    /// Shape of the reference tensor.
    pub shape_ref: Vec<i32>,
    /// Element count actually compared (min of the two flattened lengths).
    pub n_compared: usize,
    /// Maximum absolute element-wise difference.
    pub max_abs_diff: f32,
    /// Mean absolute element-wise difference.
    pub mean_abs_diff: f32,
    /// Cosine similarity over the flattened pair.
    pub cosine_similarity: f32,
    /// Magnitude of the reference tensor — denominator for `rtol`.
    pub max_abs_ref: f32,
    /// Tolerance gate this report was checked against.
    pub tolerance: Tolerance,
    /// Per-position breakdown along axis 1, only populated when explicitly
    /// requested (most checkpoints leave this `None` to keep the table
    /// printout compact).
    pub per_position_max_abs: Option<Vec<f32>>,
    /// Per-position cosine similarity, only populated when requested.
    pub per_position_cosine: Option<Vec<f32>>,
}

impl ParityReport {
    /// Compute a parity report for a single (rust, reference) pair.
    pub fn compute(
        name: &str,
        rust: &Array,
        reference: &Array,
        tol: Tolerance,
    ) -> Self {
        let shape_rust = rust.shape().to_vec();
        let shape_ref = reference.shape().to_vec();
        let n_compared = to_f32_vec_eval(rust).len().min(to_f32_vec_eval(reference).len());
        Self {
            name: name.to_string(),
            shape_rust,
            shape_ref,
            n_compared,
            max_abs_diff: max_abs_diff(rust, reference),
            mean_abs_diff: mean_abs_diff(rust, reference),
            cosine_similarity: cosine_similarity_flat(rust, reference),
            max_abs_ref: max_abs_value(reference),
            tolerance: tol,
            per_position_max_abs: None,
            per_position_cosine: None,
        }
    }

    /// Compute a parity report and populate the per-position breakdown
    /// along axis 1.
    pub fn compute_with_per_position(
        name: &str,
        rust: &Array,
        reference: &Array,
        tol: Tolerance,
    ) -> Self {
        let mut report = Self::compute(name, rust, reference, tol);
        report.per_position_max_abs = Some(per_position_max_abs_diff(rust, reference));
        report.per_position_cosine = Some(per_position_cosine_similarity(rust, reference));
        report
    }

    /// `true` when shapes match and the checkpoint passes either the
    /// absolute or relative tolerance.
    pub fn passed(&self) -> bool {
        if self.shape_rust != self.shape_ref {
            return false;
        }
        let abs_pass = self.max_abs_diff <= self.tolerance.atol;
        let rel_pass = self.max_abs_diff <= self.tolerance.rtol * self.max_abs_ref;
        abs_pass || rel_pass
    }

    /// Short status string for the table.
    pub fn status_str(&self) -> &'static str {
        if self.passed() { "PASS" } else { "FAIL" }
    }
}

/// Pretty-print a slice of `ParityReport` as a fixed-width table. Used by
/// integration tests to surface the diff at a glance, especially when one
/// checkpoint fails.
pub fn print_report_table(reports: &[ParityReport]) {
    println!(
        "{:<28} {:<22} {:>11} {:>11} {:>10} {:>10} {:>4}",
        "checkpoint", "shape", "max_abs", "mean_abs", "cos_sim", "ref_mag", "pass"
    );
    println!("{}", "-".repeat(100));
    for r in reports {
        let shape_str = format!("{:?}", r.shape_rust);
        println!(
            "{:<28} {:<22} {:>11.3e} {:>11.3e} {:>10.6} {:>10.3e} {:>4}",
            r.name,
            shape_str,
            r.max_abs_diff,
            r.mean_abs_diff,
            r.cosine_similarity,
            r.max_abs_ref,
            r.status_str()
        );
        if let Some(pos) = &r.per_position_max_abs {
            let preview: Vec<String> = pos
                .iter()
                .enumerate()
                .map(|(t, v)| format!("t={}:{:.2e}", t, v))
                .collect();
            println!("    per-pos max_abs:  {}", preview.join("  "));
        }
        if let Some(pos) = &r.per_position_cosine {
            let preview: Vec<String> = pos
                .iter()
                .enumerate()
                .map(|(t, v)| format!("t={}:{:.5}", t, v))
                .collect();
            println!("    per-pos cos_sim:  {}", preview.join("  "));
        }
    }
}
