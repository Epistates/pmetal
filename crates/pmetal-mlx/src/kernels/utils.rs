//! Dispatch helpers shared by the kernels.

use pmetal_runtime::compat::Array;

/// Whether autograd is tracing any of `arrays`, so whatever is computed from
/// them has to stay in MLX's graph.
///
/// A kernel that builds its output outside the graph (from a Metal buffer, or
/// from values read back to the host) hands autograd a constant, and one whose
/// primitive has no VJP (a custom Metal kernel, an op like `tri_inv`) fails or
/// returns NaN under it; either way everything upstream trains on nothing.
/// Each such kernel's dispatcher asks this and takes its graph path when the
/// answer is yes. A frozen model's forward (a distillation teacher, a
/// reference policy) isn't traced even inside `value_and_grad` and keeps the
/// fast kernels.
pub fn any_traced(arrays: &[&Array]) -> bool {
    arrays.iter().any(|a| a.is_tracer())
}
