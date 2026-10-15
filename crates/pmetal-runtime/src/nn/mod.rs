//! The operations a model's forward and its training loss are built from, over
//! the runtime's arrays.
//!
//! - [`attention`]: scaled dot-product attention with masks, sinks and logit
//!   softcapping
//! - [`gated_delta`]: the gated delta rule (Qwen 3.5's linear attention), with
//!   a chunkwise parallel form for training
//! - [`loss`]: the LM-head loss without materializing the logits
//! - [`mod@rope`]: rotary embeddings at contiguous or explicit positions
//! - [`moe`]: the expert MLP and the combine step of a mixture-of-experts block
//! - [`training`]: the training-mode flag the model forwards read

pub mod attention;
pub mod gated_delta;
pub mod loss;
pub mod moe;
pub mod rope;
pub mod training;

pub use attention::*;
pub use gated_delta::*;
pub use loss::*;
pub use moe::{Expert, moe_combine};
pub use rope::*;
pub use training::*;

use crate::compat::Array;

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
