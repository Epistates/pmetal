//! Merge method implementations.
//!
//! Each merge method implements a different algorithm for combining model weights.

mod dare;
mod linear;
mod model_stock;
mod passthrough;
mod slerp;
mod task_arithmetic;
mod ties;

pub use dare::DareMerge;
pub use linear::LinearMerge;
pub use model_stock::ModelStockMerge;
pub use passthrough::PassthroughMerge;
pub use slerp::SlerpMerge;
pub use task_arithmetic::TaskArithmeticMerge;
pub use ties::TiesMerge;

use crate::{MergeParameters, Result};
use mlx_rs::Array;

/// Trait for merge method implementations.
pub trait MergeMethod: Send + Sync {
    /// Name of the merge method.
    fn name(&self) -> &'static str;

    /// Human-readable description.
    fn description(&self) -> &'static str;

    /// Whether this method requires a base model.
    fn requires_base_model(&self) -> bool;

    /// Merge a set of tensors.
    ///
    /// # Arguments
    /// * `tensors` - Tensors to merge (one per model)
    /// * `base_tensor` - Base model tensor (if required)
    /// * `params` - Merge parameters for each model
    /// * `global_params` - Global merge parameters
    fn merge(
        &self,
        tensors: &[Array],
        base_tensor: Option<&Array>,
        params: &[MergeParameters],
        global_params: &MergeParameters,
    ) -> Result<Array>;
}
