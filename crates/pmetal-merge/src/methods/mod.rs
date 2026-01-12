//! Merge method implementations.
//!
//! Each merge method implements a different algorithm for combining model weights.

mod linear;
mod slerp;
mod ties;
mod dare;
mod model_stock;
mod passthrough;
mod task_arithmetic;

pub use linear::LinearMerge;
pub use slerp::SlerpMerge;
pub use ties::TiesMerge;
pub use dare::DareMerge;
pub use model_stock::ModelStockMerge;
pub use passthrough::PassthroughMerge;
pub use task_arithmetic::TaskArithmeticMerge;

use mlx_rs::Array;
use crate::{MergeParameters, Result};

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
