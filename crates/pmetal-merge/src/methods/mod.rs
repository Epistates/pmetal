//! Merge method implementations.
//!
//! Each merge method implements a different algorithm for combining model weights.
//!
//! # Q1 2026 SOTA Methods
//!
//! - **TIES**: Task arithmetic with sparsification and sign consensus
//! - **DARE**: Random pruning with rescaling
//! - **Model Breadcrumbs**: Deterministic dual-masking (removes outliers AND noise)
//! - **Souper**: Optimal coefficient model soup with inverse-deviation weighting
//! - **DELLA**: Adaptive magnitude-based pruning
//! - **Model Stock**: Geometric interpolation based on task vector similarity

mod breadcrumbs;
mod dare;
mod della;
mod fisher;
mod linear;
mod model_stock;
mod multislerp;
mod nearswap;
mod passthrough;
mod ram;
mod regmean;
mod slerp;
mod souper;
mod task_arithmetic;
mod ties;

pub use breadcrumbs::BreadcrumbsMerge;
pub use dare::DareMerge;
pub use della::DellaMerge;
pub use fisher::FisherMerge;
pub use linear::LinearMerge;
pub use model_stock::ModelStockMerge;
pub use multislerp::MultiSlerpMerge;
pub use nearswap::NearswapMerge;
pub use passthrough::PassthroughMerge;
pub use ram::RamMerge;
pub use regmean::RegMeanMerge;
pub use slerp::SlerpMerge;
pub use souper::SouperMerge;
pub use task_arithmetic::TaskArithmeticMerge;
pub use ties::TiesMerge;

use crate::{MergeParameters, Result};
use pmetal_runtime::compat::Array;

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

    /// Name-aware merge entry. The default implementation forwards to
    /// `merge`, discarding the name; methods that need per-tensor side
    /// data (Fisher information, RegMean Gram matrices) override this to
    /// look up that state.
    fn merge_named(
        &self,
        _name: &str,
        tensors: &[Array],
        base_tensor: Option<&Array>,
        params: &[MergeParameters],
        global_params: &MergeParameters,
    ) -> Result<Array> {
        self.merge(tensors, base_tensor, params, global_params)
    }
}
