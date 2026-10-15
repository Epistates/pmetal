//! Merge method implementations.
//!
//! Each merge method implements a different algorithm for combining model weights.
//!
//! # Methods
//!
//! - **Linear / task arithmetic**: weighted averages of weights or task vectors
//! - **SLERP / Multi-SLERP**: spherical interpolation and the weighted
//!   Karcher mean on the sphere
//! - **TIES**: trim, elect a sign by mass, disjoint mean (Yadav et al.)
//! - **DARE**: random drop and rescale (Yu et al.)
//! - **DELLA**: magnitude-ranked drop and rescale (Deep et al.)
//! - **Model Breadcrumbs**: masks the smallest entries and the outliers
//! - **Consensus**: TALL-mask consensus merging (Wang et al.)
//! - **Model Stock**: geometric interpolation toward the pre-trained weights
//! - **Fisher / RegMean**: data-informed averaging from precomputed statistics

mod breadcrumbs;
mod consensus;
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
mod task_arithmetic;
mod ties;

pub use breadcrumbs::BreadcrumbsMerge;
pub use consensus::ConsensusMerge;
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

    /// Name-aware merge entry, the one every pipeline calls.
    ///
    /// The default implementation resolves each filter-conditional parameter
    /// (`weight: [{value, filter}, ...]`) against `name`, so `merge` only
    /// ever sees scalars, and forwards to it. Methods that need per-tensor
    /// side data (Fisher information, RegMean Gram matrices) override this
    /// to look that state up, and call [`MergeParameters::resolved`] themselves.
    fn merge_named(
        &self,
        name: &str,
        tensors: &[Array],
        base_tensor: Option<&Array>,
        params: &[MergeParameters],
        global_params: &MergeParameters,
    ) -> Result<Array> {
        let params: Vec<MergeParameters> = params
            .iter()
            .map(|p| p.resolved_in(name, global_params.layer_count))
            .collect();
        self.merge(tensors, base_tensor, &params, &global_params.resolved(name))
    }
}
