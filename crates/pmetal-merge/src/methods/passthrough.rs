use mlx_rs::Array;
use crate::{MergeMethod, MergeParameters, Result, MergeError};

/// Passthrough "merge" - copies the tensor from the first model that contains it.
///
/// This effectively implements a priority merge where the first model in the list
/// that has a specific tensor "wins". Useful for assembling models from disjoint
/// parts (Frankenmerging) or format conversion.
#[derive(Debug, Clone, Default)]
pub struct PassthroughMerge;

impl PassthroughMerge {
    pub fn new() -> Self {
        Self
    }
}

impl MergeMethod for PassthroughMerge {
    fn name(&self) -> &'static str {
        "passthrough"
    }

    fn description(&self) -> &'static str {
        "Copy first available tensor (priority merge)"
    }

    fn requires_base_model(&self) -> bool {
        false
    }

    fn merge(
        &self,
        tensors: &[Array],
        _base_tensor: Option<&Array>,
        _params: &[MergeParameters],
        _global_params: &MergeParameters,
    ) -> Result<Array> {
        if tensors.is_empty() {
            return Err(MergeError::NotEnoughModels {
                expected: 1,
                actual: 0,
            });
        }
        // Return the first one (highest priority)
        Ok(tensors[0].clone())
    }
}
