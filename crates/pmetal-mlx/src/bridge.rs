//! Handing a runtime array to a Metal kernel in `pmetal-metal`.
//!
//! `pmetal-metal` and `pmetal-runtime` do not depend on each other, so the
//! crossing lives here: the one place that knows both an [`Array`] and a
//! `MetalBuffer`.

use pmetal_metal::{
    buffer::{BufferUsage, MetalBuffer},
    context::MetalContext,
    error::{MetalError, Result as MetalResult},
};
use pmetal_runtime::compat::Array;

/// Array to Metal buffer.
pub struct MlxMetalBridge;

impl MlxMetalBridge {
    /// Copy an array into a new f32 Metal buffer, converting the dtype if
    /// needed. The array is evaluated first.
    ///
    /// The buffer owns its data, so the array may be dropped or reused; the
    /// copy is for small inputs (a decode step's hidden state), not weights.
    pub fn copy_as_f32(ctx: &MetalContext, array: &Array) -> MetalResult<MetalBuffer<f32>> {
        let data = array.clone().to_f32_vec(array.size()).ok_or_else(|| {
            MetalError::InvalidConfig("MLX array could not be read as f32".into())
        })?;
        MetalBuffer::from_slice(ctx, &data, BufferUsage::Shared)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_runtime::compat::Dtype;

    #[test]
    fn a_copy_holds_the_values_in_f32() {
        let ctx = MetalContext::new().unwrap();
        let values = [1.0f32, -2.5, 3.25, 0.0, 7.0, 8.0];
        for dtype in [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16] {
            let array = Array::from_f32_slice(&values, &[2, 3]).as_dtype(dtype.as_i32());
            let buffer = MlxMetalBridge::copy_as_f32(&ctx, &array).unwrap();
            assert_eq!(buffer.len(), values.len());
            assert_eq!(buffer.as_slice(), &values, "{dtype:?}");
        }
    }

    #[test]
    fn a_copy_outlives_the_array_it_came_from() {
        let ctx = MetalContext::new().unwrap();
        let array = Array::from_f32_slice(&[4.0, 5.0], &[2]);
        let buffer = MlxMetalBridge::copy_as_f32(&ctx, &array).unwrap();
        drop(array);
        assert_eq!(buffer.as_slice(), &[4.0, 5.0]);
    }
}
