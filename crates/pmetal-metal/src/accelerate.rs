//! Accelerate.framework wrappers for CPU-side vector operations.
//!
//! Provides vDSP-accelerated routines for gradient norm computation
//! and gradient scaling, achieving 5-10x speedup over scalar loops
//! for CPU fallback paths.

#[cfg(target_os = "macos")]
mod ffi {
    unsafe extern "C" {
        /// Compute sum of squares: result = sum(data[i]^2)
        /// vDSP_svesq(A, IA, C, N) — single-precision vector sum of element-squared
        pub fn vDSP_svesq(a: *const f32, ia: isize, c: *mut f32, n: usize);

        /// Vector-scalar multiply: C[i] = A[i] * B
        /// vDSP_vsmul(A, IA, B, C, IC, N) — single-precision vector-scalar multiply
        pub fn vDSP_vsmul(
            a: *const f32,
            ia: isize,
            b: *const f32,
            c: *mut f32,
            ic: isize,
            n: usize,
        );
    }
}

/// Compute the sum of squares of a float slice using vDSP.
///
/// Returns `sum(data[i]^2)` for all elements.
/// Uses `vDSP_svesq` on macOS for hardware-accelerated computation.
///
/// This is used for gradient norm computation on the CPU fallback path.
pub fn sum_of_squares(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }

    #[cfg(target_os = "macos")]
    {
        let mut result: f32 = 0.0;
        unsafe {
            ffi::vDSP_svesq(
                data.as_ptr(),
                1, // stride
                &mut result,
                data.len(),
            );
        }
        result
    }

    #[cfg(not(target_os = "macos"))]
    {
        data.iter().map(|x| x * x).sum()
    }
}

/// Scale a float slice in-place using vDSP.
///
/// Computes `data[i] = data[i] * scale` for all elements.
/// Uses `vDSP_vsmul` on macOS for hardware-accelerated computation.
///
/// This is used for gradient clipping on the CPU fallback path.
pub fn scale_inplace(data: &mut [f32], scale: f32) {
    if data.is_empty() {
        return;
    }

    #[cfg(target_os = "macos")]
    {
        unsafe {
            ffi::vDSP_vsmul(
                data.as_ptr(),
                1, // input stride
                &scale,
                data.as_mut_ptr(),
                1, // output stride
                data.len(),
            );
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        for x in data.iter_mut() {
            *x *= scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_of_squares() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let result = sum_of_squares(&data);
        assert!((result - 30.0).abs() < 1e-6, "Expected 30.0, got {result}");
    }

    #[test]
    fn test_sum_of_squares_empty() {
        let data: Vec<f32> = vec![];
        assert_eq!(sum_of_squares(&data), 0.0);
    }

    #[test]
    fn test_scale_inplace() {
        let mut data = vec![1.0f32, 2.0, 3.0, 4.0];
        scale_inplace(&mut data, 0.5);
        assert!((data[0] - 0.5).abs() < 1e-6);
        assert!((data[1] - 1.0).abs() < 1e-6);
        assert!((data[2] - 1.5).abs() < 1e-6);
        assert!((data[3] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_scale_inplace_empty() {
        let mut data: Vec<f32> = vec![];
        scale_inplace(&mut data, 2.0);
        // Should not panic
    }

    #[test]
    fn test_sum_of_squares_large() {
        // Test with a larger buffer to exercise vectorized path
        let data: Vec<f32> = (1..=1000).map(|x| x as f32).collect();
        let expected: f32 = data.iter().map(|x| x * x).sum();
        let result = sum_of_squares(&data);
        assert!(
            (result - expected).abs() / expected < 1e-5,
            "Expected {expected}, got {result}"
        );
    }

    #[test]
    fn test_sum_of_squares_negative_values() {
        let data = vec![-1.0f32, -2.0, -3.0, -4.0];
        let result = sum_of_squares(&data);
        // (-1)^2 + (-2)^2 + (-3)^2 + (-4)^2 = 1 + 4 + 9 + 16 = 30
        assert!((result - 30.0).abs() < 1e-6, "Expected 30.0, got {result}");
    }

    #[test]
    fn test_sum_of_squares_single_element() {
        let data = vec![7.0f32];
        let result = sum_of_squares(&data);
        assert!((result - 49.0).abs() < 1e-6, "Expected 49.0, got {result}");
    }

    #[test]
    fn test_sum_of_squares_very_large() {
        // 1M+ elements to stress the vDSP path
        let n = 1_048_576;
        let data: Vec<f32> = vec![1.0; n];
        let result = sum_of_squares(&data);
        assert!(
            (result - n as f32).abs() / (n as f32) < 1e-5,
            "Expected {n}, got {result}"
        );
    }

    #[test]
    fn test_scale_inplace_negative_values() {
        let mut data = vec![-1.0f32, -2.0, -3.0, -4.0];
        scale_inplace(&mut data, -2.0);
        assert!((data[0] - 2.0).abs() < 1e-6);
        assert!((data[1] - 4.0).abs() < 1e-6);
        assert!((data[2] - 6.0).abs() < 1e-6);
        assert!((data[3] - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_scale_inplace_single_element() {
        let mut data = vec![5.0f32];
        scale_inplace(&mut data, 3.0);
        assert!(
            (data[0] - 15.0).abs() < 1e-6,
            "Expected 15.0, got {}",
            data[0]
        );
    }

    #[test]
    fn test_scale_inplace_very_large() {
        // 1M+ elements to stress the vDSP path
        let n = 1_048_576;
        let mut data: Vec<f32> = vec![2.0; n];
        scale_inplace(&mut data, 0.5);
        for (i, val) in data.iter().enumerate().take(10) {
            assert!(
                (*val - 1.0).abs() < 1e-6,
                "Mismatch at index {i}: expected 1.0, got {val}"
            );
        }
        // Also check last elements
        assert!((data[n - 1] - 1.0).abs() < 1e-6);
    }
}
