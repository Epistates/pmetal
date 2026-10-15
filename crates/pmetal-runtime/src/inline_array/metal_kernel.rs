//! Custom Metal kernels (`mlx.fast.metal_kernel`).
//!
//! [`MetalKernel`] wraps a kernel *body* written in the Metal Shading
//! Language. MLX generates the function signature from the input and output
//! names, the array shapes the body reads (`x_shape`, `x_strides`, `x_ndim`
//! for an input named `x`, when the body mentions them) and the template
//! arguments, compiles it on first use, and caches it. The kernel runs in the
//! same lazy graph as every other op.
//!
//! A custom kernel has no gradient: MLX cannot differentiate through it, so
//! use one for inference paths and keep an op-based twin for training.

use super::{InlineArray, RawBuf};
use crate::compat::Dtype;
use crate::error::{RuntimeError, RuntimeResult, check_last_error};
use std::ffi::{CString, c_char};
use std::mem::MaybeUninit;

bridge_extern! {
    fn pmetal_runtime_metal_kernel(
        outputs: *mut RawBuf,
        n_out: i32,
        name: *const c_char,
        source: *const c_char,
        header: *const c_char,
        in_names: *const *const c_char,
        n_in: i32,
        out_names: *const *const c_char,
        inputs: *const *const RawBuf,
        out_shapes: *const i32,
        out_ndims: *const i32,
        out_dtypes: *const i32,
        grid: *const i32,
        threadgroup: *const i32,
        tmpl_names: *const *const c_char,
        tmpl_kinds: *const i32,
        tmpl_vals: *const i32,
        n_tmpl: i32,
        ensure_row_contiguous: bool,
    );
}

/// A template argument of a Metal kernel: an integer constant or a dtype
/// (usable in the body as a type, e.g. `T`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemplateArg {
    /// An `int` template parameter.
    Int(i32),
    /// A dtype, substituted as the corresponding Metal type.
    Dtype(Dtype),
}

/// A Metal kernel defined by the source of its body.
///
/// Kernels are cached process-wide by `name`, so the name must identify the
/// source: reusing a name for different source text is an error.
#[derive(Debug, Clone)]
pub struct MetalKernel {
    name: CString,
    source: CString,
    header: CString,
    input_names: Vec<CString>,
    output_names: Vec<CString>,
    ensure_row_contiguous: bool,
}

fn c_string(what: &str, s: &str) -> RuntimeResult<CString> {
    CString::new(s).map_err(|_| {
        RuntimeError::CxxException(format!("[metal_kernel] {what} contains a NUL byte"))
    })
}

impl MetalKernel {
    /// Define a kernel.
    ///
    /// * `name` - a unique kernel name.
    /// * `input_names` / `output_names` - the names the body refers to its
    ///   buffers by.
    /// * `source` - the kernel body.
    /// * `header` - declarations placed before the generated function (helper
    ///   functions, `#include`s); may be empty.
    pub fn new(
        name: &str,
        input_names: &[&str],
        output_names: &[&str],
        source: &str,
        header: &str,
    ) -> RuntimeResult<Self> {
        Ok(Self {
            name: c_string("kernel name", name)?,
            source: c_string("kernel source", source)?,
            header: c_string("kernel header", header)?,
            input_names: input_names
                .iter()
                .map(|n| c_string("input name", n))
                .collect::<RuntimeResult<_>>()?,
            output_names: output_names
                .iter()
                .map(|n| c_string("output name", n))
                .collect::<RuntimeResult<_>>()?,
            ensure_row_contiguous: true,
        })
    }

    /// Allow inputs that are not row-contiguous (the body then reads
    /// `x_strides`). The default makes MLX copy them first.
    pub fn allow_strided_inputs(mut self) -> Self {
        self.ensure_row_contiguous = false;
        self
    }

    /// Launch the kernel.
    ///
    /// * `inputs` - one array per input name, in order.
    /// * `outputs` - `(shape, dtype)` of each output, in order.
    /// * `grid` - total thread count per dimension (`dispatchThreads`).
    /// * `threadgroup` - threads per threadgroup.
    /// * `template` - template arguments by name.
    ///
    /// Returns the output arrays, lazily: they are computed when evaluated.
    pub fn call(
        &self,
        inputs: &[&InlineArray],
        outputs: &[(&[i32], Dtype)],
        grid: [i32; 3],
        threadgroup: [i32; 3],
        template: &[(&str, TemplateArg)],
    ) -> RuntimeResult<Vec<InlineArray>> {
        if inputs.len() != self.input_names.len() || outputs.len() != self.output_names.len() {
            return Err(RuntimeError::CxxException(format!(
                "kernel {:?} takes {} inputs and {} outputs, got {} and {}",
                self.name,
                self.input_names.len(),
                self.output_names.len(),
                inputs.len(),
                outputs.len()
            )));
        }
        let in_names: Vec<*const c_char> = self.input_names.iter().map(|s| s.as_ptr()).collect();
        let out_names: Vec<*const c_char> = self.output_names.iter().map(|s| s.as_ptr()).collect();
        let input_ptrs: Vec<*const RawBuf> =
            inputs.iter().map(|a| &a.raw as *const RawBuf).collect();

        let shapes: Vec<i32> = outputs
            .iter()
            .flat_map(|(s, _)| s.iter().copied())
            .collect();
        let ndims: Vec<i32> = outputs.iter().map(|(s, _)| s.len() as i32).collect();
        let dtypes: Vec<i32> = outputs.iter().map(|(_, d)| d.as_i32()).collect();

        let tmpl_names: Vec<CString> = template
            .iter()
            .map(|(n, _)| c_string("template name", n))
            .collect::<RuntimeResult<_>>()?;
        let tmpl_name_ptrs: Vec<*const c_char> = tmpl_names.iter().map(|s| s.as_ptr()).collect();
        let (kinds, vals): (Vec<i32>, Vec<i32>) = template
            .iter()
            .map(|(_, a)| match a {
                TemplateArg::Int(v) => (0, *v),
                TemplateArg::Dtype(d) => (1, d.as_i32()),
            })
            .unzip();

        let mut out: Vec<MaybeUninit<RawBuf>> =
            (0..outputs.len()).map(|_| MaybeUninit::uninit()).collect();
        // SAFETY: every pointer is valid for the call; C++ initialises all
        // `outputs.len()` output slots on every path, with placeholders on error.
        unsafe {
            pmetal_runtime_metal_kernel(
                out.as_mut_ptr().cast(),
                outputs.len() as i32,
                self.name.as_ptr(),
                self.source.as_ptr(),
                self.header.as_ptr(),
                in_names.as_ptr(),
                inputs.len() as i32,
                out_names.as_ptr(),
                input_ptrs.as_ptr(),
                shapes.as_ptr(),
                ndims.as_ptr(),
                dtypes.as_ptr(),
                grid.as_ptr(),
                threadgroup.as_ptr(),
                tmpl_name_ptrs.as_ptr(),
                kinds.as_ptr(),
                vals.as_ptr(),
                template.len() as i32,
                self.ensure_row_contiguous,
            );
        }
        let arrays: Vec<InlineArray> = out
            .into_iter()
            // SAFETY: initialised by the call above.
            .map(|slot| InlineArray {
                raw: unsafe { slot.assume_init() },
            })
            .collect();
        check_last_error()?;
        Ok(arrays)
    }
}
