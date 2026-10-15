//! Reading model checkpoints.
//!
//! NVIDIA releases BigVGAN as PyTorch `.pt` files (`bigvgan_generator.pt`,
//! `bigvgan_discriminator_optimizer.pt`); the community also circulates
//! safetensors conversions. Both load into a [`StateDict`] of float32 arrays
//! under the reference's tensor names, from where a model takes what it needs.
//!
//! The `.pt` reader is `pmetal_hub::torch`: it maps the ZIP container and
//! interprets the pickle with an allow-list of globals, so opening an
//! untrusted file cannot execute code.

use crate::error::{Result, VocoderError};
use crate::nn::StateDict;
use pmetal_hub::torch::{TorchArchive, TorchDtype};
use pmetal_runtime::compat::Array;
use std::collections::HashMap;
use std::path::Path;

/// Load the tensors of a checkpoint file, by extension: `.safetensors`, or a
/// PyTorch archive (`.pt`, `.pth`, `.bin`, ...).
///
/// For a PyTorch archive, `key` selects a nested state dict
/// (`"generator"`, `"mpd"`, `"mrd"`); `None` uses the `"generator"` entry if the
/// file has one and the top-level dict otherwise.
pub fn load_state_dict(path: &Path, key: Option<&str>) -> Result<StateDict> {
    let is_safetensors = path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("safetensors"));
    if is_safetensors {
        load_safetensors(path)
    } else {
        load_torch(path, key)
    }
}

/// Load every tensor of a safetensors file as float32.
pub fn load_safetensors(path: &Path) -> Result<StateDict> {
    let bytes = std::fs::read(path)?;
    let st = safetensors::SafeTensors::deserialize(&bytes)
        .map_err(|e| VocoderError::WeightLoad(format!("{}: {e}", path.display())))?;
    let mut tensors = HashMap::new();
    for (name, view) in st.tensors() {
        let shape: Vec<i32> = view.shape().iter().map(|&d| d as i32).collect();
        let elem = match view.dtype() {
            safetensors::Dtype::F32 => Elem::F32,
            safetensors::Dtype::F16 => Elem::F16,
            safetensors::Dtype::BF16 => Elem::Bf16,
            safetensors::Dtype::F64 => Elem::F64,
            other => {
                return Err(VocoderError::WeightLoad(format!(
                    "`{name}` has unsupported dtype {other:?}"
                )));
            }
        };
        tensors.insert(
            name,
            Array::from_f32_slice(&to_f32(view.data(), elem), &shape),
        );
    }
    Ok(StateDict::new(tensors))
}

/// Floating-point element types a checkpoint can store.
#[derive(Clone, Copy)]
enum Elem {
    F32,
    F16,
    Bf16,
    F64,
}

/// Decode little-endian `data` of element type `elem` to float32.
fn to_f32(data: &[u8], elem: Elem) -> Vec<f32> {
    match elem {
        Elem::F32 => data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect(),
        Elem::F16 => data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| half::f16::from_le_bytes(*b).to_f32())
            .collect(),
        Elem::Bf16 => data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| half::bf16::from_le_bytes(*b).to_f32())
            .collect(),
        Elem::F64 => data
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| f64::from_le_bytes(*b) as f32)
            .collect(),
    }
}

/// Load a PyTorch archive's state dict (see [`load_state_dict`]).
///
/// The archive is memory-mapped and read by `pmetal_hub::torch`, whose
/// unpickler runs nothing: only the tensor rebuilders and `OrderedDict` are
/// understood, anything else is an error.
pub fn load_torch(path: &Path, key: Option<&str>) -> Result<StateDict> {
    let fail = |e: &dyn std::fmt::Display| VocoderError::WeightLoad(e.to_string());
    let archive = TorchArchive::open(path).map_err(|e| fail(&e))?;
    // The released checkpoints nest the generator's weights under `generator`.
    let group = key.or_else(|| {
        archive
            .groups()
            .contains(&"generator")
            .then_some("generator")
    });

    let mut tensors = HashMap::new();
    for t in archive.tensors(group).map_err(|e| fail(&e))? {
        let elem = match t.dtype {
            TorchDtype::F32 => Elem::F32,
            TorchDtype::F16 => Elem::F16,
            TorchDtype::Bf16 => Elem::Bf16,
            TorchDtype::F64 => Elem::F64,
            other => {
                return Err(VocoderError::WeightLoad(format!(
                    "tensor `{}`: has non-float dtype {other:?}",
                    t.name
                )));
            }
        };
        let shape: Vec<i32> = t.shape.iter().map(|&d| d as i32).collect();
        tensors.insert(
            t.name.clone(),
            Array::from_f32_slice(&to_f32(t.data, elem), &shape),
        );
    }
    Ok(StateDict::new(tensors))
}
