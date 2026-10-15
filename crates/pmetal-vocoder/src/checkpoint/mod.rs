//! Reading model checkpoints.
//!
//! NVIDIA releases BigVGAN as PyTorch `.pt` files (`bigvgan_generator.pt`,
//! `bigvgan_discriminator_optimizer.pt`); the community also circulates
//! safetensors conversions. Both load into a [`StateDict`] of float32 arrays
//! under the reference's tensor names, from where a model takes what it needs.
//!
//! The `.pt` reader is self-contained and restricted: it parses the ZIP
//! container and interprets the pickle with an allow-list of globals, so
//! opening an untrusted file cannot execute code.

mod pickle;
mod zip;

use crate::error::{Result, VocoderError};
use crate::nn::StateDict;
use pickle::{Obj, StorageDtype, TensorRef};
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
pub fn load_torch(path: &Path, key: Option<&str>) -> Result<StateDict> {
    let bytes = std::fs::read(path)?;
    let archive = zip::StoredZip::parse(&bytes)?;
    let (root_dir, pkl) = archive.find_suffix("data.pkl").ok_or_else(|| {
        VocoderError::WeightLoad(format!(
            "{}: no data.pkl; not a PyTorch archive",
            path.display()
        ))
    })?;
    let root = pickle::load(pkl)?;

    let selected = match key {
        Some(k) => root
            .get(k)
            .ok_or_else(|| VocoderError::WeightLoad(format!("checkpoint has no `{k}` entry")))?,
        None => root.get("generator").unwrap_or_else(|| root.clone()),
    };
    let items = selected.dict_items().ok_or_else(|| {
        VocoderError::WeightLoad("the selected checkpoint entry is not a dict".into())
    })?;

    let mut tensors = HashMap::new();
    for (name, value) in items {
        let Some(name) = name.as_str() else { continue };
        // Non-tensor entries (e.g. `_metadata`) are not weights.
        if let Obj::Tensor(t) = value {
            tensors.insert(name.to_owned(), materialize(&archive, root_dir, &t, name)?);
        }
    }
    if tensors.is_empty() {
        return Err(VocoderError::WeightLoad(format!(
            "{}: the selected entry holds no tensors",
            path.display()
        )));
    }
    Ok(StateDict::new(tensors))
}

/// Read a tensor's elements as float32 from its storage file.
fn materialize(
    archive: &zip::StoredZip<'_>,
    root_dir: &str,
    t: &TensorRef,
    name: &str,
) -> Result<Array> {
    let fail = |msg: &str| VocoderError::WeightLoad(format!("tensor `{name}`: {msg}"));
    let entry = format!("{root_dir}/data/{}", t.storage.key);
    let bytes = archive
        .entry(&entry)
        .ok_or_else(|| fail(&format!("storage `{entry}` is missing")))?;

    // Only densely packed row-major tensors are produced by `state_dict()`.
    let mut expected = 1usize;
    for (&dim, &stride) in t.shape.iter().zip(&t.stride).rev() {
        if dim > 1 && stride != expected {
            return Err(fail("is not stored contiguously"));
        }
        expected *= dim;
    }
    let numel: usize = t.shape.iter().product();
    let width = t.storage.dtype.size();
    let start = t.offset * width;
    let data = bytes
        .get(start..start + numel * width)
        .ok_or_else(|| fail("runs past the end of its storage"))?;

    let elem = match t.storage.dtype {
        StorageDtype::F32 => Elem::F32,
        StorageDtype::F16 => Elem::F16,
        StorageDtype::Bf16 => Elem::Bf16,
        StorageDtype::F64 => Elem::F64,
        other => return Err(fail(&format!("has non-float dtype {other:?}"))),
    };
    let shape: Vec<i32> = t.shape.iter().map(|&d| d as i32).collect();
    Ok(Array::from_f32_slice(&to_f32(data, elem), &shape))
}
