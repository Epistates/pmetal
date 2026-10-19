//! Where `pmetal quantize` reads a model's tensors from: one at a time.
//!
//! Quantizing never needs the model in memory, only the tensor in hand. A
//! checkpoint of plain float tensors (safetensors, sharded or not) is read
//! tensor by tensor, straight from the file, as `f32`; nothing else is held.
//! A checkpoint the loader has to unpack first (MLX affine or ModelOpt packed
//! weights, a PyTorch pickle) is loaded whole, as before, and its tensors are
//! handed out one at a time and let go of as they are used.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use anyhow::Context;
use half::{bf16, f16};
use pmetal_runtime::compat::{Array, Dtype};

/// A float tensor of the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TensorMeta {
    pub(crate) name: String,
    pub(crate) shape: Vec<usize>,
}

impl TensorMeta {
    pub(crate) fn elements(&self) -> usize {
        self.shape.iter().product()
    }
}

/// A model's float tensors, readable one at a time.
pub(crate) trait TensorSource {
    /// The float tensors, by name. Tensors of other kinds (integer buffers)
    /// are not quantizable and not listed.
    fn tensors(&self) -> &[TensorMeta];

    /// One tensor's values as `f32`. A source that holds its tensors in memory
    /// gives this one up (it is not read twice).
    fn read_f32(&mut self, name: &str) -> anyhow::Result<Vec<f32>>;
}

/// Names that mark a checkpoint whose weights are packed and need the loader
/// to unpack them before they can be quantized again.
const PACKED_MARKERS: [&str; 8] = [
    ".scales",
    ".biases",
    ".weight_scale",
    ".weight_scale_2",
    ".weight_scale_inv",
    ".qweight",
    ".qzeros",
    ".g_idx",
];

#[derive(Debug, Clone, serde::Deserialize)]
struct HeaderEntry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

/// Plain-float safetensors, read from the files tensor by tensor.
#[derive(Debug)]
pub(crate) struct SafetensorsSource {
    files: Vec<PathBuf>,
    /// name -> (file, dtype, offset of the data in the file, length in bytes)
    index: HashMap<String, (usize, FloatKind, u64, u64)>,
    metas: Vec<TensorMeta>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FloatKind {
    F32,
    F16,
    Bf16,
    F64,
}

impl FloatKind {
    fn parse(dtype: &str) -> Option<Self> {
        Some(match dtype {
            "F32" => Self::F32,
            "F16" => Self::F16,
            "BF16" => Self::Bf16,
            "F64" => Self::F64,
            _ => return None,
        })
    }

    fn width(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
            Self::F64 => 8,
        }
    }
}

/// The header of one safetensors file: its tensors, and where the data starts.
fn read_header(path: &Path) -> anyhow::Result<(HashMap<String, HeaderEntry>, u64)> {
    let mut file =
        File::open(path).with_context(|| format!("could not open '{}'", path.display()))?;
    let mut len = [0u8; 8];
    file.read_exact(&mut len)
        .with_context(|| format!("'{}' is shorter than a safetensors header", path.display()))?;
    let len = u64::from_le_bytes(len);
    anyhow::ensure!(
        len < (1 << 30),
        "'{}': a header of {len} bytes is not a safetensors file",
        path.display()
    );
    let mut raw = vec![0u8; len as usize];
    file.read_exact(&mut raw)
        .with_context(|| format!("'{}': the header is cut off", path.display()))?;
    let mut entries: HashMap<String, serde_json::Value> = serde_json::from_slice(&raw)
        .with_context(|| format!("'{}': the header is not JSON", path.display()))?;
    entries.remove("__metadata__");
    let mut tensors = HashMap::with_capacity(entries.len());
    for (name, value) in entries {
        let entry: HeaderEntry = serde_json::from_value(value).with_context(|| {
            format!(
                "'{}': tensor '{name}' has a malformed entry",
                path.display()
            )
        })?;
        tensors.insert(name, entry);
    }
    Ok((tensors, 8 + len))
}

impl SafetensorsSource {
    /// The source for `files`, or `None` when the checkpoint holds packed
    /// weights that need unpacking (see [`open`]).
    pub(crate) fn open(files: &[PathBuf]) -> anyhow::Result<Option<Self>> {
        let mut index = HashMap::new();
        let mut metas = Vec::new();
        for (file_index, path) in files.iter().enumerate() {
            let size = std::fs::metadata(path)?.len();
            let (header, data_start) = read_header(path)?;
            for (name, entry) in header {
                if PACKED_MARKERS.iter().any(|marker| name.ends_with(marker)) {
                    return Ok(None);
                }
                let Some(kind) = FloatKind::parse(&entry.dtype) else {
                    // An integer buffer: not a weight to quantize. (Packed
                    // weights are caught by their sidecars above.)
                    tracing::debug!("Skipping non-float tensor {name} ({})", entry.dtype);
                    continue;
                };
                let [begin, end] = entry.data_offsets;
                let elements: usize = entry.shape.iter().product();
                anyhow::ensure!(
                    end >= begin
                        && end - begin == (elements * kind.width()) as u64
                        && data_start + end <= size,
                    "'{}': tensor '{name}' says it is {:?} {} but its data ({begin}..{end}) does \
                     not fit the file",
                    path.display(),
                    entry.shape,
                    entry.dtype
                );
                anyhow::ensure!(
                    index
                        .insert(
                            name.clone(),
                            (file_index, kind, data_start + begin, end - begin)
                        )
                        .is_none(),
                    "tensor '{name}' is in more than one file"
                );
                metas.push(TensorMeta {
                    name,
                    shape: entry.shape,
                });
            }
        }
        metas.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(Some(Self {
            files: files.to_vec(),
            index,
            metas,
        }))
    }
}

impl TensorSource for SafetensorsSource {
    fn tensors(&self) -> &[TensorMeta] {
        &self.metas
    }

    fn read_f32(&mut self, name: &str) -> anyhow::Result<Vec<f32>> {
        let &(file, kind, offset, length) = self
            .index
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("tensor '{name}' is not in the model"))?;
        let path = &self.files[file];
        let mut handle =
            File::open(path).with_context(|| format!("could not open '{}'", path.display()))?;
        handle.seek(SeekFrom::Start(offset))?;
        let mut bytes = vec![0u8; length as usize];
        handle
            .read_exact(&mut bytes)
            .with_context(|| format!("'{}': reading tensor '{name}'", path.display()))?;
        Ok(decode(&bytes, kind))
    }
}

/// `bytes` as `f32`.
fn decode(bytes: &[u8], kind: FloatKind) -> Vec<f32> {
    match kind {
        FloatKind::F32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
        FloatKind::F16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f16::from_le_bytes(*c).to_f32())
            .collect(),
        FloatKind::Bf16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| bf16::from_le_bytes(*c).to_f32())
            .collect(),
        FloatKind::F64 => bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_le_bytes(*c) as f32)
            .collect(),
    }
}

/// A model the loader unpacked, held whole: the fallback for checkpoints that
/// are not plain float safetensors.
pub(crate) struct LoadedSource {
    weights: HashMap<String, Array>,
    metas: Vec<TensorMeta>,
}

impl LoadedSource {
    fn load(dir: &Path) -> anyhow::Result<Self> {
        let weights = pmetal_models::loader::load_weights(dir)
            .map_err(|e| anyhow::anyhow!("failed to load the weights: {e}"))?;
        let mut metas: Vec<TensorMeta> = weights
            .iter()
            .filter(|(_, tensor)| {
                matches!(
                    tensor.dtype(),
                    Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16
                )
            })
            .map(|(name, tensor)| TensorMeta {
                name: name.clone(),
                shape: tensor.shape().iter().map(|&d| d as usize).collect(),
            })
            .collect();
        metas.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(Self { weights, metas })
    }
}

impl TensorSource for LoadedSource {
    fn tensors(&self) -> &[TensorMeta] {
        &self.metas
    }

    fn read_f32(&mut self, name: &str) -> anyhow::Result<Vec<f32>> {
        let tensor = self
            .weights
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("tensor '{name}' is not in the model"))?;
        let float32 = tensor.as_dtype(Dtype::Float32.as_i32());
        float32.eval();
        let values = float32.as_slice::<f32>().to_vec();
        // Read once; the array's buffer is let go with it.
        self.weights.remove(name);
        Ok(values)
    }
}

/// The tensor source for the model in `dir`.
pub(crate) fn open(dir: &Path) -> anyhow::Result<Box<dyn TensorSource>> {
    let report = pmetal_hub::verify_model_dir(dir);
    if report.format == Some(pmetal_hub::WeightFormat::Safetensors) && report.is_complete() {
        if let Some(source) = SafetensorsSource::open(&report.weight_files)? {
            tracing::info!(
                tensors = source.tensors().len(),
                "Reading the weights tensor by tensor"
            );
            return Ok(Box::new(source));
        }
        tracing::info!("The checkpoint holds packed weights; loading and unpacking it whole");
    }
    Ok(Box::new(LoadedSource::load(dir)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{Dtype as StDtype, TensorView, serialize};

    fn write(path: &Path, tensors: &[(&str, StDtype, Vec<usize>, Vec<u8>)]) {
        let views: Vec<(String, TensorView<'_>)> = tensors
            .iter()
            .map(|(name, dtype, shape, data)| {
                (
                    name.to_string(),
                    TensorView::new(*dtype, shape.clone(), data).unwrap(),
                )
            })
            .collect();
        std::fs::write(path, serialize(views, None).unwrap()).unwrap();
    }

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| bf16::from_f32(*v).to_le_bytes())
            .collect()
    }

    #[test]
    fn plain_float_tensors_are_read_one_at_a_time_as_f32() {
        let dir = tempfile::tempdir().unwrap();
        let first = dir.path().join("a.safetensors");
        let second = dir.path().join("b.safetensors");
        write(
            &first,
            &[
                (
                    "w",
                    StDtype::BF16,
                    vec![2, 2],
                    bf16_bytes(&[1.0, -2.0, 0.5, 4.0]),
                ),
                (
                    "pos_ids",
                    StDtype::I64,
                    vec![2],
                    [0i64, 1].iter().flat_map(|v| v.to_le_bytes()).collect(),
                ),
            ],
        );
        write(
            &second,
            &[(
                "x",
                StDtype::F32,
                vec![3],
                [1.5f32, 2.5, -3.5]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
            )],
        );
        let mut source = SafetensorsSource::open(&[first, second])
            .unwrap()
            .expect("plain floats stream");
        let listed: Vec<(&str, Vec<usize>)> = source
            .tensors()
            .iter()
            .map(|m| (m.name.as_str(), m.shape.clone()))
            .collect();
        assert_eq!(listed, [("w", vec![2, 2]), ("x", vec![3])]);
        assert_eq!(source.read_f32("w").unwrap(), [1.0, -2.0, 0.5, 4.0]);
        assert_eq!(source.read_f32("x").unwrap(), [1.5, 2.5, -3.5]);
        assert!(source.read_f32("pos_ids").is_err());
    }

    #[test]
    fn a_checkpoint_with_packed_weights_is_left_to_the_loader() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("model.safetensors");
        write(
            &file,
            &[
                ("l.weight", StDtype::U32, vec![2], vec![0; 8]),
                ("l.scales", StDtype::BF16, vec![1], bf16_bytes(&[1.0])),
            ],
        );
        assert!(SafetensorsSource::open(&[file]).unwrap().is_none());
    }

    #[test]
    fn a_tensor_whose_data_does_not_fit_its_file_is_an_error_naming_it() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("model.safetensors");
        write(&file, &[("w", StDtype::F32, vec![4], vec![0; 16])]);
        // Cut the data short.
        let bytes = std::fs::read(&file).unwrap();
        std::fs::write(&file, &bytes[..bytes.len() - 4]).unwrap();
        let err = SafetensorsSource::open(&[file]).unwrap_err().to_string();
        assert!(err.contains("'w'") && err.contains("does not fit"), "{err}");
    }
}
