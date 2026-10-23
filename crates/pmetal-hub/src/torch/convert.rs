//! Turning PyTorch pickles into safetensors files.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use pmetal_core::safetensors_index::{INDEX_FILE, ShardIndex};
use pmetal_core::{PMetalError, Result};

use super::{TorchArchive, TorchTensor};

/// Write `tensors` as a safetensors file at `dst`: the tensors in name order,
/// the header padded to 8 bytes, the whole thing written to a temporary file
/// and renamed into place, so a crash never leaves half a model under the
/// final name.
pub fn write_safetensors(tensors: &[TorchTensor<'_>], dst: &Path) -> Result<()> {
    let mut sorted: Vec<&TorchTensor<'_>> = tensors.iter().collect();
    sorted.sort_by(|a, b| a.name.cmp(&b.name));
    if sorted.windows(2).any(|w| w[0].name == w[1].name) {
        return Err(PMetalError::Serialization(
            "two tensors share a name; cannot write a safetensors file".into(),
        ));
    }

    let mut header = serde_json::Map::new();
    header.insert(
        "__metadata__".to_owned(),
        serde_json::json!({ "format": "pt" }),
    );
    let mut offset = 0u64;
    for tensor in &sorted {
        let end = offset + tensor.data.len() as u64;
        header.insert(
            tensor.name.clone(),
            serde_json::json!({
                "dtype": tensor.dtype.safetensors_name(),
                "shape": tensor.shape,
                "data_offsets": [offset, end],
            }),
        );
        offset = end;
    }
    let mut header =
        serde_json::to_vec(&header).map_err(|e| PMetalError::Serialization(e.to_string()))?;
    while header.len() % 8 != 0 {
        header.push(b' ');
    }

    let tmp = dst.with_extension("safetensors.tmp");
    let write = || -> std::io::Result<()> {
        let mut out = std::io::BufWriter::with_capacity(1 << 20, std::fs::File::create(&tmp)?);
        out.write_all(&(header.len() as u64).to_le_bytes())?;
        out.write_all(&header)?;
        for tensor in &sorted {
            out.write_all(tensor.data)?;
        }
        out.flush()?;
        out.into_inner().map_err(|e| e.into_error())?.sync_all()
    };
    if let Err(e) = write() {
        let _ = std::fs::remove_file(&tmp);
        return Err(e.into());
    }
    std::fs::rename(&tmp, dst)?;
    Ok(())
}

/// The safetensors name for a pickle shard: `pytorch_model-00001-of-00002.bin`
/// becomes `model-00001-of-00002.safetensors`, `pytorch_model.bin` becomes
/// `model.safetensors`, and any other `x.bin` / `x.pt` / `x.pth` becomes
/// `x.safetensors`.
fn safetensors_name(pickle_name: &str) -> String {
    let stem = pickle_name
        .rsplit_once('.')
        .map_or(pickle_name, |(stem, _)| stem);
    let stem = stem
        .strip_prefix("pytorch_model")
        .map_or_else(|| stem.to_owned(), |rest| format!("model{rest}"));
    format!("{stem}.safetensors")
}

/// Convert the PyTorch checkpoint in `dir` (`pytorch_model.bin`, or the shards
/// `pytorch_model.bin.index.json` names) to safetensors files in the same
/// directory, with a `model.safetensors.index.json` for shards. Returns the
/// safetensors files, in shard order. A shard that already has a safetensors
/// file next to it is left alone, so a repeated call converts nothing twice.
///
/// The pickles are read with the restricted interpreter of [`TorchArchive`]:
/// no code in them runs.
pub fn convert_pytorch_dir(dir: &Path) -> Result<Vec<PathBuf>> {
    let index_path = dir.join("pytorch_model.bin.index.json");
    let (shards, weight_map): (Vec<String>, Option<ShardIndex>) = if index_path.exists() {
        let index = ShardIndex::read(&index_path)
            .map_err(|e| PMetalError::Serialization(format!("{}: {e}", index_path.display())))?;
        let shards: Vec<String> = index.shards().into_iter().map(str::to_owned).collect();
        if shards.is_empty() {
            return Err(PMetalError::Serialization(format!(
                "{} names no shards",
                index_path.display()
            )));
        }
        (shards, Some(index))
    } else if dir.join("pytorch_model.bin").exists() {
        (vec!["pytorch_model.bin".to_owned()], None)
    } else {
        return Err(PMetalError::Serialization(format!(
            "no pytorch_model.bin or pytorch_model.bin.index.json in {}",
            dir.display()
        )));
    };

    let mut outputs = Vec::new();
    let mut total_size = 0u64;
    for shard in &shards {
        let dst = dir.join(safetensors_name(shard));
        // Already converted (and intact): nothing to do.
        if let Ok(done) = crate::tensors::summarize_safetensors(&dst) {
            total_size += done.data_bytes();
            outputs.push(dst);
            continue;
        }
        let archive = TorchArchive::open(dir.join(shard))?;
        let tensors = archive.tensors(None)?;
        total_size += tensors.iter().map(|t| t.data.len() as u64).sum::<u64>();
        write_safetensors(&tensors, &dst)?;
        outputs.push(dst);
    }

    if let Some(index) = weight_map {
        let renamed: BTreeMap<String, String> = index
            .weight_map
            .iter()
            .map(|(tensor, shard)| (tensor.clone(), safetensors_name(shard)))
            .collect();
        let path = dir.join(INDEX_FILE);
        ShardIndex::new(renamed, total_size)
            .write(&path)
            .map_err(|e| PMetalError::Serialization(format!("{}: {e}", path.display())))?;
    }
    Ok(outputs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shard_names_follow_the_hub_convention() {
        assert_eq!(safetensors_name("pytorch_model.bin"), "model.safetensors");
        assert_eq!(
            safetensors_name("pytorch_model-00001-of-00003.bin"),
            "model-00001-of-00003.safetensors"
        );
        assert_eq!(
            safetensors_name("adapter_model.bin"),
            "adapter_model.safetensors"
        );
        assert_eq!(safetensors_name("weights.pth"), "weights.safetensors");
    }
}
