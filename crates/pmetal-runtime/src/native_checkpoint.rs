//! A checkpoint's tensors, claimed one module at a time.
//!
//! The native engines load a checkpoint the same way: read every shard, drop
//! what the text model does not use, normalise the key namespace, and then ask
//! for each module by name. A module is a dense linear when the checkpoint has
//! `{module}.weight` alone, and a packed one when `{module}.scales` (and, for
//! the affine mode, `{module}.biases`) sit beside it. MLX models quantization
//! as a property of the layer, so the engines ask for "the linear at this
//! path" and get whichever [`LayerWeight`] the checkpoint holds.
//!
//! Every tensor an engine takes leaves the map, so what is left when it is
//! done is what the engine did not know what to do with. [`Checkpoint::unclaimed`]
//! hands that back: a loader that ignores a tensor silently runs a different
//! model from the one the checkpoint holds, which is how an unsupported
//! feature (attention biases, an indexer) should fail instead.

use crate::{RuntimeError, RuntimeResult};
use std::collections::HashMap;
use std::path::Path;

use crate::InlineArray;
use crate::native_weight::{
    EmbeddingWeight, LayerWeight, MlxQuantization, QuantParams, detect_model_dtype,
};

/// The tensors of a checkpoint directory, and how its packed ones are read.
pub struct Checkpoint {
    tag: &'static str,
    tensors: HashMap<String, InlineArray>,
    quantization: Option<MlxQuantization>,
    dtype: i32,
}

impl Checkpoint {
    /// Read every shard under `model_dir`.
    ///
    /// `normalize` maps a checkpoint key (and the module paths of its
    /// `quantization` block) onto the namespace the engine asks for, and
    /// returns `None` for tensors the engine never loads (vision towers).
    /// `tag` names the engine in error messages.
    pub fn open(
        tag: &'static str,
        model_dir: &Path,
        normalize: impl Fn(&str) -> Option<String>,
    ) -> RuntimeResult<Self> {
        let shards = crate::native_loader::discover_safetensors_shards(model_dir)?;
        let mut tensors = HashMap::new();
        for shard in &shards {
            let path = shard.to_str().ok_or_else(|| {
                RuntimeError::Weights(format!("{tag}: non-UTF-8 shard path: {shard:?}"))
            })?;
            let entries = crate::inline_array::load_safetensors_shard(path).ok_or_else(|| {
                RuntimeError::Weights(format!("{tag}: failed to load shard: {path}"))
            })?;
            for (key, array) in entries {
                let Some(name) = normalize(&key) else {
                    continue;
                };
                // Two layouts normalising onto one key would resolve silently
                // in whatever order the shards happened to be read.
                if tensors.insert(name.clone(), array).is_some() {
                    return Err(RuntimeError::Weights(format!(
                        "{tag}: two checkpoint keys both read as {name}"
                    )));
                }
            }
        }
        if tensors.is_empty() {
            return Err(RuntimeError::Weights(format!(
                "{tag}: no weights in {}",
                model_dir.display()
            )));
        }
        let quantization = crate::native_loader::load_mlx_quantization(model_dir, &normalize)?;
        let dtype = detect_model_dtype(|key| tensors.get(key).map(|a| a.dtype_raw()));
        Ok(Self {
            tag,
            tensors,
            quantization,
            dtype,
        })
    }

    /// The dtype the trunk computes in; see [`detect_model_dtype`].
    pub fn dtype(&self) -> i32 {
        self.dtype
    }

    /// Whether `key` is still in the checkpoint.
    pub fn contains(&self, key: &str) -> bool {
        self.tensors.contains_key(key)
    }

    /// Every remaining key, in no particular order.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    /// The keys nothing has claimed, sorted.
    pub fn unclaimed(&self) -> Vec<&str> {
        let mut keys: Vec<&str> = self.keys().collect();
        keys.sort_unstable();
        keys
    }

    /// Remove and return `key`, or `None`.
    pub fn take_opt(&mut self, key: &str) -> Option<InlineArray> {
        self.tensors.remove(key)
    }

    /// Remove and return `key`; its absence names the nearest keys.
    pub fn take(&mut self, key: &str) -> RuntimeResult<InlineArray> {
        match self.tensors.remove(key) {
            Some(array) => Ok(array),
            None => {
                let suffix = key.rsplit_once('.').map_or(key, |(_, s)| s);
                let mut close: Vec<&str> = self
                    .keys()
                    .filter(|k| k.ends_with(suffix))
                    .take(5)
                    .collect();
                close.sort_unstable();
                Err(RuntimeError::Weights(format!(
                    "{}: missing weight key: {key} (close matches: {close:?})",
                    self.tag
                )))
            }
        }
    }

    /// Replace a tensor, e.g. with its dequantized form.
    pub fn put(&mut self, key: String, array: InlineArray) {
        self.tensors.insert(key, array);
    }

    /// Every tensor, for a loader's own rewrites (FP8 dequantization).
    pub fn tensors_mut(&mut self) -> &mut HashMap<String, InlineArray> {
        &mut self.tensors
    }

    /// How `module`'s packed tensors are read: its own override, else the
    /// file-level block, else MLX's affine defaults. Only consulted for a
    /// module that has scales, so the last case decodes nothing.
    pub fn params_for(&self, module: &str) -> QuantParams {
        self.quantization
            .as_ref()
            .and_then(|q| q.params_for(module).or_else(|| q.default_params()))
            .unwrap_or_else(|| QuantParams::defaults_for(crate::QuantizedMode::Affine))
    }

    /// The linear at `module`, `[in, out]` dense or packed as stored.
    pub fn linear(&mut self, module: &str) -> RuntimeResult<LayerWeight> {
        let weight = self.take(&format!("{module}.weight"))?;
        let scales = self.take_opt(&format!("{module}.scales"));
        let biases = self.take_opt(&format!("{module}.biases"));
        Ok(LayerWeight::new(
            weight,
            scales,
            biases,
            self.params_for(module),
        ))
    }

    /// The embedding table at `module`, dense or packed.
    pub fn embedding(&mut self, module: &str) -> RuntimeResult<EmbeddingWeight> {
        let weight = self.take(&format!("{module}.weight"))?;
        let scales = self.take_opt(&format!("{module}.scales"));
        let biases = self.take_opt(&format!("{module}.biases"));
        Ok(EmbeddingWeight::new(
            weight,
            scales,
            biases,
            self.params_for(module),
        ))
    }

    /// Experts stacked over a leading axis at `module`, stored `[E, out, in]`
    /// like a Linear's weight: `[E, in, out]` when dense (what `gather_mm`
    /// takes), packed as stored otherwise.
    pub fn stacked(&mut self, module: &str) -> RuntimeResult<LayerWeight> {
        let weight = self.take(&format!("{module}.weight"))?;
        let scales = self.take_opt(&format!("{module}.scales"));
        let biases = self.take_opt(&format!("{module}.biases"));
        Ok(self.stacked_from(module, weight, scales, biases))
    }

    /// [`Self::stacked`] for tensors the caller already holds.
    pub fn stacked_from(
        &self,
        module: &str,
        weight: InlineArray,
        scales: Option<InlineArray>,
        biases: Option<InlineArray>,
    ) -> LayerWeight {
        match scales {
            Some(scales) => LayerWeight::Quantized {
                weight,
                scales,
                biases,
                tensor_scale: None,
                params: self.params_for(module),
            },
            None => LayerWeight::Dense(weight.transpose_axes(&[0, 2, 1])),
        }
    }

    /// Stack per-expert tensors `{prefix}.{e}.{name}` (`e` in `0..experts`)
    /// into one `[E, ...]` tensor, removing them. `None` when expert 0 is
    /// absent; a partial set is an error.
    pub fn stack_experts(
        &mut self,
        prefix: &str,
        experts: usize,
        name: &str,
    ) -> RuntimeResult<Option<InlineArray>> {
        if !self.contains(&format!("{prefix}.0.{name}")) {
            return Ok(None);
        }
        let parts = (0..experts)
            .map(|e| self.take(&format!("{prefix}.{e}.{name}")))
            .collect::<Result<Vec<_>, _>>()?;
        let stacked = crate::compat::ops::stack_axis(&parts, 0);
        crate::check_last_error().map_err(|e| {
            RuntimeError::Weights(format!("{}: stacking {prefix}.*.{name}: {e}", self.tag))
        })?;
        Ok(Some(stacked))
    }
}
