//! Architecture-agnostic LoRA loading.
//!
//! This is a thin shell over [`AdaptedModel`], which holds the *same*
//! [`DynamicModel`] the inference path builds and attaches adapters to its
//! projections. It used to be a fourteen-arm enum over per-architecture
//! `*LoraForCausalLM` types, each of which re-derived its architecture from the
//! config rather than reusing the one `pmetal serve` runs. Two forward passes
//! per architecture is how eight of them came to fine-tune a different model
//! from the one they served, and how packed training silently dropped position
//! IDs on eleven of the fourteen.
//!
//! The shell stays because the trainer, the CLI and the GUI all name the type.
//! Everything it does is delegation.
//!
//! # Example
//!
//! ```ignore
//! use pmetal_lora::DynamicLoraModel;
//! use pmetal_core::LoraConfig;
//!
//! let mut model = DynamicLoraModel::from_pretrained(
//!     "/path/to/model",
//!     LoraConfig::default(),
//! )?;
//! let logits = model.forward(&input_ids, None)?;
//! ```

use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

use pmetal_bridge::compat::Array;
use pmetal_bridge::compat::Exception;
use pmetal_core::LoraConfig;
use pmetal_mlx::kv_cache::KVCache;
use pmetal_models::dispatcher::{DynamicModel, DynamicModelLoadOptions};
use pmetal_models::{ModelArchitecture, WeightFormatError};

use crate::{AdaptedModel, LoraError, TrainableModel};

/// A model loaded for LoRA training, whatever its architecture.
pub struct DynamicLoraModel {
    inner: AdaptedModel,
}

impl std::fmt::Debug for DynamicLoraModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DynamicLoraModel::{:?}", self.architecture())
    }
}

impl DynamicLoraModel {
    /// Load a checkpoint and attach adapters.
    ///
    /// Architecture detection, config quirks (Gemma's version flags, the
    /// `text_config` nesting Qwen 3.5 and Gemma 4 use, RoPE parameter
    /// application) and weight loading all happen inside
    /// [`DynamicModel::load`], which is the one place that knows them.
    pub fn from_pretrained(
        model_dir: impl AsRef<Path>,
        lora_config: LoraConfig,
    ) -> Result<Self, DynamicLoraError> {
        let model_dir = model_dir.as_ref();
        let model = DynamicModel::load(model_dir)?;
        tracing::info!("Loaded {} for LoRA training", model.architecture());
        Ok(Self {
            inner: AdaptedModel::attach(model, lora_config)?,
        })
    }

    /// Load from a GGUF file or a directory containing one, then attach
    /// adapters.
    pub fn from_gguf(
        gguf_path: impl AsRef<Path>,
        lora_config: LoraConfig,
    ) -> Result<Self, DynamicLoraError> {
        let gguf_path = gguf_path.as_ref();
        let gguf_file = if gguf_path.is_file() {
            gguf_path.to_path_buf()
        } else {
            std::fs::read_dir(gguf_path)?
                .filter_map(|entry| entry.ok())
                .map(|entry| entry.path())
                .find(|path| {
                    path.extension()
                        .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
                })
                .ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        format!("No .gguf file found in {gguf_path:?}"),
                    )
                })?
        };

        let model = DynamicModel::load_gguf(&gguf_file, DynamicModelLoadOptions::default())?;
        tracing::info!(
            "Loaded {} from GGUF for LoRA training",
            model.architecture()
        );
        Ok(Self {
            inner: AdaptedModel::attach(model, lora_config)?,
        })
    }

    /// The adapted model underneath, for callers that need the inference API.
    pub fn adapted(&self) -> &AdaptedModel {
        &self.inner
    }

    /// Mutable access to the adapted model.
    pub fn adapted_mut(&mut self) -> &mut AdaptedModel {
        &mut self.inner
    }

    /// The base model, adapters attached.
    pub fn model(&self) -> &DynamicModel {
        self.inner.model()
    }

    /// Mutable access to the base model.
    pub fn model_mut(&mut self) -> &mut DynamicModel {
        self.inner.model_mut()
    }

    /// The detected architecture.
    pub fn architecture(&self) -> ModelArchitecture {
        self.inner.architecture()
    }

    /// The architecture as a bare identifier.
    ///
    /// Distinct from `ModelArchitecture`'s `Display`, which is prose ("Gemma
    /// 4"). Callers compare this against a literal, so the spelling is part of
    /// the contract.
    pub fn architecture_name(&self) -> &'static str {
        self.architecture().identifier()
    }

    /// Fold every adapter into its base weight, reversibly.
    pub fn merge_lora(&mut self) -> Result<(), LoraError> {
        self.inner.merge();
        Ok(())
    }

    /// Undo [`merge_lora`](Self::merge_lora).
    pub fn unmerge_lora(&mut self) -> Result<(), LoraError> {
        self.inner.unmerge();
        Ok(())
    }

    /// Quantize merged base linear weights to FP8 E4M3 for inference.
    pub fn quantize_fp8(&mut self) -> Result<(), LoraError> {
        self.inner
            .model_mut()
            .quantize_fp8()
            .map_err(LoraError::Mlx)
    }

    /// The recurrent state cache for hybrid architectures, or `None`.
    pub fn create_mamba_cache(&self) -> Option<pmetal_mlx::kv_cache::MambaCache> {
        self.inner.model().create_mamba_cache()
    }

    /// The attention KV cache for inference.
    ///
    /// A hybrid architecture still needs one for its attention layers even
    /// though its recurrent layers use a separate Mamba cache, so this is
    /// always `Some` where [`create_cache`](TrainableModel::create_cache)
    /// might not be.
    pub fn create_inference_kv_cache(&self, max_seq_len: usize) -> Option<KVCache> {
        Some(self.inner.create_cache(max_seq_len))
    }

    /// Materialise every parameter, so the next forward does not re-evaluate
    /// the load graph.
    pub fn eval_all(&mut self) -> Result<(), LoraError> {
        self.inner.eval_all()
    }

    /// Forward with both caches, for architectures that interleave attention
    /// and recurrent layers. Collapses to the plain cached forward elsewhere.
    pub fn forward_with_hybrid_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        kv_cache: Option<&mut KVCache>,
        mamba_cache: Option<&mut pmetal_mlx::kv_cache::MambaCache>,
    ) -> Result<Array, LoraError> {
        self.inner
            .model_mut()
            .forward_with_hybrid_cache(input_ids, mask, kv_cache, mamba_cache)
            .map_err(LoraError::Mlx)
    }
}

impl pmetal_bridge::compat::ModuleParameters for DynamicLoraModel {
    fn num_parameters(&self) -> usize {
        self.inner.num_parameters()
    }

    fn parameters(&self) -> pmetal_bridge::compat::ModuleParamRef<'_> {
        self.inner.parameters()
    }

    fn parameters_mut(&mut self) -> pmetal_bridge::compat::ModuleParamMut<'_> {
        self.inner.parameters_mut()
    }

    fn trainable_parameters(&self) -> pmetal_bridge::compat::ModuleParamRef<'_> {
        self.inner.trainable_parameters()
    }

    fn freeze_parameters(&mut self, recursive: bool) {
        self.inner.freeze_parameters(recursive)
    }

    fn unfreeze_parameters(&mut self, recursive: bool) {
        self.inner.unfreeze_parameters(recursive)
    }

    fn all_frozen(&self) -> Option<bool> {
        self.inner.all_frozen()
    }

    fn any_frozen(&self) -> Option<bool> {
        self.inner.any_frozen()
    }
}

impl TrainableModel for DynamicLoraModel {
    fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        TrainableModel::forward(&mut self.inner, input_ids, mask)
    }

    fn forward_noised(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        noise_alpha: f32,
    ) -> Result<Array, LoraError> {
        TrainableModel::forward_noised(&mut self.inner, input_ids, mask, noise_alpha)
    }

    fn forward_with_positions(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        position_ids: &Array,
    ) -> Result<Array, LoraError> {
        TrainableModel::forward_with_positions(&mut self.inner, input_ids, mask, position_ids)
    }

    fn supports_packed_positions(&self) -> bool {
        TrainableModel::supports_packed_positions(&self.inner)
    }

    fn forward_with_images(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        pixel_values: Option<&Array>,
    ) -> Result<Array, LoraError> {
        TrainableModel::forward_with_images(&mut self.inner, input_ids, mask, pixel_values)
    }

    fn is_multimodal(&self) -> bool {
        TrainableModel::is_multimodal(&self.inner)
    }

    fn num_trainable_params(&self) -> usize {
        TrainableModel::num_trainable_params(&self.inner)
    }

    fn lora_parameters(&self) -> HashMap<Rc<str>, Array> {
        TrainableModel::lora_parameters(&self.inner)
    }

    fn set_lora_parameters(&mut self, params: &HashMap<Rc<str>, Array>) {
        TrainableModel::set_lora_parameters(&mut self.inner, params)
    }

    fn save_lora_weights(&self, path: impl AsRef<Path>) -> Result<(), LoraError> {
        TrainableModel::save_lora_weights(&self.inner, path)
    }

    fn load_lora_weights(&mut self, path: impl AsRef<Path>) -> Result<(), LoraError> {
        TrainableModel::load_lora_weights(&mut self.inner, path)
    }

    fn enable_gradient_checkpointing(&mut self, layers_per_block: usize) {
        TrainableModel::enable_gradient_checkpointing(&mut self.inner, layers_per_block)
    }

    fn disable_gradient_checkpointing(&mut self) {
        TrainableModel::disable_gradient_checkpointing(&mut self.inner)
    }

    fn supports_gradient_checkpointing(&self) -> bool {
        TrainableModel::supports_gradient_checkpointing(&self.inner)
    }

    fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, LoraError> {
        TrainableModel::forward_with_cache(&mut self.inner, input_ids, mask, cache)
    }

    fn create_cache(&self, max_seq_len: usize) -> Option<KVCache> {
        TrainableModel::create_cache(&self.inner, max_seq_len)
    }

    fn supports_kv_cache(&self) -> bool {
        TrainableModel::supports_kv_cache(&self.inner)
    }

    fn forward_hidden(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
    ) -> Option<Result<Array, LoraError>> {
        TrainableModel::forward_hidden(&mut self.inner, input_ids, mask)
    }

    fn forward_hidden_with_positions(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        position_ids: &Array,
    ) -> Option<Result<Array, LoraError>> {
        TrainableModel::forward_hidden_with_positions(
            &mut self.inner,
            input_ids,
            mask,
            position_ids,
        )
    }

    fn lm_head_weight(&self) -> Option<Array> {
        TrainableModel::lm_head_weight(&self.inner)
    }
}

/// Errors during dynamic LoRA model loading.
#[derive(Debug, thiserror::Error)]
pub enum DynamicLoraError {
    /// MLX exception.
    #[error("MLX error: {0}")]
    Mlx(#[from] Exception),
    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// JSON parsing error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    /// LoRA-specific error.
    #[error("LoRA error: {0}")]
    Lora(#[from] LoraError),
    /// Weight format error.
    #[error("Weight format error: {0}")]
    WeightFormat(#[from] WeightFormatError),
    /// Architecture not implemented for LoRA training.
    #[error("Architecture not implemented for LoRA training: {0}")]
    NotImplemented(ModelArchitecture),
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_bridge::compat::ModuleParameters;

    /// Every architecture the dispatcher builds is now trainable, where the
    /// enum covered fourteen of twenty.
    fn tiny_llama() -> DynamicLoraModel {
        let model = DynamicModel::from_config(
            r#"{
                "model_type": "llama",
                "vocab_size": 128, "hidden_size": 32, "intermediate_size": 64,
                "num_hidden_layers": 2, "num_attention_heads": 4,
                "num_key_value_heads": 2, "head_dim": 8,
                "max_position_embeddings": 64, "tie_word_embeddings": false
            }"#,
        )
        .expect("llama builds");
        DynamicLoraModel {
            inner: AdaptedModel::attach(
                model,
                LoraConfig {
                    r: 4,
                    alpha: 8.0,
                    dropout: 0.0,
                    ..Default::default()
                },
            )
            .expect("attach"),
        }
    }

    #[test]
    fn the_architecture_name_is_the_one_callers_compare_against() {
        // `orchestrator.rs` gates its sequence-length warning on the literal
        // "Qwen3Next", so this spelling is load-bearing.
        assert_eq!(ModelArchitecture::Qwen3Next.identifier(), "Qwen3Next");
        assert_eq!(tiny_llama().architecture_name(), "Llama");
    }

    /// Only the adapters are trainable, and the frozen base is not counted.
    #[test]
    fn trainable_parameters_are_the_adapters() {
        let model = tiny_llama();
        let trainable = model.trainable_parameters();
        assert!(!trainable.is_empty(), "no adapter was reported trainable");
        assert!(
            trainable
                .keys()
                .all(|k| k.ends_with(".lora_a") || k.ends_with(".lora_b")),
            "something other than an adapter is trainable: {:?}",
            trainable.keys().collect::<Vec<_>>()
        );
    }

    /// Packing is on by default, so an architecture that quietly drops position
    /// IDs trains the second sequence in a packed row on positions that
    /// continue from the first. The per-architecture path claimed support and
    /// then fell back to plain `forward` for eleven of its fourteen arms.
    #[test]
    fn packed_positions_reach_the_rotation() {
        let mut model = tiny_llama();
        assert!(model.supports_packed_positions());

        let ids = Array::from_i32_slice_shaped(&[1, 2, 3, 4, 5, 6], &[1, 6]);
        let sequential = Array::from_i32_slice_shaped(&[0, 1, 2, 3, 4, 5], &[1, 6]);
        // Two packed sequences of three, so the second restarts at zero.
        let packed = Array::from_i32_slice_shaped(&[0, 1, 2, 0, 1, 2], &[1, 6]);

        let a = model
            .forward_with_positions(&ids, None, &sequential)
            .expect("sequential");
        let b = model
            .forward_with_positions(&ids, None, &packed)
            .expect("packed");

        let diff = a.subtract(&b);
        assert!(
            diff.multiply(&diff).sum(None).item_f32() > 0.0,
            "restarting the positions changed nothing, so they never reached RoPE"
        );
    }

    #[test]
    fn neftune_noise_reaches_the_embedding() {
        let mut model = tiny_llama();
        let ids = Array::from_i32_slice_shaped(&[1, 2, 3, 4], &[1, 4]);

        let clean = TrainableModel::forward(&mut model, &ids, None).unwrap();
        let noised = TrainableModel::forward_noised(&mut model, &ids, None, 10.0).unwrap();

        let diff = clean.subtract(&noised);
        assert!(
            diff.multiply(&diff).sum(None).item_f32() > 0.0,
            "NEFTune fell back to a plain forward"
        );
    }

    /// Cut cross-entropy needs both halves; neither is architecture-specific
    /// any more.
    #[test]
    fn cut_cross_entropy_gets_what_it_needs() {
        let mut model = tiny_llama();
        let ids = Array::from_i32_slice_shaped(&[1, 2, 3, 4], &[1, 4]);

        let hidden = TrainableModel::forward_hidden(&mut model, &ids, None)
            .expect("forward_hidden is available")
            .expect("forward_hidden succeeds");
        assert_eq!(hidden.shape(), &[1, 4, 32]);

        let head = TrainableModel::lm_head_weight(&model).expect("lm_head is available");
        assert_eq!(head.shape(), &[128, 32]);
    }
}
