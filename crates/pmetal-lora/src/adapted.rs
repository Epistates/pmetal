//! LoRA as a pass over the model the inference path already builds.
//!
//! The per-architecture `*LoraForCausalLM` types this replaces each re-derive
//! their architecture from its `Config`, which is how eight of them came to
//! compute something different from the model they would be served as. An
//! [`AdaptedModel`] holds a [`DynamicModel`] — the *same* type `pmetal serve`
//! runs — and attaches [`LoraAdapter`](pmetal_bridge::compat::LoraAdapter)s to
//! its projections. There is one forward pass, so there is nothing to drift.
//!
//! This is the shape PEFT and mlx-lm use: build the model, then walk it and
//! adapt the layers named by `target_modules`.

use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

use pmetal_bridge::compat::{
    Array, Exception, ModuleParameters, ModuleParametersExt, NestedValue, VisitLinears,
};
use pmetal_core::LoraConfig;
use pmetal_models::dispatcher::DynamicModel;

use crate::{LoraError, load_safetensors_map, save_safetensors_map};

/// Prefix the dispatcher's parameter tree puts in front of the decoder stack.
///
/// Adapter files written by the per-architecture path omit it, so it is stripped
/// on save and tolerated on load.
const TRUNK_PREFIX: &str = "model.";

/// A loaded model with low-rank adapters on its targeted projections.
pub struct AdaptedModel {
    model: DynamicModel,
    config: LoraConfig,
    /// Parameter paths of the projections carrying an adapter, in walk order.
    adapted: Vec<String>,
}

impl std::fmt::Debug for AdaptedModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdaptedModel")
            .field("model", &self.model)
            .field("adapted_projections", &self.adapted.len())
            .finish()
    }
}

impl AdaptedModel {
    /// Attach adapters to a model that is already built and loaded.
    ///
    /// Every projection whose *name* — the last segment of its path — appears in
    /// `config.target_modules` gets one. An empty `target_modules` adapts every
    /// projection, matching [`crate::effective_rank`].
    pub fn attach(mut model: DynamicModel, config: LoraConfig) -> Result<Self, LoraError> {
        let rank = config.r as i32;
        if rank <= 0 {
            return Err(LoraError::InvalidState(format!(
                "LoRA rank must be positive, got {rank}"
            )));
        }

        let targets = &config.target_modules;
        let mut adapted = Vec::new();
        let mut failure: Option<Exception> = None;

        let use_dora = config.use_dora;
        let dropout = config.dropout;
        let alpha = config.alpha;
        let use_rslora = config.use_rslora;

        model.visit_linears_mut("", &mut |path, linear| {
            if failure.is_some() || !is_targeted(path, targets) {
                return;
            }
            // Read the base weight before attaching: DoRA seeds its magnitude
            // from it, and the `&mut` to the adapter would rule that out after.
            let weight = use_dora.then(|| linear.weight.value.clone());
            match linear.attach_lora(rank, alpha, use_rslora) {
                Ok(adapter) => {
                    adapter.dropout = dropout;
                    if let Some(weight) = weight {
                        adapter.set_dora(&weight);
                    }
                    adapted.push(path.to_string());
                }
                Err(e) => failure = Some(e),
            }
        });

        if let Some(e) = failure {
            return Err(LoraError::Mlx(e));
        }

        if adapted.is_empty() {
            return Err(LoraError::InvalidState(format!(
                "no projection matched target_modules {targets:?}; the model exposes \
                 none of those names"
            )));
        }

        Ok(Self {
            model,
            config,
            adapted,
        })
    }

    /// Load a checkpoint through the production dispatcher, then adapt it.
    pub fn from_pretrained(
        model_dir: impl AsRef<Path>,
        config: LoraConfig,
    ) -> Result<Self, LoraError> {
        let model = DynamicModel::load(model_dir.as_ref()).map_err(LoraError::Mlx)?;
        Self::attach(model, config)
    }

    /// The wrapped model, for callers that need the inference API.
    pub fn model(&self) -> &DynamicModel {
        &self.model
    }

    /// Mutable access to the wrapped model.
    pub fn model_mut(&mut self) -> &mut DynamicModel {
        &mut self.model
    }

    /// Paths of the projections carrying an adapter.
    pub fn adapted_projections(&self) -> &[String] {
        &self.adapted
    }

    /// The config the adapters were built from.
    pub fn config(&self) -> &LoraConfig {
        &self.config
    }

    /// Turn adapter dropout on or off across the model.
    pub fn set_training(&mut self, training: bool) {
        self.model
            .visit_linears_mut("", &mut |_, linear| linear.set_adapter_training(training));
    }

    /// Fold every adapter into its base weight, leaving a plain dense model.
    ///
    /// This is what `pmetal fuse` wants: afterwards the model is byte-for-byte
    /// servable without pmetal-lora in the picture.
    pub fn merge(&mut self) {
        self.model
            .visit_linears_mut("", &mut |_, linear| linear.merge_lora());
        self.adapted.clear();
    }

    /// Forward pass. Identical to what inference runs, because it *is* what
    /// inference runs.
    pub fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        self.model.forward(input_ids, mask).map_err(LoraError::Mlx)
    }

    /// Adapter tensors, keyed the way an adapter file is.
    pub fn lora_parameters(&self) -> HashMap<Rc<str>, Array> {
        self.model
            .flatten_params()
            .into_iter()
            .filter(|(path, _)| is_adapter_key(path))
            .map(|(path, value)| (Rc::from(adapter_key(&path).as_str()), value))
            .collect()
    }

    /// Overwrite adapter tensors from a map keyed as [`lora_parameters`] emits.
    ///
    /// Keys that match nothing are ignored, and keys carrying the older
    /// `model.`-prefixed form are accepted too.
    ///
    /// [`lora_parameters`]: Self::lora_parameters
    pub fn set_lora_parameters(&mut self, params: &HashMap<Rc<str>, Array>) {
        let mut tree = self.model.flatten_params_mut();
        for (path, slot) in tree.iter_mut() {
            if !is_adapter_key(path) {
                continue;
            }
            let key = adapter_key(path);
            if let Some(value) = params
                .get(key.as_str())
                .or_else(|| params.get(path.as_str()))
            {
                **slot = value.clone();
            }
        }
    }

    /// Trainable element count: the adapters, and nothing else.
    pub fn num_trainable_params(&self) -> usize {
        self.lora_parameters()
            .values()
            .map(|a| a.shape().iter().map(|d| *d as usize).product::<usize>())
            .sum()
    }

    /// Write the adapters to a safetensors file.
    pub fn save_lora_weights(&self, path: impl AsRef<Path>) -> Result<(), LoraError> {
        save_safetensors_map(path, &self.lora_parameters())
    }

    /// Read adapters back from a safetensors file.
    pub fn load_lora_weights(&mut self, path: impl AsRef<Path>) -> Result<(), LoraError> {
        let loaded = load_safetensors_map(path)?;
        let keyed: HashMap<Rc<str>, Array> = loaded
            .into_iter()
            .map(|(k, v)| (Rc::from(k.as_str()), v))
            .collect();
        self.set_lora_parameters(&keyed);
        Ok(())
    }

    /// Materialise every parameter, so the next forward does not re-evaluate the
    /// load graph.
    pub fn eval_all(&self) -> Result<(), LoraError> {
        self.model.eval().map_err(LoraError::Mlx)
    }
}

/// Whether a projection at `path` should carry an adapter.
///
/// Matches on the final path segment, which is the projection's name as a
/// checkpoint and `target_modules` both spell it (`q_proj`, `gate_proj`).
fn is_targeted(path: &str, targets: &[String]) -> bool {
    if targets.is_empty() {
        return true;
    }
    let name = path.rsplit('.').next().unwrap_or(path);
    targets.iter().any(|t| t == name)
}

fn is_adapter_key(path: &str) -> bool {
    path.ends_with(".lora_a") || path.ends_with(".lora_b") || path.ends_with(".lora_magnitude")
}

/// Drop the dispatcher's trunk prefix so the key matches what the
/// per-architecture path wrote.
fn adapter_key(path: &str) -> String {
    path.strip_prefix(TRUNK_PREFIX).unwrap_or(path).to_string()
}

/// Everything the trainer asks of a model it is fine-tuning.
///
/// Every method here already exists as an inherent method, because they are
/// operations on the adapters rather than on the architecture. That is the
/// point: the per-architecture types had to reimplement all of this thirty
/// times over.
impl crate::TrainableModel for AdaptedModel {
    fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        AdaptedModel::forward(self, input_ids, mask)
    }

    fn num_trainable_params(&self) -> usize {
        AdaptedModel::num_trainable_params(self)
    }

    fn lora_parameters(&self) -> HashMap<Rc<str>, Array> {
        AdaptedModel::lora_parameters(self)
    }

    fn set_lora_parameters(&mut self, params: &HashMap<Rc<str>, Array>) {
        AdaptedModel::set_lora_parameters(self, params)
    }

    fn save_lora_weights(&self, path: impl AsRef<Path>) -> Result<(), LoraError> {
        AdaptedModel::save_lora_weights(self, path)
    }

    fn load_lora_weights(&mut self, path: impl AsRef<Path>) -> Result<(), LoraError> {
        AdaptedModel::load_lora_weights(self, path)
    }

    fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut pmetal_mlx::kv_cache::KVCache>,
    ) -> Result<Array, LoraError> {
        self.model
            .forward_with_cache(input_ids, mask, cache)
            .map_err(LoraError::Mlx)
    }
}

/// Delegates straight through to the wrapped model, so the optimizer and the
/// checkpointer see the same parameter tree the loader wrote.
impl ModuleParameters for AdaptedModel {
    fn num_parameters(&self) -> usize {
        self.model.num_parameters()
    }

    fn parameters(&self) -> pmetal_bridge::compat::ModuleParamRef<'_> {
        self.model.parameters()
    }

    fn parameters_mut(&mut self) -> pmetal_bridge::compat::ModuleParamMut<'_> {
        self.model.parameters_mut()
    }

    /// The adapters, and nothing else.
    ///
    /// An adapted `Linear` already withholds its frozen weight, but a projection
    /// that `target_modules` skipped is still an ordinary trainable layer as far
    /// as it knows, and so are the norms and embeddings. Freezing the base is
    /// the adapted model's decision, not the layer's -- the same split PEFT
    /// makes when it calls `freeze()` on the model and then unfreezes the
    /// adapters.
    ///
    /// Returned flat, with dotted paths, which is how a consumer sees it after
    /// flattening anyway.
    fn trainable_parameters(&self) -> pmetal_bridge::compat::ModuleParamRef<'_> {
        fn collect<'a>(
            tree: &pmetal_bridge::compat::ModuleParamRef<'a>,
            prefix: &str,
            out: &mut pmetal_bridge::compat::ModuleParamRef<'a>,
        ) {
            for (name, value) in tree {
                let path = pmetal_bridge::compat::child_path(prefix, name);
                match value {
                    pmetal_bridge::compat::NestedValue::Value(array) => {
                        if is_adapter_key(&path) {
                            out.insert(Rc::from(path.as_str()), NestedValue::Value(array));
                        }
                    }
                    pmetal_bridge::compat::NestedValue::Map(inner) => collect(inner, &path, out),
                }
            }
        }

        let tree = self.model.parameters();
        let mut out = pmetal_bridge::compat::ModuleParamRef::new();
        collect(&tree, "", &mut out);
        out
    }
}
