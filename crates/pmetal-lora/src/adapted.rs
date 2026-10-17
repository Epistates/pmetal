//! LoRA as a pass over the model the inference path already builds.
//!
//! The per-architecture `*LoraForCausalLM` types this replaces each re-derive
//! their architecture from its `Config`, which is how eight of them came to
//! compute something different from the model they would be served as. An
//! [`AdaptedModel`] holds a [`DynamicModel`] — the *same* type `pmetal serve`
//! runs — and attaches [`LoraAdapter`](pmetal_runtime::compat::LoraAdapter)s to
//! its projections. There is one forward pass, so there is nothing to drift.
//!
//! This is the shape PEFT uses: build the model, then walk it and
//! adapt the layers named by `target_modules`.

use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

use pmetal_core::{LoraConfig, LoraInit};
use pmetal_models::dispatcher::DynamicModel;
use pmetal_runtime::compat::{
    Array, ModuleParameters, ModuleParametersExt, NestedValue, VisitLinears,
};

use crate::init::initialise;
use crate::peft::{
    ADAPTER_WEIGHTS, LEGACY_ADAPTER_WEIGHTS, PEFT_PREFIX, find_adapter_weights, hf_module_name,
    peft_tensor_name, write_adapter_config,
};
use crate::{
    LoraError, PackedBase, QLoraConfig, load_safetensors_map, quantize_base, save_safetensors_map,
};

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
    /// `config.target_modules` gets one. An empty `target_modules` (the
    /// default) or `all-linear` adapts every projection except the output
    /// layer (the LM head), as PEFT's `all-linear` does; name it (`lm_head`)
    /// to adapt it. MoE routers and routed experts are never adapted.
    ///
    /// `config.init_lora_weights` says how the adapters start. PiSSA and
    /// OLoRA move part of each adapted weight into its adapter and need a
    /// dense base; LoftQ is fitted to a quantizer and goes through
    /// [`attach_quantized`](Self::attach_quantized).
    pub fn attach(model: DynamicModel, config: LoraConfig) -> Result<Self, LoraError> {
        Self::attach_fitted(model, config, None)
    }

    /// Pack the base for QLoRA and attach adapters: the whole of QLoRA, in the
    /// order its initialisation needs.
    ///
    /// With the standard initialisation the base is packed first, so a DoRA
    /// magnitude is seeded from the weight each layer computes with. PiSSA and
    /// OLoRA split the *dense* weight and pack the residual they leave (QPiSSA
    /// and QOLoRA: the principal part stays out of the quantizer), and LoftQ
    /// fits the adapter to the quantizer's round trip, so for those the
    /// adapters come first and the pack second.
    pub fn attach_quantized(
        mut model: DynamicModel,
        config: LoraConfig,
        qlora: &QLoraConfig,
    ) -> Result<(Self, PackedBase), LoraError> {
        qlora.validate()?;
        if config.init_lora_weights == LoraInit::Standard {
            let packed = quantize_base(&mut model, qlora)?;
            return Ok((Self::attach(model, config)?, packed));
        }
        let mut adapted = Self::attach_fitted(model, config, Some(qlora))?;
        let packed = quantize_base(&mut adapted.model, qlora)?;
        Ok((adapted, packed))
    }

    /// [`attach`](Self::attach), with the quantizer LoftQ fits its adapter to.
    fn attach_fitted(
        mut model: DynamicModel,
        config: LoraConfig,
        quantizer: Option<&QLoraConfig>,
    ) -> Result<Self, LoraError> {
        let rank = config.r as i32;
        if rank <= 0 {
            return Err(LoraError::InvalidState(format!(
                "LoRA rank must be positive, got {rank}"
            )));
        }

        // `all-linear` is the spelling PEFT uses for an empty list.
        let targets: &[String] = if config.targets_all_linear() {
            &[]
        } else {
            &config.target_modules
        };
        let output = if targets.is_empty() || quantizer.is_some() {
            model.output_projection_path()
        } else {
            None
        };
        let mut adapted = Vec::new();
        let mut failure: Option<LoraError> = None;

        let use_dora = config.use_dora;
        let dropout = config.dropout;
        let alpha = config.alpha;
        let use_rslora = config.use_rslora;
        let init = config.init_lora_weights;

        model.visit_linears_mut("", &mut |path, linear| {
            if failure.is_some() || !is_targeted(path, targets, output.as_deref()) {
                return;
            }
            // Read the base weight before attaching: DoRA seeds its magnitude
            // from it, and the `&mut` to the adapter would rule that out after.
            // Unpacked, when the base is packed for QLoRA.
            let weight = use_dora.then(|| linear.dense_weight());
            match linear.attach_lora(rank, alpha, use_rslora) {
                Ok(adapter) => {
                    adapter.dropout = dropout;
                    if let Some(weight) = weight {
                        adapter.set_dora(&weight);
                    }
                    adapted.push(path.to_string());
                }
                Err(e) => {
                    failure = Some(LoraError::Mlx(e));
                    return;
                }
            }
            // A projection QLoRA leaves unpacked (the head, a narrow one) has
            // no quantization error for LoftQ to correct.
            let init = match quantizer {
                Some(q) if init == LoraInit::Loftq && !q.packs(path, linear, output.as_deref()) => {
                    LoraInit::Standard
                }
                _ => init,
            };
            let dtype = linear.weight.value.dtype();
            let round_trip = quantizer.map(|q| {
                move |dense: &Array| q.round_trip(&dense.as_dtype(dtype.as_i32()))
            });
            let fitted = initialise(
                linear,
                init,
                round_trip
                    .as_ref()
                    .map(|f| f as &dyn Fn(&Array) -> Result<Array, LoraError>),
            );
            if let Err(e) = fitted {
                failure = Some(e);
            }
        });

        if let Some(e) = failure {
            return Err(e);
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

    /// Fold every adapter into its base weight, keeping them attached.
    ///
    /// Reversible with [`unmerge`](Self::unmerge). A merged model runs one
    /// matmul per projection instead of three, which is worth it for an
    /// evaluation pass mid-run.
    pub fn merge(&mut self) {
        self.model
            .visit_linears_mut("", &mut |_, linear| linear.merge_lora());
    }

    /// Undo [`merge`](Self::merge) and resume training.
    pub fn unmerge(&mut self) {
        self.model
            .visit_linears_mut("", &mut |_, linear| linear.unmerge_lora());
    }

    /// Fold every adapter in and drop it, leaving a plain dense model.
    ///
    /// This is what `pmetal fuse` produces: servable without pmetal-lora in the
    /// picture, and not reversible.
    pub fn fuse(&mut self) {
        self.model
            .visit_linears_mut("", &mut |_, linear| linear.fuse_lora());
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

    /// The adapted modules by full name in the Hugging Face model, sorted: how
    /// PEFT's own `adapter_config.json` lists what `all-linear` matched. A name
    /// is exact where a bare projection name would also match the modules a
    /// model has more of than this one adapts (a MoE layer's experts).
    pub fn adapter_target_modules(&self) -> Vec<String> {
        let names: std::collections::BTreeSet<String> =
            self.adapted.iter().map(|path| hf_module_name(path)).collect();
        names.into_iter().collect()
    }

    /// Write the adapters to a safetensors file in PEFT's layout
    /// (`base_model.model.<module>.lora_A.weight`, ...; see [`crate::peft`]).
    pub fn save_lora_weights(&self, path: impl AsRef<Path>) -> Result<(), LoraError> {
        let params = self.lora_parameters();
        let mut out: HashMap<String, Array> = HashMap::new();
        for module in &self.adapted {
            let stem = adapter_key(module);
            let part = |suffix: &str| params.get(format!("{stem}.{suffix}").as_str());
            let missing = || LoraError::InvalidState(format!("`{module}` has no adapter tensors"));
            let (a, b) = (part("lora_a").ok_or_else(missing)?, part("lora_b").ok_or_else(missing)?);
            out.insert(peft_tensor_name(module, "lora_A.weight"), a.clone());
            out.insert(peft_tensor_name(module, "lora_B.weight"), b.clone());
            if let Some(magnitude) = part("lora_magnitude") {
                // PEFT holds the DoRA magnitude as one value per output row.
                let rows = magnitude.shape().iter().product::<i32>();
                out.insert(
                    peft_tensor_name(module, "lora_magnitude_vector"),
                    magnitude.reshape(&[rows]),
                );
            }
        }
        save_safetensors_map(path, &out)
    }

    /// Write an adapter directory PEFT can load: `adapter_model.safetensors`
    /// and `adapter_config.json`, which names `base_model` when it is known.
    pub fn save_adapter(
        &self,
        dir: impl AsRef<Path>,
        base_model: Option<&str>,
    ) -> Result<(), LoraError> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)
            .map_err(|e| LoraError::Io(format!("{}: {e}", dir.display())))?;
        self.save_lora_weights(dir.join(ADAPTER_WEIGHTS))?;
        write_adapter_config(dir, &self.config, &self.adapter_target_modules(), base_model)
    }

    /// Read adapters back: a PEFT adapter directory or weights file, or one in
    /// the layout earlier releases wrote.
    ///
    /// Fails if the file holds an adapter tensor this model has no slot for,
    /// or one of another shape (a different rank): dropping or reshaping it
    /// would serve a different model than the one trained.
    pub fn load_lora_weights(&mut self, path: impl AsRef<Path>) -> Result<(), LoraError> {
        let path = path.as_ref();
        let file = find_adapter_weights(path).ok_or_else(|| {
            LoraError::Io(format!(
                "no {ADAPTER_WEIGHTS} (or {LEGACY_ADAPTER_WEIGHTS}) at {}",
                path.display()
            ))
        })?;
        let loaded = load_safetensors_map(&file)?;
        let ours = self.lora_parameters();
        let modules = self.module_index();

        let mut keyed: HashMap<Rc<str>, Array> = HashMap::new();
        let mut unmatched: Vec<String> = Vec::new();
        for (name, value) in loaded {
            let Some(key) = native_key(&name, &modules, &ours) else {
                if name.contains("lora_") || name.starts_with(PEFT_PREFIX) {
                    unmatched.push(name);
                }
                continue;
            };
            let slot = &ours[key.as_str()];
            // PEFT's DoRA magnitude is `[out]`; ours is `[out, 1]`.
            let value = if value.shape() != slot.shape()
                && value.shape().iter().product::<i32>() == slot.shape().iter().product::<i32>()
                && key.ends_with(".lora_magnitude")
            {
                value.reshape(slot.shape())
            } else {
                value
            };
            if value.shape() != slot.shape() {
                return Err(LoraError::ShapeMismatch(format!(
                    "adapter tensor {name} is {:?}, this model's {key} is {:?}; the adapter was \
                     trained at another rank or width, so build the model with the adapter's \
                     adapter_config.json",
                    value.shape(),
                    slot.shape()
                )));
            }
            keyed.insert(Rc::from(key.as_str()), value);
        }
        if !unmatched.is_empty() {
            unmatched.sort_unstable();
            return Err(LoraError::InvalidState(format!(
                "the adapter file holds {} tensor(s) for projections this model has no \
                 adapter on ({:?}{}); name those projections in target_modules, which \
                 when empty leaves out the LM head",
                unmatched.len(),
                &unmatched[..unmatched.len().min(4)],
                if unmatched.len() > 4 { ", ..." } else { "" },
            )));
        }
        self.set_lora_parameters(&keyed);
        Ok(())
    }

    /// Adapted modules by their canonical Hugging Face name, to the stem of
    /// their in-memory adapter keys.
    fn module_index(&self) -> HashMap<String, String> {
        self.adapted
            .iter()
            .map(|path| (canonical_module(&hf_module_name(path)), adapter_key(path)))
            .collect()
    }

    /// Materialise every parameter, so the next forward does not re-evaluate the
    /// load graph.
    pub fn eval_all(&self) -> Result<(), LoraError> {
        self.model.eval().map_err(LoraError::Mlx)
    }

    /// Which architecture is loaded.
    pub fn architecture(&self) -> pmetal_models::ModelArchitecture {
        self.model.architecture()
    }

    /// Hidden states before the LM head, for cut cross-entropy.
    pub fn forward_hidden(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
    ) -> Result<Array, LoraError> {
        self.model
            .forward_hidden(input_ids, mask)
            .map_err(LoraError::Mlx)
    }

    /// A KV cache sized for this architecture.
    pub fn create_cache(&self, max_seq_len: usize) -> pmetal_cache::kv_cache::KVCache {
        self.model.create_cache(max_seq_len)
    }
}

/// Whether a projection at `path` should carry an adapter.
///
/// A target matches a module whose name it equals or ends, at a `.`
/// boundary: PEFT's rule, so a projection's name (`q_proj`, `gate_proj`) and a
/// full module name (`model.layers.0.mlp.down_proj`, how PEFT lists the
/// modules `all-linear` found) both work.
///
/// No targets means every projection but `output`, the model's output layer
/// (see [`DynamicModel::output_projection_path`]). MoE routers and routed
/// experts never do; see [`moe_role`].
fn is_targeted(path: &str, targets: &[String], output: Option<&str>) -> bool {
    if moe_role(path).is_some() {
        return false;
    }
    if targets.is_empty() {
        return output != Some(path);
    }
    let module = canonical_module(&hf_module_name(path));
    targets.iter().any(|target| {
        let target = canonical_module(target);
        module == target
            || module
                .strip_suffix(target.as_str())
                .is_some_and(|rest| rest.ends_with('.'))
    })
}

/// What a projection does in a mixture-of-experts block that puts it out of
/// LoRA's and QLoRA's reach, if anything.
///
/// Routed experts run through stacked or gathered expert kernels that read
/// their weights directly, so an adapter on one is never applied (GPT-OSS's
/// experts are named `gate_proj`, so targeting the MLP would have adapted
/// them). A router's adapter is not reliably applied either, and its argmax
/// picks which computation runs at all. DeepSeek's router is a bare matrix, not
/// a `Linear`, so it is never visited; Granite 4's keeps its `Linear` in a
/// field named `layer`, so its path ends `router.layer`, and Gemma 4's (and
/// DiffusionGemma's) in one named `proj`, so its path ends `router.proj`.
pub(crate) fn moe_role(path: &str) -> Option<&'static str> {
    if pmetal_models::moe_routing::is_router_path(path) {
        return Some("a router picks experts by argmax");
    }
    if path.contains(".experts.") {
        return Some("routed experts are read by the stacked expert kernels");
    }
    None
}

/// A module name with the spellings that differ between a text model and the
/// multimodal wrapper around it removed: `model.language_model.layers.0.x` and
/// `model.layers.0.x` and `layers.0.x` are the same module.
fn canonical_module(name: &str) -> String {
    let name = name.replace("model.language_model.", "model.");
    name.strip_prefix(TRUNK_PREFIX).unwrap_or(&name).to_string()
}

/// The in-memory key (see [`AdaptedModel::lora_parameters`]) an adapter file's
/// tensor `name` is for, in PEFT's layout or the earlier one, or `None` if
/// this model has no such slot.
fn native_key(
    name: &str,
    modules: &HashMap<String, String>,
    ours: &HashMap<Rc<str>, Array>,
) -> Option<String> {
    if let Some(rest) = name.strip_prefix(PEFT_PREFIX) {
        const PARTS: [(&str, &str); 4] = [
            (".lora_A.weight", ".lora_a"),
            (".lora_B.weight", ".lora_b"),
            (".lora_magnitude_vector.weight", ".lora_magnitude"),
            (".lora_magnitude_vector", ".lora_magnitude"),
        ];
        let (module, native) = PARTS
            .iter()
            .find_map(|(suffix, native)| rest.strip_suffix(suffix).map(|m| (m, *native)))?;
        let key = format!("{}{native}", modules.get(&canonical_module(module))?);
        return ours.contains_key(key.as_str()).then_some(key);
    }
    if !is_adapter_key(name) {
        return None;
    }
    [name.to_string(), adapter_key(name)]
        .into_iter()
        .find(|key| ours.contains_key(key.as_str()))
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

    fn adapter_target_modules(&self) -> Vec<String> {
        AdaptedModel::adapter_target_modules(self)
    }

    fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut pmetal_cache::kv_cache::KVCache>,
    ) -> Result<Array, LoraError> {
        self.model
            .forward_with_cache(input_ids, mask, cache)
            .map_err(LoraError::Mlx)
    }

    fn forward_with_cache_last(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut pmetal_cache::kv_cache::KVCache>,
    ) -> Result<Array, LoraError> {
        self.model
            .forward_with_cache_last(input_ids, mask, cache)
            .map_err(LoraError::Mlx)
    }

    fn forward_with_positions(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        position_ids: &Array,
    ) -> Result<Array, LoraError> {
        self.model
            .forward_with_positions(input_ids, mask, Some(position_ids))
            .map_err(LoraError::Mlx)
    }

    fn supports_packed_positions(&self) -> bool {
        self.model.supports_packed_positions()
    }

    /// An architecture with recurrent layers is one whose decoding needs a
    /// recurrent cache beside the KV cache, which is what the dispatcher's
    /// `create_mamba_cache` answers for every such model; the ones whose
    /// layers reset at a packed boundary on their own are not counted.
    fn has_recurrent_layers(&self) -> bool {
        self.model.has_unresettable_recurrent_layers()
    }

    fn output_projection_path(&mut self) -> Option<String> {
        self.model.output_projection_path()
    }

    /// `pixel_values` here means vision features already projected to the text
    /// hidden size, which is what the VLM adapter path computes once and
    /// reuses. Running the tower from raw pixels needs the processor's tile
    /// geometry, which no uniform batch carries; see
    /// [`DynamicModel::forward_with_cross_states`].
    ///
    /// A model with no cross-attention errors rather than quietly running
    /// text-only, so a VLM run that was handed images and could not use them
    /// stops instead of reporting a loss that ignored them.
    fn forward_with_images(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        pixel_values: Option<&Array>,
    ) -> Result<Array, LoraError> {
        match pixel_values {
            Some(cross_states) => self
                .model
                .forward_with_cross_states(input_ids, mask, cross_states)
                .map_err(LoraError::Mlx),
            None => AdaptedModel::forward(self, input_ids, mask),
        }
    }

    fn is_multimodal(&self) -> bool {
        self.model.supports_cross_attention()
    }

    fn supports_kv_cache(&self) -> bool {
        true
    }

    fn create_cache(&self, max_seq_len: usize) -> Option<pmetal_cache::kv_cache::KVCache> {
        Some(AdaptedModel::create_cache(self, max_seq_len))
    }

    /// `layers_per_block` is accepted and ignored.
    ///
    /// One decoder layer is the checkpoint unit, which is what PyTorch
    /// uses. The parameter stays on the trait because the CLI still
    /// carries the flag; it never selected anything.
    fn enable_gradient_checkpointing(&mut self, _layers_per_block: usize) {
        self.model.set_gradient_checkpointing(true);
    }

    fn disable_gradient_checkpointing(&mut self) {
        self.model.set_gradient_checkpointing(false);
    }

    fn supports_gradient_checkpointing(&self) -> bool {
        self.model.supports_gradient_checkpointing()
    }

    /// Hidden states before the LM head, so cut cross-entropy never has to
    /// materialise `[batch, seq, vocab]` logits.
    ///
    /// `None` only when this architecture has no trunk forward to call, which
    /// is what the dispatcher reports by erroring.
    fn forward_hidden(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
    ) -> Option<Result<Array, LoraError>> {
        Some(
            self.model
                .forward_hidden(input_ids, mask)
                .map_err(LoraError::Mlx),
        )
    }

    /// The same for a packed batch, through the trunk
    /// [`forward_with_positions`](crate::TrainableModel::forward_with_positions)
    /// runs, so cut cross-entropy covers packed training too.
    fn forward_hidden_with_positions(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        position_ids: &Array,
    ) -> Option<Result<Array, LoraError>> {
        Some(
            self.model
                .forward_hidden_with_positions(input_ids, mask, Some(position_ids))
                .map_err(LoraError::Mlx),
        )
    }

    fn lm_head(&self) -> Option<pmetal_models::dispatcher::LmHead> {
        self.model.lm_head()
    }

    /// Forward with NEFTune noise on the embedding output.
    ///
    /// The alpha is set on the embedding for the duration of the call and
    /// cleared afterwards, so nothing else that shares this model (an eval
    /// pass, a generation) inherits it.
    fn forward_noised(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        noise_alpha: f32,
    ) -> Result<Array, LoraError> {
        if !self.model.set_neftune_alpha(Some(noise_alpha)) {
            return AdaptedModel::forward(self, input_ids, mask);
        }
        let out = AdaptedModel::forward(self, input_ids, mask);
        self.model.set_neftune_alpha(None);
        out
    }
}

/// Delegates straight through to the wrapped model, so the optimizer and the
/// checkpointer see the same parameter tree the loader wrote.
impl ModuleParameters for AdaptedModel {
    fn num_parameters(&self) -> usize {
        self.model.num_parameters()
    }

    fn parameters(&self) -> pmetal_runtime::compat::ModuleParamRef<'_> {
        self.model.parameters()
    }

    fn parameters_mut(&mut self) -> pmetal_runtime::compat::ModuleParamMut<'_> {
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
    fn trainable_parameters(&self) -> pmetal_runtime::compat::ModuleParamRef<'_> {
        fn collect<'a>(
            tree: &pmetal_runtime::compat::ModuleParamRef<'a>,
            prefix: &str,
            out: &mut pmetal_runtime::compat::ModuleParamRef<'a>,
        ) {
            for (name, value) in tree {
                let path = pmetal_runtime::compat::child_path(prefix, name);
                match value {
                    pmetal_runtime::compat::NestedValue::Value(array) => {
                        if is_adapter_key(&path) {
                            out.insert(Rc::from(path.as_str()), NestedValue::Value(array));
                        }
                    }
                    pmetal_runtime::compat::NestedValue::Map(inner) => collect(inner, &path, out),
                }
            }
        }

        let tree = self.model.parameters();
        let mut out = pmetal_runtime::compat::ModuleParamRef::new();
        collect(&tree, "", &mut out);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::is_targeted;

    /// The output layer is left out by identity, not by name: a model whose
    /// head is `embed_out` loses that, and keeps an adapter on a projection
    /// that happens to be called `lm_head` but is not its output.
    #[test]
    fn empty_targets_leave_out_the_output_layer_whatever_it_is_called() {
        let none: &[String] = &[];
        assert!(!is_targeted("embed_out", none, Some("embed_out")));
        assert!(is_targeted("lm_head", none, Some("embed_out")));
        assert!(is_targeted(
            "model.layers.0.self_attn.q_proj",
            none,
            Some("embed_out")
        ));
        // A tied model has no output projection to leave out.
        assert!(is_targeted("model.layers.0.mlp.down_proj", none, None));
        // Routers stay out whatever the targets.
        assert!(!is_targeted("model.layers.0.mlp.gate", none, None));
        // Named, the output layer is adapted.
        assert!(is_targeted("embed_out", &["embed_out".to_string()], None));
        assert!(is_targeted("lm_head", &["lm_head".to_string()], None));
    }
}
