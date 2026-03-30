//! Shared helper infrastructure for LoRA architecture files.
//!
//! This module provides:
//! - `LoraDecoderStack`: trait that each architecture's model type implements to
//!   expose its layers' projections through a uniform interface.
//! - Generic functions (`collect_lora_parameters`, `set_lora_parameters`,
//!   `save_lora_weights_impl`, `load_lora_weights_impl`, `count_trainable_params`)
//!   that operate on any `LoraDecoderStack`.
//! - `impl_trainable_model!` macro that generates the boilerplate `TrainableModel`
//!   impl for architectures whose `ForCausalLM` type exposes all required methods
//!   as inherent methods.

use std::collections::HashMap;
use std::rc::Rc;

use pmetal_bridge::compat::Array;

use crate::LoraError;
use crate::lora::LoraProjection;

// ─── LoraDecoderStack ────────────────────────────────────────────────────────

/// Object-safe interface over a stack of transformer decoder layers.
///
/// Implement this trait on the inner `*LoraModel` struct (not the `ForCausalLM`
/// wrapper) so that the generic helper functions below can iterate over layers
/// and access projections without knowing the concrete architecture.
///
/// Notes on projection ordering:
/// - `attn_projections` must return projections in the same order as
///   `attn_projection_names` (both of length N).
/// - `mlp_projections` / `mlp_projection_names` follow the same contract.
pub trait LoraDecoderStack {
    /// Total number of decoder layers.
    fn num_layers(&self) -> usize;

    /// Canonical names for the attention projections, in the same order that
    /// `attn_projections` returns them.
    ///
    /// Default: `["q_proj", "k_proj", "v_proj", "o_proj"]`.
    fn attn_projection_names(&self) -> &[&'static str] {
        &["q_proj", "k_proj", "v_proj", "o_proj"]
    }

    /// Canonical names for the MLP projections.
    ///
    /// Default: `["gate_proj", "up_proj", "down_proj"]`.
    fn mlp_projection_names(&self) -> &[&'static str] {
        &["gate_proj", "up_proj", "down_proj"]
    }

    /// Immutable references to attention projections for layer `layer`.
    fn attn_projections(&self, layer: usize) -> Vec<&dyn LoraProjection>;

    /// Mutable references to attention projections for layer `layer`.
    fn attn_projections_mut(&mut self, layer: usize) -> Vec<&mut dyn LoraProjection>;

    /// Immutable references to MLP projections for layer `layer`.
    fn mlp_projections(&self, layer: usize) -> Vec<&dyn LoraProjection>;

    /// Mutable references to MLP projections for layer `layer`.
    fn mlp_projections_mut(&mut self, layer: usize) -> Vec<&mut dyn LoraProjection>;
}

// ─── Generic helper functions ────────────────────────────────────────────────

/// Collect all LoRA adapter parameters into a flat `HashMap`.
///
/// Keys follow the pattern `layers.{i}.{section}.{proj_name}.lora_{a|b}`,
/// with optional extra keys for DoRA magnitude etc.
pub fn collect_lora_parameters(stack: &dyn LoraDecoderStack) -> HashMap<Rc<str>, Array> {
    let mut params = HashMap::new();
    let attn_names = stack.attn_projection_names();
    let mlp_names = stack.mlp_projection_names();

    for i in 0..stack.num_layers() {
        let layer_prefix = format!("layers.{}", i);

        for (proj, name) in stack.attn_projections(i).iter().zip(attn_names.iter()) {
            let key_prefix = format!("{}.self_attn.{}", layer_prefix, name);
            params.insert(
                Rc::from(format!("{}.lora_a", key_prefix)),
                proj.lora_a().clone(),
            );
            params.insert(
                Rc::from(format!("{}.lora_b", key_prefix)),
                proj.lora_b().clone(),
            );
            for (extra_name, arr) in proj.extra_params() {
                params.insert(
                    Rc::from(format!("{}.{}", key_prefix, extra_name)),
                    arr.clone(),
                );
            }
        }

        for (proj, name) in stack.mlp_projections(i).iter().zip(mlp_names.iter()) {
            let key_prefix = format!("{}.mlp.{}", layer_prefix, name);
            params.insert(
                Rc::from(format!("{}.lora_a", key_prefix)),
                proj.lora_a().clone(),
            );
            params.insert(
                Rc::from(format!("{}.lora_b", key_prefix)),
                proj.lora_b().clone(),
            );
            for (extra_name, arr) in proj.extra_params() {
                params.insert(
                    Rc::from(format!("{}.{}", key_prefix, extra_name)),
                    arr.clone(),
                );
            }
        }
    }

    params
}

/// Apply parameters from a `HashMap` back into the stack's projections.
///
/// Silently ignores keys that do not match any projection.
pub fn set_lora_parameters(stack: &mut dyn LoraDecoderStack, params: &HashMap<Rc<str>, Array>) {
    let attn_names: Vec<&'static str> = stack.attn_projection_names().to_vec();
    let mlp_names: Vec<&'static str> = stack.mlp_projection_names().to_vec();

    for i in 0..stack.num_layers() {
        let layer_prefix = format!("layers.{}", i);

        // Collect keys first so we can borrow stack mutably per projection
        for (idx, name) in attn_names.iter().enumerate() {
            let key_prefix = format!("{}.self_attn.{}", layer_prefix, name);
            let a_key: Rc<str> = Rc::from(format!("{}.lora_a", key_prefix));
            let b_key: Rc<str> = Rc::from(format!("{}.lora_b", key_prefix));

            if let Some(value) = params.get(&a_key) {
                let proj = &mut stack.attn_projections_mut(i)[idx];
                *proj.lora_a_mut() = value.clone();
            }
            if let Some(value) = params.get(&b_key) {
                let proj = &mut stack.attn_projections_mut(i)[idx];
                *proj.lora_b_mut() = value.clone();
            }

            // Extra params (e.g. DoRA magnitude)
            // We need to collect extra param names before borrowing mutably
            let extra_names: Vec<String> = {
                let projs = stack.attn_projections(i);
                projs[idx]
                    .extra_params()
                    .iter()
                    .map(|(n, _)| n.to_string())
                    .collect()
            };
            for extra_name in extra_names {
                let extra_key: Rc<str> = Rc::from(format!("{}.{}", key_prefix, extra_name));
                if let Some(value) = params.get(&extra_key) {
                    let proj = &mut stack.attn_projections_mut(i)[idx];
                    for (n, arr) in proj.extra_params_mut() {
                        if n == extra_name {
                            *arr = value.clone();
                            break;
                        }
                    }
                }
            }
        }

        for (idx, name) in mlp_names.iter().enumerate() {
            let key_prefix = format!("{}.mlp.{}", layer_prefix, name);
            let a_key: Rc<str> = Rc::from(format!("{}.lora_a", key_prefix));
            let b_key: Rc<str> = Rc::from(format!("{}.lora_b", key_prefix));

            if let Some(value) = params.get(&a_key) {
                let proj = &mut stack.mlp_projections_mut(i)[idx];
                *proj.lora_a_mut() = value.clone();
            }
            if let Some(value) = params.get(&b_key) {
                let proj = &mut stack.mlp_projections_mut(i)[idx];
                *proj.lora_b_mut() = value.clone();
            }

            let extra_names: Vec<String> = {
                let projs = stack.mlp_projections(i);
                projs[idx]
                    .extra_params()
                    .iter()
                    .map(|(n, _)| n.to_string())
                    .collect()
            };
            for extra_name in extra_names {
                let extra_key: Rc<str> = Rc::from(format!("{}.{}", key_prefix, extra_name));
                if let Some(value) = params.get(&extra_key) {
                    let proj = &mut stack.mlp_projections_mut(i)[idx];
                    for (n, arr) in proj.extra_params_mut() {
                        if n == extra_name {
                            *arr = value.clone();
                            break;
                        }
                    }
                }
            }
        }
    }
}

/// Save LoRA weights from a stack to a safetensors file.
pub fn save_lora_weights_impl(
    stack: &dyn LoraDecoderStack,
    path: impl AsRef<std::path::Path>,
) -> Result<(), LoraError> {
    let params = collect_lora_parameters(stack);
    crate::save_safetensors_map(path, &params)
}

/// Load LoRA weights into a stack from a safetensors file or directory.
///
/// If `path` is a directory, looks for `lora_weights.safetensors` inside it.
pub fn load_lora_weights_impl(
    stack: &mut dyn LoraDecoderStack,
    path: impl AsRef<std::path::Path>,
) -> Result<(), LoraError> {
    let path = path.as_ref();
    let file_path = if path.is_dir() {
        path.join("lora_weights.safetensors")
    } else {
        path.to_path_buf()
    };
    let loaded = crate::load_safetensors_map(&file_path)?;
    // Convert the string-keyed map to Rc<str>-keyed for set_lora_parameters
    let params: HashMap<Rc<str>, Array> = loaded
        .into_iter()
        .map(|(k, v)| (Rc::from(k.as_str()), v))
        .collect();
    set_lora_parameters(stack, &params);
    Ok(())
}

/// Count the total number of trainable parameters across all layers.
pub fn count_trainable_params(stack: &dyn LoraDecoderStack) -> usize {
    let mut total = 0usize;
    for i in 0..stack.num_layers() {
        for proj in stack.attn_projections(i) {
            total += proj.num_trainable_params();
        }
        for proj in stack.mlp_projections(i) {
            total += proj.num_trainable_params();
        }
    }
    total
}

// ─── impl_trainable_model! macro ─────────────────────────────────────────────

/// Generate a `TrainableModel` impl for a `ForCausalLM` type that delegates to
/// its own inherent methods.
///
/// The generated impl requires the concrete type to expose:
/// - `forward(&mut self, input_ids, mask) -> Result<Array, LoraError>`
/// - `num_trainable_params(&self) -> usize`
/// - `lora_parameters(&self) -> HashMap<Rc<str>, Array>`
/// - `set_lora_parameters(&mut self, params)`
/// - `save_lora_weights(&self, path) -> Result<(), LoraError>`
/// - `load_lora_weights(&mut self, path) -> Result<(), LoraError>`
/// - `enable_gradient_checkpointing(&mut self, layers_per_block)`
/// - `disable_gradient_checkpointing(&mut self)`
/// - `create_cache(&self, max_seq_len) -> KVCache`
/// - `forward_with_cache(&mut self, input_ids, mask, cache) -> Result<Array, LoraError>`
/// - `forward_noised(&mut self, input_ids, mask, noise_alpha) -> Result<Array, LoraError>`
/// - `forward_hidden_states(&mut self, input_ids, mask) -> Result<Array, LoraError>`
/// - `forward_hidden_states_with_positions(&mut self, input_ids, mask, pos) -> Result<Array, LoraError>`
/// - `get_lm_head_weight(&self) -> Option<Array>`
#[macro_export]
macro_rules! impl_trainable_model {
    ($type:ty) => {
        impl $crate::TrainableModel for $type {
            fn forward(
                &mut self,
                input_ids: &Array,
                mask: Option<&Array>,
            ) -> Result<Array, $crate::LoraError> {
                <$type>::forward(self, input_ids, mask)
            }

            fn num_trainable_params(&self) -> usize {
                <$type>::num_trainable_params(self)
            }

            fn lora_parameters(&self) -> std::collections::HashMap<std::rc::Rc<str>, Array> {
                <$type>::lora_parameters(self)
            }

            fn set_lora_parameters(
                &mut self,
                params: &std::collections::HashMap<std::rc::Rc<str>, Array>,
            ) {
                <$type>::set_lora_parameters(self, params)
            }

            fn save_lora_weights(
                &self,
                path: impl AsRef<std::path::Path>,
            ) -> Result<(), $crate::LoraError> {
                <$type>::save_lora_weights(self, path)
            }

            fn load_lora_weights(
                &mut self,
                path: impl AsRef<std::path::Path>,
            ) -> Result<(), $crate::LoraError> {
                <$type>::load_lora_weights(self, path)
            }

            fn enable_gradient_checkpointing(&mut self, layers_per_block: usize) {
                <$type>::enable_gradient_checkpointing(self, layers_per_block)
            }

            fn disable_gradient_checkpointing(&mut self) {
                <$type>::disable_gradient_checkpointing(self)
            }

            fn supports_gradient_checkpointing(&self) -> bool {
                true
            }

            fn supports_kv_cache(&self) -> bool {
                true
            }

            fn create_cache(&self, max_seq_len: usize) -> Option<pmetal_mlx::kv_cache::KVCache> {
                Some(<$type>::create_cache(self, max_seq_len))
            }

            fn forward_noised(
                &mut self,
                input_ids: &Array,
                mask: Option<&Array>,
                noise_alpha: f32,
            ) -> Result<Array, $crate::LoraError> {
                <$type>::forward_noised(self, input_ids, mask, noise_alpha)
            }

            fn forward_with_cache(
                &mut self,
                input_ids: &Array,
                mask: Option<&Array>,
                cache: Option<&mut pmetal_mlx::kv_cache::KVCache>,
            ) -> Result<Array, $crate::LoraError> {
                <$type>::forward_with_cache(self, input_ids, mask, cache)
            }

            fn forward_hidden(
                &mut self,
                input_ids: &Array,
                mask: Option<&Array>,
            ) -> Option<Result<Array, $crate::LoraError>> {
                Some(<$type>::forward_hidden_states(self, input_ids, mask))
            }

            fn forward_hidden_with_positions(
                &mut self,
                input_ids: &Array,
                mask: Option<&Array>,
                position_ids: &Array,
            ) -> Option<Result<Array, $crate::LoraError>> {
                Some(<$type>::forward_hidden_states_with_positions(
                    self,
                    input_ids,
                    mask,
                    position_ids,
                ))
            }

            fn lm_head_weight(&self) -> Option<Array> {
                <$type>::get_lm_head_weight(self)
            }
        }
    };
}

pub use impl_trainable_model;
