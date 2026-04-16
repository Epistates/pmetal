//! Depth-aware parameter initialization for pretraining from scratch.
//!
//! Standard Xavier/Kaiming init (which `nn::Linear::new` uses) is too wide
//! for deep transformers — the residual stream variance explodes. GPT-NeoX /
//! Megatron-LM scale the output projections of each residual block by
//! `1 / sqrt(2 * n_layers)`, keeping the residual signal stable regardless
//! of model depth. We apply the same correction here.

use pmetal_bridge::compat::{Array, Exception, ModuleParametersExt, module::ModuleParameters, ops};

/// Re-scale residual-stream output projections (o_proj, down_proj) by
/// `1 / sqrt(2 * n_layers)` following the GPT-NeoX convention.
///
/// Other weights keep their default initialization (uniform Xavier from
/// `nn::Linear::new`). Embeddings are left untouched.
///
/// This must be called **once** immediately after model construction and
/// before the first optimizer step.
pub fn apply_depth_scaled_init<M: ModuleParameters>(
    model: &mut M,
    n_layers: usize,
) -> Result<(), Exception> {
    if n_layers == 0 {
        return Ok(());
    }
    let scale = 1.0 / ((2.0 * n_layers as f64).sqrt());
    let scale_arr = Array::from_f32(scale as f32);

    let mut flat = model.flatten_params_mut();
    for (key, param) in flat.iter_mut() {
        // Match residual-output projections by convention: o_proj (attention
        // output) and down_proj (MLP output). These are the weights that
        // feed directly into the residual stream addition.
        if key.ends_with(".o_proj.weight") || key.ends_with(".down_proj.weight") {
            **param = param.multiply(&scale_arr);
        }
    }

    Ok(())
}

/// Zero-initialize all bias parameters. Many modern transformer architectures
/// (Llama, Qwen, Gemma) have no bias, but GPT-OSS does — zeroing prevents
/// the random bias from shifting the residual stream at init.
pub fn zero_biases<M: ModuleParameters>(model: &mut M) {
    let mut flat = model.flatten_params_mut();
    for (key, param) in flat.iter_mut() {
        if key.ends_with(".bias") {
            let shape = param.shape().to_vec();
            **param = ops::zeros_dtype(&shape, param.dtype());
        }
    }
}
