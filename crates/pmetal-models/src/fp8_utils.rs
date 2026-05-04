//! Generic FP8 weight quantization utilities for all model architectures.
//!
//! This module provides architecture-agnostic FP8 quantization by traversing
//! a model's flattened parameter map and replacing every eligible linear
//! weight tensor with its `to_fp8` equivalent.  It is intentionally separate
//! from the NemotronH-specific implementation in `architectures/nemotron_h.rs`,
//! which operates on concrete `nn::Linear` structs with custom FP8-aware
//! forward passes.
//!
//! # How it works
//!
//! `ModuleParameters::parameters_mut().flatten()` yields a
//! `HashMap<Rc<str>, &mut Array>` whose keys are dot-separated parameter paths
//! (e.g. `"model.layers.0.self_attn.q_proj.weight"`).  Any key whose final
//! component is `"weight"` and whose path is not an embedding or normalization
//! parameter is eligible for FP8 quantization.
//!
//! After quantisation the parameter arrays are stored as `uint8` (E4M3 format)
//! in-place.  Inference code that reads these parameters must call
//! `pmetal_bridge::compat::ops::from_fp8` to dequantise before computation — this matches the
//! semantics already used by `pmetal_mlx::fp8_quantization`.

use pmetal_bridge::compat::ops::{from_fp8, to_fp8};
use pmetal_bridge::compat::{Array, Dtype, Exception, ModuleParameters, ModuleParametersExt};

fn is_fp8_weight_candidate(key: &str, arr: &Array) -> bool {
    if !(key.ends_with(".weight") || key == "weight") {
        return false;
    }
    if arr.ndim() != 2 {
        return false;
    }

    let lower = key.to_ascii_lowercase();
    let excluded = [
        "embed",
        "embedding",
        "word_embeddings",
        "position_embeddings",
        "token_embedding",
        "rotary",
        "norm",
        "layernorm",
        "layer_norm",
        "rmsnorm",
        "groupnorm",
        "group_norm",
    ];
    !excluded.iter().any(|needle| lower.contains(needle))
}

/// Materialize an FP8-stored weight to the compute dtype expected by matmul paths.
///
/// MLX stores FP8 E4M3 tensors as `uint8` payload arrays.  Any direct matmul
/// path that bypasses `nn::Linear::forward` must call this before using a
/// quantized weight.
pub fn dequantize_fp8_weight_for_compute(weight: &Array) -> Result<Array, Exception> {
    if weight.dtype() == Dtype::Uint8 {
        from_fp8(weight, Dtype::Bfloat16)
    } else {
        Ok(weight.clone())
    }
}

/// Quantize every eligible linear weight parameter of `model` to FP8 (E4M3) in-place.
///
/// The function iterates the fully-flattened parameter map and replaces 2-D
/// non-embedding, non-normalization arrays whose key ends with `".weight"` (or
/// equals `"weight"` for top-level linear modules) with its `to_fp8`
/// representation.
///
/// Biases, embedding tables, and normalisation scale vectors are left untouched.
///
/// # Errors
///
/// Returns the first `Exception` produced by `pmetal_bridge::compat::ops::to_fp8` if any
/// quantisation call fails.
pub fn quantize_model_linears<M: ModuleParameters>(model: &mut M) -> Result<(), Exception> {
    // Collect the keys we need to quantize first (avoid borrow issues).
    // We need owned quantized arrays, then write them back through parameters_mut.
    let keys_to_quantize: Vec<std::rc::Rc<str>> = {
        let flat = model.flatten_params();
        flat.iter()
            .filter(|(k, arr)| is_fp8_weight_candidate(k.as_ref(), arr))
            .map(|(k, _)| k.clone())
            .collect()
    };

    if keys_to_quantize.is_empty() {
        return Ok(());
    }

    // Quantize each eligible weight.  We do this in two passes to satisfy the
    // borrow checker: first read + compute FP8 tensors (immutable borrow),
    // then write them back (mutable borrow).
    let quantized: Vec<(std::rc::Rc<str>, pmetal_bridge::compat::Array)> = {
        let flat = model.flatten_params();
        keys_to_quantize
            .iter()
            .filter_map(|k| flat.get(k).map(|arr| (k.clone(), arr.clone())))
            .map(|(k, arr)| to_fp8(&arr).map(|q| (k, q)))
            .collect::<Result<_, _>>()?
    };

    // Write the FP8 tensors back through the mutable parameter map.
    {
        let mut flat_mut = model.flatten_params_mut();
        for (k, q) in quantized {
            let k_str: &str = k.as_ref();
            if let Some(dest) = flat_mut.get_mut(k_str) {
                **dest = q;
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_bridge::compat::nn;
    use pmetal_bridge::impl_module_params;

    #[derive(Debug)]
    struct TinyModel {
        linear: nn::Linear,
        embed_tokens: nn::Embedding,
        norm: nn::RmsNorm,
        lm_head: nn::Linear,
    }

    impl_module_params!(TinyModel; linear, embed_tokens, norm, lm_head);

    #[test]
    fn quantize_model_linears_skips_embeddings_and_norms() {
        let mut model = TinyModel {
            linear: nn::LinearBuilder::new(4, 3).bias(false).build().unwrap(),
            embed_tokens: nn::Embedding::new(8, 4).unwrap(),
            norm: nn::RmsNorm::new(4).unwrap(),
            lm_head: nn::LinearBuilder::new(4, 8).bias(false).build().unwrap(),
        };

        quantize_model_linears(&mut model).unwrap();

        assert_eq!(model.linear.weight.as_ref().dtype(), Dtype::Uint8);
        assert_eq!(model.lm_head.weight.as_ref().dtype(), Dtype::Uint8);
        assert_ne!(model.embed_tokens.weight.as_ref().dtype(), Dtype::Uint8);
        assert_ne!(model.norm.weight.as_ref().dtype(), Dtype::Uint8);
    }
}
