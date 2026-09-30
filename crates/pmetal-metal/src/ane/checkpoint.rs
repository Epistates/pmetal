//! What the ANE engines can load, and loading it without silent fallbacks.
//!
//! The two engines implement fixed architectures. Training
//! ([`super::dynamic_trainer`]) is Llama-shaped: SiLU-gated FFN, no per-head
//! q/k norm. Inference ([`super::inference`]) is Qwen3-shaped: the same FFN
//! plus per-head q/k RMSNorm. Neither routes experts, applies `rope_scaling`,
//! adds attention or MLP biases, uses a sliding window, or has an output
//! projection separate from the embeddings. A checkpoint outside that shape
//! used to load anyway and compute a different function with no error (#34),
//! as did one whose tensors failed to convert or had the wrong size.

use std::collections::HashSet;

use crate::error::{MetalError, Result};

/// Config limits both engines share. `engine` names the caller in errors.
pub(crate) fn check_shared_limits(
    config: &serde_json::Value,
    engine: &str,
) -> std::result::Result<(), String> {
    let int = |key: &str| config.get(key).and_then(|v| v.as_i64()).unwrap_or(0);
    let flag = |key: &str| config.get(key).and_then(|v| v.as_bool());
    let set = |key: &str| config.get(key).is_some_and(|v| !v.is_null());

    if ["num_experts", "num_local_experts", "n_routed_experts"]
        .iter()
        .any(|k| int(k) > 0)
    {
        return Err(format!(
            "MoE models with routed experts are not supported by ANE {engine}."
        ));
    }
    if set("rope_scaling") {
        return Err(format!(
            "rope_scaling is not implemented by the ANE {engine} engine, which applies plain RoPE."
        ));
    }
    if set("sliding_window") && flag("use_sliding_window") != Some(false) {
        return Err(format!(
            "Sliding-window attention is not implemented by the ANE {engine} engine."
        ));
    }
    for key in ["attention_bias", "mlp_bias"] {
        if flag(key) == Some(true) {
            return Err(format!(
                "{key} is not implemented by the ANE {engine} engine."
            ));
        }
    }
    // Hugging Face defaults tie_word_embeddings to true when absent.
    if flag("tie_word_embeddings") == Some(false) {
        return Err(format!(
            "The ANE {engine} engine takes logits from the embedding matrix, so a model \
             with a separate lm_head (tie_word_embeddings: false) is not supported."
        ));
    }
    Ok(())
}

/// Reject a `model_type` the engine doesn't implement.
pub(crate) fn check_model_type(
    config: &serde_json::Value,
    supported: &[&str],
    engine: &str,
) -> std::result::Result<(), String> {
    let model_type = config
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if supported.contains(&model_type) {
        Ok(())
    } else {
        Err(format!(
            "Model type '{model_type}' is not implemented by the ANE {engine} engine \
             (supported: {}). Use the GPU.",
            supported.join(", ")
        ))
    }
}

/// Copies checkpoint tensors into an engine's buffers, refusing anything it
/// would otherwise have to guess at, and records what it loaded.
#[derive(Default)]
pub(crate) struct StrictLoader {
    loaded: HashSet<String>,
}

impl StrictLoader {
    /// Copy `name` into `dst`, which must have room for `expected` elements.
    /// `data` is the tensor's f32 conversion: an unsupported dtype (such as a
    /// packed quantized weight) or a size other than `expected` is an error.
    pub(crate) fn copy<E: std::fmt::Display>(
        &mut self,
        name: &str,
        data: &std::result::Result<Vec<f32>, E>,
        dst: &mut [f32],
        expected: usize,
    ) -> Result<()> {
        let src = data.as_ref().map_err(|why| {
            MetalError::InvalidConfig(format!(
                "{name}: {why} (the ANE engines read f32, f16 and bf16 weights only)"
            ))
        })?;
        if src.len() != expected || dst.len() < expected {
            return Err(MetalError::InvalidConfig(format!(
                "{name} has {} elements; the ANE engine expects {expected}",
                src.len()
            )));
        }
        dst[..expected].copy_from_slice(src);
        self.loaded.insert(name.to_string());
        Ok(())
    }

    /// Fail unless every tensor in `required` was loaded.
    pub(crate) fn require(&self, required: impl IntoIterator<Item = String>) -> Result<()> {
        let missing: Vec<String> = required
            .into_iter()
            .filter(|name| !self.loaded.contains(name))
            .collect();
        if missing.is_empty() {
            return Ok(());
        }
        let shown = missing
            .iter()
            .take(5)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ");
        let more = missing.len().saturating_sub(5);
        Err(MetalError::InvalidConfig(format!(
            "checkpoint is missing {} tensor(s) the ANE engine needs: {shown}{}",
            missing.len(),
            if more > 0 {
                format!(" and {more} more")
            } else {
                String::new()
            }
        )))
    }
}

/// Per-layer tensor names under `model.layers.{i}.`.
pub(crate) fn layer_tensors<'a>(
    n_layers: usize,
    suffixes: &'a [&'a str],
) -> impl Iterator<Item = String> + 'a {
    (0..n_layers).flat_map(move |i| {
        suffixes
            .iter()
            .map(move |suffix| format!("model.layers.{i}.{suffix}"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn shared_limits_reject_what_neither_engine_implements() {
        let base = json!({"model_type": "llama"});
        assert!(check_shared_limits(&base, "training").is_ok());

        for (key, value) in [
            ("num_experts", json!(8)),
            (
                "rope_scaling",
                json!({"rope_type": "llama3", "factor": 8.0}),
            ),
            ("sliding_window", json!(4096)),
            ("attention_bias", json!(true)),
            ("tie_word_embeddings", json!(false)),
        ] {
            let mut config = base.clone();
            config[key] = value;
            assert!(
                check_shared_limits(&config, "training").is_err(),
                "{key} should be rejected"
            );
        }

        // Present but null or disabled is fine.
        let config = json!({
            "rope_scaling": null,
            "sliding_window": 4096,
            "use_sliding_window": false,
            "tie_word_embeddings": true,
        });
        assert!(check_shared_limits(&config, "inference").is_ok());
    }

    #[test]
    fn the_loader_refuses_bad_dtype_wrong_size_and_missing_tensors() {
        let mut loader = StrictLoader::default();
        let mut dst = vec![0.0f32; 4];

        let unsupported: std::result::Result<Vec<f32>, &str> = Err("dtype U32");
        assert!(loader.copy("a", &unsupported, &mut dst, 4).is_err());

        let short: std::result::Result<Vec<f32>, &str> = Ok(vec![1.0; 3]);
        assert!(loader.copy("b", &short, &mut dst, 4).is_err());
        assert_eq!(
            dst,
            vec![0.0; 4],
            "a rejected tensor must not be half-copied"
        );

        let good: std::result::Result<Vec<f32>, &str> = Ok(vec![1.0, 2.0, 3.0, 4.0]);
        loader.copy("c", &good, &mut dst, 4).unwrap();
        assert_eq!(dst, vec![1.0, 2.0, 3.0, 4.0]);

        assert!(loader.require(["c".to_string()]).is_ok());
        let err = loader
            .require([
                "c".to_string(),
                "model.layers.0.self_attn.q_norm.weight".to_string(),
            ])
            .unwrap_err();
        assert!(err.to_string().contains("q_norm"));
    }
}
