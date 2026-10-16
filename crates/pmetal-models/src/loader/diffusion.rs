//! Loaders for the text-to-image pipeline's components: the CLIP and T5 text
//! encoders, the VAE and the Flux transformer.

use pmetal_runtime::compat::Array;
use std::collections::HashMap;

use super::assign::assign_renamed;
use super::{Exemption, LoadError, LoadPolicy};
use crate::architectures::clip::CLIPTextModel;
use crate::architectures::flux::FluxDiT;
use crate::architectures::t5::T5EncoderModel;
use crate::architectures::vae::AutoencoderKL;
/// The model path of a CLIP text encoder checkpoint key, `transformers`
/// layout, with or without the `text_model.` namespace a standalone text tower
/// is saved under.
fn clip_param_name(key: &str) -> String {
    let key = key.strip_prefix("text_model.").unwrap_or(key);
    if let Some(rest) = key.strip_prefix("embeddings.") {
        return rest.to_string();
    }
    if key.starts_with("final_layer_norm.") || key.starts_with("text_projection.") {
        return key.to_string();
    }
    let Some(rest) = key.strip_prefix("encoder.layers.") else {
        return key.to_string();
    };
    let Some((index, rest)) = rest.split_once('.') else {
        return key.to_string();
    };
    let mapped = if let Some(rest) = rest.strip_prefix("self_attn.") {
        format!("attn.{rest}")
    } else if let Some(rest) = rest.strip_prefix("layer_norm1.") {
        format!("norm1.{rest}")
    } else if let Some(rest) = rest.strip_prefix("layer_norm2.") {
        format!("norm2.{rest}")
    } else if rest.starts_with("mlp.") {
        rest.to_string()
    } else {
        return key.to_string();
    };
    format!("layers.{index}.{mapped}")
}

/// Load a CLIP text encoder from a `transformers` checkpoint (`CLIPTextModel`,
/// `CLIPTextModelWithProjection`, or the text half of a full `CLIPModel`).
///
/// Fails, naming the difference, unless every tensor of the text tower is
/// consumed and every parameter filled and each shape matches.
pub fn load_clip_weights(
    model: &mut CLIPTextModel,
    weights: HashMap<String, Array>,
) -> Result<(), LoadError> {
    let mut policy = LoadPolicy::strict().allowing_unused_tensors(&[
        Exemption::new(
            "vision_model.*",
            "the image tower of a full CLIP checkpoint; this model is the text encoder",
        ),
        Exemption::new(
            "visual_projection.weight",
            "the image projection of a full CLIP checkpoint; this model is the text encoder",
        ),
        Exemption::new(
            "logit_scale",
            "the contrastive temperature of a full CLIP checkpoint; not used to encode text",
        ),
    ]);
    if model.text_projection.is_none() {
        policy = policy.allowing_unused_tensors(&[Exemption::new(
            "*text_projection.weight",
            "the projection of a full CLIP checkpoint; a CLIPTextModel pools without it",
        )]);
    }
    assign_renamed(
        model,
        weights,
        |key, value| (clip_param_name(key), value),
        &policy,
    )
}

/// The model path of a T5 encoder checkpoint key, `transformers` layout.
fn t5_param_name(key: &str) -> String {
    if let Some(rest) = key.strip_prefix("encoder.final_layer_norm.") {
        return format!("final_layer_norm.{rest}");
    }
    let Some(rest) = key.strip_prefix("encoder.block.") else {
        return key.to_string();
    };
    let Some((index, rest)) = rest.split_once('.') else {
        return key.to_string();
    };
    let mapped = if let Some(rest) = rest.strip_prefix("layer.0.SelfAttention.") {
        match rest.strip_prefix("relative_attention_bias.") {
            Some(rest) => format!("layer_0_attn.relative_attention_bias.embedding.{rest}"),
            None => format!("layer_0_attn.{rest}"),
        }
    } else if let Some(rest) = rest.strip_prefix("layer.0.layer_norm.") {
        format!("layer_0_norm.{rest}")
    } else if let Some(rest) = rest.strip_prefix("layer.1.DenseReluDense.") {
        format!("layer_1_mlp.{rest}")
    } else if let Some(rest) = rest.strip_prefix("layer.1.layer_norm.") {
        format!("layer_1_norm.{rest}")
    } else {
        return key.to_string();
    };
    format!("blocks.{index}.{mapped}")
}

/// Load a T5 encoder from a `transformers` checkpoint (`T5EncoderModel`, or
/// the full `T5Model` / `T5ForConditionalGeneration`, whose decoder and head
/// are not part of this model).
///
/// Fails, naming the difference, unless every tensor of the encoder is
/// consumed and every parameter filled.
pub fn load_t5_weights(
    model: &mut T5EncoderModel,
    mut weights: HashMap<String, Array>,
) -> Result<(), LoadError> {
    // `encoder.embed_tokens` is the same table as `shared`, which the
    // reference ties; a checkpoint may carry either or both.
    if weights.contains_key("shared.weight") {
        weights.remove("encoder.embed_tokens.weight");
    } else if let Some(table) = weights.remove("encoder.embed_tokens.weight") {
        weights.insert("shared.weight".to_string(), table);
    }
    let policy = LoadPolicy::strict().allowing_unused_tensors(&[
        Exemption::new(
            "decoder.*",
            "the decoder stack of a full T5 checkpoint; this model is the encoder",
        ),
        Exemption::new(
            "lm_head.weight",
            "the language-model head of a full T5 checkpoint; this model is the encoder",
        ),
    ]);
    assign_renamed(
        model,
        weights,
        |key, value| (t5_param_name(key), value),
        &policy,
    )
}

/// Map a `diffusers` `AutoencoderKL` tensor onto the model's parameter and
/// layout.
///
/// The names already agree; what differs is layout. Convolution weights are
/// `[out, in, h, w]` and the model's are `[out, h, w, in]`. The attention's
/// projections are `Linear`s (`[out, in]`), except in checkpoints from before
/// `diffusers` 0.14, which call them `query`, `key`, `value` and `proj_attn`
/// and may store them as 1x1 convolutions; those are renamed and squeezed the
/// way `diffusers` converts them on load.
fn vae_tensor(key: &str, value: Array) -> (String, Array) {
    let mut name = key.to_string();
    if key.contains(".attentions.") {
        for (old, new) in [
            (".query.", ".to_q."),
            (".key.", ".to_k."),
            (".value.", ".to_v."),
            (".proj_attn.", ".to_out.0."),
        ] {
            name = name.replace(old, new);
        }
    }
    let is_attention_projection = name.contains(".attentions.")
        && [".to_q.", ".to_k.", ".to_v.", ".to_out.0."]
            .iter()
            .any(|p| name.contains(p));
    let value = if value.ndim() == 4 && name.ends_with(".weight") {
        if is_attention_projection {
            value.reshape(&[value.dim(0), value.dim(1)])
        } else {
            value.transpose_axes(&[0, 2, 3, 1])
        }
    } else {
        value
    };
    (name, value)
}

/// Load an `AutoencoderKL` from a `diffusers` checkpoint.
///
/// Fails, naming the difference, unless every tensor is consumed, every
/// parameter filled and every shape matches (after the layout change).
pub fn load_vae_weights(
    model: &mut AutoencoderKL,
    weights: HashMap<String, Array>,
) -> Result<(), LoadError> {
    assign_renamed(model, weights, vae_tensor, &LoadPolicy::strict())
}

/// Rows `[start, end)` of `a`, along the first axis.
fn rows(a: &Array, start: i32, end: i32) -> Array {
    pmetal_runtime::compat::ops::slice_axis(a, 0, start, end)
}

/// Rewrite an original-layout (Black Forest Labs, as in the single-file
/// `flux1-*.safetensors` releases and ComfyUI) FLUX transformer checkpoint as
/// `diffusers`' tensors, the way `diffusers`' `convert_flux_to_diffusers.py`
/// does: fused `qkv` and `linear1` projections split into their parts, the
/// final layer's scale and shift (stored shift first) swapped, and the names
/// mapped. A key the mapping does not know is left as it is, so the load
/// reports it.
fn flux_original_to_diffusers(
    weights: HashMap<String, Array>,
    hidden: i32,
) -> HashMap<String, Array> {
    let mlp = 4 * hidden;
    let mut out = HashMap::with_capacity(weights.len() + 8 * 38);
    for (key, value) in weights {
        let name = key.strip_prefix("model.diffusion_model.").unwrap_or(&key);
        let (head, tail) = name.split_once('.').unwrap_or((name, ""));
        match head {
            "time_in" | "vector_in" | "guidance_in" => {
                let embedder = match head {
                    "time_in" => "timestep_embedder",
                    "vector_in" => "text_embedder",
                    _ => "guidance_embedder",
                };
                let mapped = tail.replacen("in_layer.", "linear_1.", 1).replacen(
                    "out_layer.",
                    "linear_2.",
                    1,
                );
                out.insert(format!("time_text_embed.{embedder}.{mapped}"), value);
            }
            "txt_in" => {
                out.insert(format!("context_embedder.{tail}"), value);
            }
            "img_in" => {
                out.insert(format!("x_embedder.{tail}"), value);
            }
            "final_layer" => {
                if let Some(rest) = tail.strip_prefix("linear.") {
                    out.insert(format!("proj_out.{rest}"), value);
                } else if let Some(rest) = tail.strip_prefix("adaLN_modulation.1.") {
                    // Stored (shift, scale); `diffusers` reads (scale, shift).
                    let half = value.dim(0) / 2;
                    let swapped = pmetal_runtime::compat::ops::concatenate_axis(
                        &[&rows(&value, half, 2 * half), &rows(&value, 0, half)],
                        0,
                    );
                    out.insert(format!("norm_out.linear.{rest}"), swapped);
                } else {
                    out.insert(key.clone(), value);
                }
            }
            "double_blocks" | "single_blocks" => {
                let Some((index, rest)) = tail.split_once('.') else {
                    out.insert(key.clone(), value);
                    continue;
                };
                let double = head == "double_blocks";
                let block = if double {
                    format!("transformer_blocks.{index}")
                } else {
                    format!("single_transformer_blocks.{index}")
                };
                // Three equal parts of a fused projection, then the rest.
                let mut split = |names: &[&str], suffix: &str, value: &Array| {
                    for (i, part) in names.iter().enumerate() {
                        let (from, to) = (i as i32 * hidden, (i as i32 + 1) * hidden);
                        out.insert(format!("{block}.{part}.{suffix}"), rows(value, from, to));
                    }
                };
                if let Some(s) = rest.strip_prefix("img_attn.qkv.").filter(|_| double) {
                    split(&["attn.to_q", "attn.to_k", "attn.to_v"], s, &value);
                } else if let Some(s) = rest.strip_prefix("txt_attn.qkv.").filter(|_| double) {
                    split(
                        &["attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"],
                        s,
                        &value,
                    );
                } else if let Some(s) = rest.strip_prefix("linear1.").filter(|_| !double) {
                    split(&["attn.to_q", "attn.to_k", "attn.to_v"], s, &value);
                    out.insert(
                        format!("{block}.proj_mlp.{s}"),
                        rows(&value, 3 * hidden, 3 * hidden + mlp),
                    );
                } else {
                    let table: &[(&str, &str)] = if double {
                        &[
                            ("img_mod.lin.", "norm1.linear."),
                            ("txt_mod.lin.", "norm1_context.linear."),
                            ("img_attn.norm.query_norm.scale", "attn.norm_q.weight"),
                            ("img_attn.norm.key_norm.scale", "attn.norm_k.weight"),
                            ("txt_attn.norm.query_norm.scale", "attn.norm_added_q.weight"),
                            ("txt_attn.norm.key_norm.scale", "attn.norm_added_k.weight"),
                            ("img_attn.proj.", "attn.to_out.0."),
                            ("txt_attn.proj.", "attn.to_add_out."),
                            ("img_mlp.0.", "ff.net.0.proj."),
                            ("img_mlp.2.", "ff.net.2."),
                            ("txt_mlp.0.", "ff_context.net.0.proj."),
                            ("txt_mlp.2.", "ff_context.net.2."),
                        ]
                    } else {
                        &[
                            ("modulation.lin.", "norm.linear."),
                            ("norm.query_norm.scale", "attn.norm_q.weight"),
                            ("norm.key_norm.scale", "attn.norm_k.weight"),
                            ("linear2.", "proj_out."),
                        ]
                    };
                    let mapped = table.iter().find_map(|(old, new)| {
                        rest.strip_prefix(old).map(|s| format!("{block}.{new}{s}"))
                    });
                    out.insert(mapped.unwrap_or_else(|| key.clone()), value);
                }
            }
            _ => {
                out.insert(key.clone(), value);
            }
        }
    }
    out
}

/// The model path of a `diffusers` FLUX transformer tensor: the same name,
/// except that the feed-forwards' `net.0.proj` and `net.2` are `proj_in` and
/// `proj_out`.
fn flux_param_name(key: &str) -> String {
    for feed_forward in ["ff", "ff_context"] {
        let marker = format!(".{feed_forward}.net.");
        if let Some(at) = key.find(&marker) {
            let head = &key[..at + marker.len() - "net.".len()];
            let tail = &key[at + marker.len()..];
            if let Some(rest) = tail.strip_prefix("0.proj.") {
                return format!("{head}proj_in.{rest}");
            }
            if let Some(rest) = tail.strip_prefix("2.") {
                return format!("{head}proj_out.{rest}");
            }
        }
    }
    key.to_string()
}

/// Load the FLUX.1 transformer from a `diffusers` `transformer/` checkpoint,
/// or from an original-layout single file (`double_blocks.*`, `single_blocks.*`,
/// converted the way `diffusers`' own script does).
///
/// Fails, naming the difference, unless every tensor is consumed, every
/// parameter filled and every shape matches. A checkpoint with a guidance
/// embedder into a model built without one (or the reverse) is such a
/// difference.
pub fn load_flux_weights(
    model: &mut FluxDiT,
    weights: HashMap<String, Array>,
) -> Result<(), LoadError> {
    let original = weights.keys().any(|key| {
        let key = key.strip_prefix("model.diffusion_model.").unwrap_or(key);
        key.starts_with("double_blocks.") || key.starts_with("single_blocks.")
    });
    let weights = if original {
        flux_original_to_diffusers(weights, model.config.hidden_size() as i32)
    } else {
        weights
    };
    assign_renamed(
        model,
        weights,
        |key, value| (flux_param_name(key), value),
        &LoadPolicy::strict(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::architectures::clip::{CLIPConfig, CLIPTextModel};
    use crate::architectures::flux::FluxConfig;
    use crate::architectures::t5::{T5Config, T5EncoderModel};
    use crate::architectures::vae::VAEConfig;
    use pmetal_runtime::compat::{Dtype, ModuleParameters, ModuleParametersExt};
    use serde_json::Value;
    use std::collections::BTreeMap;

    /// The tensor names and shapes of the released FLUX.1-dev components, read
    /// from their safetensors headers.
    fn released(component: &str) -> BTreeMap<String, Vec<i32>> {
        let all: Value =
            serde_json::from_str(include_str!("../../tests/fixtures/flux_dev_layout.json"))
                .expect("fixture parses");
        all[component]
            .as_object()
            .expect("component")
            .iter()
            .map(|(name, entry)| {
                let shape = entry["shape"]
                    .as_array()
                    .expect("shape")
                    .iter()
                    .map(|d| d.as_i64().expect("dim") as i32)
                    .collect();
                (name.clone(), shape)
            })
            .collect()
    }

    /// What `load_*_weights` would assign: the released tensors after the
    /// loader's rename and layout change, as `name -> shape`. No tensor is
    /// ever read, so a model of any size can be checked.
    fn assigned(
        released: &BTreeMap<String, Vec<i32>>,
        map: impl Fn(&str, Array) -> (String, Array),
        exempt: impl Fn(&str) -> bool,
    ) -> BTreeMap<String, Vec<i32>> {
        released
            .iter()
            .filter(|(name, _)| !exempt(name))
            .map(|(name, shape)| {
                let (target, tensor) = map(name, ops_zeros(shape));
                (target, tensor.shape().to_vec())
            })
            .collect()
    }

    fn ops_zeros(shape: &[i32]) -> Array {
        pmetal_runtime::compat::ops::zeros(shape, Dtype::Bfloat16)
    }

    fn parameters<M: ModuleParameters + ModuleParametersExt>(
        model: &M,
    ) -> BTreeMap<String, Vec<i32>> {
        model
            .flatten_params()
            .into_iter()
            .map(|(name, array)| (name.to_string(), array.shape().to_vec()))
            .collect()
    }

    fn assert_same(
        what: &str,
        assigned: BTreeMap<String, Vec<i32>>,
        model: BTreeMap<String, Vec<i32>>,
    ) {
        let unfilled: Vec<_> = model
            .keys()
            .filter(|k| !assigned.contains_key(*k))
            .take(5)
            .collect();
        let unused: Vec<_> = assigned
            .keys()
            .filter(|k| !model.contains_key(*k))
            .take(5)
            .collect();
        assert!(
            unfilled.is_empty() && unused.is_empty(),
            "{what}: parameters no released tensor fills {unfilled:?}; released tensors no parameter takes {unused:?}"
        );
        for (name, shape) in &model {
            assert_eq!(&assigned[name], shape, "{what}: {name}");
        }
    }

    // The four tests below build the *released-size* models without
    // evaluating them: parameters are lazy, so only names and shapes exist.

    #[test]
    fn the_released_flux_transformer_layout_loads() {
        let model = crate::architectures::flux::FluxDiT::new(FluxConfig {
            guidance_embeds: true,
            ..Default::default()
        })
        .unwrap();
        assert_same(
            "FLUX.1-dev transformer",
            assigned(
                &released("transformer"),
                |k, v| (flux_param_name(k), v),
                |_| false,
            ),
            parameters(&model),
        );
    }

    #[test]
    fn the_released_clip_layout_loads() {
        let model = CLIPTextModel::new(CLIPConfig::clip_vit_l_14()).unwrap();
        assert_same(
            "CLIP-L text encoder",
            assigned(
                &released("text_encoder"),
                |k, v| (clip_param_name(k), v),
                |name| name.ends_with("position_ids"),
            ),
            parameters(&model),
        );
    }

    #[test]
    fn the_released_t5_xxl_layout_loads() {
        let model = T5EncoderModel::new(T5Config {
            d_model: 4096,
            d_ff: 10240,
            d_kv: 64,
            num_layers: 24,
            num_heads: 64,
            feed_forward_proj: "gated-gelu".into(),
            ..Default::default()
        })
        .unwrap();
        assert_same(
            "T5-XXL encoder",
            assigned(
                &released("text_encoder_2"),
                |k, v| (t5_param_name(k), v),
                |_| false,
            ),
            parameters(&model),
        );
    }

    #[test]
    fn the_released_flux_vae_layout_loads() {
        let model = AutoencoderKL::new(VAEConfig::flux()).unwrap();
        assert_same(
            "FLUX.1 VAE",
            assigned(&released("vae"), vae_tensor, |_| false),
            parameters(&model),
        );
    }
}
