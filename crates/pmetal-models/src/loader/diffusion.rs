//! Loaders for the text-to-image pipeline's components: the CLIP and T5 text
//! encoders, the VAE and the Flux transformer.

use pmetal_runtime::compat::{Array, ModuleParameters, ModuleParametersExt, nn};
use std::collections::HashMap;

use super::{Exemption, LoadError, LoadPolicy, assign_weights};
use crate::architectures::clip::CLIPTextModel;
use crate::architectures::flux::FluxDiT;
use crate::architectures::t5::T5EncoderModel;
use crate::architectures::vae::AutoencoderKL;

/// Rename a checkpoint's tensors to the model's parameter paths, check each
/// shape against the parameter it lands on, and assign them under `policy`,
/// which fails unless checkpoint and model account for each other.
///
/// `map` turns a checkpoint key and tensor into the parameter path and tensor
/// to assign (a rename, and a transpose where the layouts differ). A key it
/// leaves alone keeps its checkpoint name, so it reaches no parameter and is
/// reported as unconsumed unless `policy` exempts it. Two tensors mapped onto
/// one parameter are an error rather than a race.
fn assign_renamed<M: ModuleParameters + ModuleParametersExt>(
    model: &mut M,
    weights: HashMap<String, Array>,
    map: impl Fn(&str, Array) -> (String, Array),
    policy: &LoadPolicy,
) -> Result<(), LoadError> {
    let mut renamed: HashMap<String, Array> = HashMap::with_capacity(weights.len());
    for (key, value) in weights {
        let (target, value) = map(&key, value);
        if renamed.insert(target.clone(), value).is_some() {
            return Err(LoadError::SafeTensors(format!(
                "two checkpoint tensors map onto the parameter `{target}` (one is `{key}`)"
            )));
        }
    }
    {
        let params = model.flatten_params();
        for (name, value) in &renamed {
            if let Some(param) = params.get(name.as_str())
                && param.shape() != value.shape()
            {
                return Err(LoadError::ShapeMismatch {
                    key: name.clone(),
                    expected: param.shape().to_vec(),
                    actual: value.shape().to_vec(),
                });
            }
        }
    }
    assign_weights(model, renamed, policy)
}

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
    assign_renamed(model, weights, |key, value| (clip_param_name(key), value), &policy)
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
    assign_renamed(model, weights, |key, value| (t5_param_name(key), value), &policy)
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

pub fn load_flux_weights(
    model: &mut FluxDiT,
    weights: &HashMap<String, Array>,
) -> Result<(), LoadError> {
    for (name, weight) in weights {
        let name = if let Some(stripped) = name.strip_prefix("model.diffusion_model.") {
            stripped
        } else {
            name
        };

        if name.starts_with("double_blocks.") {
            let parts: Vec<&str> = name.split('.').collect();
            let idx = parts[1].parse::<usize>().map_err(|_| {
                LoadError::SafeTensors(format!("Invalid block index in key: {}", name))
            })?;
            if idx >= model.blocks.len() {
                continue;
            }
            let sub_path = parts[2..].join(".");

            match sub_path.as_str() {
                "img_mod.lin.weight" => {
                    model.blocks[idx].norm1_a.linear.weight =
                        pmetal_runtime::compat::module::Param::new(weight.clone())
                }
                "img_mod.lin.bias" => {
                    model.blocks[idx].norm1_a.linear.bias =
                        pmetal_runtime::compat::module::Param::new(Some(weight.clone()))
                }
                "txt_mod.lin.weight" => {
                    model.blocks[idx].norm1_b.linear.weight =
                        pmetal_runtime::compat::module::Param::new(weight.clone())
                }
                "txt_mod.lin.bias" => {
                    model.blocks[idx].norm1_b.linear.bias =
                        pmetal_runtime::compat::module::Param::new(Some(weight.clone()))
                }

                "img_attn.qkv.weight" => {
                    model.blocks[idx].attn.a_to_qkv.weight =
                        pmetal_runtime::compat::module::Param::new(weight.clone())
                }
                "img_attn.qkv.bias" => {
                    model.blocks[idx].attn.a_to_qkv.bias =
                        pmetal_runtime::compat::module::Param::new(Some(weight.clone()))
                }
                "txt_attn.qkv.weight" => {
                    model.blocks[idx].attn.b_to_qkv.weight =
                        pmetal_runtime::compat::module::Param::new(weight.clone())
                }
                "txt_attn.qkv.bias" => {
                    model.blocks[idx].attn.b_to_qkv.bias =
                        pmetal_runtime::compat::module::Param::new(Some(weight.clone()))
                }

                "img_attn.norm.query_norm.scale" => {
                    model.blocks[idx].attn.norm_q_a.weight =
                        pmetal_runtime::compat::module::Param::new(weight.clone())
                }
                "img_attn.norm.key_norm.scale" => {
                    model.blocks[idx].attn.norm_k_a.weight =
                        pmetal_runtime::compat::module::Param::new(weight.clone())
                }
                "txt_attn.norm.query_norm.scale" => {
                    model.blocks[idx].attn.norm_q_b.weight =
                        pmetal_runtime::compat::module::Param::new(weight.clone())
                }
                "txt_attn.norm.key_norm.scale" => {
                    model.blocks[idx].attn.norm_k_b.weight =
                        pmetal_runtime::compat::module::Param::new(weight.clone())
                }

                "img_attn.proj.weight" => {
                    model.blocks[idx].attn.a_to_out.weight =
                        pmetal_runtime::compat::module::Param::new(weight.clone())
                }
                "img_attn.proj.bias" => {
                    model.blocks[idx].attn.a_to_out.bias =
                        pmetal_runtime::compat::module::Param::new(Some(weight.clone()))
                }
                "txt_attn.proj.weight" => {
                    model.blocks[idx].attn.b_to_out.weight =
                        pmetal_runtime::compat::module::Param::new(weight.clone())
                }
                "txt_attn.proj.bias" => {
                    model.blocks[idx].attn.b_to_out.bias =
                        pmetal_runtime::compat::module::Param::new(Some(weight.clone()))
                }

                "img_mlp.0.weight" => {
                    model.blocks[idx].ff_a[0].weight =
                        pmetal_runtime::compat::module::Param::new(weight.clone())
                }
                "img_mlp.0.bias" => {
                    model.blocks[idx].ff_a[0].bias =
                        pmetal_runtime::compat::module::Param::new(Some(weight.clone()))
                }
                "img_mlp.2.weight" => {
                    model.blocks[idx].ff_a[1].weight =
                        pmetal_runtime::compat::module::Param::new(weight.clone())
                }
                "img_mlp.2.bias" => {
                    model.blocks[idx].ff_a[1].bias =
                        pmetal_runtime::compat::module::Param::new(Some(weight.clone()))
                }

                "txt_mlp.0.weight" => {
                    model.blocks[idx].ff_b[0].weight =
                        pmetal_runtime::compat::module::Param::new(weight.clone())
                }
                "txt_mlp.0.bias" => {
                    model.blocks[idx].ff_b[0].bias =
                        pmetal_runtime::compat::module::Param::new(Some(weight.clone()))
                }
                "txt_mlp.2.weight" => {
                    model.blocks[idx].ff_b[1].weight =
                        pmetal_runtime::compat::module::Param::new(weight.clone())
                }
                "txt_mlp.2.bias" => {
                    model.blocks[idx].ff_b[1].bias =
                        pmetal_runtime::compat::module::Param::new(Some(weight.clone()))
                }
                _ => {}
            }
        } else if name.starts_with("single_blocks.") {
            let parts: Vec<&str> = name.split('.').collect();
            let idx = parts[1].parse::<usize>().map_err(|_| {
                LoadError::SafeTensors(format!("Invalid block index in key: {}", name))
            })?;
            if idx >= model.single_blocks.len() {
                continue;
            }
            let sub_path = parts[2..].join(".");

            match sub_path.as_str() {
                "modulation.lin.weight" => {
                    model.single_blocks[idx].norm.linear.weight =
                        pmetal_runtime::compat::module::Param::new(weight.clone())
                }
                "modulation.lin.bias" => {
                    model.single_blocks[idx].norm.linear.bias =
                        pmetal_runtime::compat::module::Param::new(Some(weight.clone()))
                }
                "linear1.weight" => {
                    model.single_blocks[idx].to_qkv_mlp.weight =
                        pmetal_runtime::compat::module::Param::new(weight.clone())
                }
                "linear1.bias" => {
                    model.single_blocks[idx].to_qkv_mlp.bias =
                        pmetal_runtime::compat::module::Param::new(Some(weight.clone()))
                }
                "linear2.weight" => {
                    model.single_blocks[idx].proj_out.weight =
                        pmetal_runtime::compat::module::Param::new(weight.clone())
                }
                "linear2.bias" => {
                    model.single_blocks[idx].proj_out.bias =
                        pmetal_runtime::compat::module::Param::new(Some(weight.clone()))
                }
                "norm.query_norm.scale" => {
                    model.single_blocks[idx].norm_q_a.weight =
                        pmetal_runtime::compat::module::Param::new(weight.clone())
                }
                "norm.key_norm.scale" => {
                    model.single_blocks[idx].norm_k_a.weight =
                        pmetal_runtime::compat::module::Param::new(weight.clone())
                }
                _ => {}
            }
        } else {
            match name {
                "time_in.in_layer.weight" => {
                    model.time_embedder.linear_1.weight =
                        pmetal_runtime::compat::module::Param::new(weight.clone())
                }
                "time_in.in_layer.bias" => {
                    model.time_embedder.linear_1.bias =
                        pmetal_runtime::compat::module::Param::new(Some(weight.clone()))
                }
                "time_in.out_layer.weight" => {
                    model.time_embedder.linear_2.weight =
                        pmetal_runtime::compat::module::Param::new(weight.clone())
                }
                "time_in.out_layer.bias" => {
                    model.time_embedder.linear_2.bias =
                        pmetal_runtime::compat::module::Param::new(Some(weight.clone()))
                }

                "txt_in.weight" => {
                    model.context_embedder.weight =
                        pmetal_runtime::compat::module::Param::new(weight.clone())
                }
                "txt_in.bias" => {
                    model.context_embedder.bias =
                        pmetal_runtime::compat::module::Param::new(Some(weight.clone()))
                }

                "vector_in.in_layer.weight" => {
                    model.pooled_text_embedder[0].weight =
                        pmetal_runtime::compat::module::Param::new(weight.clone())
                }
                "vector_in.in_layer.bias" => {
                    model.pooled_text_embedder[0].bias =
                        pmetal_runtime::compat::module::Param::new(Some(weight.clone()))
                }
                "vector_in.out_layer.weight" => {
                    model.pooled_text_embedder[1].weight =
                        pmetal_runtime::compat::module::Param::new(weight.clone())
                }
                "vector_in.out_layer.bias" => {
                    model.pooled_text_embedder[1].bias =
                        pmetal_runtime::compat::module::Param::new(Some(weight.clone()))
                }

                "guidance_in.in_layer.weight" => {
                    if let Some(ref mut ge) = model.guidance_embedder {
                        ge.linear_1.weight =
                            pmetal_runtime::compat::module::Param::new(weight.clone());
                    }
                }
                "guidance_in.in_layer.bias" => {
                    if let Some(ref mut ge) = model.guidance_embedder {
                        ge.linear_1.bias =
                            pmetal_runtime::compat::module::Param::new(Some(weight.clone()));
                    }
                }
                "guidance_in.out_layer.weight" => {
                    if let Some(ref mut ge) = model.guidance_embedder {
                        ge.linear_2.weight =
                            pmetal_runtime::compat::module::Param::new(weight.clone());
                    }
                }
                "guidance_in.out_layer.bias" => {
                    if let Some(ref mut ge) = model.guidance_embedder {
                        ge.linear_2.bias =
                            pmetal_runtime::compat::module::Param::new(Some(weight.clone()));
                    }
                }

                "img_in.weight" => {
                    model.x_embedder.weight =
                        pmetal_runtime::compat::module::Param::new(weight.clone())
                }
                "img_in.bias" => {
                    model.x_embedder.bias =
                        pmetal_runtime::compat::module::Param::new(Some(weight.clone()))
                }

                "final_layer.adaLN_modulation.1.weight" => {
                    model.final_norm_out.linear.weight =
                        pmetal_runtime::compat::module::Param::new(weight.clone())
                }
                "final_layer.adaLN_modulation.1.bias" => {
                    model.final_norm_out.linear.bias =
                        pmetal_runtime::compat::module::Param::new(Some(weight.clone()))
                }
                "final_layer.linear.weight" => {
                    model.final_proj_out.weight =
                        pmetal_runtime::compat::module::Param::new(weight.clone())
                }
                "final_layer.linear.bias" => {
                    model.final_proj_out.bias =
                        pmetal_runtime::compat::module::Param::new(Some(weight.clone()))
                }
                _ => {}
            }
        }
    }
    Ok(())
}
