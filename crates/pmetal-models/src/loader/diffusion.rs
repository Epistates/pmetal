//! Loaders for the text-to-image pipeline's components: the CLIP and T5 text
//! encoders, the VAE and the Flux transformer.

use pmetal_runtime::compat::{Array, ModuleParameters, ModuleParametersExt, nn};
use std::collections::HashMap;

use super::{Exemption, LoadError, LoadPolicy, assign_weights};
use crate::architectures::clip::CLIPTextModel;
use crate::architectures::flux::FluxDiT;
use crate::architectures::t5::T5EncoderModel;
use crate::architectures::vae::FluxVAE;

/// Rename a checkpoint's tensors to the model's parameter paths, check each
/// shape against the parameter it lands on, and assign them under `policy`,
/// which fails unless checkpoint and model account for each other.
///
/// A key `rename` leaves alone keeps its checkpoint name, so it reaches no
/// parameter and is reported as unconsumed unless `policy` exempts it. Two
/// tensors renamed onto one parameter are an error rather than a race.
fn assign_renamed<M: ModuleParameters + ModuleParametersExt>(
    model: &mut M,
    weights: HashMap<String, Array>,
    rename: impl Fn(&str) -> String,
    policy: &LoadPolicy,
) -> Result<(), LoadError> {
    let mut renamed: HashMap<String, Array> = HashMap::with_capacity(weights.len());
    for (key, value) in weights {
        let target = rename(&key);
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
    assign_renamed(model, weights, clip_param_name, &policy)
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
    assign_renamed(model, weights, t5_param_name, &policy)
}

pub fn load_vae_weights(
    model: &mut FluxVAE,
    weights: &HashMap<String, Array>,
) -> Result<(), LoadError> {
    // Helper to load ResnetBlock weights from HF-style keys
    fn load_resnet_block(
        block: &mut crate::architectures::vae::ResnetBlock,
        weights: &HashMap<String, Array>,
        prefix: &str,
    ) -> Result<(), LoadError> {
        load_group_norm_weight(&mut block.norm1, weights, &format!("{prefix}.norm1"))?;
        load_conv2d_weight(&mut block.conv1, weights, &format!("{prefix}.conv1"))?;
        load_group_norm_weight(&mut block.norm2, weights, &format!("{prefix}.norm2"))?;
        load_conv2d_weight(&mut block.conv2, weights, &format!("{prefix}.conv2"))?;
        if let Some(ref mut shortcut) = block.conv_shortcut {
            // Only load if the weights exist (skip silently if they don't)
            let key = format!("{prefix}.conv_shortcut.weight");
            if weights.contains_key(&key) {
                load_conv2d_weight(shortcut, weights, &format!("{prefix}.conv_shortcut"))?;
            }
        }
        Ok(())
    }

    // Helper to load VAEAttentionBlock weights
    fn load_attn_block(
        attn: &mut crate::architectures::vae::VAEAttentionBlock,
        weights: &HashMap<String, Array>,
        prefix: &str,
    ) -> Result<(), LoadError> {
        load_group_norm_weight(&mut attn.norm, weights, &format!("{prefix}.group_norm"))?;
        load_conv2d_weight(&mut attn.q, weights, &format!("{prefix}.to_q"))?;
        load_conv2d_weight(&mut attn.k, weights, &format!("{prefix}.to_k"))?;
        load_conv2d_weight(&mut attn.v, weights, &format!("{prefix}.to_v"))?;
        load_conv2d_weight(&mut attn.proj_out, weights, &format!("{prefix}.to_out.0"))?;
        Ok(())
    }

    // Encoder weights
    if let Some(ref mut encoder) = model.encoder {
        load_conv2d_weight(&mut encoder.conv_in, weights, "encoder.conv_in")?;

        // Down blocks: block 0 has no downsampler, blocks 1-3 have downsamplers
        load_resnet_block(
            &mut encoder.down_1_0,
            weights,
            "encoder.down_blocks.0.resnets.0",
        )?;
        load_resnet_block(
            &mut encoder.down_1_1,
            weights,
            "encoder.down_blocks.0.resnets.1",
        )?;

        load_resnet_block(
            &mut encoder.down_2_0,
            weights,
            "encoder.down_blocks.1.resnets.0",
        )?;
        load_resnet_block(
            &mut encoder.down_2_1,
            weights,
            "encoder.down_blocks.1.resnets.1",
        )?;
        load_conv2d_weight(
            &mut encoder.down_2_sampler.conv,
            weights,
            "encoder.down_blocks.1.downsamplers.0.conv",
        )?;

        load_resnet_block(
            &mut encoder.down_3_0,
            weights,
            "encoder.down_blocks.2.resnets.0",
        )?;
        load_resnet_block(
            &mut encoder.down_3_1,
            weights,
            "encoder.down_blocks.2.resnets.1",
        )?;
        load_conv2d_weight(
            &mut encoder.down_3_sampler.conv,
            weights,
            "encoder.down_blocks.2.downsamplers.0.conv",
        )?;

        load_resnet_block(
            &mut encoder.down_4_0,
            weights,
            "encoder.down_blocks.3.resnets.0",
        )?;
        load_resnet_block(
            &mut encoder.down_4_1,
            weights,
            "encoder.down_blocks.3.resnets.1",
        )?;
        load_conv2d_weight(
            &mut encoder.down_4_sampler.conv,
            weights,
            "encoder.down_blocks.3.downsamplers.0.conv",
        )?;

        // Mid block
        load_resnet_block(
            &mut encoder.mid_block_1,
            weights,
            "encoder.mid_block.resnets.0",
        )?;
        load_attn_block(
            &mut encoder.mid_attn,
            weights,
            "encoder.mid_block.attentions.0",
        )?;
        load_resnet_block(
            &mut encoder.mid_block_2,
            weights,
            "encoder.mid_block.resnets.1",
        )?;

        load_group_norm_weight(&mut encoder.norm_out, weights, "encoder.conv_norm_out")?;
        load_conv2d_weight(&mut encoder.conv_out, weights, "encoder.conv_out")?;
    }

    // Decoder weights
    let decoder = &mut model.decoder;
    load_conv2d_weight(&mut decoder.conv_in, weights, "decoder.conv_in")?;

    // Mid block
    load_resnet_block(
        &mut decoder.mid_block_1,
        weights,
        "decoder.mid_block.resnets.0",
    )?;
    load_attn_block(
        &mut decoder.mid_attn,
        weights,
        "decoder.mid_block.attentions.0",
    )?;
    load_resnet_block(
        &mut decoder.mid_block_2,
        weights,
        "decoder.mid_block.resnets.1",
    )?;

    // Up blocks: blocks 0-2 have upsamplers, block 3 does not
    load_resnet_block(
        &mut decoder.up_1_0,
        weights,
        "decoder.up_blocks.0.resnets.0",
    )?;
    load_resnet_block(
        &mut decoder.up_1_1,
        weights,
        "decoder.up_blocks.0.resnets.1",
    )?;
    load_resnet_block(
        &mut decoder.up_1_2,
        weights,
        "decoder.up_blocks.0.resnets.2",
    )?;
    load_conv2d_weight(
        &mut decoder.up_1_sampler.conv,
        weights,
        "decoder.up_blocks.0.upsamplers.0.conv",
    )?;

    load_resnet_block(
        &mut decoder.up_2_0,
        weights,
        "decoder.up_blocks.1.resnets.0",
    )?;
    load_resnet_block(
        &mut decoder.up_2_1,
        weights,
        "decoder.up_blocks.1.resnets.1",
    )?;
    load_resnet_block(
        &mut decoder.up_2_2,
        weights,
        "decoder.up_blocks.1.resnets.2",
    )?;
    load_conv2d_weight(
        &mut decoder.up_2_sampler.conv,
        weights,
        "decoder.up_blocks.1.upsamplers.0.conv",
    )?;

    load_resnet_block(
        &mut decoder.up_3_0,
        weights,
        "decoder.up_blocks.2.resnets.0",
    )?;
    load_resnet_block(
        &mut decoder.up_3_1,
        weights,
        "decoder.up_blocks.2.resnets.1",
    )?;
    load_resnet_block(
        &mut decoder.up_3_2,
        weights,
        "decoder.up_blocks.2.resnets.2",
    )?;
    load_conv2d_weight(
        &mut decoder.up_3_sampler.conv,
        weights,
        "decoder.up_blocks.2.upsamplers.0.conv",
    )?;

    load_resnet_block(
        &mut decoder.up_4_0,
        weights,
        "decoder.up_blocks.3.resnets.0",
    )?;
    load_resnet_block(
        &mut decoder.up_4_1,
        weights,
        "decoder.up_blocks.3.resnets.1",
    )?;
    load_resnet_block(
        &mut decoder.up_4_2,
        weights,
        "decoder.up_blocks.3.resnets.2",
    )?;

    load_group_norm_weight(&mut decoder.norm_out, weights, "decoder.conv_norm_out")?;
    load_conv2d_weight(&mut decoder.conv_out, weights, "decoder.conv_out")?;

    Ok(())
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

fn load_conv2d_weight(
    conv: &mut nn::Conv2d,
    weights: &HashMap<String, Array>,
    prefix: &str,
) -> Result<(), LoadError> {
    let weight_key = format!("{prefix}.weight");
    if let Some(w) = weights.get(&weight_key) {
        // HF/PyTorch Conv2d weights are [O, I, H, W], MLX uses [O, H, W, I]
        let w = w.transpose_axes(&[0, 2, 3, 1]);
        conv.weight = pmetal_runtime::compat::module::Param::new(w);
    } else {
        return Err(LoadError::MissingWeight(weight_key));
    }
    let bias_key = format!("{prefix}.bias");
    if let Some(b) = weights.get(&bias_key) {
        conv.bias = pmetal_runtime::compat::module::Param::new(Some(b.clone()));
    }
    Ok(())
}

fn load_group_norm_weight(
    norm: &mut nn::GroupNorm,
    weights: &HashMap<String, Array>,
    prefix: &str,
) -> Result<(), LoadError> {
    let weight_key = format!("{prefix}.weight");
    if let Some(w) = weights.get(&weight_key) {
        norm.weight = pmetal_runtime::compat::module::Param::new(Some(w.clone()));
    } else {
        return Err(LoadError::MissingWeight(weight_key));
    }
    let bias_key = format!("{prefix}.bias");
    if let Some(b) = weights.get(&bias_key) {
        norm.bias = pmetal_runtime::compat::module::Param::new(Some(b.clone()));
    }
    Ok(())
}
