//! Model loading utilities for PMetal.
//!
//! Provides functionality to load model weights from safetensor files,
//! with support for HuggingFace model formats and weight name mapping.

use std::collections::{HashMap, HashSet};
use std::path::Path;

pub use mlx_rs::module::ModuleParametersExt;
use mlx_rs::{Array, nn};

use crate::architectures::clip::CLIPTextModel;
use crate::architectures::flux::FluxDiT;
use crate::architectures::gemma::GemmaForCausalLM;
use crate::architectures::llama::{LlamaConfig, LlamaForCausalLM};
use crate::architectures::mistral::MistralForCausalLM;
use crate::architectures::mllama::MllamaForConditionalGeneration;
use crate::architectures::nemotron_h::{
    NemotronHForCausalLM, load_nemotron_weights as load_nemotron,
};
use crate::architectures::phi::PhiForCausalLM;
use crate::architectures::qwen2::Qwen2ForCausalLM;
use crate::architectures::qwen3::Qwen3ForCausalLM;
use crate::architectures::t5::T5EncoderModel;
use crate::architectures::vae::FluxVAE;

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("SafeTensors error: {0}")]
    SafeTensors(String),
    #[error("Missing weight: {0}")]
    MissingWeight(String),
    #[error("Shape mismatch for {key}: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        key: String,
        expected: Vec<i32>,
        actual: Vec<i32>,
    },
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("MLX error: {0}")]
    Mlx(String),
    #[error("MLX IO error: {0}")]
    MlxIo(#[from] mlx_rs::error::IoError),
}

impl From<mlx_rs::error::Exception> for LoadError {
    fn from(e: mlx_rs::error::Exception) -> Self {
        Self::Mlx(e.to_string())
    }
}

pub fn load_clip_weights(
    model: &mut CLIPTextModel,
    weights: &HashMap<String, Array>,
) -> Result<(), LoadError> {
    for (name, weight) in weights {
        let name = if name.starts_with("text_model.") {
            &name["text_model.".len()..]
        } else {
            name
        };

        match name {
            "embeddings.token_embedding.weight" => {
                model.token_embedding.weight = mlx_rs::module::Param::new(weight.clone())
            }
            "embeddings.position_embedding.weight" => {
                model.position_embedding = mlx_rs::module::Param::new(weight.clone())
            }
            "final_layer_norm.weight" => {
                model.final_layer_norm.weight = mlx_rs::module::Param::new(Some(weight.clone()))
            }
            "final_layer_norm.bias" => {
                model.final_layer_norm.bias = mlx_rs::module::Param::new(Some(weight.clone()))
            }
            _ if name.starts_with("encoder.layers.") => {
                let parts: Vec<&str> = name.split('.').collect();
                let idx = parts[2].parse::<usize>().map_err(|_| {
                    LoadError::SafeTensors(format!("Invalid layer index in key: {}", name))
                })?;
                if idx >= model.layers.len() {
                    continue;
                }
                let sub_path = parts[3..].join(".");
                match sub_path.as_str() {
                    "self_attn.q_proj.weight" => {
                        model.layers[idx].attn.q_proj.weight =
                            mlx_rs::module::Param::new(weight.clone())
                    }
                    "self_attn.q_proj.bias" => {
                        model.layers[idx].attn.q_proj.bias =
                            mlx_rs::module::Param::new(Some(weight.clone()))
                    }
                    "self_attn.k_proj.weight" => {
                        model.layers[idx].attn.k_proj.weight =
                            mlx_rs::module::Param::new(weight.clone())
                    }
                    "self_attn.k_proj.bias" => {
                        model.layers[idx].attn.k_proj.bias =
                            mlx_rs::module::Param::new(Some(weight.clone()))
                    }
                    "self_attn.v_proj.weight" => {
                        model.layers[idx].attn.v_proj.weight =
                            mlx_rs::module::Param::new(weight.clone())
                    }
                    "self_attn.v_proj.bias" => {
                        model.layers[idx].attn.v_proj.bias =
                            mlx_rs::module::Param::new(Some(weight.clone()))
                    }
                    "self_attn.out_proj.weight" => {
                        model.layers[idx].attn.out_proj.weight =
                            mlx_rs::module::Param::new(weight.clone())
                    }
                    "self_attn.out_proj.bias" => {
                        model.layers[idx].attn.out_proj.bias =
                            mlx_rs::module::Param::new(Some(weight.clone()))
                    }
                    "layer_norm1.weight" => {
                        model.layers[idx].norm1.weight =
                            mlx_rs::module::Param::new(Some(weight.clone()))
                    }
                    "layer_norm1.bias" => {
                        model.layers[idx].norm1.bias =
                            mlx_rs::module::Param::new(Some(weight.clone()))
                    }
                    "layer_norm2.weight" => {
                        model.layers[idx].norm2.weight =
                            mlx_rs::module::Param::new(Some(weight.clone()))
                    }
                    "layer_norm2.bias" => {
                        model.layers[idx].norm2.bias =
                            mlx_rs::module::Param::new(Some(weight.clone()))
                    }
                    "mlp.fc1.weight" => {
                        model.layers[idx].mlp.fc1.weight =
                            mlx_rs::module::Param::new(weight.clone())
                    }
                    "mlp.fc1.bias" => {
                        model.layers[idx].mlp.fc1.bias =
                            mlx_rs::module::Param::new(Some(weight.clone()))
                    }
                    "mlp.fc2.weight" => {
                        model.layers[idx].mlp.fc2.weight =
                            mlx_rs::module::Param::new(weight.clone())
                    }
                    "mlp.fc2.bias" => {
                        model.layers[idx].mlp.fc2.bias =
                            mlx_rs::module::Param::new(Some(weight.clone()))
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }
    Ok(())
}

pub fn load_t5_weights(
    model: &mut T5EncoderModel,
    weights: &HashMap<String, Array>,
) -> Result<(), LoadError> {
    for (name, weight) in weights {
        match name.as_str() {
            "shared.weight" => model.shared.weight = mlx_rs::module::Param::new(weight.clone()),
            "encoder.final_layer_norm.weight" => {
                model.final_layer_norm.weight = mlx_rs::module::Param::new(weight.clone())
            }
            _ if name.starts_with("encoder.block.") => {
                let parts: Vec<&str> = name.split('.').collect();
                let idx = parts[2].parse::<usize>().map_err(|_| {
                    LoadError::SafeTensors(format!("Invalid layer index in key: {}", name))
                })?;
                if idx >= model.blocks.len() {
                    continue;
                }
                let sub_path = parts[3..].join(".");
                match sub_path.as_str() {
                    "layer.0.SelfAttention.q.weight" => {
                        model.blocks[idx].layer_0_attn.q.weight =
                            mlx_rs::module::Param::new(weight.clone())
                    }
                    "layer.0.SelfAttention.k.weight" => {
                        model.blocks[idx].layer_0_attn.k.weight =
                            mlx_rs::module::Param::new(weight.clone())
                    }
                    "layer.0.SelfAttention.v.weight" => {
                        model.blocks[idx].layer_0_attn.v.weight =
                            mlx_rs::module::Param::new(weight.clone())
                    }
                    "layer.0.SelfAttention.o.weight" => {
                        model.blocks[idx].layer_0_attn.o.weight =
                            mlx_rs::module::Param::new(weight.clone())
                    }
                    "layer.0.SelfAttention.relative_attention_bias.weight" => {
                        if let Some(ref mut rel_bias) =
                            model.blocks[idx].layer_0_attn.relative_attention_bias
                        {
                            rel_bias.embedding.weight = mlx_rs::module::Param::new(weight.clone());
                        }
                    }
                    "layer.0.layer_norm.weight" => {
                        model.blocks[idx].layer_0_norm.weight =
                            mlx_rs::module::Param::new(weight.clone())
                    }
                    "layer.1.DenseReluDense.wi_0.weight" => {
                        model.blocks[idx].layer_1_mlp.wi_0.weight =
                            mlx_rs::module::Param::new(weight.clone())
                    }
                    "layer.1.DenseReluDense.wi_1.weight" => {
                        model.blocks[idx].layer_1_mlp.wi_1.weight =
                            mlx_rs::module::Param::new(weight.clone())
                    }
                    "layer.1.DenseReluDense.wo.weight" => {
                        model.blocks[idx].layer_1_mlp.wo.weight =
                            mlx_rs::module::Param::new(weight.clone())
                    }
                    "layer.1.layer_norm.weight" => {
                        model.blocks[idx].layer_1_norm.weight =
                            mlx_rs::module::Param::new(weight.clone())
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }
    Ok(())
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
        let name = if name.starts_with("model.diffusion_model.") {
            &name["model.diffusion_model.".len()..]
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
                        mlx_rs::module::Param::new(weight.clone())
                }
                "img_mod.lin.bias" => {
                    model.blocks[idx].norm1_a.linear.bias =
                        mlx_rs::module::Param::new(Some(weight.clone()))
                }
                "txt_mod.lin.weight" => {
                    model.blocks[idx].norm1_b.linear.weight =
                        mlx_rs::module::Param::new(weight.clone())
                }
                "txt_mod.lin.bias" => {
                    model.blocks[idx].norm1_b.linear.bias =
                        mlx_rs::module::Param::new(Some(weight.clone()))
                }

                "img_attn.qkv.weight" => {
                    model.blocks[idx].attn.a_to_qkv.weight =
                        mlx_rs::module::Param::new(weight.clone())
                }
                "img_attn.qkv.bias" => {
                    model.blocks[idx].attn.a_to_qkv.bias =
                        mlx_rs::module::Param::new(Some(weight.clone()))
                }
                "txt_attn.qkv.weight" => {
                    model.blocks[idx].attn.b_to_qkv.weight =
                        mlx_rs::module::Param::new(weight.clone())
                }
                "txt_attn.qkv.bias" => {
                    model.blocks[idx].attn.b_to_qkv.bias =
                        mlx_rs::module::Param::new(Some(weight.clone()))
                }

                "img_attn.norm.query_norm.scale" => {
                    model.blocks[idx].attn.norm_q_a.weight =
                        mlx_rs::module::Param::new(weight.clone())
                }
                "img_attn.norm.key_norm.scale" => {
                    model.blocks[idx].attn.norm_k_a.weight =
                        mlx_rs::module::Param::new(weight.clone())
                }
                "txt_attn.norm.query_norm.scale" => {
                    model.blocks[idx].attn.norm_q_b.weight =
                        mlx_rs::module::Param::new(weight.clone())
                }
                "txt_attn.norm.key_norm.scale" => {
                    model.blocks[idx].attn.norm_k_b.weight =
                        mlx_rs::module::Param::new(weight.clone())
                }

                "img_attn.proj.weight" => {
                    model.blocks[idx].attn.a_to_out.weight =
                        mlx_rs::module::Param::new(weight.clone())
                }
                "img_attn.proj.bias" => {
                    model.blocks[idx].attn.a_to_out.bias =
                        mlx_rs::module::Param::new(Some(weight.clone()))
                }
                "txt_attn.proj.weight" => {
                    model.blocks[idx].attn.b_to_out.weight =
                        mlx_rs::module::Param::new(weight.clone())
                }
                "txt_attn.proj.bias" => {
                    model.blocks[idx].attn.b_to_out.bias =
                        mlx_rs::module::Param::new(Some(weight.clone()))
                }

                "img_mlp.0.weight" => {
                    model.blocks[idx].ff_a[0].weight = mlx_rs::module::Param::new(weight.clone())
                }
                "img_mlp.0.bias" => {
                    model.blocks[idx].ff_a[0].bias =
                        mlx_rs::module::Param::new(Some(weight.clone()))
                }
                "img_mlp.2.weight" => {
                    model.blocks[idx].ff_a[1].weight = mlx_rs::module::Param::new(weight.clone())
                }
                "img_mlp.2.bias" => {
                    model.blocks[idx].ff_a[1].bias =
                        mlx_rs::module::Param::new(Some(weight.clone()))
                }

                "txt_mlp.0.weight" => {
                    model.blocks[idx].ff_b[0].weight = mlx_rs::module::Param::new(weight.clone())
                }
                "txt_mlp.0.bias" => {
                    model.blocks[idx].ff_b[0].bias =
                        mlx_rs::module::Param::new(Some(weight.clone()))
                }
                "txt_mlp.2.weight" => {
                    model.blocks[idx].ff_b[1].weight = mlx_rs::module::Param::new(weight.clone())
                }
                "txt_mlp.2.bias" => {
                    model.blocks[idx].ff_b[1].bias =
                        mlx_rs::module::Param::new(Some(weight.clone()))
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
                        mlx_rs::module::Param::new(weight.clone())
                }
                "modulation.lin.bias" => {
                    model.single_blocks[idx].norm.linear.bias =
                        mlx_rs::module::Param::new(Some(weight.clone()))
                }
                "linear1.weight" => {
                    model.single_blocks[idx].to_qkv_mlp.weight =
                        mlx_rs::module::Param::new(weight.clone())
                }
                "linear1.bias" => {
                    model.single_blocks[idx].to_qkv_mlp.bias =
                        mlx_rs::module::Param::new(Some(weight.clone()))
                }
                "linear2.weight" => {
                    model.single_blocks[idx].proj_out.weight =
                        mlx_rs::module::Param::new(weight.clone())
                }
                "linear2.bias" => {
                    model.single_blocks[idx].proj_out.bias =
                        mlx_rs::module::Param::new(Some(weight.clone()))
                }
                "norm.query_norm.scale" => {
                    model.single_blocks[idx].norm_q_a.weight =
                        mlx_rs::module::Param::new(weight.clone())
                }
                "norm.key_norm.scale" => {
                    model.single_blocks[idx].norm_k_a.weight =
                        mlx_rs::module::Param::new(weight.clone())
                }
                _ => {}
            }
        } else {
            match name {
                "time_in.in_layer.weight" => {
                    model.time_embedder.linear_1.weight = mlx_rs::module::Param::new(weight.clone())
                }
                "time_in.in_layer.bias" => {
                    model.time_embedder.linear_1.bias =
                        mlx_rs::module::Param::new(Some(weight.clone()))
                }
                "time_in.out_layer.weight" => {
                    model.time_embedder.linear_2.weight = mlx_rs::module::Param::new(weight.clone())
                }
                "time_in.out_layer.bias" => {
                    model.time_embedder.linear_2.bias =
                        mlx_rs::module::Param::new(Some(weight.clone()))
                }

                "txt_in.weight" => {
                    model.context_embedder.weight = mlx_rs::module::Param::new(weight.clone())
                }
                "txt_in.bias" => {
                    model.context_embedder.bias = mlx_rs::module::Param::new(Some(weight.clone()))
                }

                "vector_in.in_layer.weight" => {
                    model.pooled_text_embedder[0].weight =
                        mlx_rs::module::Param::new(weight.clone())
                }
                "vector_in.in_layer.bias" => {
                    model.pooled_text_embedder[0].bias =
                        mlx_rs::module::Param::new(Some(weight.clone()))
                }
                "vector_in.out_layer.weight" => {
                    model.pooled_text_embedder[1].weight =
                        mlx_rs::module::Param::new(weight.clone())
                }
                "vector_in.out_layer.bias" => {
                    model.pooled_text_embedder[1].bias =
                        mlx_rs::module::Param::new(Some(weight.clone()))
                }

                "guidance_in.in_layer.weight" => {
                    if let Some(ref mut ge) = model.guidance_embedder {
                        ge.linear_1.weight = mlx_rs::module::Param::new(weight.clone());
                    }
                }
                "guidance_in.in_layer.bias" => {
                    if let Some(ref mut ge) = model.guidance_embedder {
                        ge.linear_1.bias = mlx_rs::module::Param::new(Some(weight.clone()));
                    }
                }
                "guidance_in.out_layer.weight" => {
                    if let Some(ref mut ge) = model.guidance_embedder {
                        ge.linear_2.weight = mlx_rs::module::Param::new(weight.clone());
                    }
                }
                "guidance_in.out_layer.bias" => {
                    if let Some(ref mut ge) = model.guidance_embedder {
                        ge.linear_2.bias = mlx_rs::module::Param::new(Some(weight.clone()));
                    }
                }

                "img_in.weight" => {
                    model.x_embedder.weight = mlx_rs::module::Param::new(weight.clone())
                }
                "img_in.bias" => {
                    model.x_embedder.bias = mlx_rs::module::Param::new(Some(weight.clone()))
                }

                "final_layer.adaLN_modulation.1.weight" => {
                    model.final_norm_out.linear.weight = mlx_rs::module::Param::new(weight.clone())
                }
                "final_layer.adaLN_modulation.1.bias" => {
                    model.final_norm_out.linear.bias =
                        mlx_rs::module::Param::new(Some(weight.clone()))
                }
                "final_layer.linear.weight" => {
                    model.final_proj_out.weight = mlx_rs::module::Param::new(weight.clone())
                }
                "final_layer.linear.bias" => {
                    model.final_proj_out.bias = mlx_rs::module::Param::new(Some(weight.clone()))
                }
                _ => {}
            }
        }
    }
    Ok(())
}

/// Weight index for sharded models.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct WeightIndex {
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
    pub weight_map: HashMap<String, String>,
}

/// Validate that a shard path does not escape the model directory (path traversal protection).
fn validate_shard_path(
    model_dir: &Path,
    shard_file: &str,
) -> Result<std::path::PathBuf, LoadError> {
    let shard_path = model_dir.join(shard_file);
    // Canonicalize both paths to resolve symlinks and ../ components
    let canonical_dir = model_dir.canonicalize()?;
    let canonical_shard = shard_path.canonicalize().map_err(|e| {
        LoadError::Io(std::io::Error::new(
            e.kind(),
            format!(
                "Shard file not found: {} (in {})",
                shard_file,
                model_dir.display()
            ),
        ))
    })?;
    if !canonical_shard.starts_with(&canonical_dir) {
        return Err(LoadError::Io(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            format!("Shard path escapes model directory: {:?}", shard_path),
        )));
    }
    Ok(canonical_shard)
}

/// Load generic weights using safetensors.
pub fn load_generic_weights<M: ModuleParametersExt>(
    model: &mut M,
    model_dir: impl AsRef<Path>,
) -> Result<(), LoadError> {
    let model_dir = model_dir.as_ref();
    let single_file = model_dir.join("model.safetensors");
    if single_file.exists() {
        model
            .load_safetensors(&single_file)
            .map_err(|e| LoadError::SafeTensors(e.to_string()))?;
        return Ok(());
    }
    let index_path = model_dir.join("model.safetensors.index.json");
    if !index_path.exists() {
        return Err(LoadError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "No weights found",
        )));
    }
    let index_content = std::fs::read_to_string(&index_path)?;
    let index: WeightIndex = serde_json::from_str(&index_content)?;
    let shard_files: HashSet<&String> = index.weight_map.values().collect();
    for shard_file in shard_files {
        let shard_path = validate_shard_path(model_dir, shard_file)?;
        model
            .load_safetensors(&shard_path)
            .map_err(|e| LoadError::SafeTensors(e.to_string()))?;
    }
    Ok(())
}

pub fn load_nemotron_weights(
    model: &mut NemotronHForCausalLM,
    model_dir: impl AsRef<Path>,
) -> Result<(), LoadError> {
    let weights = load_weights(model_dir)?;
    load_nemotron(model, &weights).map_err(|e| LoadError::SafeTensors(format!("{:?}", e)))
}

/// Load all safetensor weights from a model directory.
pub fn load_weights(model_dir: impl AsRef<Path>) -> Result<HashMap<String, Array>, LoadError> {
    let model_dir = model_dir.as_ref();
    let mut all_weights = HashMap::new();
    let single_file = model_dir.join("model.safetensors");
    if single_file.exists() {
        let weights =
            Array::load_safetensors(&single_file).map_err(|e| LoadError::Mlx(e.to_string()))?;
        return Ok(weights);
    }
    let index_path = model_dir.join("model.safetensors.index.json");
    if !index_path.exists() {
        return Err(LoadError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "No weights found",
        )));
    }
    let index_content = std::fs::read_to_string(&index_path)?;
    let index: WeightIndex = serde_json::from_str(&index_content)?;
    let shard_files: HashSet<&String> = index.weight_map.values().collect();
    for shard_file in shard_files {
        let shard_path = validate_shard_path(model_dir, shard_file)?;
        let shard_weights =
            Array::load_safetensors(&shard_path).map_err(|e| LoadError::Mlx(e.to_string()))?;
        all_weights.extend(shard_weights);
    }
    Ok(all_weights)
}

pub fn load_llama_weights(
    model: &mut LlamaForCausalLM,
    weights: &HashMap<String, Array>,
) -> Result<(), LoadError> {
    if let Some(w) = weights.get("model.embed_tokens.weight") {
        model.model.embed_tokens.weight = mlx_rs::module::Param::new(w.clone());
    } else {
        return Err(LoadError::MissingWeight("model.embed_tokens.weight".into()));
    }
    for (i, layer) in model.model.layers.iter_mut().enumerate() {
        let prefix = format!("model.layers.{i}");
        load_linear_weight(
            &mut layer.self_attn.q_proj,
            weights,
            &format!("{prefix}.self_attn.q_proj"),
        )?;
        load_linear_weight(
            &mut layer.self_attn.k_proj,
            weights,
            &format!("{prefix}.self_attn.k_proj"),
        )?;
        load_linear_weight(
            &mut layer.self_attn.v_proj,
            weights,
            &format!("{prefix}.self_attn.v_proj"),
        )?;
        load_linear_weight(
            &mut layer.self_attn.o_proj,
            weights,
            &format!("{prefix}.self_attn.o_proj"),
        )?;
        load_linear_weight(
            &mut layer.mlp.gate_proj,
            weights,
            &format!("{prefix}.mlp.gate_proj"),
        )?;
        load_linear_weight(
            &mut layer.mlp.up_proj,
            weights,
            &format!("{prefix}.mlp.up_proj"),
        )?;
        load_linear_weight(
            &mut layer.mlp.down_proj,
            weights,
            &format!("{prefix}.mlp.down_proj"),
        )?;
        load_rms_norm_weight(
            &mut layer.input_layernorm,
            weights,
            &format!("{prefix}.input_layernorm"),
        )?;
        load_rms_norm_weight(
            &mut layer.post_attention_layernorm,
            weights,
            &format!("{prefix}.post_attention_layernorm"),
        )?;
    }
    load_rms_norm_weight(&mut model.model.norm, weights, "model.norm")?;
    if let Some(ref mut lm_head) = model.lm_head {
        if let Some(w) = weights.get("lm_head.weight") {
            lm_head.weight = mlx_rs::module::Param::new(w.clone());
        }
    }
    Ok(())
}

pub fn load_mistral_weights(
    model: &mut MistralForCausalLM,
    weights: &HashMap<String, Array>,
) -> Result<(), LoadError> {
    if let Some(w) = weights.get("model.embed_tokens.weight") {
        model.model.embed_tokens.weight = mlx_rs::module::Param::new(w.clone());
    } else {
        return Err(LoadError::MissingWeight("model.embed_tokens.weight".into()));
    }
    for (i, layer) in model.model.layers.iter_mut().enumerate() {
        let prefix = format!("model.layers.{i}");
        load_linear_weight(
            &mut layer.self_attn.q_proj,
            weights,
            &format!("{prefix}.self_attn.q_proj"),
        )?;
        load_linear_weight(
            &mut layer.self_attn.k_proj,
            weights,
            &format!("{prefix}.self_attn.k_proj"),
        )?;
        load_linear_weight(
            &mut layer.self_attn.v_proj,
            weights,
            &format!("{prefix}.self_attn.v_proj"),
        )?;
        load_linear_weight(
            &mut layer.self_attn.o_proj,
            weights,
            &format!("{prefix}.self_attn.o_proj"),
        )?;
        load_linear_weight(
            &mut layer.mlp.gate_proj,
            weights,
            &format!("{prefix}.mlp.gate_proj"),
        )?;
        load_linear_weight(
            &mut layer.mlp.up_proj,
            weights,
            &format!("{prefix}.mlp.up_proj"),
        )?;
        load_linear_weight(
            &mut layer.mlp.down_proj,
            weights,
            &format!("{prefix}.mlp.down_proj"),
        )?;
        load_rms_norm_weight(
            &mut layer.input_layernorm,
            weights,
            &format!("{prefix}.input_layernorm"),
        )?;
        load_rms_norm_weight(
            &mut layer.post_attention_layernorm,
            weights,
            &format!("{prefix}.post_attention_layernorm"),
        )?;
    }
    load_rms_norm_weight(&mut model.model.norm, weights, "model.norm")?;
    if let Some(ref mut lm_head) = model.lm_head {
        if let Some(w) = weights.get("lm_head.weight") {
            lm_head.weight = mlx_rs::module::Param::new(w.clone());
        }
    }
    Ok(())
}

pub fn load_gemma_weights(
    model: &mut GemmaForCausalLM,
    weights: &HashMap<String, Array>,
) -> Result<(), LoadError> {
    if let Some(w) = weights.get("model.embed_tokens.weight") {
        model.model.embed_tokens.weight = mlx_rs::module::Param::new(w.clone());
    } else {
        return Err(LoadError::MissingWeight("model.embed_tokens.weight".into()));
    }
    if let Some(ref mut layers) = model.model.layers.gemma1 {
        for (i, layer) in layers.iter_mut().enumerate() {
            let prefix = format!("model.layers.{i}");
            load_linear_weight(
                &mut layer.self_attn.q_proj,
                weights,
                &format!("{prefix}.self_attn.q_proj"),
            )?;
            load_linear_weight(
                &mut layer.self_attn.k_proj,
                weights,
                &format!("{prefix}.self_attn.k_proj"),
            )?;
            load_linear_weight(
                &mut layer.self_attn.v_proj,
                weights,
                &format!("{prefix}.self_attn.v_proj"),
            )?;
            load_linear_weight(
                &mut layer.self_attn.o_proj,
                weights,
                &format!("{prefix}.self_attn.o_proj"),
            )?;
            load_linear_weight(
                &mut layer.mlp.gate_proj,
                weights,
                &format!("{prefix}.mlp.gate_proj"),
            )?;
            load_linear_weight(
                &mut layer.mlp.up_proj,
                weights,
                &format!("{prefix}.mlp.up_proj"),
            )?;
            load_linear_weight(
                &mut layer.mlp.down_proj,
                weights,
                &format!("{prefix}.mlp.down_proj"),
            )?;
            load_gemma_rms_norm_weight(
                &mut layer.input_layernorm,
                weights,
                &format!("{prefix}.input_layernorm"),
            )?;
            load_gemma_rms_norm_weight(
                &mut layer.post_attention_layernorm,
                weights,
                &format!("{prefix}.post_attention_layernorm"),
            )?;
        }
    } else if let Some(ref mut layers) = model.model.layers.gemma2 {
        for (i, layer) in layers.iter_mut().enumerate() {
            let prefix = format!("model.layers.{i}");
            load_linear_weight(
                &mut layer.self_attn.q_proj,
                weights,
                &format!("{prefix}.self_attn.q_proj"),
            )?;
            load_linear_weight(
                &mut layer.self_attn.k_proj,
                weights,
                &format!("{prefix}.self_attn.k_proj"),
            )?;
            load_linear_weight(
                &mut layer.self_attn.v_proj,
                weights,
                &format!("{prefix}.self_attn.v_proj"),
            )?;
            load_linear_weight(
                &mut layer.self_attn.o_proj,
                weights,
                &format!("{prefix}.self_attn.o_proj"),
            )?;
            load_linear_weight(
                &mut layer.mlp.gate_proj,
                weights,
                &format!("{prefix}.mlp.gate_proj"),
            )?;
            load_linear_weight(
                &mut layer.mlp.up_proj,
                weights,
                &format!("{prefix}.mlp.up_proj"),
            )?;
            load_linear_weight(
                &mut layer.mlp.down_proj,
                weights,
                &format!("{prefix}.mlp.down_proj"),
            )?;
            load_gemma_rms_norm_weight(
                &mut layer.input_layernorm,
                weights,
                &format!("{prefix}.input_layernorm"),
            )?;
            load_gemma_rms_norm_weight(
                &mut layer.post_attention_layernorm,
                weights,
                &format!("{prefix}.post_attention_layernorm"),
            )?;
            load_gemma_rms_norm_weight(
                &mut layer.pre_feedforward_layernorm,
                weights,
                &format!("{prefix}.pre_feedforward_layernorm"),
            )?;
            load_gemma_rms_norm_weight(
                &mut layer.post_feedforward_layernorm,
                weights,
                &format!("{prefix}.post_feedforward_layernorm"),
            )?;
        }
    }
    load_gemma_rms_norm_weight(&mut model.model.norm, weights, "model.norm")?;
    Ok(())
}

pub fn load_phi_weights(
    model: &mut PhiForCausalLM,
    weights: &HashMap<String, Array>,
) -> Result<(), LoadError> {
    if let Some(w) = weights.get("model.embed_tokens.weight") {
        model.model.embed_tokens.weight = mlx_rs::module::Param::new(w.clone());
    } else {
        return Err(LoadError::MissingWeight("model.embed_tokens.weight".into()));
    }
    for (i, layer) in model.model.layers.iter_mut().enumerate() {
        let prefix = format!("model.layers.{i}");
        load_linear_weight(
            &mut layer.self_attn.q_proj,
            weights,
            &format!("{prefix}.self_attn.q_proj"),
        )?;
        load_linear_weight(
            &mut layer.self_attn.k_proj,
            weights,
            &format!("{prefix}.self_attn.k_proj"),
        )?;
        load_linear_weight(
            &mut layer.self_attn.v_proj,
            weights,
            &format!("{prefix}.self_attn.v_proj"),
        )?;
        load_linear_weight(
            &mut layer.self_attn.o_proj,
            weights,
            &format!("{prefix}.self_attn.o_proj"),
        )?;
        load_linear_weight(
            &mut layer.mlp.gate_up_proj,
            weights,
            &format!("{prefix}.mlp.gate_up_proj"),
        )?;
        load_linear_weight(
            &mut layer.mlp.down_proj,
            weights,
            &format!("{prefix}.mlp.down_proj"),
        )?;
        load_phi_rms_norm_weight(
            &mut layer.input_layernorm,
            weights,
            &format!("{prefix}.input_layernorm"),
        )?;
        load_phi_rms_norm_weight(
            &mut layer.post_attention_layernorm,
            weights,
            &format!("{prefix}.post_attention_layernorm"),
        )?;
    }
    load_phi_rms_norm_weight(&mut model.model.norm, weights, "model.norm")?;
    load_linear_weight(&mut model.lm_head, weights, "lm_head")?;
    Ok(())
}

fn load_linear_weight(
    linear: &mut mlx_rs::nn::Linear,
    weights: &HashMap<String, Array>,
    prefix: &str,
) -> Result<(), LoadError> {
    let weight_key = format!("{prefix}.weight");
    if let Some(w) = weights.get(&weight_key) {
        linear.weight = mlx_rs::module::Param::new(w.clone());
    } else {
        return Err(LoadError::MissingWeight(weight_key));
    }
    let bias_key = format!("{prefix}.bias");
    if let Some(b) = weights.get(&bias_key) {
        linear.bias = mlx_rs::module::Param::new(Some(b.clone()));
    }
    Ok(())
}

fn load_rms_norm_weight(
    norm: &mut mlx_rs::nn::RmsNorm,
    weights: &HashMap<String, Array>,
    prefix: &str,
) -> Result<(), LoadError> {
    let weight_key = format!("{prefix}.weight");
    if let Some(w) = weights.get(&weight_key) {
        norm.weight = mlx_rs::module::Param::new(w.clone());
    } else {
        return Err(LoadError::MissingWeight(weight_key));
    }
    Ok(())
}

fn load_layer_norm_weight(
    norm: &mut mlx_rs::nn::LayerNorm,
    weights: &HashMap<String, Array>,
    prefix: &str,
) -> Result<(), LoadError> {
    let weight_key = format!("{prefix}.weight");
    if let Some(w) = weights.get(&weight_key) {
        norm.weight = mlx_rs::module::Param::new(Some(w.clone()));
    } else {
        return Err(LoadError::MissingWeight(weight_key));
    }
    let bias_key = format!("{prefix}.bias");
    if let Some(b) = weights.get(&bias_key) {
        norm.bias = mlx_rs::module::Param::new(Some(b.clone()));
    }
    Ok(())
}

fn load_gemma_rms_norm_weight(
    norm: &mut crate::architectures::gemma::GemmaRmsNorm,
    weights: &HashMap<String, Array>,
    prefix: &str,
) -> Result<(), LoadError> {
    let weight_key = format!("{prefix}.weight");
    if let Some(w) = weights.get(&weight_key) {
        norm.weight = mlx_rs::module::Param::new(w.clone());
    } else {
        return Err(LoadError::MissingWeight(weight_key));
    }
    Ok(())
}

fn load_phi_rms_norm_weight(
    norm: &mut crate::architectures::phi::PhiRMSNorm,
    weights: &HashMap<String, Array>,
    prefix: &str,
) -> Result<(), LoadError> {
    let weight_key = format!("{prefix}.weight");
    if let Some(w) = weights.get(&weight_key) {
        norm.weight = mlx_rs::module::Param::new(w.clone());
    } else {
        return Err(LoadError::MissingWeight(weight_key));
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
        let w = w
            .transpose_axes(&[0, 2, 3, 1])
            .map_err(|e| LoadError::Mlx(e.to_string()))?;
        conv.weight = mlx_rs::module::Param::new(w);
    } else {
        return Err(LoadError::MissingWeight(weight_key));
    }
    let bias_key = format!("{prefix}.bias");
    if let Some(b) = weights.get(&bias_key) {
        conv.bias = mlx_rs::module::Param::new(Some(b.clone()));
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
        norm.weight = mlx_rs::module::Param::new(Some(w.clone()));
    } else {
        return Err(LoadError::MissingWeight(weight_key));
    }
    let bias_key = format!("{prefix}.bias");
    if let Some(b) = weights.get(&bias_key) {
        norm.bias = mlx_rs::module::Param::new(Some(b.clone()));
    }
    Ok(())
}
