//! Qwen 3.5-family vision: the vision tower (`model.visual.*`) of Qwen3.5,
//! Qwen3.6 and Qwen3.8 (`model_type: "qwen3_5"`), the 3-D position ids of a
//! prompt carrying images and videos, and the merge of vision features into
//! the text embeddings.
//!
//! Ported from Hugging Face transformers' `Qwen3_5VisionModel`,
//! `Qwen3_5Model.get_rope_index` and `Qwen3_5Model.forward`:
//!
//! * **Patch embedding**: a `Conv3d` whose kernel equals its stride over a
//!   `C × T × p × p` patch, so a `[hidden, C·T·p·p]` matmul plus bias.
//! * **Learned positions**: a square `side × side` table
//!   (`num_position_embeddings`), bilinearly resampled (`align_corners=True`)
//!   to each image's patch grid, in f32, then cast to the model dtype.
//! * **2-D rotary**: the row and column index of each patch, each rotating a
//!   quarter of the head's frequencies at `theta = 10000`, over the whole head
//!   (`head_dim = hidden / heads`).
//! * **Blocks**: pre-LayerNorm (eps 1e-6) attention with a fused biased
//!   `qkv` and full (bidirectional) attention *within each frame*, then an MLP
//!   with the configured activation (`gelu_pytorch_tanh` in every release:
//!   the tanh approximation, resolved through [`resolve_activation`]).
//! * **Merger**: LayerNorm, then each `merge × merge` block of patches
//!   (contiguous, since the processor emits patches block by block)
//!   concatenated and projected by `fc2(gelu(fc1(x)))` to the text width. This
//!   GELU is `nn.GELU()`, the exact erf one, not the configured activation.
//!
//! DeepStack (`deepstack_visual_indexes`) is empty in every Qwen3.5-family
//! release, and transformers' Qwen3.5 has no DeepStack path; a config that
//! names layers is refused.
//!
//! The prompt side ([`rope_index`], [`merge_media_features`]) is engine
//! agnostic: both text engines take the merged embeddings and the `[3, T]`
//! positions (`qwen3_native::forward_embeddings_hidden`,
//! `Qwen3NextForCausalLM::forward_embeddings`).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use pmetal_bridge::compat::{Array, Dtype, Exception, Param, nn, ops};
use pmetal_data::qwen_vl_processing::{ProcessedMedia, QwenVlProcessor};
use serde::Deserialize;

use super::utils::{Activation, resolve_activation};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

fn default_depth() -> usize {
    27
}
fn default_hidden_size() -> usize {
    1152
}
fn default_hidden_act() -> String {
    "gelu_pytorch_tanh".into()
}
fn default_intermediate_size() -> usize {
    4304
}
fn default_num_heads() -> usize {
    16
}
fn default_in_channels() -> usize {
    3
}
fn default_patch_size() -> usize {
    16
}
fn default_spatial_merge_size() -> usize {
    2
}
fn default_temporal_patch_size() -> usize {
    2
}
fn default_out_hidden_size() -> usize {
    3584
}
fn default_num_position_embeddings() -> usize {
    2304
}

/// `Qwen3_5VisionConfig`, with the reference class's defaults.
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3_5VisionConfig {
    /// Number of transformer blocks.
    #[serde(default = "default_depth")]
    pub depth: usize,
    /// Width of the tower.
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    /// Activation of the blocks' MLP.
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    /// Width of the blocks' MLP.
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    /// Attention heads.
    #[serde(default = "default_num_heads")]
    pub num_heads: usize,
    /// Input channels (RGB).
    #[serde(default = "default_in_channels")]
    pub in_channels: usize,
    /// Side of a patch, in pixels.
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    /// Side of the patch block merged into one text token.
    #[serde(default = "default_spatial_merge_size")]
    pub spatial_merge_size: usize,
    /// Frames per temporal patch.
    #[serde(default = "default_temporal_patch_size")]
    pub temporal_patch_size: usize,
    /// Width of the merged features: the text model's hidden size.
    #[serde(default = "default_out_hidden_size")]
    pub out_hidden_size: usize,
    /// Entries of the learned position table, a square number.
    #[serde(default = "default_num_position_embeddings")]
    pub num_position_embeddings: usize,
    /// DeepStack taps; must be empty.
    #[serde(default)]
    pub deepstack_visual_indexes: Vec<usize>,
    /// Rotary base of the 2-D vision RoPE.
    #[serde(skip, default = "default_vision_rope_theta")]
    pub rope_theta: f32,
}

fn default_vision_rope_theta() -> f32 {
    10_000.0
}

impl Qwen3_5VisionConfig {
    /// Per-head width.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    /// Width of one patch row, `C · T · p²`.
    pub fn patch_dim(&self) -> usize {
        self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
    }

    /// Side of the learned position table.
    pub fn grid_side(&self) -> usize {
        (self.num_position_embeddings as f64).sqrt() as usize
    }

    fn parse(value: &serde_json::Value) -> Result<Self, Exception> {
        let mut config: Self = serde_json::from_value(value.clone())
            .map_err(|e| Exception::custom(format!("vision_config: {e}")))?;
        if let Some(rope) = value.get("rope_parameters").filter(|v| !v.is_null()) {
            if let Some(kind) = rope.get("rope_type").and_then(|v| v.as_str())
                && kind != "axial"
            {
                return Err(Exception::custom(format!(
                    "vision_config.rope_parameters.rope_type {kind:?} is not supported; the \
                     Qwen3.5 vision tower uses axial RoPE"
                )));
            }
            if let Some(theta) = rope.get("rope_theta").and_then(|v| v.as_f64()) {
                config.rope_theta = theta as f32;
            }
        }
        if !config.deepstack_visual_indexes.is_empty() {
            return Err(Exception::custom(format!(
                "vision_config.deepstack_visual_indexes = {:?}: DeepStack is not implemented \
                 (no Qwen3.5-family release uses it)",
                config.deepstack_visual_indexes
            )));
        }
        let side = config.grid_side();
        if config.num_heads == 0
            || config.hidden_size % config.num_heads != 0
            || config.head_dim() % 4 != 0
            || side * side != config.num_position_embeddings
            || config.spatial_merge_size == 0
            || config.temporal_patch_size == 0
        {
            return Err(Exception::custom(format!(
                "vision_config is inconsistent: hidden_size {} over {} heads (the head width must \
                 split into four rotary quarters), {} position embeddings (must be square), merge \
                 {}, temporal patch {}",
                config.hidden_size,
                config.num_heads,
                config.num_position_embeddings,
                config.spatial_merge_size,
                config.temporal_patch_size
            )));
        }
        resolve_activation(&config.hidden_act).ok_or_else(|| {
            Exception::custom(format!(
                "vision_config.hidden_act {:?} is not supported",
                config.hidden_act
            ))
        })?;
        Ok(config)
    }
}

/// The multimodal half of a `Qwen3_5ForConditionalGeneration` config: the
/// vision tower and the media token ids.
#[derive(Debug, Clone)]
pub struct Qwen3_5MultimodalConfig {
    /// The vision tower.
    pub vision: Qwen3_5VisionConfig,
    /// `<|image_pad|>`.
    pub image_token_id: u32,
    /// `<|video_pad|>`.
    pub video_token_id: u32,
    /// `<|vision_start|>`.
    pub vision_start_token_id: u32,
    /// `<|vision_end|>`.
    pub vision_end_token_id: u32,
}

impl Qwen3_5MultimodalConfig {
    /// Read the multimodal fields of a `config.json`. Errors when the
    /// checkpoint has no vision tower (`vision_config` absent, or
    /// `language_model_only`).
    pub fn from_config_value(config: &serde_json::Value) -> Result<Self, Exception> {
        if config
            .get("language_model_only")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            return Err(Exception::custom(
                "this checkpoint is language-model only (`language_model_only: true`): it ships \
                 no vision tower",
            ));
        }
        let vision = config
            .get("vision_config")
            .filter(|v| v.is_object())
            .ok_or_else(|| {
                Exception::custom("config.json has no vision_config: this is a text-only model")
            })?;
        let id = |key: &str, default: u32| -> u32 {
            config
                .get(key)
                .and_then(|v| v.as_u64())
                .map_or(default, |v| v as u32)
        };
        Ok(Self {
            vision: Qwen3_5VisionConfig::parse(vision)?,
            image_token_id: id("image_token_id", 248_056),
            video_token_id: id("video_token_id", 248_057),
            vision_start_token_id: id("vision_start_token_id", 248_053),
            vision_end_token_id: id("vision_end_token_id", 248_054),
        })
    }

    /// Expand each `<|image_pad|>` id of a tokenized prompt into its image's
    /// token run. Equivalent to expanding the placeholder in the text, since
    /// the pad is a special token; videos carry timestamp text and must be
    /// expanded before tokenizing.
    pub fn expand_image_tokens(
        &self,
        input_ids: &[u32],
        images: &[ProcessedMedia],
    ) -> Result<Vec<u32>, Exception> {
        let merge = self.vision.spatial_merge_size;
        if input_ids.contains(&self.video_token_id) {
            return Err(Exception::custom(
                "video placeholders must be expanded in the prompt text, before tokenizing",
            ));
        }
        let pads = input_ids
            .iter()
            .filter(|&&id| id == self.image_token_id)
            .count();
        if pads != images.len() {
            return Err(Exception::custom(format!(
                "the prompt has {pads} image placeholders for {} images",
                images.len()
            )));
        }
        let mut images = images.iter();
        let mut out = Vec::with_capacity(input_ids.len());
        for &id in input_ids {
            if id == self.image_token_id {
                let tokens = images.next().expect("counted above").num_tokens(merge);
                out.extend(std::iter::repeat_n(id, tokens));
            } else {
                out.push(id);
            }
        }
        Ok(out)
    }

    /// [`from_config_value`](Self::from_config_value) on a checkpoint
    /// directory's `config.json`.
    pub fn from_model_dir(dir: &Path) -> Result<Self, Exception> {
        let text = std::fs::read_to_string(dir.join("config.json"))
            .map_err(|e| Exception::custom(format!("{}/config.json: {e}", dir.display())))?;
        let value: serde_json::Value =
            json5::from_str(&text).map_err(|e| Exception::custom(format!("config.json: {e}")))?;
        Self::from_config_value(&value)
    }
}

/// Whether a checkpoint directory is a Qwen3.5-family model with a vision
/// tower.
pub fn has_vision_tower(dir: &Path) -> bool {
    let Ok(text) = std::fs::read_to_string(dir.join("config.json")) else {
        return false;
    };
    let Ok(value) = json5::from_str::<serde_json::Value>(&text) else {
        return false;
    };
    value.get("model_type").and_then(|v| v.as_str()) == Some("qwen3_5")
        && Qwen3_5MultimodalConfig::from_config_value(&value).is_ok()
}

// ---------------------------------------------------------------------------
// Tower
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct VisionBlock {
    norm1: nn::LayerNorm,
    norm2: nn::LayerNorm,
    qkv: nn::Linear,
    proj: nn::Linear,
    fc1: nn::Linear,
    fc2: nn::Linear,
}

#[derive(Debug)]
struct PatchMerger {
    norm: nn::LayerNorm,
    fc1: nn::Linear,
    fc2: nn::Linear,
}

/// The Qwen 3.5-family vision tower: preprocessed patches to features of the
/// text model's width, one row per prompt media token.
#[derive(Debug)]
pub struct Qwen3_5VisionModel {
    config: Qwen3_5VisionConfig,
    /// The `Conv3d`, flattened to `[hidden, C·T·p²]`.
    patch_embed: nn::Linear,
    /// `[side², hidden]`.
    pos_embed: Param<Array>,
    blocks: Vec<VisionBlock>,
    merger: PatchMerger,
    act: Activation,
    inv_freq: Vec<f32>,
}

/// The prefix the vision tower's tensors carry in a released checkpoint.
const PREFIXES: [&str; 2] = ["model.visual.", "visual."];

const LAYER_NORM_EPS: f32 = 1e-6;

fn linear(input: usize, output: usize) -> Result<nn::Linear, Exception> {
    nn::LinearBuilder::new(input as i32, output as i32)
        .bias(true)
        .build()
}

fn layer_norm(dims: usize) -> Result<nn::LayerNorm, Exception> {
    nn::LayerNormBuilder::new(dims as i32)
        .eps(LAYER_NORM_EPS)
        .build()
}

impl Qwen3_5VisionModel {
    /// A tower with freshly initialized weights.
    pub fn new(config: Qwen3_5VisionConfig) -> Result<Self, Exception> {
        let h = config.hidden_size;
        let merged = h * config.spatial_merge_size * config.spatial_merge_size;
        let blocks = (0..config.depth)
            .map(|_| {
                Ok(VisionBlock {
                    norm1: layer_norm(h)?,
                    norm2: layer_norm(h)?,
                    qkv: linear(h, 3 * h)?,
                    proj: linear(h, h)?,
                    fc1: linear(h, config.intermediate_size)?,
                    fc2: linear(config.intermediate_size, h)?,
                })
            })
            .collect::<Result<Vec<_>, Exception>>()?;
        let act = resolve_activation(&config.hidden_act).ok_or_else(|| {
            Exception::custom(format!("unsupported activation {:?}", config.hidden_act))
        })?;
        // `compute_axial_rope_parameters`: a quarter of the head per axis.
        let spatial = config.head_dim() / 2;
        let inv_freq = (0..spatial / 2)
            .map(|j| 1.0 / config.rope_theta.powf((2 * j) as f32 / spatial as f32))
            .collect();
        Ok(Self {
            patch_embed: linear(config.patch_dim(), h)?,
            pos_embed: Param::new(Array::zeros_f32(&[
                config.num_position_embeddings as i32,
                h as i32,
            ])),
            blocks,
            merger: PatchMerger {
                norm: layer_norm(h)?,
                fc1: linear(merged, merged)?,
                fc2: linear(merged, config.out_hidden_size)?,
            },
            act,
            inv_freq,
            config,
        })
    }

    /// The tower's config.
    pub fn config(&self) -> &Qwen3_5VisionConfig {
        &self.config
    }

    /// The dtype the tower computes in: its weights'.
    pub fn dtype(&self) -> Dtype {
        self.patch_embed.weight.as_ref().dtype()
    }

    /// Cast every weight to `dtype`, which the tower then computes in.
    pub fn set_dtype(&mut self, dtype: Dtype) {
        let cast = |linear: &mut nn::Linear| {
            linear.weight = Param::new(linear.weight.as_ref().as_dtype(dtype.as_i32()));
            if let Some(bias) = linear.bias.as_ref() {
                linear.bias = Param::new(Some(bias.as_dtype(dtype.as_i32())));
            }
        };
        let cast_norm = |norm: &mut nn::LayerNorm| {
            for slot in [&mut norm.weight, &mut norm.bias] {
                if let Some(value) = slot.as_ref() {
                    *slot = Param::new(Some(value.as_dtype(dtype.as_i32())));
                }
            }
        };
        cast(&mut self.patch_embed);
        self.pos_embed = Param::new(self.pos_embed.as_ref().as_dtype(dtype.as_i32()));
        for block in &mut self.blocks {
            cast_norm(&mut block.norm1);
            cast_norm(&mut block.norm2);
            for linear in [
                &mut block.qkv,
                &mut block.proj,
                &mut block.fc1,
                &mut block.fc2,
            ] {
                cast(linear);
            }
        }
        cast_norm(&mut self.merger.norm);
        cast(&mut self.merger.fc1);
        cast(&mut self.merger.fc2);
    }

    /// Load the tower of a checkpoint directory: `config.json`'s
    /// `vision_config` and the `model.visual.*` tensors.
    pub fn load(dir: &Path) -> Result<Self, Exception> {
        let config = Qwen3_5MultimodalConfig::from_model_dir(dir)?;
        let mut model = Self::new(config.vision)?;
        let weights = load_visual_tensors(dir)?;
        model.load_weights(weights)?;
        Ok(model)
    }

    /// Install weights keyed relative to the tower (`patch_embed.proj.weight`,
    /// `blocks.0.attn.qkv.weight`, ...). Every tensor must be present with
    /// its expected shape and every given tensor must be consumed.
    pub fn load_weights(&mut self, mut weights: HashMap<String, Array>) -> Result<(), Exception> {
        if let Some(key) = weights
            .keys()
            .find(|k| k.ends_with(".scales") || k.ends_with(".weight_scale_inv"))
        {
            return Err(Exception::custom(format!(
                "quantized vision tensors ({key}) are not supported; the vision tower loads \
                 from a bf16/f16/f32 checkpoint"
            )));
        }
        let c = self.config.clone();
        let h = c.hidden_size as i32;
        let mut take = |key: &str, shape: &[i32]| -> Result<Array, Exception> {
            let array = weights
                .remove(key)
                .ok_or_else(|| Exception::custom(format!("vision tower: missing {key}")))?;
            if !matches!(
                array.dtype(),
                Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16
            ) {
                return Err(Exception::custom(format!(
                    "vision tower: {key} is {:?}; only float weights are supported",
                    array.dtype()
                )));
            }
            if array.shape() != shape {
                return Err(Exception::custom(format!(
                    "vision tower: {key} has shape {:?}, expected {shape:?}",
                    array.shape()
                )));
            }
            Ok(array)
        };
        let set = |linear: &mut nn::Linear, weight: Array, bias: Array| {
            linear.weight = Param::new(weight);
            linear.bias = Param::new(Some(bias));
        };
        let set_norm = |norm: &mut nn::LayerNorm, weight: Array, bias: Array| {
            norm.weight = Param::new(Some(weight));
            norm.bias = Param::new(Some(bias));
        };

        // Conv3d `[hidden, C, T, p, p]`, flattened in the processor's row
        // layout (C, T, p, p).
        let (ch, t, p) = (
            c.in_channels as i32,
            c.temporal_patch_size as i32,
            c.patch_size as i32,
        );
        let conv = take("patch_embed.proj.weight", &[h, ch, t, p, p])?;
        let bias = take("patch_embed.proj.bias", &[h])?;
        set(
            &mut self.patch_embed,
            conv.reshape(&[h, c.patch_dim() as i32]),
            bias,
        );
        self.pos_embed = Param::new(take(
            "pos_embed.weight",
            &[c.num_position_embeddings as i32, h],
        )?);

        let inter = c.intermediate_size as i32;
        for (i, block) in self.blocks.iter_mut().enumerate() {
            let k = |name: &str| format!("blocks.{i}.{name}");
            let (w, b) = (
                take(&k("norm1.weight"), &[h])?,
                take(&k("norm1.bias"), &[h])?,
            );
            set_norm(&mut block.norm1, w, b);
            let (w, b) = (
                take(&k("norm2.weight"), &[h])?,
                take(&k("norm2.bias"), &[h])?,
            );
            set_norm(&mut block.norm2, w, b);
            let (w, b) = (
                take(&k("attn.qkv.weight"), &[3 * h, h])?,
                take(&k("attn.qkv.bias"), &[3 * h])?,
            );
            set(&mut block.qkv, w, b);
            let (w, b) = (
                take(&k("attn.proj.weight"), &[h, h])?,
                take(&k("attn.proj.bias"), &[h])?,
            );
            set(&mut block.proj, w, b);
            let (w, b) = (
                take(&k("mlp.linear_fc1.weight"), &[inter, h])?,
                take(&k("mlp.linear_fc1.bias"), &[inter])?,
            );
            set(&mut block.fc1, w, b);
            let (w, b) = (
                take(&k("mlp.linear_fc2.weight"), &[h, inter])?,
                take(&k("mlp.linear_fc2.bias"), &[h])?,
            );
            set(&mut block.fc2, w, b);
        }

        let merged = h * (c.spatial_merge_size * c.spatial_merge_size) as i32;
        let (w, b) = (
            take("merger.norm.weight", &[h])?,
            take("merger.norm.bias", &[h])?,
        );
        set_norm(&mut self.merger.norm, w, b);
        let (w, b) = (
            take("merger.linear_fc1.weight", &[merged, merged])?,
            take("merger.linear_fc1.bias", &[merged])?,
        );
        set(&mut self.merger.fc1, w, b);
        let (w, b) = (
            take(
                "merger.linear_fc2.weight",
                &[c.out_hidden_size as i32, merged],
            )?,
            take("merger.linear_fc2.bias", &[c.out_hidden_size as i32])?,
        );
        set(&mut self.merger.fc2, w, b);

        if !weights.is_empty() {
            let mut extra: Vec<_> = weights.into_keys().collect();
            extra.sort();
            return Err(Exception::custom(format!(
                "vision tower: {} checkpoint tensors are not consumed (first: {})",
                extra.len(),
                extra[0]
            )));
        }
        Ok(())
    }

    /// Run the tower over `pixel_values` (`[patches, C·T·p²]`, the rows of
    /// every media item concatenated) whose items have the given `(t, h, w)`
    /// patch grids. Returns `[patches / merge², out_hidden_size]` features in
    /// the tower's dtype, one row per media token, in prompt order.
    pub fn forward(&self, pixel_values: &Array, grids: &[[usize; 3]]) -> Result<Array, Exception> {
        let c = &self.config;
        let total: usize = grids.iter().map(|g| g.iter().product::<usize>()).sum();
        if pixel_values.ndim() != 2
            || pixel_values.dim(0) as usize != total
            || pixel_values.dim(1) as usize != c.patch_dim()
        {
            return Err(Exception::custom(format!(
                "vision tower: pixel_values {:?} do not match grids {grids:?} (expected [{total}, \
                 {}])",
                pixel_values.shape(),
                c.patch_dim()
            )));
        }
        let m = c.spatial_merge_size;
        if grids.iter().any(|&[_, h, w]| h % m != 0 || w % m != 0) {
            return Err(Exception::custom(format!(
                "vision tower: grids {grids:?} do not tile into {m}x{m} merge blocks"
            )));
        }
        let dtype = self.dtype();
        let n = total as i32;
        let h = c.hidden_size as i32;
        let hd = c.head_dim() as i32;
        let heads = c.num_heads as i32;

        let mut x = self
            .patch_embed
            .forward(&pixel_values.as_dtype(dtype.as_i32()));

        // Learned positions, bilinearly resampled to each grid, in f32.
        let (indices, weights) = interpolation_taps(grids, c.grid_side(), m);
        let taps = self
            .pos_embed
            .as_ref()
            .take_axis(&Array::from_i32_slice(&indices), 0)
            .reshape(&[n, 4, h])
            .as_dtype(Dtype::Float32.as_i32());
        let weights = Array::from_f32_slice(&weights, &[n, 4, 1]);
        let positions = taps.multiply(&weights).sum_axis(1, false);
        x = x.add(&positions.as_dtype(dtype.as_i32()));

        // 2-D rotary tables, [N, 1, head_dim] f32.
        let (cos, sin) = rotary_tables(grids, m, &self.inv_freq);
        let cos = Array::from_f32_slice(&cos, &[n, 1, hd]);
        let sin = Array::from_f32_slice(&sin, &[n, 1, hd]);

        // Attention never crosses a frame.
        let segments: Vec<i32> = grids
            .iter()
            .flat_map(|&[t, gh, gw]| std::iter::repeat_n((gh * gw) as i32, t))
            .collect();
        let scale = (hd as f32).powf(-0.5);

        for block in &self.blocks {
            let normed = block.norm1.forward(&x);
            let qkv = block.qkv.forward(&normed).reshape(&[n, 3, heads, hd]);
            let q = ops::slice_axis(&qkv, 1, 0, 1).squeeze_axes(&[1]);
            let k = ops::slice_axis(&qkv, 1, 1, 2).squeeze_axes(&[1]);
            let v = ops::slice_axis(&qkv, 1, 2, 3).squeeze_axes(&[1]);
            let q = rotate(&q, &cos, &sin)
                .transpose_axes(&[1, 0, 2])
                .expand_dims(0);
            let k = rotate(&k, &cos, &sin)
                .transpose_axes(&[1, 0, 2])
                .expand_dims(0);
            let v = v.transpose_axes(&[1, 0, 2]).expand_dims(0);
            let mut outputs = Vec::with_capacity(segments.len());
            let mut start = 0;
            for &len in &segments {
                let slice = |a: &Array| ops::slice_axis(a, 2, start, start + len);
                outputs.push(slice(&q).sdpa_with_mask(&slice(&k), &slice(&v), scale, None));
                start += len;
            }
            let attn = if outputs.len() == 1 {
                outputs.pop().expect("one segment")
            } else {
                ops::concatenate_axis(&outputs.iter().collect::<Vec<_>>(), 2)
            };
            let attn = attn
                .squeeze_axes(&[0])
                .transpose_axes(&[1, 0, 2])
                .reshape(&[n, h]);
            x = x.add(&block.proj.forward(&attn));

            let normed = block.norm2.forward(&x);
            let mlp = block.fc2.forward(&(self.act)(&block.fc1.forward(&normed)));
            x = x.add(&mlp);
        }

        let merged = (c.hidden_size * m * m) as i32;
        let x = self
            .merger
            .norm
            .forward(&x)
            .reshape(&[n / (m * m) as i32, merged]);
        let x = nn::gelu_erf(&self.merger.fc1.forward(&x));
        Ok(self.merger.fc2.forward(&x))
    }

    /// Features for each media item: images (`t = 1`) and videos alike.
    pub fn encode(&self, media: &[ProcessedMedia]) -> Result<Option<Array>, Exception> {
        if media.is_empty() {
            return Ok(None);
        }
        let grids: Vec<[usize; 3]> = media.iter().map(|m| m.grid_thw).collect();
        let rows: usize = media.iter().map(ProcessedMedia::num_patches).sum();
        let mut values = Vec::with_capacity(rows * self.config.patch_dim());
        for item in media {
            values.extend_from_slice(&item.pixel_values);
        }
        let pixels = Array::from_f32_slice(&values, &[rows as i32, self.config.patch_dim() as i32]);
        self.forward(&pixels, &grids).map(Some)
    }
}

/// `x * cos + rotate_half(x) * sin` in f32, back in `x`'s dtype.
fn rotate(x: &Array, cos: &Array, sin: &Array) -> Array {
    let dtype = x.dtype();
    let x = x.as_dtype(Dtype::Float32.as_i32());
    let d = x.dim(-1);
    let x1 = ops::slice_axis(&x, -1, 0, d / 2);
    let x2 = ops::slice_axis(&x, -1, d / 2, d);
    let rotated = ops::concatenate_axis(&[&ops::negative(&x2), &x1], -1);
    x.multiply(cos)
        .add(&rotated.multiply(sin))
        .as_dtype(dtype.as_i32())
}

/// `(row, col)` of each patch, in the processor's merge-block order, the
/// whole frame repeated `t` times.
fn patch_coordinates(grids: &[[usize; 3]], merge: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for &[t, h, w] in grids {
        let mut frame = Vec::with_capacity(h * w);
        for block_row in 0..h / merge {
            for block_col in 0..w / merge {
                for in_row in 0..merge {
                    for in_col in 0..merge {
                        frame.push((block_row * merge + in_row, block_col * merge + in_col));
                    }
                }
            }
        }
        for _ in 0..t {
            out.extend_from_slice(&frame);
        }
    }
    out
}

/// `get_vision_interpolation_indices_and_weights(mode="bilinear",
/// align_corners=True)`: four table rows and their weights per patch.
fn interpolation_taps(grids: &[[usize; 3]], side: usize, merge: usize) -> (Vec<i32>, Vec<f32>) {
    // One axis: `src = index · (side − 1) / max(size − 1, 1)` in f32, taps at
    // floor and floor + 1 clamped to the table, linear-hat weights.
    let axis = |index: usize, size: usize| -> ([usize; 2], [f32; 2]) {
        let src = (index as f32 * (side - 1) as f32) / (size.max(2) - 1) as f32;
        let floor = src.floor();
        let mut taps = [0usize; 2];
        let mut weights = [0f32; 2];
        for (offset, (tap, weight)) in taps.iter_mut().zip(&mut weights).enumerate() {
            let raw = floor + offset as f32;
            *tap = (raw.max(0.0) as usize).min(side - 1);
            *weight = (1.0 - (src - floor - offset as f32).abs()).max(0.0);
        }
        (taps, weights)
    };
    let mut indices = Vec::new();
    let mut weights = Vec::new();
    let mut grid_of_patch = Vec::new();
    for &[t, h, w] in grids {
        grid_of_patch.extend(std::iter::repeat_n((h, w), t * h * w));
    }
    for ((row, col), (h, w)) in patch_coordinates(grids, merge)
        .into_iter()
        .zip(grid_of_patch)
    {
        let (row_taps, row_weights) = axis(row, h);
        let (col_taps, col_weights) = axis(col, w);
        for i in 0..2 {
            for j in 0..2 {
                indices.push((row_taps[i] * side + col_taps[j]) as i32);
                weights.push(row_weights[i] * col_weights[j]);
            }
        }
    }
    (indices, weights)
}

/// The 2-D rotary cos/sin, `[patches, head_dim]` each: per patch the row
/// frequencies then the column frequencies, that pair twice.
fn rotary_tables(grids: &[[usize; 3]], merge: usize, inv_freq: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let coords = patch_coordinates(grids, merge);
    let quarter = inv_freq.len();
    let mut cos = Vec::with_capacity(coords.len() * quarter * 4);
    let mut sin = Vec::with_capacity(coords.len() * quarter * 4);
    for (row, col) in coords {
        let angles: Vec<f32> = inv_freq
            .iter()
            .map(|&f| row as f32 * f)
            .chain(inv_freq.iter().map(|&f| col as f32 * f))
            .collect();
        for _ in 0..2 {
            cos.extend(angles.iter().map(|&a| (a as f64).cos() as f32));
            sin.extend(angles.iter().map(|&a| (a as f64).sin() as f32));
        }
    }
    (cos, sin)
}

/// Read the vision tower's tensors from a checkpoint directory, keyed
/// relative to the tower. Only the shards that hold them are opened, and
/// tensors load lazily, so the text weights are never read.
pub fn load_visual_tensors(dir: &Path) -> Result<HashMap<String, Array>, Exception> {
    let index = dir.join("model.safetensors.index.json");
    let shards: Vec<PathBuf> = if index.is_file() {
        let text = std::fs::read_to_string(&index)
            .map_err(|e| Exception::custom(format!("{}: {e}", index.display())))?;
        let value: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| Exception::custom(format!("{}: {e}", index.display())))?;
        let map = value
            .get("weight_map")
            .and_then(|v| v.as_object())
            .ok_or_else(|| Exception::custom("model.safetensors.index.json has no weight_map"))?;
        let mut files: Vec<String> = map
            .iter()
            .filter(|(key, _)| PREFIXES.iter().any(|p| key.starts_with(p)))
            .filter_map(|(_, file)| file.as_str().map(str::to_string))
            .collect();
        files.sort();
        files.dedup();
        files.into_iter().map(|f| dir.join(f)).collect()
    } else {
        vec![dir.join("model.safetensors")]
    };
    let mut tensors = HashMap::new();
    for shard in shards {
        let loaded = crate::loader::load_safetensors_file(&shard)
            .map_err(|e| Exception::custom(format!("{}: {e}", shard.display())))?;
        for (key, value) in loaded {
            if let Some(rest) = PREFIXES.iter().find_map(|p| key.strip_prefix(p)) {
                tensors.insert(rest.to_string(), value);
            }
        }
    }
    if tensors.is_empty() {
        return Err(Exception::custom(format!(
            "{} has no vision tower tensors (model.visual.*)",
            dir.display()
        )));
    }
    Ok(tensors)
}

// ---------------------------------------------------------------------------
// The bundle the surfaces use
// ---------------------------------------------------------------------------

/// Everything a Qwen3.5-family checkpoint needs to read images and videos:
/// its processors, its vision tower and its media token ids.
#[derive(Debug)]
pub struct Qwen3_5Vision {
    /// The checkpoint's image and video processors.
    pub processor: QwenVlProcessor,
    /// The vision tower.
    pub tower: Qwen3_5VisionModel,
    /// Media token ids and the tower's config.
    pub config: Qwen3_5MultimodalConfig,
}

/// A prompt's media, encoded: vision features and 3-D positions, ready for
/// either text engine.
#[derive(Debug, Clone)]
pub struct EncodedMedia {
    /// `[image tokens, hidden]`, one row per `<|image_pad|>` in the prompt.
    pub image_features: Option<Array>,
    /// `[video tokens, hidden]`, one row per `<|video_pad|>` in the prompt.
    pub video_features: Option<Array>,
    /// The prompt's positions.
    pub positions: MropePositions,
}

impl EncodedMedia {
    /// The prompt's input embeddings: `text_embeddings` (`[1, T, hidden]`, the
    /// token embeddings of `input_ids`) with the media tokens' rows replaced.
    pub fn merge(
        &self,
        text_embeddings: &Array,
        input_ids: &[u32],
        config: &Qwen3_5MultimodalConfig,
    ) -> Result<Array, Exception> {
        merge_media_features(
            text_embeddings,
            input_ids,
            self.image_features.as_ref(),
            self.video_features.as_ref(),
            config.image_token_id,
            config.video_token_id,
        )
    }
}

impl Qwen3_5Vision {
    /// Load a checkpoint's processors and vision tower.
    pub fn load(dir: &Path) -> Result<Self, Exception> {
        let config = Qwen3_5MultimodalConfig::from_model_dir(dir)?;
        let processor =
            QwenVlProcessor::from_model_dir(dir).map_err(|e| Exception::custom(e.to_string()))?;
        if processor.image.patch_size != config.vision.patch_size
            || processor.image.merge_size != config.vision.spatial_merge_size
            || processor.image.temporal_patch_size != config.vision.temporal_patch_size
        {
            return Err(Exception::custom(format!(
                "the image processor (patch {}, merge {}, temporal {}) does not match the vision \
                 tower (patch {}, merge {}, temporal {})",
                processor.image.patch_size,
                processor.image.merge_size,
                processor.image.temporal_patch_size,
                config.vision.patch_size,
                config.vision.spatial_merge_size,
                config.vision.temporal_patch_size
            )));
        }
        let mut tower = Qwen3_5VisionModel::new(config.vision.clone())?;
        tower.load_weights(load_visual_tensors(dir)?)?;
        Ok(Self {
            processor,
            tower,
            config,
        })
    }

    /// Replace each `<|image_pad|>` / `<|video_pad|>` in prompt `text` with
    /// its media's token run (timestamps included for videos).
    pub fn expand_placeholders(
        &self,
        text: &str,
        images: &[ProcessedMedia],
        videos: &[ProcessedMedia],
    ) -> Result<String, Exception> {
        pmetal_data::qwen_vl_processing::expand_placeholders(
            text,
            images,
            videos,
            self.config.vision.spatial_merge_size,
        )
        .map_err(|e| Exception::custom(e.to_string()))
    }

    /// Run the tower over a prompt's media and lay out its positions.
    /// `input_ids` is the expanded prompt. The features are evaluated, so the
    /// tower can be dropped before the text model loads.
    pub fn encode(
        &self,
        input_ids: &[u32],
        images: &[ProcessedMedia],
        videos: &[ProcessedMedia],
    ) -> Result<EncodedMedia, Exception> {
        let grids = |media: &[ProcessedMedia]| media.iter().map(|m| m.grid_thw).collect::<Vec<_>>();
        let positions = rope_index(
            input_ids,
            &grids(images),
            &grids(videos),
            self.config.vision.spatial_merge_size,
            self.config.image_token_id,
            self.config.video_token_id,
        )?;
        let image_features = self.tower.encode(images)?;
        let video_features = self.tower.encode(videos)?;
        for features in image_features.iter().chain(&video_features) {
            features
                .try_eval()
                .map_err(|e| Exception::custom(format!("vision tower: {e}")))?;
        }
        pmetal_bridge::check_last_error()
            .map_err(|e| Exception::custom(format!("vision tower: {e}")))?;
        let encoded = EncodedMedia {
            image_features,
            video_features,
            positions,
        };
        // The counts are checked here, where the error can be reported,
        // rather than inside an engine's prefill.
        let count = |id: u32| input_ids.iter().filter(|&&t| t == id).count();
        let rows = |f: &Option<Array>| f.as_ref().map_or(0, |f| f.dim(0) as usize);
        if count(self.config.image_token_id) != rows(&encoded.image_features)
            || count(self.config.video_token_id) != rows(&encoded.video_features)
        {
            return Err(Exception::custom(
                "the prompt's media tokens do not match the media's features",
            ));
        }
        Ok(encoded)
    }
}

// ---------------------------------------------------------------------------
// Prompt: positions and embeddings
// ---------------------------------------------------------------------------

/// A prompt's 3-D positions (`get_rope_index`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MropePositions {
    /// `[3, T]` row-major: temporal, row, column.
    pub positions: Vec<i32>,
    /// Prompt length `T`.
    pub len: usize,
    /// One past the largest position: where decoding continues.
    pub next_position: i32,
}

impl MropePositions {
    /// The `[3, T]` int32 array the engines take.
    pub fn array(&self) -> Array {
        Array::from_i32_slice_shaped(&self.positions, &[3, self.len as i32])
    }
}

/// `Qwen3_5Model.get_rope_index` for one unpadded prompt.
///
/// Runs of text tokens advance all three axes together. A run of image tokens
/// takes the next image grid `(t, h, w)` and lays its merged `h/m × w/m` tokens
/// out at `start + (t, row, col)`, then the next run starts `max(h, w)/m`
/// later. A video is split into its `t` frames first, each a `t = 1` grid,
/// since its frames are separated by timestamp text.
pub fn rope_index(
    input_ids: &[u32],
    image_grids: &[[usize; 3]],
    video_grids: &[[usize; 3]],
    spatial_merge_size: usize,
    image_token_id: u32,
    video_token_id: u32,
) -> Result<MropePositions, Exception> {
    let m = spatial_merge_size;
    let modality = |id: u32| -> u8 {
        if id == image_token_id {
            1
        } else if id == video_token_id {
            2
        } else {
            0
        }
    };
    let mut images = image_grids.iter();
    let frame_grids: Vec<[usize; 3]> = video_grids
        .iter()
        .flat_map(|&[t, h, w]| std::iter::repeat_n([1, h, w], t))
        .collect();
    let mut frames = frame_grids.iter();

    let len = input_ids.len();
    let mut axes: [Vec<i32>; 3] = [
        Vec::with_capacity(len),
        Vec::with_capacity(len),
        Vec::with_capacity(len),
    ];
    let mut current: i64 = 0;
    let mut i = 0;
    while i < len {
        let kind = modality(input_ids[i]);
        let run = input_ids[i..]
            .iter()
            .take_while(|&&id| modality(id) == kind)
            .count();
        if kind == 0 {
            for offset in 0..run as i64 {
                for axis in &mut axes {
                    axis.push((current + offset) as i32);
                }
            }
            current += run as i64;
        } else {
            let (name, grid) = if kind == 1 {
                ("image", images.next())
            } else {
                ("video frame", frames.next())
            };
            let &[t, h, w] = grid.ok_or_else(|| {
                Exception::custom(format!(
                    "the prompt has more {name} token runs than {name}s were given"
                ))
            })?;
            let (gh, gw) = (h / m, w / m);
            if run != t * gh * gw {
                return Err(Exception::custom(format!(
                    "a run of {run} {name} tokens does not match its {t}x{h}x{w} grid ({} tokens)",
                    t * gh * gw
                )));
            }
            for ti in 0..t {
                for row in 0..gh {
                    for col in 0..gw {
                        axes[0].push((ti as i64 + current) as i32);
                        axes[1].push((row as i64 + current) as i32);
                        axes[2].push((col as i64 + current) as i32);
                    }
                }
            }
            current += (h.max(w) / m) as i64;
        }
        i += run;
    }
    if images.next().is_some() || frames.next().is_some() {
        return Err(Exception::custom(
            "more images or video frames were given than the prompt has token runs for",
        ));
    }
    let next_position = axes
        .iter()
        .flat_map(|axis| axis.iter().copied())
        .max()
        .map_or(0, |max| max + 1);
    let mut positions = Vec::with_capacity(3 * len);
    for axis in axes {
        positions.extend(axis);
    }
    Ok(MropePositions {
        positions,
        len,
        next_position,
    })
}

/// Replace the embeddings of a prompt's image and video tokens with vision
/// features: `get_placeholder_mask` and `masked_scatter`.
///
/// `text_embeddings` is `[1, T, hidden]`. The `k`-th image token takes row `k`
/// of `image_features`, the `k`-th video token row `k` of `video_features`;
/// the counts must match exactly. Features are cast to the embeddings' dtype.
pub fn merge_media_features(
    text_embeddings: &Array,
    input_ids: &[u32],
    image_features: Option<&Array>,
    video_features: Option<&Array>,
    image_token_id: u32,
    video_token_id: u32,
) -> Result<Array, Exception> {
    let rows = |features: Option<&Array>| features.map_or(0, |f| f.dim(0) as usize);
    let (image_rows, video_rows) = (rows(image_features), rows(video_features));
    let image_tokens = input_ids.iter().filter(|&&id| id == image_token_id).count();
    let video_tokens = input_ids.iter().filter(|&&id| id == video_token_id).count();
    if image_tokens != image_rows {
        return Err(Exception::custom(format!(
            "Image features and image tokens do not match, tokens: {image_tokens}, features: \
             {image_rows}"
        )));
    }
    if video_tokens != video_rows {
        return Err(Exception::custom(format!(
            "Video features and video tokens do not match, tokens: {video_tokens}, features: \
             {video_rows}"
        )));
    }
    if image_rows + video_rows == 0 {
        return Ok(text_embeddings.clone());
    }
    let dtype = text_embeddings.dtype().as_i32();
    let features: Vec<Array> = [image_features, video_features]
        .into_iter()
        .flatten()
        .map(|f| f.as_dtype(dtype))
        .collect();
    let features = if features.len() == 1 {
        features[0].clone()
    } else {
        ops::concatenate_axis(&features.iter().collect::<Vec<_>>(), 0)
    };
    let (mut next_image, mut next_video) = (0i32, image_rows as i32);
    let mut index = Vec::with_capacity(input_ids.len());
    let mut is_media = Vec::with_capacity(input_ids.len());
    for &id in input_ids {
        if id == image_token_id {
            index.push(next_image);
            next_image += 1;
            is_media.push(1);
        } else if id == video_token_id {
            index.push(next_video);
            next_video += 1;
            is_media.push(1);
        } else {
            index.push(0);
            is_media.push(0);
        }
    }
    let t = input_ids.len() as i32;
    let gathered = features
        .take_axis(&Array::from_i32_slice(&index), 0)
        .expand_dims(0);
    let mask = ops::greater(
        &Array::from_i32_slice_shaped(&is_media, &[1, t, 1]),
        &Array::from_i32(0),
    );
    Ok(ops::where_fn(&mask, &gathered, text_embeddings))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_index_lays_out_an_image_and_continues_after_it() {
        // 2 text, one 1x4x6 image (merge 2 -> 2x3 tokens), 3 text.
        let ids = [5, 6, 9, 9, 9, 9, 9, 9, 7, 8, 4];
        let pos = rope_index(&ids, &[[1, 4, 6]], &[], 2, 9, 10).unwrap();
        let axis = |a: usize| &pos.positions[a * ids.len()..(a + 1) * ids.len()];
        assert_eq!(axis(0), &[0, 1, 2, 2, 2, 2, 2, 2, 5, 6, 7]);
        assert_eq!(axis(1), &[0, 1, 2, 2, 2, 3, 3, 3, 5, 6, 7]);
        assert_eq!(axis(2), &[0, 1, 2, 3, 4, 2, 3, 4, 5, 6, 7]);
        assert_eq!(pos.next_position, 8);
    }

    #[test]
    fn rope_index_refuses_a_grid_that_does_not_fit() {
        assert!(rope_index(&[9, 9, 9], &[[1, 4, 4]], &[], 2, 9, 10).is_err());
        assert!(rope_index(&[1, 2], &[[1, 2, 2]], &[], 2, 9, 10).is_err());
    }

    #[test]
    fn coordinates_follow_merge_blocks() {
        let coords = patch_coordinates(&[[1, 2, 4]], 2);
        assert_eq!(
            coords,
            vec![
                (0, 0),
                (0, 1),
                (1, 0),
                (1, 1),
                (0, 2),
                (0, 3),
                (1, 2),
                (1, 3)
            ]
        );
    }

    #[test]
    fn interpolation_weights_sum_to_one() {
        let (_, weights) = interpolation_taps(&[[1, 6, 10]], 48, 2);
        for patch in weights.chunks(4) {
            let sum: f32 = patch.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "{sum}");
        }
    }
}
