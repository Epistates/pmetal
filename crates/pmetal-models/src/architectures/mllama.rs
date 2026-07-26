//! Llama 3.2 Vision (Mllama) architecture.
//!
//! Mllama pairs a Llama 3.2 text decoder with a *tiled* vision encoder. What
//! makes it unlike every other vision tower in this crate is that the image
//! geometry is a model **input**, not just a preprocessing detail: the processor
//! (see `pmetal_data::image_processing::MllamaImageProcessor`) picks an
//! arrangement of up to `max_num_tiles` square tiles, reports it back as
//! `aspect_ratio_ids`, and the tower looks up *precomputed per-arrangement*
//! embeddings with that id. Feeding the wrong id silently selects a different
//! embedding, so the processor and the tower have to agree exactly.
//!
//! The tower runs two stacks over the tiles:
//!
//! * `transformer` — `num_hidden_layers` un-gated layers over all tiles'
//!   patches concatenated, with tile padding masked.
//! * `global_transformer` — `num_global_layers` **tanh-gated** layers,
//!   preceded by a second gated per-arrangement embedding.
//!
//! and its output is not the last hidden state alone: the final state is
//! concatenated with the outputs of the layers named by
//! `intermediate_layers_indices`, which is why `vision_output_dim` is
//! `hidden_size · (1 + len(indices))` = 7680, not 1280.
//!
//! # Divergence from the reference
//!
//! `_prepare_aspect_ratio_attention_mask` zeroes padding rows with
//! `mask[:, :, -pad_patches:] = 0`. When `pad_patches == 0` that slice is
//! `[:, :, 0:]` — the *whole* tensor — so the reference masks every position
//! and softmax sees an all-`-inf` row. It cannot trigger on a released
//! checkpoint (`(448/14)² + 1 = 1025` pads to 1032), and it is plainly not the
//! intent, so [`aspect_ratio_attention_mask`] builds the mask additively and
//! leaves nothing masked when there is no padding.
use pmetal_bridge::compat::{
    Array, Dtype, Exception, Module, ModuleParameters, ModuleParametersExt, Param, nn, ops, random,
};
use pmetal_bridge::impl_module_params;

use std::collections::HashMap;

use pmetal_mlx::kernels::{
    AttentionMaskType, FusedAttentionConfig, differentiable_attention, fused_sdpa,
    get_training_context, rope::apply_rope,
};
use pmetal_mlx::kv_cache::KVCache;
use serde::{Deserialize, Serialize};

use crate::architectures::llama::{LlamaAttention, LlamaConfig, LlamaMLP, RopeScalingValue};
use crate::architectures::utils::{
    LoadReport, load_layer_norm, load_linear, load_optional_param, load_param,
};
use crate::traits::ModelConfig;

/// Mllama vision-tower configuration (`MllamaVisionConfig`).
///
/// Field names follow the reference config, including `attention_heads` (which
/// `transformers` exposes under the `num_attention_heads` alias) and `norm_eps`.
/// The defaults are the reference *class* defaults; released checkpoints
/// override `image_size` to 560.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MllamaVisionConfig {
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_hidden_layers: i32,
    /// Second, tanh-gated encoder stack run after the per-tile one.
    pub num_global_layers: i32,
    #[serde(alias = "num_attention_heads")]
    pub attention_heads: i32,
    pub num_channels: i32,
    pub image_size: i32,
    pub patch_size: i32,
    pub hidden_act: String,
    pub norm_eps: f32,
    pub max_num_tiles: i32,
    /// Encoder layers whose outputs are concatenated onto the final hidden
    /// state. Indices are into the `transformer` stack's *post-layer* outputs,
    /// so `0` means "the output of layer 0".
    pub intermediate_layers_indices: Vec<i32>,
    /// `hidden_size · (1 + intermediate_layers_indices.len())`. Kept as a field
    /// because the reference config carries it and the text-side projector reads
    /// it; [`MllamaVisionConfig::validate`] checks the two agree.
    pub vision_output_dim: i32,
    /// Number of tile arrangements the checkpoint's embedding tables were built
    /// for. `None` means the default rule (every `(w, h)` with `w·h ≤
    /// max_num_tiles`), whose *count* [`Self::max_aspect_ratio_id`] derives
    /// arithmetically. The arrangement **order** — which is what an id means —
    /// lives in `pmetal_data::image_processing::supported_aspect_ratios`; this
    /// crate only ever needs the count, so the list is not duplicated here.
    pub supported_aspect_ratios: Option<Vec<(i32, i32)>>,
}

impl Default for MllamaVisionConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1280,
            intermediate_size: 5120,
            num_hidden_layers: 32,
            num_global_layers: 8,
            attention_heads: 16,
            num_channels: 3,
            image_size: 448,
            patch_size: 14,
            hidden_act: "gelu".to_string(),
            norm_eps: 1e-5,
            max_num_tiles: 4,
            intermediate_layers_indices: vec![3, 7, 15, 23, 30],
            vision_output_dim: 7680,
            supported_aspect_ratios: None,
        }
    }
}

impl MllamaVisionConfig {
    /// Patches per tile, **including** the prepended class token.
    pub fn num_patches(&self) -> i32 {
        (self.image_size / self.patch_size).pow(2) + 1
    }

    /// Highest valid `aspect_ratio_id`; the embedding tables have
    /// `max_aspect_ratio_id + 1` rows because `0` is reserved for "no image".
    pub fn max_aspect_ratio_id(&self) -> i32 {
        match &self.supported_aspect_ratios {
            Some(ratios) => ratios.len() as i32,
            // |{(w, h) : w·h ≤ max}| = Σ_w ⌊max / w⌋ — the same set
            // `supported_aspect_ratios` enumerates, counted instead of listed.
            None => (1..=self.max_num_tiles)
                .map(|w| self.max_num_tiles / w)
                .sum(),
        }
    }

    /// Concatenated output width: the final hidden state plus one copy per
    /// collected intermediate layer.
    pub fn output_dim(&self) -> i32 {
        self.hidden_size * (1 + self.intermediate_layers_indices.len() as i32)
    }

    fn validate(&self) -> Result<(), Exception> {
        if self.hidden_size % self.attention_heads != 0 {
            return Err(Exception::custom(format!(
                "mllama vision: hidden_size {} is not divisible by attention_heads {}",
                self.hidden_size, self.attention_heads
            )));
        }
        if self.image_size % self.patch_size != 0 {
            return Err(Exception::custom(format!(
                "mllama vision: image_size {} is not divisible by patch_size {}",
                self.image_size, self.patch_size
            )));
        }
        let n = self.num_hidden_layers;
        if let Some(&bad) = self
            .intermediate_layers_indices
            .iter()
            .find(|&&i| i < 0 || i >= n)
        {
            return Err(Exception::custom(format!(
                "mllama vision: intermediate_layers_indices contains {bad}, outside 0..{n}"
            )));
        }
        // A mismatch here means the checkpoint's projector expects a different
        // concatenation width than this config produces — a silent 6x shape
        // error at the text boundary if it goes unchecked.
        if self.vision_output_dim != self.output_dim() {
            return Err(Exception::custom(format!(
                "mllama vision: vision_output_dim {} != hidden_size {} x (1 + {} intermediate layers) = {}",
                self.vision_output_dim,
                self.hidden_size,
                self.intermediate_layers_indices.len(),
                self.output_dim()
            )));
        }
        Ok(())
    }

    /// Resolve `hidden_act`. Mllama's vision MLP is CLIP's, so the reference
    /// runs `ACT2FN[hidden_act]`; released checkpoints say `"gelu"`, which is
    /// the **exact** erf definition, not either fast approximation.
    fn activation(&self) -> Result<fn(&Array) -> Array, Exception> {
        match self.hidden_act.as_str() {
            "gelu" => Ok(nn::gelu_erf),
            "gelu_pytorch_tanh" | "gelu_new" => Ok(nn::gelu_tanh_approximate),
            "quick_gelu" => Ok(nn::gelu),
            "relu" => Ok(nn::relu),
            "silu" => Ok(nn::silu),
            other => Err(Exception::custom(format!(
                "mllama vision: unsupported hidden_act {other:?}"
            ))),
        }
    }
}

fn default_cross_attention_layers() -> Vec<i32> {
    vec![3, 8, 13, 18, 23, 28, 33, 38]
}

fn default_mllama_model_type() -> String {
    "mllama".to_string()
}

/// Mllama text-decoder configuration (`MllamaTextConfig`).
///
/// A Llama decoder plus the list of layers that carry cross-attention to the
/// vision features. Those layers are *extra*: `num_hidden_layers` counts them,
/// and they have no self-attention at all.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MllamaTextConfig {
    #[serde(flatten)]
    pub llama: LlamaConfig,
    #[serde(default = "default_cross_attention_layers")]
    pub cross_attention_layers: Vec<i32>,
}

impl Default for MllamaTextConfig {
    fn default() -> Self {
        Self {
            llama: LlamaConfig::default(),
            cross_attention_layers: default_cross_attention_layers(),
        }
    }
}

/// Mllama full model configuration.
///
/// Note that `text_config` is *nested* in the reference `config.json`, not
/// flattened at the top level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MllamaConfig {
    #[serde(default = "default_mllama_model_type")]
    pub model_type: String,
    /// Text model configuration.
    #[serde(default)]
    pub text_config: MllamaTextConfig,
    /// Vision model configuration.
    #[serde(default)]
    pub vision_config: MllamaVisionConfig,
    /// Id of the `<|image|>` placeholder token.
    #[serde(default = "default_image_token_index")]
    pub image_token_index: i32,
}

fn default_image_token_index() -> i32 {
    128_256
}

impl Default for MllamaConfig {
    fn default() -> Self {
        Self {
            model_type: default_mllama_model_type(),
            text_config: MllamaTextConfig::default(),
            vision_config: MllamaVisionConfig::default(),
            image_token_index: default_image_token_index(),
        }
    }
}

impl ModelConfig for MllamaConfig {
    fn model_type(&self) -> &str {
        &self.model_type
    }

    fn vocab_size(&self) -> i32 {
        self.text_config.llama.vocab_size
    }

    fn hidden_size(&self) -> i32 {
        self.text_config.llama.hidden_size
    }

    fn num_hidden_layers(&self) -> i32 {
        self.text_config.llama.num_hidden_layers
    }

    fn num_attention_heads(&self) -> i32 {
        self.text_config.llama.num_attention_heads
    }

    fn num_kv_heads(&self) -> i32 {
        self.text_config.llama.num_kv_heads()
    }

    fn head_dim(&self) -> i32 {
        self.text_config.llama.get_head_dim()
    }

    fn intermediate_size(&self) -> i32 {
        self.text_config.llama.intermediate_size
    }

    fn max_position_embeddings(&self) -> i32 {
        self.text_config.llama.max_position_embeddings
    }

    fn norm_eps(&self) -> f32 {
        self.text_config.llama.rms_norm_eps
    }

    fn rope_theta(&self) -> f32 {
        self.text_config.llama.rope_theta
    }

    fn tie_word_embeddings(&self) -> bool {
        self.text_config.llama.tie_word_embeddings
    }
}

/// `tanh(gate)` broadcast against `x`, or `x` unchanged when the layer is
/// un-gated. Mllama gates with a *scalar* parameter in five places, always this
/// way round (gate the branch, then add the residual).
fn apply_gate(gate: &Param<Option<Array>>, x: &Array) -> Array {
    match gate.value.as_ref() {
        Some(g) => ops::tanh(g).multiply(x),
        None => x.clone(),
    }
}

/// A zero-initialised scalar gate — the reference's `nn.Parameter(torch.zeros(1))`.
fn zero_gate() -> Param<Option<Array>> {
    Param::new(Some(ops::zeros(&[1], Dtype::Float32)))
}

/// Per-arrangement tile embedding (`MllamaPrecomputedAspectRatioEmbedding`).
///
/// One `max_num_tiles · hidden` vector per supported tile arrangement, looked up
/// by `aspect_ratio_id` and added to every patch of the corresponding tile.
#[derive(Debug)]
pub struct MllamaPrecomputedAspectRatioEmbedding {
    pub embedding: nn::Embedding,
    /// `None` when the module is built un-gated (never, for released
    /// checkpoints — both instances in the tower are gated).
    pub gate: Param<Option<Array>>,
    max_num_tiles: i32,
    hidden_size: i32,
}
impl_module_params!(MllamaPrecomputedAspectRatioEmbedding; embedding, gate);

impl MllamaPrecomputedAspectRatioEmbedding {
    pub fn new(config: &MllamaVisionConfig, is_gated: bool) -> Result<Self, Exception> {
        Ok(Self {
            embedding: nn::Embedding::new(
                config.max_aspect_ratio_id() + 1,
                config.max_num_tiles * config.hidden_size,
            )?,
            gate: if is_gated {
                zero_gate()
            } else {
                Param::new(None)
            },
            max_num_tiles: config.max_num_tiles,
            hidden_size: config.hidden_size,
        })
    }

    /// `hidden_state`: `[rows, max_num_tiles, patches, hidden]`;
    /// `aspect_ratio_ids`: `[rows, 1]`. Broadcasts over the patch axis.
    pub fn forward(&self, hidden_state: &Array, aspect_ratio_ids: &Array) -> Array {
        let embeddings = self.embedding.forward(aspect_ratio_ids).reshape(&[
            -1,
            self.max_num_tiles,
            1,
            self.hidden_size,
        ]);
        hidden_state.add(&apply_gate(&self.gate, &embeddings))
    }
}

/// Patch + per-arrangement position embedding (`MllamaPrecomputedPositionEmbedding`).
///
/// Holds a tile-agnostic `[patches, hidden]` table and a per-arrangement
/// `[max_num_tiles, patches, hidden]` one, mixed by a *single* gate: the
/// tile-agnostic table is scaled by `1 - tanh(gate)` and the per-arrangement one
/// by `tanh(gate)`, so a zero gate falls back to plain position embeddings.
#[derive(Debug)]
pub struct MllamaPrecomputedPositionEmbedding {
    pub gate: Param<Option<Array>>,
    /// `[num_patches, hidden]` — a raw parameter, not an `nn.Embedding`.
    pub embedding: Param<Array>,
    pub tile_embedding: nn::Embedding,
    max_num_tiles: i32,
    num_patches: i32,
    hidden_size: i32,
}
impl_module_params!(MllamaPrecomputedPositionEmbedding; gate, embedding, tile_embedding);

impl MllamaPrecomputedPositionEmbedding {
    pub fn new(config: &MllamaVisionConfig) -> Result<Self, Exception> {
        let num_patches = config.num_patches();
        let scale = (config.hidden_size as f32).powf(-0.5);
        Ok(Self {
            gate: zero_gate(),
            embedding: Param::new(
                random::normal(&[num_patches, config.hidden_size], Dtype::Float32)
                    .multiply(&Array::from_f32(scale)),
            ),
            tile_embedding: nn::Embedding::new(
                config.max_aspect_ratio_id() + 1,
                config.max_num_tiles * num_patches * config.hidden_size,
            )?,
            max_num_tiles: config.max_num_tiles,
            num_patches,
            hidden_size: config.hidden_size,
        })
    }

    /// `hidden_state`: `[rows, max_num_tiles, num_patches, hidden]`;
    /// `aspect_ratio_ids`: `[rows, 1]`.
    pub fn forward(&self, hidden_state: &Array, aspect_ratio_ids: &Array) -> Array {
        let gate_tanh = match self.gate.value.as_ref() {
            Some(g) => ops::tanh(g),
            None => Array::from_f32(0.0),
        };

        let ungated = Array::from_f32(1.0).subtract(&gate_tanh);
        let position = self.embedding.as_ref().multiply(&ungated).reshape(&[
            1,
            1,
            self.num_patches,
            self.hidden_size,
        ]);
        let hidden_state = hidden_state.add(&position);

        let rows = hidden_state.dim(0);
        let tile = self.tile_embedding.forward(aspect_ratio_ids).reshape(&[
            rows,
            self.max_num_tiles,
            self.num_patches,
            self.hidden_size,
        ]);
        hidden_state.add(&tile.multiply(&gate_tanh))
    }
}

/// Vision MLP (`MllamaVisionMLP`) — CLIP's: two biased linears around
/// `hidden_act`.
#[derive(Debug)]
pub struct MllamaVisionMLP {
    pub fc1: nn::Linear,
    pub fc2: nn::Linear,
    activation: fn(&Array) -> Array,
}
impl_module_params!(MllamaVisionMLP; fc1, fc2);

impl MllamaVisionMLP {
    pub fn new(config: &MllamaVisionConfig) -> Result<Self, Exception> {
        Ok(Self {
            fc1: nn::LinearBuilder::new(config.hidden_size, config.intermediate_size)
                .bias(true)
                .build()?,
            fc2: nn::LinearBuilder::new(config.intermediate_size, config.hidden_size)
                .bias(true)
                .build()?,
            activation: config.activation()?,
        })
    }

    pub fn forward(&self, x: &Array) -> Array {
        self.fc2.forward(&(self.activation)(&self.fc1.forward(x)))
    }
}

/// Vision self-attention (`MllamaVisionAttention`) — plain MHA, no GQA, no
/// bias, no RoPE, with an optional additive mask for tile padding.
#[derive(Debug)]
pub struct MllamaVisionAttention {
    pub q_proj: nn::Linear,
    pub k_proj: nn::Linear,
    pub v_proj: nn::Linear,
    pub o_proj: nn::Linear,
    num_heads: i32,
    head_dim: i32,
    scaling: f32,
}
impl_module_params!(MllamaVisionAttention; q_proj, k_proj, v_proj, o_proj);

impl MllamaVisionAttention {
    pub fn new(config: &MllamaVisionConfig) -> Result<Self, Exception> {
        let num_heads = config.attention_heads;
        let head_dim = config.hidden_size / num_heads;
        let dims = num_heads * head_dim;
        Ok(Self {
            q_proj: nn::LinearBuilder::new(config.hidden_size, dims)
                .bias(false)
                .build()?,
            k_proj: nn::LinearBuilder::new(config.hidden_size, dims)
                .bias(false)
                .build()?,
            v_proj: nn::LinearBuilder::new(config.hidden_size, dims)
                .bias(false)
                .build()?,
            o_proj: nn::LinearBuilder::new(dims, config.hidden_size)
                .bias(false)
                .build()?,
            num_heads,
            head_dim,
            scaling: (head_dim as f32).powf(-0.5),
        })
    }

    /// `x`: `[rows, seq, hidden]`; `mask`: optional additive `[rows, 1, seq, seq]`.
    pub fn forward(&self, x: &Array, mask: Option<&Array>) -> Array {
        let rows = x.dim(0);
        let seq = x.dim(1);
        let to_heads = |p: &nn::Linear| {
            p.forward(x)
                .reshape(&[rows, seq, self.num_heads, self.head_dim])
                .transpose_axes(&[0, 2, 1, 3])
        };
        let out = to_heads(&self.q_proj).sdpa_with_mask(
            &to_heads(&self.k_proj),
            &to_heads(&self.v_proj),
            self.scaling,
            mask,
        );
        self.o_proj
            .forward(&out.transpose_axes(&[0, 2, 1, 3]).reshape(&[
                rows,
                seq,
                self.num_heads * self.head_dim,
            ]))
    }
}

/// One vision encoder layer. The `global_transformer`'s layers are gated
/// (`gate_attn` / `gate_ffn`, both initialised to `π/4`); the per-tile
/// `transformer`'s are not.
#[derive(Debug)]
pub struct MllamaVisionEncoderLayer {
    pub self_attn: MllamaVisionAttention,
    pub mlp: MllamaVisionMLP,
    pub input_layernorm: nn::LayerNorm,
    pub post_attention_layernorm: nn::LayerNorm,
    pub gate_attn: Param<Option<Array>>,
    pub gate_ffn: Param<Option<Array>>,
}
impl_module_params!(
    MllamaVisionEncoderLayer;
    self_attn,
    mlp,
    input_layernorm,
    post_attention_layernorm,
    gate_attn,
    gate_ffn
);

impl MllamaVisionEncoderLayer {
    pub fn new(config: &MllamaVisionConfig, is_gated: bool) -> Result<Self, Exception> {
        // The reference seeds both gates at π/4 rather than 0; only relevant for
        // a from-scratch init, but matching it keeps an un-loaded model sane.
        let gate = || {
            if is_gated {
                Param::new(Some(ops::full(
                    &[1],
                    std::f32::consts::FRAC_PI_4,
                    Dtype::Float32,
                )))
            } else {
                Param::new(None)
            }
        };
        Ok(Self {
            self_attn: MllamaVisionAttention::new(config)?,
            mlp: MllamaVisionMLP::new(config)?,
            input_layernorm: nn::LayerNormBuilder::new(config.hidden_size)
                .eps(config.norm_eps)
                .build()?,
            post_attention_layernorm: nn::LayerNormBuilder::new(config.hidden_size)
                .eps(config.norm_eps)
                .build()?,
            gate_attn: gate(),
            gate_ffn: gate(),
        })
    }

    pub fn forward(&self, x: &Array, mask: Option<&Array>) -> Array {
        let attn = self
            .self_attn
            .forward(&self.input_layernorm.forward(x), mask);
        let h = x.add(&apply_gate(&self.gate_attn, &attn));

        let ff = self.mlp.forward(&self.post_attention_layernorm.forward(&h));
        h.add(&apply_gate(&self.gate_ffn, &ff))
    }
}

/// A stack of vision encoder layers (`MllamaVisionEncoder`).
#[derive(Debug)]
pub struct MllamaVisionEncoder {
    pub layers: Vec<MllamaVisionEncoderLayer>,
}
impl_module_params!(MllamaVisionEncoder; layers);

impl MllamaVisionEncoder {
    pub fn new(
        config: &MllamaVisionConfig,
        num_layers: i32,
        is_gated: bool,
    ) -> Result<Self, Exception> {
        Ok(Self {
            layers: (0..num_layers)
                .map(|_| MllamaVisionEncoderLayer::new(config, is_gated))
                .collect::<Result<Vec<_>, _>>()?,
        })
    }

    /// Runs the stack, additionally returning the output of each layer named in
    /// `collect`, **in the order `collect` gives them** — that order is the
    /// concatenation order of the tower's output, so it is load-bearing.
    pub fn forward(&self, x: &Array, mask: Option<&Array>, collect: &[i32]) -> (Array, Vec<Array>) {
        let mut hidden = x.clone();
        let mut by_index: HashMap<i32, Array> = HashMap::new();
        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, mask);
            let i = i as i32;
            if collect.contains(&i) {
                by_index.insert(i, hidden.clone());
            }
        }
        let collected = collect
            .iter()
            .filter_map(|i| by_index.get(i).cloned())
            .collect();
        (hidden, collected)
    }
}

/// Additive attention mask over `max_num_tiles · target_length` positions,
/// from a `[rows, max_num_tiles]` tile mask (`1` = real tile).
///
/// Two things about this are surprising enough to spell out, because both are
/// reference behaviour rather than oversights on our side:
///
/// * The mask is the **outer product** of the padding indicator with itself, so
///   a position is masked only when *both* the query and the key are padding. A
///   real query is left free to attend to padding keys.
/// * `num_patches` counts the real patches per tile; positions beyond it inside
///   each tile are the 8-multiple alignment padding and count as padding.
///
/// Returns `None` when nothing is masked, which lets the attention skip the mask
/// entirely.
pub fn aspect_ratio_attention_mask(
    tile_mask: &Array,
    num_patches: i32,
    target_length: i32,
) -> Option<Array> {
    let rows = tile_mask.dim(0);
    let max_num_tiles = tile_mask.dim(1);

    // keep[row, tile, pos] = tile is real AND pos is a real patch.
    let patch_keep = ops::less(
        &ops::arange(target_length, Dtype::Float32),
        &Array::from_f32(num_patches as f32),
    )
    .as_type::<f32>()
    .reshape(&[1, 1, target_length]);
    let keep = tile_mask
        .as_type::<f32>()
        .reshape(&[rows, max_num_tiles, 1])
        .multiply(&patch_keep);

    let padding =
        Array::from_f32(1.0)
            .subtract(&keep)
            .reshape(&[rows, max_num_tiles * target_length, 1]);
    let mask = padding
        .matmul(&padding.transpose_axes(&[0, 2, 1]))
        .multiply(&Array::from_f32(f32::MIN))
        .expand_dims(1);
    Some(mask)
}

/// The Mllama vision tower (`MllamaVisionModel`).
///
/// Consumes the six-dimensional output of
/// `pmetal_data::image_processing::MllamaImageProcessor` and returns
/// `[batch, images, tiles, num_patches, vision_output_dim]`.
#[derive(Debug)]
pub struct MllamaVisionModel {
    pub config: MllamaVisionConfig,

    pub patch_embedding: nn::Conv2d,
    /// `[hidden]` — prepended to every tile's patches as a CLS token.
    pub class_embedding: Param<Array>,
    pub gated_positional_embedding: MllamaPrecomputedPositionEmbedding,
    pub pre_tile_positional_embedding: MllamaPrecomputedAspectRatioEmbedding,
    pub post_tile_positional_embedding: MllamaPrecomputedAspectRatioEmbedding,
    pub layernorm_pre: nn::LayerNorm,
    pub layernorm_post: nn::LayerNorm,
    pub transformer: MllamaVisionEncoder,
    pub global_transformer: MllamaVisionEncoder,
}
impl_module_params!(
    MllamaVisionModel;
    patch_embedding,
    class_embedding,
    gated_positional_embedding,
    pre_tile_positional_embedding,
    post_tile_positional_embedding,
    layernorm_pre,
    layernorm_post,
    transformer,
    global_transformer
);

impl MllamaVisionModel {
    pub fn new(config: MllamaVisionConfig) -> Result<Self, Exception> {
        config.validate()?;
        let scale = (config.hidden_size as f32).powf(-0.5);
        Ok(Self {
            patch_embedding: nn::Conv2dBuilder::new(
                config.num_channels,
                config.hidden_size,
                config.patch_size,
            )
            .stride(config.patch_size)
            .bias(false)
            .build()?,
            class_embedding: Param::new(
                random::normal(&[config.hidden_size], Dtype::Float32)
                    .multiply(&Array::from_f32(scale)),
            ),
            gated_positional_embedding: MllamaPrecomputedPositionEmbedding::new(&config)?,
            pre_tile_positional_embedding: MllamaPrecomputedAspectRatioEmbedding::new(
                &config, true,
            )?,
            post_tile_positional_embedding: MllamaPrecomputedAspectRatioEmbedding::new(
                &config, true,
            )?,
            // The reference builds these two with `nn.LayerNorm(hidden_size)`,
            // i.e. torch's default eps, *not* `config.norm_eps` — which only
            // differ if a config overrides norm_eps away from 1e-5.
            layernorm_pre: nn::LayerNormBuilder::new(config.hidden_size)
                .eps(nn::LayerNorm::DEFAULT_EPS)
                .build()?,
            layernorm_post: nn::LayerNormBuilder::new(config.hidden_size)
                .eps(nn::LayerNorm::DEFAULT_EPS)
                .build()?,
            transformer: MllamaVisionEncoder::new(&config, config.num_hidden_layers, false)?,
            global_transformer: MllamaVisionEncoder::new(&config, config.num_global_layers, true)?,
            config,
        })
    }

    /// Patches per tile including the class token — the number of vision tokens
    /// each tile contributes, which the text side needs to size its
    /// cross-attention mask.
    pub fn num_patches(&self) -> i32 {
        self.config.num_patches()
    }

    /// Prepend the class token to `[rows, patches, hidden]`.
    fn apply_class_embedding(&self, hidden_state: &Array) -> Array {
        let rows = hidden_state.dim(0);
        let hidden = hidden_state.dim(2);
        let class = ops::broadcast_to(
            &self.class_embedding.as_ref().reshape(&[1, 1, hidden]),
            &[rows, 1, hidden],
        );
        ops::concatenate_axis(&[&class, hidden_state], 1)
    }

    /// * `pixel_values`: `[batch, images, tiles, channels, height, width]`
    /// * `aspect_ratio_ids`: `[batch, images]`
    /// * `aspect_ratio_mask`: `[batch, images, tiles]`
    ///
    /// Returns `[batch, images, tiles, num_patches, vision_output_dim]`.
    pub fn forward(
        &mut self,
        pixel_values: &Array,
        aspect_ratio_ids: &Array,
        aspect_ratio_mask: &Array,
    ) -> Result<Array, Exception> {
        let shape = pixel_values.shape().to_vec();
        if shape.len() != 6 {
            return Err(Exception::custom(format!(
                "mllama vision: pixel_values must be [batch, images, tiles, channels, height, width], got {shape:?}"
            )));
        }
        let (batch, images, tiles) = (shape[0], shape[1], shape[2]);
        let (channels, height, width) = (shape[3], shape[4], shape[5]);
        if tiles != self.config.max_num_tiles {
            // The per-arrangement embeddings are shaped for exactly
            // `max_num_tiles`; the processor always pads up to it.
            return Err(Exception::custom(format!(
                "mllama vision: pixel_values has {tiles} tile slots but max_num_tiles is {}",
                self.config.max_num_tiles
            )));
        }
        let rows = batch * images;
        let dim = self.config.hidden_size;

        // Patch embedding. MLX convolves NHWC and yields [N, h/p, w/p, hidden],
        // whose row-major flattening already matches the reference's
        // `flatten(2).transpose(1, 2)` on NCHW.
        let flat_tiles = pixel_values
            .reshape(&[rows * tiles, channels, height, width])
            .transpose_axes(&[0, 2, 3, 1]);
        let patch_embeds = Module::forward(&mut self.patch_embedding, &flat_tiles)?;
        let mut hidden_state = patch_embeds.reshape(&[rows * tiles, -1, dim]);
        let mut num_patches = hidden_state.dim(1);

        let ids = aspect_ratio_ids.reshape(&[rows, -1]);

        // Per-arrangement tile embedding, then the class token, then positions.
        hidden_state = self.pre_tile_positional_embedding.forward(
            &hidden_state.reshape(&[rows, tiles, num_patches, dim]),
            &ids,
        );

        hidden_state =
            self.apply_class_embedding(&hidden_state.reshape(&[rows * tiles, num_patches, dim]));
        num_patches += 1;
        if num_patches != self.config.num_patches() {
            return Err(Exception::custom(format!(
                "mllama vision: a {height}x{width} tile yields {num_patches} patches but the config \
                 (image_size {}, patch_size {}) describes {}; the position tables would be misaligned",
                self.config.image_size,
                self.config.patch_size,
                self.config.num_patches()
            )));
        }

        hidden_state = self.gated_positional_embedding.forward(
            &hidden_state.reshape(&[rows, tiles, num_patches, dim]),
            &ids,
        );
        hidden_state = self.layernorm_pre.forward(&hidden_state);

        // Pad the patch axis up to a multiple of 8.
        let num_padding = (8 - (num_patches % 8)) % 8;
        let padded = num_patches + num_padding;
        if num_padding > 0 {
            hidden_state = ops::pad(
                &hidden_state,
                &[(0, 0), (0, 0), (0, num_padding), (0, 0)],
                None,
                Some(0.0),
            );
        }

        let mask = aspect_ratio_attention_mask(
            &aspect_ratio_mask.reshape(&[rows, -1]),
            num_patches,
            padded,
        );

        // Per-tile encoder, over all tiles' patches at once.
        let (hidden, intermediates) = self.transformer.forward(
            &hidden_state.reshape(&[rows, tiles * padded, dim]),
            mask.as_ref(),
            &self.config.intermediate_layers_indices,
        );
        let hidden = self.layernorm_post.forward(&hidden);

        // Global encoder, after a second per-arrangement embedding.
        let hidden = self
            .post_tile_positional_embedding
            .forward(&hidden.reshape(&[rows, tiles, padded, dim]), &ids);
        let (hidden, _) = self.global_transformer.forward(
            &hidden.reshape(&[rows, tiles * padded, dim]),
            mask.as_ref(),
            &[],
        );

        // Strip the alignment padding from both halves and concatenate. The
        // intermediates are stacked on a *new last* axis before the reshape, so
        // the collected layers end up interleaved per hidden unit — not
        // block-concatenated. Getting that backwards is invisible in the shape.
        let unpad = |x: &Array, width: i32| {
            let x = x.reshape(&[rows, tiles, padded, width]);
            let x = if num_padding > 0 {
                ops::slice_axis(&x, 2, 0, num_patches)
            } else {
                x
            };
            x.reshape(&[batch, images, tiles, num_patches, width])
        };

        let last = unpad(&hidden, dim);
        if intermediates.len() != self.config.intermediate_layers_indices.len() {
            return Err(Exception::custom(
                "mllama vision: intermediate layer collection came back short",
            ));
        }
        let stacked = ops::stack_axis(&intermediates, 3);
        let stacked = unpad(&stacked, dim * intermediates.len() as i32);

        Ok(ops::concatenate_axis(&[&last, &stacked], -1))
    }
}

// =============================================================================
// Cross-attention & text decoder
// =============================================================================

/// Text-side cross-attention onto the projected vision features
/// (`MllamaTextCrossAttention`).
///
/// Unlike the text self-attention it carries per-head `q_norm`/`k_norm` RMSNorms
/// and takes no RoPE — the keys are image patches, which have no position in the
/// text sequence.
#[derive(Debug)]
pub struct MllamaTextCrossAttention {
    pub q_proj: nn::Linear,
    pub k_proj: nn::Linear,
    pub v_proj: nn::Linear,
    pub o_proj: nn::Linear,
    pub q_norm: nn::RmsNorm,
    pub k_norm: nn::RmsNorm,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    scaling: f32,
}
impl_module_params!(MllamaTextCrossAttention; q_proj, k_proj, v_proj, o_proj, q_norm, k_norm);

impl MllamaTextCrossAttention {
    pub fn new(config: &MllamaTextConfig) -> Result<Self, Exception> {
        let hidden = config.llama.hidden_size;
        let num_heads = config.llama.num_attention_heads;
        let num_kv_heads = config.llama.num_kv_heads();
        let head_dim = config.llama.get_head_dim();
        let eps = config.llama.rms_norm_eps;
        Ok(Self {
            q_proj: nn::LinearBuilder::new(hidden, num_heads * head_dim)
                .bias(false)
                .build()?,
            k_proj: nn::LinearBuilder::new(hidden, num_kv_heads * head_dim)
                .bias(false)
                .build()?,
            v_proj: nn::LinearBuilder::new(hidden, num_kv_heads * head_dim)
                .bias(false)
                .build()?,
            o_proj: nn::LinearBuilder::new(num_heads * head_dim, hidden)
                .bias(false)
                .build()?,
            q_norm: nn::RmsNormBuilder::new(head_dim).eps(eps).build()?,
            k_norm: nn::RmsNormBuilder::new(head_dim).eps(eps).build()?,
            num_heads,
            num_kv_heads,
            head_dim,
            scaling: (head_dim as f32).powf(-0.5),
        })
    }

    /// * `x`: `[batch, seq, hidden]`
    /// * `cross_states`: `[batch · images · tiles, patches, hidden]` — every
    ///   leading axis beyond the text batch collapses into the key sequence.
    /// * `mask`: optional additive `[batch, 1, seq, images · tiles · patches]`.
    pub fn forward(&self, x: &Array, cross_states: &Array, mask: Option<&Array>) -> Array {
        let batch = x.dim(0);
        let seq = x.dim(1);

        let q = self
            .q_proj
            .forward(x)
            .reshape(&[batch, seq, self.num_heads, self.head_dim])
            .transpose_axes(&[0, 2, 1, 3]);
        let q = self.q_norm.forward(&q);

        let kv = |p: &nn::Linear| {
            p.forward(cross_states)
                .reshape(&[batch, -1, self.num_kv_heads, self.head_dim])
                .transpose_axes(&[0, 2, 1, 3])
        };
        let k = self.k_norm.forward(&kv(&self.k_proj));
        let v = kv(&self.v_proj);

        let out = q.sdpa_with_mask(&k, &v, self.scaling, mask);
        self.o_proj
            .forward(&out.transpose_axes(&[0, 2, 1, 3]).reshape(&[
                batch,
                seq,
                self.num_heads * self.head_dim,
            ]))
    }
}

/// One text decoder layer.
///
/// Mllama's decoder interleaves two layer kinds — `MllamaSelfAttentionDecoderLayer`
/// and `MllamaCrossAttentionDecoderLayer` — that share `mlp`,
/// `input_layernorm` and `post_attention_layernorm` under exactly those names
/// and differ only in which attention they hold. Modelling that as one struct
/// with an either/or attention keeps the parameter keys identical to the
/// reference's while avoiding a second near-copy of the MLP half.
///
/// A cross-attention layer has **no self-attention**: it consumes one of the
/// `num_hidden_layers` slots outright.
#[derive(Debug)]
pub struct MllamaTextDecoderLayer {
    pub self_attn: Option<LlamaAttention>,
    pub cross_attn: Option<MllamaTextCrossAttention>,
    pub mlp: LlamaMLP,
    pub input_layernorm: nn::RmsNorm,
    pub post_attention_layernorm: nn::RmsNorm,
    /// Both gates are zero-initialised, so an untrained cross-attention layer
    /// starts out as the identity.
    pub cross_attn_attn_gate: Param<Option<Array>>,
    pub cross_attn_mlp_gate: Param<Option<Array>>,
}
impl_module_params!(
    MllamaTextDecoderLayer;
    self_attn,
    cross_attn,
    mlp,
    input_layernorm,
    post_attention_layernorm,
    cross_attn_attn_gate,
    cross_attn_mlp_gate
);

impl MllamaTextDecoderLayer {
    pub fn new(config: &MllamaTextConfig, layer_id: usize) -> Result<Self, Exception> {
        let hidden = config.llama.hidden_size;
        let eps = config.llama.rms_norm_eps;
        let norm = || nn::RmsNormBuilder::new(hidden).eps(eps).build();

        let is_cross = config.cross_attention_layers.contains(&(layer_id as i32));
        let (self_attn, cross_attn) = if is_cross {
            (None, Some(MllamaTextCrossAttention::new(config)?))
        } else {
            (Some(LlamaAttention::new(&config.llama, layer_id)?), None)
        };

        Ok(Self {
            self_attn,
            cross_attn,
            mlp: LlamaMLP::new(&config.llama)?,
            input_layernorm: norm()?,
            post_attention_layernorm: norm()?,
            cross_attn_attn_gate: if is_cross {
                zero_gate()
            } else {
                Param::new(None)
            },
            cross_attn_mlp_gate: if is_cross {
                zero_gate()
            } else {
                Param::new(None)
            },
        })
    }

    /// True for a cross-attention layer, which must be skipped entirely on a
    /// text-only forward (the reference `continue`s past it).
    pub fn is_cross_attention(&self) -> bool {
        self.cross_attn.is_some()
    }

    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cross: Option<&CrossAttentionInputs>,
        cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array, Exception> {
        let normed = self.input_layernorm.forward(x);

        let attn = match (self.self_attn.as_mut(), self.cross_attn.as_ref()) {
            (Some(self_attn), _) => self_attn.forward_with_cache(&normed, mask, cache)?,
            (None, Some(cross_attn)) => {
                let cross = cross.ok_or_else(|| {
                    Exception::custom(
                        "mllama: a cross-attention layer was run without vision features",
                    )
                })?;
                apply_gate(
                    &self.cross_attn_attn_gate,
                    &cross_attn.forward(&normed, &cross.states, cross.mask.as_ref()),
                )
            }
            (None, None) => {
                return Err(Exception::custom(
                    "mllama: decoder layer has neither self- nor cross-attention",
                ));
            }
        };
        let h = x.add(&attn);

        let mut ff = self
            .mlp
            .forward(&self.post_attention_layernorm.forward(&h))?;
        if self.cross_attn.is_some() {
            // Text rows that attend to no image at all are zeroed here rather
            // than masked in the softmax — see `prepare_cross_attention_mask`.
            if let Some(rows) = cross.and_then(|c| c.full_text_row_mask.as_ref()) {
                ff = ff.multiply(rows);
            }
            ff = apply_gate(&self.cross_attn_mlp_gate, &ff);
        }
        Ok(h.add(&ff))
    }
}

/// Everything the cross-attention layers need from the vision side.
#[derive(Debug, Clone)]
pub struct CrossAttentionInputs {
    /// Projected vision features, `[batch · images · tiles, patches, hidden]`.
    pub states: Array,
    /// Additive `[batch, 1, seq, images · tiles · patches]` mask.
    pub mask: Option<Array>,
    /// `[batch, seq, 1]` — `0` for text rows that attend to no image.
    pub full_text_row_mask: Option<Array>,
}

/// Build the cross-attention mask from the processor's
/// `[batch, seq, images, tiles]` 0/1 mask (`_prepare_cross_attention_mask`).
///
/// Each tile expands to `num_vision_tokens` key positions. The second return
/// value marks text rows with no visible image: rather than leave such a row
/// fully `-inf` (which would make softmax NaN), the reference *unmasks* it and
/// instead zeroes that row's MLP output in the decoder layer.
pub fn prepare_cross_attention_mask(
    cross_attention_mask: &Array,
    num_vision_tokens: i32,
) -> (Array, Array) {
    let batch = cross_attention_mask.dim(0);
    let seq = cross_attention_mask.dim(1);

    let visible = ops::repeat_axis(cross_attention_mask.as_type::<f32>(), num_vision_tokens, 3)
        .reshape(&[batch, seq, -1])
        .expand_dims(1);

    // 1 - visible is 0/1, so scaling by f32::MIN is the reference's
    // `masked_fill(inverted.bool(), finfo.min)`.
    let mask = Array::from_f32(1.0)
        .subtract(&visible)
        .multiply(&Array::from_f32(f32::MIN));

    // A row is live if it can see at least one vision token.
    let live = ops::greater(&visible.sum_axis(3, true), &Array::from_f32(0.0)).as_type::<f32>();

    (mask.multiply(&live), live.squeeze(1))
}

/// Mllama text decoder (`MllamaTextModel`).
#[derive(Debug)]
pub struct MllamaTextModel {
    pub config: MllamaTextConfig,

    pub embed_tokens: nn::Embedding,
    pub layers: Vec<MllamaTextDecoderLayer>,
    pub norm: nn::RmsNorm,
}
impl_module_params!(MllamaTextModel; embed_tokens, layers, norm);

impl MllamaTextModel {
    pub fn new(config: MllamaTextConfig) -> Result<Self, Exception> {
        // `vocab_size + 8`: the reference over-allocates the embedding table so
        // the added multimodal tokens (`<|image|>` among them) land past the
        // text vocabulary. The checkpoint ships the larger table, so anything
        // narrower fails to load.
        let embed_tokens =
            nn::Embedding::new(config.llama.vocab_size + 8, config.llama.hidden_size)?;

        let layers = (0..config.llama.num_hidden_layers)
            .map(|layer_id| MllamaTextDecoderLayer::new(&config, layer_id as usize))
            .collect::<Result<Vec<_>, _>>()?;

        let norm = nn::RmsNormBuilder::new(config.llama.hidden_size)
            .eps(config.llama.rms_norm_eps)
            .build()?;

        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
        })
    }

    pub fn forward(
        &mut self,
        input_ids: &Array,
        cross: Option<&CrossAttentionInputs>,
        mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        self.forward_with_cache(input_ids, cross, mask, None)
    }

    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        cross: Option<&CrossAttentionInputs>,
        mask: Option<&Array>,
        mut cache: Option<&mut KVCache>,
    ) -> Result<Array, Exception> {
        let mut hidden_states = self.embed_tokens.forward(input_ids);

        // Same convention as the other decoders here: build a causal mask only
        // when the caller supplied neither a mask nor a cache.
        let mask_owned;
        let mask = if mask.is_none() && cache.is_none() {
            mask_owned = crate::architectures::utils::create_causal_mask(input_ids.dim(1))?;
            Some(&mask_owned)
        } else {
            mask
        };

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            // Text-only inputs skip the cross-attention layers outright.
            if layer.is_cross_attention() && cross.is_none() {
                continue;
            }
            // Cross-attention layers hold no self-attention KV, so they leave
            // their cache slot untouched; indexing by the true layer id keeps
            // the remaining slots aligned with the reference's.
            let slot = cache.as_deref_mut().map(|c| (c, layer_idx));
            hidden_states = layer.forward(&hidden_states, mask, cross, slot)?;
        }

        Ok(self.norm.forward(&hidden_states))
    }
}

/// The three tensors the vision tower needs, as produced by
/// `pmetal_data::image_processing::MllamaImageProcessor`.
#[derive(Debug, Clone, Copy)]
pub struct MllamaVisionInputs<'a> {
    pub pixel_values: &'a Array,
    pub aspect_ratio_ids: &'a Array,
    pub aspect_ratio_mask: &'a Array,
}

/// Mllama for conditional generation (`MllamaForConditionalGeneration`).
#[derive(Debug)]
pub struct MllamaForConditionalGeneration {
    pub config: MllamaConfig,

    pub vision_model: MllamaVisionModel,
    /// A single biased linear from `vision_output_dim` to the text hidden size —
    /// not an MLP.
    pub multi_modal_projector: nn::Linear,
    pub language_model: MllamaTextModel,
    pub lm_head: nn::Linear,
}
impl_module_params!(MllamaForConditionalGeneration; vision_model, multi_modal_projector, language_model, lm_head);

impl MllamaForConditionalGeneration {
    pub fn new(config: MllamaConfig) -> Result<Self, Exception> {
        let vision_model = MllamaVisionModel::new(config.vision_config.clone())?;
        let multi_modal_projector = nn::LinearBuilder::new(
            config.vision_config.vision_output_dim,
            config.text_config.llama.hidden_size,
        )
        .bias(true)
        .build()?;
        let language_model = MllamaTextModel::new(config.text_config.clone())?;
        let lm_head = nn::LinearBuilder::new(
            config.text_config.llama.hidden_size,
            config.text_config.llama.vocab_size,
        )
        .bias(false)
        .build()?;

        Ok(Self {
            config,
            vision_model,
            multi_modal_projector,
            language_model,
            lm_head,
        })
    }

    /// Encode images and project them into the text embedding space.
    ///
    /// The projector's output keeps one row per `(image, tile)` and is folded
    /// into the key sequence by the cross-attention itself, so the leading axes
    /// collapse to `[batch · images · tiles, patches, hidden]`.
    pub fn encode_images(&mut self, vision: MllamaVisionInputs<'_>) -> Result<Array, Exception> {
        let features = self.vision_model.forward(
            vision.pixel_values,
            vision.aspect_ratio_ids,
            vision.aspect_ratio_mask,
        )?;
        let patches = features.dim(features.shape().len() as i32 - 2);
        let projected = self.multi_modal_projector.forward(&features);
        Ok(projected.reshape(&[-1, patches, self.config.text_config.llama.hidden_size]))
    }

    /// Run the vision tower once and package everything the cross-attention
    /// layers need.
    ///
    /// A generation loop should call this once and hand the result to every
    /// [`Self::forward_full`] step: the vision features are fixed, so there is
    /// nothing to cache across steps except the key/value projections of them
    /// (which the reference does cache, and we currently recompute).
    ///
    /// * `cross_attention_mask` — the processor's `[batch, seq, images, tiles]`
    ///   0/1 mask saying which text tokens may look at which tile. `None` lets
    ///   every token see every tile.
    pub fn prepare_cross_attention(
        &mut self,
        vision: MllamaVisionInputs<'_>,
        cross_attention_mask: Option<&Array>,
    ) -> Result<CrossAttentionInputs, Exception> {
        let states = self.encode_images(vision)?;
        let (mask, full_text_row_mask) = match cross_attention_mask {
            Some(m) => {
                let (mask, rows) = prepare_cross_attention_mask(m, self.vision_model.num_patches());
                (Some(mask), Some(rows))
            }
            None => (None, None),
        };
        Ok(CrossAttentionInputs {
            states,
            mask,
            full_text_row_mask,
        })
    }

    /// The full entry point. `cross` of `None` runs text-only and skips every
    /// cross-attention layer, exactly as the reference does.
    pub fn forward_full(
        &mut self,
        input_ids: &Array,
        cross: Option<&CrossAttentionInputs>,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, Exception> {
        let hidden_states = self
            .language_model
            .forward_with_cache(input_ids, cross, mask, cache)?;
        Ok(self.lm_head.forward(&hidden_states))
    }

    /// Single-shot multimodal forward: encode the images, then run the decoder.
    pub fn forward(
        &mut self,
        input_ids: &Array,
        vision: Option<MllamaVisionInputs<'_>>,
        cross_attention_mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        let cross = match vision {
            Some(vision) => Some(self.prepare_cross_attention(vision, cross_attention_mask)?),
            None => None,
        };
        self.forward_full(input_ids, cross.as_ref(), None, None)
    }

    /// Text-only cached decode, for the uniform `DynamicModel` path.
    ///
    /// Image-conditioned decoding cannot come through here — there is no vision
    /// channel in this signature — so use [`Self::prepare_cross_attention`] plus
    /// [`Self::forward_full`] for that.
    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, Exception> {
        self.forward_full(input_ids, None, mask, cache)
    }

    /// Final hidden state (pre-`lm_head`), for embedding / distillation callers.
    pub fn forward_hidden(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        self.language_model.forward(input_ids, None, mask)
    }
}

// =============================================================================
// Weight loading
// =============================================================================

/// Pick whichever of `candidates` a checkpoint actually uses, by probing for
/// `{prefix}{probe}`.
///
/// Mllama's released weights were saved when `MllamaForConditionalGeneration`
/// held `vision_model` / `language_model` directly; current `transformers`
/// nests both under a `MllamaModel` called `model`. Both layouts are in the
/// wild, and probing beats guessing.
fn resolve_prefix<'a>(
    weights: &HashMap<String, Array>,
    candidates: &[&'a str],
    probe: &str,
) -> &'a str {
    candidates
        .iter()
        .copied()
        .find(|prefix| weights.contains_key(&format!("{prefix}{probe}")))
        .unwrap_or(candidates[0])
}

/// Load HuggingFace `MllamaForConditionalGeneration` weights.
///
/// The one layout conversion is the patch-embedding convolution: PyTorch stores
/// `[out, in, kh, kw]` and MLX wants `[out, kh, kw, in]`.
pub fn load_mllama_weights(
    model: &mut MllamaForConditionalGeneration,
    weights: &HashMap<String, Array>,
) -> Result<LoadReport, Exception> {
    let mut report = LoadReport::default();

    let vision_root = resolve_prefix(
        weights,
        &["vision_model.", "model.vision_model."],
        "patch_embedding.weight",
    );
    let text_root = resolve_prefix(
        weights,
        &["language_model.model.", "model.language_model."],
        "embed_tokens.weight",
    );
    let projector_root = resolve_prefix(
        weights,
        &["multi_modal_projector.", "model.multi_modal_projector."],
        "weight",
    );

    load_vision_weights(&mut model.vision_model, weights, vision_root, &mut report);
    load_linear(
        &mut model.multi_modal_projector,
        weights,
        projector_root.trim_end_matches('.'),
        &mut report,
    );
    load_text_weights(&mut model.language_model, weights, text_root, &mut report);

    // `lm_head` is outside the text model in both layouts, but at different
    // depths.
    let lm_head_key = ["language_model.lm_head.weight", "lm_head.weight"]
        .into_iter()
        .find(|k| weights.contains_key(*k))
        .unwrap_or("lm_head.weight");
    load_param(&mut model.lm_head.weight, weights, lm_head_key, &mut report);

    Ok(report)
}

fn load_vision_encoder(
    encoder: &mut MllamaVisionEncoder,
    weights: &HashMap<String, Array>,
    root: &str,
    report: &mut LoadReport,
) {
    for (i, layer) in encoder.layers.iter_mut().enumerate() {
        let p = format!("{root}layers.{i}");
        for (proj, name) in [
            (&mut layer.self_attn.q_proj, "q_proj"),
            (&mut layer.self_attn.k_proj, "k_proj"),
            (&mut layer.self_attn.v_proj, "v_proj"),
            (&mut layer.self_attn.o_proj, "o_proj"),
        ] {
            load_linear(proj, weights, &format!("{p}.self_attn.{name}"), report);
        }
        load_linear(&mut layer.mlp.fc1, weights, &format!("{p}.mlp.fc1"), report);
        load_linear(&mut layer.mlp.fc2, weights, &format!("{p}.mlp.fc2"), report);
        load_layer_norm(
            &mut layer.input_layernorm,
            weights,
            &format!("{p}.input_layernorm"),
            report,
        );
        load_layer_norm(
            &mut layer.post_attention_layernorm,
            weights,
            &format!("{p}.post_attention_layernorm"),
            report,
        );
        // Only the global stack is gated; an un-gated layer must not report the
        // absent gates as skipped.
        if layer.gate_attn.value.is_some() {
            load_optional_param(
                &mut layer.gate_attn,
                weights,
                &format!("{p}.gate_attn"),
                report,
            );
            load_optional_param(
                &mut layer.gate_ffn,
                weights,
                &format!("{p}.gate_ffn"),
                report,
            );
        }
    }
}

fn load_vision_weights(
    vision: &mut MllamaVisionModel,
    weights: &HashMap<String, Array>,
    root: &str,
    report: &mut LoadReport,
) {
    let conv_key = format!("{root}patch_embedding.weight");
    match weights.get(&conv_key) {
        Some(w) => {
            // PyTorch [O, I, kh, kw] -> MLX [O, kh, kw, I].
            vision.patch_embedding.weight = Param::new(w.transpose_axes(&[0, 2, 3, 1]));
            report.loaded += 1;
        }
        None => report.skipped.push(conv_key),
    }

    load_param(
        &mut vision.class_embedding,
        weights,
        &format!("{root}class_embedding"),
        report,
    );

    let gpe = format!("{root}gated_positional_embedding");
    load_param(
        &mut vision.gated_positional_embedding.embedding,
        weights,
        &format!("{gpe}.embedding"),
        report,
    );
    load_optional_param(
        &mut vision.gated_positional_embedding.gate,
        weights,
        &format!("{gpe}.gate"),
        report,
    );
    load_param(
        &mut vision.gated_positional_embedding.tile_embedding.weight,
        weights,
        &format!("{gpe}.tile_embedding.weight"),
        report,
    );

    for (tile, name) in [
        (
            &mut vision.pre_tile_positional_embedding,
            "pre_tile_positional_embedding",
        ),
        (
            &mut vision.post_tile_positional_embedding,
            "post_tile_positional_embedding",
        ),
    ] {
        load_param(
            &mut tile.embedding.weight,
            weights,
            &format!("{root}{name}.embedding.weight"),
            report,
        );
        load_optional_param(
            &mut tile.gate,
            weights,
            &format!("{root}{name}.gate"),
            report,
        );
    }

    load_layer_norm(
        &mut vision.layernorm_pre,
        weights,
        &format!("{root}layernorm_pre"),
        report,
    );
    load_layer_norm(
        &mut vision.layernorm_post,
        weights,
        &format!("{root}layernorm_post"),
        report,
    );

    load_vision_encoder(
        &mut vision.transformer,
        weights,
        &format!("{root}transformer."),
        report,
    );
    load_vision_encoder(
        &mut vision.global_transformer,
        weights,
        &format!("{root}global_transformer."),
        report,
    );
}

fn load_text_weights(
    text: &mut MllamaTextModel,
    weights: &HashMap<String, Array>,
    root: &str,
    report: &mut LoadReport,
) {
    load_param(
        &mut text.embed_tokens.weight,
        weights,
        &format!("{root}embed_tokens.weight"),
        report,
    );
    load_param(
        &mut text.norm.weight,
        weights,
        &format!("{root}norm.weight"),
        report,
    );

    for (i, layer) in text.layers.iter_mut().enumerate() {
        let p = format!("{root}layers.{i}");

        if let Some(self_attn) = layer.self_attn.as_mut() {
            for (proj, name) in [
                (&mut self_attn.q_proj, "q_proj"),
                (&mut self_attn.k_proj, "k_proj"),
                (&mut self_attn.v_proj, "v_proj"),
                (&mut self_attn.o_proj, "o_proj"),
            ] {
                load_linear(proj, weights, &format!("{p}.self_attn.{name}"), report);
            }
        }

        if let Some(cross_attn) = layer.cross_attn.as_mut() {
            for (proj, name) in [
                (&mut cross_attn.q_proj, "q_proj"),
                (&mut cross_attn.k_proj, "k_proj"),
                (&mut cross_attn.v_proj, "v_proj"),
                (&mut cross_attn.o_proj, "o_proj"),
            ] {
                load_linear(proj, weights, &format!("{p}.cross_attn.{name}"), report);
            }
            load_param(
                &mut cross_attn.q_norm.weight,
                weights,
                &format!("{p}.cross_attn.q_norm.weight"),
                report,
            );
            load_param(
                &mut cross_attn.k_norm.weight,
                weights,
                &format!("{p}.cross_attn.k_norm.weight"),
                report,
            );
            load_optional_param(
                &mut layer.cross_attn_attn_gate,
                weights,
                &format!("{p}.cross_attn_attn_gate"),
                report,
            );
            load_optional_param(
                &mut layer.cross_attn_mlp_gate,
                weights,
                &format!("{p}.cross_attn_mlp_gate"),
                report,
            );
        }

        for (proj, name) in [
            (&mut layer.mlp.gate_proj, "gate_proj"),
            (&mut layer.mlp.up_proj, "up_proj"),
            (&mut layer.mlp.down_proj, "down_proj"),
        ] {
            load_linear(proj, weights, &format!("{p}.mlp.{name}"), report);
        }
        load_param(
            &mut layer.input_layernorm.weight,
            weights,
            &format!("{p}.input_layernorm.weight"),
            report,
        );
        load_param(
            &mut layer.post_attention_layernorm.weight,
            weights,
            &format!("{p}.post_attention_layernorm.weight"),
            report,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_bridge::compat::ModuleParametersExt;
    use serial_test::serial;

    const VOCAB: i32 = 64;
    const TILES: i32 = 4;

    /// A tiny but *structurally faithful* config: one gated global layer, one
    /// collected intermediate layer, and a 2x2 patch grid (so `num_patches` = 5
    /// and the 8-alignment pads by 3).
    fn tiny_config() -> MllamaConfig {
        let mut config = MllamaConfig::default();

        config.text_config.llama.vocab_size = VOCAB;
        config.text_config.llama.hidden_size = 64;
        config.text_config.llama.intermediate_size = 128;
        config.text_config.llama.num_hidden_layers = 3;
        config.text_config.llama.num_attention_heads = 4;
        config.text_config.llama.num_key_value_heads = Some(2);
        // Layer 1 of 3 carries cross-attention.
        config.text_config.cross_attention_layers = vec![1];

        config.vision_config.hidden_size = 32;
        config.vision_config.intermediate_size = 64;
        config.vision_config.num_hidden_layers = 2;
        config.vision_config.num_global_layers = 1;
        config.vision_config.attention_heads = 4;
        config.vision_config.image_size = 28;
        config.vision_config.patch_size = 14;
        config.vision_config.max_num_tiles = TILES;
        config.vision_config.intermediate_layers_indices = vec![1];
        config.vision_config.vision_output_dim = 64;

        config
    }

    /// Processor-shaped vision inputs for `batch` rows with one image each.
    fn vision_inputs(config: &MllamaConfig, batch: i32) -> (Array, Array, Array) {
        let vc = &config.vision_config;
        let pixels = random::normal(
            &[batch, 1, vc.max_num_tiles, 3, vc.image_size, vc.image_size],
            Dtype::Float32,
        );
        // Arrangement 2 for every image, all tiles real.
        let ids = Array::from_slice(&vec![2_i32; batch as usize], &[batch, 1]);
        let mask = Array::from_slice(
            &vec![1_i32; (batch * vc.max_num_tiles) as usize],
            &[batch, 1, vc.max_num_tiles],
        );
        (pixels, ids, mask)
    }

    #[test]
    #[serial]
    fn instantiation_exposes_parameters_for_both_towers() {
        let model = MllamaForConditionalGeneration::new(tiny_config()).unwrap();
        let params = model.flatten_params();
        assert!(!params.is_empty());

        // The gated global stack and the tile embeddings are the pieces most
        // easily left unbuilt; assert they are present by key.
        for key in [
            "vision_model.global_transformer.layers.0.gate_attn",
            "vision_model.global_transformer.layers.0.gate_ffn",
            "vision_model.pre_tile_positional_embedding.embedding.weight",
            "vision_model.post_tile_positional_embedding.gate",
            "vision_model.gated_positional_embedding.tile_embedding.weight",
            "vision_model.class_embedding",
            "language_model.layers.1.cross_attn.q_norm.weight",
            "language_model.layers.1.cross_attn_attn_gate",
            "multi_modal_projector.bias",
        ] {
            assert!(params.contains_key(key), "missing parameter {key}");
        }

        // A self-attention layer must carry no cross-attention parameters, and
        // vice versa — that is what makes the shared-struct layout safe.
        assert!(!params.contains_key("language_model.layers.0.cross_attn.q_proj.weight"));
        assert!(!params.contains_key("language_model.layers.1.self_attn.q_proj.weight"));
    }

    #[test]
    #[serial]
    fn vision_tower_output_is_the_concatenated_width() {
        let config = tiny_config();
        let mut tower = MllamaVisionModel::new(config.vision_config.clone()).unwrap();
        let (pixels, ids, mask) = vision_inputs(&config, 1);

        let out = tower.forward(&pixels, &ids, &mask).unwrap();
        assert_eq!(
            out.shape(),
            &[
                1,
                1,
                TILES,
                config.vision_config.num_patches(),
                config.vision_config.vision_output_dim
            ]
        );

        let mut out = out;
        out.eval();
        let n = out.size();
        let v = out.to_f32_vec(n).unwrap();
        assert!(v.iter().all(|x| x.is_finite()), "vision output has NaN/inf");
    }

    /// The tile-alignment padding is what makes the attention mask non-trivial;
    /// a `num_patches` that is already a multiple of 8 must not mask everything
    /// (the reference's `[-0:]` slice does exactly that).
    #[test]
    #[serial]
    fn attention_mask_masks_nothing_when_there_is_no_padding() {
        let tile_mask = Array::from_slice(&[1_i32, 1], &[1, 2]);
        let mut mask = aspect_ratio_attention_mask(&tile_mask, 8, 8).unwrap();
        mask.eval();
        let v = mask.to_f32_vec(16 * 16).unwrap();
        assert!(
            v.iter().all(|&x| x == 0.0),
            "fully-real mask masked a position"
        );
    }

    #[test]
    #[serial]
    fn attention_mask_masks_only_padding_against_padding() {
        // 2 tiles, the second one padding; 3 real patches out of 4 positions.
        let tile_mask = Array::from_slice(&[1_i32, 0], &[1, 2]);
        let mut mask = aspect_ratio_attention_mask(&tile_mask, 3, 4).unwrap();
        mask.eval();
        let v = mask.to_f32_vec(8 * 8).unwrap();

        // Padding positions: index 3 (tile 0's alignment pad) and 4..8 (tile 1).
        let is_pad = |i: usize| i == 3 || i >= 4;
        for q in 0..8 {
            for k in 0..8 {
                let want_masked = is_pad(q) && is_pad(k);
                let got = v[q * 8 + k];
                assert_eq!(
                    got != 0.0,
                    want_masked,
                    "query {q} vs key {k}: got {got}, want {}",
                    if want_masked { "masked" } else { "free" }
                );
            }
        }
    }

    #[test]
    #[serial]
    fn full_model_forward_with_and_without_images() {
        let config = tiny_config();
        let mut model = MllamaForConditionalGeneration::new(config.clone()).unwrap();
        let input_ids = Array::from_slice(&[1_i32, 2, 3, 4], &[1, 4]);
        let (pixels, ids, mask) = vision_inputs(&config, 1);

        let logits = model
            .forward(
                &input_ids,
                Some(MllamaVisionInputs {
                    pixel_values: &pixels,
                    aspect_ratio_ids: &ids,
                    aspect_ratio_mask: &mask,
                }),
                None,
            )
            .unwrap();
        assert_eq!(logits.shape(), &[1, 4, VOCAB]);

        // Text-only: the cross-attention layer is skipped, not run with zeros.
        let text_only = model.forward(&input_ids, None, None).unwrap();
        assert_eq!(text_only.shape(), &[1, 4, VOCAB]);
    }

    /// A cross-attention mask row that sees no tile must be zeroed rather than
    /// left as an all-`-inf` softmax row.
    #[test]
    #[serial]
    fn cross_attention_mask_frees_rows_that_see_no_image() {
        // batch 1, seq 2, 1 image, 2 tiles; row 0 sees both tiles, row 1 none.
        let visible = Array::from_slice(&[1_i32, 1, 0, 0], &[1, 2, 1, 2]);
        let (mut mask, mut rows) = prepare_cross_attention_mask(&visible, 2);
        assert_eq!(mask.shape(), &[1, 1, 2, 4]);
        assert_eq!(rows.shape(), &[1, 2, 1]);

        mask.eval();
        rows.eval();
        let m = mask.to_f32_vec(8).unwrap();
        let r = rows.to_f32_vec(2).unwrap();

        assert!(m[0..4].iter().all(|&x| x == 0.0), "visible row was masked");
        assert!(
            m[4..8].iter().all(|&x| x == 0.0),
            "a row seeing no tile must be freed, not left at -inf"
        );
        assert_eq!(r, vec![1.0, 0.0]);
    }

    #[test]
    fn config_rejects_a_vision_output_dim_that_cannot_be_produced() {
        let mut config = tiny_config();
        config.vision_config.vision_output_dim = 1234;
        assert!(MllamaVisionModel::new(config.vision_config).is_err());
    }

    #[test]
    fn default_geometry_matches_the_reference() {
        let config = MllamaVisionConfig::default();
        assert_eq!(config.num_patches(), 1025);
        assert_eq!(config.max_aspect_ratio_id(), 8);
        assert_eq!(config.output_dim(), 7680);
        assert_eq!(config.vision_output_dim, config.output_dim());
        // 1025 pads to 1032, so the reference's `[-pad:]` slice is well-formed
        // on a released checkpoint.
        assert_eq!((8 - config.num_patches() % 8) % 8, 7);
    }

    #[test]
    fn max_aspect_ratio_id_counts_the_processor_arrangements() {
        // Must agree with `pmetal_data::image_processing::supported_aspect_ratios`,
        // which lists 8 / 3 / 1 arrangements for max_num_tiles 4 / 2 / 1.
        for (max_num_tiles, want) in [(1, 1), (2, 3), (4, 8), (16, 50)] {
            let config = MllamaVisionConfig {
                max_num_tiles,
                ..Default::default()
            };
            assert_eq!(config.max_aspect_ratio_id(), want, "max {max_num_tiles}");
        }
    }
}
