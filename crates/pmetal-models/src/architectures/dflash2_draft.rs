//! DFlash 2 block-diffusion draft model.
//!
//! DFlash 2 (`architectures: ["DFlash2DraftModel"]`, e.g.
//! `z-lab/Qwen3.8-27B-DFlash2`) keeps the first generation's shape: a few
//! decoder layers that draft a block `[last token, mask, ...]` in one pass,
//! attending to the target's hidden states at a handful of its layers
//! (projected by `fc`), and reading the guesses off the target's LM head. It
//! adds two things.
//!
//! * **Two-tap dynamic convolutions** around each sublayer
//!   ([`GroupedDynamicCausalConv`]). The normed input of the attention and of
//!   the MLP is convolved causally along the block before the sublayer and
//!   its output after, with kernels that are a learned base plus a
//!   per-position term projected from the normed input. They carry each
//!   position's neighbour into it, which keeps the late positions of a block
//!   from decaying.
//! * **A candidate selector** ([`CandidateSelector`]): instead of each
//!   position's argmax, it keeps the top `selector_top_k` tokens at every
//!   position and walks one path through them, scoring each candidate by its
//!   logit plus a low-rank edge between the token chosen before it and itself,
//!   modulated by the position's hidden state.
//!
//! Its layers are sliding-window (2048 for Qwen3.8-27B) and attend within the
//! block bidirectionally (`is_causal: false`). Attention, the MLP and the
//! context cache are the first generation's
//! ([`super::dflash_draft`]); this module adds the convolutions, the
//! selector, and a strict loader.
//!
//! The port follows the reference implementation (z-lab/dflash): its MLX
//! `model_mlx.py` for the operation order, which is what Apple Silicon runs,
//! and its PyTorch `model.py`, which the parity fixture is dumped from.

use std::collections::HashMap;

use pmetal_bridge::compat::{Array, Exception, ModuleParametersExt, Param, nn, ops};
use pmetal_bridge::impl_module_params;
use pmetal_mlx::kv_cache::KVCache;

use super::dflash_draft::{
    DFlashAttention, DFlashDraftConfig, DFlashMlp, LoadReport, make_context_cache,
};

// ----------------------------------------------------------------------------
// Dynamic convolution
// ----------------------------------------------------------------------------

/// A causal convolution along the block whose kernel is a learned base plus
/// a term projected from the input at each position, one weight per group of
/// `group_size` channels.
///
/// It runs in two halves around a sublayer: [`prepare`](Self::prepare)
/// convolves the sublayer's input with the first base kernel and the first
/// half of the projected kernels, and hands back the second half, with which
/// [`finish`](Self::finish) convolves the sublayer's output.
#[derive(Debug)]
pub struct GroupedDynamicCausalConv {
    /// `[2, kernel_size, hidden]`: the static kernel of each half.
    pub base_kernel: Param<Array>,
    /// `hidden -> 2 * kernel_size * groups`.
    pub kernel_projection: nn::Linear,
    pub kernel_size: i32,
    pub group_size: i32,
}
impl_module_params!(GroupedDynamicCausalConv; base_kernel, kernel_projection);

impl GroupedDynamicCausalConv {
    pub fn new(hidden_size: i32, kernel_size: i32, group_size: i32) -> Result<Self, Exception> {
        if kernel_size < 1 || group_size < 1 || hidden_size % group_size != 0 {
            return Err(Exception::custom(format!(
                "DFlash 2 convolution: kernel {kernel_size} and group {group_size} \
                 don't fit hidden size {hidden_size}"
            )));
        }
        let groups = hidden_size / group_size;
        Ok(Self {
            base_kernel: Param::new(ops::zeros(
                &[2, kernel_size, hidden_size],
                pmetal_bridge::compat::Dtype::Float32,
            )),
            kernel_projection: nn::LinearBuilder::new(hidden_size, 2 * kernel_size * groups)
                .bias(false)
                .build()?,
            kernel_size,
            group_size,
        })
    }

    /// Convolve `hidden` `[B, L, hidden]` with the first half's kernels and
    /// return it with the second half's projected kernels, `[B, L, K, G]`.
    pub fn prepare(&mut self, hidden: &Array) -> (Array, Array) {
        let (b, l, h) = (hidden.dim(0), hidden.dim(1), hidden.dim(2));
        let (k, groups) = (self.kernel_size, h / self.group_size);
        let dynamic = self
            .kernel_projection
            .forward(hidden)
            .reshape(&[b, l, 2, k, groups]);
        let half = |i: i32| {
            dynamic
                .slice(&[0, 0, i, 0, 0], &[b, l, i + 1, k, groups])
                .reshape(&[b, l, k, groups])
        };
        let out = self.convolve(hidden, &half(0), 0);
        (out, half(1))
    }

    /// Convolve `hidden` with the second half's kernels.
    pub fn finish(&self, hidden: &Array, dynamic: &Array) -> Array {
        self.convolve(hidden, dynamic, 1)
    }

    /// `out[t] = sum_o (base[o] + dynamic[t, o]) * x[t - o]`, zero before the
    /// block, the dynamic weight shared within each group of channels. The
    /// operation order is the reference's.
    fn convolve(&self, hidden: &Array, dynamic: &Array, half: i32) -> Array {
        let (b, l, h) = (hidden.dim(0), hidden.dim(1), hidden.dim(2));
        let (k, gs) = (self.kernel_size, self.group_size);
        let groups = h / gs;
        let blocks = hidden.reshape(&[b, l, groups, gs]);
        let dynamic = dynamic.reshape(&[b, l, k, groups, 1]);
        let base = self.base_kernel.value.as_ref();
        let mut out = ops::zeros_like(&blocks);
        for offset in 0..k {
            let values = if offset == 0 {
                blocks.clone()
            } else if offset >= l {
                ops::zeros_like(&blocks)
            } else {
                let pad = ops::zeros(&[b, offset, groups, gs], blocks.dtype());
                let head = blocks.slice(&[0, 0, 0, 0], &[b, l - offset, groups, gs]);
                ops::concatenate_axis(&[&pad, &head], 1)
            };
            let kernel = base
                .slice(&[half, offset, 0], &[half + 1, offset + 1, h])
                .reshape(&[1, 1, groups, gs])
                .as_dtype(hidden.dtype().as_i32());
            let weight = dynamic
                .slice(&[0, 0, offset, 0, 0], &[b, l, offset + 1, groups, 1])
                .reshape(&[b, l, groups, 1]);
            out = out.add(&kernel.multiply(&values));
            out = out.add(&weight.multiply(&values));
        }
        out.reshape(&[b, l, h])
    }
}

// ----------------------------------------------------------------------------
// Decoder layer
// ----------------------------------------------------------------------------

/// A DFlash 2 layer: the first generation's attention and MLP, each wrapped
/// in a dynamic convolution.
#[derive(Debug)]
pub struct DFlash2DecoderLayer {
    pub input_layernorm: nn::RmsNorm,
    pub self_attn: DFlashAttention,
    pub post_attention_layernorm: nn::RmsNorm,
    pub mlp: DFlashMlp,
    pub attention_conv: GroupedDynamicCausalConv,
    pub mlp_conv: GroupedDynamicCausalConv,
}
impl_module_params!(
    DFlash2DecoderLayer;
    input_layernorm,
    self_attn,
    post_attention_layernorm,
    mlp,
    attention_conv,
    mlp_conv
);

impl DFlash2DecoderLayer {
    pub fn new(config: &DFlashDraftConfig, layer: usize) -> Result<Self, Exception> {
        let norm = || {
            nn::RmsNormBuilder::new(config.hidden_size)
                .eps(config.rms_norm_eps)
                .build()
        };
        let extras = &config.dflash_config;
        let (kernel, group) = match (extras.conv_kernel_size, extras.conv_group_size) {
            (Some(k), Some(g)) => (k, g),
            _ => {
                return Err(Exception::custom(
                    "DFlash 2 draft: dflash_config needs conv_kernel_size and conv_group_size",
                ));
            }
        };
        let conv = || GroupedDynamicCausalConv::new(config.hidden_size, kernel, group);
        Ok(Self {
            input_layernorm: norm()?,
            self_attn: DFlashAttention::new(config, layer)?,
            post_attention_layernorm: norm()?,
            mlp: DFlashMlp::new(config)?,
            attention_conv: conv()?,
            mlp_conv: conv()?,
        })
    }

    pub fn forward(
        &mut self,
        hidden_states: &Array,
        target_hidden: &Array,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, Exception> {
        let normed = self.input_layernorm.forward(hidden_states);
        let (x, kernel) = self.attention_conv.prepare(&normed);
        let attn = self.self_attn.forward(&x, target_hidden, cache)?;
        let hidden_states = hidden_states.add(&self.attention_conv.finish(&attn, &kernel));

        let normed = self.post_attention_layernorm.forward(&hidden_states);
        let (x, kernel) = self.mlp_conv.prepare(&normed);
        let mlp = self.mlp.forward(&x)?;
        Ok(hidden_states.add(&self.mlp_conv.finish(&mlp, &kernel)))
    }
}

// ----------------------------------------------------------------------------
// Candidate selector
// ----------------------------------------------------------------------------

/// Picks one path through each position's top-k candidates.
///
/// At position `t` with hidden state `h_t` the selector scores candidate `c`
/// as `logit_t[c] + sum_r P[prev][r] * (W h_t)[r] * S[c][r]`, where `prev` is
/// the token chosen at `t - 1` (the block's anchor at `t = 0`), `P` and `S`
/// the predecessor and successor codebooks, and `W` the hidden projection.
#[derive(Debug)]
pub struct CandidateSelector {
    /// `[vocab, rank]`.
    pub predecessor_codebook: Param<Array>,
    /// `[vocab, rank]`.
    pub successor_codebook: Param<Array>,
    /// `hidden -> rank`.
    pub hidden_projection: nn::Linear,
    pub top_k: i32,
}
impl_module_params!(
    CandidateSelector;
    predecessor_codebook,
    successor_codebook,
    hidden_projection
);

impl CandidateSelector {
    pub fn new(config: &DFlashDraftConfig) -> Result<Self, Exception> {
        let extras = &config.dflash_config;
        let (rank, top_k) = match (extras.selector_rank, extras.selector_top_k) {
            (Some(r), Some(k)) if r > 0 && k > 0 && k <= config.vocab_size => (r, k),
            _ => {
                return Err(Exception::custom(
                    "DFlash 2 draft: dflash_config needs a positive selector_rank and a \
                     selector_top_k within the vocabulary",
                ));
            }
        };
        let codebook = || {
            Param::new(ops::zeros(
                &[config.vocab_size, rank],
                pmetal_bridge::compat::Dtype::Float32,
            ))
        };
        Ok(Self {
            predecessor_codebook: codebook(),
            successor_codebook: codebook(),
            hidden_projection: nn::LinearBuilder::new(config.hidden_size, rank)
                .bias(false)
                .build()?,
            top_k,
        })
    }

    /// The greedy path, `[B, L]` token ids (lazy), from the draft's hidden
    /// states `[B, L, hidden]`, its logits `[B, L, vocab]`, and the block's
    /// anchor tokens `[B]`.
    ///
    /// The candidates are the reference's `argpartition` top-k and ties go to
    /// the first in that order, so a tie breaks as it does there.
    pub fn select(&mut self, hidden: &Array, logits: &Array, anchor: &Array) -> Array {
        let (b, l, vocab) = (logits.dim(0), logits.dim(1), logits.dim(2));
        let k = self.top_k;
        let candidates =
            ops::argpartition_axis(logits, vocab - k, -1).slice(&[0, 0, vocab - k], &[b, l, vocab]);
        let unary = ops::take_along_axis(logits, &candidates, -1);
        let projected = self.hidden_projection.forward(hidden);
        let rank = projected.dim(2);
        let predecessors = self.predecessor_codebook.value.as_ref();
        let successors = self.successor_codebook.value.as_ref();

        let mut previous = anchor.clone();
        let mut path = Vec::with_capacity(l as usize);
        for t in 0..l {
            let at = |x: &Array| {
                let d = x.dim(2);
                x.slice(&[0, t, 0], &[b, t + 1, d]).reshape(&[b, d])
            };
            let position_candidates = at(&candidates);
            // [B, 1, R] * [B, 1, R] * [B, k, R] -> [B, k]
            let edges = predecessors
                .take_axis(&previous, 0)
                .reshape(&[b, 1, rank])
                .multiply(&at(&projected).reshape(&[b, 1, rank]))
                .multiply(&successors.take_axis(&position_candidates, 0));
            let scores = at(&unary).add(&ops::sum_axis(&edges, -1, false));
            let chosen = ops::argmax_axis(&scores, -1).reshape(&[b, 1]);
            previous = ops::take_along_axis(&position_candidates, &chosen, -1).reshape(&[b]);
            path.push(previous.clone());
        }
        ops::stack_axis(&path, 1)
    }
}

// ----------------------------------------------------------------------------
// Draft model
// ----------------------------------------------------------------------------

/// DFlash 2 draft model. Like the first generation it owns no embedding or
/// LM head; the caller lends the target's.
#[derive(Debug)]
pub struct DFlash2DraftModel {
    pub layers: Vec<DFlash2DecoderLayer>,
    /// Projects `[B, T, taps * hidden]` target hidden states to `[B, T, hidden]`.
    pub fc: nn::Linear,
    pub hidden_norm: nn::RmsNorm,
    pub norm: nn::RmsNorm,
    pub candidate_selector: CandidateSelector,
    pub config: DFlashDraftConfig,
}
impl_module_params!(
    DFlash2DraftModel;
    layers,
    fc,
    hidden_norm,
    norm,
    candidate_selector
);

impl DFlash2DraftModel {
    pub fn new(config: DFlashDraftConfig) -> Result<Self, Exception> {
        config.validate()?;
        let taps = config.num_target_layers() as i32;
        if taps == 0 {
            return Err(Exception::custom(
                "DFlash2DraftModel requires at least one target_layer_id",
            ));
        }
        let layers = (0..config.num_hidden_layers as usize)
            .map(|layer| DFlash2DecoderLayer::new(&config, layer))
            .collect::<Result<Vec<_>, _>>()?;
        let norm = || {
            nn::RmsNormBuilder::new(config.hidden_size)
                .eps(config.rms_norm_eps)
                .build()
        };
        Ok(Self {
            layers,
            fc: nn::LinearBuilder::new(taps * config.hidden_size, config.hidden_size)
                .bias(false)
                .build()?,
            hidden_norm: norm()?,
            norm: norm()?,
            candidate_selector: CandidateSelector::new(&config)?,
            config,
        })
    }

    /// Tokens per drafted block, the anchor included.
    pub fn block_size(&self) -> usize {
        self.config.block_size() as usize
    }

    /// Token id filling the block's guessed positions.
    pub fn mask_token_id(&self) -> i32 {
        self.config.dflash_config.mask_token_id
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// The draft's hidden states over the block, normed, `[B, L, hidden]`.
    /// `noise_embedding` is the block's embedding (scaled by the config's
    /// `input_embedding_scale`) and `target_hidden` `[B, T, taps * hidden]`
    /// the target's tapped states for the context: with a cache from
    /// [`make_cache`](Self::make_cache) the positions new since the last
    /// draft, which join it; without one, the whole context.
    pub fn forward(
        &mut self,
        noise_embedding: &Array,
        target_hidden: &Array,
        mut cache: Option<&mut [KVCache]>,
    ) -> Result<Array, Exception> {
        let context = self.hidden_norm.forward(&self.fc.forward(target_hidden));
        let mut hidden = noise_embedding.clone();
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let layer_cache = cache.as_deref_mut().and_then(|c| c.get_mut(i));
            hidden = layer.forward(&hidden, &context, layer_cache)?;
        }
        Ok(self.norm.forward(&hidden))
    }

    /// Draft the block whose embedding is `noise_embedding` `[1, L, hidden]`
    /// (anchor first): the guesses for its last `L - 1` positions, `[1, L - 1]`
    /// (lazy), read through `lm_head` and the selector. `anchor` `[1]` is the
    /// block's first token.
    pub fn propose(
        &mut self,
        noise_embedding: &Array,
        anchor: &Array,
        target_hidden: &Array,
        cache: &mut [KVCache],
        lm_head: &mut dyn FnMut(&Array) -> Result<Array, Exception>,
    ) -> Result<Array, Exception> {
        let hidden = self.forward(noise_embedding, target_hidden, Some(cache))?;
        let (b, l, h) = (hidden.dim(0), hidden.dim(1), hidden.dim(2));
        let guesses = hidden.slice(&[0, 1, 0], &[b, l, h]);
        let logits = self.config.scale_logits(lm_head(&guesses)?);
        Ok(self.candidate_selector.select(&guesses, &logits, anchor))
    }

    /// A KV cache per layer holding the context; see
    /// [`make_context_cache`].
    pub fn make_cache(&self, context: usize) -> Vec<KVCache> {
        make_context_cache(&self.config, context)
    }

    /// Pack every projection for MLX's quantized matmul (the codebooks and
    /// convolution base kernels stay as loaded).
    pub fn quantize(
        &mut self,
        params: pmetal_bridge::native_weight::QuantParams,
    ) -> Result<(), Exception> {
        let mut result = Ok(());
        pmetal_bridge::compat::VisitLinears::visit_linears_mut(self, "", &mut |_, linear| {
            if result.is_ok() {
                result = linear.quantize(params);
            }
        });
        result
    }

    /// Load every parameter from `weights`, the checkpoint's own names
    /// (`layers.{i}.…`, `fc.weight`, `candidate_selector.…`, optionally under
    /// `model.`). Strict both ways: a tensor the model has no place for, a
    /// parameter the checkpoint doesn't fill, or a shape that doesn't match
    /// is an error.
    pub fn load_weights(
        &mut self,
        weights: &HashMap<String, Array>,
    ) -> Result<LoadReport, Exception> {
        let mut params = self.flatten_params_mut();
        let mut report = LoadReport::default();
        let mut unknown = Vec::new();
        for (name, weight) in weights {
            let stripped = name.strip_prefix("model.").unwrap_or(name);
            match params.remove(stripped) {
                Some(slot) => {
                    if slot.shape() != weight.shape() {
                        return Err(Exception::custom(format!(
                            "DFlash 2 draft: {name} is {:?}, the model's is {:?}",
                            weight.shape(),
                            slot.shape()
                        )));
                    }
                    *slot = weight.clone();
                    report.loaded += 1;
                }
                None => unknown.push(name.clone()),
            }
        }
        if !unknown.is_empty() || !params.is_empty() {
            unknown.sort();
            let mut missing: Vec<_> = params.into_keys().collect();
            missing.sort();
            return Err(Exception::custom(format!(
                "DFlash 2 draft: checkpoint tensors the model has no place for: {unknown:?}; \
                 parameters the checkpoint doesn't fill: {missing:?}"
            )));
        }
        Ok(report)
    }
}
