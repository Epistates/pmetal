//! A DFlash drafter: a small block-diffusion model that guesses the next block
//! of tokens from the target model's hidden states, on the GPU.
//!
//! The target hands it the residual stream after a few of its layers for each
//! context token as the token is decided ([`DFlashDrafter::observe`]). The
//! drafter projects those into its own KV cache, so every draft attends to the
//! whole context. A draft ([`DFlashDrafter::propose`]) runs the block
//! `[last token, mask, mask, ...]` through the draft layers in one pass and
//! reads its guesses off the target's LM head, as dflash-mlx does
//! (`dflash_mlx/runtime.py`).

use std::path::Path;

use pmetal_bridge::compat::{Array, Exception, ops};
use pmetal_bridge::native_weight::{EmbeddingWeight, LayerWeight, QuantParams};
use pmetal_mlx::kv_cache::KVCache;

use crate::architectures::dflash_draft::DFlashDraftModel;
use crate::dflash_decoder::load_dflash_draft_from_dir;

/// Tensors per draft layer: q, k, v, o, q/k norms, two layer norms, and the
/// three MLP projections.
const TENSORS_PER_LAYER: usize = 11;

/// Whether the model in `dir` is a DFlash draft model, by its `config.json`.
pub fn is_dflash_draft(dir: &Path) -> bool {
    std::fs::read_to_string(dir.join("config.json"))
        .ok()
        .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
        .is_some_and(|config| config.get("dflash_config").is_some())
}

/// A DFlash draft model paired with its target's embedding and LM head.
pub struct DFlashDrafter {
    draft: DFlashDraftModel,
    /// One per draft layer: the projected context, and the block while a
    /// draft runs.
    cache: Vec<KVCache>,
    /// The target's token embedding, `[vocab, dim]`.
    embed: EmbeddingWeight,
    /// The target's LM head when it isn't the embedding.
    lm_head: Option<LayerWeight>,
    /// The activations' dtype: the target's, as its embedding was stored.
    dtype: i32,
    taps: Vec<usize>,
    dim: usize,
    /// Observed hidden states not yet in the cache, `[n, taps * dim]`.
    pending: Vec<f32>,
    /// Context positions the cache holds.
    max_context: usize,
}

impl DFlashDrafter {
    /// Load the draft model in `draft_dir` for the target in `target_dir`,
    /// with room for `max_context` tokens of context, its weights and its
    /// copy of the target's embedding and LM head packed as `quant` says
    /// (dense for `None`).
    pub fn load(
        draft_dir: &Path,
        target_dir: &Path,
        max_context: usize,
        quant: Option<QuantParams>,
    ) -> Result<Self, Exception> {
        let (mut draft, report) = load_dflash_draft_from_dir(draft_dir)?;
        let expected = TENSORS_PER_LAYER * draft.num_layers() + 3;
        if report.loaded != expected || !report.skipped.is_empty() {
            return Err(Exception::custom(format!(
                "DFlash draft {}: loaded {} of {expected} tensors; unrecognised: {:?}",
                draft_dir.display(),
                report.loaded,
                report.skipped
            )));
        }

        let target: serde_json::Value = std::fs::read_to_string(target_dir.join("config.json"))
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .ok_or_else(|| {
                Exception::custom(format!(
                    "DFlash target {}: can't read config.json",
                    target_dir.display()
                ))
            })?;
        let cfg = &draft.config;
        let dim = cfg.hidden_size as usize;
        let get = |key: &str| target[key].as_u64().unwrap_or(0) as usize;
        let (target_dim, target_vocab) = (get("hidden_size"), get("vocab_size"));
        let target_layers = get("num_hidden_layers");
        let taps = cfg.target_layer_ids();
        if target_dim != dim || target_vocab != cfg.vocab_size as usize {
            return Err(Exception::custom(format!(
                "DFlash draft {} is for a model {dim} wide with {} tokens; the target is \
                 {target_dim} wide with {target_vocab}",
                draft_dir.display(),
                cfg.vocab_size,
            )));
        }
        if taps.iter().any(|&t| t >= target_layers) {
            return Err(Exception::custom(format!(
                "DFlash draft reads target layers {taps:?}, past the target's {target_layers}"
            )));
        }

        let tied = target["tie_word_embeddings"].as_bool().unwrap_or(true);
        let mut weights = crate::loader::load_weights_filtered(target_dir, |key| {
            key == "model.embed_tokens.weight" || (!tied && key == "lm_head.weight")
        })
        .map_err(|e| Exception::custom(format!("DFlash target embedding: {e}")))?;
        let mut take = |key: &str| {
            weights.remove(key).ok_or_else(|| {
                Exception::custom(format!(
                    "DFlash target {} has no {key}",
                    target_dir.display()
                ))
            })
        };
        let embed = take("model.embed_tokens.weight")?;
        let lm_head = if tied {
            None
        } else {
            Some(take("lm_head.weight")?)
        };
        let dtype = embed.dtype().as_i32();
        let (embed, lm_head) = match quant {
            None => (
                EmbeddingWeight::new(
                    embed,
                    None,
                    None,
                    QuantParams::defaults_for(pmetal_bridge::QuantizedMode::Affine),
                ),
                lm_head.map(|w| {
                    LayerWeight::new(
                        w,
                        None,
                        None,
                        QuantParams::defaults_for(pmetal_bridge::QuantizedMode::Affine),
                    )
                }),
            ),
            Some(params) => {
                draft.quantize(params)?;
                let pack = |w: Array| {
                    let (w, s, b) = w.quantize_weights(params.group_size, params.bits);
                    pmetal_bridge::check_last_error()
                        .map(|()| (w, s, b))
                        .map_err(|e| Exception::custom(format!("DFlash target head: {e}")))
                };
                let (w, s, b) = pack(embed)?;
                let embed = EmbeddingWeight::new(w, Some(s), Some(b), params);
                let lm_head = match lm_head {
                    Some(head) => {
                        let (w, s, b) = pack(head)?;
                        Some(LayerWeight::new(w, Some(s), Some(b), params))
                    }
                    None => None,
                };
                (embed, lm_head)
            }
        };

        let mut drafter = Self {
            cache: Vec::new(),
            draft,
            embed,
            lm_head,
            dtype,
            taps,
            dim,
            pending: Vec::new(),
            max_context,
        };
        // MLX builds its kernels on first use, ~1.5 s that would otherwise
        // land on the first request. A draft after a prompt reads many rows
        // of context and one mid-generation a few, which take different
        // kernels (packed matmuls especially), so warm both.
        drafter.reset();
        let row = drafter.taps.len() * dim;
        for n in [64, 1] {
            drafter.observe(&vec![0.0; n * row], n)?;
            drafter.propose(0, drafter.max_guesses())?;
        }
        drafter.reset();
        Ok(drafter)
    }

    /// Target layers whose output the draft reads, ascending.
    pub fn target_layer_ids(&self) -> &[usize] {
        &self.taps
    }

    /// Tokens a draft guesses at most: the block less its first token.
    pub fn max_guesses(&self) -> usize {
        self.draft.block_size() - 1
    }

    /// Forget the context.
    pub fn reset(&mut self) {
        self.cache = self.draft.make_cache(self.max_context);
        self.pending.clear();
    }

    /// Add `n` context tokens' hidden states, `[n, taps * dim]` token-major,
    /// the target layers in [`target_layer_ids`](Self::target_layer_ids)
    /// order.
    pub fn observe(&mut self, hidden: &[f32], n: usize) -> Result<(), Exception> {
        let row = self.taps.len() * self.dim;
        if hidden.len() != n * row {
            return Err(Exception::custom(format!(
                "DFlash observe: {} values for {n} tokens of {row}",
                hidden.len()
            )));
        }
        self.pending.extend_from_slice(hidden);
        Ok(())
    }

    /// Up to `max` guesses at the tokens after `last`, the token after the
    /// observed context.
    pub fn propose(&mut self, last: u32, max: usize) -> Result<Vec<u32>, Exception> {
        let max = max.min(self.max_guesses());
        if max == 0 {
            // The observed states stay pending for the next draft.
            return Ok(Vec::new());
        }
        let (bs, dim) = (self.draft.block_size(), self.dim);
        let row = self.taps.len() * dim;
        let n = self.pending.len() / row;
        if n == 0 {
            return Err(Exception::custom(
                "DFlash propose: no context observed since the last draft",
            ));
        }

        let mut block = vec![self.draft.mask_token_id(); bs];
        block[0] = last as i32;
        let ids = Array::from_slice(&block, &[bs as i32]);
        let noise = self.embed.lookup(&ids).reshape(&[1, bs as i32, dim as i32]);
        let context =
            Array::from_slice(&self.pending, &[1, n as i32, row as i32]).as_dtype(self.dtype);
        let hidden = self.draft.draft_block(&noise, &context, &mut self.cache)?;
        self.pending.clear();

        let guesses = hidden.slice(&[0, 1, 0], &[1, 1 + max as i32, dim as i32]);
        let logits = match &self.lm_head {
            Some(head) => head.matmul_from(&guesses),
            None => self.embed.as_linear(&guesses),
        };
        let tokens = ops::argmax_axis(&logits, -1);
        let _ = tokens.eval();
        pmetal_bridge::check_last_error()
            .map_err(|e| Exception::custom(format!("DFlash draft: {e}")))?;
        Ok(tokens.as_slice::<u32>().to_vec())
    }
}

#[cfg(feature = "ane")]
impl pmetal_metal::ane::lm::Drafter for DFlashDrafter {
    fn taps(&self) -> &[usize] {
        &self.taps
    }

    fn reset(&mut self) {
        DFlashDrafter::reset(self);
    }

    fn observe(&mut self, hidden: &[f32], n: usize) -> pmetal_metal::error::Result<()> {
        DFlashDrafter::observe(self, hidden, n)
            .map_err(|e| pmetal_metal::error::MetalError::ExecutionFailed(e.to_string()))
    }

    fn draft(&mut self, context: &[u32], max: usize) -> pmetal_metal::error::Result<Vec<u32>> {
        let last = *context.last().ok_or_else(|| {
            pmetal_metal::error::MetalError::InvalidConfig("DFlash draft: empty context".into())
        })?;
        self.propose(last, max)
            .map_err(|e| pmetal_metal::error::MetalError::ExecutionFailed(e.to_string()))
    }
}
