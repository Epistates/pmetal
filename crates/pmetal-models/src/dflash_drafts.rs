//! Both generations of DFlash draft model behind one interface: telling them
//! apart, loading them strictly, and drafting a block.
//!
//! A checkpoint is DFlash 2 when its `config.json` declares the
//! `DFlash2DraftModel` architecture, as the reference implementation decides;
//! anything else with a `dflash_config` is the first generation. Both draft
//! the same way from the caller's side: embed the block `[anchor, mask, ...]`
//! with the target's embedding, hand over the target's tapped hidden states
//! for the context decided since the last draft, and lend the target's LM
//! head. The first generation guesses each position's argmax; DFlash 2 walks
//! its candidate selector.

use std::path::Path;

use pmetal_bridge::compat::{Array, Exception, ModuleParametersExt, ops};
use pmetal_bridge::native_weight::QuantParams;
use pmetal_mlx::kv_cache::KVCache;

use crate::architectures::dflash_draft::{
    DFlashDraftConfig, DFlashDraftModel, LoadReport, make_context_cache,
};
use crate::architectures::dflash2_draft::DFlash2DraftModel;
use crate::dflash_decoder::DFlashDraftQuant;

/// A DFlash draft model of either generation.
#[derive(Debug)]
pub enum DFlashDraft {
    /// `DFlashDraftModel`: per-position argmax.
    V1(DFlashDraftModel),
    /// `DFlash2DraftModel`: dynamic convolutions and a candidate selector.
    V2(DFlash2DraftModel),
}

impl From<DFlashDraftModel> for DFlashDraft {
    fn from(model: DFlashDraftModel) -> Self {
        Self::V1(model)
    }
}

impl From<DFlash2DraftModel> for DFlashDraft {
    fn from(model: DFlash2DraftModel) -> Self {
        Self::V2(model)
    }
}

impl DFlashDraft {
    pub fn config(&self) -> &DFlashDraftConfig {
        match self {
            Self::V1(m) => &m.config,
            Self::V2(m) => &m.config,
        }
    }

    /// Whether this is a DFlash 2 draft.
    pub fn is_dflash2(&self) -> bool {
        matches!(self, Self::V2(_))
    }

    /// Tokens per drafted block, the anchor included.
    pub fn block_size(&self) -> usize {
        self.config().block_size() as usize
    }

    /// Token id filling the block's guessed positions.
    pub fn mask_token_id(&self) -> i32 {
        self.config().dflash_config.mask_token_id
    }

    /// Target layers whose output the draft reads, in the order it
    /// concatenates them.
    pub fn target_layer_ids(&self) -> Vec<usize> {
        self.config().target_layer_ids()
    }

    pub fn num_layers(&self) -> usize {
        self.config().num_hidden_layers as usize
    }

    /// A KV cache per layer holding the context, with room for `context`
    /// positions (a sliding layer keeps its window).
    pub fn make_cache(&self, context: usize) -> Vec<KVCache> {
        make_context_cache(self.config(), context)
    }

    /// Pack every projection for MLX's quantized matmul.
    pub fn quantize(&mut self, params: QuantParams) -> Result<(), Exception> {
        match self {
            Self::V1(m) => m.quantize(params),
            Self::V2(m) => m.quantize(params),
        }
    }

    /// The block's embedding as the draft takes it: the target's, scaled by
    /// the config's `input_embedding_scale`.
    fn scale_embedding(&self, embedding: &Array) -> Array {
        let scale = self.config().input_embedding_scale();
        if scale == 1.0 {
            embedding.clone()
        } else {
            embedding.multiply(&Array::from_f32(scale).as_dtype(embedding.dtype().as_i32()))
        }
    }

    /// Guess the `guesses` tokens after the block's anchor, `[1, guesses]`
    /// (lazy, unsigned ids).
    ///
    /// `block_embedding` `[1, L, hidden]` is the target's embedding of the
    /// block `[anchor, mask, ...]` and `anchor` its first token;
    /// `target_hidden` `[1, T, taps * hidden]` holds the target's tapped
    /// states for the `T` context positions decided since the last draft,
    /// which join `cache`; `lm_head` is the target's LM head. `guesses` is at
    /// most `L - 1`.
    pub fn propose(
        &mut self,
        block_embedding: &Array,
        anchor: i32,
        target_hidden: &Array,
        cache: &mut [KVCache],
        lm_head: &mut dyn FnMut(&Array) -> Result<Array, Exception>,
        guesses: usize,
    ) -> Result<Array, Exception> {
        let block = block_embedding.dim(1);
        let guesses = guesses as i32;
        if guesses < 1 || guesses >= block {
            return Err(Exception::custom(format!(
                "DFlash draft: {guesses} guesses from a block of {block}"
            )));
        }
        let embedding = self.scale_embedding(block_embedding);
        match self {
            Self::V1(model) => {
                let logits = v1_logits(model, &embedding, target_hidden, cache, lm_head, guesses)?;
                Ok(ops::argmax_axis(&logits, -1))
            }
            Self::V2(model) => {
                let anchor = Array::from_slice(&[anchor], &[1]);
                let path = model.propose(&embedding, &anchor, target_hidden, cache, lm_head)?;
                Ok(path.slice(&[0, 0], &[1, guesses]))
            }
        }
    }

    /// The first generation's logits for the `guesses` positions after the
    /// anchor, `[1, guesses, vocab]`, which a tree verify branches on. `None`
    /// for DFlash 2, whose guesses are a path its selector picks rather than
    /// independent per-position distributions.
    pub fn guess_logits(
        &mut self,
        block_embedding: &Array,
        target_hidden: &Array,
        cache: &mut [KVCache],
        lm_head: &mut dyn FnMut(&Array) -> Result<Array, Exception>,
        guesses: usize,
    ) -> Result<Option<Array>, Exception> {
        let embedding = self.scale_embedding(block_embedding);
        match self {
            Self::V1(model) => v1_logits(
                model,
                &embedding,
                target_hidden,
                cache,
                lm_head,
                guesses as i32,
            )
            .map(Some),
            Self::V2(_) => Ok(None),
        }
    }
}

fn v1_logits(
    model: &mut DFlashDraftModel,
    embedding: &Array,
    target_hidden: &Array,
    cache: &mut [KVCache],
    lm_head: &mut dyn FnMut(&Array) -> Result<Array, Exception>,
    guesses: i32,
) -> Result<Array, Exception> {
    let hidden = model.draft_block(embedding, target_hidden, cache)?;
    let dim = hidden.dim(2);
    let suffix = hidden.slice(&[0, 1, 0], &[1, 1 + guesses, dim]);
    Ok(model.config.scale_logits(lm_head(&suffix)?))
}

/// Parse the draft config in `dir`.
pub fn read_dflash_config(dir: &Path) -> Result<DFlashDraftConfig, Exception> {
    let path = dir.join("config.json");
    let bytes = std::fs::read(&path).map_err(|e| {
        Exception::custom(format!(
            "DFlash draft: failed to read {}: {e}",
            path.display()
        ))
    })?;
    serde_json::from_slice(&bytes).map_err(|e| {
        Exception::custom(format!(
            "DFlash draft: failed to parse {}: {e}",
            path.display()
        ))
    })
}

/// Load the DFlash draft in `dir`, either generation, then optionally
/// quantize it.
///
/// Strict: every checkpoint tensor must fill a parameter and every parameter
/// be filled, or loading fails naming the difference.
pub fn load_dflash_draft(dir: &Path, quant: DFlashDraftQuant) -> Result<DFlashDraft, Exception> {
    let config = read_dflash_config(dir)?;
    let weights = crate::loader::load_weights(dir)
        .map_err(|e| Exception::custom(format!("DFlash draft: weight load failed: {e}")))?;
    let mut draft = if config.is_dflash2() {
        let mut model = DFlash2DraftModel::new(config)?;
        model.load_weights(&weights)?;
        DFlashDraft::V2(model)
    } else {
        let mut model = DFlashDraftModel::new(config)?;
        let report: LoadReport = model.load_weights(&weights)?;
        let expected = model.flatten_params().len();
        if report.loaded != expected || !report.skipped.is_empty() {
            return Err(Exception::custom(format!(
                "DFlash draft {}: filled {} of {expected} parameters; unrecognised: {:?}",
                dir.display(),
                report.loaded,
                report.skipped
            )));
        }
        DFlashDraft::V1(model)
    };
    match quant {
        DFlashDraftQuant::None => {}
        DFlashDraftQuant::Fp8 => match &mut draft {
            DFlashDraft::V1(m) => crate::fp8_utils::quantize_model_linears(m)?,
            DFlashDraft::V2(m) => crate::fp8_utils::quantize_model_linears(m)?,
        },
    }
    Ok(draft)
}
