//! Native-bridge [`DFlashTarget`] for Qwen3 / Qwen3.5.
//!
//! Routes the DFlash verify-step target forward through the fused
//! `pmetal_bridge::qwen3_native::forward_step_with_capture` path instead
//! of the dynamic `pmetal-models` forward. Two wins fall out:
//!
//! 1. **Speed** — the native path matches mlx-lm's parallel-replay
//!    verify forward in kernel selection and per-op dispatch, so the
//!    target-forward cost per verify step drops to parity with mlx-lm.
//! 2. **Numerical parity with upstream** — the native bridge uses the
//!    same fused attention + MLP kernels as mlx-lm for Qwen3, which is
//!    what the DFlash draft was trained against. Without this, the
//!    draft's cross-attention sees subtly different target hidden
//!    states and acceptance rate drops.
//!
//! Only Qwen3 / Qwen3.5 are wired today. Other architectures continue
//! to use the dynamic `DynamicModel` path. The dispatch lives in
//! `pmetal/src/commands/dflash.rs`.

use std::path::Path;

use pmetal_bridge::qwen3_native::{self, NativeCache, NativeWeights};
use pmetal_mlx::kv_cache::{KVCache, MambaCache};
use pmetal_mlx::speculative::SpecCapture;
use pmetal_mlx::{Array, Exception};

use crate::dflash_decoder::DFlashTarget;

/// DFlash target backed by the fused native-bridge Qwen3 forward.
pub struct NativeQwen3Target {
    weights: NativeWeights,
    cache: NativeCache,
    hidden_size: i32,
    num_layers: usize,
    num_kv_heads: i32,
    head_dim: i32,
}

impl NativeQwen3Target {
    /// Load weights from `model_dir` into the fused native bridge.
    /// Returns a target that owns its own `NativeCache`.
    ///
    /// Fails cleanly if the model is not a supported Qwen3 / Qwen3.5
    /// checkpoint. Callers should fall back to the dynamic path on
    /// error.
    pub fn load(model_dir: &Path) -> Result<Self, String> {
        let config = qwen3_native::load_config(model_dir)
            .map_err(|e| format!("qwen3_native::load_config: {e}"))?;
        let weights = qwen3_native::load_model(model_dir, &config)
            .map_err(|e| format!("qwen3_native::load_model: {e}"))?;
        let cache = NativeCache::new_empty(&weights);
        let hidden_size = config.hidden_size;
        let num_layers = config.num_hidden_layers as usize;
        // For pure-attention Qwen3 all layers have the same num_kv_heads.
        // Qwen3.5 is hybrid (GDN + attention); we report the attention
        // layers' head config, which is what the dummy external KV cache
        // sizing wants.
        let num_kv_heads = config
            .num_key_value_heads
            .unwrap_or(config.num_attention_heads);
        let head_dim = config.head_dim.unwrap_or(config.hidden_size / config.num_attention_heads);
        Ok(Self {
            weights,
            cache,
            hidden_size,
            num_layers,
            num_kv_heads,
            head_dim,
        })
    }
}

impl DFlashTarget for NativeQwen3Target {
    fn embed_tokens(&mut self, input_ids: &Array) -> Result<Array, Exception> {
        // Quantized embeddings dequantize inside the native forward; for
        // DFlash's `noise_embedding = target.embed_tokens(block_input)`
        // call we replicate the same path the forward takes.
        if let (Some(scales), Some(biases)) = (
            self.weights.embed_scales.as_ref(),
            self.weights.embed_biases.as_ref(),
        ) {
            let qcfg = self.weights.quantization_config.as_ref();
            let gs = qcfg.map(|q| q.group_size).unwrap_or(64);
            let bits = qcfg.map(|q| q.bits).unwrap_or(4);
            let w_rows = self.weights.embed_w.take_axis(input_ids, 0);
            let s_rows = scales.take_axis(input_ids, 0);
            let b_rows = biases.take_axis(input_ids, 0);
            Ok(w_rows.dequantize(&s_rows, &b_rows, gs, bits))
        } else {
            Ok(self.weights.embed_w.take_axis(input_ids, 0))
        }
    }

    fn forward_with_capture(
        &mut self,
        input_ids: &Array,
        _mask: Option<&Array>,
        _kv_cache: Option<&mut KVCache>,
        _mamba_cache: Option<&mut MambaCache>,
        capture: &mut SpecCapture,
    ) -> Result<Array, Exception> {
        // Snapshot the requested layers the decoder is tapping. The
        // native forward writes them in ascending layer order; we then
        // record each into the SpecCapture at the same key.
        let tap_layers: Vec<usize> = capture.requested_hidden_layers.clone();
        let mut captured: Vec<Array> = Vec::with_capacity(tap_layers.len());
        let logits = qwen3_native::forward_step_with_capture(
            &self.weights,
            input_ids,
            &mut self.cache,
            &tap_layers,
            &mut captured,
        );
        if captured.len() != tap_layers.len() {
            return Err(Exception::custom(format!(
                "NativeQwen3Target: forward_step_with_capture returned {} hidden states, \
                 expected {} (taps={:?})",
                captured.len(),
                tap_layers.len(),
                tap_layers
            )));
        }
        // Diagnostic: dump last-position values at each tapped layer so
        // we can compare to upstream mlx-lm's per-layer output and
        // localize the DFlash acceptance-gap divergence. Only active
        // when `PMETAL_DFLASH_TAP_TRACE` is set; the f32 copy costs a
        // few microseconds but is only taken once the var is present.
        if std::env::var_os("PMETAL_DFLASH_TAP_TRACE").is_some() {
            use pmetal_bridge::compat::Dtype;
            // Dump per-position tap values so we can diff against
            // upstream iter-by-iter. When the tap sequence is short
            // (verify block or accepted-prefix slice) we print every
            // position; for a full prompt we only print the last.
            for (idx, h) in tap_layers.iter().zip(captured.iter()) {
                let t = h.dim(1);
                let hidden_dim = h.dim(2);
                let positions: Vec<i32> = if t <= 16 {
                    (0..t).collect()
                } else {
                    vec![t - 1]
                };
                for pos in positions {
                    let slice = h.slice(&[0, pos, 0], &[1, pos + 1, hidden_dim]);
                    let slice_f32 = slice.as_dtype(Dtype::Float32.as_i32());
                    let _ = slice_f32.eval();
                    let data = slice_f32.as_slice::<f32>();
                    let first4: Vec<f32> = data.iter().take(4).copied().collect();
                    let l2: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
                    eprintln!(
                        "[pmetal tap] layer_{idx:02} pos={pos:02} [:4]={:?} ||.||={:.4}",
                        first4.iter().map(|x| (x * 10000.0).round() / 10000.0).collect::<Vec<_>>(),
                        l2
                    );
                }
            }
        }
        for (idx, h) in tap_layers.iter().zip(captured.into_iter()) {
            capture.record_hidden(*idx, h);
        }
        Ok(logits)
    }

    fn lm_head_project(&mut self, hidden: &Array) -> Result<Array, Exception> {
        if self.weights.tie_word_embeddings {
            if let (Some(scales), Some(biases)) = (
                self.weights.embed_scales.as_ref(),
                self.weights.embed_biases.as_ref(),
            ) {
                let qcfg = self.weights.quantization_config.as_ref();
                let gs = qcfg.map(|q| q.group_size).unwrap_or(64);
                let bits = qcfg.map(|q| q.bits).unwrap_or(4);
                Ok(hidden.quantized_matmul(
                    &self.weights.embed_w,
                    scales,
                    Some(biases),
                    true,
                    gs,
                    bits,
                ))
            } else {
                Ok(hidden.matmul(&self.weights.embed_w.t()))
            }
        } else {
            let lm_head = self.weights.lm_head_w.as_ref().ok_or_else(|| {
                Exception::custom(
                    "NativeQwen3Target: untied model has no lm_head_w loaded".to_string(),
                )
            })?;
            Ok(lm_head.matmul_from(hidden))
        }
    }

    fn target_hidden_size(&self) -> i32 {
        self.hidden_size
    }

    fn target_num_layers(&self) -> usize {
        self.num_layers
    }

    fn target_num_kv_heads(&self) -> i32 {
        self.num_kv_heads
    }

    fn target_head_dim(&self) -> i32 {
        self.head_dim
    }

    /// The external KV cache from [`DFlashTarget::make_kv_cache`] is a
    /// placeholder that the decode loop never writes to. Rewind our
    /// internal `NativeCache` instead.
    fn rollback_rejected(&mut self, _kv_cache: &mut KVCache, n: usize) {
        qwen3_native::rollback_cache(&mut self.cache, n as i32);
    }

    fn supports_tree_verify(&self) -> bool {
        true
    }

    fn forward_tree_verify(
        &mut self,
        input_ids: &Array,
        position_ids: &Array,
        attention_mask: &Array,
        capture: &mut SpecCapture,
    ) -> Result<Array, Exception> {
        let tap_layers: Vec<usize> = capture.requested_hidden_layers.clone();
        let mut captured: Vec<Array> = Vec::with_capacity(tap_layers.len());
        let logits = qwen3_native::forward_step_tree_verify(
            &self.weights,
            input_ids,
            &mut self.cache,
            position_ids,
            attention_mask,
            &tap_layers,
            &mut captured,
        );
        if captured.len() != tap_layers.len() {
            return Err(Exception::custom(format!(
                "NativeQwen3Target: forward_step_tree_verify returned {} hidden states, \
                 expected {} (taps={:?})",
                captured.len(),
                tap_layers.len(),
                tap_layers
            )));
        }
        for (idx, h) in tap_layers.iter().zip(captured.into_iter()) {
            capture.record_hidden(*idx, h);
        }
        Ok(logits)
    }

    fn compact_tree_cache(
        &mut self,
        past_length: usize,
        tree_length: usize,
        accepted_indices: &[usize],
    ) {
        qwen3_native::compact_tree_cache(
            &mut self.cache,
            past_length as i32,
            tree_length as i32,
            accepted_indices,
        );
    }
}
