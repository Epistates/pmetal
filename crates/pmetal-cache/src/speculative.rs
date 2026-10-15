//! State a target model's forward pass records for speculative decoding.
//!
//! [`SpecCapture`] carries the hidden states a draft model conditions on and
//! the recurrent-layer inputs a partial-accept rollback replays. The draft
//! and verify loops that use it live in `pmetal_models`: the layer-split
//! `speculative` decoder, the MTP and DFlash decoders, and the seeded draws
//! they share (`generation::SpeculativeDraws`).

use std::collections::BTreeMap;

use pmetal_runtime::compat::{Array, Exception, ops};

use crate::kv_cache::GdnVerifyInputs;

/// Intermediate state captured from a target-model forward pass for use by
/// a speculative draft (e.g. a DFlash block-diffusion drafter) and rollback.
///
/// Two kinds of state live here:
///
/// * **Hidden states** — the activations at a set of tapped layers. The
///   DFlash drafter conditions on these rather than just on logits, so a
///   single target forward can feed both the verifier and the next draft
///   block without a second pass.
///
/// * **GDN verify inputs** — for hybrid-attention models (Qwen3.5) that
///   carry recurrent linear-attention state, each linear layer records the
///   per-token `k,v,g,β,conv_input` it fed through `gated_delta_update`
///   during verify. On a partial-accept rollback, these records are
///   replayed over the accepted prefix to reconstruct the exact SSM state
///   (see [`crate::kv_cache::MambaCacheEntry::rewind`]).
///
/// Layer indices used in both maps are the model's absolute layer index
/// (0-based). A model is free to populate only the hidden-state map if it
/// has no recurrent layers.
#[derive(Debug, Default, Clone)]
pub struct SpecCapture {
    /// Which layer indices the caller wants hidden states captured at.
    /// If empty, no hidden-state capture is performed.
    pub requested_hidden_layers: Vec<usize>,
    /// Captured hidden states, keyed by layer index. Shape `[B, T, hidden]`.
    pub hidden_states: BTreeMap<usize, Array>,
    /// Per-layer GDN verify inputs, keyed by layer index. Layers that are
    /// not linear-attention do not appear here.
    pub gdn_inputs: BTreeMap<usize, GdnVerifyInputs>,
    /// When true, the model should capture the post-embedding hidden state
    /// (after any embedding scale multiply) into [`Self::embedding`]. This
    /// is the earliest tap point and exists so parity tests can localise
    /// drift introduced before the first decoder layer (e.g. a dtype
    /// promotion in the embed-scale multiply). Defaults to `false`.
    pub capture_embedding: bool,
    /// Captured post-embedding hidden state, shape `[B, T, hidden]`.
    /// Only populated when `capture_embedding` is true.
    pub embedding: Option<Array>,
}

impl SpecCapture {
    /// Construct a capture buffer that wants hidden states at the given
    /// layer indices. Duplicate indices are removed.
    pub fn with_layers(mut layer_ids: Vec<usize>) -> Self {
        layer_ids.sort();
        layer_ids.dedup();
        Self {
            requested_hidden_layers: layer_ids,
            hidden_states: BTreeMap::new(),
            gdn_inputs: BTreeMap::new(),
            capture_embedding: false,
            embedding: None,
        }
    }

    /// Construct a capture buffer that additionally taps the post-embedding
    /// hidden state. Equivalent to [`Self::with_layers`] followed by setting
    /// `capture_embedding = true`.
    pub fn with_layers_and_embedding(layer_ids: Vec<usize>, capture_embedding: bool) -> Self {
        let mut this = Self::with_layers(layer_ids);
        this.capture_embedding = capture_embedding;
        this
    }

    /// Returns true if `layer_idx` is one of the requested hidden-state taps.
    pub fn wants_hidden_for(&self, layer_idx: usize) -> bool {
        self.requested_hidden_layers
            .binary_search(&layer_idx)
            .is_ok()
    }

    /// Returns true if the caller wants the post-embedding tap populated.
    pub fn wants_embedding(&self) -> bool {
        self.capture_embedding
    }

    /// Record a hidden-state tap for `layer_idx`. Callers only need to call
    /// this after checking [`Self::wants_hidden_for`] so the extra clone is
    /// paid only for layers the caller asked about.
    pub fn record_hidden(&mut self, layer_idx: usize, hidden: Array) {
        self.hidden_states.insert(layer_idx, hidden);
    }

    /// Record the post-embedding hidden state. Callers should check
    /// [`Self::wants_embedding`] first so uninterested captures pay no clone.
    pub fn record_embedding(&mut self, hidden: Array) {
        self.embedding = Some(hidden);
    }

    /// Record the GDN verify inputs for `layer_idx`.
    pub fn record_gdn(&mut self, layer_idx: usize, inputs: GdnVerifyInputs) {
        self.gdn_inputs.insert(layer_idx, inputs);
    }

    /// Stack the captured hidden states into `[B, T, L * hidden]` in the
    /// order of [`Self::requested_hidden_layers`]. This is the format a
    /// DFlash drafter expects as its `target_hidden` input.
    ///
    /// Returns `Err` if any requested layer was not captured.
    pub fn stack_hidden(&self) -> Result<Array, Exception> {
        if self.requested_hidden_layers.is_empty() {
            return Err(Exception::custom(
                "SpecCapture::stack_hidden: no hidden layers were requested",
            ));
        }
        let mut tensors: Vec<Array> = Vec::with_capacity(self.requested_hidden_layers.len());
        for idx in &self.requested_hidden_layers {
            match self.hidden_states.get(idx) {
                Some(h) => tensors.push(h.clone()),
                None => {
                    return Err(Exception::custom(format!(
                        "SpecCapture::stack_hidden: layer {idx} was requested but not captured"
                    )));
                }
            }
        }
        let refs: Vec<&Array> = tensors.iter().collect();
        // Concatenate along the hidden dim (last axis).
        Ok(ops::concatenate_axis(&refs, -1))
    }

    /// Clear all captured state so the buffer can be reused for the next
    /// verify step without reallocating the requested-layer list.
    pub fn clear(&mut self) {
        self.hidden_states.clear();
        self.gdn_inputs.clear();
        self.embedding = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hidden(values: &[f32]) -> Array {
        Array::from_f32_slice(values, &[1, 2, values.len() as i32 / 2])
    }

    fn read(a: &Array) -> Vec<f32> {
        let mut a = a.clone();
        a.eval();
        a.to_f32_vec(a.size()).unwrap()
    }

    /// The tap list is sorted and de-duplicated, and a layer is wanted
    /// exactly when it is on it.
    #[test]
    fn the_requested_layers_are_sorted_and_unique() {
        let capture = SpecCapture::with_layers(vec![9, 2, 9, 5, 2]);
        assert_eq!(capture.requested_hidden_layers, vec![2, 5, 9]);
        for layer in 0..12 {
            assert_eq!(
                capture.wants_hidden_for(layer),
                [2, 5, 9].contains(&layer),
                "layer {layer}"
            );
        }
        assert!(!capture.wants_embedding());
        assert!(SpecCapture::with_layers_and_embedding(vec![], true).wants_embedding());
        assert!(!SpecCapture::default().wants_hidden_for(0));
    }

    /// `stack_hidden` concatenates the taps along the hidden axis in the order
    /// of the requested layers, whatever order they were recorded in.
    #[test]
    fn the_taps_stack_along_the_hidden_axis_in_layer_order() {
        let mut capture = SpecCapture::with_layers(vec![7, 3]);
        capture.record_hidden(7, hidden(&[5.0, 6.0, 7.0, 8.0]));
        capture.record_hidden(3, hidden(&[1.0, 2.0, 3.0, 4.0]));
        let stacked = capture.stack_hidden().unwrap();
        assert_eq!(stacked.shape(), &[1, 2, 4]);
        assert_eq!(read(&stacked), vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]);
    }

    #[test]
    fn stacking_names_what_is_missing() {
        let none = SpecCapture::default().stack_hidden().unwrap_err();
        assert!(none.to_string().contains("no hidden layers"), "{none}");

        let mut capture = SpecCapture::with_layers(vec![1, 4]);
        capture.record_hidden(1, hidden(&[0.0, 0.0]));
        let missing = capture.stack_hidden().unwrap_err();
        assert!(missing.to_string().contains("layer 4"), "{missing}");
    }

    /// `clear` empties the recordings and keeps the request, so a buffer is
    /// reused across verify steps.
    #[test]
    fn clearing_keeps_the_request() {
        let mut capture = SpecCapture::with_layers(vec![2]);
        capture.record_hidden(2, hidden(&[1.0, 2.0]));
        capture.record_embedding(hidden(&[3.0, 4.0]));
        capture.clear();
        assert!(capture.hidden_states.is_empty() && capture.embedding.is_none());
        assert_eq!(capture.requested_hidden_layers, vec![2]);
    }
}
