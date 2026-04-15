//! Mamba SSM state cache for hybrid architectures.

use pmetal_bridge::compat::{Array, Exception, ops};

use crate::array_ext::ArrayDtypeExt;
use crate::kernels::gated_delta_state_advance;

/// Cache for Mamba-2 SSM state during autoregressive generation.
///
/// Mamba layers require two types of state for incremental generation:
/// 1. **Conv state**: Last (kernel_size - 1) conv1d inputs for causal convolution
/// 2. **SSM state**: The hidden state matrix from the state space model
///
/// Without this cache, each generated token is processed without context from
/// previous tokens through Mamba layers, producing incoherent output.
#[derive(Debug, Clone)]
pub struct MambaCache {
    /// Per-layer cache entries.
    /// Each entry is (conv_state, ssm_state) where both may be None initially.
    layers: Vec<MambaCacheEntry>,
}

/// Cache entry for a single Mamba layer.
#[derive(Debug, Clone, Default)]
pub struct MambaCacheEntry {
    /// Convolutional state - last (kernel_size - 1) inputs.
    /// Shape: [batch, kernel_size - 1, conv_dim]
    pub conv_state: Option<Array>,
    /// SSM hidden state.
    /// Shape: [batch, num_heads, head_dim, state_dim]
    pub ssm_state: Option<Array>,
}

/// Snapshot of a single Mamba/GDN layer's state, captured before a
/// speculative verify step.
///
/// `MambaSnapshot` is an immutable, cheap clone of the two state arrays; MLX
/// arrays are ref-counted so cloning here does not copy data, but any future
/// in-place update on the live cache leaves the snapshot unchanged.
#[derive(Debug, Clone, Default)]
pub struct MambaSnapshot {
    /// Conv state as of the snapshot, shape `[B, kernel-1, conv_dim]`.
    pub conv_state: Option<Array>,
    /// SSM state as of the snapshot, shape `[B, Hv, Dv, Dk]`.
    pub ssm_state: Option<Array>,
}

/// Per-token inputs captured during a speculative verify step so that a
/// rollback can replay the GDN recurrence from a snapshot over only the
/// accepted prefix.
///
/// All arrays should cover exactly the `T_verify` tokens that the verify
/// pass appended; a rollback with `accepted < T_verify` slices them to
/// `accepted` along axis 1.
#[derive(Debug, Clone)]
pub struct GdnVerifyInputs {
    /// Keys used by the verify pass, shape `[B, T_verify, Hk, Dk]`.
    pub keys: Array,
    /// Values used by the verify pass, shape `[B, T_verify, Hv, Dv]`.
    pub values: Array,
    /// Gating decay, shape `[B, T_verify, Hv]` (scalar gating).
    pub g: Array,
    /// Beta gate, shape `[B, T_verify, Hv]`.
    pub beta: Array,
    /// Conv1d input for this verify step (pre-conv), shape
    /// `[B, T_verify, conv_dim]`. Used to reconstruct `conv_state` on rollback.
    pub conv_input: Array,
    /// Conv1d kernel size (needed to size the rolled-back conv state).
    pub conv_kernel_size: usize,
}

impl MambaCache {
    /// Create a new Mamba cache with the specified number of layers.
    pub fn new(num_layers: usize) -> Self {
        let layers = (0..num_layers)
            .map(|_| MambaCacheEntry::default())
            .collect();
        Self { layers }
    }

    /// Number of layers tracked by this cache.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get a mutable reference to a layer's cache entry.
    pub fn get_mut(&mut self, layer_idx: usize) -> Option<&mut MambaCacheEntry> {
        self.layers.get_mut(layer_idx)
    }

    /// Get an immutable reference to a layer's cache entry.
    pub fn get(&self, layer_idx: usize) -> Option<&MambaCacheEntry> {
        self.layers.get(layer_idx)
    }

    /// Reset all cache entries to None.
    pub fn reset(&mut self) {
        for entry in &mut self.layers {
            entry.conv_state = None;
            entry.ssm_state = None;
        }
    }

    /// Check if the cache is empty (no state stored).
    pub fn is_empty(&self) -> bool {
        self.layers
            .iter()
            .all(|e| e.conv_state.is_none() && e.ssm_state.is_none())
    }

    /// Snapshot every layer's current state for a speculative verify step.
    ///
    /// Returns a parallel `Vec<MambaSnapshot>` that can later be replayed via
    /// [`MambaCache::rewind_from_snapshots`]. This is `O(num_layers)` and
    /// performs no data copies — MLX arrays are ref-counted.
    pub fn snapshot(&self) -> Vec<MambaSnapshot> {
        self.layers.iter().map(|entry| entry.snapshot()).collect()
    }

    /// Rewind every layer to the state implied by the accepted prefix of
    /// the verify inputs.
    ///
    /// `snapshots` must have been produced by [`MambaCache::snapshot`] on
    /// this cache prior to the verify pass. `per_layer_inputs[i]` is the
    /// verify inputs captured during the verify forward for layer `i`; it
    /// may be `None` for layers that are not GDN (pure attention layers).
    /// `accepted_tokens` is how many of the drafted tokens were accepted.
    ///
    /// When `accepted_tokens == 0` every GDN layer is restored exactly to its
    /// snapshot. Otherwise each layer's GDN state is replayed forward from
    /// its snapshot through the first `accepted_tokens` entries of its
    /// verify inputs.
    pub fn rewind_from_snapshots(
        &mut self,
        snapshots: &[MambaSnapshot],
        per_layer_inputs: &[Option<GdnVerifyInputs>],
        accepted_tokens: usize,
    ) -> Result<(), Exception> {
        if snapshots.len() != self.layers.len() {
            return Err(Exception::custom(format!(
                "snapshot count {} does not match cache layer count {}",
                snapshots.len(),
                self.layers.len()
            )));
        }
        if per_layer_inputs.len() != self.layers.len() {
            return Err(Exception::custom(format!(
                "verify input count {} does not match cache layer count {}",
                per_layer_inputs.len(),
                self.layers.len()
            )));
        }
        for (idx, entry) in self.layers.iter_mut().enumerate() {
            entry.rewind(&snapshots[idx], per_layer_inputs[idx].as_ref(), accepted_tokens)?;
        }
        Ok(())
    }
}

impl MambaCacheEntry {
    /// Update the conv state with new input, returning the padded input for conv1d.
    ///
    /// This implements causal convolution by:
    /// 1. Concatenating stored state with new input
    /// 2. Storing the last (kernel_size - 1) values for next call
    /// 3. Returning the padded input for conv1d processing
    ///
    /// # Arguments
    /// * `input` - New input tensor [batch, seq_len, conv_dim]
    /// * `kernel_size` - Conv1d kernel size
    ///
    /// # Returns
    /// Padded input [batch, seq_len + kernel_size - 1, conv_dim]
    pub fn update_conv_state(
        &mut self,
        input: &Array,
        kernel_size: i32,
    ) -> Result<Array, Exception> {
        let pad_len = (kernel_size - 1) as usize;
        let shape = input.shape();
        let batch = shape[0] as i32;
        let conv_dim = shape[2] as i32;

        // Get or initialize conv state with matching dtype
        let conv_state = if let Some(ref state) = self.conv_state {
            state.clone()
        } else {
            // Initialize to zeros with shape [batch, pad_len, conv_dim]
            // Match the input dtype to avoid dtype mismatch issues
            ops::zeros(&[batch, pad_len as i32, conv_dim], input.dtype())
        };

        // Concatenate state with new input along sequence dimension
        let padded = ops::concatenate_axis(&[&conv_state, input], 1);

        // Store last (kernel_size - 1) values for next call
        let seq_len = padded.dim(1) as usize;
        let start_idx = seq_len - pad_len;
        let b = padded.dim(0) as usize;
        let cd = padded.dim(2) as usize;
        self.conv_state = Some(padded.slice(
            &[0, start_idx as i32, 0],
            &[b as i32, seq_len as i32, cd as i32],
        ));

        Ok(padded)
    }

    /// Get the current SSM state, returning None if not initialized.
    pub fn get_ssm_state(&self) -> Option<&Array> {
        self.ssm_state.as_ref()
    }

    /// Update the SSM state.
    pub fn set_ssm_state(&mut self, state: Array) {
        self.ssm_state = Some(state);
    }

    /// Clone this entry's current state into an immutable snapshot.
    ///
    /// See [`MambaSnapshot`] — this is the single-layer form of
    /// [`MambaCache::snapshot`].
    pub fn snapshot(&self) -> MambaSnapshot {
        MambaSnapshot {
            conv_state: self.conv_state.clone(),
            ssm_state: self.ssm_state.clone(),
        }
    }

    /// Restore this entry exactly to the given snapshot (no replay).
    pub fn restore(&mut self, snapshot: &MambaSnapshot) {
        self.conv_state = snapshot.conv_state.clone();
        self.ssm_state = snapshot.ssm_state.clone();
    }

    /// Rewind this GDN layer to the state after `accepted_tokens` of the
    /// captured verify inputs, using the pre-verify snapshot as the base.
    ///
    /// * If `verify_inputs` is `None` the layer is restored verbatim —
    ///   appropriate for pure-attention layers that never contributed GDN
    ///   state during verify.
    /// * If `accepted_tokens == 0` the layer is restored verbatim even when
    ///   verify inputs are supplied — no replay is needed.
    /// * Otherwise the SSM state is replayed forward from
    ///   `snapshot.ssm_state` through the first `accepted_tokens` entries of
    ///   `verify_inputs.{keys,values,g,beta}`, and the conv state is rebuilt
    ///   by taking the last `kernel_size - 1` positions of
    ///   `[snapshot.conv_state || verify_inputs.conv_input[..accepted]]`.
    pub fn rewind(
        &mut self,
        snapshot: &MambaSnapshot,
        verify_inputs: Option<&GdnVerifyInputs>,
        accepted_tokens: usize,
    ) -> Result<(), Exception> {
        let Some(inputs) = verify_inputs else {
            self.restore(snapshot);
            return Ok(());
        };
        if accepted_tokens == 0 {
            self.restore(snapshot);
            return Ok(());
        }

        let verify_len = inputs.keys.dim(1) as usize;
        let t = accepted_tokens.min(verify_len);

        // ── SSM state replay ─────────────────────────────────────────────
        let initial_ssm = snapshot.ssm_state.clone().ok_or_else(|| {
            Exception::custom(
                "MambaCacheEntry::rewind: snapshot.ssm_state is None; \
                 cannot replay GDN without an initial state",
            )
        })?;
        let k_slice = slice_leading_time(&inputs.keys, t as i32);
        let v_slice = slice_leading_time(&inputs.values, t as i32);
        let g_slice = slice_leading_time(&inputs.g, t as i32);
        let beta_slice = slice_leading_time(&inputs.beta, t as i32);
        self.ssm_state = Some(gated_delta_state_advance(
            &initial_ssm,
            &k_slice,
            &v_slice,
            &g_slice,
            &beta_slice,
        )?);

        // ── Conv state rewind ────────────────────────────────────────────
        // The live conv_state after verify is the last (kernel-1) rows of
        //   [initial_conv_state || inputs.conv_input].
        // After accepting only the first `t` tokens, the rewound conv_state
        // should be the last (kernel-1) rows of
        //   [initial_conv_state || inputs.conv_input[:, :t]].
        if inputs.conv_kernel_size >= 1 {
            let keep = inputs.conv_kernel_size.saturating_sub(1);
            let conv_input_prefix = slice_leading_time(&inputs.conv_input, t as i32);
            let initial_conv = snapshot
                .conv_state
                .clone()
                .unwrap_or_else(|| zero_conv_state(&inputs.conv_input, keep));
            let extended = ops::concatenate_axis(&[&initial_conv, &conv_input_prefix], 1);
            let b = extended.dim(0);
            let ext_len = extended.dim(1);
            let cd = extended.dim(2);
            let keep_i32 = keep.min(ext_len as usize) as i32;
            let start = (ext_len - keep_i32).max(0);
            self.conv_state = Some(extended.slice(&[0, start, 0], &[b, ext_len, cd]));
        } else {
            self.conv_state = snapshot.conv_state.clone();
        }

        Ok(())
    }
}

/// Slice a per-token tensor along axis 1 (time) to the first `t` entries.
fn slice_leading_time(arr: &Array, t: i32) -> Array {
    let rank = arr.shape().len();
    let mut start = vec![0i32; rank];
    let _ = &mut start; // keep zero-initialized
    let mut stop: Vec<i32> = arr.shape().to_vec();
    if rank >= 2 {
        stop[1] = t;
    }
    arr.slice(&start, &stop)
}

/// Construct a zero-initialized conv state matching the dtype/batch/channels
/// of `conv_input` when no prior state existed.
fn zero_conv_state(conv_input: &Array, keep: usize) -> Array {
    let b = conv_input.dim(0);
    let cd = conv_input.dim(2);
    ops::zeros(&[b, keep as i32, cd], conv_input.dtype())
}
