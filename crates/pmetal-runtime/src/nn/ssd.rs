//! The Mamba-2 selective state space scan, as the chunked state space duality
//! algorithm (Dao and Gu, "Transformers are SSMs: Generalized Models and
//! Efficient Algorithms Through Structured State Space Duality", ICML 2024).
//!
//! Per head, with state `S ∈ ℝ^{P×N}` (`P` the head width, `N` the state
//! size), a scalar decay and groups of heads sharing `B` and `C`:
//!
//! ```text
//! dt_t = clamp(softplus(dt_raw_t + dt_bias), dt_min, dt_max)
//! A    = -exp(A_log)
//! S_t  = exp(dt_t · A) · S_{t-1} + (dt_t · x_t) ⊗ B_t
//! y_t  = S_t · C_t + D · x_t
//! ```
//!
//! [`ssd_scan`] computes a whole sequence in chunks of `Q` tokens: inside a
//! chunk the output is a masked, decayed attention-like matrix product (the
//! "dual" quadratic form, `Q × Q` per head), and between chunks only the
//! `P × N` state is handed on, through one batched matmul over the chunk
//! decays instead of a sequential loop. The cost is linear in the length and
//! the memory is `O(L · Q)` rather than `O(L²)`, which a single pass over the
//! sequence would be. [`ssd_step`] is the recurrence itself, for one token.
//!
//! Both run in float32 whatever the inputs' dtype, take and return the state
//! in float32, and return the output in float32 (the caller normalizes it
//! before casting back, as the reference does). A head `h` of `H` reads group
//! `h / (H / G)` of `B` and `C`: the groups are contiguous runs of heads, the
//! layout the released kernels use. The groups are never expanded to heads,
//! the matmuls broadcast over the heads of a group.
//!
//! Both are built from differentiable ops, so training and the adapters run
//! the same code as inference. The tests hold them to a float64 evaluation of
//! the recurrence above, values and gradients.
//!
//! # Shapes
//!
//! - `x`: `[B, L, H, P]`
//! - `dt`: `[B, L, H]`, before the bias and the softplus
//! - `b`, `c`: `[B, L, G, N]`
//! - `state`: `[B, H, P, N]`

use crate::compat::{Array, Dtype, Exception, nn, ops};

/// The most chunks one pass over a sequence holds. The decay between chunks
/// is a `[chunks + 1, chunks + 1]` matrix per head and the products of
/// decays are formed as differences of running sums, which lose precision as
/// the sums grow; a longer sequence is cut into passes of this many chunks
/// and the state carried from one to the next.
const MAX_CHUNKS_PER_PASS: i32 = 32;

/// The parameters of one Mamba-2 layer's scan.
#[derive(Clone, Copy)]
pub struct SsdParams<'a> {
    /// `log(-A)`, `[H]`.
    pub a_log: &'a Array,
    /// The skip connection, `[H]`.
    pub d: &'a Array,
    /// Added to `dt` before the softplus, `[H]`.
    pub dt_bias: &'a Array,
    /// Floor of the step size after the softplus.
    pub dt_min: f32,
    /// Ceiling of the step size; `f32::INFINITY` for none.
    pub dt_max: f32,
}

fn to_f32(a: &Array) -> Array {
    if a.dtype() == Dtype::Float32 {
        a.clone()
    } else {
        a.as_dtype(Dtype::Float32.as_i32())
    }
}

impl SsdParams<'_> {
    /// `dt = clamp(softplus(dt_raw + dt_bias))`, `[.., H]` in float32.
    fn step_size(&self, dt_raw: &Array) -> Array {
        let dt = nn::softplus(&to_f32(dt_raw).add(&to_f32(self.dt_bias)));
        let lo = (self.dt_min > 0.0).then(|| Array::from_f32(self.dt_min));
        let hi = self
            .dt_max
            .is_finite()
            .then(|| Array::from_f32(self.dt_max));
        if lo.is_none() && hi.is_none() {
            dt
        } else {
            ops::clip(&dt, lo.as_ref(), hi.as_ref())
        }
    }

    /// `A = -exp(A_log)`, `[H]` in float32.
    fn decay_rate(&self) -> Array {
        ops::negative(&ops::exp(&to_f32(self.a_log)))
    }
}

fn check_shapes(x: &Array, dt: &Array, b: &Array, c: &Array) -> Result<(), Exception> {
    let (bsz, l, h) = (x.dim(0), x.dim(1), x.dim(2));
    let g = b.dim(2);
    let ok = x.ndim() == 4
        && b.ndim() == 4
        && b.shape() == c.shape()
        && dt.shape() == [bsz, l, h]
        && b.dim(0) == bsz
        && b.dim(1) == l
        && g >= 1
        && h % g == 0;
    if ok {
        Ok(())
    } else {
        Err(Exception::custom(format!(
            "ssd: x {:?}, dt {:?}, B {:?}, C {:?} are not [B, L, H, P], [B, L, H], \
             [B, L, G, N] twice with G dividing H",
            x.shape(),
            dt.shape(),
            b.shape(),
            c.shape()
        )))
    }
}

/// One token of the recurrence: `x` `[B, 1, H, P]`, `dt` `[B, 1, H]`, `b` and
/// `c` `[B, 1, G, N]` and the previous `state` `[B, H, P, N]`, to the output
/// `[B, 1, H, P]` and the new state, both float32.
pub fn ssd_step(
    x: &Array,
    dt: &Array,
    b: &Array,
    c: &Array,
    params: SsdParams<'_>,
    state: &Array,
) -> Result<(Array, Array), Exception> {
    check_shapes(x, dt, b, c)?;
    if x.dim(1) != 1 {
        return Err(Exception::custom("ssd_step takes exactly one token"));
    }
    let (bsz, h, p) = (x.dim(0), x.dim(2), x.dim(3));
    let (g, n) = (b.dim(2), b.dim(3));
    let r = h / g;

    let x = to_f32(x).reshape(&[bsz, g, r, p]);
    let dt = params.step_size(&dt.reshape(&[bsz, h])); // [B, H]
    let decay = ops::exp(&dt.multiply(&params.decay_rate())); // [B, H]

    let xdt = x.multiply(&dt.reshape(&[bsz, g, r, 1])); // [B, G, r, P]
    let b = to_f32(b).reshape(&[bsz, g, 1, 1, n]);
    let c = to_f32(c).reshape(&[bsz, g, 1, n, 1]);

    let state = to_f32(state).reshape(&[bsz, g, r, p, n]);
    let state = state
        .multiply(&decay.reshape(&[bsz, g, r, 1, 1]))
        .add(&xdt.reshape(&[bsz, g, r, p, 1]).multiply(&b));

    // [B, G, r, P, N] @ [B, G, 1, N, 1] -> [B, G, r, P, 1]
    let y = ops::matmul(&state, &c).reshape(&[bsz, g, r, p]);
    let skip = x.multiply(&to_f32(params.d).reshape(&[g, r, 1]));
    let y = y.add(&skip).reshape(&[bsz, 1, h, p]);
    Ok((y, state.reshape(&[bsz, h, p, n])))
}

/// A sequence through the recurrence, in chunks of `chunk_size` tokens (see
/// the module docs): `x` `[B, L, H, P]`, `dt` `[B, L, H]`, `b` and `c`
/// `[B, L, G, N]`, from `state` `[B, H, P, N]` (zeros when `None`). Returns
/// the output `[B, L, H, P]` and the state after the last token, `[B, H, P,
/// N]`, both float32.
///
/// Every chunk size gives the same result up to rounding; it sets only how
/// much of the work is the dense `Q × Q` form. A sequence shorter than the
/// chunk is one chunk of its own length.
pub fn ssd_scan(
    x: &Array,
    dt: &Array,
    b: &Array,
    c: &Array,
    params: SsdParams<'_>,
    state: Option<&Array>,
    chunk_size: i32,
) -> Result<(Array, Array), Exception> {
    scan(x, dt, b, c, params, state, chunk_size, None)
}

/// [`ssd_scan`] over several sequences packed into one row: `segments`
/// `[B, L]` (integers) numbers the sequence each token belongs to, a
/// sequence being a run of equal numbers. The state is reset where the number
/// changes, so a token sees none of the tokens of the sequences before its
/// own, as if each had been scanned alone; `state`, when given, is the state
/// the row's first sequence continues from, and the state returned is the one
/// after the row's last token.
///
/// The attention mask of a packed row cannot reach a recurrent layer, which
/// is why one needs this.
#[expect(
    clippy::too_many_arguments,
    reason = "the recurrence's inputs, one per tensor"
)]
pub fn ssd_scan_segmented(
    x: &Array,
    dt: &Array,
    b: &Array,
    c: &Array,
    params: SsdParams<'_>,
    state: Option<&Array>,
    chunk_size: i32,
    segments: &Array,
) -> Result<(Array, Array), Exception> {
    if segments.shape() != [x.dim(0), x.dim(1)] {
        return Err(Exception::custom(format!(
            "ssd_scan_segmented: segments {:?} for x {:?}",
            segments.shape(),
            x.shape()
        )));
    }
    scan(
        x,
        dt,
        b,
        c,
        params,
        state,
        chunk_size,
        Some(&segments.as_dtype(Dtype::Int32.as_i32())),
    )
}

#[allow(clippy::too_many_arguments)]
fn scan(
    x: &Array,
    dt: &Array,
    b: &Array,
    c: &Array,
    params: SsdParams<'_>,
    state: Option<&Array>,
    chunk_size: i32,
    segments: Option<&Array>,
) -> Result<(Array, Array), Exception> {
    check_shapes(x, dt, b, c)?;
    let (bsz, l, h, p) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3));
    let (g, n) = (b.dim(2), b.dim(3));
    if l == 0 || chunk_size < 1 {
        return Err(Exception::custom(format!(
            "ssd_scan: {l} tokens in chunks of {chunk_size}"
        )));
    }

    let x = to_f32(x);
    let dt = params.step_size(dt); // [B, L, H]
    let da = dt.multiply(&params.decay_rate()); // [B, L, H]
    let xdt = x.multiply(&dt.reshape(&[bsz, l, h, 1]));
    let (b, c) = (to_f32(b), to_f32(c));

    let chunk = chunk_size.min(l);
    let pass_len = chunk * MAX_CHUNKS_PER_PASS;
    let mut state = state.map(to_f32);
    let mut outputs = Vec::new();
    let mut start = 0;
    while start < l {
        let end = (start + pass_len).min(l);
        let piece = |a: &Array| ops::slice_axis(a, 1, start, end);
        // The state a pass starts from belongs to the sequence of the token
        // before it (the row's first, for the first pass).
        let pass_segments = segments.map(|s| PassSegments {
            ids: piece(s),
            carried: ops::slice_axis(s, 1, (start - 1).max(0), start.max(1)).reshape(&[bsz]),
        });
        let (y, next) = ssd_pass(
            &piece(&xdt),
            &piece(&da),
            &piece(&b),
            &piece(&c),
            state.as_ref(),
            chunk,
            (bsz, h, p, g, n),
            pass_segments.as_ref(),
        )?;
        outputs.push(y);
        state = Some(next);
        start = end;
    }
    let y = if outputs.len() == 1 {
        outputs.pop().expect("one pass")
    } else {
        let refs: Vec<&Array> = outputs.iter().collect();
        ops::concatenate_axis(&refs, 1)
    };
    // The skip connection reads the input itself, not the discretized one.
    let skip = x.multiply(&to_f32(params.d).reshape(&[1, 1, h, 1]));
    let state = state.expect("at least one pass");
    Ok((y.add(&skip), state))
}

/// The sequence numbers of one pass's tokens, and the number of the sequence
/// the state the pass starts from belongs to.
struct PassSegments {
    /// `[B, L]`, `int32`.
    ids: Array,
    /// `[B]`, `int32`.
    carried: Array,
}

/// Which pairs of positions, chunks and states may exchange state, as 0/1
/// factors that broadcast against the pass's `[B, G, r, ...]` tensors.
struct SegmentMasks {
    /// Position `j` feeds position `i` of its chunk: `[B, 1, 1, chunks, Q, Q]`.
    within: Array,
    /// Position `j` is in the sequence its chunk ends in, so it reaches the
    /// state the chunk hands on: `[B, 1, 1, chunks, Q]`.
    to_end: Array,
    /// Entry `j` of the states `[carried, chunk 0, chunk 1, ...]` still
    /// reaches the state after entry `i`: `[B, 1, 1, steps, steps]`.
    between: Array,
    /// Position `i` is in the sequence of the state entering its chunk:
    /// `[B, 1, 1, chunks, Q, 1]`.
    from_state: Array,
}

impl SegmentMasks {
    fn new(segments: &PassSegments, chunks: i32, q: i32, pad: i32) -> Self {
        let f32 = Dtype::Float32.as_i32();
        let bsz = segments.ids.dim(0);
        let l = segments.ids.dim(1);
        // The padding at the end continues the last sequence: it is inert.
        let ids = if pad == 0 {
            segments.ids.clone()
        } else {
            let tail = ops::slice_axis(&segments.ids, 1, l - 1, l).broadcast_to(&[bsz, pad]);
            ops::concatenate_axis(&[&segments.ids, &tail], 1)
        }
        .reshape(&[bsz, chunks, q]);

        let within = ops::expand_dims(&ids, -1)
            .equal(&ops::expand_dims(&ids, -2))
            .as_dtype(f32); // [B, chunks, Q, Q]
        let last = ops::slice_axis(&ids, 2, q - 1, q); // [B, chunks, 1]
        let to_end = ids.equal(&last).as_dtype(f32); // [B, chunks, Q]

        // The sequence of each state in `[carried, chunk 0, chunk 1, ...]`.
        let owners = ops::concatenate_axis(
            &[
                &segments.carried.reshape(&[bsz, 1]),
                &last.reshape(&[bsz, chunks]),
            ],
            1,
        ); // [B, steps]
        let between = ops::expand_dims(&owners, -1)
            .equal(&ops::expand_dims(&owners, -2))
            .as_dtype(f32); // [B, steps, steps]
        let entering = ops::slice_axis(&owners, 1, 0, chunks); // [B, chunks]
        let from_state = ids.equal(&ops::expand_dims(&entering, -1)).as_dtype(f32); // [B, chunks, Q]

        Self {
            within: within.reshape(&[bsz, 1, 1, chunks, q, q]),
            to_end: to_end.reshape(&[bsz, 1, 1, chunks, q]),
            between: between.reshape(&[bsz, 1, 1, chunks + 1, chunks + 1]),
            from_state: from_state.reshape(&[bsz, 1, 1, chunks, q, 1]),
        }
    }
}

/// One pass of at most [`MAX_CHUNKS_PER_PASS`] chunks, without the skip
/// connection: `xdt` is `x · dt`, `da` is `dt · A`. The head axis is held as
/// `[G, r]` (group, head within the group) throughout, so `B` and `C` stay
/// per group and broadcast over `r`.
#[allow(clippy::too_many_arguments)]
fn ssd_pass(
    xdt: &Array,
    da: &Array,
    b: &Array,
    c: &Array,
    state: Option<&Array>,
    chunk: i32,
    (bsz, h, p, g, n): (i32, i32, i32, i32, i32),
    segments: Option<&PassSegments>,
) -> Result<(Array, Array), Exception> {
    let r = h / g;
    let l = xdt.dim(1);
    let q = chunk.min(l);
    let pad = (q - l % q) % q;
    let chunks = (l + pad) / q;
    let masks = segments.map(|s| SegmentMasks::new(s, chunks, q, pad));

    // Zero padding is inert: no decay (`da = 0`), no input (`xdt = 0`, and
    // `B` of zero), so the state after the padding is the state before it.
    let padded = |a: &Array| {
        if pad == 0 {
            a.clone()
        } else {
            let mut spec = vec![(0, 0); a.ndim() as usize];
            spec[1] = (0, pad);
            ops::pad(a, &spec, None, Some(0.0))
        }
    };
    let (xdt, da, b, c) = (padded(xdt), padded(da), padded(b), padded(c));

    // [B, G, r, chunks, Q, P], [B, G, r, chunks, Q]
    let xdt = xdt
        .reshape(&[bsz, chunks, q, g, r, p])
        .transpose_axes(&[0, 3, 4, 1, 2, 5]);
    let da = da
        .reshape(&[bsz, chunks, q, g, r])
        .transpose_axes(&[0, 3, 4, 1, 2]);
    // [B, G, chunks, Q, N]
    let b = b
        .reshape(&[bsz, chunks, q, g, n])
        .transpose_axes(&[0, 3, 1, 2, 4]);
    let c = c
        .reshape(&[bsz, chunks, q, g, n])
        .transpose_axes(&[0, 3, 1, 2, 4]);
    let b_t = b.transpose_axes(&[0, 1, 2, 4, 3]);

    // The decay from the start of a chunk to each of its positions.
    let acum = da.cumsum(-1); // [B, G, r, chunks, Q]

    // 1. Inside a chunk: y_i = Σ_{j<=i} (C_i · B_j) exp(acum_i - acum_j) xdt_j.
    let lower = Array::tri(q, q, 0, Dtype::Float32.as_i32());
    let zero = Array::from_f32(0.0);
    let masked = lower
        .equal(&zero)
        .where_cond(&Array::from_f32(f32::NEG_INFINITY), &zero);
    let within = ops::exp(
        &ops::expand_dims(&acum, -1)
            .subtract(&ops::expand_dims(&acum, -2))
            .add(&masked),
    ); // [B, G, r, chunks, Q, Q]
    let within = match &masks {
        Some(m) => within.multiply(&m.within),
        None => within,
    };
    let scores = ops::expand_dims(&ops::matmul(&c, &b_t), 2); // [B, G, 1, chunks, Q, Q]
    let y_within = ops::matmul(&scores.multiply(&within), &xdt);

    // 2. What each chunk adds to the state, decayed to the chunk's end:
    //    Σ_j exp(acum_last - acum_j) xdt_j ⊗ B_j.
    let last = acum.slice(&[0, 0, 0, 0, q - 1], &[bsz, g, r, chunks, q]); // [B, G, r, chunks, 1]
    let to_end = ops::exp(&last.subtract(&acum));
    let to_end = match &masks {
        Some(m) => to_end.multiply(&m.to_end),
        None => to_end,
    };
    let weighted = xdt.multiply(&ops::expand_dims(&to_end, -1));
    let added = ops::matmul(
        &weighted.transpose_axes(&[0, 1, 2, 3, 5, 4]),
        &ops::expand_dims(&b, 2),
    ); // [B, G, r, chunks, P, N]

    // 3. The state entering each chunk (and leaving the last): a matmul of
    //    the chunk-to-chunk decays with the states, one row per chunk.
    let carried = match state {
        Some(s) => to_f32(s).reshape(&[bsz, g, r, 1, p, n]),
        None => ops::zeros(&[bsz, g, r, 1, p, n], Dtype::Float32),
    };
    let all =
        ops::concatenate_axis(&[&carried, &added], 3).reshape(&[bsz, g, r, chunks + 1, p * n]);
    let chunk_log = last.squeeze(-1); // [B, G, r, chunks]
    let front = ops::zeros(&[bsz, g, r, 1], Dtype::Float32);
    let running = ops::concatenate_axis(&[&front, &chunk_log], 3).cumsum(-1); // [B, G, r, chunks + 1]
    let steps = chunks + 1;
    let upper = Array::tri(steps, steps, 0, Dtype::Float32.as_i32());
    let upper_mask = upper
        .equal(&zero)
        .where_cond(&Array::from_f32(f32::NEG_INFINITY), &zero);
    let between = ops::exp(
        &ops::expand_dims(&running, -1)
            .subtract(&ops::expand_dims(&running, -2))
            .add(&upper_mask),
    ); // [B, G, r, steps, steps]
    let between = match &masks {
        Some(m) => between.multiply(&m.between),
        None => between,
    };
    let entering = ops::matmul(&between, &all).reshape(&[bsz, g, r, steps, p, n]);
    let into_chunks = entering.slice(&[0, 0, 0, 0, 0, 0], &[bsz, g, r, chunks, p, n]);
    let final_state = entering
        .slice(&[0, 0, 0, chunks, 0, 0], &[bsz, g, r, steps, p, n])
        .reshape(&[bsz, h, p, n]);

    // 4. What the entering state contributes to each position:
    //    exp(acum_i) C_i · S_entering.
    let from_state = ops::matmul(
        &ops::expand_dims(&c, 2),
        &into_chunks.transpose_axes(&[0, 1, 2, 3, 5, 4]),
    ); // [B, G, r, chunks, Q, P]
    let y_state = from_state.multiply(&ops::expand_dims(&ops::exp(&acum), -1));
    let y_state = match &masks {
        Some(m) => y_state.multiply(&m.from_state),
        None => y_state,
    };

    let y = y_within
        .add(&y_state)
        .transpose_axes(&[0, 3, 4, 1, 2, 5])
        .reshape(&[bsz, chunks * q, h, p]);
    let y = if pad == 0 {
        y
    } else {
        ops::slice_axis(&y, 1, 0, l)
    };
    Ok((y, final_state))
}

/// The gated, group-wise RMS norm that closes a Mamba-2 mixer: the scan
/// output `y` is multiplied by `silu(gate)`, each run of `group_size`
/// channels is divided by its own root mean square, and `weight` scales the
/// result (`norm_before_gate = False`). Computed in float32 and returned in
/// `y`'s dtype; `group_size` equal to the width is the ordinary RMS norm over
/// the whole row.
pub fn gated_group_rms_norm(
    y: &Array,
    gate: Option<&Array>,
    weight: &Array,
    eps: f32,
    group_size: i32,
) -> Result<Array, Exception> {
    let width = y.dim(-1);
    if group_size < 1 || width % group_size != 0 || weight.shape() != [width] {
        return Err(Exception::custom(format!(
            "gated_group_rms_norm: width {width}, group size {group_size}, weight {:?}",
            weight.shape()
        )));
    }
    let dtype = y.dtype();
    let mut gated = to_f32(y);
    if let Some(gate) = gate {
        gated = gated.multiply(&nn::silu(&to_f32(gate)));
    }
    let mut dims: Vec<i32> = gated.shape().to_vec();
    dims.pop();
    let rows = dims.clone();
    dims.extend([width / group_size, group_size]);
    let normed = gated.reshape(&dims).rms_norm(None, eps);
    let mut flat = rows;
    flat.push(width);
    let out = normed.reshape(&flat).multiply(&to_f32(weight));
    Ok(if dtype == Dtype::Float32 {
        out
    } else {
        out.as_dtype(dtype.as_i32())
    })
}

#[cfg(test)]
#[path = "ssd_tests.rs"]
mod tests;
