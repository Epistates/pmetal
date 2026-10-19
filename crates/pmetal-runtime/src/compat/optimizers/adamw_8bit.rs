//! AdamW with 8-bit optimizer states (Dettmers et al., *8-bit Optimizers via
//! Block-wise Quantization*, ICLR 2022, arXiv 2110.02861).
//!
//! The two moments of every tensor of at least [`MIN_8BIT_SIZE`] elements are
//! held as one byte each: split into blocks of [`BLOCK`] elements, each block
//! is scaled by its own absolute maximum, and the scaled value is stored as
//! the index of the nearest entry of a 256-entry *dynamic* code, which has
//! the resolution of a floating-point number around zero and so represents
//! both tiny and large moments well. The first moment uses the signed code and
//! the second, which is never negative, the unsigned one. Each step
//! dequantizes, takes the AdamW update in f32 and quantizes the new moments
//! against their new block maxima, in one fused Metal kernel; the
//! states take a quarter of the memory of f32 ones. Smaller tensors, where
//! the saving is nothing and the quantization noise matters most, keep f32
//! moments, as the reference does.
//!
//! The update itself is [`crate::optimizer::AdamW`]'s (`torch.optim.AdamW`);
//! only the storage of the moments differs.

use super::first_order::{f32_of, master_of};
use super::state::{export_masters, import_masters};
use super::{AdamW, Optimizer, OptimizerState, State, StatefulOptimizer, Updatable, apply_updates};
use crate::MetalKernel;
use crate::compat::{Array, Dtype, Exception, FlattenedModuleParam, ModuleParameters};
use crate::optimizer::{AdamWStepScalars, MasterWeight, excluded_from_weight_decay};
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::OnceLock;

/// Elements per quantization block.
pub const BLOCK: usize = 256;

/// Tensors with fewer elements keep f32 moments.
pub const MIN_8BIT_SIZE: usize = 4096;

/// The dynamic quantization code: 256 values in `[-1, 1]` (signed) or
/// `[0, 1]`, sorted, dense near zero and sparse near one.
///
/// Each of the seven exponent ranges `10^(i-6)` spans a share of the
/// codes proportional to the fraction bits it leaves: the value is the mean
/// of consecutive points of a linear grid over `[0.1, 1]` times the range's
/// power of ten. This is `create_dynamic_map(signed, max_exponent_bits=7,
/// total_bits=8)` of the reference implementation.
pub fn dynamic_map(signed: bool) -> [f32; 256] {
    const MAX_EXPONENT_BITS: i32 = 7;
    const NON_SIGN_BITS: i32 = 7;
    let mut data: Vec<f64> = Vec::with_capacity(256);
    let push_range = |exponent: i32, points: usize, data: &mut Vec<f64>| {
        // Means of consecutive points of linspace(0.1, 1, points).
        let step = 0.9 / (points - 1) as f64;
        let scale = 10f64.powi(exponent);
        for j in 0..points - 1 {
            let mean = 0.1 + step * (j as f64 + 0.5);
            data.push(scale * mean);
            if signed {
                data.push(-scale * mean);
            }
        }
    };
    let mut last = 0;
    for i in 0..MAX_EXPONENT_BITS {
        let points = if signed {
            (1usize << (i + NON_SIGN_BITS - MAX_EXPONENT_BITS)) + 1
        } else {
            (1usize << (i + NON_SIGN_BITS - MAX_EXPONENT_BITS + 1)) + 1
        };
        push_range(-(MAX_EXPONENT_BITS - 1) + i, points, &mut data);
        last = i;
    }
    let additional = (1usize << (NON_SIGN_BITS - MAX_EXPONENT_BITS)) - 1;
    if additional > 0 {
        push_range(-(MAX_EXPONENT_BITS - 1) + last, additional + 1, &mut data);
    }
    data.push(0.0);
    data.push(1.0);
    data.resize(256, 0.0);
    data.sort_by(|a, b| a.total_cmp(b));
    let mut out = [0f32; 256];
    for (o, d) in out.iter_mut().zip(data) {
        *o = d as f32;
    }
    out
}

/// The index of the entry of the sorted `code` nearest `x` (the lower one at
/// a tie): the reference the Metal kernel's search is tested against.
pub fn nearest_code(code: &[f32; 256], x: f32) -> u8 {
    let mut best = 0usize;
    for (i, &c) in code.iter().enumerate() {
        if (c - x).abs() < (code[best] - x).abs() {
            best = i;
        }
    }
    best as u8
}

const HEADER: &str = r#"
inline uint nearest_code(const device float* code, float x) {
    int lo = 0;
    int hi = 255;
    while (hi - lo > 1) {
        int mid = (lo + hi) >> 1;
        if (code[mid] <= x) { lo = mid; } else { hi = mid; }
    }
    return (fabs(code[lo] - x) <= fabs(code[hi] - x)) ? uint(lo) : uint(hi);
}
"#;

/// One threadgroup of 256 threads per block of 256 elements.
const SOURCE: &str = r#"
    uint block = threadgroup_position_in_grid.x;
    uint lid = thread_position_in_threadgroup.x;
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint i = block * 256 + lid;
    uint n = p_shape[0];
    bool live = i < n;

    float b1 = hp[0];
    float omb1 = hp[1];
    float b2 = hp[2];
    float omb2 = hp[3];
    float decay = hp[4];
    float step_size = hp[5];
    float bc2_sqrt = hp[6];
    float eps = hp[7];

    float gv = live ? g[i] : 0.0f;
    float m = live ? code_m[mq[i]] * m_absmax[block] : 0.0f;
    float v = live ? code_v[vq[i]] * v_absmax[block] : 0.0f;
    m = b1 * m + omb1 * gv;
    v = b2 * v + omb2 * gv * gv;
    if (live) {
        p_out[i] = p[i] * decay - step_size * m / (sqrt(v) / bc2_sqrt + eps);
    }

    threadgroup float red_m[8];
    threadgroup float red_v[8];
    float am = simd_max(fabs(m));
    float av = simd_max(v);
    if (lane == 0) {
        red_m[sg] = am;
        red_v[sg] = av;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float bm = red_m[0];
    float bv = red_v[0];
    for (int k = 1; k < 8; k++) {
        bm = max(bm, red_m[k]);
        bv = max(bv, red_v[k]);
    }
    if (live) {
        mq_out[i] = uint8_t(nearest_code(code_m, bm > 0.0f ? m / bm : 0.0f));
        vq_out[i] = uint8_t(nearest_code(code_v, bv > 0.0f ? v / bv : 0.0f));
    }
    if (lid == 0) {
        m_absmax_out[block] = bm;
        v_absmax_out[block] = bv;
    }
"#;

fn kernel() -> &'static MetalKernel {
    static KERNEL: OnceLock<MetalKernel> = OnceLock::new();
    KERNEL.get_or_init(|| {
        MetalKernel::new(
            "pmetal_adamw_8bit_blockwise",
            &[
                "p", "g", "mq", "vq", "m_absmax", "v_absmax", "code_m", "code_v", "hp",
            ],
            &["p_out", "mq_out", "vq_out", "m_absmax_out", "v_absmax_out"],
            SOURCE,
            HEADER,
        )
        .expect("the 8-bit AdamW kernel's source has no NUL byte")
    })
}

/// A tensor's quantized moments.
#[derive(Debug)]
struct Quantized {
    /// First moment, one code index per element.
    mq: Array,
    /// Second moment, one code index per element.
    vq: Array,
    /// Per-block absolute maximum of the first moment.
    m_absmax: Array,
    /// Per-block maximum of the second moment.
    v_absmax: Array,
}

/// AdamW with blockwise 8-bit moments.
pub struct AdamW8bit {
    lr: f32,
    weight_decay: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    min_8bit_size: usize,
    step: u64,
    /// f32 AdamW for the tensors too small to quantize.
    small: AdamW,
    quantized: State<Quantized>,
    masters: State<MasterWeight>,
    code_m: Array,
    code_v: Array,
}

impl std::fmt::Debug for AdamW8bit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdamW8bit")
            .field("lr", &self.lr)
            .field("step", &self.step)
            .field("quantized", &self.quantized.len())
            .finish_non_exhaustive()
    }
}

impl AdamW8bit {
    /// 8-bit AdamW with `torch.optim.AdamW`'s defaults (β = (0.9, 0.999),
    /// ε = 1e-8) and the reference's 4096-element threshold.
    pub fn new(lr: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            weight_decay,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            min_8bit_size: MIN_8BIT_SIZE,
            step: 0,
            small: AdamW::new(lr, weight_decay),
            quantized: HashMap::new(),
            masters: HashMap::new(),
            code_m: Array::from_f32_slice(&dynamic_map(true), &[256]),
            code_v: Array::from_f32_slice(&dynamic_map(false), &[256]),
        }
    }

    /// Quantize the moments of tensors of at least `size` elements.
    pub fn with_min_8bit_size(mut self, size: usize) -> Self {
        self.min_8bit_size = size;
        self
    }

    /// The learning rate the next update uses.
    pub fn lr(&self) -> f32 {
        self.lr
    }

    /// Set the learning rate the next update uses.
    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
        self.small.set_lr(lr);
    }

    /// Advance the step counter. Once per optimizer step, before the
    /// per-parameter calls; [`Optimizer::update`] does it itself.
    pub fn advance_step(&mut self) {
        self.step += 1;
        self.small.advance_step();
    }

    /// Steps taken.
    pub fn step_count(&self) -> u64 {
        self.step
    }

    /// Tensors whose moments are held in 8 bits.
    pub fn num_quantized(&self) -> usize {
        self.quantized.len()
    }

    /// Bytes the quantized moments occupy.
    pub fn quantized_state_bytes(&self) -> usize {
        self.quantized
            .values()
            .map(|q| q.mq.nbytes() + q.vq.nbytes() + q.m_absmax.nbytes() + q.v_absmax.nbytes())
            .sum()
    }

    /// Number of tensors the optimizer holds state for.
    pub fn num_params(&self) -> usize {
        self.quantized.len() + self.small.state.len()
    }

    fn step_quantized(
        &mut self,
        key: &Rc<str>,
        gradient: &Array,
        parameter: &mut Array,
    ) -> Result<(), Exception> {
        let n = parameter.size();
        let blocks = n.div_ceil(BLOCK);
        let wd = if excluded_from_weight_decay(key) {
            0.0
        } else {
            self.weight_decay
        };
        let s = AdamWStepScalars::new(
            self.lr,
            wd,
            self.beta1,
            self.beta2,
            self.eps,
            self.step.max(1),
        );
        let hp = Array::from_f32_slice(
            &[
                self.beta1,
                s.one_minus_beta1,
                s.beta2,
                s.one_minus_beta2,
                s.decay,
                s.step_size,
                s.bias_correction2_sqrt,
                s.eps,
            ],
            &[8],
        );
        let f32_ = Dtype::Float32.as_i32();
        let state = self
            .quantized
            .entry(key.clone())
            .or_insert_with(|| Quantized {
                mq: Array::zeros(&[n as i32], Dtype::Uint8.as_i32()),
                vq: Array::zeros(&[n as i32], Dtype::Uint8.as_i32()),
                m_absmax: Array::zeros(&[blocks as i32], f32_),
                v_absmax: Array::zeros(&[blocks as i32], f32_),
            });
        let master = master_of(&mut self.masters, key);
        let shape = parameter.shape().to_vec();
        let weight = master.weight(parameter).reshape(&[-1]);
        let g = f32_of(gradient).reshape(&[-1]);
        let n_i32 = n as i32;
        let b_i32 = blocks as i32;
        let mut out = kernel()
            .call(
                &[
                    &weight,
                    &g,
                    &state.mq,
                    &state.vq,
                    &state.m_absmax,
                    &state.v_absmax,
                    &self.code_m,
                    &self.code_v,
                    &hp,
                ],
                &[
                    (&[n_i32], Dtype::Float32),
                    (&[n_i32], Dtype::Uint8),
                    (&[n_i32], Dtype::Uint8),
                    (&[b_i32], Dtype::Float32),
                    (&[b_i32], Dtype::Float32),
                ],
                [(blocks * BLOCK) as i32, 1, 1],
                [BLOCK as i32, 1, 1],
                &[],
            )
            .map_err(|e| Exception::custom(format!("8-bit AdamW update of `{key}`: {e}")))?
            .into_iter();
        let p_out = out.next().expect("five outputs");
        state.mq = out.next().expect("five outputs");
        state.vq = out.next().expect("five outputs");
        state.m_absmax = out.next().expect("five outputs");
        state.v_absmax = out.next().expect("five outputs");
        master.commit(parameter, p_out.reshape(&shape));
        Ok(())
    }
}

impl Optimizer for AdamW8bit {
    type State = Self;
    fn state(&self) -> &Self::State {
        self
    }
    fn state_mut(&mut self) -> &mut Self::State {
        self
    }
    fn update_single(
        &mut self,
        key: &Rc<str>,
        gradient: &Array,
        parameter: &mut Array,
    ) -> Result<(), Exception> {
        if parameter.size() >= self.min_8bit_size {
            self.step_quantized(key, gradient, parameter)
        } else {
            self.small.update_single(key, gradient, parameter)
        }
    }
    fn update<M: ModuleParameters>(
        &mut self,
        model: &mut M,
        gradients: FlattenedModuleParam,
    ) -> Result<(), Exception> {
        self.advance_step();
        apply_updates(self, model, gradients)
    }
}

impl StatefulOptimizer for AdamW8bit {
    fn export_state(&self) -> OptimizerState {
        let mut out = OptimizerState::new("adamw_8bit");
        out.set_scalar("step", self.step as f64);
        self.small.export_state().nest("small", &mut out);
        for (key, q) in &self.quantized {
            out.set_tensor("mq", key, &q.mq);
            out.set_tensor("vq", key, &q.vq);
            out.set_tensor("m_absmax", key, &q.m_absmax);
            out.set_tensor("v_absmax", key, &q.v_absmax);
        }
        export_masters(&self.masters, &mut out);
        out
    }

    fn import_state(&mut self, state: &OptimizerState) -> Result<(), Exception> {
        state.expect_kind("adamw_8bit")?;
        self.step = state.scalar("step")? as u64;
        self.small.import_state(&state.unnest("small", "adamw"))?;
        self.quantized.clear();
        for (key, mq) in state.slot("mq") {
            let get = |slot: &str| {
                state.tensor(slot, key).cloned().ok_or_else(|| {
                    Exception::custom(format!(
                        "8-bit AdamW state has `mq` for `{key}` but no `{slot}`"
                    ))
                })
            };
            self.quantized.insert(
                Rc::from(key),
                Quantized {
                    mq: mq.clone(),
                    vq: get("vq")?,
                    m_absmax: get("m_absmax")?,
                    v_absmax: get("v_absmax")?,
                },
            );
        }
        self.masters = import_masters(state);
        Ok(())
    }
}

impl Updatable for AdamW8bit {
    fn updatable_states_len(&self) -> usize {
        self.updatable_states().len()
    }
    fn updatable_states(&self) -> Vec<&Array> {
        let mut states = self.small.updatable_states();
        for q in self.quantized.values() {
            states.extend([&q.mq, &q.vq, &q.m_absmax, &q.v_absmax]);
        }
        states
    }
    fn updatable_states_mut(&mut self) -> Vec<&mut Array> {
        let mut states = self.small.updatable_states_mut();
        for q in self.quantized.values_mut() {
            states.extend([&mut q.mq, &mut q.vq, &mut q.m_absmax, &mut q.v_absmax]);
        }
        states
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compat::random;

    fn read(a: &Array) -> Vec<f32> {
        let mut a = a.as_dtype(Dtype::Float32.as_i32());
        a.eval();
        let n = a.size();
        let v = a.to_f32_vec(n).expect("readable");
        crate::check_unobserved_error().expect("no bridge error");
        v
    }

    fn read_u8(a: &Array) -> Vec<u8> {
        let wide = a.as_dtype(Dtype::Uint32.as_i32());
        wide.eval();
        let v = wide.as_slice::<u32>().iter().map(|&x| x as u8).collect();
        crate::check_unobserved_error().expect("no bridge error");
        v
    }

    /// The reference's maps, from the construction in its source (a linear
    /// grid per exponent range, means of consecutive points).
    #[test]
    fn the_dynamic_maps_are_the_reference_codes() {
        let signed = dynamic_map(true);
        let unsigned = dynamic_map(false);
        for (name, map) in [("signed", &signed), ("unsigned", &unsigned)] {
            assert!(map.windows(2).all(|w| w[0] <= w[1]), "{name} is sorted");
            assert_eq!(map[255], 1.0, "{name} ends at one");
            assert!(map.contains(&0.0), "{name} has zero");
        }
        assert_eq!(signed[0], -0.992_968_75);
        assert_eq!(signed[1], -0.978_906_3_f32);
        assert_eq!(signed[255], 1.0);
        // The signed code sums to its extra +1.0 and nothing else: every
        // other entry has its negation.
        assert!((signed.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert_eq!(&unsigned[..3], &[0.0, 3.25e-7, 7.75e-7]);
        assert_eq!(unsigned[254], 0.996_484_4);
        assert!((unsigned.iter().sum::<f32>() - 75.105_26).abs() < 1e-3);
        // Dense near zero: half the unsigned code is below 0.1.
        assert_eq!(unsigned.iter().filter(|&&c| c < 0.1).count(), 127);
    }

    /// The kernel's moments after one step are the quantization of the f64
    /// reference step's moments with the reference's blockwise nearest-code
    /// rule, entry for entry (a one-code disagreement is allowed where the
    /// f32 kernel and the f64 reference round a tie differently), and the
    /// parameter is AdamW's own.
    #[test]
    fn one_step_matches_a_float64_reference() {
        random::seed(91);
        let n = 700; // three blocks, the last partial
        let p0 = read(&random::uniform_range(-1.0, 1.0, &[n], Dtype::Float32));
        // Gradients over several decades, so the code's dynamic range works.
        let g: Vec<f32> = read(&random::uniform_range(-1.0, 1.0, &[n], Dtype::Float32))
            .iter()
            .enumerate()
            .map(|(i, v)| v * 10f32.powi(-((i % 5) as i32)))
            .collect();
        let (lr, wd) = (0.01_f32, 0.1_f32);
        let mut opt = AdamW8bit::new(lr, wd).with_min_8bit_size(256);
        let key: Rc<str> = Rc::from("w");
        let mut p = Array::from_f32_slice(&p0, &[n]);
        opt.advance_step();
        opt.update_single(&key, &Array::from_f32_slice(&g, &[n]), &mut p)
            .expect("update");
        assert_eq!(opt.num_quantized(), 1);

        let (b1, b2, eps) = (f64::from(0.9f32), f64::from(0.999f32), f64::from(1e-8f32));
        let m: Vec<f64> = g.iter().map(|&x| (1.0 - b1) * x as f64).collect();
        let v: Vec<f64> = g.iter().map(|&x| (1.0 - b2) * (x as f64).powi(2)).collect();
        // The parameter: torch.optim.AdamW, step one.
        let got_p = read(&p);
        for i in 0..n as usize {
            let want = p0[i] as f64 * (1.0 - lr as f64 * wd as f64)
                - (lr as f64 / (1.0 - b1)) * m[i] / ((v[i]).sqrt() / (1.0 - b2).sqrt() + eps);
            assert!(
                (got_p[i] as f64 - want).abs() < 1e-5,
                "p[{i}]: {} vs {want}",
                got_p[i]
            );
        }

        let state = opt.export_state();
        let (mq, vq) = (
            read_u8(state.tensor("mq", "w").unwrap()),
            read_u8(state.tensor("vq", "w").unwrap()),
        );
        let (m_abs, v_abs) = (
            read(state.tensor("m_absmax", "w").unwrap()),
            read(state.tensor("v_absmax", "w").unwrap()),
        );
        assert_eq!(m_abs.len(), 3);
        let (code_m, code_v) = (dynamic_map(true), dynamic_map(false));
        for block in 0..3 {
            let range = block * BLOCK..((block + 1) * BLOCK).min(n as usize);
            let bm = m[range.clone()].iter().fold(0f64, |a, x| a.max(x.abs()));
            let bv = v[range.clone()].iter().fold(0f64, |a, x| a.max(*x));
            assert!(
                (m_abs[block] as f64 - bm).abs() <= 1e-6 * bm,
                "m absmax {block}"
            );
            assert!(
                (v_abs[block] as f64 - bv).abs() <= 1e-6 * bv,
                "v absmax {block}"
            );
            for i in range {
                let (nm, nv) = ((m[i] / bm) as f32, (v[i] / bv) as f32);
                let (wm, wv) = (nearest_code(&code_m, nm), nearest_code(&code_v, nv));
                assert!(mq[i].abs_diff(wm) <= 1, "mq[{i}]: {} vs {wm}", mq[i]);
                assert!(vq[i].abs_diff(wv) <= 1, "vq[{i}]: {} vs {wv}", vq[i]);
            }
        }
    }

    /// Over many steps the 8-bit moments track the f32 optimizer's: every
    /// parameter stays within the quantization's error of AdamW's.
    #[test]
    fn follows_float32_adamw_over_many_steps() {
        random::seed(92);
        let n = 8192;
        let p0 = random::uniform_range(-1.0, 1.0, &[n], Dtype::Float32);
        let mut full = AdamW::new(1e-3, 0.01);
        let mut small = AdamW8bit::new(1e-3, 0.01);
        let key: Rc<str> = Rc::from("w");
        let (mut pf, mut pq) = (p0.clone(), p0.clone());
        for _ in 0..40 {
            let g = random::normal(&[n], Dtype::Float32);
            full.advance_step();
            full.update_single(&key, &g, &mut pf).expect("f32");
            small.advance_step();
            small.update_single(&key, &g, &mut pq).expect("8-bit");
            pf.eval();
            pq.eval();
        }
        assert_eq!(small.num_quantized(), 1);
        let (a, b) = (read(&pf), read(&pq));
        let worst = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        let moved = a
            .iter()
            .zip(read(&p0))
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        assert!(moved > 0.02, "the parameters moved ({moved})");
        // The step is lr-sized, and a few percent of it per step is the
        // code's resolution: well under a tenth of the distance moved.
        assert!(
            worst < 0.1 * moved,
            "worst deviation {worst} against {moved} moved"
        );
    }

    /// The states are held at a quarter of f32's size, and a tensor under the
    /// threshold keeps f32 moments that match AdamW's exactly.
    #[test]
    fn quantized_states_take_a_quarter_and_small_tensors_stay_exact() {
        random::seed(93);
        let big: Rc<str> = Rc::from("big");
        let tiny: Rc<str> = Rc::from("tiny");
        let mut opt = AdamW8bit::new(1e-3, 0.0);
        let mut reference = AdamW::new(1e-3, 0.0);
        let mut pb = random::uniform_range(-1.0, 1.0, &[64, 128], Dtype::Float32);
        let mut pt = random::uniform_range(-1.0, 1.0, &[100], Dtype::Float32);
        let mut pt_ref = pt.clone();
        let (gb, gt) = (
            random::normal(&[64, 128], Dtype::Float32),
            random::normal(&[100], Dtype::Float32),
        );
        opt.advance_step();
        opt.update_single(&big, &gb, &mut pb).expect("big");
        opt.update_single(&tiny, &gt, &mut pt).expect("tiny");
        reference.advance_step();
        reference
            .update_single(&tiny, &gt, &mut pt_ref)
            .expect("ref");
        assert_eq!(read(&pt), read(&pt_ref), "a small tensor is plain AdamW");
        assert_eq!(opt.num_quantized(), 1);
        let n = 64 * 128;
        let f32_bytes = 2 * n * 4;
        let held = opt.quantized_state_bytes();
        assert!(held * 3 < f32_bytes, "{held} bytes against {f32_bytes}");
        assert_eq!(pb.shape(), [64, 128], "the parameter keeps its shape");
    }

    /// State out and back in continues bit-exactly, and a bf16 parameter's
    /// f32 master copy is part of it.
    #[test]
    fn resumes_bit_exactly_with_a_bf16_master() {
        random::seed(94);
        let bf16 = Dtype::Bfloat16.as_i32();
        let key: Rc<str> = Rc::from("w");
        let p0 = random::uniform_range(-1.0, 1.0, &[4096], Dtype::Float32).as_dtype(bf16);
        let grads: Vec<Array> = (0..4)
            .map(|_| random::normal(&[4096], Dtype::Float32).mul_scalar(1e-3))
            .collect();
        let run = |opt: &mut AdamW8bit, p: &mut Array, range: std::ops::Range<usize>| {
            for i in range {
                opt.advance_step();
                opt.update_single(&key, &grads[i], p).expect("update");
                p.eval();
            }
        };
        let mut whole = AdamW8bit::new(1e-4, 0.01);
        let mut pw = p0.clone();
        run(&mut whole, &mut pw, 0..4);

        let mut first = AdamW8bit::new(1e-4, 0.01);
        let mut p = p0.clone();
        run(&mut first, &mut p, 0..2);
        let state = first.export_state();
        assert!(state.tensor("master", "w").is_some(), "the master is saved");

        let mut second = AdamW8bit::new(1e-4, 0.01);
        second.import_state(&state).expect("import");
        // A resumed run loads the weights the checkpoint holds: the master
        // rounded to bf16.
        let mut p2 = Array::from_f32_slice(&read(&p), &[4096]).as_dtype(bf16);
        run(&mut second, &mut p2, 2..4);
        assert_eq!(read(&p2), read(&pw));
    }
}
