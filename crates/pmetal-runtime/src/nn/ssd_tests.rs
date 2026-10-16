//! The Mamba-2 recurrence in float64 and the checks that hold the chunked
//! scan, the single step and the gated norm to it: values and gradients,
//! across chunk sizes, padding, groups, a carried state and several passes.
//!
//! The recurrence (Dao and Gu, ICML 2024), per head and group `g(h)`:
//!
//! ```text
//! dt_t = clamp(softplus(dt_raw_t + dt_bias), dt_min, dt_max)
//! S_t  = exp(-dt_t · exp(A_log)) · S_{t-1} + (dt_t · x_t) ⊗ B_t
//! y_t  = S_t · C_t + D · x_t
//! ```

use super::*;
use crate::f64_reference::*;
use serial_test::serial;

#[derive(Clone)]
struct Case {
    b: usize,
    l: usize,
    h: usize,
    p: usize,
    g: usize,
    n: usize,
    x: Vec<f64>,
    dt: Vec<f64>,
    bm: Vec<f64>,
    cm: Vec<f64>,
    a_log: Vec<f64>,
    d: Vec<f64>,
    dt_bias: Vec<f64>,
    state: Vec<f64>,
    dt_min: f32,
    dt_max: f32,
}

fn case(
    (b, l, h, p, g, n): (usize, usize, usize, usize, usize, usize),
    seed: u64,
    clamp: (f32, f32),
) -> Case {
    let mut rng = Rng(seed);
    let mut draw = |count: usize, scale: f64, shift: f64| -> Vec<f64> {
        (0..count)
            .map(|_| round32(shift + scale * rng.normal()))
            .collect()
    };
    Case {
        b,
        l,
        h,
        p,
        g,
        n,
        x: draw(b * l * h * p, 1.0, 0.0),
        dt: draw(b * l * h, 1.0, -0.5),
        bm: draw(b * l * g * n, 0.7, 0.0),
        cm: draw(b * l * g * n, 0.7, 0.0),
        a_log: draw(h, 0.4, 0.0),
        d: draw(h, 0.5, 1.0),
        dt_bias: draw(h, 0.3, -0.2),
        state: draw(b * h * p * n, 0.5, 0.0),
        dt_min: clamp.0,
        dt_max: clamp.1,
    }
}

fn softplus(x: f64) -> f64 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

fn step_size(c: &Case, raw: f64, head: usize) -> f64 {
    softplus(raw + c.dt_bias[head])
        .max(f64::from(c.dt_min))
        .min(f64::from(c.dt_max))
}

/// `(y [B,L,H,P], final state [B,H,P,N])`, token by token in float64. `group`
/// maps a head to the group it reads. Where `segments` (`[B, L]`) changes the
/// state is reset to zero, as if each run of equal numbers were scanned alone.
fn reference_with(
    c: &Case,
    state: Option<&[f64]>,
    group: impl Fn(usize) -> usize,
    segments: Option<&[i32]>,
) -> (Vec<f64>, Vec<f64>) {
    let (b, l, h, p, g, n) = (c.b, c.l, c.h, c.p, c.g, c.n);
    let mut s = state.map_or_else(|| vec![0.0; b * h * p * n], <[f64]>::to_vec);
    let mut y = vec![0.0; b * l * h * p];
    for bi in 0..b {
        for hi in 0..h {
            let base = (bi * h + hi) * p * n;
            let gi = group(hi);
            for t in 0..l {
                if let Some(seg) = segments
                    && t > 0
                    && seg[bi * l + t] != seg[bi * l + t - 1]
                {
                    s[base..base + p * n].fill(0.0);
                }
                let dt = step_size(c, c.dt[(bi * l + t) * h + hi], hi);
                let decay = (-dt * c.a_log[hi].exp()).exp();
                let xo = ((bi * l + t) * h + hi) * p;
                let go = ((bi * l + t) * g + gi) * n;
                for pi in 0..p {
                    for ni in 0..n {
                        let cell = &mut s[base + pi * n + ni];
                        *cell = decay * *cell + dt * c.x[xo + pi] * c.bm[go + ni];
                    }
                    let acc: f64 = (0..n).map(|ni| s[base + pi * n + ni] * c.cm[go + ni]).sum();
                    y[xo + pi] = acc + c.d[hi] * c.x[xo + pi];
                }
            }
        }
    }
    (y, s)
}

/// Heads read contiguous groups: head `h` of `H` reads group `h / (H / G)`.
fn reference(c: &Case, state: Option<&[f64]>) -> (Vec<f64>, Vec<f64>) {
    let per = c.h / c.g;
    reference_with(c, state, move |h| h / per, None)
}

/// [`reference`] with the state reset where `segments` changes.
fn reference_segmented(c: &Case, state: Option<&[f64]>, segments: &[i32]) -> (Vec<f64>, Vec<f64>) {
    let per = c.h / c.g;
    reference_with(c, state, move |h| h / per, Some(segments))
}

struct Fields {
    x: Array,
    dt: Array,
    b: Array,
    c: Array,
    a_log: Array,
    d: Array,
    dt_bias: Array,
    state: Array,
}

fn fields(c: &Case) -> Fields {
    Fields {
        x: arr(&c.x, &[c.b, c.l, c.h, c.p]),
        dt: arr(&c.dt, &[c.b, c.l, c.h]),
        b: arr(&c.bm, &[c.b, c.l, c.g, c.n]),
        c: arr(&c.cm, &[c.b, c.l, c.g, c.n]),
        a_log: arr(&c.a_log, &[c.h]),
        d: arr(&c.d, &[c.h]),
        dt_bias: arr(&c.dt_bias, &[c.h]),
        state: arr(&c.state, &[c.b, c.h, c.p, c.n]),
    }
}

fn scan(c: &Case, f: &Fields, with_state: bool, chunk: i32) -> (Array, Array) {
    ssd_scan(
        &f.x,
        &f.dt,
        &f.b,
        &f.c,
        SsdParams {
            a_log: &f.a_log,
            d: &f.d,
            dt_bias: &f.dt_bias,
            dt_min: c.dt_min,
            dt_max: c.dt_max,
        },
        with_state.then_some(&f.state),
        chunk,
    )
    .unwrap()
}

fn check_scan(what: &str, c: &Case, with_state: bool, chunk: i32) {
    let f = fields(c);
    let (y, s) = scan(c, &f, with_state, chunk);
    drain();
    let (ry, rs) = reference(c, with_state.then_some(c.state.as_slice()));
    assert_close(&format!("{what}: output"), &vec_of(&y), &ry, 2e-5);
    assert_close(&format!("{what}: final state"), &vec_of(&s), &rs, 2e-5);
}

#[test]
#[serial]
fn the_chunked_scan_matches_the_recurrence_for_every_chunking() {
    let c = case((2, 23, 4, 3, 2, 5), 1, (0.0, f32::INFINITY));
    // Smaller than the sequence with and without a remainder, equal to it,
    // larger than it, and one token per chunk.
    for chunk in [1, 3, 4, 8, 23, 64] {
        check_scan(&format!("chunk {chunk}"), &c, false, chunk);
        check_scan(&format!("chunk {chunk} from a state"), &c, true, chunk);
    }
}

#[test]
#[serial]
fn a_group_count_of_one_and_of_every_head_both_hold() {
    for groups in [1, 6] {
        let c = case((1, 19, 6, 4, groups, 4), 2, (0.0, f32::INFINITY));
        check_scan(&format!("{groups} groups"), &c, true, 5);
    }
}

#[test]
#[serial]
fn a_sequence_longer_than_one_pass_carries_its_state_across_passes() {
    // Two-token chunks make 35 chunks, past MAX_CHUNKS_PER_PASS.
    let c = case((1, 70, 2, 3, 1, 3), 3, (0.0, f32::INFINITY));
    assert!(c.l > 2 * MAX_CHUNKS_PER_PASS as usize);
    check_scan("two passes", &c, true, 2);
}

#[test]
#[serial]
fn the_step_size_limits_apply_and_are_visible_in_the_fixture() {
    // A floor above most steps and a ceiling below the rest.
    let clamped = case((1, 16, 4, 3, 2, 4), 4, (0.35, 0.9));
    check_scan("clamped", &clamped, true, 6);
    let free = Case {
        dt_min: 0.0,
        dt_max: f32::INFINITY,
        ..clamped.clone()
    };
    let (a, _) = reference(&clamped, None);
    let (b, _) = reference(&free, None);
    let moved = a
        .iter()
        .zip(&b)
        .map(|(p, q)| (p - q).abs())
        .fold(0.0, f64::max);
    assert!(moved > 0.1, "the limits change nothing here ({moved})");
}

#[test]
#[serial]
fn heads_read_contiguous_groups_and_the_fixture_can_tell() {
    let c = case((1, 12, 4, 3, 2, 4), 5, (0.0, f32::INFINITY));
    // Tiled instead: head h reads group h mod G.
    let (contiguous, _) = reference(&c, None);
    let (tiled, _) = reference_with(&c, None, |h| h % 2, None);
    let apart = contiguous
        .iter()
        .zip(&tiled)
        .map(|(p, q)| (p - q).abs())
        .fold(0.0, f64::max);
    assert!(apart > 0.1, "the two group layouts agree ({apart})");
    check_scan("contiguous groups", &c, false, 4);
}

#[test]
#[serial]
fn one_step_is_one_token_of_the_scan() {
    let c = case((2, 1, 4, 3, 2, 5), 6, (0.0, f32::INFINITY));
    let f = fields(&c);
    let params = SsdParams {
        a_log: &f.a_log,
        d: &f.d,
        dt_bias: &f.dt_bias,
        dt_min: c.dt_min,
        dt_max: c.dt_max,
    };
    let (y, s) = ssd_step(&f.x, &f.dt, &f.b, &f.c, params, &f.state).unwrap();
    drain();
    let (ry, rs) = reference(&c, Some(&c.state));
    assert_close("step output", &vec_of(&y), &ry, 2e-5);
    assert_close("step state", &vec_of(&s), &rs, 2e-5);
}

#[test]
#[serial]
fn a_bf16_input_is_scanned_in_float32() {
    let c = case((1, 14, 4, 3, 2, 4), 7, (0.0, f32::INFINITY));
    let f = fields(&c);
    let bf = |a: &Array| a.as_dtype(Dtype::Bfloat16.as_i32());
    let rounded = Case {
        x: vec_of(&bf(&f.x)),
        dt: vec_of(&bf(&f.dt)),
        bm: vec_of(&bf(&f.b)),
        cm: vec_of(&bf(&f.c)),
        ..c.clone()
    };
    let (y, s) = ssd_scan(
        &bf(&f.x),
        &bf(&f.dt),
        &bf(&f.b),
        &bf(&f.c),
        SsdParams {
            a_log: &f.a_log,
            d: &f.d,
            dt_bias: &f.dt_bias,
            dt_min: 0.0,
            dt_max: f32::INFINITY,
        },
        None,
        5,
    )
    .unwrap();
    drain();
    assert_eq!(y.dtype(), Dtype::Float32);
    assert_eq!(s.dtype(), Dtype::Float32);
    let (ry, rs) = reference(&rounded, None);
    assert_close("bf16 inputs: output", &vec_of(&y), &ry, 2e-5);
    assert_close("bf16 inputs: state", &vec_of(&s), &rs, 2e-5);
}

// ---------------------------------------------------------------------------
// Gradients
// ---------------------------------------------------------------------------

const NAMES: [&str; 8] = ["x", "dt", "B", "C", "A_log", "D", "dt_bias", "state"];

fn bases(c: &Case) -> Vec<Vec<f64>> {
    vec![
        c.x.clone(),
        c.dt.clone(),
        c.bm.clone(),
        c.cm.clone(),
        c.a_log.clone(),
        c.d.clone(),
        c.dt_bias.clone(),
        c.state.clone(),
    ]
}

fn shapes(c: &Case) -> Vec<Vec<usize>> {
    vec![
        vec![c.b, c.l, c.h, c.p],
        vec![c.b, c.l, c.h],
        vec![c.b, c.l, c.g, c.n],
        vec![c.b, c.l, c.g, c.n],
        vec![c.h],
        vec![c.h],
        vec![c.h],
        vec![c.b, c.h, c.p, c.n],
    ]
}

fn with_bases(c: &Case, mut v: Vec<Vec<f64>>) -> Case {
    let mut c = c.clone();
    c.state = v.pop().unwrap();
    c.dt_bias = v.pop().unwrap();
    c.d = v.pop().unwrap();
    c.a_log = v.pop().unwrap();
    c.cm = v.pop().unwrap();
    c.bm = v.pop().unwrap();
    c.dt = v.pop().unwrap();
    c.x = v.pop().unwrap();
    c
}

/// A scalar loss over the output and the final state, from random weights.
fn gradient_check(what: &str, c: &Case, chunk: i32, step: bool) {
    let mut rng = Rng(0xBEEF);
    let wy: Vec<f64> = (0..c.b * c.l * c.h * c.p)
        .map(|_| round32(rng.normal()))
        .collect();
    let ws: Vec<f64> = (0..c.b * c.h * c.p * c.n)
        .map(|_| round32(rng.normal()))
        .collect();
    let wy_arr = arr(&wy, &[c.b, c.l, c.h, c.p]);
    let ws_arr = arr(&ws, &[c.b, c.h, c.p, c.n]);
    let (dt_min, dt_max) = (c.dt_min, c.dt_max);

    check_gradient(
        what,
        &NAMES,
        &bases(c),
        &shapes(c),
        |a: &[Array]| {
            let params = SsdParams {
                a_log: &a[4],
                d: &a[5],
                dt_bias: &a[6],
                dt_min,
                dt_max,
            };
            let (y, s) = if step {
                ssd_step(&a[0], &a[1], &a[2], &a[3], params, &a[7]).unwrap()
            } else {
                ssd_scan(&a[0], &a[1], &a[2], &a[3], params, Some(&a[7]), chunk).unwrap()
            };
            y.multiply(&wy_arr)
                .sum_all()
                .add(&s.multiply(&ws_arr).sum_all())
        },
        |v: &[Vec<f64>]| {
            let c = with_bases(c, v.to_vec());
            let (y, s) = reference(&c, Some(&c.state));
            dot(&y, &wy) + dot(&s, &ws)
        },
        2e-4,
    );
}

#[test]
#[serial]
fn the_scan_differentiates_like_the_recurrence() {
    let c = case((2, 11, 4, 3, 2, 4), 8, (0.0, f32::INFINITY));
    gradient_check("chunked scan", &c, 4, false);
    gradient_check("single chunk", &c, 64, false);
}

#[test]
#[serial]
fn the_scan_differentiates_through_the_step_size_limits() {
    let c = case((1, 9, 4, 3, 2, 4), 9, (0.35, 0.9));
    gradient_check("clamped scan", &c, 4, false);
}

#[test]
#[serial]
fn the_step_differentiates_like_the_recurrence() {
    let c = case((2, 1, 4, 3, 2, 4), 10, (0.0, f32::INFINITY));
    gradient_check("step", &c, 1, true);
}

// ---------------------------------------------------------------------------
// Packed rows
// ---------------------------------------------------------------------------

/// Sequence numbers for rows packed from the given sequence lengths.
fn segment_ids(rows: &[&[usize]]) -> Vec<i32> {
    rows.iter()
        .flat_map(|lengths| {
            lengths
                .iter()
                .enumerate()
                .flat_map(|(n, &len)| std::iter::repeat_n(n as i32 * 3 + 1, len))
        })
        .collect()
}

fn scan_segmented(
    c: &Case,
    f: &Fields,
    with_state: bool,
    chunk: i32,
    ids: &[i32],
) -> (Array, Array) {
    let segments = Array::from_i32_slice(ids).reshape(&[c.b as i32, c.l as i32]);
    ssd_scan_segmented(
        &f.x,
        &f.dt,
        &f.b,
        &f.c,
        SsdParams {
            a_log: &f.a_log,
            d: &f.d,
            dt_bias: &f.dt_bias,
            dt_min: c.dt_min,
            dt_max: c.dt_max,
        },
        with_state.then_some(&f.state),
        chunk,
        &segments,
    )
    .unwrap()
}

fn check_segmented(what: &str, c: &Case, with_state: bool, chunk: i32, ids: &[i32]) {
    let f = fields(c);
    let (y, s) = scan_segmented(c, &f, with_state, chunk, ids);
    drain();
    let (ry, rs) = reference_segmented(c, with_state.then_some(c.state.as_slice()), ids);
    assert_close(&format!("{what}: output"), &vec_of(&y), &ry, 2e-5);
    assert_close(&format!("{what}: final state"), &vec_of(&s), &rs, 2e-5);
}

#[test]
#[serial]
fn packed_sequences_scan_as_if_each_were_alone() {
    // Boundaries inside a chunk, on a chunk edge and one chunk long, in two
    // rows that are cut differently.
    let c = case((2, 20, 4, 3, 2, 4), 13, (0.0, f32::INFINITY));
    let ids = segment_ids(&[&[5, 3, 8, 4], &[2, 9, 9]]);
    for chunk in [1, 3, 4, 8, 20, 64] {
        check_segmented(&format!("chunk {chunk}"), &c, false, chunk, &ids);
        check_segmented(
            &format!("chunk {chunk}, from a state"),
            &c,
            true,
            chunk,
            &ids,
        );
    }
}

#[test]
#[serial]
fn a_packed_row_longer_than_one_pass_resets_across_the_pass_edge() {
    let c = case((1, 70, 2, 3, 1, 3), 14, (0.0, f32::INFINITY));
    assert!(c.l > 2 * MAX_CHUNKS_PER_PASS as usize);
    // A boundary exactly on the first pass's edge (64 tokens at chunk 2) and
    // others before and after it.
    for lengths in [&[13usize, 30, 27][..], &[64, 6][..], &[1, 63, 6][..]] {
        let ids = segment_ids(&[lengths]);
        check_segmented(&format!("{lengths:?}"), &c, true, 2, &ids);
    }
}

#[test]
#[serial]
fn the_fixture_can_tell_a_reset_from_a_carried_state() {
    let c = case((1, 20, 4, 3, 2, 4), 15, (0.0, f32::INFINITY));
    let ids = segment_ids(&[&[7, 6, 7]]);
    let (packed, _) = reference_segmented(&c, None, &ids);
    let (carried, _) = reference(&c, None);
    let apart = packed
        .iter()
        .zip(&carried)
        .map(|(p, q)| (p - q).abs())
        .fold(0.0, f64::max);
    assert!(apart > 0.1, "a reset changes nothing here ({apart})");
    // Unsegmented, the scan carries the state through the boundary.
    check_scan("unsegmented", &c, false, 5);
}

#[test]
#[serial]
fn the_segmented_scan_differentiates_like_the_recurrence() {
    let c = case((2, 11, 4, 3, 2, 4), 16, (0.0, f32::INFINITY));
    let ids = segment_ids(&[&[4, 3, 4], &[11]]);
    let segments = Array::from_i32_slice(&ids).reshape(&[2, 11]);
    let mut rng = Rng(0xFACE);
    let wy: Vec<f64> = (0..c.b * c.l * c.h * c.p)
        .map(|_| round32(rng.normal()))
        .collect();
    let ws: Vec<f64> = (0..c.b * c.h * c.p * c.n)
        .map(|_| round32(rng.normal()))
        .collect();
    let wy_arr = arr(&wy, &[c.b, c.l, c.h, c.p]);
    let ws_arr = arr(&ws, &[c.b, c.h, c.p, c.n]);
    check_gradient(
        "segmented scan",
        &NAMES,
        &bases(&c),
        &shapes(&c),
        |a: &[Array]| {
            let params = SsdParams {
                a_log: &a[4],
                d: &a[5],
                dt_bias: &a[6],
                dt_min: 0.0,
                dt_max: f32::INFINITY,
            };
            let (y, s) = ssd_scan_segmented(
                &a[0],
                &a[1],
                &a[2],
                &a[3],
                params,
                Some(&a[7]),
                4,
                &segments,
            )
            .unwrap();
            y.multiply(&wy_arr)
                .sum_all()
                .add(&s.multiply(&ws_arr).sum_all())
        },
        |v: &[Vec<f64>]| {
            let c = with_bases(&c, v.to_vec());
            let (y, s) = reference_segmented(&c, Some(&c.state), &ids);
            dot(&y, &wy) + dot(&s, &ws)
        },
        2e-4,
    );
}

// ---------------------------------------------------------------------------
// The gated norm
// ---------------------------------------------------------------------------

#[test]
#[serial]
fn the_gated_norm_normalizes_each_group_after_gating() {
    let (rows, groups, size) = (3usize, 2usize, 5usize);
    let width = groups * size;
    let mut rng = Rng(11);
    let y: Vec<f64> = (0..rows * width)
        .map(|_| round32(2.0 * rng.normal()))
        .collect();
    let gate: Vec<f64> = (0..rows * width).map(|_| round32(rng.normal())).collect();
    let weight: Vec<f64> = (0..width)
        .map(|_| round32(1.0 + 0.3 * rng.normal()))
        .collect();
    let eps = 1e-5_f64;

    let silu = |v: f64| v / (1.0 + (-v).exp());
    let mut want = vec![0.0; rows * width];
    for r in 0..rows {
        for gi in 0..groups {
            let span = r * width + gi * size..r * width + (gi + 1) * size;
            let gated: Vec<f64> = span.clone().map(|i| y[i] * silu(gate[i])).collect();
            let mean = gated.iter().map(|v| v * v).sum::<f64>() / size as f64;
            let scale = 1.0 / (mean + eps).sqrt();
            for (k, i) in span.enumerate() {
                want[i] = gated[k] * scale * weight[gi * size + k];
            }
        }
    }
    let got = gated_group_rms_norm(
        &arr(&y, &[1, rows, width]),
        Some(&arr(&gate, &[1, rows, width])),
        &arr(&weight, &[width]),
        eps as f32,
        size as i32,
    )
    .unwrap();
    drain();
    assert_close("gated group norm", &vec_of(&got), &want, 2e-5);

    // Whole-row statistics are another function: the fixture sees the groups.
    let whole = gated_group_rms_norm(
        &arr(&y, &[1, rows, width]),
        Some(&arr(&gate, &[1, rows, width])),
        &arr(&weight, &[width]),
        eps as f32,
        width as i32,
    )
    .unwrap();
    let apart = vec_of(&whole)
        .iter()
        .zip(&want)
        .map(|(p, q)| (p - q).abs())
        .fold(0.0, f64::max);
    assert!(apart > 0.05, "group and whole-row norms agree ({apart})");
}

#[test]
#[serial]
fn the_gated_norm_gradient_matches_finite_differences() {
    let (rows, width, size) = (2usize, 6usize, 3usize);
    let mut rng = Rng(12);
    let y: Vec<f64> = (0..rows * width).map(|_| round32(rng.normal())).collect();
    let gate: Vec<f64> = (0..rows * width).map(|_| round32(rng.normal())).collect();
    let weight: Vec<f64> = (0..width)
        .map(|_| round32(1.0 + 0.3 * rng.normal()))
        .collect();
    let probe: Vec<f64> = (0..rows * width).map(|_| round32(rng.normal())).collect();
    let probe_arr = arr(&probe, &[1, rows, width]);
    let eps = 1e-5_f64;

    let forward64 = |v: &[Vec<f64>]| -> f64 {
        let (y, gate, weight) = (&v[0], &v[1], &v[2]);
        let silu = |v: f64| v / (1.0 + (-v).exp());
        let mut total = 0.0;
        for r in 0..rows {
            for gi in 0..width / size {
                let span: Vec<usize> = (0..size).map(|k| r * width + gi * size + k).collect();
                let gated: Vec<f64> = span.iter().map(|&i| y[i] * silu(gate[i])).collect();
                let mean = gated.iter().map(|v| v * v).sum::<f64>() / size as f64;
                let scale = 1.0 / (mean + eps).sqrt();
                for (k, &i) in span.iter().enumerate() {
                    total += probe[i] * gated[k] * scale * weight[gi * size + k];
                }
            }
        }
        total
    };
    check_gradient(
        "gated norm",
        &["y", "gate", "weight"],
        &[y, gate, weight],
        &[vec![1, rows, width], vec![1, rows, width], vec![width]],
        |a: &[Array]| {
            gated_group_rms_norm(&a[0], Some(&a[1]), &a[2], eps as f32, size as i32)
                .unwrap()
                .multiply(&probe_arr)
                .sum_all()
        },
        forward64,
        2e-4,
    );
}
