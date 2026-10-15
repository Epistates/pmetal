//! A compiled Gemma 4 block traces once per cache key and replays the trace
//! on a hit, so the key has to name everything the trace bakes in.
//!
//! Each block baked values the key left out (norm epsilons, the RoPE base and
//! rotated width): a second model, or a second layer type, that shared a
//! geometry but not those values replayed the first one's trace.
//!
//! Every case runs a baseline, then the same geometry with one baked value
//! changed, then that changed call again under a key that differs for an
//! unrelated reason (a longer KV capacity, which the attention mask makes
//! irrelevant, or a second batch row). The second must equal the third; on the
//! old keys it equalled the baseline.

use pmetal_runtime::{InlineArray, LayerWeight, check_unobserved_error};

/// Deterministic pseudo-random values in `[-scale, scale]`.
fn noise(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

fn arr(values: Vec<f32>, shape: &[i32]) -> InlineArray {
    InlineArray::from_f32_slice(&values, shape)
}

fn read(mut a: InlineArray) -> Vec<f32> {
    let n = a.size();
    check_unobserved_error().expect("bridge error");
    a.to_f32_vec(n).expect("readback")
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// The three runs described in the module docs.
fn assert_key_covers(
    what: &str,
    baseline: impl FnOnce() -> Vec<f32>,
    changed: impl FnOnce() -> Vec<f32>,
    changed_under_another_key: impl FnOnce() -> Vec<f32>,
) {
    let baseline = baseline();
    let changed = changed();
    let reference = changed_under_another_key();
    assert!(
        max_diff(&baseline, &reference) > 1e-4,
        "{what}: changing the value does not change the output, so this case checks nothing"
    );
    let stale = max_diff(&changed, &reference);
    assert!(
        stale < 1e-4,
        "{what}: the second call replayed the baseline's trace (differs from a fresh trace by {stale})"
    );
}

// ── attention block ──────────────────────────────────────────────────────

const HIDDEN: i32 = 8;
const N_HEADS: i32 = 2;
const N_KV: i32 = 1;
const HEAD_DIM: i32 = 4;

#[derive(Clone, Copy)]
struct Attn {
    in_eps: f32,
    qk_eps: f32,
    post_eps: f32,
    base: f32,
    rope_dims: i32,
}

const ATTN: Attn = Attn {
    in_eps: 1e-6,
    qk_eps: 1e-6,
    post_eps: 1e-6,
    base: 10_000.0,
    rope_dims: HEAD_DIM,
};

fn dense(values: Vec<f32>, shape: &[i32]) -> LayerWeight {
    LayerWeight::Dense(arr(values, shape))
}

/// One decode step of the full block at position 5 over a cache holding five
/// earlier positions of noise, in a cache of capacity `cache_len`.
fn attn_block(p: Attn, cache_len: i32) -> Vec<f32> {
    let kv = N_KV * HEAD_DIM;
    let x = arr(noise(HIDDEN as usize, 1, 0.02), &[1, 1, HIDDEN]);
    let mut keys = vec![0.0; (N_KV * cache_len * HEAD_DIM) as usize];
    let mut vals = keys.clone();
    let k_noise = noise((5 * HEAD_DIM) as usize, 2, 1.0);
    let v_noise = noise((5 * HEAD_DIM) as usize, 3, 1.0);
    keys[..k_noise.len()].copy_from_slice(&k_noise);
    vals[..v_noise.len()].copy_from_slice(&v_noise);
    let (out, _, _) = InlineArray::compiled_gemma4_attn_block(
        &x,
        &arr(vec![1.0; HIDDEN as usize], &[HIDDEN]),
        &dense(
            noise((HIDDEN * N_HEADS * HEAD_DIM) as usize, 4, 0.5),
            &[HIDDEN, N_HEADS * HEAD_DIM],
        ),
        &dense(noise((HIDDEN * kv) as usize, 5, 0.5), &[HIDDEN, kv]),
        Some(&dense(noise((HIDDEN * kv) as usize, 6, 0.5), &[HIDDEN, kv])),
        &dense(
            noise((N_HEADS * HEAD_DIM * HIDDEN) as usize, 7, 0.5),
            &[N_HEADS * HEAD_DIM, HIDDEN],
        ),
        &arr(
            noise(HEAD_DIM as usize, 8, 1.0)
                .iter()
                .map(|v| v + 1.5)
                .collect(),
            &[HEAD_DIM],
        ),
        &arr(
            noise(HEAD_DIM as usize, 9, 1.0)
                .iter()
                .map(|v| v + 1.5)
                .collect(),
            &[HEAD_DIM],
        ),
        &arr(vec![1.0; HIDDEN as usize], &[HIDDEN]),
        None,
        &arr(keys, &[1, N_KV, cache_len, HEAD_DIM]),
        &arr(vals, &[1, N_KV, cache_len, HEAD_DIM]),
        5,
        0,
        N_HEADS,
        N_KV,
        HEAD_DIM,
        p.in_eps,
        p.qk_eps,
        p.post_eps,
        0,
        p.base,
        p.rope_dims,
    );
    read(out)
}

fn attn_case(what: &str, changed: Attn, tag: i32) {
    // Each case has capacities of its own so no two share a cache entry.
    let l = 16 + 16 * tag;
    assert_key_covers(
        what,
        || attn_block(ATTN, l),
        || attn_block(changed, l),
        || attn_block(changed, l + 8),
    );
}

#[test]
fn the_attention_blocks_key_covers_the_input_norm_epsilon() {
    attn_case(
        "in_norm_eps",
        Attn {
            in_eps: 1e-1,
            ..ATTN
        },
        0,
    );
}

#[test]
fn the_attention_blocks_key_covers_the_qk_norm_epsilon() {
    attn_case(
        "qk_norm_eps",
        Attn {
            qk_eps: 1.0,
            ..ATTN
        },
        1,
    );
}

#[test]
fn the_attention_blocks_key_covers_the_post_norm_epsilon() {
    attn_case(
        "post_norm_eps",
        Attn {
            post_eps: 1e-1,
            ..ATTN
        },
        2,
    );
}

#[test]
fn the_attention_blocks_key_covers_the_rope_base() {
    attn_case(
        "rope_base",
        Attn {
            base: 1_000_000.0,
            ..ATTN
        },
        3,
    );
}

#[test]
fn the_attention_blocks_key_covers_the_rotated_width() {
    attn_case(
        "rope_dims",
        Attn {
            rope_dims: 2,
            ..ATTN
        },
        4,
    );
}

// ── shared-KV attention (q only) ─────────────────────────────────────────

fn shared_attn(p: Attn, cache_len: i32) -> Vec<f32> {
    let x = arr(noise(HIDDEN as usize, 11, 0.02), &[1, 1, HIDDEN]);
    let mut keys = vec![0.0; (N_KV * cache_len * HEAD_DIM) as usize];
    let mut vals = keys.clone();
    let k_noise = noise((6 * HEAD_DIM) as usize, 12, 1.0);
    let v_noise = noise((6 * HEAD_DIM) as usize, 13, 1.0);
    keys[..k_noise.len()].copy_from_slice(&k_noise);
    vals[..v_noise.len()].copy_from_slice(&v_noise);
    read(InlineArray::compiled_gemma4_shared_attn_decode(
        &x,
        &arr(vec![1.0; HIDDEN as usize], &[HIDDEN]),
        &dense(
            noise((HIDDEN * N_HEADS * HEAD_DIM) as usize, 14, 0.5),
            &[HIDDEN, N_HEADS * HEAD_DIM],
        ),
        &dense(
            noise((N_HEADS * HEAD_DIM * HIDDEN) as usize, 15, 0.5),
            &[N_HEADS * HEAD_DIM, HIDDEN],
        ),
        &arr(
            noise(HEAD_DIM as usize, 16, 1.0)
                .iter()
                .map(|v| v + 1.5)
                .collect(),
            &[HEAD_DIM],
        ),
        &arr(vec![1.0; HIDDEN as usize], &[HIDDEN]),
        None,
        &arr(keys, &[1, N_KV, cache_len, HEAD_DIM]),
        &arr(vals, &[1, N_KV, cache_len, HEAD_DIM]),
        6,
        6,
        0,
        N_HEADS,
        N_KV,
        HEAD_DIM,
        p.in_eps,
        p.qk_eps,
        p.post_eps,
        0,
        p.base,
        p.rope_dims,
    ))
}

fn shared_case(what: &str, changed: Attn, tag: i32) {
    let l = 16 + 16 * tag;
    assert_key_covers(
        what,
        || shared_attn(ATTN, l),
        || shared_attn(changed, l),
        || shared_attn(changed, l + 8),
    );
}

#[test]
fn the_shared_attention_key_covers_the_input_norm_epsilon() {
    shared_case(
        "in_norm_eps",
        Attn {
            in_eps: 1e-1,
            ..ATTN
        },
        0,
    );
}

#[test]
fn the_shared_attention_key_covers_the_q_norm_epsilon() {
    shared_case(
        "q_norm_eps",
        Attn {
            qk_eps: 1.0,
            ..ATTN
        },
        1,
    );
}

#[test]
fn the_shared_attention_key_covers_the_post_norm_epsilon() {
    shared_case(
        "post_norm_eps",
        Attn {
            post_eps: 1e-1,
            ..ATTN
        },
        2,
    );
}

#[test]
fn the_shared_attention_key_covers_the_rope_base() {
    shared_case(
        "rope_base",
        Attn {
            base: 1_000_000.0,
            ..ATTN
        },
        3,
    );
}

#[test]
fn the_shared_attention_key_covers_the_rotated_width() {
    shared_case(
        "rope_dims",
        Attn {
            rope_dims: 2,
            ..ATTN
        },
        4,
    );
}

// ── MLP and per-layer-input blocks (no cache: the second key is a second
//    batch row) ─────────────────────────────────────────────────────────

const INTER: i32 = 16;

/// Row 0 of the MLP block over `batch` identical rows.
fn mlp(pre_eps: f32, post_eps: f32, batch: i32) -> Vec<f32> {
    let row = noise(HIDDEN as usize, 21, 0.02);
    let x: Vec<f32> = (0..batch).flat_map(|_| row.clone()).collect();
    let out = InlineArray::compiled_gemma4_mlp_block(
        &arr(x, &[batch, 1, HIDDEN]),
        &arr(vec![1.0; HIDDEN as usize], &[HIDDEN]),
        &dense(noise((HIDDEN * INTER) as usize, 22, 0.5), &[HIDDEN, INTER]),
        &dense(noise((HIDDEN * INTER) as usize, 23, 0.5), &[HIDDEN, INTER]),
        &dense(noise((INTER * HIDDEN) as usize, 24, 0.5), &[INTER, HIDDEN]),
        &arr(vec![1.0; HIDDEN as usize], &[HIDDEN]),
        pre_eps,
        post_eps,
    );
    read(out)[..HIDDEN as usize].to_vec()
}

#[test]
fn the_mlp_blocks_key_covers_the_pre_norm_epsilon() {
    assert_key_covers(
        "pre_norm_eps",
        || mlp(1e-6, 1e-6, 1),
        || mlp(1e-1, 1e-6, 1),
        || mlp(1e-1, 1e-6, 2),
    );
}

#[test]
fn the_mlp_blocks_key_covers_the_post_norm_epsilon() {
    assert_key_covers(
        "post_norm_eps",
        || mlp(1e-6, 1e-6, 3),
        || mlp(1e-6, 1e-1, 3),
        || mlp(1e-6, 1e-1, 4),
    );
}

const PER_LAYER: i32 = 4;

fn per_layer_input(post_eps: f32, batch: i32) -> Vec<f32> {
    let row = noise(HIDDEN as usize, 31, 0.02);
    let x: Vec<f32> = (0..batch).flat_map(|_| row.clone()).collect();
    let layer_row = noise(PER_LAYER as usize, 32, 1.0);
    let layer: Vec<f32> = (0..batch).flat_map(|_| layer_row.clone()).collect();
    let out = InlineArray::compiled_gemma4_per_layer_input_block(
        &arr(x, &[batch, 1, HIDDEN]),
        &arr(layer, &[batch, 1, PER_LAYER]),
        &dense(
            noise((HIDDEN * PER_LAYER) as usize, 33, 0.5),
            &[HIDDEN, PER_LAYER],
        ),
        &dense(
            noise((PER_LAYER * HIDDEN) as usize, 34, 0.5),
            &[PER_LAYER, HIDDEN],
        ),
        &arr(vec![1.0; HIDDEN as usize], &[HIDDEN]),
        post_eps,
    );
    read(out)[..HIDDEN as usize].to_vec()
}

#[test]
fn the_per_layer_input_blocks_key_covers_the_post_norm_epsilon() {
    assert_key_covers(
        "post_norm_eps",
        || per_layer_input(1e-6, 1),
        || per_layer_input(1e-1, 1),
        || per_layer_input(1e-1, 2),
    );
}
