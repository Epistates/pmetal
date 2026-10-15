use pmetal_runtime::InlineArray;
use std::time::Instant;

fn bench(label: &str, mut setup: impl FnMut() -> InlineArray, iters: usize) {
    for _ in 0..5 {
        let r = setup();
        r.eval();
    }
    let mut times = Vec::new();
    for _ in 0..iters {
        let r = setup();
        let t0 = Instant::now();
        r.eval();
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let avg = times[5..].iter().sum::<f64>() / (times.len() - 5) as f64;
    eprintln!("{label}: {avg:.3}ms");
}

fn bench2(label: &str, mut setup: impl FnMut() -> (InlineArray, InlineArray), iters: usize) {
    for _ in 0..5 {
        let (a, b) = setup();
        a.eval();
        b.eval();
    }
    let mut times = Vec::new();
    for _ in 0..iters {
        let (mut a, mut b) = setup();
        let t0 = Instant::now();
        InlineArray::eval_2(&mut a, &mut b);
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let avg = times[5..].iter().sum::<f64>() / (times.len() - 5) as f64;
    eprintln!("{label}: {avg:.3}ms");
}

#[test]
fn profile_gdn_specific_ops() {
    let dt = 11; // bf16
    let h = 1024i32;
    let nk = 16i32;
    let nv = 16i32;
    let dk = 128i32;
    let dv = 128i32;
    let cd = 6144i32; // conv_dim
    let ck = 4i32;
    let n_heads = 8i32;
    let n_kv = 2i32;
    let head_dim = 256i32;

    // ── Conv1d (depthwise, groups=cd) ──
    let conv_input = InlineArray::ones(&[1, ck, cd], dt);
    let conv_weight = InlineArray::ones(&[cd, ck, 1], dt);
    bench(
        "conv1d depthwise (groups=6144)",
        || conv_input.conv1d(&conv_weight, 1, 0, 1, cd),
        30,
    );

    // ── GDN Metal kernel ──
    let q = InlineArray::ones(&[1, 1, nk, dk], dt);
    let k = InlineArray::ones(&[1, 1, nk, dk], dt);
    let v = InlineArray::ones(&[1, 1, nv, dv], dt);
    let g = InlineArray::ones(&[1, 1, nv], dt);
    let beta = InlineArray::ones(&[1, 1, nv], dt);
    let state = InlineArray::zeros(&[1, nv, dv, dk], 10); // f32 state
    bench2(
        "GDN metal_step (nv=16, dk=128, dv=128)",
        || InlineArray::gdn_metal_step(&q, &k, &v, &g, &beta, &state, 1),
        30,
    );

    // ── SDPA ──
    let sq = InlineArray::ones(&[1, n_heads, 1, head_dim], dt);
    let sk = InlineArray::ones(&[1, n_kv, 50, head_dim], dt);
    let sv = InlineArray::ones(&[1, n_kv, 50, head_dim], dt);
    bench(
        "SDPA (8 heads, seq=50, hd=256)",
        || sq.sdpa(&sk, &sv, 0.0625, "causal"),
        30,
    );

    // ── RoPE ──
    let rx = InlineArray::ones(&[1, 1, n_heads, head_dim], dt);
    bench("RoPE", || rx.rope(64, false, 1000000.0, 1.0, 10), 30);

    // ── KV cache slice_set ──
    let kv_buf = InlineArray::zeros(&[1, n_kv, 256, head_dim], dt);
    let kv_new = InlineArray::ones(&[1, n_kv, 1, head_dim], dt);
    bench(
        "slice_set (KV cache update)",
        || kv_buf.slice_set(&kv_new, &[0, 0, 10, 0], &[1, n_kv, 11, head_dim]),
        30,
    );

    // ── fused_compute_g ──
    let a_log = InlineArray::ones(&[nv], dt);
    let a_val = InlineArray::ones(&[1, 1, nv], dt);
    let dt_bias = InlineArray::ones(&[nv], dt);
    bench(
        "fused_compute_g",
        || InlineArray::fused_compute_g(&a_log, &a_val, &dt_bias),
        30,
    );

    // ── fused_swiglu ──
    let gate = InlineArray::ones(&[1, 1, 3584], dt);
    let up = InlineArray::ones(&[1, 1, 3584], dt);
    bench("fused_swiglu", || InlineArray::fused_swiglu(&gate, &up), 30);

    // ── 18x GDN layers total (just conv + kernel) ──
    bench(
        "18x (conv1d + gdn_kernel)",
        || {
            let mut total = InlineArray::ones(&[1, 1, h], dt);
            for _ in 0..18 {
                let _c = conv_input.conv1d(&conv_weight, 1, 0, 1, cd);
                let (out, _state) = InlineArray::gdn_metal_step(&q, &k, &v, &g, &beta, &state, 1);
                total = total.add(&out.reshape(&[1, 1, -1]).slice(&[0, 0, 0], &[1, 1, h]));
            }
            total
        },
        30,
    );
}
