use pmetal_runtime::InlineArray;
use std::time::Instant;

fn bench_eval(label: &str, mut setup: impl FnMut() -> InlineArray, iters: usize) {
    // Warm up
    for _ in 0..3 {
        let r = setup();
        r.eval();
    }
    // Measure
    let mut times = Vec::new();
    for _ in 0..iters {
        let r = setup();
        let t0 = Instant::now();
        r.eval();
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let avg = times[5..].iter().sum::<f64>() / (times.len() - 5) as f64;
    eprintln!("{label}: {avg:.2}ms");
}

#[test]
fn profile_real_ops() {
    let h = 1024i32;
    let vocab = 248320i32;
    let hd = 256i32;
    let nh = 8i32;
    let nkv = 2i32;

    // lm_head: the largest single matmul
    let x = InlineArray::ones(&[1, 1, h], 10);
    let ew = InlineArray::ones(&[vocab, h], 10);
    let ewt = ew.t();
    bench_eval("lm_head [1,1,1024]@[1024,248K]", || x.matmul(&ewt), 20);

    // SDPA at growing seq lengths
    for sl in [10, 50, 100, 200] {
        let q = InlineArray::ones(&[1, nh, 1, hd], 10);
        let k = InlineArray::ones(&[1, nkv, sl, hd], 10);
        let v = InlineArray::ones(&[1, nkv, sl, hd], 10);
        bench_eval(
            &format!("SDPA seq={sl}"),
            || q.sdpa(&k, &v, 0.0625, "causal"),
            20,
        );
    }

    // KV append
    let ck = InlineArray::ones(&[1, nkv, 100, hd], 10);
    let nk = InlineArray::ones(&[1, nkv, 1, hd], 10);
    bench_eval("KV append 100+1", || ck.kv_cache_append(&nk, 2), 20);

    // Embedding
    let em = InlineArray::ones(&[vocab, h], 10);
    let tok = InlineArray::from_i32(42).reshape(&[1, 1]);
    bench_eval("Embed lookup [248K,1024]", || em.take_axis(&tok, 0), 20);

    // Full 24-layer MLP (real dims: hidden=1024, intermediate=3584)
    // Weights are pre-transposed: storage shape [out, in] -> .t() for [in, out]
    // so matmul(x[...,in], w[in,out]) -> [...,out].
    let gw = InlineArray::ones(&[3584, h], 10).t(); // [h=1024, 3584]
    gw.eval();
    let uw = InlineArray::ones(&[3584, h], 10).t(); // [h=1024, 3584]
    uw.eval();
    let dw = InlineArray::ones(&[h, 3584], 10).t(); // [3584, h=1024]
    dw.eval();
    let nw = InlineArray::ones(&[h], 10);
    bench_eval(
        "24-layer MLP (real dims)",
        || {
            let mut x = InlineArray::ones(&[1, 1, h], 10);
            for _ in 0..24 {
                let n = x.rms_norm(Some(&nw), 1e-6);
                let g = InlineArray::fused_silu(&n.matmul(&gw));
                let u = n.matmul(&uw);
                x = x.add(&g.multiply(&u).matmul(&dw));
            }
            x
        },
        20,
    );
}
