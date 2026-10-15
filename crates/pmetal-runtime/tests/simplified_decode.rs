use pmetal_runtime::InlineArray;
use std::time::Instant;

/// Create a materialized (non-broadcast) weight tensor.
/// ones() and zeros() create broadcast scalars — 2 bytes regardless of shape.
/// arange() creates unique elements, forcing a FULL Metal buffer allocation.
fn rand_w(shape: &[i32], dtype: i32) -> InlineArray {
    let n: i32 = shape.iter().product();
    let w = InlineArray::arange(n, dtype).reshape(shape);
    w.eval();
    w
}

fn silu(x: &InlineArray) -> InlineArray {
    x.multiply(&x.sigmoid())
}

#[test]
fn simplified_decode_benchmark() {
    let h = 1024i32;
    let n_layers = 24;
    let inter = 3584i32;
    let dtype = 11; // bfloat16

    struct LayerWeights {
        ln1_w: InlineArray,
        ln2_w: InlineArray,
        gate_wt: InlineArray,
        up_wt: InlineArray,
        down_wt: InlineArray,
        is_gdn: bool,
        qkvz_wt: Option<InlineArray>,
        ba_wt: Option<InlineArray>,
        out_wt: Option<InlineArray>,
        q_wt: Option<InlineArray>,
        k_wt: Option<InlineArray>,
        v_wt: Option<InlineArray>,
        o_wt: Option<InlineArray>,
    }

    // Use rand_w for REAL materialized weights (full Metal buffers)
    let embed_w = rand_w(&[248320, h], dtype);
    let final_ln_w = rand_w(&[h], dtype);

    let mut layers = Vec::new();
    for i in 0..n_layers {
        let is_gdn = (i % 4) != 3;
        let mut lw = LayerWeights {
            ln1_w: rand_w(&[h], dtype),
            ln2_w: rand_w(&[h], dtype),
            gate_wt: rand_w(&[h, inter], dtype),
            up_wt: rand_w(&[h, inter], dtype),
            down_wt: rand_w(&[inter, h], dtype),
            is_gdn,
            qkvz_wt: None,
            ba_wt: None,
            out_wt: None,
            q_wt: None,
            k_wt: None,
            v_wt: None,
            o_wt: None,
        };
        if is_gdn {
            lw.qkvz_wt = Some(rand_w(&[h, 8192], dtype));
            lw.ba_wt = Some(rand_w(&[h, 32], dtype));
            lw.out_wt = Some(rand_w(&[h, h], dtype));
        } else {
            lw.q_wt = Some(rand_w(&[h, 4096], dtype));
            lw.k_wt = Some(rand_w(&[h, 512], dtype));
            lw.v_wt = Some(rand_w(&[h, 512], dtype));
            lw.o_wt = Some(rand_w(&[h, h], dtype));
        }
        layers.push(lw);
    }

    // Eval all weights
    for lw in &layers {
        let w = lw.ln1_w.clone();
        w.eval();
    }

    // Simulate the real model's Metal buffer footprint (~1550 MB)
    // by allocating many additional buffers.
    let mut dummy_buffers: Vec<InlineArray> = Vec::new();
    let dummy_bytes: i32 = 1550 * 1024 * 1024 / 2; // ~1550 MB in bf16 elements
    let chunk = 10_000_000i32; // 10M elements per chunk = 20MB
    let n_chunks = dummy_bytes / chunk;
    for _ in 0..n_chunks {
        let d = InlineArray::arange(chunk, 11);
        d.eval();
        dummy_buffers.push(d);
    }
    eprintln!(
        "Allocated {} dummy buffers ({:.0} MB)",
        n_chunks,
        n_chunks as f64 * chunk as f64 * 2.0 / 1e6
    );
    eprintln!(
        "Active memory: {:.0} MB",
        pmetal_runtime::inline_array::get_active_memory() as f64 / 1e6
    );

    let decode_step = |token_id: i32| -> InlineArray {
        let tok = InlineArray::from_i32(token_id).reshape(&[1, 1]);
        let idx = tok.squeeze(0);
        let emb = embed_w.take_axis(&idx, 0);
        let mut hidden = emb.reshape(&[1, 1, h]);

        for lw in &layers {
            let normed = hidden.rms_norm(Some(&lw.ln1_w), 1e-6);
            let r = if lw.is_gdn {
                let _proj = normed.matmul(lw.qkvz_wt.as_ref().unwrap());
                let _ba = normed.matmul(lw.ba_wt.as_ref().unwrap());
                normed.matmul(lw.out_wt.as_ref().unwrap())
            } else {
                let _q = normed.matmul(lw.q_wt.as_ref().unwrap());
                let _k = normed.matmul(lw.k_wt.as_ref().unwrap());
                let _v = normed.matmul(lw.v_wt.as_ref().unwrap());
                normed.matmul(lw.o_wt.as_ref().unwrap())
            };
            let h2 = hidden.add(&r.reshape(&[1, 1, h]));
            let mlp_in = h2.rms_norm(Some(&lw.ln2_w), 1e-6);
            let gate = mlp_in.matmul(&lw.gate_wt);
            let up = mlp_in.matmul(&lw.up_wt);
            let mlp_out = silu(&gate).multiply(&up).matmul(&lw.down_wt);
            hidden = h2.add(&mlp_out);
        }
        let out = hidden.rms_norm(Some(&final_ln_w), 1e-6);
        out.matmul(&embed_w.t())
    };

    // Warmup
    for i in 0..5 {
        let r = decode_step(42 + i);
        r.eval();
    }

    // Benchmark
    let mut times = Vec::new();
    for i in 0..50 {
        let t0 = Instant::now();
        let r = decode_step(100 + i);
        r.eval();
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let avg = times[5..].iter().sum::<f64>() / (times.len() - 5) as f64;
    let p50 = times[times.len() / 2];
    eprintln!(
        "InlineArray simplified decode: avg={avg:.2}ms p50={p50:.2}ms = {:.0} tok/s",
        1000.0 / avg
    );
}
