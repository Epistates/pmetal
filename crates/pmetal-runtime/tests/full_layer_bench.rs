use pmetal_runtime::InlineArray;
use std::time::Instant;

/// Create a materialized weight tensor (full Metal buffer, not broadcast).
fn real_w(shape: &[i32], dtype: i32) -> InlineArray {
    let n: i32 = shape.iter().product();
    let w = InlineArray::arange(n, dtype).reshape(shape);
    w.eval();
    w
}

/// Create a pre-transposed weight matching the real model's pattern:
/// arange → reshape to [out, in] → .t() → eval
/// This creates a column-major view, exactly like ia_from_array(weight).t()
fn real_wt(in_dim: i32, out_dim: i32, dtype: i32) -> InlineArray {
    let n = in_dim * out_dim;
    let w = InlineArray::arange(n, dtype)
        .reshape(&[out_dim, in_dim])
        .t();
    w.eval();
    w
}

/// Build and eval a FULL 24-layer forward pass (all GDN+attention ops)
/// and compare with the simplified version to isolate the overhead.
#[test]
fn full_layer_forward() {
    let dt = 11; // bf16
    let h = 1024i32;
    let nk = 16i32;
    let nv = 16i32;
    let dk = 128i32;
    let dv = 128i32;
    let cd = 6144i32;
    let ck = 4i32;
    let n_heads = 8i32;
    let n_kv = 2i32;
    let head_dim = 256i32;
    let inter = 3584i32;

    // Weights
    struct LW {
        ln1_w: InlineArray,
        ln2_w: InlineArray,
        gate_wt: InlineArray,
        up_wt: InlineArray,
        down_wt: InlineArray,
        is_gdn: bool,
        // GDN
        qkvz_wt: Option<InlineArray>,
        ba_wt: Option<InlineArray>,
        conv_w: Option<InlineArray>,
        q_nw: Option<InlineArray>,
        k_nw: Option<InlineArray>,
        a_log: Option<InlineArray>,
        dt_bias: Option<InlineArray>,
        out_wt: Option<InlineArray>,
        norm_w: Option<InlineArray>,
        // Attn
        q_wt: Option<InlineArray>,
        k_wt: Option<InlineArray>,
        v_wt: Option<InlineArray>,
        o_wt: Option<InlineArray>,
    }

    let embed_w = real_w(&[248320, h], dt);
    let final_ln_w = real_w(&[h], dt);

    let mut layers: Vec<LW> = Vec::new();
    for i in 0..24 {
        let is_gdn = (i % 4) != 3;
        let mut lw = LW {
            ln1_w: real_w(&[h], dt),
            ln2_w: real_w(&[h], dt),
            gate_wt: real_wt(h, inter, dt),
            up_wt: real_wt(h, inter, dt),
            down_wt: real_wt(inter, h, dt),
            is_gdn,
            qkvz_wt: None,
            ba_wt: None,
            conv_w: None,
            q_nw: None,
            k_nw: None,
            a_log: None,
            dt_bias: None,
            out_wt: None,
            norm_w: None,
            q_wt: None,
            k_wt: None,
            v_wt: None,
            o_wt: None,
        };
        if is_gdn {
            lw.qkvz_wt = Some(real_wt(h, cd + nv * dv, dt));
            lw.ba_wt = Some(real_wt(h, nv * 2, dt));
            lw.conv_w = Some(real_w(&[cd, ck, 1], dt));
            lw.q_nw = Some(real_w(&[dk], dt));
            lw.k_nw = Some(real_w(&[dk], dt));
            lw.a_log = Some(real_w(&[nv], dt));
            lw.dt_bias = Some(real_w(&[nv], dt));
            lw.out_wt = Some(real_wt(nv * dv, h, dt));
            lw.norm_w = Some(real_w(&[nv * dv], dt));
        } else {
            lw.q_wt = Some(real_wt(h, n_heads * head_dim * 2, dt));
            lw.k_wt = Some(real_wt(h, n_kv * head_dim, dt));
            lw.v_wt = Some(real_wt(h, n_kv * head_dim, dt));
            lw.o_wt = Some(real_wt(h, h, dt));
        }
        layers.push(lw);
    }

    // ── Test A: Simplified (matmul-only, same as Python benchmark) ──
    let dummy_hxh = real_wt(h, h, dt); // [H, H] for simplified output
    let simplified_step = |tok: i32| -> InlineArray {
        let t = InlineArray::from_i32(tok).reshape(&[1, 1]);
        let mut hidden = embed_w.take_axis(&t.squeeze(0), 0).reshape(&[1, 1, h]);
        for lw in &layers {
            let normed = hidden.rms_norm(Some(&lw.ln1_w), 1e-6);
            let r = if lw.is_gdn {
                let _p = normed.matmul(lw.qkvz_wt.as_ref().unwrap());
                let _b = normed.matmul(lw.ba_wt.as_ref().unwrap());
                normed.matmul(&dummy_hxh)
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
            let mlp_out = InlineArray::fused_swiglu(&gate, &up).matmul(&lw.down_wt);
            hidden = h2.add(&mlp_out);
        }
        hidden
            .rms_norm(Some(&final_ln_w), 1e-6)
            .matmul(&embed_w.t())
    };

    // ── Test B: With GDN conv + kernel (adds ~480 dispatches) ──
    let mut gdn_states: Vec<InlineArray> = (0..18)
        .map(|_| InlineArray::zeros(&[1, nv, dv, dk], 10))
        .collect();
    let mut conv_states: Vec<InlineArray> = (0..18)
        .map(|_| InlineArray::zeros(&[1, ck - 1, cd], dt))
        .collect();

    let full_step = |tok: i32,
                     gdn_states: &mut Vec<InlineArray>,
                     conv_states: &mut Vec<InlineArray>|
     -> InlineArray {
        let t = InlineArray::from_i32(tok).reshape(&[1, 1]);
        let mut hidden = embed_w.take_axis(&t.squeeze(0), 0).reshape(&[1, 1, h]);
        let mut gdn_idx = 0usize;
        for lw in &layers {
            let normed = hidden.rms_norm(Some(&lw.ln1_w), 1e-6);
            let r = if lw.is_gdn {
                let qkvz = normed.matmul(lw.qkvz_wt.as_ref().unwrap());
                let ba = normed.matmul(lw.ba_wt.as_ref().unwrap());

                // Split qkvz → qkv + z
                let qkv = qkvz.slice(&[0, 0, 0], &[1, 1, cd]);
                let _z = qkvz.slice(&[0, 0, cd], &[1, 1, cd + nv * dv]);

                // Conv1d (depthwise)
                let cs = &conv_states[gdn_idx];
                let conv_in = cs.concatenate_2(&qkv, 1);
                conv_states[gdn_idx] = conv_in.slice(&[0, 1, 0], &[1, ck, cd]);
                let conv_out = conv_in.conv1d(lw.conv_w.as_ref().unwrap(), 1, 0, 1, cd);
                let conv_out = InlineArray::fused_silu(&conv_out);

                // Split conv_out → q, k, v
                let kd = nk * dk;
                let q = conv_out
                    .slice(&[0, 0, 0], &[1, 1, kd])
                    .reshape(&[1, 1, nk, dk]);
                let k = conv_out
                    .slice(&[0, 0, kd], &[1, 1, kd * 2])
                    .reshape(&[1, 1, nk, dk]);
                let v = conv_out
                    .slice(&[0, 0, kd * 2], &[1, 1, cd])
                    .reshape(&[1, 1, nv, dv]);

                // QK norm
                let q = q.rms_norm(lw.q_nw.as_ref(), 1e-6);
                let k = k.rms_norm(lw.k_nw.as_ref(), 1e-6);

                // GDN gating
                let a_parts = ba.slice(&[0, 0, nv], &[1, 1, nv * 2]);
                let b_parts = ba.slice(&[0, 0, 0], &[1, 1, nv]);
                let g = InlineArray::fused_compute_g(
                    lw.a_log.as_ref().unwrap(),
                    &a_parts,
                    lw.dt_bias.as_ref().unwrap(),
                );
                let beta = b_parts.sigmoid();

                // GDN Metal kernel
                let st = &gdn_states[gdn_idx];
                let (out, new_state) = InlineArray::gdn_metal_step(&q, &k, &v, &g, &beta, st, 1);
                gdn_states[gdn_idx] = new_state;
                gdn_idx += 1;

                // Output projection — reshape to [1,1,nv*dv] then project to H
                out.reshape(&[1, 1, nv * dv])
                    .matmul(lw.out_wt.as_ref().unwrap())
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
            let mlp_out = InlineArray::fused_swiglu(&gate, &up).matmul(&lw.down_wt);
            hidden = h2.add(&mlp_out);
        }
        hidden
            .rms_norm(Some(&final_ln_w), 1e-6)
            .matmul(&embed_w.t())
    };

    // Warmup simplified
    for i in 0..5 {
        let r = simplified_step(i);
        r.eval();
    }
    let mut times_a = Vec::new();
    for i in 0..30 {
        let t0 = Instant::now();
        let r = simplified_step(100 + i);
        r.eval();
        times_a.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    times_a.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let avg_a = times_a[5..].iter().sum::<f64>() / (times_a.len() - 5) as f64;

    // Warmup full
    for i in 0..5 {
        let r = full_step(i, &mut gdn_states, &mut conv_states);
        r.eval();
    }
    let mut times_b = Vec::new();
    for i in 0..30 {
        let t0 = Instant::now();
        let r = full_step(200 + i, &mut gdn_states, &mut conv_states);
        r.eval();
        times_b.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    times_b.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let avg_b = times_b[5..].iter().sum::<f64>() / (times_b.len() - 5) as f64;

    eprintln!("Simplified: avg={avg_a:.2}ms = {:.0} tok/s", 1000.0 / avg_a);
    eprintln!("Full GDN:   avg={avg_b:.2}ms = {:.0} tok/s", 1000.0 / avg_b);
    eprintln!(
        "GDN overhead: {:.2}ms ({:.1}x)",
        avg_b - avg_a,
        avg_b / avg_a
    );
}
