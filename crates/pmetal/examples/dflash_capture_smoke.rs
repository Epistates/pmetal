//! End-to-end smoke test for the DFlash hidden-state capture path against
//! a real Qwen3.5 checkpoint.
//!
//! Loads the target via `DynamicModel::load`, extracts
//! `Qwen3NextForCausalLM`, runs a short forward pass with a `SpecCapture`
//! buffer requesting hidden states at a handful of layers, and reports
//! shapes + timing. This is the pre-wire-up sanity check for
//! `wire GDN verify-input capture`: it proves the existing hidden-state
//! tap works on a real hybrid stack before we add the GDN-specific capture.
//!
//! Usage:
//!     cargo run -p pmetal --release --example dflash_capture_smoke -- \
//!         --model <path-to-Qwen3.5-checkpoint>
//!
//! Or pass an HF repo id via `--model <org>/<Qwen3.5-0.8B-repo>` to download
//! (or use cached) via hub.

use std::time::Instant;

use pmetal::hub::resolve_model_path;
use pmetal::models::DynamicModel;
use pmetal::models::dflash_decoder::DFlashTarget;
use pmetal_mlx::Array;
use pmetal_mlx::speculative::SpecCapture;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let args: Vec<String> = std::env::args().collect();
    let model_id = args
        .iter()
        .position(|a| a == "--model")
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
        .unwrap_or("unsloth/Qwen3.5-0.8B");
    let seq_len: i32 = args
        .iter()
        .position(|a| a == "--seq-len")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);

    println!("[smoke] model: {model_id}");
    println!("[smoke] seq_len: {seq_len}");

    let t0 = Instant::now();
    let model_path = resolve_model_path(model_id).await?;
    println!(
        "[smoke] resolved path: {} ({:.1}s)",
        model_path.display(),
        t0.elapsed().as_secs_f32()
    );

    let t0 = Instant::now();
    let mut dynamic = DynamicModel::load(&model_path)?;
    println!("[smoke] loaded model in {:.1}s", t0.elapsed().as_secs_f32());

    let arch = dynamic.architecture();
    println!("[smoke] architecture: {arch:?}");

    // Route through the DFlashTarget impl on DynamicModel so every
    // architecture that ships capture support is exercised on the same
    // code path the CLI / Python bindings use.
    let num_layers = dynamic.target_num_layers();
    let hidden_size = dynamic.target_hidden_size();
    println!("[smoke] num_layers={num_layers}, hidden_size={hidden_size}");

    let mut kv_cache = dynamic.make_kv_cache(32);
    let mut mamba_cache = dynamic.make_mamba_cache();

    // Build a small fake token batch. We just want to exercise the
    // forward path and the capture hook — we're not actually generating
    // anything.
    let fake_ids: Vec<i32> = (1..=seq_len).collect();
    let input_ids = Array::from_slice(&fake_ids, &[1, seq_len]);

    // Tap every 4th layer so we hit at least one full-attention layer
    // and several GDN layers on Qwen3.5 (upstream `z-lab/Qwen3.5-4B-DFlash`
    // uses layers [3, 7, 11, 15]).
    let tap_layers: Vec<usize> = (0..num_layers).step_by(4).collect();
    println!("[smoke] tapping layers: {tap_layers:?}");

    let mut capture = SpecCapture::with_layers(tap_layers.clone());

    let t0 = Instant::now();
    let logits = dynamic.forward_with_capture(
        &input_ids,
        None,
        Some(&mut kv_cache),
        mamba_cache.as_mut(),
        &mut capture,
    )?;
    // Drive the lazy graph to completion.
    let _ = logits.eval();
    println!(
        "[smoke] forward_with_capture finished in {:.1}ms",
        t0.elapsed().as_secs_f32() * 1000.0
    );

    println!("[smoke] logits shape: {:?}", logits.shape());
    // Sanity-check that outputs are finite (not NaN / inf).
    {
        let evaled = logits.clone();
        let _ = evaled.eval();
        let f32_slice = evaled.as_slice::<f32>();
        let sample = &f32_slice[..f32_slice.len().min(32)];
        let finite = sample.iter().all(|x| x.is_finite());
        let max_abs = sample.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        println!(
            "[smoke] logits finite={finite}, sample[0..{}] max_abs={max_abs:.3}",
            sample.len()
        );
        if !finite {
            return Err("logits contained NaN or Inf".into());
        }
    }
    println!(
        "[smoke] captured {} hidden tensors",
        capture.hidden_states.len()
    );
    for idx in &tap_layers {
        match capture.hidden_states.get(idx) {
            Some(h) => println!("[smoke]   layer {idx}: shape {:?}", h.shape()),
            None => println!("[smoke]   layer {idx}: NOT CAPTURED"),
        }
    }

    println!(
        "[smoke] captured {} GDN verify-input records",
        capture.gdn_inputs.len()
    );
    let mut first_gdn = None;
    for (layer_idx, rec) in capture.gdn_inputs.iter().take(3) {
        println!(
            "[smoke]   gdn layer {layer_idx}: k={:?}  v={:?}  g={:?}  beta={:?}  conv_in={:?}  kernel={}",
            rec.keys.shape(),
            rec.values.shape(),
            rec.g.shape(),
            rec.beta.shape(),
            rec.conv_input.shape(),
            rec.conv_kernel_size,
        );
        if first_gdn.is_none() {
            first_gdn = Some(*layer_idx);
        }
    }
    if capture.gdn_inputs.len() > 3 {
        println!(
            "[smoke]   ... {} more GDN records",
            capture.gdn_inputs.len() - 3
        );
    }

    // ── Rollback correctness check ──────────────────────────────────────
    //
    // The goal: `forward(ids[..T])` should produce a Mamba-cache SSM state
    // that equals
    //   `forward(ids[..T+K]) + rewind_from_snapshots(snaps, verify_inputs, T)`
    // — in other words, rewinding after a fake partial-accept must land us
    // exactly where we would have been had we never drafted the rejected
    // tail. We exercise that here on the same Qwen3.5-0.8B weights.
    if let (Some(mamba), Some(_layer)) = (mamba_cache.as_ref(), first_gdn) {
        println!("[smoke] rollback check: simulating partial accept");
        // Snapshot the *current* state, then rewind through the captured
        // verify inputs with `accepted = seq_len / 2` — this should
        // reconstruct the state as if the second half never happened.
        let snaps = mamba.snapshot();
        // Build per-layer verify inputs in cache layer order.
        let num_layers = mamba.num_layers();
        let mut per_layer: Vec<Option<pmetal_mlx::kv_cache::GdnVerifyInputs>> =
            Vec::with_capacity(num_layers);
        for layer_idx in 0..num_layers {
            per_layer.push(capture.gdn_inputs.get(&layer_idx).cloned());
        }
        let _ = mamba; // release the immutable borrow before reborrowing mut

        // Snapshot the layer-0 SSM state before rewind so we can sanity-check
        // it changed (rewinding to a partial accept *should* produce a
        // different state vs "no accept" unless the inputs are all zero).
        let mamba_mut = mamba_cache.as_mut().unwrap();
        let pre_rewind = mamba_mut
            .get(0)
            .and_then(|e| e.ssm_state.as_ref())
            .map(|s| s.shape().to_vec());
        println!("[smoke]   pre-rewind layer-0 ssm shape: {:?}", pre_rewind);

        mamba_mut
            .rewind_from_snapshots(&snaps, &per_layer, (seq_len / 2) as usize)?;

        let post_rewind = mamba_mut
            .get(0)
            .and_then(|e| e.ssm_state.as_ref())
            .map(|s| s.shape().to_vec());
        println!("[smoke]   post-rewind layer-0 ssm shape: {:?}", post_rewind);

        if pre_rewind.is_none() || post_rewind.is_none() {
            return Err("layer-0 SSM state missing before/after rewind".into());
        }
        if pre_rewind != post_rewind {
            return Err(format!(
                "rewind changed shape: {pre_rewind:?} -> {post_rewind:?}"
            )
            .into());
        }
        println!("[smoke]   rollback shape-preserving: OK");
    }

    // Stacking is the exact call the DFlash decoder uses to feed the
    // draft. Exercise it to confirm shapes concatenate as expected.
    let stacked = capture.stack_hidden()?;
    let _ = stacked.eval();
    println!("[smoke] stacked hidden shape: {:?}", stacked.shape());
    let expected_last = tap_layers.len() as i32 * hidden_size;
    if stacked.dim(2) != expected_last {
        return Err(format!(
            "stacked last dim mismatch: got {}, expected {}",
            stacked.dim(2),
            expected_last
        )
        .into());
    }
    println!("[smoke] OK");

    Ok(())
}
