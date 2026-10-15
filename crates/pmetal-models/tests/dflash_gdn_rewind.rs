//! A speculative verify on a Qwen3.5 target (GDN + gated attention, the
//! native engine), then a rewind to a prefix of the block, leaves the target
//! where running only that prefix would have: the next token's logits match
//! a straight run over the same tokens.
//!
//! The checkpoint is the `qwen3_5_swish` parity fixture (fp32, two GDN and
//! two attention layers). A KV cache rewinds by moving its offset, a GDN
//! layer's recurrent state only by replaying the kept prefix. Rewinding the
//! attention caches alone, as the target did before, moves these logits by
//! 2.0; replaying from the block's start for the convolution state, or one
//! token short, by as much.

mod common;

use common::{fixture_path, load_shard, ref_tensor};
use pmetal_cache::speculative::SpecCapture;
use pmetal_cache::test_utils::{ParityReport, Tolerance};
use pmetal_models::dflash_decoder::DFlashTarget;
use pmetal_models::dflash_native_target::NativeQwen3Target;
use pmetal_runtime::compat::{Array, ops::slice_axis};
use serial_test::serial;

const PROMPT: i32 = 20;
const BLOCK: i32 = 8;

/// fp32, the same tokens through differently shaped forwards (a block of 8
/// against one at a time). They agree bit for bit on this checkpoint; the
/// gate leaves room for kernels that reassociate by shape.
const TOL: Tolerance = Tolerance::new(2e-5, 0.0);

struct Checkpoint {
    _dir: tempfile::TempDir,
    ids: Array,
}

fn checkpoint() -> Checkpoint {
    let dir = tempfile::tempdir().expect("tempdir");
    for (from, to) in [
        ("qwen3_5_swish_config.json", "config.json"),
        ("qwen3_5_swish_weights.safetensors", "model.safetensors"),
    ] {
        std::fs::copy(fixture_path(from), dir.path().join(to)).expect("copy fixture");
    }
    let reference = load_shard(&fixture_path("qwen3_5_swish_reference.safetensors"));
    let ids = ref_tensor(&reference, "input_ids").clone();
    Checkpoint { _dir: dir, ids }
}

fn drain(op: &str) {
    pmetal_runtime::check_unobserved_error().unwrap_or_else(|e| panic!("{op}: bridge error: {e}"));
}

fn forward(target: &mut NativeQwen3Target, ids: &Array) -> Array {
    let mut capture = SpecCapture::with_layers(vec![1]);
    let mut kv = target.make_kv_cache(1);
    target
        .forward_with_capture(ids, None, Some(&mut kv), None, &mut capture)
        .expect("forward")
}

#[test]
#[serial]
fn rewinding_a_verify_matches_running_the_kept_prefix() {
    let ck = checkpoint();
    let mut target = NativeQwen3Target::load(ck._dir.path()).expect("native target loads");
    drain("load");
    let prompt = slice_axis(&ck.ids, 1, 0, PROMPT);
    let block = slice_axis(&ck.ids, 1, PROMPT, PROMPT + BLOCK);

    for kept in [1, 3, 7, BLOCK] {
        let next = slice_axis(&ck.ids, 1, PROMPT + kept, PROMPT + kept + 1);

        // Prompt, verify the block, keep `kept` of it, then one more token.
        target.reset_state();
        forward(&mut target, &prompt);
        let mut capture = SpecCapture::with_layers(vec![1]);
        let mut kv = target.make_kv_cache(1);
        target
            .verify_with_capture(&block, Some(&mut kv), None, &mut capture)
            .expect("verify");
        target
            .rollback_rejected(&mut kv, (BLOCK - kept) as usize)
            .expect("rewind");
        let rewound = forward(&mut target, &next);
        drain("speculative");

        // The same tokens straight through.
        target.reset_state();
        let kept_ids = slice_axis(&ck.ids, 1, 0, PROMPT + kept);
        forward(&mut target, &kept_ids);
        let straight = forward(&mut target, &next);
        drain("straight");

        let report = ParityReport::compute(&format!("kept {kept}"), &rewound, &straight, TOL);
        println!(
            "kept {kept} of {BLOCK}: max |diff| {:.3e}",
            report.max_abs_diff
        );
        assert!(
            report.passed(),
            "kept {kept} of {BLOCK}: next-token logits differ by {:.3e}",
            report.max_abs_diff
        );
    }
}

/// The same contract on the `DynamicModel` Qwen3.5 target the MTP decode runs
/// on when the native engine can't: a verify records each GDN layer's
/// recurrence inputs, a partial accept replays them over the kept prefix from
/// the pre-verify snapshot, and the KV caches roll back by their offsets.
#[test]
#[serial]
fn rewinding_a_dynamic_verify_matches_running_the_kept_prefix() {
    use pmetal_models::DynamicModel;
    use pmetal_models::mtp::MtpTarget;
    use pmetal_models::qwen3_next_mtp::Qwen3NextMtpVerifier;

    let ck = checkpoint();
    let mut model = DynamicModel::load(ck._dir.path()).expect("checkpoint loads");
    drain("load");
    let tokens = |from: i32, to: i32| -> Vec<u32> {
        pmetal_cache::test_utils::to_f32_vec_eval(&slice_axis(&ck.ids, 1, from, to))
            .into_iter()
            .map(|v| v as u32)
            .collect()
    };
    let prompt = tokens(0, PROMPT);
    let block = tokens(PROMPT, PROMPT + BLOCK);

    // `first`, then optionally a verified block cut back to `kept`, then
    // `next`: the logits of `next`.
    let mut run = |first: &[u32], verified: Option<(&[u32], usize)>, next: &[u32]| {
        let mut kv = model.create_cache(128);
        let mut mamba = model
            .create_mamba_cache()
            .expect("hybrid model has a GDN cache");
        let target = model.as_qwen3_next_mut().expect("Qwen3Next");
        let mut verifier = Qwen3NextMtpVerifier::new(target, &mut kv, &mut mamba);
        verifier.forward(first).expect("prompt");
        if let Some((block, kept)) = verified {
            verifier.verify(block).expect("verify");
            verifier.keep(kept).expect("keep");
        }
        let (_, logits) = verifier.forward(next).expect("next token");
        drain("dynamic rewind");
        logits
    };

    for kept in [1usize, 3, 7, BLOCK as usize] {
        let next = tokens(PROMPT + kept as i32, PROMPT + kept as i32 + 1);
        let rewound = run(&prompt, Some((&block, kept)), &next);
        let mut straight_prefix = prompt.clone();
        straight_prefix.extend_from_slice(&block[..kept]);
        let straight = run(&straight_prefix, None, &next);

        let report = ParityReport::compute(&format!("kept {kept}"), &rewound, &straight, TOL);
        println!(
            "dynamic: kept {kept} of {BLOCK}: max |diff| {:.3e}",
            report.max_abs_diff
        );
        assert!(
            report.passed(),
            "dynamic: kept {kept} of {BLOCK}: next-token logits differ by {:.3e}",
            report.max_abs_diff
        );
    }
}
