//! DFlash 2 draft parity against the reference implementation's PyTorch
//! `DFlash2DraftModel` (z-lab/dflash), dumped by
//! `.strategy/parity/dump_dflash2_reference.py` on a tiny random config.
//!
//! The draft drafts five blocks in a row against a context that grows by the
//! prompt, then by each step's accepted tokens, the way speculative decoding
//! drives it, with its context carried by the draft cache. The window (12)
//! is small enough that the third draft's block reaches past it and the
//! fourth's context alone overflows it, so the sliding cache trims. Each
//! step's draft hidden states, logits and the candidate selector's path must
//! match the reference: the path exactly, the rest to fp32 noise.
//!
//! The checkpoint loads through the production loader, which is strict, so a
//! tensor it misnames or leaves unfilled fails here too.

mod common;

use std::collections::HashMap;

use common::{fixture_path, load_shard, ref_tensor};
use pmetal_bridge::compat::{Array, Dtype, ops};
use pmetal_mlx::test_utils::{ParityReport, Tolerance, print_report_table};
use pmetal_models::dflash_decoder::DFlashDraftQuant;
use pmetal_models::dflash_drafts::{DFlashDraft, load_dflash_draft};
use serial_test::serial;

/// fp32 against fp32. The reference's own fp32-vs-fp64 noise is 9e-7 on the
/// hidden states and 3.5e-6 on the logits (the fixture's meta); the gates are
/// ~10x that. The selector's smallest winning margin along any path is 2e-2,
/// so the exact path comparison isn't a photo finish.
const HIDDEN_TOL: Tolerance = Tolerance::new(1e-5, 0.0);
const LOGITS_TOL: Tolerance = Tolerance::new(4e-5, 0.0);

const STEPS: usize = 5;

struct Fixture {
    _dir: tempfile::TempDir,
    reference: HashMap<String, Array>,
}

impl Fixture {
    fn load() -> Self {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::copy(
            fixture_path("dflash2_config.json"),
            dir.path().join("config.json"),
        )
        .expect("copy config");
        std::fs::copy(
            fixture_path("dflash2_weights.safetensors"),
            dir.path().join("model.safetensors"),
        )
        .expect("copy weights");
        let reference = load_shard(&fixture_path("dflash2_reference.safetensors"));
        Self {
            _dir: dir,
            reference,
        }
    }

    fn draft(&self) -> DFlashDraft {
        load_dflash_draft(self._dir.path(), DFlashDraftQuant::None).expect("draft loads")
    }

    fn tensor(&self, key: &str) -> Array {
        ref_tensor(&self.reference, key).clone()
    }

    fn anchor(&self, step: usize) -> i32 {
        let anchors = self.tensor("anchors");
        let _ = anchors.eval();
        anchors.as_slice::<i32>()[step]
    }

    /// The target's embedding of block `step`: `[anchor, mask, ...]`.
    fn block_embedding(&self, draft: &DFlashDraft, step: usize) -> Array {
        let mut ids = vec![draft.mask_token_id(); draft.block_size()];
        ids[0] = self.anchor(step);
        let ids = Array::from_slice(&ids, &[ids.len() as i32]);
        let embed = self.tensor("target.embed_tokens");
        embed
            .take_axis(&ids, 0)
            .reshape(&[1, ids.dim(0), embed.dim(1)])
    }

    fn lm_head(&self, hidden: &Array) -> Array {
        hidden.matmul(&self.tensor("target.lm_head").t())
    }

    fn path(&self, step: usize) -> Vec<i32> {
        ints(&self.tensor(&format!("step{step}.path")))
    }
}

fn ints(a: &Array) -> Vec<i32> {
    let a = a.as_dtype(Dtype::Int32.as_i32());
    let _ = a.eval();
    a.as_slice::<i32>().to_vec()
}

fn drain_bridge(op: &str) {
    pmetal_bridge::check_last_error().unwrap_or_else(|e| panic!("{op}: bridge error: {e}"));
}

/// Hidden states, logits and the selector's path at every step, drafted
/// through the model's own pieces so each is compared on its own.
#[test]
#[serial]
fn dflash2_draft_matches_reference_step_by_step() {
    let fx = Fixture::load();
    let mut draft = fx.draft();
    let mut cache = draft.make_cache(64);
    let DFlashDraft::V2(model) = &mut draft else {
        panic!("a DFlash2DraftModel config loads as DFlash 2");
    };
    assert_eq!(model.block_size(), 4, "block size comes from dflash_config");

    let mut reports = Vec::new();
    let mut selector_reorders = false;
    for step in 0..STEPS {
        let mut ids = vec![model.mask_token_id(); model.block_size()];
        ids[0] = fx.anchor(step);
        let embed = fx.tensor("target.embed_tokens");
        let noise = embed
            .take_axis(&Array::from_slice(&ids, &[ids.len() as i32]), 0)
            .reshape(&[1, ids.len() as i32, embed.dim(1)]);
        let context = fx.tensor(&format!("step{step}.context"));
        let hidden = model
            .forward(&noise, &context, Some(&mut cache))
            .expect("draft forward");
        let guesses = ops::slice_axis(&hidden, 1, 1, hidden.dim(1));
        let logits = model.config.scale_logits(fx.lm_head(&guesses));
        let anchor = Array::from_slice(&[fx.anchor(step)], &[1]);
        let path = model.candidate_selector.select(&guesses, &logits, &anchor);
        drain_bridge(&format!("step {step}"));

        reports.push(ParityReport::compute(
            &format!("step{step}.hidden"),
            &guesses,
            &fx.tensor(&format!("step{step}.hidden")),
            HIDDEN_TOL,
        ));
        reports.push(ParityReport::compute(
            &format!("step{step}.logits"),
            &logits,
            &fx.tensor(&format!("step{step}.logits")),
            LOGITS_TOL,
        ));
        assert_eq!(
            ints(&path),
            fx.path(step),
            "step {step}: the selector's path differs from the reference"
        );
        selector_reorders |= ints(&ops::argmax_axis(&logits, -1)) != fx.path(step);
    }
    println!("\n== DFlash 2 draft vs reference ==");
    print_report_table(&reports);
    let failed: Vec<&str> = reports
        .iter()
        .filter(|r| !r.passed())
        .map(|r| r.name.as_str())
        .collect();
    assert!(failed.is_empty(), "{failed:?} out of tolerance");
    assert!(
        selector_reorders,
        "the fixture never has the selector overrule the argmax, so it can't catch a selector bug"
    );
}

/// The same drafts through [`DFlashDraft::propose`], what the decoder and
/// the drafter call: the embedding scale, the LM head and the path's
/// truncation to the guesses wanted.
#[test]
#[serial]
fn dflash2_propose_matches_reference_paths() {
    let fx = Fixture::load();
    let mut draft = fx.draft();
    let mut cache = draft.make_cache(64);
    for step in 0..STEPS {
        let embedding = fx.block_embedding(&draft, step);
        let context = fx.tensor(&format!("step{step}.context"));
        // The last step asks for fewer guesses than the block holds.
        let guesses = if step == STEPS - 1 { 2 } else { 3 };
        let path = draft
            .propose(
                &embedding,
                fx.anchor(step),
                &context,
                &mut cache,
                &mut |hidden| Ok(fx.lm_head(hidden)),
                guesses,
            )
            .expect("propose");
        drain_bridge(&format!("propose step {step}"));
        assert_eq!(
            ints(&path),
            fx.path(step)[..guesses],
            "step {step}: propose's path differs from the reference"
        );
    }
}

/// The loader refuses a checkpoint missing a tensor, rather than leaving the
/// parameter at its initial value.
#[test]
#[serial]
fn dflash2_loader_is_strict() {
    let fx = Fixture::load();
    let dir = fx._dir.path();
    let mut weights = load_shard(&dir.join("model.safetensors"));
    weights
        .remove("layers.1.mlp_conv.base_kernel")
        .expect("fixture has the tensor");
    let arrays: Vec<(&str, &Array)> = weights.iter().map(|(k, v)| (k.as_str(), v)).collect();
    Array::save_safetensors(dir.join("model.safetensors").to_str().unwrap(), &arrays);
    let err = load_dflash_draft(dir, DFlashDraftQuant::None).unwrap_err();
    assert!(
        err.to_string().contains("layers.1.mlp_conv.base_kernel"),
        "the error names the missing tensor: {err}"
    );
}
