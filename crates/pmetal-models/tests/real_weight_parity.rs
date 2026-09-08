//! Runs each architecture's forward pass against a **real released
//! checkpoint** and diffs it against `transformers` over the same tensors.
//!
//! This is the third and last rung of the parity ladder, and the only one that
//! can fail for a reason the other two structurally cannot:
//!
//! * The synthetic fixtures (`gemma2_parity`, `gpt_oss_parity`, …) prove the
//!   *math*, but their weights are written by the same `transformers` run that
//!   produced the reference. Both sides agree on key names by construction, so
//!   no fixture can ever catch a loader that looks for `mlp.gate_proj` in a
//!   checkpoint that ships `mlp.w1`.
//! * `real_config_parity` proves pmetal *reads* the released `config.json` the
//!   way the checkpoint states it. It never runs a forward, so a field that is
//!   parsed correctly and then applied in the wrong place is invisible to it.
//!   Granite's `attention_multiplier` was exactly that: parse it right, forget
//!   that it *replaces* `1/√head_dim` rather than composing with it.
//!
//! Both sides load the same bf16 tensors — the oracle is not upcast — so the
//! only legitimate source of divergence is accumulation order.
//!
//! How much divergence that buys is not a constant, which is why nothing here
//! is compared against a fixed tolerance. The dumper measures each model's
//! **noise floor** by running the same `transformers` code twice, once in bf16
//! and once in fp32. Those floors span two orders of magnitude: `1 - cos` is
//! 5.6e-5 for Granite and 5.5e-3 for Gemma 3, whose residual stream reaches
//! ~9.6e3 where a bf16 step is 64. Gemma 3 simply cannot reach Granite's
//! agreement in this format, and holding it to that would be measuring the
//! format rather than the port. So each checkpoint is asked the only question
//! that means anything: is pmetal as close to `transformers` as bf16 permits?
//!
//! Two taps per checkpoint, which localize a failure without needing per-arch
//! hooks:
//!
//! | trunk | logits | reading |
//! |-------|--------|---------|
//! | pass  | pass   | the architecture is right |
//! | pass  | FAIL   | the LM head, tied-weight handling, or a logit scale |
//! | FAIL  | FAIL   | the decoder stack: an embedding scale, a norm, attention |
//!
//! Checkpoints are *not* committed — they carry the upstream model licenses,
//! and they are gigabytes. So this is `#[ignore]`d and gated on
//! `PMETAL_REAL_WEIGHTS`, the same pattern as `real_config_parity` and the real
//! Mllama test. Populate it with:
//!
//! ```bash
//! HF_HOME=/Volumes/AmBa/huggingface .venv-parity/bin/python -u \
//!     .strategy/parity/dump_real_weight_reference.py \
//!     weight_models.txt /Volumes/AmBa/huggingface/parity-weights
//! PMETAL_REAL_WEIGHTS=/Volumes/AmBa/huggingface/parity-weights \
//!     cargo test -p pmetal-models --test real_weight_parity -- --ignored --nocapture
//! ```
//!
//! Everything lands on `/Volumes/AmBa`, never the boot volume.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use pmetal_bridge::compat::Array;
// Not via `common`: that module's `fixture_path` keys off `tests/fixtures/`,
// and every path here comes from `PMETAL_REAL_WEIGHTS` instead.
use pmetal_mlx::test_utils::{
    ParityReport, Tolerance, load_shard, print_report_table, ref_tensor, to_f32_vec_eval,
};
use pmetal_models::dispatcher::{DynamicModel, ModelArchitecture};
use serde::Deserialize;

/// How much further than bf16 itself pmetal is allowed to be.
///
/// A fixed cosine threshold cannot work here. The dumper measures each model's
/// *noise floor* by running the same `transformers` code twice, in bf16 and in
/// fp32, and those floors span two orders of magnitude: `1 - cos` is 5.6e-5 for
/// Granite and 5.5e-3 for Gemma 3, whose residual stream reaches ~9.6e3 where a
/// bf16 step is 64. Any single threshold is either unreachable for Gemma or
/// meaningless for Granite.
///
/// So the question asked is: *is pmetal as close to `transformers` as the
/// format permits?* Across the five architectures that pass, pmetal's own
/// `1 - cos` sits between 0.4x and 1.6x the floor. 2.5 leaves real headroom
/// above that spread while staying far below any defect this sweep has found —
/// the mildest was Gemma 3 with its QK-norm missing, at 43x.
const TOLERANCE_FACTOR: f32 = 2.5;

/// Floor for the budget itself, so a model whose bf16 and fp32 runs happen to
/// agree exactly still gets a little room rather than demanding bit equality.
const MIN_COSINE_BUDGET: f32 = 1e-4;
const MIN_REL_BUDGET: f32 = 5e-3;

/// How far down pmetal's ranking the oracle's argmax may sit.
///
/// Not a fudge factor for a weak assertion: near the top of a converged
/// distribution two tokens routinely sit within bf16 noise of each other, so
/// demanding an exact argmax match tests the format. Demanding the oracle's
/// choice stay in the top 5 tests the model, and a real defect scatters it far
/// past 5 at every position.
const MAX_RANK: usize = 5;

/// How far bf16 alone moves one tap, measured by the dumper.
#[derive(Debug, Clone, Copy, Deserialize)]
struct NoiseFloor {
    cosine: f32,
    rel_max_abs: f32,
}

impl NoiseFloor {
    /// The `1 - cos` this tap is allowed to reach.
    fn cosine_budget(&self) -> f32 {
        (1.0 - self.cosine).max(MIN_COSINE_BUDGET) * TOLERANCE_FACTOR
    }

    /// Relative `max_abs_diff` budget, as a `Tolerance` the report can use.
    fn tolerance(&self) -> Tolerance {
        Tolerance::new(0.0, self.rel_max_abs.max(MIN_REL_BUDGET) * TOLERANCE_FACTOR)
    }
}

/// One entry of `_references/_manifest.json`, written by
/// `dump_real_weight_reference.py`.
#[derive(Debug, Deserialize)]
struct RefMeta {
    arch: String,
    model_id: String,
    model_dir: String,
    seq_len: usize,
    vocab_size: i32,
    hidden_size: i32,
    /// The oracle's argmax token at each position.
    top1: Vec<i32>,
    /// Per-tap bf16-vs-fp32 divergence, keyed `trunk` / `logits`.
    noise_floor: BTreeMap<String, NoiseFloor>,
}

impl RefMeta {
    fn floor(&self, tap: &str) -> NoiseFloor {
        self.noise_floor.get(tap).copied().unwrap_or(NoiseFloor {
            cosine: 1.0,
            rel_max_abs: 0.0,
        })
    }
}

struct Finding {
    slug: String,
    detail: String,
}

/// The dumper's token ids, recomputed rather than read, so a mismatch between
/// the two sides is itself a failure the fixture cannot paper over.
fn input_ids(seq_len: usize, vocab_size: i32) -> Vec<i32> {
    (0..seq_len)
        .map(|i| ((i as i32 + 1) * 7 + 13) % vocab_size)
        .collect()
}

/// Rank of `token` in `row` — the number of logits strictly greater than it.
fn rank_of(row: &[f32], token: i32) -> usize {
    let value = row[token as usize];
    row.iter().filter(|&&v| v > value).count()
}

/// MLX ops that throw route through a thread-local error channel and return a
/// placeholder rather than unwinding, and a degenerate placeholder *broadcasts*
/// through a residual stream — leaving output that is correctly shaped, finite,
/// and plausibly scaled. Drain after every stage or the whole sweep can pass
/// while attention computes nothing.
fn drain(stage: &str) -> Result<(), String> {
    pmetal_bridge::check_last_error().map_err(|e| format!("{stage} raised a bridge exception: {e}"))
}

/// Everything one checkpoint contributes: the taps it produced, whatever went
/// wrong, and the row the summary table prints for it.
#[derive(Default)]
struct Outcome {
    reports: Vec<ParityReport>,
    details: Vec<String>,
    arch: Option<ModelArchitecture>,
    worst_rank: Option<usize>,
}

impl Outcome {
    fn failed(detail: String) -> Self {
        Self {
            details: vec![detail],
            ..Self::default()
        }
    }
}

fn check_one(slug: &str, meta: &RefMeta, refs_root: &Path) -> Outcome {
    let mut out = Outcome::default();

    let model_dir = PathBuf::from(&meta.model_dir);
    if !model_dir.exists() {
        return Outcome::failed(format!("{model_dir:?} does not exist"));
    }
    let shard = load_shard(&refs_root.join(format!("{slug}.safetensors")));

    let mut model = match DynamicModel::load(&model_dir) {
        Ok(m) => m,
        Err(e) => return Outcome::failed(format!("{} failed to load: {e}", meta.model_id)),
    };
    if let Err(e) = drain("loading") {
        return Outcome::failed(e);
    }

    // The manifest names the architecture the checkpoint is *supposed* to
    // resolve to; without this a Granite silently running as plain Llama looks
    // like a numerical failure rather than a dispatch one.
    let arch = model.architecture();
    out.arch = Some(arch);
    if format!("{arch:?}") != meta.arch {
        out.details
            .push(format!("detected {arch:?}, manifest expects {}", meta.arch));
        return out;
    }
    if model.vocab_size() != meta.vocab_size || model.hidden_size() != meta.hidden_size {
        out.details.push(format!(
            "geometry mismatch: pmetal says vocab {} hidden {}, config says vocab {} hidden {}",
            model.vocab_size(),
            model.hidden_size(),
            meta.vocab_size,
            meta.hidden_size
        ));
        return out;
    }

    let ids = input_ids(meta.seq_len, meta.vocab_size);
    let ids_array = Array::from_slice(&ids, &[1, meta.seq_len as i32]);
    let ref_ids: Vec<i32> = to_f32_vec_eval(ref_tensor(&shard, "input_ids"))
        .into_iter()
        .map(|v| v as i32)
        .collect();
    if ref_ids != ids {
        out.details.push(format!(
            "token ids drifted from the dumper: rust {ids:?} vs fixture {ref_ids:?}"
        ));
        return out;
    }

    let trunk = match model.forward_hidden(&ids_array, None) {
        Ok(t) => {
            t.eval();
            t
        }
        Err(e) => {
            out.details.push(format!("forward_hidden failed: {e}"));
            return out;
        }
    };
    if let Err(e) = drain("forward_hidden") {
        out.details.push(e);
        return out;
    }
    out.reports.push(ParityReport::compute(
        &format!("{slug}/trunk"),
        &trunk,
        ref_tensor(&shard, "trunk"),
        meta.floor("trunk").tolerance(),
    ));

    let logits = match model.forward(&ids_array, None) {
        Ok(l) => {
            l.eval();
            l
        }
        Err(e) => {
            out.details.push(format!("forward failed: {e}"));
            return out;
        }
    };
    if let Err(e) = drain("forward") {
        out.details.push(e);
        return out;
    }
    out.reports.push(ParityReport::compute(
        &format!("{slug}/logits"),
        &logits,
        ref_tensor(&shard, "logits"),
        meta.floor("logits").tolerance(),
    ));

    for (tap, report) in ["trunk", "logits"].iter().zip(&out.reports) {
        let floor = meta.floor(tap);
        let budget = floor.cosine_budget();
        let gap = 1.0 - report.cosine_similarity;
        if gap > budget {
            out.details.push(format!(
                "{} is {:.1}x further from transformers than bf16 itself: 1-cos {:.3e} \
                 against a floor of {:.3e} (budget {:.3e})",
                report.name,
                gap / (1.0 - floor.cosine).max(MIN_COSINE_BUDGET),
                gap,
                1.0 - floor.cosine,
                budget
            ));
        } else if !report.passed() {
            out.details.push(format!(
                "{} is aligned (cos {:.6}) but off by a factor: max_abs {:.3e} against ref \
                 magnitude {:.3e}, more than {TOLERANCE_FACTOR}x the bf16 floor of {:.3e}",
                report.name,
                report.cosine_similarity,
                report.max_abs_diff,
                report.max_abs_ref,
                floor.rel_max_abs
            ));
        }
    }

    // Ranking agreement, position by position. Cosine similarity is a global
    // measure and can stay high while one position is wrong; this is the local
    // one, and it is what a user of the model actually observes.
    let flat = to_f32_vec_eval(&logits);
    let vocab = meta.vocab_size as usize;
    let mut worst = (0usize, 0usize);
    for (t, &token) in meta.top1.iter().enumerate() {
        let row = &flat[t * vocab..(t + 1) * vocab];
        let rank = rank_of(row, token);
        if rank > worst.1 {
            worst = (t, rank);
        }
    }
    if worst.1 >= MAX_RANK {
        out.details.push(format!(
            "the oracle's argmax at position {} sits at rank {} in pmetal's logits \
             (allowed < {MAX_RANK})",
            worst.0, worst.1
        ));
    }

    out.worst_rank = Some(worst.1);
    out
}

/// Read the manifest the dumper wrote, or explain what is missing.
fn load_manifest() -> (PathBuf, BTreeMap<String, RefMeta>) {
    let Ok(root) = std::env::var("PMETAL_REAL_WEIGHTS") else {
        panic!("set PMETAL_REAL_WEIGHTS to a directory built by dump_real_weight_reference.py");
    };
    let refs_root = PathBuf::from(root).join("_references");
    let manifest_path = refs_root.join("_manifest.json");
    let manifest: BTreeMap<String, RefMeta> = serde_json::from_str(
        &std::fs::read_to_string(&manifest_path)
            .unwrap_or_else(|e| panic!("missing {manifest_path:?}: {e}")),
    )
    .expect("_manifest.json is slug -> reference metadata");
    assert!(
        !manifest.is_empty(),
        "manifest is empty — nothing was dumped"
    );
    (refs_root, manifest)
}

/// Every parameter the checkpoint should fill must actually be filled.
///
/// The generic loader silently skips a checkpoint key it cannot map to a
/// parameter, and leaves the unclaimed parameter at its random init. That is
/// invisible to every other check here: a randomly-initialised tensor is
/// finite, correctly shaped, and plausibly scaled.
///
/// So load each checkpoint twice under different seeds. A fully-loaded model is
/// a pure function of its weights, and both runs agree bit for bit. Anything
/// left at init moves, and the size of the movement says how much of the model
/// is not really loaded.
#[test]
#[ignore = "requires PMETAL_REAL_WEIGHTS and multi-GB checkpoints; see the module docs"]
fn released_checkpoints_leave_no_parameter_at_random_init() {
    use pmetal_bridge::compat::random;

    let (_refs_root, manifest) = load_manifest();
    let mut findings = Vec::new();

    for (slug, meta) in &manifest {
        let ids = input_ids(meta.seq_len, meta.vocab_size);
        let ids_array = Array::from_slice(&ids, &[1, meta.seq_len as i32]);

        // Both taps, because *which* one moves says where the hole is: a trunk
        // that holds still while the logits move is an unloaded LM head, and
        // both moving is the decoder stack.
        let run = |seed: u64| -> Option<(Vec<f32>, Vec<f32>)> {
            random::seed(seed);
            let mut model = DynamicModel::load(&meta.model_dir).ok()?;
            let trunk = model.forward_hidden(&ids_array, None).ok()?;
            let logits = model.forward(&ids_array, None).ok()?;
            Some((to_f32_vec_eval(&trunk), to_f32_vec_eval(&logits)))
        };
        let (Some(a), Some(b)) = (run(11), run(9_973)) else {
            findings.push(format!("{slug}: could not run both seeds"));
            continue;
        };

        let moved = |x: &[f32], y: &[f32]| x.iter().zip(y).filter(|(p, q)| p != q).count();
        let (trunk_moved, logits_moved) = (moved(&a.0, &b.0), moved(&a.1, &b.1));
        if trunk_moved > 0 || logits_moved > 0 {
            findings.push(format!(
                "{slug}: the init seed changes {trunk_moved}/{} trunk values and \
                 {logits_moved}/{} logits — the loader left parameters at random init",
                a.0.len(),
                a.1.len()
            ));
        } else {
            println!("  ok    {slug:44}  seed-invariant");
        }
    }

    assert!(
        findings.is_empty(),
        "\n{} checkpoint(s) are not fully loaded:\n  {}\n",
        findings.len(),
        findings.join("\n  ")
    );
}

#[test]
#[ignore = "requires PMETAL_REAL_WEIGHTS and multi-GB checkpoints; see the module docs"]
fn released_checkpoints_forward_the_way_transformers_does() {
    let (refs_root, manifest) = load_manifest();

    let mut findings: Vec<Finding> = Vec::new();
    let mut reports = Vec::new();
    let mut rows = Vec::new();
    for (slug, meta) in &manifest {
        let outcome = check_one(slug, meta, &refs_root);
        let status = if outcome.details.is_empty() {
            "ok"
        } else {
            "FAIL"
        };
        let arch = outcome
            .arch
            .map(|a| format!("{a:?}"))
            .unwrap_or_else(|| "-".to_string());
        let rank = outcome
            .worst_rank
            .map(|r| r.to_string())
            .unwrap_or_else(|| "-".to_string());
        rows.push(format!(
            "  {status:4}  {slug:44}  {arch:12}  worst rank {rank}"
        ));
        reports.extend(outcome.reports);
        findings.extend(outcome.details.into_iter().map(|detail| Finding {
            slug: slug.clone(),
            detail,
        }));
    }

    println!("\nreal released checkpoints, {} models:", manifest.len());
    print_report_table(&reports);
    for row in &rows {
        println!("{row}");
    }

    if !findings.is_empty() {
        let mut msg = format!(
            "\n{} divergence(s) from transformers on real weights:\n",
            findings.len()
        );
        for f in &findings {
            msg.push_str(&format!("  {}: {}\n", f.slug, f.detail));
        }
        panic!("{msg}");
    }
}
