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
//! Three taps per checkpoint, which localize a failure without needing
//! per-arch hooks:
//!
//! | trunk | logits | reading |
//! |-------|--------|---------|
//! | pass  | pass   | the architecture is right |
//! | pass  | FAIL   | the LM head, tied-weight handling, or a logit scale |
//! | FAIL  | FAIL   | the decoder stack: an embedding scale, a norm, attention |
//!
//! The third is `trunk_long`, the same trunk over 640 tokens instead of 16.
//! It is the only run here that crosses a sliding-window boundary: at 16
//! tokens a local layer and a global one produce identical masks, so an
//! architecture that gets the local/global split wrong — Gemma 3 alternates
//! five to one — looks perfect. Models whose window is larger than 640
//! (Gemma 2 and Mistral state 4096) still run it, and the summary row says so
//! rather than implying coverage they do not have.
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
/// format permits?* This is the aggregate gate, over the whole tensor. It is
/// deliberately the looser of the two: summed over a sequence, a model that
/// merely drifts and a model that is wrong at one position land in the same
/// place — Mistral (correct, its worst position *is* the floor's worst
/// position) and Gemma 3 (a real defect) both sit near 3x. The sharp
/// discrimination is [`POSITION_TOLERANCE_FACTOR`]; this catches whole-tensor
/// errors like a forgotten scale that no single position would reveal.
const TOLERANCE_FACTOR: f32 = 5.0;

/// How much further than the floor pmetal may be *at any single position*.
///
/// Position-matched against the floor's own profile, this is the measurement
/// that separates drift from divergence, and the observed spread has a wide
/// gap in the middle. Architectures that agree with `transformers`: Qwen3Next
/// 1x, Qwen3 2x, Granite 3x, Mistral 3x, Qwen2.5 4x, Gemma 2 4x, Phi-3 8x.
/// Architectures with a real defect: Llama 3.2 127x (no llama3 RoPE scaling),
/// Phi-4-mini 598x, Gemma 3 731x (past its sliding window). 20 sits in the gap
/// with more than a factor of two of room on either side.
const POSITION_TOLERANCE_FACTOR: f32 = 20.0;

/// Floor for the budget itself, so a model whose bf16 and fp32 runs happen to
/// agree exactly still gets a little room rather than demanding bit equality.
const MIN_COSINE_BUDGET: f32 = 1e-4;
const MIN_REL_BUDGET: f32 = 5e-3;

/// Stand-in floor for a checkpoint whose fp32 reference could not be run.
///
/// DeepSeek-V2-Lite is 63 GB in fp32 and its load is SIGKILLed here, so its
/// `noise_floor` is empty. Reading an absent floor as a *perfect* one would
/// hold it to the tightest budget in the sweep and fail it for no reason, so
/// these are the loosest floors any measured model produced (Gemma 3's), and
/// the summary row says the budget is a stand-in. It still catches everything
/// this sweep has found — the mildest real defect was 43x its own floor.
const UNMEASURED_FLOOR: NoiseFloor = NoiseFloor {
    cosine: 0.9945,
    rel_max_abs: 0.17,
};

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
    /// Second, longer run, present to cross a sliding-window boundary. `None`
    /// when the dumper could not run it — NemotronH's naive SSM fallback
    /// materializes the whole scan and does not fit at 640 tokens.
    long_seq_len: Option<usize>,
    /// The window this checkpoint states, if any. Only used to report whether
    /// the long run actually reached it.
    sliding_window: Option<usize>,
    vocab_size: i32,
    hidden_size: i32,
    /// The oracle's argmax token at each position.
    top1: Vec<i32>,
    /// Per-tap bf16-vs-fp32 divergence, keyed `trunk` / `trunk_long` / `logits`.
    noise_floor: BTreeMap<String, NoiseFloor>,
    /// False when the fp32 reference could not be run, so `noise_floor` is
    /// empty by declaration rather than by omission.
    noise_floor_measured: bool,
    /// Whether the shard carries `floor_pos_cos_long`, the floor's per-position
    /// profile over the long run.
    has_position_floor: bool,
}

impl RefMeta {
    fn floor(&self, tap: &str) -> NoiseFloor {
        if !self.noise_floor_measured {
            return UNMEASURED_FLOOR;
        }
        self.noise_floor.get(tap).copied().unwrap_or(NoiseFloor {
            cosine: 1.0,
            rel_max_abs: 0.0,
        })
    }

    /// Whether the long run is long enough to cross this model's window.
    ///
    /// Reported rather than asserted: Gemma 2 and Mistral state 4096, so 640
    /// tokens never reach their boundary and claiming otherwise would be
    /// coverage theatre.
    fn long_run_crosses_window(&self) -> bool {
        match (self.sliding_window, self.long_seq_len) {
            (Some(w), Some(len)) => w > 0 && len > w,
            _ => false,
        }
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

/// The position where pmetal is furthest past the floor, and by what multiple.
///
/// Both series are `1 - cos` at the same position, so this asks the same
/// question the aggregate gate asks, one token at a time. Positions where the
/// floor itself is essentially perfect get `MIN_COSINE_BUDGET` as a
/// denominator, so a single well-behaved token cannot manufacture a huge ratio.
fn worst_position_ratio(pmetal: &[f32], floor: &[f32]) -> Option<(usize, f32)> {
    pmetal
        .iter()
        .zip(floor)
        .enumerate()
        .map(|(t, (got, base))| (t, (1.0 - got) / (1.0 - base).max(MIN_COSINE_BUDGET)))
        .max_by(|a, b| a.1.total_cmp(&b.1))
}

/// Hand MLX's buffer cache back to the OS between checkpoints.
///
/// Dropping a `DynamicModel` frees its arrays, but MLX keeps the underlying
/// allocations in a cache for reuse. Across a sweep that loads a dozen models
/// in one process — DeepSeek-V2-Lite alone is 31 GB, NemotronH another 16 —
/// that accumulates until the test is SIGKILLed, which is exactly what
/// happened before this call existed.
fn release_mlx_memory() {
    pmetal_bridge::inline_array::clear_cache();
}

/// Cosine similarity per sequence position, over `hidden` values each.
///
/// One materialisation of each tensor and then plain arithmetic. The MLX-side
/// equivalent slices per position, which at 640 positions next to a resident
/// multi-billion-parameter model is enough allocation to be killed for.
fn per_position_cosine(rust: &Array, reference: &Array, hidden: usize) -> Vec<f32> {
    let (a, b) = (to_f32_vec_eval(rust), to_f32_vec_eval(reference));
    let positions = a.len().min(b.len()) / hidden.max(1);
    (0..positions)
        .map(|t| {
            let (x, y) = (
                &a[t * hidden..(t + 1) * hidden],
                &b[t * hidden..(t + 1) * hidden],
            );
            let dot: f32 = x.iter().zip(y).map(|(p, q)| p * q).sum();
            let nx: f32 = x.iter().map(|p| p * p).sum::<f32>().sqrt();
            let ny: f32 = y.iter().map(|q| q * q).sum::<f32>().sqrt();
            if nx == 0.0 || ny == 0.0 {
                1.0
            } else {
                dot / (nx * ny)
            }
        })
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
    /// From the long run: the worst position and its cosine.
    long_shape: Option<(usize, f32)>,
    /// Worst position-matched multiple of the floor, and where.
    long_ratio: Option<(usize, f32)>,
}

impl Outcome {
    fn failed(detail: String) -> Self {
        Self {
            details: vec![detail],
            ..Self::default()
        }
    }
}

/// One checkpoint, with MLX's buffer cache released afterwards whichever way
/// the check exits — including the early returns, where a model that failed
/// halfway through is still holding gigabytes.
fn check_one(slug: &str, meta: &RefMeta, refs_root: &Path) -> Outcome {
    let outcome = check_one_inner(slug, meta, refs_root);
    release_mlx_memory();
    outcome
}

fn check_one_inner(slug: &str, meta: &RefMeta, refs_root: &Path) -> Outcome {
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

    // Both runs' ids are recomputed rather than read, so a drift between the
    // two sides is itself a failure the fixture cannot paper over.
    let id_checks: Vec<(&str, usize)> = std::iter::once(("input_ids", meta.seq_len))
        .chain(meta.long_seq_len.map(|len| ("input_ids_long", len)))
        .collect();
    for (key, len) in id_checks {
        let ids = input_ids(len, meta.vocab_size);
        let from_fixture: Vec<i32> = to_f32_vec_eval(ref_tensor(&shard, key))
            .into_iter()
            .map(|v| v as i32)
            .collect();
        if from_fixture != ids {
            out.details.push(format!(
                "{key} drifted from the dumper: rust starts {:?}, fixture starts {:?}",
                &ids[..ids.len().min(6)],
                &from_fixture[..from_fixture.len().min(6)]
            ));
            return out;
        }
    }

    let short = input_ids(meta.seq_len, meta.vocab_size);
    let short_array = Array::from_slice(&short, &[1, meta.seq_len as i32]);
    let long_array = meta.long_seq_len.map(|len| {
        let long = input_ids(len, meta.vocab_size);
        Array::from_slice(&long, &[1, len as i32])
    });

    let mut taps: Vec<&'static str> = Vec::new();

    // The trunk at both lengths. The long one is the only thing here that
    // crosses a sliding-window boundary — at 16 tokens a local layer and a
    // global one mask identically, so an arch that gets the split wrong looks
    // perfect.
    let trunk_runs: Vec<(&'static str, &Array)> = std::iter::once(("trunk", &short_array))
        .chain(long_array.as_ref().map(|a| ("trunk_long", a)))
        .collect();
    for (tap, ids_array) in trunk_runs {
        let hidden = match model.forward_hidden(ids_array, None) {
            Ok(h) => {
                h.eval();
                h
            }
            Err(e) => {
                out.details
                    .push(format!("forward_hidden ({tap}) failed: {e}"));
                return out;
            }
        };
        if let Err(e) = drain("forward_hidden") {
            out.details.push(e);
            return out;
        }
        let name = format!("{slug}/{tap}");
        let reference = ref_tensor(&shard, tap);
        if tap == "trunk_long" {
            // *Where* the long run diverges is the whole question: a wrong
            // sliding-window mask shows up as a step at the boundary, while
            // accumulation drifts smoothly from position zero.
            //
            // Computed over flat f32 vectors rather than through
            // `ParityReport::compute_with_per_position`, which slices the MLX
            // array twice per position. At 640 positions with a 31 GB DeepSeek
            // resident that is enough allocation churn to get SIGKILLed.
            let per_pos = per_position_cosine(&hidden, reference, meta.hidden_size as usize);
            out.long_shape = per_pos
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, c)| (i, *c));

            // Position-matched against the floor's own profile, which is the
            // only comparison that separates drift from divergence. Aggregated
            // over the sequence Mistral and Gemma 3 both sit near 3x their
            // floor; matched position by position, Mistral's worst position is
            // the floor's worst position at the same value, while Gemma 3 is
            // 950x the floor at t=637. One is the format, the other is a bug.
            if meta.has_position_floor {
                let floor_pos = to_f32_vec_eval(ref_tensor(&shard, "floor_pos_cos_long"));
                if let Some((t, ratio)) = worst_position_ratio(&per_pos, &floor_pos) {
                    out.long_ratio = Some((t, ratio));
                    if ratio > POSITION_TOLERANCE_FACTOR {
                        out.details.push(format!(
                            "{slug}/trunk_long diverges at position {t}: {ratio:.0}x the bf16 \
                             floor there (pmetal 1-cos {:.3e}, floor {:.3e})",
                            1.0 - per_pos[t],
                            1.0 - floor_pos[t]
                        ));
                    }
                }
            }
        }
        out.reports.push(ParityReport::compute(
            &name,
            &hidden,
            reference,
            meta.floor(tap).tolerance(),
        ));
        taps.push(tap);
    }

    let logits = match model.forward(&short_array, None) {
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
    taps.push("logits");

    for (tap, report) in taps.iter().zip(&out.reports) {
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
    drop(model);
    release_mlx_memory();
    out
}

/// Collapse `layers.7.` to `layers.N.` so a 40-layer stack is one entry.
fn collapse_indices(key: &str) -> String {
    key.split('.')
        .map(|part| {
            if !part.is_empty() && part.chars().all(|c| c.is_ascii_digit()) {
                "N"
            } else {
                part
            }
        })
        .collect::<Vec<_>>()
        .join(".")
}

/// Checkpoint tensor patterns that legitimately have no parameter behind them.
///
/// Kept deliberately short. Every entry is a statement that pmetal does not
/// need the tensor, and the whole value of the check is that adding a line here
/// is more uncomfortable than fixing the mapping.
fn unclaimed_is_expected(key: &str) -> bool {
    // A tied head ships no `lm_head.weight`; when a checkpoint carries one
    // anyway the embedding is used instead and the extra copy is redundant.
    key == "lm_head.weight"
        // Buffers, not parameters: recomputed from the config every load.
        || key.ends_with(".rotary_emb.inv_freq")
        || key.ends_with(".attention.masked_bias")
        || key.ends_with(".attention.bias")
        // Non-language towers. pmetal runs the language stack of a multimodal
        // wrapper and drops the rest by design, which is a scope decision
        // rather than a mapping gap.
        || ["visual.", "vision_tower.", "vision_model.", "audio_tower.",
            "multi_modal_projector.", "embed_vision.", "embed_audio."]
            .iter()
            .any(|tower| key.contains(tower))
}

/// The loader's key rewrites, applied so the comparison sees the names the
/// loader actually assigns.
///
/// Multimodal wrappers nest the text stack under `model.language_model.` and
/// the dispatcher strips the infix back to `model.` before assignment. DeepSeek
/// goes further and renames its whole mixture; that rewrite is the loader's own
/// `deepseek_param_name`, called here rather than restated, so the check cannot
/// drift from what actually runs.
fn loader_normalized(arch: ModelArchitecture, key: &str) -> String {
    let stripped = key.replace("model.language_model.", "model.");
    let renamed = if arch == ModelArchitecture::DeepSeek {
        pmetal_models::loader::deepseek_param_name(&stripped).unwrap_or(stripped)
    } else {
        stripped
    };
    collapse_indices(&renamed)
}

/// Whether this architecture assigns weights by matching parameter names
/// against checkpoint names, which is the premise the tensor-claim check rests
/// on.
///
/// Several architectures do not. Gemma walks an explicit key map in
/// `load_gemma_weights` (its layers live behind a `gemma1`/`gemma2` enum, so
/// the parameter path is `layers.gemma2.N.…` and matches nothing by name), Phi
/// splits a fused `qkv_proj` before assignment, and NemotronH and Qwen3Next
/// have bespoke loaders of their own. For those, a name difference is the
/// design rather than a defect, and reporting it would be noise that trains
/// the reader to ignore this test.
///
/// The narrower claim is still worth making: the generic path is exactly where
/// an unmatched key is dropped in silence.
fn uses_generic_name_matching(arch: ModelArchitecture) -> bool {
    use ModelArchitecture as A;
    matches!(
        arch,
        A::Llama | A::Qwen2 | A::Qwen3 | A::Qwen3MoE | A::Mistral | A::Granite | A::DeepSeek
    )
}

/// Tensor names in a checkpoint directory, from the shard index when there is
/// one and from the safetensors header otherwise.
///
/// Reading the header directly rather than loading the shard: half of these
/// checkpoints are a single `model.safetensors` with no index, and the point is
/// the key set, not the tensors. The format is an 8-byte little-endian header
/// length followed by that many bytes of JSON.
fn checkpoint_tensor_names(model_dir: &Path) -> Option<Vec<String>> {
    let index = model_dir.join("model.safetensors.index.json");
    if let Ok(raw) = std::fs::read_to_string(&index) {
        let json: serde_json::Value = serde_json::from_str(&raw).ok()?;
        return Some(json["weight_map"].as_object()?.keys().cloned().collect());
    }

    let single = model_dir.join("model.safetensors");
    let bytes = std::fs::read(&single).ok()?;
    let len = u64::from_le_bytes(bytes.get(..8)?.try_into().ok()?) as usize;
    let header: serde_json::Value = serde_json::from_slice(bytes.get(8..8 + len)?).ok()?;
    Some(
        header
            .as_object()?
            .keys()
            .filter(|k| *k != "__metadata__")
            .cloned()
            .collect(),
    )
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
            let taps = (to_f32_vec_eval(&trunk), to_f32_vec_eval(&logits));
            drop(model);
            release_mlx_memory();
            Some(taps)
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

/// Every tensor the checkpoint ships must be claimed by some parameter.
///
/// This is the complement of the seed test, and it is not redundant with it.
/// `assign_loaded_weights` matches by exact name and drops non-matches in both
/// directions, so there are two distinct failures:
///
/// * a **parameter** nothing fills, which stays at random init — the seed test
///   catches that, and it is how Granite's unmapped attention surfaced;
/// * a **checkpoint tensor** nothing claims, which is a feature pmetal simply
///   does not have. The seed test is blind to it, and so is every forward
///   comparison that happens to still look plausible.
///
/// The second kind is what this catches, and it has found two so far: Gemma 3's
/// `q_norm` / `k_norm`, and NemotronH's entire Mamba2 SSM parameter set
/// (`A_log`, `D`, `dt_bias`, the gated norm) — a model whose every parameter
/// loaded and whose output was still noise.
#[test]
#[ignore = "requires PMETAL_REAL_WEIGHTS and multi-GB checkpoints; see the module docs"]
fn released_checkpoints_claim_every_tensor() {
    use pmetal_bridge::compat::ModuleParametersExt;
    use std::collections::BTreeSet;

    let (_refs_root, manifest) = load_manifest();
    let mut findings = Vec::new();

    for (slug, meta) in &manifest {
        let model = match DynamicModel::load(&meta.model_dir) {
            Ok(m) => m,
            Err(e) => {
                findings.push(format!("{slug}: failed to load: {e}"));
                continue;
            }
        };
        let arch = model.architecture();
        let params: BTreeSet<String> = model
            .flatten_params()
            .keys()
            .map(|k| collapse_indices(k))
            .collect();
        drop(model);
        release_mlx_memory();

        if !uses_generic_name_matching(arch) {
            println!("  n/a   {slug:44}  {arch:?} maps keys explicitly");
            continue;
        }

        let Some(names) = checkpoint_tensor_names(Path::new(&meta.model_dir)) else {
            findings.push(format!("{slug}: could not read any tensor names"));
            continue;
        };
        let unclaimed: BTreeSet<String> = names
            .iter()
            .filter(|k| !unclaimed_is_expected(k))
            .map(|k| loader_normalized(arch, k))
            .filter(|k| !params.contains(k))
            .collect();

        if unclaimed.is_empty() {
            println!("  ok    {slug:44}  every tensor claimed");
        } else {
            findings.push(format!(
                "{slug}: {} tensor pattern(s) no parameter claims:\n      {}",
                unclaimed.len(),
                unclaimed.into_iter().collect::<Vec<_>>().join("\n      ")
            ));
        }
    }

    assert!(
        findings.is_empty(),
        "\n{} checkpoint(s) ship tensors pmetal has nowhere to put:\n  {}\n",
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
        // Say plainly whether the long run reached this model's window rather
        // than letting a green row imply sliding-window coverage it lacks.
        let window = match (meta.sliding_window, meta.long_seq_len) {
            _ if meta.long_run_crosses_window() => {
                format!("window {} crossed", meta.sliding_window.unwrap_or(0))
            }
            (_, None) => "no long run".to_string(),
            (Some(w), Some(len)) => format!("window {w} NOT reached at {len}"),
            (None, Some(_)) => "no window".to_string(),
        };
        let floor = if meta.noise_floor_measured {
            ""
        } else {
            "  [stand-in floor]"
        };
        let long = match outcome.long_shape {
            Some((pos, cos)) => format!(
                "  long worst t={pos} cos {cos:.5}{}",
                match outcome.long_ratio {
                    Some((t, r)) => format!(", {r:.0}x floor at t={t}"),
                    None => String::new(),
                }
            ),
            _ => String::new(),
        };
        rows.push(format!(
            "  {status:4}  {slug:44}  {arch:12}  worst rank {rank:6}  {window}{floor}{long}"
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
