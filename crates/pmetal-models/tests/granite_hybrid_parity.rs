//! Numerical parity for the Granite MoE and hybrid families against the
//! authoritative Hugging Face `transformers` oracle
//! (`GraniteMoeHybridForCausalLM`, `GraniteMoeForCausalLM`,
//! `GraniteMoeSharedForCausalLM`).
//!
//! Each fixture is a whole checkpoint directory, dumped by
//! `.strategy/parity/dump_granite_hybrid_reference.py`: a `config.json` and an
//! fp32 `model.safetensors` under the key names IBM's checkpoints ship
//! (`block_sparse_moe.input_linear.weight`, `router.layer.weight`,
//! `mamba.A_log`, the conv as `[C, 1, K]`). The model is loaded through
//! `DynamicModel::load`, the production path, which refuses a checkpoint it
//! cannot place in full.
//!
//! One profile per feature combination a release uses:
//!
//! * `hybrid_moe`: Granite 4.0-H Tiny / Small. NoPE, Mamba-2 + attention,
//!   routed experts summed with a shared MLP, tied head.
//! * `hybrid_dense`: Granite 4.0-H 350M / 1B / Micro. NoPE, Mamba-2 +
//!   attention, shared MLP only. Two Mamba groups (no release uses more than
//!   one) so the group broadcast and the full-width gated norm are pinned, and
//!   an untied head.
//! * `dense_rope`: Granite 4.0 350M / 1B / Micro. Every layer attention, RoPE,
//!   shared MLP only.
//! * `moe`: Granite 3.x `a400m` / `a800m`. RoPE, experts, no shared MLP.
//! * `moe_shared`: `granitemoeshared`, which IBM has not released.
//!
//! Every profile is checked on an uncached prefill, per layer, and on a cached
//! decode: 24 tokens of prefill, then the remaining 16 one at a time, each
//! step against the reference's row for that position. The Mamba chunk is 16
//! tokens against a 40-token sequence, so the prefill crosses a chunk boundary
//! and the decode carries conv and SSM state across calls.

mod common;

use std::collections::HashMap;
use std::path::Path;

use common::{fixture_path, load_shard, ref_tensor};
use pmetal_bridge::compat::{Array, Module, ops::slice_axis};
use pmetal_mlx::test_utils::{ParityReport, Tolerance, argmax_last_axis, print_report_table};
use pmetal_models::DynamicModel;
use serial_test::serial;

/// Tokens fed as one prefill before the cached decode takes over.
const PREFILL: i32 = 24;

/// fp32 against fp32. The reference's own fp32-vs-fp64 noise on these logits is
/// 0.8-2.8e-7 (recorded in each fixture's meta), and pmetal lands at or under
/// 1.8e-7 on every logit row and 7.2e-7 on every hidden state (magnitude ~3),
/// prefill and decode, every profile. The gate is ~3x the hidden-state figure.
/// Each of the mutations this suite was checked against (per-group gated norm,
/// no `D` skip, state dropped at a chunk boundary, conv history not carried,
/// RoPE on a NoPE model, shared MLP dropped, router weights left unnormalized,
/// gate/up halves swapped) moves something by 1e-4 or more.
const TOL: Tolerance = Tolerance::new(2e-6, 0.0);

struct Fixture {
    _dir: tempfile::TempDir,
    reference: HashMap<String, Array>,
}

impl Fixture {
    fn path(&self) -> &Path {
        self._dir.path()
    }

    fn tensor(&self, key: &str) -> Array {
        ref_tensor(&self.reference, key).clone()
    }

    fn input_ids(&self) -> Array {
        self.tensor("input_ids")
    }
}

/// Lay the profile out as a checkpoint directory: `config.json` +
/// `model.safetensors`, nothing else.
fn checkpoint(profile: &str) -> Fixture {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::copy(
        fixture_path(&format!("granite_{profile}_config.json")),
        dir.path().join("config.json"),
    )
    .expect("copy config");
    std::fs::copy(
        fixture_path(&format!("granite_{profile}_weights.safetensors")),
        dir.path().join("model.safetensors"),
    )
    .expect("copy weights");
    let reference = load_shard(&fixture_path(&format!(
        "granite_{profile}_reference.safetensors"
    )));
    Fixture {
        _dir: dir,
        reference,
    }
}

fn rows(a: &Array, start: i32, end: i32) -> Array {
    slice_axis(a, 1, start, end)
}

fn drain_bridge(op: &str) {
    pmetal_bridge::check_last_error().unwrap_or_else(|e| panic!("{op}: bridge error: {e}"));
}

fn assert_all_pass(title: &str, reports: &[ParityReport]) {
    println!("\n== {title} ==");
    print_report_table(reports);
    let failed: Vec<&str> = reports
        .iter()
        .filter(|r| !r.passed())
        .map(|r| r.name.as_str())
        .collect();
    assert!(failed.is_empty(), "{title}: {failed:?} out of tolerance");
}

fn prefill(profile: &str) {
    let fx = checkpoint(profile);
    let mut model = DynamicModel::load(fx.path()).expect("checkpoint loads");
    let ids = fx.input_ids();

    let mut reports = Vec::new();
    {
        let DynamicModel::Granite(granite) = &mut model else {
            panic!("{profile} routes to Granite");
        };
        // The layer loop by hand, to compare every layer's output.
        let inner = &mut granite.model;
        let mut h = Module::forward(&mut inner.embed_tokens, &ids)
            .expect("embed")
            .mul_scalar(inner.config.embedding_multiplier);
        for (idx, layer) in inner.layers.iter_mut().enumerate() {
            h = layer
                .forward_with_cache(&h, None, None, None)
                .expect("layer forward");
            let name = format!("layer_{idx}_hidden");
            reports.push(ParityReport::compute(&name, &h, &fx.tensor(&name), TOL));
        }
        let hidden = Module::forward(&mut inner.norm, &h).expect("final norm");
        reports.push(ParityReport::compute(
            "final_hidden",
            &hidden,
            &fx.tensor("final_hidden"),
            TOL,
        ));
    }
    let logits = model.forward(&ids, None).expect("forward");
    drain_bridge("prefill");
    reports.push(ParityReport::compute_with_per_position(
        "logits",
        &logits,
        &fx.tensor("logits"),
        TOL,
    ));
    assert_all_pass(&format!("{profile}: prefill"), &reports);
    assert_eq!(
        argmax_last_axis(&logits),
        argmax_last_axis(&fx.tensor("logits")),
        "{profile}: greedy tokens differ from the reference"
    );
}

fn cached_decode(profile: &str, hybrid: bool) {
    let fx = checkpoint(profile);
    let mut model = DynamicModel::load(fx.path()).expect("checkpoint loads");
    let ids = fx.input_ids();
    let t = ids.dim(1);
    let want = fx.tensor("logits");

    let mut kv = model.create_cache(t as usize + 1);
    let mut mamba = model.create_mamba_cache();
    assert_eq!(
        mamba.is_some(),
        hybrid,
        "{profile}: a recurrent cache exactly when the model has Mamba layers"
    );

    let first = model
        .forward_with_hybrid_cache(&rows(&ids, 0, PREFILL), None, Some(&mut kv), mamba.as_mut())
        .expect("cached prefill");
    drain_bridge("cached prefill");
    let mut reports = vec![ParityReport::compute(
        "prefill_logits",
        &first,
        &rows(&want, 0, PREFILL),
        TOL,
    )];
    for pos in PREFILL..t {
        let step = model
            .forward_with_hybrid_cache(
                &rows(&ids, pos, pos + 1),
                None,
                Some(&mut kv),
                mamba.as_mut(),
            )
            .expect("decode step");
        drain_bridge("decode step");
        reports.push(ParityReport::compute(
            &format!("step_{pos}"),
            &step,
            &rows(&want, pos, pos + 1),
            TOL,
        ));
    }
    assert_all_pass(&format!("{profile}: cached decode"), &reports);

    if hybrid {
        // A KV cache alone is refused for a hybrid, rather than decoding the
        // Mamba layers without their state.
        let mut kv = model.create_cache(4);
        assert!(
            model
                .forward_with_cache(&rows(&ids, 0, 1), None, Some(&mut kv))
                .is_err(),
            "{profile}: a cached forward without Mamba state must be refused"
        );
    }
}

#[test]
#[serial]
fn hybrid_moe_prefill() {
    prefill("hybrid_moe");
}

#[test]
#[serial]
fn hybrid_moe_cached_decode() {
    cached_decode("hybrid_moe", true);
}

#[test]
#[serial]
fn hybrid_dense_prefill() {
    prefill("hybrid_dense");
}

#[test]
#[serial]
fn hybrid_dense_cached_decode() {
    cached_decode("hybrid_dense", true);
}

#[test]
#[serial]
fn dense_rope_prefill() {
    prefill("dense_rope");
}

#[test]
#[serial]
fn dense_rope_cached_decode() {
    cached_decode("dense_rope", false);
}

#[test]
#[serial]
fn moe_prefill() {
    prefill("moe");
}

#[test]
#[serial]
fn moe_cached_decode() {
    cached_decode("moe", false);
}

#[test]
#[serial]
fn moe_shared_prefill() {
    prefill("moe_shared");
}

#[test]
#[serial]
fn moe_shared_cached_decode() {
    cached_decode("moe_shared", false);
}

/// A checkpoint missing a tensor the config calls for is refused, not run
/// with that tensor at its random init.
#[test]
#[serial]
fn incomplete_checkpoint_is_refused() {
    let fx = checkpoint("hybrid_moe");
    let mut weights = load_shard(&fx.path().join("model.safetensors"));
    weights.remove("model.layers.0.block_sparse_moe.input_linear.weight");
    let path = fx.path().join("model.safetensors");
    std::fs::remove_file(&path).unwrap();
    let refs: Vec<(&str, &Array)> = weights.iter().map(|(k, v)| (k.as_str(), v)).collect();
    Array::save_safetensors(path.to_str().unwrap(), &refs);
    drain_bridge("rewrite weights");

    let err = DynamicModel::load(fx.path()).expect_err("an expert bank is missing");
    assert!(
        err.to_string().contains("block_sparse_moe.input_linear"),
        "{err}"
    );
}

/// The MLX conversion of a checkpoint (split expert projections, a dense
/// model's `shared_mlp` renamed to `mlp`, the conv as `[C, K, 1]`) loads to the
/// same model.
#[test]
#[serial]
fn mlx_layout_loads_to_the_same_model() {
    for profile in ["hybrid_moe", "hybrid_dense"] {
        let fx = checkpoint(profile);
        let mut weights = load_shard(&fx.path().join("model.safetensors"));
        let keys: Vec<String> = weights.keys().cloned().collect();
        for key in keys {
            if let Some(p) = key.strip_suffix(".block_sparse_moe.input_linear.weight") {
                let fused = weights.remove(&key).unwrap();
                let i = fused.dim(1) / 2;
                weights.insert(
                    format!("{p}.block_sparse_moe.switch_mlp.gate_proj.weight"),
                    slice_axis(&fused, 1, 0, i),
                );
                weights.insert(
                    format!("{p}.block_sparse_moe.switch_mlp.up_proj.weight"),
                    slice_axis(&fused, 1, i, 2 * i),
                );
            } else if let Some(p) = key.strip_suffix(".block_sparse_moe.output_linear.weight") {
                let w = weights.remove(&key).unwrap();
                weights.insert(
                    format!("{p}.block_sparse_moe.switch_mlp.down_proj.weight"),
                    w,
                );
            } else if profile == "hybrid_dense" {
                if let Some(p) = key.strip_suffix(".shared_mlp.input_linear.weight") {
                    let fused = weights.remove(&key).unwrap();
                    let i = fused.dim(0) / 2;
                    weights.insert(
                        format!("{p}.mlp.gate_proj.weight"),
                        slice_axis(&fused, 0, 0, i),
                    );
                    weights.insert(
                        format!("{p}.mlp.up_proj.weight"),
                        slice_axis(&fused, 0, i, 2 * i),
                    );
                } else if let Some(p) = key.strip_suffix(".shared_mlp.output_linear.weight") {
                    let w = weights.remove(&key).unwrap();
                    weights.insert(format!("{p}.mlp.down_proj.weight"), w);
                }
            }
            if key.ends_with(".conv1d.weight") {
                let w = weights.remove(&key).unwrap();
                weights.insert(key.clone(), w.transpose_axes(&[0, 2, 1]));
            }
        }
        let path = fx.path().join("model.safetensors");
        std::fs::remove_file(&path).unwrap();
        let refs: Vec<(&str, &Array)> = weights.iter().map(|(k, v)| (k.as_str(), v)).collect();
        Array::save_safetensors(path.to_str().unwrap(), &refs);
        drain_bridge("rewrite weights");

        let mut model = DynamicModel::load(fx.path()).expect("MLX layout loads");
        let logits = model.forward(&fx.input_ids(), None).expect("forward");
        drain_bridge("mlx layout forward");
        let report = ParityReport::compute("logits", &logits, &fx.tensor("logits"), TOL);
        assert_all_pass(&format!("{profile}: MLX layout"), &[report]);
    }
}
