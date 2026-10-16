//! Numerical parity for Nemotron-H against the authoritative Hugging Face
//! `transformers` oracle (`NemotronHForCausalLM`).
//!
//! Each fixture is a whole checkpoint directory from transformers 5.20 (torch
//! 2.14.1, seed 11): a `config.json` and an fp32 `model.safetensors` under the
//! key names NVIDIA's checkpoints ship (`backbone.*`, an
//! `experts.{j}.{up,down}_proj.weight` per expert, the depthwise conv as
//! `[C, 1, K]`), loaded through `DynamicModel::load`, the production path.
//!
//! * `hybrid_moe`: Mamba-2 (two groups, chunks of 16 against 40 tokens),
//!   attention, MLP, and grouped MoE routing with a nonzero
//!   `e_score_correction_bias`, `routed_scaling_factor` 2.5 and renormalized
//!   weights. One head's step size is under the floor.
//! * `latent_moe`: one group, no renormalization, and the latent projections of
//!   Nemotron 3 Super; the head is tied.
//!
//! Every profile is checked per layer on an uncached prefill, per MoE layer on
//! the router's experts and weights (given the reference's own input, so the
//! routing is isolated from everything before it), on a cached decode, on a
//! prefill cut into pieces that carry the conv and SSM state across calls, and
//! on the last position alone.

mod common;

use std::collections::HashMap;
use std::path::Path;

use common::{fixture_path, load_shard, ref_tensor};
use pmetal_cache::test_utils::{
    ParityReport, Tolerance, argmax_last_axis, print_report_table, to_f32_vec_eval,
};
use pmetal_models::DynamicModel;
use pmetal_models::architectures::nemotron_h::NemotronHMixer;
use pmetal_runtime::compat::{Array, Module, ModuleParametersExt, ops::slice_axis};
use serial_test::serial;

/// Tokens fed as one prefill before the cached decode takes over.
const PREFILL: i32 = 24;

/// fp32 against fp32; the reference's own fp32-vs-fp64 noise on the logits is
/// 1e-6 (recorded in each fixture meta) and the hidden states reach 3.
const TOL: Tolerance = Tolerance::new(5e-6, 0.0);

struct Fixture {
    _dir: tempfile::TempDir,
    reference: HashMap<String, Array>,
    meta: serde_json::Value,
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

    fn sensitivity(&self, what: &str) -> f64 {
        self.meta["sensitivity"][what]
            .as_f64()
            .unwrap_or_else(|| panic!("the fixture's meta records no {what}"))
    }
}

/// Lay the profile out as a checkpoint directory: `config.json` +
/// `model.safetensors`, nothing else.
fn checkpoint(profile: &str) -> Fixture {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::copy(
        fixture_path(&format!("nemotron_h_{profile}_config.json")),
        dir.path().join("config.json"),
    )
    .expect("copy config");
    std::fs::copy(
        fixture_path(&format!("nemotron_h_{profile}_weights.safetensors")),
        dir.path().join("model.safetensors"),
    )
    .expect("copy weights");
    let reference = load_shard(&fixture_path(&format!(
        "nemotron_h_{profile}_reference.safetensors"
    )));
    let meta = serde_json::from_str(
        &std::fs::read_to_string(fixture_path(&format!(
            "nemotron_h_{profile}_reference.safetensors.meta.json"
        )))
        .expect("meta"),
    )
    .expect("meta parses");
    Fixture {
        _dir: dir,
        reference,
        meta,
    }
}

fn rows(a: &Array, start: i32, end: i32) -> Array {
    slice_axis(a, 1, start, end)
}

fn drain_bridge(op: &str) {
    pmetal_runtime::check_unobserved_error().unwrap_or_else(|e| panic!("{op}: bridge error: {e}"));
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

fn max_abs_diff(a: &Array, b: &Array) -> f32 {
    let (a, b) = (to_f32_vec_eval(a), to_f32_vec_eval(b));
    assert_eq!(a.len(), b.len());
    a.iter().zip(&b).fold(0.0, |worst: f32, (x, y)| {
        let d = (x - y).abs();
        assert!(!d.is_nan(), "NaN in a comparison");
        worst.max(d)
    })
}

fn load(profile: &str) -> (Fixture, DynamicModel) {
    let fx = checkpoint(profile);
    let model = DynamicModel::load(fx.path()).expect("checkpoint loads");
    (fx, model)
}

fn prefill(profile: &str) {
    let (fx, mut model) = load(profile);
    let ids = fx.input_ids();

    let mut reports = Vec::new();
    {
        let DynamicModel::NemotronH(nemotron) = &mut model else {
            panic!("{profile} routes to NemotronH");
        };
        let backbone = &mut nemotron.backbone;
        // The layer loop by hand, to compare every layer's output.
        let mut h = Module::forward(&mut backbone.embeddings, &ids).expect("embed");
        for (idx, layer) in backbone.layers.iter_mut().enumerate() {
            h = layer
                .forward_with_cache(&h, None, None, None)
                .expect("layer forward");
            let name = format!("layer_{idx}_hidden");
            reports.push(ParityReport::compute(&name, &h, &fx.tensor(&name), TOL));
        }
        let hidden = Module::forward(&mut backbone.norm_f, &h).expect("final norm");
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

/// The router's experts and weights, per token, from the reference's own
/// input to the layer: the same set of experts with the same weights.
fn routing(profile: &str) {
    let (fx, mut model) = load(profile);
    let DynamicModel::NemotronH(nemotron) = &mut model else {
        panic!("{profile} routes to NemotronH");
    };
    let mut checked = 0;
    for (idx, layer) in nemotron.backbone.layers.iter().enumerate() {
        let NemotronHMixer::Moe(moe) = &layer.mixer else {
            continue;
        };
        let input = fx.tensor(&format!("layer_{idx}_moe_input"));
        let flat = input.reshape(&[-1, input.dim(-1)]);
        let (inds, weights) = moe.gate.route(&flat);
        drain_bridge("route");
        let want_inds = to_f32_vec_eval(&fx.tensor(&format!("layer_{idx}_topk_indices")));
        let want_weights = to_f32_vec_eval(&fx.tensor(&format!("layer_{idx}_topk_weights")));
        let (got_inds, got_weights) = (to_f32_vec_eval(&inds), to_f32_vec_eval(&weights));
        let k = inds.dim(1) as usize;
        assert_eq!(got_inds.len(), want_inds.len(), "layer {idx}: shapes");
        for token in 0..got_inds.len() / k {
            // The order within a token's k is not defined, the pairing is.
            let pairs = |inds: &[f32], weights: &[f32]| {
                let mut p: Vec<(i32, f32)> = (0..k)
                    .map(|j| (inds[token * k + j] as i32, weights[token * k + j]))
                    .collect();
                p.sort_by_key(|(e, _)| *e);
                p
            };
            let (got, want) = (
                pairs(&got_inds, &got_weights),
                pairs(&want_inds, &want_weights),
            );
            for ((ge, gw), (we, ww)) in got.iter().zip(&want) {
                assert_eq!(
                    ge, we,
                    "layer {idx}, token {token}: experts {got:?} vs {want:?}"
                );
                assert!(
                    (gw - ww).abs() <= 1e-5 * (1.0 + ww.abs()),
                    "layer {idx}, token {token}, expert {ge}: weight {gw} vs {ww}"
                );
            }
        }
        checked += 1;
    }
    assert!(checked > 0, "{profile} has no MoE layer");
}

fn cached_decode(profile: &str) {
    let (fx, mut model) = load(profile);
    let ids = fx.input_ids();
    let t = ids.dim(1);
    let want = fx.tensor("logits");

    let mut kv = model.create_cache(t as usize + 1);
    let mut mamba = model.create_mamba_cache();
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

/// A compressed KV cache serves a hybrid whose first layer is not attention:
/// the cache's length and token count cannot be read off layer 0.
#[test]
#[serial]
fn a_quantized_kv_cache_decodes_a_hybrid() {
    use pmetal_cache::kv_cache::CacheMode;

    let (fx, mut model) = load("hybrid_moe");
    let ids = fx.input_ids();
    let t = ids.dim(1);
    let want = fx.tensor("logits");
    let mut kv = model.create_cache_with_mode(
        t as usize + 1,
        CacheMode::Quantized {
            bits: 8,
            group_size: 16,
        },
    );
    let mut mamba = model.create_mamba_cache();
    let first = model
        .forward_with_hybrid_cache(&rows(&ids, 0, PREFILL), None, Some(&mut kv), mamba.as_mut())
        .expect("quantized-cache prefill");
    let mut got = vec![first];
    for pos in PREFILL..t {
        got.push(
            model
                .forward_with_hybrid_cache(
                    &rows(&ids, pos, pos + 1),
                    None,
                    Some(&mut kv),
                    mamba.as_mut(),
                )
                .expect("quantized-cache decode"),
        );
    }
    drain_bridge("quantized cache");
    let refs: Vec<&Array> = got.iter().collect();
    let got = pmetal_runtime::compat::ops::concatenate_axis(&refs, 1);
    // 8-bit keys and values: the logits move a little, the tokens not at all.
    let report = ParityReport::compute("logits", &got, &want, Tolerance::new(5e-2, 0.0));
    assert_all_pass("quantized KV cache", &[report]);
    assert_eq!(
        argmax_last_axis(&got),
        argmax_last_axis(&want),
        "greedy tokens differ from the reference"
    );
    assert!(kv.seq_len() > 0, "the cache reports no tokens");
}

/// A prompt cut into pieces of every awkward size (shorter than the conv
/// kernel, shorter than a scan chunk, longer) carries the conv history and
/// the SSM state across the cuts.
fn chunked_prefill(profile: &str) {
    let (fx, mut model) = load(profile);
    let ids = fx.input_ids();
    let want = fx.tensor("logits");
    let mut kv = model.create_cache(ids.dim(1) as usize + 1);
    let mut mamba = model.create_mamba_cache();

    let mut reports = Vec::new();
    let mut start = 0;
    for len in [2, 1, 7, 17, 13] {
        let end = (start + len).min(ids.dim(1));
        let piece = model
            .forward_with_hybrid_cache(&rows(&ids, start, end), None, Some(&mut kv), mamba.as_mut())
            .expect("prefill piece");
        drain_bridge("prefill piece");
        reports.push(ParityReport::compute(
            &format!("piece_{start}_{end}"),
            &piece,
            &rows(&want, start, end),
            TOL,
        ));
        start = end;
    }
    assert_eq!(start, ids.dim(1), "the pieces cover the prompt");
    assert_all_pass(&format!("{profile}: chunked prefill"), &reports);
}

fn last_position(profile: &str) {
    let (fx, mut model) = load(profile);
    let ids = fx.input_ids();
    let want = rows(&fx.tensor("logits"), ids.dim(1) - 1, ids.dim(1));
    let mut kv = model.create_cache(ids.dim(1) as usize + 1);
    let mut mamba = model.create_mamba_cache();
    let last = model
        .forward_with_hybrid_cache_last(&ids, None, Some(&mut kv), mamba.as_mut())
        .expect("last-position forward");
    drain_bridge("last position");
    assert_eq!(last.shape(), want.shape());
    assert_all_pass(
        &format!("{profile}: last position"),
        &[ParityReport::compute("last_logits", &last, &want, TOL)],
    );
}

#[test]
#[serial]
fn hybrid_moe_prefill() {
    prefill("hybrid_moe");
}

#[test]
#[serial]
fn hybrid_moe_routing() {
    routing("hybrid_moe");
}

#[test]
#[serial]
fn hybrid_moe_cached_decode() {
    cached_decode("hybrid_moe");
}

#[test]
#[serial]
fn hybrid_moe_chunked_prefill() {
    chunked_prefill("hybrid_moe");
}

#[test]
#[serial]
fn hybrid_moe_last_position() {
    last_position("hybrid_moe");
}

#[test]
#[serial]
fn latent_moe_prefill() {
    prefill("latent_moe");
}

#[test]
#[serial]
fn latent_moe_routing() {
    routing("latent_moe");
}

#[test]
#[serial]
fn latent_moe_cached_decode() {
    cached_decode("latent_moe");
}

#[test]
#[serial]
fn latent_moe_chunked_prefill() {
    chunked_prefill("latent_moe");
}

/// The load-balancing bias steers the choice of experts: leaving it at zero
/// moves the logits by far more than the tolerance, so the parity above is
/// not blind to it.
#[test]
#[serial]
fn the_fixture_is_sensitive_to_the_router_bias() {
    let (fx, mut model) = load("hybrid_moe");
    assert!(
        fx.sensitivity("router_bias_max_logit_shift") > 1e3 * f64::from(TOL.atol),
        "the reference does not move with the bias"
    );
    let DynamicModel::NemotronH(nemotron) = &mut model else {
        panic!("NemotronH");
    };
    let mut touched = 0;
    for layer in &mut nemotron.backbone.layers {
        if let NemotronHMixer::Moe(moe) = &mut layer.mixer {
            *moe.gate.e_score_correction_bias = Array::zeros_f32(&[moe.switch_mlp.num_experts()]);
            touched += 1;
        }
    }
    assert!(touched > 0);
    let logits = model.forward(&fx.input_ids(), None).expect("forward");
    let shift = max_abs_diff(&logits, &fx.tensor("logits"));
    assert!(
        shift > 1e3 * TOL.atol,
        "zeroing the bias moved the logits by only {shift:e}"
    );
}

/// The step size is clamped to the config's `time_step_limit`, and
/// `time_step_min` (0.001 in the fixture's config) is only the range `dt_bias`
/// was initialised from: flooring the step size at it, as `transformers` does,
/// moves the logits by far more than the tolerance, and the model is better
/// for not doing it (see the CHANGELOG).
#[test]
#[serial]
fn the_step_size_floor_is_the_limit_and_not_the_init_range() {
    let fx = checkpoint("hybrid_moe");
    assert!(
        fx.sensitivity("time_step_floor_max_logit_shift") > 50.0 * f64::from(TOL.atol),
        "the reference does not move with the floor"
    );
    let path = fx.path().join("config.json");
    let mut config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    assert_eq!(config["time_step_min"], 0.001);
    config["time_step_limit"] = serde_json::json!([0.001, 1.0e30]);
    std::fs::write(&path, config.to_string()).unwrap();

    let mut model = DynamicModel::load(fx.path()).expect("loads");
    let logits = model.forward(&fx.input_ids(), None).expect("forward");
    let shift = max_abs_diff(&logits, &fx.tensor("logits"));
    assert!(
        shift > 50.0 * TOL.atol,
        "a floor of 0.001 moved the logits by only {shift:e}"
    );
}

/// The multi-token-prediction head some releases bundle is not run; its
/// tensors are accepted. Any other tensor the model has no place for is not.
#[test]
#[serial]
fn bundled_mtp_tensors_are_accepted_and_strangers_are_not() {
    let fx = checkpoint("hybrid_moe");
    let path = fx.path().join("model.safetensors");
    let mut weights = load_shard(&path);
    let any = weights["backbone.norm_f.weight"].clone();
    let rewrite = |weights: &HashMap<String, Array>| {
        std::fs::remove_file(&path).unwrap();
        let refs: Vec<(&str, &Array)> = weights.iter().map(|(k, v)| (k.as_str(), v)).collect();
        Array::save_safetensors(path.to_str().unwrap(), &refs).expect("rewrite weights");
    };

    weights.insert("mtp.layers.0.norm.weight".to_string(), any.clone());
    rewrite(&weights);
    DynamicModel::load(fx.path()).expect("mtp tensors are exempt");

    weights.insert("backbone.layers.0.mixer.surprise.weight".to_string(), any);
    rewrite(&weights);
    let err = DynamicModel::load(fx.path()).expect_err("a stranger tensor");
    assert!(err.to_string().contains("surprise"), "{err}");
}

/// A checkpoint missing a tensor the config calls for is refused, not run
/// with that tensor at its random init.
#[test]
#[serial]
fn incomplete_checkpoint_is_refused() {
    let fx = checkpoint("hybrid_moe");
    let path = fx.path().join("model.safetensors");
    let mut weights = load_shard(&path);
    weights.remove("backbone.layers.1.mixer.gate.e_score_correction_bias");
    std::fs::remove_file(&path).unwrap();
    let refs: Vec<(&str, &Array)> = weights.iter().map(|(k, v)| (k.as_str(), v)).collect();
    Array::save_safetensors(path.to_str().unwrap(), &refs).expect("rewrite weights");

    let err = DynamicModel::load(fx.path()).expect_err("the router bias is missing");
    assert!(err.to_string().contains("e_score_correction_bias"), "{err}");
}

/// The other two layouts a checkpoint arrives in load to the same model: the
/// MLX conversion (experts stacked as `switch_mlp.fc1` / `fc2`, the conv as
/// `[C, K, 1]`) and transformers 5's save (two fused expert tensors).
#[test]
#[serial]
fn the_other_expert_layouts_load_to_the_same_model() {
    for layout in ["mlx", "fused"] {
        let fx = checkpoint("hybrid_moe");
        let path = fx.path().join("model.safetensors");
        let released = load_shard(&path);
        let mut weights: HashMap<String, Array> = HashMap::new();
        let mut per_expert: HashMap<(String, String), Vec<(usize, Array)>> = HashMap::new();
        for (key, value) in released {
            if let Some((base, rest)) = key.split_once(".mixer.experts.") {
                let (index, proj) = rest.split_once('.').expect("expert key");
                let proj = proj.trim_end_matches(".weight").to_string();
                per_expert
                    .entry((base.to_string(), proj))
                    .or_default()
                    .push((index.parse().unwrap(), value));
            } else if layout == "mlx" && key.ends_with(".conv1d.weight") {
                weights.insert(key, value.transpose_axes(&[0, 2, 1]));
            } else {
                weights.insert(key, value);
            }
        }
        for ((base, proj), mut experts) in per_expert {
            experts.sort_by_key(|(i, _)| *i);
            let stacked = pmetal_runtime::compat::ops::stack_axis(
                &experts.into_iter().map(|(_, w)| w).collect::<Vec<_>>(),
                0,
            );
            let key = match (layout, proj.as_str()) {
                ("mlx", "up_proj") => format!("{base}.mixer.switch_mlp.fc1.weight"),
                ("mlx", "down_proj") => format!("{base}.mixer.switch_mlp.fc2.weight"),
                (_, proj) => format!("{base}.mixer.experts.{proj}"),
            };
            weights.insert(key, stacked);
        }
        std::fs::remove_file(&path).unwrap();
        let refs: Vec<(&str, &Array)> = weights.iter().map(|(k, v)| (k.as_str(), v)).collect();
        Array::save_safetensors(path.to_str().unwrap(), &refs).expect("rewrite weights");
        drain_bridge("rewrite weights");

        let mut model = DynamicModel::load(fx.path()).expect("the layout loads");
        let logits = model.forward(&fx.input_ids(), None).expect("forward");
        drain_bridge("forward");
        let report = ParityReport::compute("logits", &logits, &fx.tensor("logits"), TOL);
        assert_all_pass(&format!("{layout} layout"), &[report]);
    }
}

/// `flatten_params` names the parameters as the checkpoint does, and the
/// router bias is among them without being trainable.
#[test]
#[serial]
fn the_parameter_tree_is_the_checkpoints() {
    let (fx, model) = load("hybrid_moe");
    let DynamicModel::NemotronH(nemotron) = &model else {
        panic!("NemotronH");
    };
    let params = nemotron.flatten_params();
    for key in [
        "backbone.embeddings.weight",
        "backbone.layers.0.mixer.A_log",
        "backbone.layers.0.mixer.conv1d.bias",
        "backbone.layers.0.mixer.norm.weight",
        "backbone.layers.1.mixer.gate.weight",
        "backbone.layers.1.mixer.switch_mlp.fc1.weight",
        "backbone.layers.1.mixer.shared_experts.up_proj.weight",
        "backbone.layers.2.mixer.q_proj.weight",
        "backbone.layers.4.mixer.up_proj.weight",
        "backbone.norm_f.weight",
        "lm_head.weight",
    ] {
        assert!(params.contains_key(key), "no parameter {key}");
    }
    let trainable = nemotron.flatten_trainable_params();
    assert!(
        !trainable
            .keys()
            .any(|k| k.ends_with("e_score_correction_bias")),
        "the router bias is trained"
    );
    drop(fx);
}
