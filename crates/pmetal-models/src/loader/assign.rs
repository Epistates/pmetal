//! The generic loader: assign a checkpoint's tensors to a model's parameters by
//! name, and hold the result to account.
//!
//! This is the path every architecture whose parameter tree matches its
//! checkpoint takes (Llama, Qwen, Mistral, Cohere, ...), and the one the GGUF
//! and Phi paths assign through. It fails, naming the difference, when a
//! checkpoint tensor reaches no parameter or a parameter is never filled; see
//! [`super::audit`].

use pmetal_runtime::compat::{Array, ModuleParameters, ModuleParametersExt};

use std::collections::{HashMap, HashSet};
use std::path::Path;

use super::LoadError;
use super::audit::{LoadPolicy, ParamSnapshot};
use crate::architectures::utils::LoadReport;

const PARAM_EVAL_BATCH_SIZE: usize = 128;

/// How many checkpoint keys [`detect_namespace_prefix`] probes before deciding.
/// Each probe narrows the candidate set, and one alone is enough whenever the
/// parameter tree has no repeated tails.
const NAMESPACE_PROBE_KEYS: usize = 8;

/// The namespace shift, if any, that reconciles a checkpoint's keys with a
/// model's parameter names.
///
/// Publishers disagree on whether the text tower carries its wrapper's
/// namespace. `Qwen/Qwen3-Embedding-8B` ships `embed_tokens.weight`,
/// `layers.0.…` and `norm.weight` where `Qwen3ForCausalLM` holds all three
/// under `model.`, so an exact-name load matches nothing and the model answers
/// from its random init.
///
/// The shift is detected once for the whole checkpoint rather than per key. A
/// per-key fallback would let a checkpoint carrying both `layers.0.x` and
/// `model.layers.0.x` resolve both onto the same parameter, in whatever order
/// the `HashMap` happened to yield them.
///
/// Returns `None` when the names already line up, when nothing explains them,
/// or when two prefixes explain them equally well — which leaves the mismatch
/// for [`LoadReport`] to report rather than papering over it with a guess.
fn detect_namespace_prefix(
    params: &HashMap<String, &mut Array>,
    loaded: &HashMap<String, Array>,
) -> Option<String> {
    let explains = |prefix: &str| -> usize {
        loaded
            .keys()
            .filter(|key| params.contains_key(&format!("{prefix}{key}")))
            .count()
    };

    let already = explains("");
    if already == loaded.len() {
        return None;
    }

    // A candidate is whatever the parameter tree puts in front of a checkpoint
    // key. A handful of probes surfaces every prefix in play; scoring then runs
    // each candidate against the whole checkpoint, so one stray key among the
    // probes cannot veto a prefix that explains all the rest.
    let mut probes: Vec<&String> = loaded.keys().collect();
    probes.sort_unstable();
    probes.truncate(NAMESPACE_PROBE_KEYS);

    let mut candidates: HashSet<String> = HashSet::new();
    for probe in probes {
        let tail = format!(".{probe}");
        candidates.extend(
            params
                .keys()
                .filter_map(|name| name.strip_suffix(tail.as_str()))
                .map(|prefix| format!("{prefix}.")),
        );
    }

    let mut ranked: Vec<(usize, String)> = candidates
        .into_iter()
        .map(|prefix| (explains(&prefix), prefix))
        .collect();
    ranked.sort_unstable();

    match ranked.pop() {
        // A prefix that explains no more than the bare names is not a shift,
        // and two prefixes explaining equally much is a coin flip that would
        // silently half-load the model.
        Some((best, prefix)) if best > already && !ranked.iter().any(|(n, _)| *n == best) => {
            Some(prefix)
        }
        _ => None,
    }
}

/// Assign every checkpoint tensor whose name matches a parameter path,
/// reporting the ones that matched nothing.
///
/// Matching is by exact name, after [`detect_namespace_prefix`] reconciles a
/// checkpoint that omits the model's namespace. A checkpoint laid out
/// differently in some other way contributes nothing to the model, so the
/// unmatched keys come back in the report for the caller to judge against a
/// [`LoadPolicy`].
pub(crate) fn assign_loaded_weights<M: ModuleParameters + ModuleParametersExt>(
    model: &mut M,
    loaded: HashMap<String, Array>,
) -> LoadReport {
    let mut params = model.flatten_params_mut();
    let prefix = detect_namespace_prefix(&params, &loaded);
    let mut report = LoadReport::default();

    // A key that already names a parameter keeps its meaning whatever the shift:
    // `lm_head.weight` sits outside the namespace a text tower is missing, and
    // moving it under that namespace would send it nowhere. Those go first, and
    // a shifted key may then only take a parameter no absolute key took, so a
    // checkpoint carrying both `layers.0.x` and `model.layers.0.x` cannot
    // resolve both onto one parameter in whatever order the map yields them.
    let (absolute, relative): (Vec<_>, Vec<_>) = loaded
        .into_iter()
        .partition(|(key, _)| params.contains_key(key));
    let mut taken: HashSet<String> = HashSet::new();
    for (key, value) in absolute {
        if let Some(param) = params.get_mut(&key) {
            **param = value;
            report.loaded += 1;
            taken.insert(key);
        }
    }
    for (key, value) in relative {
        let target = prefix.as_ref().map(|prefix| format!("{prefix}{key}"));
        match target {
            Some(name) if !taken.contains(&name) => match params.get_mut(&name) {
                Some(param) => {
                    **param = value;
                    report.loaded += 1;
                    taken.insert(name);
                }
                // The checkpoint's own spelling, not the probed one, since
                // that is what the reader has in front of them.
                None => report.skipped.push(key),
            },
            _ => report.skipped.push(key),
        }
    }
    report
}

pub(crate) fn eval_loaded_parameters<M: ModuleParametersExt>(model: &M) -> Result<(), LoadError> {
    let params = model.flatten_params();
    let arrays: Vec<Array> = params.into_values().collect();
    for chunk in arrays.chunks(PARAM_EVAL_BATCH_SIZE) {
        pmetal_runtime::compat::transforms::eval(chunk.iter())
            .map_err(|e| LoadError::Mlx(e.to_string()))?;
    }
    Ok(())
}

/// A tied model has no `lm_head` parameter, and its checkpoint may still carry
/// the head it was tied from. transformers takes the embedding matrix for the
/// head and ignores the stored copy, so an `lm_head.weight` nothing consumes is
/// expected exactly when the model has no head of its own.
fn tied_head_is_unused<M: ModuleParametersExt>(model: &M, key: &str) -> bool {
    key == "lm_head.weight"
        && !model
            .flatten_params()
            .keys()
            .any(|name| name.starts_with("lm_head."))
}

/// Assign `weights` to `model` by name and fail unless the checkpoint is
/// accounted for: every tensor consumed or exempt under `policy`, every
/// parameter filled or exempt.
pub(crate) fn assign_checked<M: ModuleParameters + ModuleParametersExt>(
    model: &mut M,
    weights: HashMap<String, Array>,
    policy: &LoadPolicy,
    snapshot: &ParamSnapshot,
) -> Result<LoadReport, LoadError> {
    let report = assign_loaded_weights(model, weights);
    let mut unconsumed = policy.unconsumed(&report.skipped);
    unconsumed.retain(|key| !tied_head_is_unused(model, key));
    snapshot.require_complete(model, policy, unconsumed)?;
    Ok(report)
}

/// Load generic weights using safetensors.
pub fn load_generic_weights<
    M: ModuleParameters + ModuleParametersExt + pmetal_runtime::compat::VisitLinears,
>(
    model: &mut M,
    model_dir: impl AsRef<Path>,
) -> Result<(), LoadError> {
    load_generic_weights_with(model, model_dir, &LoadPolicy::strict())
}

/// [`load_generic_weights`] under an explicit [`LoadPolicy`]: the checkpoint
/// tensors that may go unused and the parameters that may stay as constructed,
/// each with its reason.
///
/// ⚠️ The checkpoint is read whole, never a shard at a time. Unpacking a
/// quantized one needs `{module}.weight` together with its `.scales` and
/// `.biases`, which are separate tensors that nothing promises will share a
/// shard, and the namespace shift is detected once for the whole checkpoint
/// rather than per shard. Reading is lazy, so assembling the map costs headers,
/// not bytes.
///
/// The projections of an NVIDIA ModelOpt NVFP4 (or per-tensor FP8) checkpoint
/// stay packed on their `Linear`s, running MLX's quantized matmul, where every
/// other quantized format is unpacked to dense as it is read. An adapter on
/// such a layer is QLoRA on that checkpoint, with no dense copy ever made.
pub fn load_generic_weights_with<
    M: ModuleParameters + ModuleParametersExt + pmetal_runtime::compat::VisitLinears,
>(
    model: &mut M,
    model_dir: impl AsRef<Path>,
    policy: &LoadPolicy,
) -> Result<(), LoadError> {
    let model_dir = model_dir.as_ref();
    let snapshot = ParamSnapshot::of(model);
    let linear_paths = policy.keep_quantized.then(|| {
        let mut paths = Vec::new();
        model.visit_linears_mut("", &mut |path, _| paths.push(path.to_string()));
        paths
    });
    let (loaded, packed) = super::load_weights_keeping_packed(model_dir, linear_paths.as_deref())?;
    let report = assign_checked_with_packed(model, loaded, packed, policy, &snapshot)
        .map_err(|e| e.with_quantization_hint(model_dir))?;
    eval_loaded_parameters(model)?;
    report.log_summary(model_dir);
    Ok(())
}

/// [`assign_checked`], then the packed projections put on their layers before
/// the parameters are held to account: each one's `weight` is replaced, which
/// is what fills it, and its scales are parameters the model gained.
fn assign_checked_with_packed<
    M: ModuleParameters + ModuleParametersExt + pmetal_runtime::compat::VisitLinears,
>(
    model: &mut M,
    weights: HashMap<String, Array>,
    packed: super::PackedLinears,
    policy: &LoadPolicy,
    snapshot: &ParamSnapshot,
) -> Result<LoadReport, LoadError> {
    let report = assign_loaded_weights(model, weights);
    let mut unconsumed = policy.unconsumed(&report.skipped);
    unconsumed.retain(|key| !tied_head_is_unused(model, key));
    if !packed.is_empty() {
        let count = packed.install(model)?;
        tracing::info!("{count} projections stay packed");
    }
    snapshot.require_complete(model, policy, unconsumed)?;
    Ok(report)
}
///
/// This is the in-memory dual of [`load_generic_weights`]: the safetensors path
/// memory-maps shards from disk, whereas the GGUF inference path dequantizes
/// tensors into a `HashMap` and assigns them here. The same rules apply: a
/// tensor no parameter consumes and a parameter no tensor fills are errors,
/// bar what `policy` exempts.
pub fn assign_weights<M: ModuleParameters + ModuleParametersExt>(
    model: &mut M,
    weights: HashMap<String, Array>,
    policy: &LoadPolicy,
) -> Result<(), LoadError> {
    let snapshot = ParamSnapshot::of(model);
    assign_checked(model, weights, policy, &snapshot)?;
    eval_loaded_parameters(model)?;
    Ok(())
}

/// Rename a checkpoint's tensors to the model's parameter paths, check each
/// shape against the parameter it lands on, and assign them under `policy`,
/// which fails unless checkpoint and model account for each other.
///
/// This is [`assign_weights`] for the architectures whose checkpoints spell
/// their parameters differently from the model (a different tree, a fused
/// projection, a transposed convolution). `map` turns a checkpoint key and
/// tensor into the parameter path and tensor to assign. A key it leaves alone
/// keeps its checkpoint name, so it reaches no parameter and is reported as
/// unconsumed unless `policy` exempts it. Two tensors mapped onto one parameter
/// are an error rather than a race.
///
/// Unlike the generic loader this checks shapes: a tensor of the wrong shape
/// for the parameter it names is a [`LoadError::ShapeMismatch`] here rather than
/// a failure at the first forward pass.
pub(crate) fn assign_renamed<M: ModuleParameters + ModuleParametersExt>(
    model: &mut M,
    weights: HashMap<String, Array>,
    map: impl Fn(&str, Array) -> (String, Array),
    policy: &LoadPolicy,
) -> Result<(), LoadError> {
    let mut renamed: HashMap<String, Array> = HashMap::with_capacity(weights.len());
    for (key, value) in weights {
        let (target, value) = map(&key, value);
        if renamed.insert(target.clone(), value).is_some() {
            return Err(LoadError::SafeTensors(format!(
                "two checkpoint tensors map onto the parameter `{target}` (one is `{key}`)"
            )));
        }
    }
    {
        let params = model.flatten_params();
        for (name, value) in &renamed {
            if let Some(param) = params.get(name.as_str())
                && param.shape() != value.shape()
            {
                return Err(LoadError::ShapeMismatch {
                    key: name.clone(),
                    expected: param.shape().to_vec(),
                    actual: value.shape().to_vec(),
                });
            }
        }
    }
    assign_weights(model, renamed, policy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::architectures::{Qwen3Config, Qwen3ForCausalLM};
    use crate::loader::Exemption;
    use tempfile::tempdir;

    fn write_safetensors(path: &Path, weights: &HashMap<String, Array>) {
        let entries: Vec<_> = weights
            .iter()
            .map(|(key, value)| (key.as_str(), value))
            .collect();
        Array::save_safetensors(path.to_string_lossy().as_ref(), &entries).unwrap();
    }

    /// A model small enough to dump and reload in a test.
    fn tiny_qwen3(tie: bool) -> Qwen3ForCausalLM {
        Qwen3ForCausalLM::new(Qwen3Config {
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: Some(1),
            head_dim: 16,
            vocab_size: 100,
            tie_word_embeddings: tie,
            ..Default::default()
        })
        .unwrap()
    }

    /// A checkpoint holding exactly `model`'s own parameters.
    fn dump(model: &Qwen3ForCausalLM, dir: &Path) -> HashMap<String, Array> {
        let weights: HashMap<String, Array> = model
            .flatten_params()
            .into_iter()
            .map(|(name, array)| (name.to_string(), array))
            .collect();
        write_safetensors(&dir.join("model.safetensors"), &weights);
        weights
    }

    #[test]
    fn a_complete_checkpoint_loads() {
        let dir = tempdir().unwrap();
        let weights = dump(&tiny_qwen3(false), dir.path());

        let mut model = tiny_qwen3(false);
        load_generic_weights(&mut model, dir.path()).unwrap();
        assert!(weights.len() > 10);
    }

    /// ⚠️ A checkpoint missing a tensor used to load, and the parameter ran on
    /// the random value it was constructed with.
    #[test]
    fn a_parameter_the_checkpoint_never_fills_is_an_error() {
        let dir = tempdir().unwrap();
        let mut weights = dump(&tiny_qwen3(false), dir.path());
        weights.remove("model.layers.1.mlp.up_proj.weight");
        write_safetensors(&dir.path().join("model.safetensors"), &weights);

        let mut model = tiny_qwen3(false);
        let err = load_generic_weights(&mut model, dir.path()).unwrap_err();
        match &err {
            LoadError::Incomplete {
                unloaded,
                unconsumed,
                ..
            } => {
                assert_eq!(unloaded, &["model.layers.1.mlp.up_proj.weight".to_string()]);
                assert!(unconsumed.is_empty());
            }
            other => panic!("expected Incomplete, got {other}"),
        }
        assert!(err.to_string().contains("up_proj"), "{err}");
    }

    /// ⚠️ A tensor no parameter consumed used to be logged at `info` and
    /// forgotten: a renamed expert, a dropped router bias, a fused projection.
    #[test]
    fn a_tensor_no_parameter_consumes_is_an_error() {
        let dir = tempdir().unwrap();
        let mut weights = dump(&tiny_qwen3(false), dir.path());
        weights.insert(
            "model.layers.0.self_attn.mystery.weight".to_string(),
            Array::from_slice(&[1.0f32], &[1]),
        );
        write_safetensors(&dir.path().join("model.safetensors"), &weights);

        let mut model = tiny_qwen3(false);
        let err = load_generic_weights(&mut model, dir.path()).unwrap_err();
        match err {
            LoadError::Incomplete {
                unconsumed,
                unloaded,
                ..
            } => {
                assert_eq!(
                    unconsumed,
                    ["model.layers.0.self_attn.mystery.weight".to_string()]
                );
                assert!(unloaded.is_empty());
            }
            other => panic!("expected Incomplete, got {other}"),
        }
    }

    #[test]
    fn buffers_and_a_tied_head_may_go_unused() {
        let dir = tempdir().unwrap();
        let mut weights = dump(&tiny_qwen3(true), dir.path());
        weights.insert(
            "model.layers.0.self_attn.rotary_emb.inv_freq".to_string(),
            Array::from_slice(&[1.0f32], &[1]),
        );
        // A tied model's checkpoint often still carries the head it was tied
        // from; transformers ignores it and so does the loader.
        weights.insert(
            "lm_head.weight".to_string(),
            Array::from_slice(&[1.0f32], &[1]),
        );
        write_safetensors(&dir.path().join("model.safetensors"), &weights);

        let mut model = tiny_qwen3(true);
        assert!(model.lm_head.is_none(), "this config should tie the head");
        load_generic_weights(&mut model, dir.path()).unwrap();
    }

    /// The same stray `lm_head.weight` on an *untied* model is the head's real
    /// weights arriving at nothing, which is a different story.
    #[test]
    fn an_untied_models_head_must_be_consumed() {
        let dir = tempdir().unwrap();
        let mut weights = dump(&tiny_qwen3(false), dir.path());
        let head = weights
            .remove("lm_head.weight")
            .expect("untied model has a head");
        weights.insert("head.weight".to_string(), head);
        write_safetensors(&dir.path().join("model.safetensors"), &weights);

        let mut model = tiny_qwen3(false);
        let err = load_generic_weights(&mut model, dir.path()).unwrap_err();
        let LoadError::Incomplete {
            unconsumed,
            unloaded,
            ..
        } = err
        else {
            panic!("expected Incomplete");
        };
        assert_eq!(unconsumed, ["head.weight".to_string()]);
        assert_eq!(unloaded, ["lm_head.weight".to_string()]);
    }

    #[test]
    fn a_policy_can_exempt_a_parameter_with_a_reason() {
        let dir = tempdir().unwrap();
        let mut weights = dump(&tiny_qwen3(false), dir.path());
        weights.remove("model.norm.weight");
        write_safetensors(&dir.path().join("model.safetensors"), &weights);

        let policy = LoadPolicy::strict().allowing_unloaded_params(&[Exemption::new(
            "model.norm.weight",
            "the test supplies its own",
        )]);
        let mut model = tiny_qwen3(false);
        load_generic_weights_with(&mut model, dir.path(), &policy).unwrap();
    }

    /// The assignment itself, not just the verdict: every parameter holds the
    /// checkpoint's values afterwards.
    #[test]
    fn loaded_parameters_hold_the_checkpoints_values() {
        let dir = tempdir().unwrap();
        let weights = dump(&tiny_qwen3(false), dir.path());

        let mut model = tiny_qwen3(false);
        load_generic_weights(&mut model, dir.path()).unwrap();

        let flat = model.flatten_params();
        assert_eq!(flat.len(), weights.len());
        for (name, want) in &weights {
            let got = &flat[name.as_str()];
            assert_eq!(got.shape(), want.shape(), "{name}");
            let diff = got.subtract(want).abs().max(None);
            assert_eq!(diff.item_f32(), 0.0, "{name} differs from the checkpoint");
        }
        pmetal_runtime::check_unobserved_error().unwrap();
    }

    /// A checkpoint key that matches no parameter has to come back in the
    /// report. Silently dropping it is how a model ends up running on random
    /// init while looking like it loaded fine.
    #[test]
    fn unmatched_checkpoint_keys_are_reported() {
        let mut model = Qwen3ForCausalLM::new(Default::default()).unwrap();
        let tensor = || Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
        let loaded = HashMap::from([
            ("model.embed_tokens.weight".to_string(), tensor()),
            ("model.not_a_parameter.weight".to_string(), tensor()),
            ("layers.0.self_attn.q_proj.weight".to_string(), tensor()),
        ]);
        let report = assign_loaded_weights(&mut model, loaded);

        assert_eq!(
            report.loaded, 1,
            "only embed_tokens matches a real parameter"
        );
        let skipped: HashSet<&str> = report.skipped.iter().map(String::as_str).collect();
        assert_eq!(
            skipped,
            HashSet::from([
                "model.not_a_parameter.weight",
                "layers.0.self_attn.q_proj.weight"
            ]),
            "both unmatched keys must be reported, including the un-prefixed layout"
        );
    }

    /// Build a `flatten_params_mut`-shaped map without a real module, so the
    /// namespace probe can be exercised on layouts no architecture ships.
    fn param_map<'a>(names: &[&str], slots: &'a mut [Array]) -> HashMap<String, &'a mut Array> {
        assert_eq!(names.len(), slots.len());
        names
            .iter()
            .map(|name| (*name).to_string())
            .zip(slots.iter_mut())
            .collect()
    }

    fn checkpoint(keys: &[&str]) -> HashMap<String, Array> {
        keys.iter()
            .map(|key| ((*key).to_string(), Array::from_slice(&[0.0f32], &[1])))
            .collect()
    }

    fn slots(n: usize) -> Vec<Array> {
        (0..n).map(|_| Array::from_slice(&[0.0f32], &[1])).collect()
    }

    /// `Qwen/Qwen3-Embedding-8B` publishes the text tower with no namespace at
    /// all, so every key missed and the model answered from random init
    /// (issue #19).
    #[test]
    fn unprefixed_checkpoint_resolves_against_the_model_namespace() {
        let mut slots = slots(4);
        let params = param_map(
            &[
                "model.embed_tokens.weight",
                "model.layers.0.self_attn.q_proj.weight",
                "model.norm.weight",
                "lm_head.weight",
            ],
            &mut slots,
        );
        let loaded = checkpoint(&[
            "embed_tokens.weight",
            "layers.0.self_attn.q_proj.weight",
            "norm.weight",
        ]);

        assert_eq!(
            detect_namespace_prefix(&params, &loaded).as_deref(),
            Some("model.")
        );
    }

    /// The checkpoint above plus an `lm_head.weight` that *does* line up. An
    /// early-out on "some key matched" would take the identity mapping here and
    /// drop every layer.
    #[test]
    fn one_already_matching_key_does_not_veto_the_shift() {
        let mut slots = slots(4);
        let params = param_map(
            &[
                "model.embed_tokens.weight",
                "model.layers.0.self_attn.q_proj.weight",
                "model.norm.weight",
                "lm_head.weight",
            ],
            &mut slots,
        );
        let loaded = checkpoint(&[
            "embed_tokens.weight",
            "layers.0.self_attn.q_proj.weight",
            "norm.weight",
            "lm_head.weight",
        ]);

        assert_eq!(
            detect_namespace_prefix(&params, &loaded).as_deref(),
            Some("model."),
            "three keys under `model.` outweigh the one that already matched"
        );
    }

    #[test]
    fn a_checkpoint_that_already_lines_up_is_left_alone() {
        let mut slots = slots(2);
        let params = param_map(
            &["model.embed_tokens.weight", "model.norm.weight"],
            &mut slots,
        );
        let loaded = checkpoint(&["model.embed_tokens.weight", "model.norm.weight"]);

        assert_eq!(detect_namespace_prefix(&params, &loaded), None);
    }

    /// Two towers with identically-named sublayers: prefixing onto either is a
    /// coin flip, so the load has to stay put and report the misses.
    #[test]
    fn an_ambiguous_shift_is_refused() {
        let mut slots = slots(2);
        let params = param_map(
            &[
                "text_model.layers.0.self_attn.q_proj.weight",
                "vision_model.layers.0.self_attn.q_proj.weight",
            ],
            &mut slots,
        );
        let loaded = checkpoint(&["layers.0.self_attn.q_proj.weight"]);

        assert_eq!(detect_namespace_prefix(&params, &loaded), None);
    }

    #[test]
    fn a_layout_no_prefix_explains_is_left_to_the_report() {
        let mut slots = slots(1);
        let params = param_map(&["model.embed_tokens.weight"], &mut slots);
        let loaded = checkpoint(&["bert.encoder.layer.0.attention.self.query.weight"]);

        assert_eq!(detect_namespace_prefix(&params, &loaded), None);
    }

    /// End to end on the real `Qwen3ForCausalLM` parameter tree: before the
    /// shift was detected this loaded nothing at all.
    #[test]
    fn unprefixed_qwen3_checkpoint_loads_into_the_model() {
        let dir = tempdir().unwrap();
        let mut weights = dump(&tiny_qwen3(false), dir.path());
        weights = weights
            .into_iter()
            .map(|(key, value)| {
                (
                    key.strip_prefix("model.").unwrap_or(&key).to_string(),
                    value,
                )
            })
            .collect();
        write_safetensors(&dir.path().join("model.safetensors"), &weights);

        let mut model = tiny_qwen3(false);
        load_generic_weights(&mut model, dir.path()).unwrap();
    }

    /// A checkpoint carrying a tensor under both spellings must not let the
    /// shifted one overwrite the absolute one, which `HashMap` order used to
    /// decide.
    #[test]
    fn a_shifted_key_never_overwrites_an_absolute_one() {
        let dir = tempdir().unwrap();
        let weights = dump(&tiny_qwen3(false), dir.path());
        let key = "model.layers.0.input_layernorm.weight";
        let truth = weights[key].clone();

        let mut loaded = weights.clone();
        // The same tensor spelled without the namespace, holding other values.
        loaded.insert(
            "layers.0.input_layernorm.weight".to_string(),
            truth.multiply(&Array::from_f32(0.0)),
        );
        // Enough shifted keys that the shift is detected at all.
        for name in ["layers.0.mlp.up_proj.weight", "layers.1.mlp.up_proj.weight"] {
            loaded.insert(name.to_string(), weights[&format!("model.{name}")].clone());
        }
        let mut model = tiny_qwen3(false);
        let report = assign_loaded_weights(&mut model, loaded);

        assert!(
            report
                .skipped
                .contains(&"layers.0.input_layernorm.weight".to_string()),
            "the duplicate should be reported, not applied: {:?}",
            report.skipped
        );
        let got = model.flatten_params()[key].clone();
        let diff = got.subtract(&truth).abs().max(None);
        assert_eq!(
            diff.item_f32(),
            0.0,
            "the absolute key lost to the shifted one"
        );
    }

    /// ⚠️ The `simple_load!` arms — llama, qwen2, mistral, gemma, phi, cohere
    /// and most of the rest — reach the model through `load_generic_weights`,
    /// which read shards straight off disk and never consulted the
    /// quantization config. Every `mlx-community` 4-bit checkpoint therefore
    /// arrived still packed, and the model built on `uint32` payloads.
    ///
    /// Asserted on the *parameter*, not on a returned map, because the gap was
    /// between reading the shard and assigning it.
    #[test]
    fn load_generic_weights_keeps_quantized_projections_packed_when_asked() {
        let dir = tempdir().unwrap();
        let mut weights = dump(&tiny_qwen3(false), dir.path());
        // An MLX conversion: the projection's words, scales and biases, and
        // the embedding quantized too (which is not a `Linear`).
        let q = "model.layers.0.self_attn.q_proj";
        let embed = "model.embed_tokens";
        for module in [q, embed] {
            let dense = weights.remove(&format!("{module}.weight")).unwrap();
            let (packed, scales, biases) = dense.quantize_weights(32, 4);
            weights.insert(format!("{module}.weight"), packed);
            weights.insert(format!("{module}.scales"), scales);
            weights.insert(format!("{module}.biases"), biases);
        }
        write_safetensors(&dir.path().join("model.safetensors"), &weights);
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"quantization": {"bits": 4, "group_size": 32}}"#,
        )
        .unwrap();

        let mut unpacked = tiny_qwen3(false);
        load_generic_weights(&mut unpacked, dir.path()).unwrap();
        let mut kept = tiny_qwen3(false);
        load_generic_weights_with(
            &mut kept,
            dir.path(),
            &LoadPolicy::strict().keeping_quantized(true),
        )
        .unwrap();

        // The projection runs on its packed words; the embedding is dense.
        let layer = &kept.model.layers[0].self_attn.q_proj;
        let quant = layer.quant.as_ref().expect("q_proj stays quantized");
        assert_eq!((quant.params.bits, quant.params.group_size), (4, 32));
        assert!(quant.biases.is_some(), "affine biases ride along");
        assert_ne!(
            layer.weight.shape(),
            unpacked.model.layers[0].self_attn.q_proj.weight.shape()
        );
        assert_eq!(
            kept.model.embed_tokens.weight.shape(),
            unpacked.model.embed_tokens.weight.shape(),
            "an embedding is unpacked to dense"
        );
        // And it computes what the unpacked layer computes.
        let x = pmetal_runtime::compat::random::normal(
            &[1, 3, unpacked.model.layers[0].self_attn.q_proj.shape().1],
            pmetal_runtime::compat::Dtype::Float32,
        );
        let a = layer.forward(&x);
        let b = unpacked.model.layers[0].self_attn.q_proj.forward(&x);
        let diff = a
            .subtract(&b.as_dtype(a.dtype().as_i32()))
            .abs()
            .max(None)
            .item_f32();
        assert!(
            diff < 2e-2,
            "packed and unpacked projections differ by {diff}"
        );
        pmetal_runtime::check_unobserved_error().unwrap();
    }

    #[test]
    fn load_generic_weights_unpacks_a_quantized_checkpoint() {
        let dir = tempdir().unwrap();
        let mut weights = dump(&tiny_qwen3(false), dir.path());

        // Quantize one projection the way MLX does and ship it as packed
        // words with sibling scales and biases.
        let key = "model.layers.0.self_attn.q_proj";
        let dense = weights.remove(&format!("{key}.weight")).unwrap();
        let shape = dense.shape().to_vec();
        let (packed, scales, biases) = dense.quantize_weights(32, 4);
        weights.insert(format!("{key}.weight"), packed);
        weights.insert(format!("{key}.scales"), scales);
        weights.insert(format!("{key}.biases"), biases);
        write_safetensors(&dir.path().join("model.safetensors"), &weights);
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"quantization": {"bits": 4, "group_size": 32}}"#,
        )
        .unwrap();

        let mut model = tiny_qwen3(false);
        load_generic_weights(&mut model, dir.path()).unwrap();

        assert_eq!(
            model.model.layers[0].self_attn.q_proj.weight.shape(),
            shape.as_slice(),
            "q_proj kept the packed uint32 width instead of being dequantized"
        );
    }
}
