//! Routed experts, brought to the stacked `switch_mlp` layout at load.
//!
//! A model holds a layer's experts as three weights stacked over a leading
//! expert axis, `mlp.switch_mlp.{gate,up,down}_proj.weight`, `[experts, out,
//! in]` (what an MLX conversion stores; see
//! [`SwitchMlp`](crate::common::switch_mlp::SwitchMlp)). Checkpoints arrive in
//! three layouts:
//!
//! * one key per expert, the released layout:
//!   `mlp.experts.{e}.{gate,up,down}_proj.weight`;
//! * transformers 5's save, experts fused into two tensors:
//!   `mlp.experts.gate_up_proj` (`[experts, 2 * intermediate, hidden]`, the
//!   gate projection in the first half of the second axis, the up projection in
//!   the second) and `mlp.experts.down_proj` (`[experts, hidden,
//!   intermediate]`);
//! * already stacked, which passes through untouched.

use std::collections::{BTreeMap, HashMap};

use pmetal_runtime::compat::{Array, Exception, ops};

/// The projections an expert has, in the order a layer lists them.
const PROJECTIONS: [&str; 3] = ["gate_proj", "up_proj", "down_proj"];

/// A checkpoint's tensors with each layer's routed experts stacked.
///
/// Keys that are not an expert's pass through. The stack of a per-expert layout
/// is one lazy concatenation: evaluated, each expert is read once and written
/// into its slot of the result, so a layer is never held twice. `family` names
/// the architecture in errors: an expert missing from the run `0..n`, a layer
/// that carries two layouts, or a fused tensor of the wrong shape.
pub fn stack_experts(
    family: &str,
    weights: HashMap<String, Array>,
) -> Result<HashMap<String, Array>, Exception> {
    // (layer prefix `…mlp`, projection) -> expert index -> weight
    let mut per_expert: HashMap<(String, String), BTreeMap<usize, Array>> = HashMap::new();
    let mut out = HashMap::with_capacity(weights.len());
    let put = |out: &mut HashMap<String, Array>, key: String, value: Array| match out
        .insert(key.clone(), value)
    {
        None => Ok(()),
        Some(_) => Err(Exception::custom(format!(
            "{family}: the checkpoint holds {key} twice, in two expert layouts"
        ))),
    };

    for (key, value) in weights {
        if let Some(layer) = key.strip_suffix(".mlp.experts.gate_up_proj") {
            if value.ndim() != 3 || value.dim(1) % 2 != 0 {
                return Err(Exception::custom(format!(
                    "{family}: {key} is {:?}, not a fused [experts, 2 * intermediate, hidden]",
                    value.shape()
                )));
            }
            let half = value.dim(1) / 2;
            let layer = format!("{layer}.mlp.switch_mlp");
            put(
                &mut out,
                format!("{layer}.gate_proj.weight"),
                ops::slice_axis(&value, 1, 0, half),
            )?;
            put(
                &mut out,
                format!("{layer}.up_proj.weight"),
                ops::slice_axis(&value, 1, half, 2 * half),
            )?;
            continue;
        }
        if let Some(layer) = key.strip_suffix(".mlp.experts.down_proj") {
            put(
                &mut out,
                format!("{layer}.mlp.switch_mlp.down_proj.weight"),
                value,
            )?;
            continue;
        }
        let parsed = key.split_once(".mlp.experts.").and_then(|(layer, rest)| {
            let (index, member) = rest.split_once('.')?;
            let proj = member.strip_suffix(".weight")?;
            PROJECTIONS.contains(&proj).then(|| {
                Some((
                    format!("{layer}.mlp"),
                    proj.to_string(),
                    index.parse::<usize>().ok()?,
                ))
            })?
        });
        match parsed {
            Some((layer, proj, index)) => {
                per_expert
                    .entry((layer, proj))
                    .or_default()
                    .insert(index, value);
            }
            None => put(&mut out, key, value)?,
        }
    }

    stack_runs(family, per_expert, &mut out, &put, |layer, proj| {
        format!("{layer}.switch_mlp.{proj}.weight")
    })?;
    pmetal_runtime::check_unobserved_error()
        .map_err(|e| Exception::custom(format!("{family}: stacking the experts failed: {e}")))?;
    Ok(out)
}

/// Stack each `(layer, projection)` run of per-expert tensors into one tensor
/// named `dest(layer, projection)`, naming a gap in a run.
fn stack_runs(
    family: &str,
    per_expert: HashMap<(String, String), BTreeMap<usize, Array>>,
    out: &mut HashMap<String, Array>,
    put: &impl Fn(&mut HashMap<String, Array>, String, Array) -> Result<(), Exception>,
    dest: impl Fn(&str, &str) -> String,
) -> Result<(), Exception> {
    for ((layer, proj), by_index) in per_expert {
        let count = by_index.len();
        if let Some(missing) = (0..count).find(|e| !by_index.contains_key(e)) {
            return Err(Exception::custom(format!(
                "{family}: {layer}.experts.{missing}.{proj}.weight is missing from a run of \
                 {count} experts"
            )));
        }
        let ordered: Vec<Array> = by_index.into_values().collect();
        put(out, dest(&layer, &proj), ops::stack_axis(&ordered, 0))?;
    }
    Ok(())
}

/// The destination of a plain (gate-less) expert projection, in an MLX
/// conversion's names: the up projection is `fc1`, the down projection `fc2`.
fn plain_projection(proj: &str) -> Option<&'static str> {
    match proj {
        "up_proj" => Some("fc1"),
        "down_proj" => Some("fc2"),
        _ => None,
    }
}

/// The experts of a layer whose experts have no gate (Nemotron-H), stacked
/// into the `mixer.switch_mlp.{fc1,fc2}` layout the model holds
/// ([`PlainSwitchMlp`](crate::common::switch_mlp::PlainSwitchMlp)).
///
/// The same layouts as [`stack_experts`]: one key per expert,
/// `mixer.experts.{e}.{up,down}_proj.weight`; transformers 5's two fused
/// tensors, `mixer.experts.up_proj` `[experts, intermediate, hidden]` and
/// `mixer.experts.down_proj` `[experts, hidden, intermediate]`; and the
/// already-stacked `switch_mlp.fc1` / `fc2`, which pass through. The up
/// projection becomes `fc1` and the down projection `fc2`.
pub fn stack_plain_experts(
    family: &str,
    weights: HashMap<String, Array>,
) -> Result<HashMap<String, Array>, Exception> {
    let mut per_expert: HashMap<(String, String), BTreeMap<usize, Array>> = HashMap::new();
    let mut out = HashMap::with_capacity(weights.len());
    let put = |out: &mut HashMap<String, Array>, key: String, value: Array| match out
        .insert(key.clone(), value)
    {
        None => Ok(()),
        Some(_) => Err(Exception::custom(format!(
            "{family}: the checkpoint holds {key} twice, in two expert layouts"
        ))),
    };

    for (key, value) in weights {
        // Fused: `<layer>.mixer.experts.up_proj`, with no `.weight`.
        let fused = key
            .rsplit_once(".mixer.experts.")
            .and_then(|(layer, proj)| {
                plain_projection(proj).map(|fc| format!("{layer}.mixer.switch_mlp.{fc}.weight"))
            });
        if let Some(dest) = fused {
            if value.ndim() != 3 {
                return Err(Exception::custom(format!(
                    "{family}: {key} is {:?}, not a fused [experts, out, in]",
                    value.shape()
                )));
            }
            put(&mut out, dest, value)?;
            continue;
        }
        let parsed = key.split_once(".mixer.experts.").and_then(|(layer, rest)| {
            let (index, member) = rest.split_once('.')?;
            let proj = member.strip_suffix(".weight")?;
            plain_projection(proj)?;
            Some((
                format!("{layer}.mixer"),
                proj.to_string(),
                index.parse::<usize>().ok()?,
            ))
        });
        match parsed {
            Some((layer, proj, index)) => {
                per_expert
                    .entry((layer, proj))
                    .or_default()
                    .insert(index, value);
            }
            None => put(&mut out, key, value)?,
        }
    }
    stack_runs(family, per_expert, &mut out, &put, |layer, proj| {
        let fc = plain_projection(proj).expect("only plain projections are collected");
        format!("{layer}.switch_mlp.{fc}.weight")
    })?;
    pmetal_runtime::check_unobserved_error()
        .map_err(|e| Exception::custom(format!("{family}: stacking the experts failed: {e}")))?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_runtime::compat::Dtype;
    use serial_test::serial;

    fn max_diff(a: &Array, b: &Array) -> f32 {
        let diff = a.subtract(b).abs().max(None);
        diff.eval();
        diff.item::<f32>()
    }

    /// A released checkpoint names an expert per key; the model holds each
    /// layer's experts stacked, in an MLX conversion's `switch_mlp` layout.
    #[test]
    #[serial]
    fn stacks_per_expert_keys() {
        let (experts, out, inp) = (3, 4, 5);
        let layer = "model.layers.2.mlp";
        let per_expert: Vec<Array> = (0..experts)
            .map(|e| Array::from_slice(&vec![e as f32 + 1.0; (out * inp) as usize], &[out, inp]))
            .collect();
        let mut weights: HashMap<String, Array> = HashMap::new();
        for (e, w) in per_expert.iter().enumerate() {
            for proj in PROJECTIONS {
                weights.insert(format!("{layer}.experts.{e}.{proj}.weight"), w.clone());
            }
        }
        weights.insert(format!("{layer}.gate.weight"), per_expert[0].clone());
        weights.insert(
            format!("{layer}.shared_experts.gate_proj.weight"),
            per_expert[1].clone(),
        );

        let stacked = stack_experts("test", weights).unwrap();
        assert_eq!(
            stacked.len(),
            3 + 2,
            "three stacks, the gate and the shared expert"
        );
        for proj in PROJECTIONS {
            let w = &stacked[&format!("{layer}.switch_mlp.{proj}.weight")];
            assert_eq!(w.shape(), &[experts, out, inp]);
            for e in 0..experts {
                let slice = ops::slice_axis(w, 0, e, e + 1);
                let want = Array::full(&[1, out, inp], e as f32 + 1.0, Dtype::Float32);
                assert!(
                    max_diff(&slice, &want) == 0.0,
                    "{proj}: expert {e} is not in slot {e}"
                );
            }
        }
        assert!(stacked.contains_key(&format!("{layer}.gate.weight")));
        assert!(stacked.contains_key(&format!("{layer}.shared_experts.gate_proj.weight")));
    }

    #[test]
    #[serial]
    fn names_a_missing_expert() {
        let w = Array::zeros_f32(&[2, 2]);
        let mut weights: HashMap<String, Array> = HashMap::new();
        for e in [0, 1, 3] {
            weights.insert(
                format!("model.layers.1.mlp.experts.{e}.up_proj.weight"),
                w.clone(),
            );
        }
        let err = stack_experts("test", weights).expect_err("expert 2 is missing");
        assert!(err.to_string().contains("experts.2.up_proj"), "{err}");
    }

    /// The fused layout splits `gate_up_proj` into a gate (first half) and an
    /// up (second half) projection and renames `down_proj`.
    #[test]
    #[serial]
    fn splits_the_fused_layout_gate_first() {
        let (experts, inter, hidden) = (2, 3, 4);
        let layer = "model.layers.0.mlp";
        let gate_up = ops::concatenate_axis(
            &[
                &Array::full(&[experts, inter, hidden], 1.0, Dtype::Float32),
                &Array::full(&[experts, inter, hidden], 2.0, Dtype::Float32),
            ],
            1,
        );
        let down = Array::full(&[experts, hidden, inter], 3.0, Dtype::Float32);
        let weights = HashMap::from([
            (format!("{layer}.experts.gate_up_proj"), gate_up),
            (format!("{layer}.experts.down_proj"), down),
            (format!("{layer}.gate.weight"), Array::zeros_f32(&[2, 4])),
        ]);
        let stacked = stack_experts("test", weights).unwrap();
        assert_eq!(stacked.len(), 4);
        for (proj, value, shape) in [
            ("gate_proj", 1.0, [experts, inter, hidden]),
            ("up_proj", 2.0, [experts, inter, hidden]),
            ("down_proj", 3.0, [experts, hidden, inter]),
        ] {
            let w = &stacked[&format!("{layer}.switch_mlp.{proj}.weight")];
            assert_eq!(w.shape(), &shape);
            assert_eq!(
                max_diff(w, &Array::full(&shape, value, Dtype::Float32)),
                0.0,
                "{proj}"
            );
        }
    }

    #[test]
    #[serial]
    fn refuses_two_layouts_of_one_layer() {
        let w = Array::zeros_f32(&[2, 4, 4]);
        let weights = HashMap::from([
            (
                "model.layers.0.mlp.experts.down_proj".to_string(),
                w.clone(),
            ),
            (
                "model.layers.0.mlp.experts.0.down_proj.weight".to_string(),
                Array::zeros_f32(&[4, 4]),
            ),
        ]);
        let err = stack_experts("test", weights).expect_err("two layouts");
        assert!(err.to_string().contains("twice"), "{err}");
    }
}
