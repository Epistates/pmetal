//! MoE expert permutation alignment.
//!
//! Background — "the MoE caveat": full-model merging matches tensors by
//! name. A routed expert is named `…experts.{i}.…`, so expert *i* in
//! checkpoint A is always merged with expert *i* in checkpoint B. If the
//! two checkpoints specialised their experts in different orders during
//! training, the index correspondence is meaningless and the merged
//! expert bank is incoherent.
//!
//! This module solves that by detecting MoE structure, computing the
//! optimal pairwise expert correspondence between models, and exposing a
//! tensor-name remapping helper. The remapping rewrites
//! `…experts.{i}.…` → `…experts.{π(i)}.…` for non-base models so the
//! downstream merge sees aligned experts.
//!
//! # Algorithm
//!
//! For each layer with experts, for each non-base model:
//!
//!   1. Build the expert "fingerprint" — the flattened, L2-normalized
//!      `gate_proj.weight` (or first available expert sublayer). Same
//!      fingerprint shape across models (a flat vector of length
//!      `in_features × out_features`).
//!   2. Compute the `N × N` pairwise cosine similarity between the base
//!      model's experts and the candidate model's experts.
//!   3. Solve a maximum-weight assignment via the Hungarian algorithm.
//!      `N` is small (≤ 128 in any modern MoE), so the textbook O(N³)
//!      implementation here is sufficient.
//!
//! The result is a permutation `π: 0..N → 0..N` for each (model, layer)
//! pair. The base model uses the identity permutation. Tensor names are
//! rewritten via [`MoeAlignment::remap_tensor_name`].
//!
//! Clean-room implementation. Hungarian solver follows Kuhn (1955) /
//! Munkres (1957) — the algorithm is over a century old; nothing
//! proprietary about it.

use std::collections::{BTreeMap, HashMap};
use std::sync::OnceLock;

use crate::{Result, TensorLoader};
use regex::Regex;

/// The unit-length flattened fingerprint of an expert tensor. An unreadable
/// tensor is an error: an empty fingerprint would align experts by noise.
fn fingerprint(mut tensor: pmetal_runtime::compat::Array, name: &str) -> Result<Vec<f32>> {
    let n: usize = tensor.shape().iter().map(|&s| s as usize).product();
    let values = tensor.to_f32_vec(n);
    pmetal_runtime::check_unobserved_error()?;
    let mut values = values.ok_or_else(|| {
        crate::MergeError::ModelLoad(format!("expert tensor {name}: failed to read {n} values"))
    })?;
    l2_normalize(&mut values);
    Ok(values)
}

/// Compiled regex matching `…experts.{idx}.…` segments.
fn experts_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\.experts\.(\d+)\.").expect("experts regex compiles"))
}

/// Compiled regex matching `…layers.{idx}.…` segments — used to group
/// expert tensors by layer.
fn layers_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\.layers\.(\d+)\.").expect("layers regex compiles"))
}

/// Per-tensor parse — returns `(layer_idx, expert_idx)` if the name is an
/// expert tensor inside a `layers.{N}` block.
fn parse_layer_expert(name: &str) -> Option<(usize, usize)> {
    let layer = layers_re().captures(name)?.get(1)?.as_str().parse().ok()?;
    let expert = experts_re().captures(name)?.get(1)?.as_str().parse().ok()?;
    Some((layer, expert))
}

/// Find the canonical "expert fingerprint" tensor name for a given
/// `(layer, expert)`. Tries `gate_proj.weight` first (DeepSeek / Qwen3MoE
/// / Llama 4 / Mistral 8x7B convention), then `w1.weight` (GPT-OSS),
/// then `up_proj.weight`. Returns `None` if no candidate matches in the
/// given loader.
fn find_fingerprint_name<L: TensorLoader + ?Sized>(
    loader: &L,
    layer: usize,
    expert: usize,
    available: &[String],
) -> Option<String> {
    let candidates = [
        format!(".layers.{}.", layer),
        // also handle Llama4-style `.feed_forward.experts.{}.`
    ];
    let suffixes = [
        format!(".experts.{}.gate_proj.weight", expert),
        format!(".experts.{}.w1.weight", expert),
        format!(".experts.{}.up_proj.weight", expert),
        format!(".experts.{}.input_linear.weight", expert),
    ];
    for prefix in &candidates {
        for suffix in &suffixes {
            for n in available {
                if n.contains(prefix) && n.ends_with(suffix.as_str()) {
                    // double-check it's available
                    let _ = loader.tensor_names();
                    return Some(n.clone());
                }
            }
        }
    }
    None
}

/// One model's permutation table: `permutations[layer] = π` such that
/// expert `i` in this model corresponds to expert `π(i)` in the base.
#[derive(Debug, Clone, Default)]
pub struct ModelPermutation {
    /// Layer-indexed table. Missing layers are left unmapped (treated as
    /// identity). Stored as a `BTreeMap` so iteration is deterministic.
    pub per_layer: BTreeMap<usize, Vec<usize>>,
}

impl ModelPermutation {
    /// Lookup the permutation for a specific layer; identity is returned
    /// when no entry exists for that layer.
    pub fn permutation_for(&self, layer: usize) -> Option<&[usize]> {
        self.per_layer.get(&layer).map(|v| v.as_slice())
    }
}

/// Per-merge alignment table. Index 0 is the *base* model (identity
/// permutation by construction).
#[derive(Debug, Clone, Default)]
pub struct MoeAlignment {
    /// One [`ModelPermutation`] per input model. Index 0 corresponds to
    /// the model treated as the alignment reference (the base of the
    /// merge); its entry is always the identity.
    pub permutations: Vec<ModelPermutation>,
}

impl MoeAlignment {
    /// Rewrite an expert-bearing tensor name from model `model_idx`'s
    /// frame into the base model's frame. Returns the original name when
    /// no permutation applies (non-MoE tensor, or unknown layer).
    pub fn remap_tensor_name(&self, model_idx: usize, name: &str) -> String {
        let perm = match self.permutations.get(model_idx) {
            Some(p) => p,
            None => return name.to_string(),
        };
        let (layer, expert) = match parse_layer_expert(name) {
            Some(t) => t,
            None => return name.to_string(),
        };
        let table = match perm.permutation_for(layer) {
            Some(t) => t,
            None => return name.to_string(),
        };
        let new_expert = match table.get(expert) {
            Some(&i) => i,
            None => return name.to_string(),
        };
        if new_expert == expert {
            return name.to_string();
        }
        // Replace the *first* occurrence of `experts.{expert}.` with the
        // remapped index. The regex pattern is unambiguous so a string
        // replace is safe.
        let needle = format!(".experts.{}.", expert);
        let replacement = format!(".experts.{}.", new_expert);
        name.replacen(&needle, &replacement, 1)
    }
}

/// Compute the alignment table given access to every loader. The first
/// loader is the base model. Loaders that don't share an expert structure
/// with the base get the identity permutation for every layer.
///
/// This pass loads only the fingerprint tensors (one per expert per
/// layer per model), so memory pressure stays modest.
pub fn align_experts(loaders: &[&dyn TensorLoader]) -> Result<MoeAlignment> {
    let mut alignment = MoeAlignment {
        permutations: vec![ModelPermutation::default(); loaders.len()],
    };
    if loaders.len() < 2 {
        return Ok(alignment);
    }

    // Discover (layer, expert) pairs from the base model's tensor names.
    let base_names: Vec<String> = loaders[0].tensor_names();
    let mut by_layer: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for n in &base_names {
        if let Some((layer, expert)) = parse_layer_expert(n) {
            let entry = by_layer.entry(layer).or_default();
            if !entry.contains(&expert) {
                entry.push(expert);
            }
        }
    }
    if by_layer.is_empty() {
        return Ok(alignment);
    }
    for v in by_layer.values_mut() {
        v.sort();
    }

    // For each layer, compute base fingerprints once.
    let mut base_fingerprints: HashMap<usize, Vec<Vec<f32>>> = HashMap::new();
    for (&layer, experts) in &by_layer {
        let mut fps = Vec::with_capacity(experts.len());
        for &e in experts {
            let fp_name = find_fingerprint_name(loaders[0], layer, e, &base_names);
            let fp_name = match fp_name {
                Some(n) => n,
                None => return Ok(alignment), // give up cleanly — treat all as identity
            };
            fps.push(fingerprint(loaders[0].load_tensor(&fp_name)?, &fp_name)?);
        }
        base_fingerprints.insert(layer, fps);
    }

    // For each non-base model, compute fingerprints and solve Hungarian.
    for (m_idx, loader) in loaders.iter().enumerate().skip(1) {
        let names = loader.tensor_names();
        for (&layer, experts) in &by_layer {
            let base_fps = base_fingerprints.get(&layer).unwrap();
            let n = experts.len();
            let mut model_fps: Vec<Vec<f32>> = Vec::with_capacity(n);
            let mut had_all = true;
            for &e in experts {
                let fp_name = match find_fingerprint_name(*loader, layer, e, &names) {
                    Some(s) => s,
                    None => {
                        had_all = false;
                        break;
                    }
                };
                model_fps.push(fingerprint(loader.load_tensor(&fp_name)?, &fp_name)?);
            }
            if !had_all {
                // Layer can't be aligned — leave identity.
                continue;
            }

            // Cost matrix: -cosine_sim (Hungarian minimizes).
            let mut cost = vec![vec![0.0_f32; n]; n];
            for i in 0..n {
                for j in 0..n {
                    cost[i][j] = -dot(&base_fps[i], &model_fps[j]);
                }
            }
            let assignment = hungarian(&cost);
            // assignment[i] = j means base expert i corresponds to model expert j.
            // Therefore, in the model's frame, expert j should be remapped to i.
            // We store the *model→base* permutation: perm[j] = i.
            let mut perm = vec![0_usize; n];
            for (i, &j) in assignment.iter().enumerate() {
                perm[j] = i;
            }
            alignment.permutations[m_idx].per_layer.insert(layer, perm);
        }
    }

    Ok(alignment)
}

fn l2_normalize(v: &mut [f32]) {
    let mut s = 0.0_f64;
    for &x in v.iter() {
        s += (x as f64) * (x as f64);
    }
    let n = s.sqrt() as f32;
    if n > 1e-12 {
        let inv = 1.0_f32 / n;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut s = 0.0_f32;
    for i in 0..n {
        s += a[i] * b[i];
    }
    s
}

/// Hungarian algorithm (textbook O(N³)) for square cost matrix
/// minimization. Returns `assignment[row] = col`.
fn hungarian(cost: &[Vec<f32>]) -> Vec<usize> {
    let n = cost.len();
    if n == 0 {
        return Vec::new();
    }
    debug_assert!(cost.iter().all(|r| r.len() == n));

    // Use Jonker-Volgenant-style implementation with arrays of size n+1
    // to handle one-based indexing common in textbook formulations.
    let inf = f32::INFINITY;
    let mut u = vec![0.0_f32; n + 1];
    let mut v = vec![0.0_f32; n + 1];
    let mut p = vec![0_usize; n + 1]; // p[j] = row assigned to col j
    let mut way = vec![0_usize; n + 1];

    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0_usize;
        let mut minv = vec![inf; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = inf;
            let mut j1 = 0_usize;
            for j in 1..=n {
                if !used[j] {
                    let cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    let mut assignment = vec![0_usize; n];
    for j in 1..=n {
        if p[j] != 0 {
            assignment[p[j] - 1] = j - 1;
        }
    }
    assignment
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_layer_expert_indices() {
        assert_eq!(
            parse_layer_expert("model.layers.7.mlp.experts.3.gate_proj.weight"),
            Some((7, 3))
        );
        assert_eq!(
            parse_layer_expert(
                "language_model.model.layers.4.feed_forward.experts.0.up_proj.weight"
            ),
            Some((4, 0))
        );
        // Non-MoE: returns None.
        assert_eq!(
            parse_layer_expert("model.layers.0.self_attn.q_proj.weight"),
            None
        );
        assert_eq!(parse_layer_expert("lm_head.weight"), None);
    }

    /// Hungarian solver: identity assignment when the cost matrix is
    /// already aligned (zero cost on the diagonal, large cost elsewhere).
    #[test]
    fn hungarian_identity_when_aligned() {
        let n = 4;
        let mut cost = vec![vec![10.0_f32; n]; n];
        for (i, row) in cost.iter_mut().enumerate().take(n) {
            row[i] = 0.0;
        }
        let a = hungarian(&cost);
        for (i, &j) in a.iter().enumerate() {
            assert_eq!(i, j, "expected identity, got {} → {}", i, j);
        }
    }

    /// Hungarian solver: swapped pairs.
    #[test]
    fn hungarian_swap_when_misaligned() {
        // Optimal assignment is the swap (0→1, 1→0): cost on the swap is 0,
        // identity costs 5.
        let cost = vec![vec![5.0_f32, 0.0], vec![0.0, 5.0]];
        let a = hungarian(&cost);
        assert_eq!(a, vec![1, 0]);
    }

    /// Tensor-name remap leaves non-MoE names untouched.
    #[test]
    fn remap_passes_through_non_moe() {
        let alignment = MoeAlignment {
            permutations: vec![ModelPermutation::default(), {
                let mut p = ModelPermutation::default();
                p.per_layer.insert(0, vec![1, 0]); // swap experts 0 ↔ 1 in layer 0
                p
            }],
        };
        let untouched = "model.layers.0.self_attn.q_proj.weight";
        assert_eq!(alignment.remap_tensor_name(1, untouched), untouched);
    }

    /// Tensor-name remap rewrites the expert index per the permutation.
    #[test]
    fn remap_swaps_expert_indices() {
        let alignment = MoeAlignment {
            permutations: vec![ModelPermutation::default(), {
                let mut p = ModelPermutation::default();
                p.per_layer.insert(0, vec![1, 0]); // model expert 0 ↔ base expert 1
                p
            }],
        };
        // In *model 1's* frame, expert 0 should be rewritten to expert 1
        // (because perm[0]=1 means model expert 0 corresponds to base 1).
        assert_eq!(
            alignment.remap_tensor_name(1, "model.layers.0.mlp.experts.0.gate_proj.weight"),
            "model.layers.0.mlp.experts.1.gate_proj.weight"
        );
        // Identity for model 0 (the base).
        assert_eq!(
            alignment.remap_tensor_name(0, "model.layers.0.mlp.experts.0.gate_proj.weight"),
            "model.layers.0.mlp.experts.0.gate_proj.weight"
        );
    }
}
