//! Holding a load to account: every checkpoint tensor consumed, every
//! parameter loaded, or the difference named and justified.
//!
//! A checkpoint that loads "successfully" while a tensor went nowhere, or while
//! a parameter kept the random value it was constructed with, is a model that
//! computes something other than what was published. Nothing downstream can
//! tell: the logits are finite and plausibly shaped. Two classes of bug lived
//! here before this module existed:
//!
//! * **A tensor with no parameter.** `assign_loaded_weights` matches by exact
//!   name and used to log the leftovers at `info`, so a renamed expert, a
//!   dropped router bias or a fused projection loaded as nothing at all.
//! * **A parameter with no tensor.** Nothing ever asked the model which of its
//!   parameters the load had touched, so an architecture that constructed a
//!   layer the checkpoint does not fill (Gemma 3's QK-norm on a Gemma 2
//!   config, a tied head that is not tied) ran on its initial values.
//!
//! The second half works for every architecture without its loader's help:
//! [`ParamSnapshot`] remembers which array each parameter held before the
//! load, and a parameter still holding that same array afterwards was never
//! assigned.

use std::borrow::Cow;
use std::collections::HashMap;

use pmetal_runtime::compat::{Array, ModuleParametersExt};

use super::LoadError;

/// A deliberate exception to the rule that a load consumes everything: a
/// checkpoint tensor or a model parameter that is allowed to go unmatched, with
/// the reason written down where the exemption is.
#[derive(Debug, Clone)]
pub struct Exemption {
    /// A name pattern: `*suffix`, `prefix*`, `*contains*` or an exact name.
    pub pattern: Cow<'static, str>,
    /// Why this is safe: what the tensor or parameter is, and why the model
    /// does not need it from this checkpoint.
    pub reason: &'static str,
}

impl Exemption {
    /// An exemption for `pattern`.
    pub const fn new(pattern: &'static str, reason: &'static str) -> Self {
        Self {
            pattern: Cow::Borrowed(pattern),
            reason,
        }
    }

    /// An exemption for a pattern built at run time, from the config.
    pub fn owned(pattern: String, reason: &'static str) -> Self {
        Self {
            pattern: Cow::Owned(pattern),
            reason,
        }
    }

    /// Whether `name` is covered.
    pub fn covers(&self, name: &str) -> bool {
        let p = self.pattern.as_ref();
        match (p.strip_prefix('*'), p.strip_suffix('*')) {
            (Some(rest), Some(_)) => {
                let inner = rest.strip_suffix('*').unwrap_or(rest);
                name.contains(inner)
            }
            (Some(suffix), None) => name.ends_with(suffix),
            (None, Some(prefix)) => name.starts_with(prefix),
            (None, None) => name == p,
        }
    }
}

/// What a load is allowed to leave unmatched.
#[derive(Debug, Clone, Default)]
pub struct LoadPolicy {
    /// Checkpoint tensors no parameter has to consume.
    pub unused_tensors: Vec<Exemption>,
    /// Parameters the checkpoint does not have to fill: they keep the value
    /// the model was constructed with, which must be the right one.
    pub unloaded_params: Vec<Exemption>,
}

/// Checkpoint tensors that are never a parameter of any architecture here.
///
/// Every architecture's policy starts from these.
pub const NEVER_PARAMETERS: &[Exemption] = &[
    Exemption::new(
        "*rotary_emb.inv_freq",
        "a RoPE frequency buffer; the runtime computes its own from the config",
    ),
    Exemption::new(
        "*.num_batches_tracked",
        "a batch-norm training counter, not a weight",
    ),
    Exemption::new(
        "*position_ids",
        "an index buffer the checkpoint serialises; positions are generated",
    ),
];

impl LoadPolicy {
    /// A policy that exempts only [`NEVER_PARAMETERS`].
    pub fn strict() -> Self {
        Self {
            unused_tensors: NEVER_PARAMETERS.to_vec(),
            unloaded_params: Vec::new(),
        }
    }

    /// Also allow checkpoint tensors matching `exemptions` to go unused.
    #[must_use]
    pub fn allowing_unused_tensors(mut self, exemptions: &[Exemption]) -> Self {
        self.unused_tensors.extend_from_slice(exemptions);
        self
    }

    /// Also allow parameters matching `exemptions` to stay as constructed.
    #[must_use]
    pub fn allowing_unloaded_params(mut self, exemptions: &[Exemption]) -> Self {
        self.unloaded_params.extend_from_slice(exemptions);
        self
    }

    /// The checkpoint tensors in `names` that are neither exempt nor consumed.
    pub(crate) fn unconsumed<'a>(
        &self,
        names: impl IntoIterator<Item = &'a String>,
    ) -> Vec<String> {
        let mut out: Vec<String> = names
            .into_iter()
            .filter(|name| !self.unused_tensors.iter().any(|e| e.covers(name)))
            .cloned()
            .collect();
        out.sort();
        out
    }
}

/// The arrays a model's parameters held before its checkpoint was loaded.
///
/// ⚠️ The snapshot keeps clones of those arrays rather than just their ids. An
/// id is an address, and an array that has been dropped hands its address to
/// the next one allocated, so a parameter that *was* assigned could otherwise
/// read as untouched.
pub struct ParamSnapshot {
    before: HashMap<String, Array>,
}

impl ParamSnapshot {
    /// Record what `model`'s parameters hold now.
    pub fn of<M: ModuleParametersExt>(model: &M) -> Self {
        let before = model
            .flatten_params()
            .into_iter()
            .map(|(name, array)| (name.to_string(), array))
            .collect();
        Self { before }
    }

    /// Parameters of `model` that were not assigned since the snapshot, and
    /// that `policy` does not exempt, sorted by name. A parameter the model
    /// gained since (stacked experts built after loading) is not one the
    /// checkpoint was asked to fill.
    pub fn unloaded<M: ModuleParametersExt>(&self, model: &M, policy: &LoadPolicy) -> Vec<String> {
        let mut out: Vec<String> = model
            .flatten_params()
            .into_iter()
            .filter(|(name, now)| {
                self.before
                    .get(&**name)
                    .is_some_and(|was| was.id() == now.id())
            })
            .map(|(name, _)| name.to_string())
            .filter(|name| !policy.unloaded_params.iter().any(|e| e.covers(name)))
            .collect();
        out.sort();
        out
    }

    /// Fail unless every parameter of `model` was assigned (bar those `policy`
    /// exempts) and nothing in `unconsumed` is left over.
    pub fn require_complete<M: ModuleParametersExt>(
        &self,
        model: &M,
        policy: &LoadPolicy,
        unconsumed: Vec<String>,
    ) -> Result<(), LoadError> {
        let unloaded = self.unloaded(model, policy);
        if unloaded.is_empty() && unconsumed.is_empty() {
            return Ok(());
        }
        Err(LoadError::Incomplete {
            unconsumed,
            unloaded,
            hint: None,
        })
    }
}

/// How many names an [`LoadError::Incomplete`] message spells out.
const SAMPLE: usize = 8;

/// `a, b, c, ... (N total)`.
pub(super) fn sample(names: &[String]) -> String {
    let shown = names
        .iter()
        .take(SAMPLE)
        .cloned()
        .collect::<Vec<_>>()
        .join(", ");
    if names.len() > SAMPLE {
        format!("{shown}, ... ({} in all)", names.len())
    } else {
        shown
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patterns_match_by_position_of_the_star() {
        let suffix = Exemption::new("*.inv_freq", "");
        assert!(suffix.covers("model.layers.0.rotary_emb.inv_freq"));
        assert!(!suffix.covers("model.inv_freq.weight"));

        let prefix = Exemption::new("model.visual.*", "");
        assert!(prefix.covers("model.visual.patch_embed.weight"));
        assert!(!prefix.covers("vision.model.visual.x"));

        let contains = Exemption::new("*.mtp.*", "");
        assert!(contains.covers("model.mtp.fc.weight"));
        assert!(!contains.covers("model.layers.0.weight"));

        let exact = Exemption::new("lm_head.weight", "");
        assert!(exact.covers("lm_head.weight"));
        assert!(!exact.covers("model.lm_head.weight"));
    }

    #[test]
    fn strict_policy_exempts_only_buffers() {
        let policy = LoadPolicy::strict();
        let names = vec![
            "model.layers.0.self_attn.rotary_emb.inv_freq".to_string(),
            "model.layers.0.self_attn.q_proj.weight".to_string(),
        ];
        assert_eq!(
            policy.unconsumed(&names),
            vec!["model.layers.0.self_attn.q_proj.weight".to_string()]
        );
    }
}
