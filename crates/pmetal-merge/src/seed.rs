//! Deterministic seed derivation for the stochastic methods.
//!
//! A merge draws one random mask per (tensor, model) pair. Seeding every
//! draw with the configured seed alone would give each model the same drop
//! pattern and every tensor of equal shape the same one, so the draws are
//! decorrelated by mixing in the tensor name and the model's slot.

/// FNV-1a over the bytes of `name`: a hash that stays the same across
/// platforms and Rust releases, which `std`'s hashers do not promise.
fn fnv1a(name: &str) -> u64 {
    name.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// The splitmix64 finalizer: a bijective avalanche over 64 bits.
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// The seed of the mask for model `slot` in tensor `name`, derived from the
/// merge's `seed`. Distinct (name, slot) pairs give unrelated seeds.
pub(crate) fn mask_seed(seed: u64, name: &str, slot: usize) -> u64 {
    splitmix64(splitmix64(seed ^ fnv1a(name)) ^ slot as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeds_are_stable_and_distinct_per_name_and_slot() {
        let base = mask_seed(7, "model.layers.0.q_proj.weight", 0);
        assert_eq!(base, mask_seed(7, "model.layers.0.q_proj.weight", 0));
        assert_ne!(base, mask_seed(7, "model.layers.0.q_proj.weight", 1));
        assert_ne!(base, mask_seed(7, "model.layers.1.q_proj.weight", 0));
        assert_ne!(base, mask_seed(8, "model.layers.0.q_proj.weight", 0));
    }

    #[test]
    fn fnv1a_matches_the_published_vectors() {
        // https://datatracker.ietf.org/doc/html/draft-eastlake-fnv
        assert_eq!(fnv1a(""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a("a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a("foobar"), 0x8594_4171_f739_67e8);
    }
}
