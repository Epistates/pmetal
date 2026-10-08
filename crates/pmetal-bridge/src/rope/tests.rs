use super::*;
use serde_json::json;

fn rotary(config: Value, head_dim: i32) -> Result<Rotary, String> {
    Rotary::from_config(head_dim, RopeConfig::from_json(&config), 10_000.0, 1.0)
}

fn max_diff(a: &Array, b: &Array) -> f32 {
    let d = a.subtract(b).abs();
    let d = (0..d.ndim()).fold(d, |d, _| d.max_axis(-1, false));
    d.item_f32()
}

fn ramp(shape: &[i32]) -> Array {
    let n: i32 = shape.iter().product();
    let values: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin()).collect();
    Array::from_f32_slice(&values, shape)
}

#[test]
fn unknown_rope_types_are_refused_by_name() {
    for t in ["ntk", "dynamic-ntk", "longrope2", "mrope_v2"] {
        let err =
            rotary(json!({"rope_scaling": {"rope_type": t, "factor": 2.0}}), 64).expect_err(t);
        assert!(
            err.contains(t) && err.contains("not supported"),
            "{t}: {err}"
        );
    }
    // The legacy `type` spelling is read, and refused just the same.
    let err = rotary(json!({"rope_scaling": {"type": "xpos"}}), 64).unwrap_err();
    assert!(err.contains("xpos"), "{err}");
}

#[test]
fn missing_required_keys_are_refused_by_name() {
    let cases = [
        (json!({"rope_type": "linear"}), "factor"),
        (
            json!({"rope_type": "llama3", "factor": 8.0, "high_freq_factor": 4.0}),
            "low_freq_factor",
        ),
        (
            json!({"rope_type": "longrope", "short_factor": vec![1.0; 32]}),
            "long_factor",
        ),
        (json!({"rope_type": "yarn"}), "factor"),
        (
            json!({"rope_type": "dynamic", "factor": 2.0}),
            "max_position_embeddings",
        ),
    ];
    for (params, key) in cases {
        let err = rotary(
            json!({"rope_scaling": params, "original_max_position_embeddings": 64}),
            64,
        )
        .expect_err(key);
        assert!(err.contains(key), "{key}: {err}");
    }
    let err = rotary(
        json!({"rope_scaling": {"rope_type": "longrope", "short_factor": vec![1.0; 4],
                "long_factor": vec![1.0; 4],"original_max_position_embeddings": 64}}),
        64,
    )
    .unwrap_err();
    assert!(err.contains("short_factor has 4 entries"), "{err}");
}

#[test]
fn the_rope_dict_and_its_keys_resolve_as_transformers_resolves_them() {
    // A non-empty `rope_scaling` wins over `rope_parameters`; `rope_theta`
    // inside the dict wins over the top-level key.
    let r = rotary(
        json!({"rope_theta": 1e4, "max_position_embeddings": 4096,
               "rope_scaling": {"rope_type": "linear", "factor": 2.0, "rope_theta": 5e5},
               "rope_parameters": {"rope_type": "default", "rope_theta": 1e6}}),
        64,
    )
    .unwrap();
    assert_eq!(r.scaling, RopeScaling::Linear { factor: 2.0 });
    assert_eq!(r.theta, 5e5);
    // An empty or null `rope_scaling` leaves `rope_parameters` in force.
    for scaling in [json!({}), Value::Null] {
        let r = rotary(
            json!({"rope_scaling": scaling,
                   "rope_parameters": {"rope_type": "default", "rope_theta": 1e6,
                                       "partial_rotary_factor": 0.5}}),
            64,
        )
        .unwrap();
        assert_eq!((r.theta, r.dims), (1e6, 32));
    }
    // A top-level original_max_position_embeddings overrides the dict's
    // (Phi-3); without either, max_position_embeddings stands in.
    let r = rotary(
        json!({"max_position_embeddings": 8192, "original_max_position_embeddings": 1024,
               "rope_scaling": {"rope_type": "yarn", "factor": 4.0,
                                "original_max_position_embeddings": 2048}}),
        64,
    )
    .unwrap();
    let RopeScaling::Yarn(y) = &r.scaling else {
        panic!()
    };
    assert_eq!(y.original_max_position_embeddings, 1024.0);
    let r = rotary(
        json!({"max_position_embeddings": 8192,
               "rope_scaling": {"rope_type": "yarn", "factor": 4.0}}),
        64,
    )
    .unwrap();
    let RopeScaling::Yarn(y) = &r.scaling else {
        panic!()
    };
    assert_eq!(y.original_max_position_embeddings, 8192.0);
    assert!(y.truncate);
    // Plain configs are plain, and say so to the fused kernel.
    assert_eq!(
        rotary(json!({}), 64).unwrap().scalar(),
        Some((10_000.0, 1.0))
    );
    let linear = rotary(
        json!({"rope_scaling": {"type": "linear", "factor": 4.0}}),
        64,
    )
    .unwrap();
    assert_eq!(linear.scalar(), Some((10_000.0, 0.25)));
}

/// The fused kernel with a period table, the explicit-position tables, and
/// the scalar kernel are three implementations of one rotation; every
/// scaling type has to land on the same numbers through each.
#[test]
fn contiguous_and_explicit_positions_agree_for_every_scaling() {
    let (heads, len, head_dim) = (2, 6, 32);
    let configs = [
        json!({"rope_scaling": {"rope_type": "default"}}),
        json!({"rope_scaling": {"rope_type": "linear", "factor": 3.0}}),
        json!({"max_position_embeddings": 16, "rope_scaling": {"rope_type": "dynamic", "factor": 2.0}}),
        json!({"rope_scaling": {"rope_type": "yarn", "factor": 4.0,
                                "original_max_position_embeddings": 16}}),
        json!({"partial_rotary_factor": 0.5, "rope_scaling": {"rope_type": "yarn", "factor": 4.0,
               "original_max_position_embeddings": 16, "truncate": false}}),
        json!({"rope_scaling": {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                                "high_freq_factor": 4.0, "original_max_position_embeddings": 64}}),
        json!({"max_position_embeddings": 128, "partial_rotary_factor": 0.5,
               "rope_scaling": {"rope_type": "longrope", "original_max_position_embeddings": 16,
                                "short_factor": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
                                "long_factor": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]}}),
        json!({"rope_scaling": {"rope_type": "proportional", "partial_rotary_factor": 0.25,
                                "factor": 2.0}}),
    ];
    let x = ramp(&[1, heads, len, head_dim]);
    for config in configs {
        let r = rotary(config.clone(), head_dim).unwrap();
        for traditional in [false, true] {
            let emb = RotaryEmbedding::new(r.clone(), traditional);
            // Offsets on both sides of every threshold above.
            for offset in [0, 13, 40] {
                let contiguous = emb.apply(&x, offset);
                let ids: Vec<i32> = (offset..offset + len).collect();
                let explicit = emb.apply_at(&x, &Array::from_i32_slice(&ids));
                let d = max_diff(&contiguous, &explicit);
                crate::check_last_error().unwrap();
                assert!(
                    d < 2e-5,
                    "{config} traditional={traditional} offset={offset}: |Δ| = {d:e}"
                );
            }
        }
    }
}

/// transformers multiplies `cos` and `sin` by the attention factor, so the
/// rotated channels carry it and the pass-through tail does not.
#[test]
fn the_attention_factor_scales_only_the_rotated_channels() {
    let (head_dim, dims) = (16, 8);
    let r = rotary(
        json!({"partial_rotary_factor": 0.5,
               "rope_scaling": {"rope_type": "yarn", "factor": 4.0,
                                "original_max_position_embeddings": 32}}),
        head_dim,
    )
    .unwrap();
    let gain = r.attention_factor();
    assert!((gain as f64 - (0.1 * 4f64.ln() + 1.0)).abs() < 1e-6);
    let x = ramp(&[1, 1, 3, head_dim]);
    let unscaled = RotaryEmbedding::new(
        Rotary {
            scaling: RopeScaling::Yarn(Yarn {
                attention_factor: 1.0,
                ..match &r.scaling {
                    RopeScaling::Yarn(y) => y.clone(),
                    _ => unreachable!(),
                }
            }),
            ..r.clone()
        },
        false,
    )
    .apply(&x, 5);
    let scaled = RotaryEmbedding::new(r, false).apply(&x, 5);
    let head = |a: &Array| ops::slice_axis(a, 3, 0, dims);
    let tail = |a: &Array| ops::slice_axis(a, 3, dims, head_dim);
    assert!(
        max_diff(
            &head(&scaled),
            &head(&unscaled).multiply(&Array::from_f32(gain))
        ) < 1e-6
    );
    assert_eq!(max_diff(&tail(&scaled), &tail(&x)), 0.0);
    crate::check_last_error().unwrap();
}

/// LongRoPE and dynamic NTK re-decide from each forward's reach.
#[test]
fn reach_dependent_scalings_switch_at_their_threshold() {
    let long = rotary(
        json!({"max_position_embeddings": 64,
               "rope_scaling": {"rope_type": "longrope", "original_max_position_embeddings": 16,
                                "short_factor": vec![1.0; 4], "long_factor": [1.0, 2.0, 4.0, 8.0]}}),
        8,
    )
    .unwrap();
    assert_eq!(long.inverse_frequencies(16), long.inverse_frequencies(0));
    assert_ne!(long.inverse_frequencies(17), long.inverse_frequencies(16));
    let dynamic = rotary(
        json!({"max_position_embeddings": 16, "rope_scaling": {"rope_type": "dynamic", "factor": 2.0}}),
        8,
    )
    .unwrap();
    assert_eq!(dynamic.base_at(16), 10_000.0);
    assert!(dynamic.base_at(17) > 10_000.0);
}
