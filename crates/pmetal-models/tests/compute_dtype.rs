//! A model's KV cache is sized and typed for the dtype the model computes in.

use pmetal_models::DynamicModel;
use pmetal_runtime::compat::{Dtype, ModuleParametersExt};

const CONFIG: &str = r#"{"model_type": "llama", "vocab_size": 64, "hidden_size": 32,
    "intermediate_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
    "num_key_value_heads": 2, "head_dim": 8, "max_position_embeddings": 128,
    "rope_theta": 10000.0, "tie_word_embeddings": false}"#;

fn model_in(dtype: Dtype) -> DynamicModel {
    let mut model = DynamicModel::from_config(CONFIG).unwrap();
    for (_, param) in model.flatten_params_mut() {
        *param = param.as_dtype(dtype.as_i32());
    }
    model
}

#[test]
fn the_compute_dtype_is_the_parameters_dtype() {
    for dtype in [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16] {
        assert_eq!(model_in(dtype).compute_dtype(), dtype);
    }
}

/// ⚠️ The cache config defaulted to f32 whatever the model ran in, so every
/// memory estimate made from it was double for a bf16 or f16 model.
#[test]
fn the_cache_takes_the_models_dtype_and_its_estimate_follows() {
    let f32_model = model_in(Dtype::Float32);
    let bf16_model = model_in(Dtype::Bfloat16);

    let wide = f32_model.create_cache(1024);
    let narrow = bf16_model.create_cache(1024);

    assert_eq!(wide.config().dtype, Dtype::Float32);
    assert_eq!(narrow.config().dtype, Dtype::Bfloat16);
    assert_eq!(
        narrow.config().memory_footprint() * 2,
        wide.config().memory_footprint(),
        "a bf16 cache is half the size of the f32 one"
    );
    // The layout is the architecture's, untouched.
    assert_eq!(narrow.config().num_layers, 2);
    assert_eq!(narrow.config().num_kv_heads, 2);
    assert_eq!(narrow.config().head_dim, 8);
}

#[test]
fn a_cache_with_a_mode_keeps_the_models_dtype() {
    use pmetal_cache::kv_cache::CacheMode;
    let cache = model_in(Dtype::Float16)
        .create_cache_with_mode(1024, CacheMode::SlidingWindow { window_size: 64 });
    assert_eq!(cache.config().dtype, Dtype::Float16);
}
