//! The shared-path DeepSeek (`DynamicModel::DeepSeek`) on the released
//! `deepseek-ai/DeepSeek-V2-Lite`, depth by depth, against the native engine
//! on the same bf16 tensors.
//!
//! The two are different programs over the same weights (the native engine
//! caches the MLA latent and runs packed kernels, the shared path expands K and
//! V and is what training runs), so a layer where their logits part is a layer
//! one of them gets wrong. The engine is held to transformers by
//! `deepseek_native_real_weights`; this holds the shared path to the engine.
//!
//! The model is 31 GB and not committed, so this is `#[ignore]`d and gated on
//! the variables (run it alone, it needs the GPU's memory):
//!
//! ```bash
//! PMETAL_DEEPSEEK_V2_LITE_DIR=<snapshot dir> \
//!     cargo test -p pmetal-models --test deepseek_dynamic_real_weights -- --ignored --nocapture
//! ```
//!
//! `PMETAL_DEEPSEEK_DEPTHS` (comma-separated layer counts) picks the depths;
//! with `PMETAL_DEEPSEEK_V2_LITE_REFERENCE` set to a transformers dump
//! (`input_ids`, `logits`), the full-depth (27 layers) logits of both are also
//! compared to it.

use std::path::PathBuf;

use pmetal_cache::test_utils::{load_shard, ref_tensor};
use pmetal_models::architectures::deepseek::{DeepSeek, DeepSeekConfig, stack_experts};
use pmetal_models::loader::load_weights;
use pmetal_runtime::compat::{Array, Dtype, ModuleParametersExt};
use pmetal_runtime::deepseek_native::{NativeCache, forward_step, load_config, load_model};
use serial_test::serial;

fn drain(op: &str) {
    pmetal_runtime::check_unobserved_error().unwrap_or_else(|e| panic!("{op}: runtime error: {e}"));
}

/// Per-position cosine of `[1, T, vocab]` logits.
fn cosines(got: &Array, want: &Array) -> Vec<f32> {
    let f32 = Dtype::Float32.as_i32();
    let (got, want) = (got.as_dtype(f32), want.as_dtype(f32));
    let dot = got.multiply(&want).sum_axis(-1, false);
    let norms = got
        .square()
        .sum_axis(-1, false)
        .sqrt()
        .multiply(&want.square().sum_axis(-1, false).sqrt());
    let cos = dot.divide(&norms).squeeze(0);
    cos.eval();
    cos.as_slice::<f32>().to_vec()
}

fn worst(cos: &[f32]) -> f32 {
    cos.iter().copied().fold(f32::INFINITY, f32::min)
}

fn input_ids() -> Array {
    // Arithmetic ids, as the sweep's references use.
    let ids: Vec<i32> = (0..16).map(|i| (i * 37 + 11) % 1000 + 100).collect();
    Array::from_i32_slice_shaped(&ids, &[1, 16])
}

#[test]
#[ignore = "needs the 31 GB DeepSeek-V2-Lite checkpoint"]
#[serial]
fn dynamic_deepseek_v2_lite_tracks_the_native_engine_by_depth() {
    let Some(dir) = std::env::var_os("PMETAL_DEEPSEEK_V2_LITE_DIR") else {
        eprintln!("PMETAL_DEEPSEEK_V2_LITE_DIR not set; nothing to check");
        return;
    };
    let dir = PathBuf::from(dir);
    let reference = std::env::var_os("PMETAL_DEEPSEEK_V2_LITE_REFERENCE")
        .map(|file| load_shard(&PathBuf::from(file)));
    let ids = reference
        .as_ref()
        .map_or_else(input_ids, |r| ref_tensor(r, "input_ids").clone());
    let depths: Vec<i32> = std::env::var("PMETAL_DEEPSEEK_DEPTHS")
        .ok()
        .map(|s| s.split(',').map(|d| d.trim().parse().unwrap()).collect())
        .unwrap_or_else(|| vec![1, 2, 3, 4, 6]);
    let config_text = std::fs::read_to_string(dir.join("config.json")).unwrap();

    for n in depths {
        let native = {
            let mut config = load_config(&dir).expect("config parses");
            config.num_hidden_layers = n;
            let weights = load_model(&dir, &config).expect("native weights load");
            let mut cache = NativeCache::new_empty(&weights);
            let logits = forward_step(&weights, &ids, &mut cache);
            drain("native forward");
            logits.eval();
            logits
        };

        // The shared path at depth n: the model built at n layers, the
        // checkpoint's other layers' tensors left unread.
        let mut config: DeepSeekConfig = json5::from_str(&config_text).expect("config parses");
        config.num_hidden_layers = n;
        let mut model = DeepSeek::new(config).expect("model builds");
        let weights = stack_experts(load_weights(&dir).expect("tensors open")).expect("stack");
        let mut params = model.flatten_params_mut();
        for (key, value) in weights {
            if let Some(param) = params.get_mut(key.as_str()) {
                **param = value;
            }
        }
        model.eval().expect("weights read");
        drain("shared-path load");
        let logits = model.forward(&ids, None, None).expect("forward");
        drain("shared-path forward");

        let cos = cosines(&logits, &native);
        println!(
            "{n:>2} layers: shared path vs native engine, worst cosine {:.6}",
            worst(&cos)
        );
        println!("    per position: {cos:.5?}");
        if let Some(r) = &reference
            && n == 27
        {
            let want = ref_tensor(r, "logits");
            println!(
                "    vs transformers: shared path worst {:.6}, native worst {:.6}",
                worst(&cosines(&logits, want)),
                worst(&cosines(&native, want)),
            );
        }
    }
}
