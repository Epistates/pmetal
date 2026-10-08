//! A Qwen3.5-family vision checkpoint against transformers on a real image
//! prompt. Opt-in: needs the checkpoint and a reference dumped by
//! `.strategy/parity/dump_qwen3_5_vision_real_reference.py`.
//!
//! ```text
//! PMETAL_QWEN3_5_VL_DIR=<snapshot> PMETAL_QWEN3_5_VL_REFERENCE=<dump dir> \
//!     cargo test -p pmetal-models --test qwen3_5_vision_real_weights -- --nocapture
//! ```
//!
//! The reference ran in f32 on CPU, pmetal runs the checkpoint's bf16, so the
//! checks are the ones bf16 can be held to: the processor's pixels exactly,
//! the vision features by cosine similarity, the next-token argmax at the end
//! of the prompt on both engines, and the greedy continuation token for token.

use std::path::{Path, PathBuf};

use pmetal_bridge::compat::{Array, Module, ops::slice_axis};
use pmetal_data::qwen_vl_processing::load_image_source;
use pmetal_mlx::test_utils::{load_shard, to_f32_vec_eval};
use pmetal_models::DynamicModel;
use pmetal_models::architectures::qwen3_5_vision::Qwen3_5Vision;
use serial_test::serial;

fn env_dir(key: &str) -> Option<PathBuf> {
    std::env::var_os(key).map(PathBuf::from)
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    dot / (na * nb)
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap()
}

fn top_k(values: &[f32], k: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[b].total_cmp(&values[a]));
    order.truncate(k);
    order
}

fn drain(op: &str) {
    pmetal_bridge::check_last_error().unwrap_or_else(|e| panic!("{op}: bridge error: {e}"));
}

fn last_row(logits: &Array) -> Vec<f32> {
    let t = logits.dim(1);
    to_f32_vec_eval(&slice_axis(logits, 1, t - 1, t))
}

#[test]
#[serial]
fn real_checkpoint_matches_transformers() {
    let (Some(model_dir), Some(reference_dir)) = (
        env_dir("PMETAL_QWEN3_5_VL_DIR"),
        env_dir("PMETAL_QWEN3_5_VL_REFERENCE"),
    ) else {
        eprintln!("skipping: set PMETAL_QWEN3_5_VL_DIR and PMETAL_QWEN3_5_VL_REFERENCE");
        return;
    };
    let reference = load_shard(&reference_dir.join("reference.safetensors"));
    let get = |key: &str| {
        reference
            .get(key)
            .unwrap_or_else(|| panic!("{key}"))
            .clone()
    };
    let ids: Vec<u32> = to_f32_vec_eval(&get("input_ids"))
        .into_iter()
        .map(|v| v as u32)
        .collect();
    let want_tokens: Vec<u32> = to_f32_vec_eval(&get("generated"))
        .into_iter()
        .map(|v| v as u32)
        .collect();

    // Processor: exact pixels from the PNG.
    let vision = Qwen3_5Vision::load(&model_dir).unwrap();
    let image = load_image_source(&reference_dir.join("image.png").to_string_lossy()).unwrap();
    let image = vision.processor.preprocess_image(&image).unwrap();
    let want_grid = to_f32_vec_eval(&get("image_grid_thw"));
    assert_eq!(
        image.grid_thw.map(|v| v as f32).to_vec(),
        want_grid,
        "image grid"
    );
    assert_eq!(
        image.pixel_values,
        to_f32_vec_eval(&get("pixel_values")),
        "pixel values"
    );

    // Vision tower.
    let media = vision.encode(&ids, &[image], &[]).unwrap();
    let features = to_f32_vec_eval(media.image_features.as_ref().unwrap());
    let want_features = to_f32_vec_eval(&get("image_features"));
    let feature_cosine = cosine(&features, &want_features);
    println!("vision features (checkpoint dtype): cosine {feature_cosine:.6}");
    assert!(
        feature_cosine > 0.99,
        "vision features cosine {feature_cosine}"
    );
    // The same tower in f32 is the reference's arithmetic: no rounding left
    // to hide a porting error behind.
    {
        let mut vision32 = Qwen3_5Vision::load(&model_dir).unwrap();
        vision32
            .tower
            .set_dtype(pmetal_bridge::compat::Dtype::Float32);
        let image = vision32
            .processor
            .preprocess_image(
                &load_image_source(&reference_dir.join("image.png").to_string_lossy()).unwrap(),
            )
            .unwrap();
        let media32 = vision32.encode(&ids, &[image], &[]).unwrap();
        let features32 = to_f32_vec_eval(media32.image_features.as_ref().unwrap());
        let max_diff = features32
            .iter()
            .zip(&want_features)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let scale = want_features.iter().map(|v| v.abs()).fold(0f32, f32::max);
        let cosine32 = cosine(&features32, &want_features);
        println!(
            "vision features (f32): cosine {cosine32:.8}, max |diff| {max_diff:.3e} of {scale:.3e}"
        );
        assert!(cosine32 > 0.99999, "f32 vision features cosine {cosine32}");
    }

    let want_logits = to_f32_vec_eval(&get("prompt_logits"));
    let report = |engine: &str, got: &[f32]| {
        let diff = got
            .iter()
            .zip(&want_logits)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let overlap = top_k(got, 5)
            .iter()
            .filter(|i| top_k(&want_logits, 5).contains(i))
            .count();
        println!(
            "{engine}: argmax {} (ref {}), max |dlogit| {diff:.3}, top-5 overlap {overlap}/5, \
             logit cosine {:.6}",
            argmax(got),
            argmax(&want_logits),
            cosine(got, &want_logits)
        );
        assert_eq!(argmax(got), argmax(&want_logits), "{engine}: next token");
    };

    let id_array = |ids: &[u32]| {
        Array::from_i32_slice_shaped(
            &ids.iter().map(|&i| i as i32).collect::<Vec<_>>(),
            &[1, ids.len() as i32],
        )
    };

    // Native engine: prefill, then greedy decode.
    {
        use pmetal_bridge::qwen3_native::{
            NativeCache, embed_tokens, forward_embeddings_hidden, forward_step_hidden, load_config,
            load_model,
        };
        let config = load_config(&model_dir).unwrap();
        let weights = load_model(&model_dir, &config).unwrap();
        let text = embed_tokens(&weights, &id_array(&ids));
        let embeddings = media.merge(&text, &ids, &vision.config).unwrap();
        let tables = config.mrope_tables(&media.positions.array());
        let mut cache = NativeCache::new_empty(&weights);
        let (_, logits) = forward_embeddings_hidden(
            &weights,
            &embeddings,
            &tables,
            media.positions.next_position,
            &mut cache,
        );
        drain("native prefill");
        let row = last_row(&logits);
        report("native", &row);
        let mut tokens = vec![argmax(&row) as u32];
        while tokens.len() < want_tokens.len() {
            let (_, logits) =
                forward_step_hidden(&weights, &id_array(&tokens[tokens.len() - 1..]), &mut cache);
            drain("native decode");
            tokens.push(argmax(&last_row(&logits)) as u32);
        }
        let agree = tokens
            .iter()
            .zip(&want_tokens)
            .take_while(|(a, b)| a == b)
            .count();
        println!(
            "native greedy: {agree}/{} tokens agree with the reference",
            want_tokens.len()
        );
        assert!(
            agree >= want_tokens.len().min(8),
            "native greedy diverged at {agree}"
        );
    }

    // Dynamic engine: prefill only.
    let mut model = DynamicModel::load(Path::new(&model_dir)).unwrap();
    let qwen = model.as_qwen3_next_mut().unwrap();
    let text = Module::forward(&mut qwen.model.embed_tokens, &id_array(&ids)).unwrap();
    let embeddings = media.merge(&text, &ids, &vision.config).unwrap();
    let (_, logits) = qwen
        .forward_embeddings(&embeddings, &media.positions.array(), None, None)
        .unwrap();
    drain("dynamic prefill");
    report("dynamic", &last_row(&logits));
}
