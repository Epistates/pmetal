//! MoE expert permutation alignment integration test.
//!
//! Two synthetic "MoE checkpoints" with two experts each, where checkpoint
//! B has its experts swapped relative to A. With `align_moe_experts: false`
//! the linear merge mixes A.expert.0 with B.expert.0 and produces a
//! garbled bank. With `align_moe_experts: true` the alignment pre-pass
//! detects the swap, remaps B's tensor names, and the merge collapses to
//! the desired per-expert blend.

use pmetal_merge::{MergeConfig, MergeMethodConfig, ModelConfig, SanityLevel, run_merge};
use safetensors::Dtype;
use safetensors::tensor::TensorView;
use std::collections::HashMap;
use std::path::Path;

/// Build an "expert fingerprint" that points in a distinctive direction.
/// `seed` selects the direction (different seeds → different unit vectors
/// after L2 normalization), `scale` sets the magnitude. Cosine similarity
/// between two stubs with the same `seed` is 1.0; with different seeds
/// it's < 1.0, which is what the alignment pre-pass needs to discriminate.
fn expert_stub(scale: f32, seed: u32, n: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; n];
    let mut state = seed.wrapping_mul(0x9E37_79B1).wrapping_add(0x1234_5678);
    for slot in v.iter_mut() {
        // tiny LCG → reproducible per-seed direction.
        state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        let bits = (state >> 16) as i32 - 16384;
        *slot = scale * (bits as f32 / 16384.0);
    }
    v
}

fn write_moe_model(dir: &Path, layer_idx: usize, experts: &[(usize, Vec<f32>)], dim: usize) {
    std::fs::create_dir_all(dir).unwrap();
    let mut tensors_owned: Vec<(String, Vec<u8>, Vec<usize>)> = Vec::new();
    // Include a non-MoE tensor so the merge has something to do beyond the
    // experts (`embed_tokens.weight`). Same in both checkpoints so this
    // doesn't interfere with the alignment behavior we're checking.
    let embed = vec![1.0_f32; dim];
    tensors_owned.push((
        "model.embed_tokens.weight".to_string(),
        embed.iter().flat_map(|f| f.to_le_bytes()).collect(),
        vec![1, dim],
    ));
    for (e_idx, vals) in experts {
        let name = format!(
            "model.layers.{}.mlp.experts.{}.gate_proj.weight",
            layer_idx, e_idx
        );
        let bytes: Vec<u8> = vals.iter().flat_map(|f| f.to_le_bytes()).collect();
        tensors_owned.push((name, bytes, vec![1, dim]));
    }
    let views: HashMap<&str, TensorView<'_>> = tensors_owned
        .iter()
        .map(|(name, bytes, shape)| {
            (
                name.as_str(),
                TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
            )
        })
        .collect();
    let payload = safetensors::serialize(views, None).unwrap();
    std::fs::write(dir.join("model.safetensors"), payload).unwrap();
    // Minimal config for sidecar copy not to error out.
    std::fs::write(dir.join("config.json"), "{\"model_type\": \"test\"}").unwrap();
}

#[test]
fn align_moe_experts_recovers_per_expert_blend() {
    let dim = 8;
    let workdir = tempfile::tempdir().unwrap();
    let model_a = workdir.path().join("a");
    let model_b = workdir.path().join("b");

    // Distinct expert "identities" (different random directions). B is
    // the same checkpoint with its experts permuted in storage:
    // B.expert(0) holds A.expert(1)'s weights and vice versa.
    let a0 = expert_stub(1.0, /*seed*/ 1, dim);
    let a1 = expert_stub(2.0, /*seed*/ 2, dim);
    let b0 = a1.clone();
    let b1 = a0.clone();
    write_moe_model(&model_a, 0, &[(0, a0.clone()), (1, a1.clone())], dim);
    write_moe_model(&model_b, 0, &[(0, b0.clone()), (1, b1.clone())], dim);

    // Without alignment: the merge averages A.expert(0) with B's stored
    // expert 0 (which is A's expert 1 in disguise) — i.e. it averages two
    // *different* expert identities. With alignment: the pre-pass detects
    // the swap and B's expert 1 is paired with A's expert 0, so the merge
    // averages two copies of the *same* identity → result equals A's
    // expert 0 (and the diff to A.expert.0 is small).
    fn run_with_alignment(a: &Path, b: &Path, align: bool, out: &Path) -> Vec<f32> {
        let config = MergeConfig {
            merge_method: MergeMethodConfig::Linear,
            models: vec![
                ModelConfig {
                    model: a.to_string_lossy().into_owned(),
                    parameters: Default::default(),
                },
                ModelConfig {
                    model: b.to_string_lossy().into_owned(),
                    parameters: Default::default(),
                },
            ],
            base_model: None,
            output_path: Some(out.to_path_buf()),
            dtype: "float32".to_string(),
            parameters: Default::default(),
            tokenizer: None,
            slices: None,
            allow_mixed_dtype: false,
            sanity: SanityLevel::Quick,
            dry_run: false,
            align_moe_experts: align,
            ..Default::default()
        };
        run_merge(&config).unwrap();

        // Reload the merged expert 0 tensor.
        use pmetal_merge::TensorLoader;
        let loader = pmetal_merge::SafetensorsLoader::new(out).unwrap();
        let mut t = loader
            .load_tensor("model.layers.0.mlp.experts.0.gate_proj.weight")
            .unwrap();
        t.to_f32_vec(8).unwrap()
    }

    let plain = run_with_alignment(&model_a, &model_b, false, &workdir.path().join("out_plain"));
    let aligned = run_with_alignment(
        &model_a,
        &model_b,
        true,
        &workdir.path().join("out_aligned"),
    );

    // Compare merged expert 0 to A.expert(0)'s identity. Plain merge
    // mixes two different identities → large distance; aligned merge
    // recovers the per-expert blend → distance ≈ 0.
    fn l2_distance(x: &[f32], y: &[f32]) -> f32 {
        let mut acc = 0.0_f64;
        for (a, b) in x.iter().zip(y.iter()) {
            acc += ((a - b) as f64).powi(2);
        }
        (acc as f32).sqrt()
    }

    let plain_dist = l2_distance(&plain, &a0);
    let aligned_dist = l2_distance(&aligned, &a0);

    assert!(
        plain_dist > aligned_dist,
        "alignment must bring the merged expert closer to A's expert 0 \
         than the plain merge: plain_dist={}, aligned_dist={}",
        plain_dist,
        aligned_dist
    );
    // Sanity: aligned merge averages two copies of A.expert(0) → equals
    // A.expert(0) exactly (within numerical noise).
    assert!(
        aligned_dist < 1e-4,
        "aligned merge should be ~A.expert(0), got distance {}",
        aligned_dist
    );
}
