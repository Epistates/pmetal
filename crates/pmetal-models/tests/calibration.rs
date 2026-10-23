//! Importance-matrix collection over a tiny Llama.
//!
//! The expected statistics are computed here on the host from the model's own
//! weights, independently of the forward pass: the input of the first layer's
//! q/k/v projections is the token embedding through the first RMSNorm, which
//! is a few lines of arithmetic.

use pmetal_models::DynamicModel;
use pmetal_models::architectures::llama::{LlamaConfig, LlamaForCausalLM};
use pmetal_models::calibration::collect_imatrix;
use pmetal_runtime::compat::{Array, Dtype, ModuleParametersExt};

const HIDDEN: usize = 32;
const VOCAB: usize = 64;
const EPS: f32 = 1e-5;

fn tiny_llama(tied: bool) -> DynamicModel {
    let config = LlamaConfig {
        vocab_size: VOCAB as i32,
        hidden_size: HIDDEN as i32,
        intermediate_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: Some(2),
        head_dim: Some(8),
        max_position_embeddings: 64,
        rms_norm_eps: EPS,
        rope_theta: 10000.0,
        rope_scaling: None,
        hidden_act: "silu".to_string(),
        tie_word_embeddings: tied,
        ..Default::default()
    };
    DynamicModel::Llama(LlamaForCausalLM::new(config).expect("model"))
}

fn param(model: &DynamicModel, name: &str) -> Vec<f32> {
    let params = model.flatten_params();
    let array = params
        .iter()
        .find(|(key, _)| &***key == name)
        .map(|(_, array)| array.clone())
        .unwrap_or_else(|| panic!("the model has no parameter {name}"));
    let n = array.size();
    let mut array = array.as_dtype(Dtype::Float32.as_i32());
    array.to_f32_vec(n).expect("read parameter")
}

/// `sum_t rmsnorm(embed[token_t])_j^2` per channel, the input of layer 0's
/// attention projections.
fn expected_attention_input(model: &DynamicModel, tokens: &[u32]) -> Vec<f32> {
    let embed = param(model, "model.embed_tokens.weight");
    let weight = param(model, "model.layers.0.input_layernorm.weight");
    let mut sums = vec![0.0f64; HIDDEN];
    for &t in tokens {
        let row = &embed[t as usize * HIDDEN..(t as usize + 1) * HIDDEN];
        let mean_sq = row.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / HIDDEN as f64;
        let inv = 1.0 / (mean_sq + f64::from(EPS)).sqrt();
        for j in 0..HIDDEN {
            sums[j] += (f64::from(row[j]) * inv * f64::from(weight[j])).powi(2);
        }
    }
    sums.into_iter().map(|v| v as f32).collect()
}

fn assert_close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (j, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-4 * w.abs().max(1e-3),
            "{what}[{j}]: got {g}, expected {w}"
        );
    }
}

#[test]
fn the_first_layers_attention_input_is_measured_exactly() {
    let mut model = tiny_llama(true);
    let chunks: Vec<Vec<u32>> = vec![
        vec![3, 7, 7, 1, 12, 0, 5, 9],
        vec![40, 2, 2, 63, 8, 8, 8, 1],
    ];
    let all_tokens: Vec<u32> = chunks.iter().flatten().copied().collect();

    // A batch of two chunks and two batches of one measure the same thing.
    for batch in [2usize, 1] {
        let run = collect_imatrix(&mut model, &chunks, batch, |_, _| {}).expect("collect");
        assert_eq!(run.chunks, 2);
        assert_eq!(run.tokens, 16);
        assert_eq!(run.rejected, 0);

        let want = expected_attention_input(&model, &all_tokens);
        let n = all_tokens.len() as f32;
        let want_mean: Vec<f32> = want.iter().map(|s| s / n).collect();
        for proj in ["attn_q", "attn_k", "attn_v"] {
            let name = format!("blk.0.{proj}.weight");
            let got = run
                .imatrix
                .importance(&name)
                .unwrap_or_else(|| panic!("no entry for {name}"));
            assert_close(&got, &want_mean, &name);
            assert_eq!(run.imatrix.counts[&name], vec![16.0], "{name} tokens");
        }

        // gate and up read the same input, so they measure the same thing.
        let gate = run.imatrix.importance("blk.1.ffn_gate.weight").unwrap();
        let up = run.imatrix.importance("blk.1.ffn_up.weight").unwrap();
        assert_eq!(gate, up);
        assert_eq!(gate.len(), HIDDEN);
        // The down projection reads the intermediate activation.
        assert_eq!(
            run.imatrix
                .importance("blk.1.ffn_down.weight")
                .unwrap()
                .len(),
            64
        );
        // The tied head reads the final hidden state, under the embedding's name.
        assert_eq!(
            run.imatrix.importance("token_embd.weight").unwrap().len(),
            HIDDEN
        );
        assert_eq!(run.imatrix.last_chunk, Some(2));
        assert_eq!(run.imatrix.chunk_size, Some(8));
    }
}

#[test]
fn every_projection_of_a_dense_model_is_measured() {
    for tied in [true, false] {
        let mut model = tiny_llama(tied);
        let chunks = vec![vec![1u32, 2, 3, 4, 5, 6]];
        let run = collect_imatrix(&mut model, &chunks, 1, |_, _| {}).expect("collect");
        assert!(
            run.unmeasured.is_empty(),
            "tied={tied}: unmeasured {:?}",
            run.unmeasured
        );
        let mut names: Vec<&str> = run.imatrix.data.keys().map(String::as_str).collect();
        names.sort_unstable();
        let mut want: Vec<String> = Vec::new();
        for layer in 0..2 {
            for t in [
                "attn_q",
                "attn_k",
                "attn_v",
                "attn_output",
                "ffn_gate",
                "ffn_up",
                "ffn_down",
            ] {
                want.push(format!("blk.{layer}.{t}.weight"));
            }
        }
        // The head: its own tensor, or the embedding's when tied.
        want.push(
            if tied {
                "token_embd.weight"
            } else {
                "output.weight"
            }
            .to_string(),
        );
        want.sort_unstable();
        assert_eq!(names, want.iter().map(String::as_str).collect::<Vec<_>>());
        for (name, values) in &run.imatrix.data {
            assert!(values.iter().all(|v| v.is_finite() && *v >= 0.0), "{name}");
            assert!(values.iter().any(|v| *v > 0.0), "{name} saw nothing");
        }
    }
}

#[test]
fn collecting_does_not_change_the_forward_and_leaves_nothing_behind() {
    let mut model = tiny_llama(false);
    let ids = Array::from_slice(&[3i32, 7, 7, 1, 12, 0], &[1, 6]);
    let read = |model: &mut DynamicModel| {
        let logits = model.forward(&ids, None).expect("forward");
        let n = logits.size();
        let mut logits = logits.as_dtype(Dtype::Float32.as_i32());
        logits.to_f32_vec(n).expect("logits")
    };
    let before = read(&mut model);
    let chunks = vec![vec![3u32, 7, 7, 1, 12, 0]];
    collect_imatrix(&mut model, &chunks, 1, |_, _| {}).expect("collect");
    assert!(!pmetal_runtime::activation_stats::is_collecting());
    let after = read(&mut model);
    assert_eq!(before, after);
}

#[test]
fn chunks_of_different_lengths_run_and_progress_is_reported() {
    let mut model = tiny_llama(false);
    let chunks = vec![vec![1u32, 2, 3, 4], vec![5, 6, 7, 8], vec![9, 10]];
    let mut seen = Vec::new();
    let run = collect_imatrix(&mut model, &chunks, 8, |done, total| {
        seen.push((done, total))
    })
    .expect("collect");
    assert_eq!(run.chunks, 3);
    assert_eq!(run.tokens, 10);
    // The two chunks of four batch together; the short one runs alone.
    assert_eq!(seen, vec![(2, 3), (3, 3)]);
    assert_eq!(run.imatrix.counts["blk.0.attn_q.weight"], vec![10.0]);
}
