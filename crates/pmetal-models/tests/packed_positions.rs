//! Two records packed into one row must come out the same as two runs.
//!
//! Sequence packing is on by default in training. Each test runs the same
//! model twice — once per record with a plain causal mask, once over the
//! concatenation with a block-diagonal mask and positions that restart at the
//! boundary — and asserts the two agree per record.
//!
//! ## Why the second assertion stretches rather than drops
//!
//! RoPE is *relative*: `⟨R_i q, R_j k⟩` depends only on `i - j`. Inside a
//! block the mask has already sealed off, a uniform shift of every position
//! therefore cancels, and running the second record at `3,4,5,6` instead of
//! `0,1,2,3` produces bit-identical logits. So "drop the positions and watch
//! it move" proves nothing here, and an architecture that ignored the
//! positions outright would sail through it.
//!
//! What the shift does reach is everything that reads the absolute position
//! rather than the difference: Llama 4 scales Q by
//! `log(floor(pos / floor_scale) + 1)` on its NoPE layers, Phi-3 picks its
//! short or long LongRoPE table by how far the forward reaches, and any
//! position past the trained window lands on extrapolated frequencies. Those
//! are the cases explicit positions exist for.
//!
//! To show the positions are genuinely consumed, the second assertion
//! *stretches* them (`0,2,4,…`), which changes the differences and so must
//! change the output of any architecture that rotates by them at all.

use pmetal_bridge::compat::{Array, Dtype, Exception};

use pmetal_models::architectures::cohere::{CohereConfig, CohereForCausalLM};
use pmetal_models::architectures::gemma::{GemmaConfig, GemmaForCausalLM};
use pmetal_models::architectures::gemma4::{Gemma4Config, Gemma4ForCausalLM};
use pmetal_models::architectures::gpt_oss::{GptOssConfig, GptOssForCausalLM};
use pmetal_models::architectures::granite::{GraniteConfig, GraniteForCausalLM};
use pmetal_models::architectures::llama::{LlamaConfig, LlamaForCausalLM};
use pmetal_models::architectures::mistral::{MistralConfig, MistralForCausalLM};
use pmetal_models::architectures::phi::{PhiConfig, PhiForCausalLM};
use pmetal_models::architectures::qwen2::{Qwen2Config, Qwen2ForCausalLM};
use pmetal_models::architectures::qwen3::{Qwen3Config, Qwen3ForCausalLM};

const FIRST: [i32; 3] = [5, 9, 2];
const SECOND: [i32; 4] = [7, 1, 4, 3];

/// Additive causal mask, matching `architectures::utils::create_causal_mask`.
fn causal_mask(seq_len: i32) -> Array {
    block_diagonal_causal_mask(&[seq_len])
}

/// Additive mask that is causal *within* each record and blocks every pair
/// that straddles a boundary.
fn block_diagonal_causal_mask(lengths: &[i32]) -> Array {
    let record: Vec<usize> = lengths
        .iter()
        .enumerate()
        .flat_map(|(index, len)| std::iter::repeat_n(index, *len as usize))
        .collect();
    let total = record.len();

    let mut values = Vec::with_capacity(total * total);
    for (query, query_record) in record.iter().enumerate() {
        for (key, key_record) in record.iter().enumerate() {
            let visible = query_record == key_record && key <= query;
            values.push(if visible { 0.0 } else { f32::NEG_INFINITY });
        }
    }
    Array::from_f32_slice(&values, &[total as i32, total as i32])
}

/// `0, 1, …` restarting at each record boundary.
fn packed_positions_values(lengths: &[i32]) -> Vec<i32> {
    lengths.iter().flat_map(|len| 0..*len).collect()
}

fn packed_positions(lengths: &[i32]) -> Array {
    let values = packed_positions_values(lengths);
    Array::from_i32_slice_shaped(&values, &[values.len() as i32])
}

fn ids(tokens: &[i32]) -> Array {
    Array::from_i32_slice_shaped(tokens, &[1, tokens.len() as i32])
}

fn rows(logits: &Array, start: i32, end: i32) -> Vec<f32> {
    let shape = logits.shape();
    let vocab = shape[2];
    let mut slice = logits.slice(&[0, start, 0], &[1, end, vocab]);
    slice.eval();
    let count = ((end - start) * vocab) as usize;
    slice.to_f32_vec(count).expect("to_f32_vec")
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .fold(0.0f32, |worst, (x, y)| worst.max((x - y).abs()))
}

/// Run the three forwards and make the two assertions. `forward` is the
/// architecture's `forward_with_positions`.
fn assert_packing_is_transparent<F>(mut forward: F)
where
    F: FnMut(&Array, Option<&Array>, Option<&Array>) -> Result<Array, Exception>,
{
    let lengths = [FIRST.len() as i32, SECOND.len() as i32];

    let separate: Vec<Vec<f32>> = [&FIRST[..], &SECOND[..]]
        .iter()
        .map(|tokens| {
            let length = tokens.len() as i32;
            let logits =
                forward(&ids(tokens), Some(&causal_mask(length)), None).expect("separate forward");
            rows(&logits, 0, length)
        })
        .collect();

    let packed: Vec<i32> = FIRST.iter().chain(SECOND.iter()).copied().collect();
    let packed_ids = ids(&packed);
    let mask = block_diagonal_causal_mask(&lengths);
    let positions = packed_positions(&lengths);

    let together = forward(&packed_ids, Some(&mask), Some(&positions)).expect("packed forward");

    // Everything below is relative to how large the logits actually are. A
    // fixed epsilon would pass trivially on a freshly-initialised model, whose
    // logits sit orders of magnitude below 1.
    let scale = separate
        .iter()
        .flatten()
        .fold(0.0f32, |worst, v| worst.max(v.abs()));
    assert!(
        scale > 1e-9,
        "logits are all but zero ({scale:e}); nothing below would mean anything"
    );

    for (index, (start, end)) in [(0, lengths[0]), (lengths[0], lengths[0] + lengths[1])]
        .into_iter()
        .enumerate()
    {
        let diff = max_abs_diff(&separate[index], &rows(&together, start, end));
        assert!(
            diff / scale < 1e-4,
            "record {index}: packed logits differ from the separate run by {diff:e} \
             (relative {:e})",
            diff / scale
        );
    }

    // Stretching the positions changes the *differences* between them, which
    // relative RoPE cannot absorb. Without this the assertion above would pass
    // on an architecture that took the positions and threw them away.
    let stretched = Array::from_i32_slice_shaped(
        &packed_positions_values(&lengths)
            .iter()
            .map(|p| p * 2)
            .collect::<Vec<_>>(),
        &[packed.len() as i32],
    );
    let stretched = forward(&packed_ids, Some(&mask), Some(&stretched)).expect("stretched forward");
    let diff = max_abs_diff(
        &rows(&together, 0, lengths[0]),
        &rows(&stretched, 0, lengths[0]),
    );
    assert!(
        diff / scale > 1e-3,
        "stretching the positions changed nothing (relative {:e}, scale {scale:e}), so this \
         architecture is not rotating by them and the assertion above proves nothing",
        diff / scale
    );
}

// ---------------------------------------------------------------------------
// Tiny configs. Sliding windows are left off so the separate runs see the same
// attention pattern the packed one does.
// ---------------------------------------------------------------------------

macro_rules! packing_case {
    ($name:ident, $build:expr) => {
        #[test]
        fn $name() {
            let mut model = $build;
            assert_packing_is_transparent(|tokens, mask, positions| {
                model.forward_with_positions(tokens, mask, positions)
            });
        }
    };
}

fn llama_config() -> LlamaConfig {
    LlamaConfig {
        vocab_size: 64,
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: Some(2),
        head_dim: Some(8),
        ..Default::default()
    }
}

packing_case!(llama, LlamaForCausalLM::new(llama_config()).unwrap());

packing_case!(
    llama3_frequency_band_scaling,
    LlamaForCausalLM::new(LlamaConfig {
        rope_scaling: Some(
            [
                ("rope_type", serde_json::Value::from("llama3")),
                ("factor", serde_json::Value::from(32.0)),
                ("low_freq_factor", serde_json::Value::from(1.0)),
                ("high_freq_factor", serde_json::Value::from(4.0)),
                (
                    "original_max_position_embeddings",
                    serde_json::Value::from(8192.0),
                ),
            ]
            .into_iter()
            .map(|(key, value)| (
                key.to_string(),
                serde_json::from_value(value).expect("rope_scaling value")
            ))
            .collect(),
        ),
        ..llama_config()
    })
    .unwrap()
);

packing_case!(
    qwen2,
    Qwen2ForCausalLM::new(Qwen2Config {
        vocab_size: 64,
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: Some(2),
        use_sliding_window: false,
        sliding_window: None,
        ..Default::default()
    })
    .unwrap()
);

packing_case!(
    qwen3,
    Qwen3ForCausalLM::new(Qwen3Config {
        vocab_size: 64,
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: Some(2),
        head_dim: 8,
        use_sliding_window: false,
        sliding_window: None,
        ..Default::default()
    })
    .unwrap()
);

packing_case!(
    mistral,
    MistralForCausalLM::new(MistralConfig {
        vocab_size: 64,
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: Some(2),
        head_dim: Some(8),
        sliding_window: None,
        ..Default::default()
    })
    .unwrap()
);

packing_case!(
    gemma,
    GemmaForCausalLM::new(GemmaConfig {
        vocab_size: 64,
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: Some(2),
        head_dim: Some(8),
        ..Default::default()
    })
    .unwrap()
);

packing_case!(
    phi,
    PhiForCausalLM::new(PhiConfig {
        vocab_size: 64,
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        ..Default::default()
    })
    .unwrap()
);

packing_case!(
    cohere,
    CohereForCausalLM::new(CohereConfig {
        vocab_size: 64,
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 8,
        use_sliding_window: false,
        ..Default::default()
    })
    .unwrap()
);

// `GraniteConfig` spells both of these `Option`.

packing_case!(
    granite,
    GraniteForCausalLM::new(GraniteConfig {
        vocab_size: 64,
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: Some(8),
        ..Default::default()
    })
    .unwrap()
);

packing_case!(
    gpt_oss,
    GptOssForCausalLM::new(GptOssConfig {
        vocab_size: 64,
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 8,
        ..Default::default()
    })
    .unwrap()
);

/// Gemma 4 has no `Default` config: its per-layer-type geometry has no
/// sensible blank. This is the same shape `gemma4_parity` uses, shrunk, with
/// a sliding window wide enough that both records see plain causal attention.
fn gemma4_config() -> Gemma4Config {
    json5::from_str(
        r#"{
            "model_type": "gemma4_text",
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "global_head_dim": 16,
            "num_global_key_value_heads": 1,
            "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-6,
            "attention_k_eq_v": true,
            "tie_word_embeddings": true,
            "sliding_window": 64,
            "final_logit_softcapping": 30.0,
            "layer_types": ["sliding_attention", "full_attention"],
            "num_kv_shared_layers": 0
        }"#,
    )
    .expect("gemma4 config parses")
}

packing_case!(gemma4, Gemma4ForCausalLM::new(gemma4_config()).unwrap());

/// The mask itself has to be right, or every case above is comparing two runs
/// of the same wrong thing.
#[test]
fn the_block_diagonal_mask_separates_the_records() {
    let mut mask = block_diagonal_causal_mask(&[2, 2]);
    mask.eval();
    let values = mask.to_f32_vec(16).expect("to_f32_vec");
    let visible: Vec<bool> = values.iter().map(|v| *v == 0.0).collect();
    assert_eq!(
        visible,
        vec![
            true, false, false, false, //
            true, true, false, false, //
            false, false, true, false, //
            false, false, true, true,
        ]
    );
}

#[test]
fn positions_restart_at_each_record() {
    let mut positions = packed_positions(&[3, 4]).as_dtype(Dtype::Float32.as_i32());
    positions.eval();
    assert_eq!(
        positions.to_f32_vec(7).expect("to_f32_vec"),
        vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 3.0]
    );
}
