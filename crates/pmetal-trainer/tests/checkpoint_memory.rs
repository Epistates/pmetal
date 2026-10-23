//! Gradient checkpointing has to save activation memory in a training step, for
//! a full fine-tune as for LoRA.
//!
//! A step's peak is the weights, the gradients and the optimizer's state (none
//! of which grow with the batch) plus the activations (which do). Two steps of
//! different lengths on one model differ only in the activations, so the
//! growth of the peak per token is the activation cost of a token. With
//! checkpointing the trunk keeps one hidden state per layer and recomputes the
//! rest, so that cost has to be a small fraction of what it is without.
//!
//! `cargo test --release -p pmetal-trainer --test checkpoint_memory -- --nocapture`

use pmetal_core::{LoraConfig, TrainingConfig};
use pmetal_data::{DataLoaderConfig, Sample, TrainingBatch, TrainingDataset};
use pmetal_lora::{AdaptedModel, FullFineTune, TrainableModel};
use pmetal_models::dispatcher::DynamicModel;
use pmetal_runtime::compat::optimizers::AdamW;
use pmetal_runtime::inline_array::{
    clear_cache, get_active_memory, get_peak_memory, reset_peak_memory,
};
use pmetal_trainer::{TrainingLoop, TrainingLoopConfig};

const LAYERS: usize = 8;
const HIDDEN: usize = 256;

/// A Qwen3-shaped trunk with a small vocabulary, so the logits do not hide
/// what the layers keep.
fn config_json() -> String {
    format!(
        r#"{{
            "model_type": "qwen3",
            "vocab_size": 512, "hidden_size": {HIDDEN}, "intermediate_size": 768,
            "num_hidden_layers": {LAYERS}, "num_attention_heads": 8,
            "num_key_value_heads": 4, "head_dim": 32,
            "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
            "tie_word_embeddings": true
        }}"#
    )
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Mode {
    Lora,
    Full,
}

fn model(mode: Mode) -> AdaptedModel {
    let base = DynamicModel::from_config(&config_json()).expect("builds");
    match mode {
        Mode::Lora => AdaptedModel::attach(
            base,
            LoraConfig {
                r: 8,
                alpha: 16.0,
                target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
                    .map(String::from)
                    .to_vec(),
                ..Default::default()
            },
        )
        .expect("adapters"),
        Mode::Full => AdaptedModel::full(base, FullFineTune::default()).expect("full"),
    }
}

fn batch(seq_len: usize) -> TrainingBatch {
    let samples = vec![Sample::new(
        (0..seq_len)
            .map(|j| ((j * 7 + 3) % 500 + 1) as u32)
            .collect(),
    )];
    let mut loader = pmetal_data::DataLoader::new(
        TrainingDataset::from_samples(samples),
        DataLoaderConfig {
            batch_size: 1,
            max_seq_len: seq_len,
            shuffle: false,
            pad_token_id: 0,
            ..Default::default()
        },
        None,
    );
    loader.next_batch().expect("a batch")
}

fn training_loop() -> TrainingLoop {
    TrainingLoop::new(TrainingLoopConfig {
        training: TrainingConfig {
            learning_rate: 1e-4,
            batch_size: 1,
            gradient_accumulation_steps: 1,
            max_grad_norm: 1.0,
            ..Default::default()
        },
        // The model carries the switch; the loop's own is the orchestrator's.
        gradient_checkpointing: false,
        use_sequence_packing: false,
        ..Default::default()
    })
    .expect("loop")
}

/// Bytes of peak memory one more token of a step costs, measured on `mode`'s
/// model with checkpointing on or off.
fn bytes_per_token(mode: Mode, checkpointing: bool) -> f64 {
    let mut model = model(mode);
    if checkpointing {
        assert!(model.supports_gradient_checkpointing());
        model.enable_gradient_checkpointing(1);
    }
    let mut lp = training_loop();
    let mut optimizer = AdamW::new(1e-4, 0.0);

    // The first step allocates the optimizer's state and compiles kernels.
    lp.train_step(&mut model, &batch(64), &mut optimizer)
        .expect("warm-up step");

    let (short, long) = (512, 1024);
    let mut peaks = Vec::new();
    for seq_len in [short, long] {
        let input = batch(seq_len);
        clear_cache();
        reset_peak_memory();
        let before = get_active_memory();
        lp.train_step(&mut model, &input, &mut optimizer)
            .expect("step");
        peaks.push(get_peak_memory().saturating_sub(before) as f64);
    }
    (peaks[1] - peaks[0]) / (long - short) as f64
}

#[test]
fn checkpointing_cuts_a_steps_activation_memory() {
    // As a training command starts: before MLX touches the GPU. Without the
    // limit the GPU's command queue holds on to the buffers a checkpoint frees
    // and a checkpointed step needs about what an uncheckpointed one does.
    pmetal_runtime::memory::bound_command_buffer_memory();

    let mut rows = Vec::new();
    for mode in [Mode::Lora, Mode::Full] {
        let off = bytes_per_token(mode, false);
        let on = bytes_per_token(mode, true);
        eprintln!(
            "{mode:?}: {:.1} KB/token without checkpointing, {:.1} KB/token with ({:.2}x)",
            off / 1e3,
            on / 1e3,
            off / on
        );
        rows.push((mode, off, on));
    }
    for (mode, off, on) in rows {
        // Without it every layer's activations stay; with it one hidden state
        // a layer does. The ratio grows with depth; measured at 8x (LoRA) and
        // 11x (full) over these 8 layers, before the fix at 1.3x and 1.0x.
        assert!(
            off > 3.0 * on,
            "{mode:?}: checkpointing saved too little: {:.1} KB/token against {:.1}",
            on / 1e3,
            off / 1e3
        );
    }
}
