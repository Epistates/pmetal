//! Pretraining over documents packed into rows: the block-diagonal mask, data
//! errors that stop a run, a configuration check, and a checkpoint that holds
//! what a resumed bf16 run needs.

use std::rc::Rc;

use pmetal_models::architectures::llama::{LlamaConfig, LlamaForCausalLM};
use pmetal_runtime::compat::optimizers::{AdamW, Optimizer};
use pmetal_runtime::compat::{Array, Dtype, FlattenedModuleParam, Param, random};
use pmetal_trainer::pretrain::{
    CausalLm, CheckpointMeta, PretrainBatch, PretrainConfig, PretrainError, document_mask,
    load_checkpoint, pretrain_step_batches, run_pretrain_with_state, save_checkpoint,
};
use serial_test::serial;

fn tiny_llama() -> LlamaForCausalLM {
    LlamaForCausalLM::new(LlamaConfig {
        vocab_size: 64,
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: Some(2),
        max_position_embeddings: 64,
        ..Default::default()
    })
    .expect("llama builds")
}

fn values(a: &Array) -> Vec<f32> {
    let a = a.as_dtype(Dtype::Float32.as_i32());
    a.eval();
    pmetal_runtime::check_unobserved_error().expect("a runtime op threw");
    a.as_slice::<f32>().to_vec()
}

#[test]
fn the_document_mask_is_block_diagonal_and_causal() {
    let mask = document_mask(&[vec![2, 3]], 5).unwrap();
    assert_eq!(mask.shape(), [1, 1, 5, 5]);
    let m = values(&mask);
    let allowed = |q: usize, k: usize| m[q * 5 + k] == 0.0;
    for q in 0..5 {
        for k in 0..5 {
            let same_doc = (q < 2) == (k < 2);
            assert_eq!(allowed(q, k), same_doc && k <= q, "({q}, {k})");
            if !allowed(q, k) {
                assert_eq!(m[q * 5 + k], f32::NEG_INFINITY);
            }
        }
    }
    // Documents that do not fill the row are an error, not a mask with holes.
    assert!(document_mask(&[vec![2, 2]], 5).is_err());
}

/// What the mask is for: with it, the second document's logits are the ones it
/// has alone (RoPE is relative, so where it starts in the row does not matter);
/// without it, the first document leaks into them.
#[test]
#[serial]
fn a_masked_row_gives_each_document_the_logits_it_has_alone() {
    random::seed(3);
    let mut model = tiny_llama();
    let first = [5, 9, 13, 21];
    let second = [7, 11, 2, 30];
    let packed: Vec<i32> = first.iter().chain(&second).copied().collect();
    let row = Array::from_i32_slice_shaped(&packed, &[1, 8]);
    let alone = Array::from_i32_slice_shaped(&second, &[1, 4]);

    let mask = document_mask(&[vec![4, 4]], 8).unwrap();
    let masked = values(&model.forward_logits_masked(&row, Some(&mask)).unwrap());
    let unmasked = values(&model.forward_logits(&row).unwrap());
    let solo = values(&model.forward_logits(&alone).unwrap());
    let vocab = 64;
    let tail = |logits: &[f32]| logits[4 * vocab..].to_vec();
    let worst = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };
    let masked_off = worst(&tail(&masked), &solo);
    let unmasked_off = worst(&tail(&unmasked), &solo);
    assert!(
        masked_off < 1e-3,
        "masked second document is {masked_off} from its solo logits"
    );
    assert!(
        unmasked_off > 10.0 * masked_off.max(1e-5),
        "the first document does not leak without the mask ({unmasked_off})"
    );
}

/// A masked step trains: the loss is finite, and the mask is applied (the loss
/// differs from the unmasked row's).
#[test]
#[serial]
fn a_document_masked_step_trains() {
    random::seed(4);
    let tokens = random::randint(0, 64, &[2, 8], Dtype::Int32, None).unwrap();
    let step = |docs: Option<Vec<Vec<usize>>>| {
        random::seed(4);
        let mut model = tiny_llama();
        let mut optimizer = AdamW::new(1e-3, 0.0);
        optimizer.advance_step();
        let batch = PretrainBatch {
            tokens: tokens.clone(),
            position: None,
            doc_lengths: docs,
        };
        pretrain_step_batches(&mut model, &mut optimizer, &[batch], None, None, Some(1.0)).unwrap()
    };
    let masked = step(Some(vec![vec![4, 4], vec![3, 5]]));
    let plain = step(None);
    assert!(masked.is_finite() && plain.is_finite());
    assert!(
        (masked - plain).abs() > 1e-6,
        "the mask changed nothing ({masked} vs {plain})"
    );
}

/// A shard that cannot be read ends the run with that error, not as the end of
/// the data.
#[test]
#[serial]
fn a_data_error_stops_the_run() {
    let mut model = tiny_llama();
    let config = PretrainConfig {
        num_steps: 5,
        warmup_steps: 0,
        ..PretrainConfig::default()
    };
    let tokens = random::randint(0, 64, &[2, 8], Dtype::Int32, None).unwrap();
    let mut served = 0;
    let batches = std::iter::from_fn(move || {
        served += 1;
        Some(if served < 3 {
            Ok(PretrainBatch::new(tokens.clone()))
        } else {
            Err(PretrainError::Data("shard 0 is truncated".into()))
        })
    });
    let err = run_pretrain_with_state(
        &mut model,
        &config,
        batches,
        AdamW::new(1e-3, 0.0),
        0,
        Option::<std::iter::Empty<Array>>::None,
        &mut pmetal_trainer::CompositeCallback::new(),
    )
    .expect_err("the data error must surface");
    assert!(err.to_string().contains("truncated"), "{err}");
}

/// What a callback sees of a pretraining run: it starts, each step reports its
/// metrics (tokens included), each checkpoint written is announced, it ends;
/// and a callback that asks to stop ends the run after the step in flight.
#[test]
#[serial]
fn callbacks_follow_pretraining_and_can_stop_it() {
    use pmetal_core::{StepMetrics, TrainingCallback};
    use std::sync::{Arc, Mutex};

    struct Recorder {
        log: Arc<Mutex<Vec<String>>>,
        stop_after: usize,
    }
    impl TrainingCallback for Recorder {
        fn on_train_start(&mut self) {
            self.log.lock().unwrap().push("start".into());
        }
        fn on_step_end_with_metrics(&mut self, m: &StepMetrics) {
            assert!(m.loss.is_finite() && m.lr > 0.0 && m.total_steps == 20);
            self.log
                .lock()
                .unwrap()
                .push(format!("step {} tokens {}", m.step, m.tokens));
        }
        fn on_save(&mut self, _path: &std::path::Path) {
            self.log.lock().unwrap().push("save".into());
        }
        fn on_train_end(&mut self) {
            self.log.lock().unwrap().push("end".into());
        }
        fn should_stop(&self) -> bool {
            self.log
                .lock()
                .unwrap()
                .iter()
                .filter(|l| l.starts_with("step"))
                .count()
                >= self.stop_after
        }
    }

    random::seed(5);
    let mut model = tiny_llama();
    let tokens = random::randint(0, 64, &[2, 8], Dtype::Int32, None).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let config = PretrainConfig {
        num_steps: 20,
        learning_rate: 1e-3,
        warmup_steps: 0,
        max_grad_norm: None,
        checkpoint_dir: Some(dir.path().to_path_buf()),
        checkpoint_every: Some(100),
        log_every: 0,
        ..PretrainConfig::default()
    };
    let log = Arc::new(Mutex::new(Vec::new()));
    let mut recorder = Recorder {
        log: Arc::clone(&log),
        stop_after: 3,
    };

    let losses = run_pretrain_with_state(
        &mut model,
        &config,
        std::iter::repeat_with(move || Ok(PretrainBatch::new(tokens.clone()))),
        AdamW::new(1e-3, 0.0),
        0,
        Option::<std::iter::Empty<Array>>::None,
        &mut recorder,
    )
    .unwrap();

    assert_eq!(losses.len(), 3, "the callback stopped the run");
    assert_eq!(
        *log.lock().unwrap(),
        [
            "start",
            "step 1 tokens 16",
            "step 2 tokens 16",
            "step 3 tokens 16",
            "save",
            "end"
        ]
    );
    assert!(dir.path().join("final").join("model.safetensors").exists());
}

#[test]
fn a_pretrain_config_that_cannot_work_is_refused_before_the_run() {
    let refused = |config: PretrainConfig| config.validate().unwrap_err().to_string();
    let base = PretrainConfig::default;
    assert!(
        refused(PretrainConfig {
            num_steps: 0,
            ..base()
        })
        .contains("num_steps")
    );
    assert!(
        refused(PretrainConfig {
            learning_rate: -1.0,
            ..base()
        })
        .contains("learning_rate")
    );
    assert!(
        refused(PretrainConfig {
            min_lr: 1.0,
            learning_rate: 1e-3,
            ..base()
        })
        .contains("min_lr")
    );
    assert!(
        refused(PretrainConfig {
            betas: (0.9, 1.0),
            ..base()
        })
        .contains("betas.1")
    );
    assert!(
        refused(PretrainConfig {
            gradient_accumulation_steps: 0,
            ..base()
        })
        .contains("accumulation")
    );
    base().validate().unwrap();

    // The loop checks, whoever builds the config.
    let mut model = tiny_llama();
    let result = run_pretrain_with_state(
        &mut model,
        &PretrainConfig {
            num_steps: 0,
            ..base()
        },
        std::iter::empty::<Result<PretrainBatch, PretrainError>>(),
        AdamW::new(1e-3, 0.0),
        0,
        Option::<std::iter::Empty<Array>>::None,
        &mut pmetal_trainer::CompositeCallback::new(),
    );
    assert!(matches!(result, Err(PretrainError::Config(_))));
}

/// A bf16 weight's f32 master copy is part of the checkpoint: updates far
/// below bf16's resolution accumulate in it, and a resume that lost it would
/// start again from the rounded weight and drop them.
struct Weight {
    w: Param<Array>,
}
pmetal_runtime::impl_module_params!(Weight; w);

#[test]
fn a_pretrain_checkpoint_keeps_the_masters_of_half_precision_weights() {
    random::seed(8);
    let bf16 = Dtype::Bfloat16.as_i32();
    let init = random::uniform_range(-1.0, 1.0, &[8, 16], Dtype::Float32).as_dtype(bf16);
    let grads: Vec<Array> = (0..6)
        .map(|_| random::normal(&[8, 16], Dtype::Float32).mul_scalar(0.05))
        .collect();
    let step = |opt: &mut AdamW, model: &mut Weight, i: usize| {
        use pmetal_runtime::compat::ModuleParametersExt;
        opt.advance_step();
        let g: FlattenedModuleParam = [(Rc::from("w"), grads[i].clone())].into();
        opt.update(model, g).unwrap();
        let params = model.flatten_params();
        pmetal_runtime::compat::transforms::eval(params.values().collect::<Vec<_>>()).unwrap();
    };

    let mut whole_model = Weight {
        w: Param::new(init.clone()),
    };
    let mut whole = AdamW::new(1e-3, 0.01);
    for i in 0..6 {
        step(&mut whole, &mut whole_model, i);
    }

    let mut first_model = Weight {
        w: Param::new(init),
    };
    let mut first = AdamW::new(1e-3, 0.01);
    for i in 0..3 {
        step(&mut first, &mut first_model, i);
    }
    let dir = tempfile::tempdir().unwrap();
    save_checkpoint(
        dir.path(),
        &first_model,
        &first,
        &CheckpointMeta {
            step: 3,
            loss: 0.0,
            learning_rate: 1e-3,
            stream_position: None,
        },
    )
    .unwrap();

    let mut resumed_model = Weight {
        w: Param::new(Array::zeros(&[8, 16], bf16)),
    };
    let mut resumed = AdamW::new(1e-3, 0.01);
    let meta = load_checkpoint(dir.path(), &mut resumed_model, &mut resumed).unwrap();
    assert_eq!(meta.step, 3);
    for i in 3..6 {
        step(&mut resumed, &mut resumed_model, i);
    }
    assert_eq!(
        values(&resumed_model.w),
        values(&whole_model.w),
        "the resumed bf16 run ended somewhere else: its masters were lost"
    );
}
