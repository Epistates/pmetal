//! Each update rule against its paper's formula, worked by hand.

use super::*;
use pmetal_runtime::compat::optimizers::AdamWBuilder;
use pmetal_runtime::expect_runtime_errors;

fn arr(v: &[f32], shape: &[i32]) -> Array {
    Array::from_slice(v, shape)
}

fn values(a: &Array) -> Vec<f32> {
    a.eval();
    pmetal_runtime::check_unobserved_error().expect("bridge op failed");
    a.as_slice::<f32>().to_vec()
}

fn key(name: &str) -> Rc<str> {
    Rc::from(name)
}

fn assert_close(got: &[f32], want: &[f64], tol: f64) {
    assert_eq!(got.len(), want.len(), "length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (*g as f64 - w).abs() <= tol,
            "element {i}: got {g}, want {w} (all: {got:?} vs {want:?})"
        );
    }
}

/// One optimizer update of one parameter.
fn step(opt: &mut TrainOptimizer, name: &str, g: &Array, p: &mut Array) {
    opt.advance_step();
    opt.update_single(&key(name), g, p).unwrap();
}

#[test]
fn adamw_first_step_moves_each_weight_by_the_learning_rate() {
    let mut opt = TrainOptimizer::new(OptimizerType::AdamW, 0.1, 0.0);
    let mut p = arr(&[1.0, 1.0], &[2]);
    step(&mut opt, "w", &arr(&[0.5, -2.0], &[2]), &mut p);
    // m̂/√v̂ = g/|g| on the first step.
    assert_close(&values(&p), &[0.9, 1.1], 1e-6);
}

#[test]
fn sgd_accumulates_momentum_and_adds_l2_decay_to_the_gradient() {
    let mut opt = TrainOptimizer::new(OptimizerType::Sgd, 0.1, 0.5);
    let p0 = [1.0f64, -2.0];
    let g = [1.0f64, -2.0];
    let mut p = arr(&[1.0, -2.0], &[2]);
    let ga = arr(&[1.0, -2.0], &[2]);
    step(&mut opt, "w", &ga, &mut p);
    // buf₁ = g + λp₀, p₁ = p₀ − lr·buf₁
    let buf1: Vec<f64> = (0..2).map(|i| g[i] + 0.5 * p0[i]).collect();
    let p1: Vec<f64> = (0..2).map(|i| p0[i] - 0.1 * buf1[i]).collect();
    assert_close(&values(&p), &p1, 1e-6);
    step(&mut opt, "w", &ga, &mut p);
    // buf₂ = 0.9·buf₁ + g + λp₁
    let p2: Vec<f64> = (0..2)
        .map(|i| p1[i] - 0.1 * (0.9 * buf1[i] + g[i] + 0.5 * p1[i]))
        .collect();
    assert_close(&values(&p), &p2, 1e-6);
}

#[test]
fn lion_steps_by_the_sign_of_the_interpolated_momentum() {
    let mut opt = TrainOptimizer::new(OptimizerType::Lion, 0.1, 0.5);
    let mut p = arr(&[1.0, -2.0, 3.0], &[3]);
    step(&mut opt, "w", &arr(&[0.3, -0.1, 0.0], &[3]), &mut p);
    // c = 0.1·g → sign [1, −1, 0]; p₁ = p₀ − lr·(sign(c) + λ·p₀)
    let p1 = [
        1.0 - 0.1 * (1.0 + 0.5 * 1.0),
        -2.0 - 0.1 * (-1.0 + 0.5 * -2.0),
        3.0 - 0.1 * (0.0 + 0.5 * 3.0),
    ];
    assert_close(&values(&p), &p1, 1e-6);
    // m₁ = 0.01·g₁ = [0.003, −0.001, 0]. With g₂ = [−0.01, 0, 0.2],
    // c = 0.9·m₁ + 0.1·g₂ = [0.0017, −0.0009, 0.02]: the first weight still
    // moves down even though its new gradient points the other way.
    step(&mut opt, "w", &arr(&[-0.01, 0.0, 0.2], &[3]), &mut p);
    let sign = [1.0, -1.0, 1.0];
    let p2: Vec<f64> = (0..3)
        .map(|i| p1[i] - 0.1 * (sign[i] + 0.5 * p1[i]))
        .collect();
    assert_close(&values(&p), &p2, 1e-6);
}

/// Adafactor's update for a `rows × cols` matrix, in f64, from the paper.
#[expect(
    clippy::too_many_arguments,
    reason = "the paper's update, spelled out over scalars"
)]
fn adafactor_matrix_reference(
    p: &mut [f64],
    row: &mut [f64],
    col: &mut [f64],
    g: &[f64],
    rows: usize,
    cols: usize,
    t: f64,
    lr: f64,
    wd: f64,
) {
    let beta2t = 1.0 - t.powf(-0.8);
    let sq: Vec<f64> = g.iter().map(|x| x * x + 1e-30).collect();
    for r in 0..rows {
        let mean = (0..cols).map(|c| sq[r * cols + c]).sum::<f64>() / cols as f64;
        row[r] = beta2t * row[r] + (1.0 - beta2t) * mean;
    }
    for c in 0..cols {
        let mean = (0..rows).map(|r| sq[r * cols + c]).sum::<f64>() / rows as f64;
        col[c] = beta2t * col[c] + (1.0 - beta2t) * mean;
    }
    let row_mean = row.iter().sum::<f64>() / rows as f64;
    let mut u = vec![0.0; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let v_hat = row[r] * col[c] / row_mean;
            u[r * cols + c] = g[r * cols + c] / v_hat.sqrt();
        }
    }
    let rms = (u.iter().map(|x| x * x).sum::<f64>() / u.len() as f64).sqrt();
    let scale = (rms / 1.0).max(1.0);
    for i in 0..p.len() {
        p[i] = p[i] - wd * lr * p[i] - lr * u[i] / scale;
    }
}

#[test]
fn adafactor_factors_a_matrix_into_row_and_column_moments() {
    let (rows, cols) = (2usize, 3usize);
    let g1 = [0.5f64, -1.0, 2.0, 0.1, 0.3, -0.7];
    let g2 = [-0.2f64, 0.4, 1.0, 0.9, -0.3, 0.05];
    let p0 = [1.0f64, 2.0, -1.0, 0.5, -0.5, 0.25];
    let to32 = |v: &[f64]| v.iter().map(|&x| x as f32).collect::<Vec<_>>();

    let mut opt = TrainOptimizer::new(OptimizerType::Adafactor, 0.01, 0.1);
    let mut p = arr(&to32(&p0), &[2, 3]);
    let mut ref_p = p0.to_vec();
    let (mut ref_row, mut ref_col) = (vec![0.0; rows], vec![0.0; cols]);

    for (t, g) in [(1.0, &g1), (2.0, &g2)] {
        step(&mut opt, "w", &arr(&to32(g), &[2, 3]), &mut p);
        adafactor_matrix_reference(
            &mut ref_p,
            &mut ref_row,
            &mut ref_col,
            g,
            rows,
            cols,
            t,
            0.01,
            0.1,
        );
        assert_close(&values(&p), &ref_p, 1e-6);
    }

    let TrainOptimizer::Adafactor(inner) = &opt else {
        unreachable!()
    };
    match inner.state.get(&key("w")) {
        Some(SecondMoment::Factored { row, col }) => {
            assert_eq!(row.shape(), &[2]);
            assert_eq!(col.shape(), &[3]);
            assert_close(&values(row), &ref_row, 1e-6);
            assert_close(&values(col), &ref_col, 1e-6);
        }
        other => panic!("expected a factored moment, got {other:?}"),
    }
}

#[test]
fn adafactor_keeps_the_full_moment_for_a_vector_and_clips_the_update() {
    // One step: β̂₂₁ = 0, so v = g² + ε₁ and u = g/|g| = ±1, RMS 1 → no clip.
    let mut opt = TrainOptimizer::new(OptimizerType::Adafactor, 0.01, 0.0);
    let mut p = arr(&[1.0, 1.0], &[2]);
    step(&mut opt, "b", &arr(&[3.0, -0.5], &[2]), &mut p);
    assert_close(&values(&p), &[0.99, 1.01], 1e-6);
    // Second step: v = 2^-0.8·g₁² + (1−2^-0.8)·g₂². A gradient 10x the first
    // makes |u| > 1; the update is scaled back to RMS 1.
    let g1 = [3.0f64, -0.5];
    let g2 = [30.0f64, -5.0];
    let b = 1.0 - 2f64.powf(-0.8);
    let u: Vec<f64> = (0..2)
        .map(|i| g2[i] / (b * g1[i] * g1[i] + (1.0 - b) * g2[i] * g2[i]).sqrt())
        .collect();
    let rms = (u.iter().map(|x| x * x).sum::<f64>() / 2.0).sqrt();
    assert!(rms > 1.0, "the case must exercise the clip");
    step(&mut opt, "b", &arr(&[30.0, -5.0], &[2]), &mut p);
    let want: Vec<f64> = [0.99, 1.01]
        .iter()
        .zip(&u)
        .map(|(p, u)| p - 0.01 * u / rms)
        .collect();
    assert_close(&values(&p), &want, 1e-6);
}

#[test]
fn a_zero_learning_rate_leaves_every_kind_where_it_was() {
    for kind in OptimizerType::ALL {
        let mut opt = TrainOptimizer::new(kind, 0.5, 0.1);
        opt.set_lr(0.0);
        assert_eq!(opt.lr(), 0.0);
        let mut p = arr(&[1.0, -2.0, 3.0, 4.0], &[2, 2]);
        step(&mut opt, "w", &arr(&[0.1, 0.2, -0.3, 0.4], &[2, 2]), &mut p);
        assert_close(&values(&p), &[1.0, -2.0, 3.0, 4.0], 0.0);
        assert_eq!(opt.kind(), kind);
    }
}

#[test]
fn every_group_follows_the_scheduled_rate_at_its_ratio() {
    // Lion moves each weight by exactly its learning rate, which makes the
    // rate each group actually used visible in the parameters.
    let mut opt = ParamGroupOptimizerBuilder::new(OptimizerType::Lion, 0.1)
        .with_weight_decay(0.5)
        .with_embedding_lr(0.05)
        .with_loraplus_lr_ratio(4.0)
        .build();
    opt.set_learning_rate(0.02);
    assert_eq!(opt.learning_rates(), (0.02, 0.01));
    assert_eq!(opt.kind(), OptimizerType::Lion);
    for g in opt.optimizers_mut() {
        g.advance_step();
    }

    let g = arr(&[1.0], &[1]);
    let moved = |opt: &mut ParamGroupOptimizer, name: &str| {
        let mut p = arr(&[0.0], &[1]);
        opt.update_single(&key(name), &g, &mut p).unwrap();
        -values(&p)[0]
    };
    // p = 0, so weight decay does not enter these.
    assert!((moved(&mut opt, "layers.0.q_proj.lora_a") - 0.02).abs() < 1e-7);
    assert!((moved(&mut opt, "layers.0.q_proj.lora_b") - 0.08).abs() < 1e-7);
    assert!((moved(&mut opt, "model.embed_tokens.weight") - 0.01).abs() < 1e-7);

    // A norm weight at 1.0 moves by lr·sign only: no decay term.
    let mut p = arr(&[1.0], &[1]);
    opt.update_single(&key("layers.0.input_layernorm.weight"), &g, &mut p)
        .unwrap();
    assert!((values(&p)[0] - 0.98).abs() < 1e-7);
    // A regular weight at 1.0 also decays: 1 − 0.02·(1 + 0.5).
    let mut p = arr(&[1.0], &[1]);
    opt.update_single(&key("layers.0.q_proj.lora_a"), &g, &mut p)
        .unwrap();
    assert!((values(&p)[0] - 0.97).abs() < 1e-7);
}

#[test]
fn plain_adamw_trains_embeddings_at_the_rate_it_was_given() {
    // The AdamW every non-grouped trainer builds once routed names with
    // "embed" or "lm_head" to a hidden fixed 5e-5, whatever the schedule.
    let mut opt = AdamWBuilder::new(0.1).weight_decay(0.0).build().unwrap();
    opt.advance_step();
    let mut p = arr(&[1.0], &[1]);
    opt.update_single(
        &key("model.embed_tokens.weight"),
        &arr(&[1.0], &[1]),
        &mut p,
    )
    .unwrap();
    assert!((values(&p)[0] - 0.9).abs() < 1e-6);
    opt.set_lr(0.0);
    assert_eq!(opt.lr(), 0.0);
}

/// One weight, for driving [`Optimizer::update`] through a model.
struct OneWeight {
    w: pmetal_runtime::compat::Param<Array>,
}
pmetal_runtime::impl_module_params!(OneWeight; w);

fn one_weight() -> OneWeight {
    OneWeight {
        w: pmetal_runtime::compat::Param::new(arr(&[1.0, 1.0], &[2])),
    }
}

/// A [3] gradient for the [2] weight: no update rule's arithmetic broadcasts.
fn misfit_gradient() -> FlattenedModuleParam {
    [(key("w"), arr(&[1.0, 1.0, 1.0], &[3]))].into()
}

#[test]
fn an_update_that_fails_on_the_gpu_stops_the_step_and_names_the_parameter() {
    // The failed op hands back a zero placeholder, so the update used to
    // finish, leave the weight where it was, and return Ok.
    for kind in OptimizerType::ALL {
        let mut opt = TrainOptimizer::new(kind, 0.1, 0.0);
        let err = expect_runtime_errors(|| opt.update(&mut one_weight(), misfit_gradient()))
            .expect_err("a gradient that doesn't fit its weight must fail the update");
        assert!(err.what().contains("`w`"), "{kind:?}: {}", err.what());
        // The failure was reported; nothing is left pending for the next kind.
        let _ = pmetal_runtime::check_unobserved_error();
    }
}

#[test]
fn the_plain_adamw_update_reports_a_failed_parameter_too() {
    let mut model = one_weight();
    let mut opt = AdamWBuilder::new(0.1).weight_decay(0.0).build().unwrap();
    assert!(expect_runtime_errors(|| opt.update(&mut model, misfit_gradient())).is_err());
    let _ = pmetal_runtime::check_unobserved_error();

    // A gradient that fits still trains.
    let mut model = one_weight();
    let mut opt = AdamWBuilder::new(0.1).weight_decay(0.0).build().unwrap();
    let grads: FlattenedModuleParam = [(key("w"), arr(&[1.0, -1.0], &[2]))].into();
    opt.update(&mut model, grads).unwrap();
    assert_close(&values(&model.w), &[0.9, 1.1], 1e-5);
    pmetal_runtime::check_unobserved_error().expect("no bridge op failed");
}

/// A small regression whose parameters land in every group: a hidden matrix,
/// an embedding, a norm weight and a LoRA B matrix.
struct Net {
    q_proj: pmetal_runtime::compat::Param<Array>,
    lora_b: pmetal_runtime::compat::Param<Array>,
    embed_tokens: pmetal_runtime::compat::Param<Array>,
    norm: pmetal_runtime::compat::Param<Array>,
}
pmetal_runtime::impl_module_params!(Net; q_proj, lora_b, embed_tokens, norm);

fn net() -> Net {
    use pmetal_runtime::compat::{Dtype, Param, random};
    random::seed(401);
    let f32_ = Dtype::Float32;
    Net {
        // 8192 elements: above the 8-bit optimizer's quantization threshold.
        q_proj: Param::new(random::normal(&[64, 128], f32_).mul_scalar(0.1)),
        lora_b: Param::new(random::normal(&[16, 8], f32_).mul_scalar(0.1)),
        embed_tokens: Param::new(random::normal(&[8, 16], f32_).mul_scalar(0.1)),
        norm: Param::new(Array::ones(&[128], f32_.as_i32())),
    }
}

/// The loss, and the gradients of every parameter: fit `x @ q^T` to a fixed
/// target through the norm weight, with the other two pulled toward zero.
fn loss_and_grads(net: &mut Net) -> (Array, FlattenedModuleParam) {
    use pmetal_runtime::compat::{Dtype, nn, random};
    random::seed(402);
    let x = random::normal(&[16, 128], Dtype::Float32);
    let target = random::normal(&[16, 64], Dtype::Float32);
    let loss_fn = |net: &mut Net, (x, t): (&Array, &Array)| {
        let pred = x.multiply(&net.norm).matmul(&net.q_proj.t());
        let fit = pred.subtract(t).square().mean_all();
        let pull = net
            .embed_tokens
            .square()
            .mean_all()
            .add(&net.lora_b.square().mean_all());
        Ok::<_, pmetal_runtime::compat::Exception>(fit.add(&pull))
    };
    nn::value_and_grad(loss_fn)(net, (&x, &target)).unwrap()
}

fn loss_value(net: &mut Net) -> f32 {
    let (loss, _) = loss_and_grads(net);
    values(&loss)[0]
}

/// Every optimizer a config can name trains a model: the loss falls, each
/// group's parameters move, and nothing is NaN. (A silent zero gradient or an
/// update that never reaches the weights fails here.)
#[test]
fn every_optimizer_kind_trains_every_group() {
    use pmetal_runtime::compat::ModuleParametersExt;
    let rates = |kind| match kind {
        OptimizerType::Sgd => 0.05,
        OptimizerType::Lion => 2e-3,
        OptimizerType::Adafactor => 5e-3,
        OptimizerType::Muon => 2e-2,
        OptimizerType::AdamWScheduleFree => 2e-2,
        _ => 5e-3,
    };
    for kind in OptimizerType::ALL {
        let mut model = net();
        let before = loss_value(&mut model);
        let start = model.flatten_params();
        let start: Vec<(String, Vec<f32>)> = start
            .iter()
            .map(|(k, v)| (k.to_string(), values(v)))
            .collect();
        let mut opt = ParamGroupOptimizerBuilder::new(kind, rates(kind))
            .with_weight_decay(0.0)
            .with_loraplus_lr_ratio(2.0)
            .build();
        for _ in 0..40 {
            let (loss, grads) = loss_and_grads(&mut model);
            loss.eval();
            opt.update(&mut model, grads).unwrap();
            let params = model.flatten_params();
            let arrays: Vec<&Array> = params.values().collect();
            pmetal_runtime::compat::transforms::eval(arrays).unwrap();
        }
        // Evaluate at the weights a run would save.
        opt.eval_weights(&mut model);
        let after = loss_value(&mut model);
        assert!(after.is_finite(), "{kind}: loss {after}");
        assert!(after < 0.8 * before, "{kind}: loss {before} -> {after}");
        let end = model.flatten_params();
        for (name, was) in &start {
            let now = values(&end[name.as_str()]);
            assert!(
                now.iter().all(|v| v.is_finite()),
                "{kind}: {name} is finite"
            );
            assert!(
                now.iter().zip(was).any(|(a, b)| (a - b).abs() > 1e-6),
                "{kind}: {name} did not move"
            );
        }
    }
}

/// Muon trains the hidden matrices only: the embedding and the norm weight
/// (and the output head, wherever the model puts it) train with AdamW.
#[test]
fn muon_trains_hidden_matrices_and_adamw_the_rest() {
    let mut model = net();
    let mut opt = ParamGroupOptimizerBuilder::new(OptimizerType::Muon, 1e-2)
        .with_loraplus_lr_ratio(2.0)
        .build();
    let (_, grads) = loss_and_grads(&mut model);
    opt.update(&mut model, grads).unwrap();
    let state = opt.export_state();
    let held = |prefix: &str, name: &str| {
        state
            .tensors
            .keys()
            .any(|k| k.starts_with(prefix) && k.ends_with(&format!("/{name}")))
    };
    // Hidden matrices (the regular group's and LoRA+'s) hold a Muon buffer.
    assert!(held("regular:muon:buf", "q_proj"));
    assert!(held("lora_b:muon:buf", "lora_b"));
    // The embedding group and the norm weight run plain AdamW.
    assert!(held("embedding:m", "embed_tokens"));
    assert!(held("no_decay:m", "norm"));
    assert_eq!(opt.kind(), OptimizerType::Muon);
}

/// A checkpoint of the whole grouped optimizer: stopping at step 4, saving
/// its state, rebuilding it and loading the state continues bit for bit what
/// six uninterrupted steps do, for every kind and with every group.
#[test]
fn every_kind_resumes_bit_exactly_through_its_groups() {
    use pmetal_runtime::compat::{ModuleParametersExt, transforms};
    let make = |kind| {
        ParamGroupOptimizerBuilder::new(kind, 5e-3)
            .with_weight_decay(0.01)
            .with_embedding_lr(1e-3)
            .with_loraplus_lr_ratio(2.0)
            .build()
    };
    let snapshot = |model: &Net| -> Vec<(String, Vec<f32>)> {
        model
            .flatten_params()
            .iter()
            .map(|(k, a)| (k.to_string(), values(a)))
            .collect()
    };
    let steps = |opt: &mut ParamGroupOptimizer, model: &mut Net, n: usize| {
        for _ in 0..n {
            let (_, grads) = loss_and_grads(model);
            opt.update(model, grads).unwrap();
            let params = model.flatten_params();
            transforms::eval(params.values().collect::<Vec<_>>()).unwrap();
        }
    };
    for kind in OptimizerType::ALL {
        let mut whole_model = net();
        let mut whole = make(kind);
        steps(&mut whole, &mut whole_model, 6);

        let mut first_model = net();
        let mut first = make(kind);
        steps(&mut first, &mut first_model, 4);
        // What a checkpoint holds: the evaluation weights and the state.
        first.eval_weights(&mut first_model);
        let state = first.export_state();
        let saved = snapshot(&first_model);

        let mut resumed_model = net();
        {
            let mut flat = resumed_model.flatten_params_mut();
            for (name, values) in &saved {
                let shape = flat[name.as_str()].shape().to_vec();
                **flat.get_mut(name.as_str()).unwrap() = Array::from_slice(values, &shape);
            }
        }
        let mut resumed = make(kind);
        resumed.import_state(&state).unwrap();
        resumed.train_weights(&mut resumed_model);
        steps(&mut resumed, &mut resumed_model, 2);

        // Whole's weights are the training weights; so are resumed's.
        assert_eq!(
            snapshot(&resumed_model),
            snapshot(&whole_model),
            "{kind}: the resumed run diverged"
        );
        pmetal_runtime::check_unobserved_error().unwrap();
    }
}
