//! Every optimizer pmetal trains with, step by step against its reference.
//!
//! `fixtures/optimizer_reference.json` holds eight steps of each optimizer
//! over three f32 parameters (a 4x6 and a 3x5 matrix, a 7-vector) with fresh
//! random gradients every step and a warmup-then-cosine learning rate:
//!
//! - AdamW: reference values from PyTorch 2.14.1 `torch.optim.AdamW`
//!   (single-tensor path, f32, CPU), at betas (0.9, 0.999), eps 1e-8, weight
//!   decay 0.1, and at betas (0.9, 0.95), eps 1e-6, no weight decay;
//! - SGD: `torch.optim.SGD` with momentum 0.9 and weight decay 0.1;
//! - Lion: the authors' PyTorch reference implementation (Chen et al., arXiv
//!   2302.06675), betas (0.9, 0.99), weight decay 0.1;
//! - Adafactor: `transformers.optimization.Adafactor` 5.19.0 with an external
//!   learning rate (`relative_step=False`, `scale_parameter=False`,
//!   `warmup_init=False`, `beta1=None`), weight decay 0.1.
//!
//! Each implementation must land on the reference after every step. The
//! gradients span three scales (1, 10 and 1e-3), so Adam's epsilon and
//! Adafactor's update clipping both take part.

use std::collections::HashMap;
use std::rc::Rc;

use pmetal_bridge::compat::{
    Array, FlattenedModuleParam, ModuleParametersExt,
    optimizers::{AdamWBuilder, Optimizer},
};
use pmetal_bridge::optimizer::{AdamW as BridgeAdamW, ParamClass, ParamSet};
use pmetal_core::OptimizerType;
use pmetal_metal::kernels::AdamWConfig;
use pmetal_trainer::{MlxMetalOptimizerBuilder, ParamGroupOptimizerBuilder, TrainOptimizer};
use serde::Deserialize;

const NAMES: [&str; 3] = ["w0", "w1", "w2"];
/// Absolute tolerance per element, scaled up for elements above 1.
const ATOL: f64 = 1e-6;

#[derive(Deserialize)]
struct Fixture {
    steps: usize,
    lr: Vec<f32>,
    params: HashMap<String, Init>,
    grads: Vec<HashMap<String, Vec<f32>>>,
    optimizers: HashMap<String, Case>,
}

#[derive(Deserialize)]
struct Init {
    shape: Vec<i32>,
    init: Vec<f32>,
}

#[derive(Deserialize)]
struct Case {
    hyper: HashMap<String, serde_json::Value>,
    /// Parameters after each step.
    trajectory: Vec<HashMap<String, Vec<f32>>>,
}

impl Case {
    fn f32(&self, key: &str) -> f32 {
        self.hyper[key]
            .as_f64()
            .unwrap_or_else(|| panic!("hyper {key}")) as f32
    }
    fn betas(&self) -> (f32, f32) {
        let b = self.hyper["betas"].as_array().expect("betas");
        (b[0].as_f64().unwrap() as f32, b[1].as_f64().unwrap() as f32)
    }
}

fn fixture() -> Fixture {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/optimizer_reference.json"
    );
    serde_json::from_str(&std::fs::read_to_string(path).expect("read fixture")).expect("parse")
}

/// The three parameters, as a model the optimizers can update.
struct Weights {
    w0: Array,
    w1: Array,
    w2: Array,
}
pmetal_bridge::impl_module_params!(Weights; w0, w1, w2);

impl Fixture {
    fn shape(&self, name: &str) -> &[i32] {
        &self.params[name].shape
    }

    fn model(&self) -> Weights {
        let p = |n: &str| Array::from_f32_slice(&self.params[n].init, self.shape(n));
        Weights {
            w0: p("w0"),
            w1: p("w1"),
            w2: p("w2"),
        }
    }

    fn grads(&self, step: usize) -> FlattenedModuleParam {
        NAMES
            .iter()
            .map(|&n| {
                (
                    Rc::from(n),
                    Array::from_f32_slice(&self.grads[step][n], self.shape(n)),
                )
            })
            .collect()
    }
}

fn values(a: &Array) -> Vec<f32> {
    a.eval();
    a.as_slice::<f32>().to_vec()
}

/// The model's parameters by name.
fn snapshot(model: &Weights) -> HashMap<String, Vec<f32>> {
    model
        .flatten_params()
        .iter()
        .map(|(k, v)| (k.to_string(), values(v)))
        .collect()
}

/// Largest error of `got` against step `step` of `case`, and where.
fn worst(case: &Case, step: usize, got: &HashMap<String, Vec<f32>>) -> (f64, String) {
    let mut worst = (0.0f64, String::new());
    for name in NAMES {
        let want = &case.trajectory[step][name];
        let got = &got[name];
        assert_eq!(got.len(), want.len(), "{name} length");
        for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
            let err = (g as f64 - w as f64).abs() / (w as f64).abs().max(1.0);
            if err > worst.0 || err.is_nan() {
                worst = (err, format!("{name}[{i}]: got {g}, want {w}"));
            }
        }
    }
    worst
}

/// Run `step(model, step_index)` over the fixture and compare every step.
fn check(
    label: &str,
    fx: &Fixture,
    case: &str,
    mut step: impl FnMut(&mut Weights, usize) -> HashMap<String, Vec<f32>>,
) {
    let case = &fx.optimizers[case];
    let mut model = fx.model();
    let mut max_err = 0.0f64;
    for s in 0..fx.steps {
        let got = step(&mut model, s);
        let (err, at) = worst(case, s, &got);
        assert!(
            err <= ATOL,
            "{label}: step {} is {err:.3e} off the reference at {at}",
            s + 1
        );
        max_err = max_err.max(err);
    }
    pmetal_bridge::check_unobserved_error().expect("no bridge op failed");
    println!("{label}: max error {max_err:.2e} over {} steps", fx.steps);
}

/// Every optimizer the trainer's loops and commands build
/// ([`TrainOptimizer`]), driven through `Optimizer::update`.
#[test]
fn train_optimizers_match_their_references() {
    let fx = fixture();
    for (kind, case) in [
        (OptimizerType::AdamW, "adamw"),
        (OptimizerType::Sgd, "sgd"),
        (OptimizerType::Lion, "lion"),
        (OptimizerType::Adafactor, "adafactor"),
    ] {
        let wd = fx.optimizers[case].f32("weight_decay");
        let mut opt = TrainOptimizer::new(kind, fx.lr[0], wd);
        check(&format!("{kind:?}"), &fx, case, |model, s| {
            opt.set_lr(fx.lr[s]);
            opt.update(model, fx.grads(s)).expect("update");
            snapshot(model)
        });
    }
}

/// The parameter-group AdamW the SFT, packed and distillation loops build.
#[test]
fn grouped_adamw_matches_torch() {
    let fx = fixture();
    let mut opt = ParamGroupOptimizerBuilder::new(OptimizerType::AdamW, fx.lr[0])
        .with_weight_decay(fx.optimizers["adamw"].f32("weight_decay"))
        .build();
    check("grouped AdamW", &fx, "adamw", |model, s| {
        opt.set_learning_rate(fx.lr[s]);
        opt.update(model, fx.grads(s)).expect("update");
        snapshot(model)
    });
}

/// `AdamWBuilder`, as the pretrain, MTP and embedding trainers build it.
#[test]
fn built_adamw_matches_torch() {
    let fx = fixture();
    for case in ["adamw", "adamw_beta2_095"] {
        let c = &fx.optimizers[case];
        let mut opt = AdamWBuilder::new(fx.lr[0])
            .betas(c.betas())
            .eps(c.f32("eps"))
            .weight_decay(c.f32("weight_decay"))
            .build()
            .unwrap();
        check(&format!("AdamWBuilder {case}"), &fx, case, |model, s| {
            opt.set_lr(fx.lr[s]);
            opt.update(model, fx.grads(s)).expect("update");
            snapshot(model)
        });
    }
}

/// The bridge AdamW over a `ParamSet`, every parameter decayed alike, as the
/// DiffusionGemma trainer runs it.
#[test]
fn bridge_adamw_matches_torch() {
    let fx = fixture();
    for case in ["adamw", "adamw_beta2_095"] {
        let c = &fx.optimizers[case];
        let (b1, b2) = c.betas();
        let mut opt = BridgeAdamW::new(fx.lr[0], c.f32("weight_decay"))
            .with_betas(b1, b2)
            .with_eps(c.f32("eps"))
            .with_classifier(|_| ParamClass::Regular);
        check(&format!("bridge AdamW {case}"), &fx, case, |model, s| {
            opt.set_lr(fx.lr[s]);
            let mut params: ParamSet = model
                .flatten_params()
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect();
            let grads: ParamSet = fx
                .grads(s)
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect();
            opt.step(&mut params, &grads);
            *model = Weights {
                w0: params.remove("w0").unwrap(),
                w1: params.remove("w1").unwrap(),
                w2: params.remove("w2").unwrap(),
            };
            snapshot(model)
        });
    }
}

/// The Metal fused AdamW `pmetal train` runs by default.
#[test]
fn metal_fused_adamw_matches_torch() {
    let fx = fixture();
    for case in ["adamw", "adamw_beta2_095"] {
        let c = &fx.optimizers[case];
        let (b1, b2) = c.betas();
        let mut opt = MlxMetalOptimizerBuilder::new(fx.lr[0])
            .beta1(b1)
            .beta2(b2)
            .epsilon(c.f32("eps"))
            .weight_decay(c.f32("weight_decay"))
            .build()
            .unwrap();
        check(
            &format!("Metal fused AdamW {case}"),
            &fx,
            case,
            |model, s| {
                opt.set_learning_rate(fx.lr[s]);
                opt.update_model_fused(model, &fx.grads(s)).expect("update");
                snapshot(model)
            },
        );
    }
}

/// The CPU AdamW the ANE trainer updates its weights with.
#[test]
fn cpu_adamw_matches_torch() {
    let fx = fixture();
    for case in ["adamw", "adamw_beta2_095"] {
        let c = &fx.optimizers[case];
        let (beta1, beta2) = c.betas();
        // Parameter, first moment, second moment.
        type Slot = (Vec<f32>, Vec<f32>, Vec<f32>);
        let mut state: HashMap<&str, Slot> = NAMES
            .iter()
            .map(|&n| {
                let p = fx.params[n].init.clone();
                let z = vec![0.0; p.len()];
                (n, (p, z.clone(), z))
            })
            .collect();
        check(&format!("CPU AdamW {case}"), &fx, case, |_, s| {
            let scalars = AdamWConfig {
                learning_rate: fx.lr[s],
                beta1,
                beta2,
                epsilon: c.f32("eps"),
                weight_decay: c.f32("weight_decay"),
                step: s as u32 + 1,
            }
            .scalars()
            .unwrap();
            for (name, (p, m, v)) in state.iter_mut() {
                pmetal_metal::accelerate::adamw_update(p, &fx.grads[s][*name], m, v, &scalars);
            }
            state
                .iter()
                .map(|(n, (p, _, _))| (n.to_string(), p.clone()))
                .collect()
        });
    }
}

/// The Metal 4 AdamW kernel, on GPUs that have it (M5 and later).
#[test]
fn mpp_fused_adamw_matches_torch_where_available() {
    use pmetal_metal::buffer::{BufferUsage, MetalBuffer};
    use pmetal_metal::context::MetalContext;
    use pmetal_metal::kernels::FusedAdamW;
    use pmetal_metal::kernels::mpp_fused_training::{
        MppFusedAdamW, MppFusedAdamWConfig, MppParamInfo,
    };

    let fx = fixture();
    let ctx = MetalContext::global().expect("Metal");
    let sizes: Vec<usize> = NAMES.iter().map(|n| fx.params[*n].init.len()).collect();
    let max = sizes.iter().copied().max().unwrap();
    let mpp = MppFusedAdamW::new(ctx.clone(), MppFusedAdamWConfig::new(sizes.len(), max));
    if !mpp.is_available() {
        println!("no Metal 4 AdamW on this GPU; skipped");
        return;
    }
    let infos = FusedAdamW::build_param_info(&sizes);
    let mpp_infos: Vec<MppParamInfo> = infos
        .iter()
        .map(|i| MppParamInfo {
            offset: i.offset,
            size: i.size,
            m_offset: i.m_offset,
            v_offset: i.v_offset,
        })
        .collect();
    let shared = BufferUsage::Shared;
    let info_buf = MetalBuffer::from_slice(&ctx, &infos, shared).unwrap();
    let flat = |f: &dyn Fn(&str) -> Vec<f32>| NAMES.iter().flat_map(|n| f(n)).collect::<Vec<_>>();
    let total: usize = sizes.iter().sum();

    for case in ["adamw", "adamw_beta2_095"] {
        let c = &fx.optimizers[case];
        let (beta1, beta2) = c.betas();
        let params =
            MetalBuffer::from_slice(&ctx, &flat(&|n| fx.params[n].init.clone()), shared).unwrap();
        let m = MetalBuffer::<f32>::zeros(&ctx, total, shared).unwrap();
        let v = MetalBuffer::<f32>::zeros(&ctx, total, shared).unwrap();
        check(&format!("Metal 4 AdamW {case}"), &fx, case, |_, s| {
            let grads =
                MetalBuffer::from_slice(&ctx, &flat(&|n| fx.grads[s][n].clone()), shared).unwrap();
            let config = AdamWConfig {
                learning_rate: fx.lr[s],
                beta1,
                beta2,
                epsilon: c.f32("eps"),
                weight_decay: c.f32("weight_decay"),
                step: s as u32 + 1,
            };
            mpp.execute(&params, &grads, &m, &v, &mpp_infos, &info_buf, &config)
                .expect("Metal 4 AdamW");
            let out = params.as_slice();
            let mut at = 0;
            NAMES
                .iter()
                .zip(&sizes)
                .map(|(n, &len)| {
                    at += len;
                    (n.to_string(), out[at - len..at].to_vec())
                })
                .collect()
        });
    }
}

/// AdamW in f64, with the two ways an implementation can miss PyTorch's.
fn adamw_f64(fx: &Fixture, case: &Case, bias_correction: bool, decay_first: bool) -> f64 {
    let (b1, b2) = case.betas();
    let (b1, b2, eps, wd) = (
        b1 as f64,
        b2 as f64,
        case.f32("eps") as f64,
        case.f32("weight_decay") as f64,
    );
    let mut worst_err = 0.0f64;
    for name in NAMES {
        let mut p: Vec<f64> = fx.params[name].init.iter().map(|&x| x as f64).collect();
        let mut m = vec![0.0; p.len()];
        let mut v = vec![0.0; p.len()];
        for s in 0..fx.steps {
            let lr = fx.lr[s] as f64;
            let t = (s + 1) as i32;
            let (bc1, bc2) = if bias_correction {
                (1.0 - b1.powi(t), 1.0 - b2.powi(t))
            } else {
                (1.0, 1.0)
            };
            for i in 0..p.len() {
                let g = fx.grads[s][name][i] as f64;
                m[i] = b1 * m[i] + (1.0 - b1) * g;
                v[i] = b2 * v[i] + (1.0 - b2) * g * g;
                let update = lr / bc1 * m[i] / (v[i].sqrt() / bc2.sqrt() + eps);
                p[i] = if decay_first {
                    p[i] * (1.0 - lr * wd) - update
                } else {
                    (p[i] - update) * (1.0 - lr * wd)
                };
                let want = case.trajectory[s][name][i] as f64;
                worst_err = worst_err.max((p[i] - want).abs() / want.abs().max(1.0));
            }
        }
    }
    worst_err
}

/// The reference tells PyTorch's AdamW from the near misses: an update
/// without bias correction, or one that decays the updated parameter.
#[test]
fn the_reference_rejects_adamw_without_bias_correction_or_with_late_decay() {
    let fx = fixture();
    let case = &fx.optimizers["adamw"];
    let exact = adamw_f64(&fx, case, true, true);
    assert!(exact <= ATOL, "f64 AdamW is {exact:.3e} off the reference");
    let no_bias_correction = adamw_f64(&fx, case, false, true);
    assert!(
        no_bias_correction > 1e-3,
        "no bias correction is only {no_bias_correction:.3e} off"
    );
    let late_decay = adamw_f64(&fx, case, true, false);
    assert!(late_decay > 1e-4, "late decay is only {late_decay:.3e} off");
    println!(
        "f64 AdamW {exact:.2e}; without bias correction {no_bias_correction:.2e}; \
         decaying after the update {late_decay:.2e}"
    );
}
