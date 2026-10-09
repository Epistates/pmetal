//! Cut cross-entropy against an f32 log-softmax on a real checkpoint.
//!
//! A bf16 model's LM head produces bf16 logits, and a loss computed from them
//! carries their rounding. This measures how far each way of computing the
//! loss lands from the same loss with every operand in f32: the loss, its
//! gradient with respect to the hidden states, and with respect to the head
//! (which an adapter on `lm_head` trains through). Each variant also runs
//! alone in a child process, for its peak memory and time.
//!
//! `PMETAL_CCE_CHECKPOINT=/path/to/Qwen3-0.6B cargo test -p pmetal-models
//! --test cce_precision -- --ignored --nocapture`, several paths separated by
//! `:` to sweep.

use std::time::Instant;

use pmetal_bridge::compat::{Array, Dtype, ops};
use pmetal_bridge::inline_array::{
    clear_cache, get_active_memory, get_peak_memory, reset_peak_memory, value_and_grad,
};
use pmetal_models::dispatcher::{DynamicModel, LmHead};

const TEST_NAME: &str = "cut_cross_entropy_precision_on_a_real_checkpoint";

const TEXT: &str = "The lighthouse keeper climbed the spiral stairs every evening at dusk, \
counting the one hundred and twelve steps out of habit rather than necessity. From the \
gallery he could see the fishing boats returning to the harbour, their wakes drawing pale \
lines across the darkening water. He trimmed the wick, polished the great lens until it \
gleamed, and wrote the weather in the logbook: wind from the south-west, sea moderate, \
visibility good. In the winter of the great storm the light had burned for three days \
without rest, and the keeper had not slept at all. Ships had found the channel by its beam \
when every other mark on the coast was lost in the spray.\n\n\
def merge_sorted(left, right):\n    result = []\n    i = j = 0\n    while i < len(left) and \
j < len(right):\n        if left[i] <= right[j]:\n            result.append(left[i])\n            \
i += 1\n        else:\n            result.append(right[j])\n            j += 1\n    \
result.extend(left[i:])\n    result.extend(right[j:])\n    return result\n\n\
Photosynthesis converts light energy into chemical energy stored in glucose. In the light \
reactions, water is split and oxygen is released, while ATP and NADPH are produced. The \
Calvin cycle then uses that ATP and NADPH to fix carbon dioxide into three-carbon sugars, \
which the plant assembles into glucose, sucrose and starch. The overall equation is \
6 CO2 + 6 H2O + light -> C6H12O6 + 6 O2, although the real pathway runs through dozens of \
intermediate steps and enzymes, the most abundant of which, RuBisCO, is thought to be the \
most common protein on Earth.";

fn f32_(a: &Array) -> Array {
    a.as_dtype(Dtype::Float32.as_i32())
}

/// `softcap(scale · (h·Wᵀ + b))` in whatever dtype `h` and `w` are.
fn logits(head: &LmHead, h: &Array, w: &Array) -> Array {
    let mut z = h.matmul(&w.t());
    if let Some(b) = &head.bias {
        z = z.add(&b.as_dtype(z.dtype().as_i32()));
    }
    if head.logit_scale != 1.0 {
        z = z.multiply(&Array::from_f32(head.logit_scale));
    }
    if let Some(cap) = head.softcap {
        let cap = Array::from_f32(cap);
        z = ops::tanh(&z.divide(&cap)).multiply(&cap);
    }
    z
}

const VARIANTS: [&str; 4] = [
    "f32 reference",
    "full logits",
    "full logits, f32 softmax",
    "cut cross-entropy",
];

/// The loss of `variant`, from the hidden states and the head weight.
fn loss(variant: &str, head: &LmHead, targets: &Array, h: &Array, w: &Array) -> Array {
    let ce = |z: &Array| pmetal_bridge::training::cross_entropy_loss(z, targets, -100);
    match variant {
        "f32 reference" => ce(&logits(head, &f32_(h), &f32_(w))),
        "full logits" => ce(&logits(head, h, w)),
        "full logits, f32 softmax" => ce(&f32_(&logits(head, h, w))),
        "cut cross-entropy" => {
            let mut head = head.clone();
            head.weight = w.clone();
            head.cut_cross_entropy(h, targets, -100).expect("cce")
        }
        other => panic!("no variant {other}"),
    }
}

/// Loss and gradients of one variant, with respect to the hidden states and,
/// when `wrt_head`, the head.
fn step(
    variant: &str,
    head: &LmHead,
    targets: &Array,
    h: &Array,
    wrt_head: bool,
) -> (Array, Vec<Array>) {
    let w = head.weight.clone();
    let (value, grads) = if wrt_head {
        value_and_grad(
            |a| loss(variant, head, targets, &a[0], &a[1]),
            &[h.clone(), w],
            &[],
        )
    } else {
        value_and_grad(
            |a| loss(variant, head, targets, &a[0], &w),
            std::slice::from_ref(h),
            &[],
        )
    };
    value.eval();
    for g in &grads {
        g.eval();
    }
    if let Err(e) = pmetal_bridge::check_last_error() {
        panic!("{variant}: a bridge op threw: {e}");
    }
    (value, grads)
}

fn rel(got: &Array, want: &Array) -> f32 {
    let d = f32_(got).subtract(want).square().sum_all().sqrt();
    let n = want.square().sum_all().sqrt();
    d.divide(&n).item_f32()
}

/// Copies of the text in the batch, `PMETAL_CCE_REPEAT` (default 1): the
/// logits' share of memory grows with the token count, so a long batch is
/// where the loss's memory shows.
fn repeats() -> usize {
    std::env::var("PMETAL_CCE_REPEAT")
        .ok()
        .and_then(|r| r.parse().ok())
        .unwrap_or(1)
}

struct Batch {
    head: LmHead,
    hidden: Array,
    targets: Array,
}

fn batch(dir: &str) -> Batch {
    let mut model = DynamicModel::load(dir).expect("load");
    let tokenizer = pmetal_data::Tokenizer::from_model_dir(dir).expect("tokenizer");
    let ids: Vec<i32> = tokenizer
        .encode(&TEXT.repeat(repeats()))
        .expect("encode")
        .into_iter()
        .map(|t| t as i32)
        .collect();
    let n = ids.len() as i32;
    let input = Array::from_i32_slice_shaped(&ids, &[1, n]);
    let hidden = model.forward_hidden(&input, None).expect("hidden");
    let head = model.lm_head().expect("lm head");
    let hidden = hidden
        .slice(&[0, 0, 0], &[1, n - 1, hidden.dim(2)])
        .reshape(&[n - 1, -1]);
    hidden.eval();
    head.weight.eval();
    Batch {
        head,
        hidden,
        targets: Array::from_i32_slice_shaped(&ids[1..], &[n - 1]),
    }
}

/// Child process: one variant alone, for its peak memory and time.
fn profile(dir: &str, variant: &str, wrt_head: bool) {
    let b = batch(dir);
    // Memory from the first step in a fresh process: a step's arrays can
    // outlive it inside the bridge until the next one replaces them, which
    // hides a later step's allocations from the peak.
    clear_cache();
    let base = get_active_memory();
    reset_peak_memory();
    drop(step(variant, &b.head, &b.targets, &b.hidden, wrt_head));
    let peak = get_peak_memory().saturating_sub(base);
    let mut times = Vec::new();
    for _ in 0..5 {
        let start = Instant::now();
        drop(step(variant, &b.head, &b.targets, &b.hidden, wrt_head));
        times.push(start.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(f64::total_cmp);
    println!("PROFILE {:.0} {:.1}", peak as f64 / 1e6, times[2]);
}

#[test]
#[ignore = "needs PMETAL_CCE_CHECKPOINT"]
fn cut_cross_entropy_precision_on_a_real_checkpoint() {
    let paths = std::env::var("PMETAL_CCE_CHECKPOINT").expect("PMETAL_CCE_CHECKPOINT");
    if let Ok(spec) = std::env::var("PMETAL_CCE_PROFILE") {
        let (variant, wrt) = spec.split_once('|').expect("variant|wrt");
        return profile(&paths, variant, wrt == "head");
    }
    for dir in paths.split(':') {
        let b = batch(dir);
        println!(
            "{dir}: {} tokens, hidden {:?}, head {:?} {:?}",
            b.targets.dim(0),
            b.hidden.shape(),
            b.head.weight.shape(),
            b.head.weight.dtype()
        );
        let (want, want_grads) = step(VARIANTS[0], &b.head, &b.targets, &b.hidden, true);
        let want = want.item_f32();
        let mut errors = Vec::new();
        for variant in VARIANTS {
            let (got, grads) = step(variant, &b.head, &b.targets, &b.hidden, true);
            let error = [
                (got.item_f32() - want).abs(),
                rel(&grads[0], &want_grads[0]),
                rel(&grads[1], &want_grads[1]),
            ];
            drop(grads);
            let mut line = format!(
                "  {variant:<26} loss {:.6} |Δ| {:.1e}  ∂h {:.1e}  ∂W {:.1e}",
                got.item_f32(),
                error[0],
                error[1],
                error[2],
            );
            errors.push(error);
            for wrt in ["hidden", "head"] {
                let out = std::process::Command::new(std::env::current_exe().unwrap())
                    .args(["--exact", TEST_NAME, "--ignored", "--nocapture"])
                    .env("PMETAL_CCE_CHECKPOINT", dir)
                    .env("PMETAL_CCE_PROFILE", format!("{variant}|{wrt}"))
                    .output()
                    .expect("child");
                let stdout = String::from_utf8_lossy(&out.stdout);
                let profile = stdout
                    .lines()
                    .find_map(|l| l.split_once("PROFILE ").map(|(_, p)| p))
                    .unwrap_or("? ?");
                let (mb, ms) = profile.split_once(' ').unwrap_or(("?", "?"));
                line.push_str(&format!("  [∂{wrt}: {mb} MB {ms} ms]"));
            }
            println!("{line}");
        }
        // Cut cross-entropy is no further from f32 than the full logits with
        // an f32 softmax: the head's own rounding, and nothing on top beyond
        // f32 summation order (a few 1e-4 of a gradient, where the logits are
        // f32 already).
        let (full, cut) = (errors[2], errors[3]);
        for (what, f, c, floor) in [
            ("loss", full[0], cut[0], 1e-5),
            ("∂h", full[1], cut[1], 5e-4),
            ("∂W", full[2], cut[2], 5e-4),
        ] {
            assert!(
                c <= 1.05 * f + floor,
                "{dir}: cut cross-entropy's {what} error {c:.2e} exceeds the full logits' {f:.2e}"
            );
        }
    }
}
