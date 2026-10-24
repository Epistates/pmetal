//! AdamW step time: the fused Metal kernel against the op-based reference.
//!
//! `cargo test --release -p pmetal-runtime --test adamw_bench -- --ignored --nocapture`
//!
//! Two parameter sets, each stepped with the gradients already computed and
//! every output evaluated before the clock stops:
//!
//! * a 0.6B-parameter dense set (a Qwen3-0.6B-shaped model in bf16: parameters
//!   and gradients in bf16, moments and master copies in f32), the shape of a
//!   full fine-tune;
//! * a LoRA-sized set (hundreds of small f32 tensors), where the cost is the
//!   number of nodes and dispatches, not the bytes.

use pmetal_runtime::compat::Dtype;
use pmetal_runtime::{AdamW, InlineArray};
use std::time::Instant;

struct Set {
    label: &'static str,
    dtype: Dtype,
    /// (name, shape)
    params: Vec<(String, Vec<i32>)>,
}

/// Qwen3-0.6B: hidden 1024, 28 layers, 16 query and 8 key/value heads of 128,
/// intermediate 3072, vocabulary 151936, head tied to the embedding.
fn dense_0_6b() -> Set {
    let (h, inter, layers, vocab, q, kv) = (1024, 3072, 28, 151_936, 2048, 1024);
    let mut params = vec![("model.embed_tokens.weight".to_string(), vec![vocab, h])];
    for l in 0..layers {
        let p = format!("model.layers.{l}");
        for (n, shape) in [
            ("self_attn.q_proj.weight", vec![q, h]),
            ("self_attn.k_proj.weight", vec![kv, h]),
            ("self_attn.v_proj.weight", vec![kv, h]),
            ("self_attn.o_proj.weight", vec![h, q]),
            ("mlp.gate_proj.weight", vec![inter, h]),
            ("mlp.up_proj.weight", vec![inter, h]),
            ("mlp.down_proj.weight", vec![h, inter]),
            ("input_layernorm.weight", vec![h]),
            ("post_attention_layernorm.weight", vec![h]),
            ("self_attn.q_norm.weight", vec![128]),
            ("self_attn.k_norm.weight", vec![128]),
        ] {
            params.push((format!("{p}.{n}"), shape));
        }
    }
    params.push(("model.norm.weight".to_string(), vec![h]));
    Set {
        label: "0.6B dense, bf16",
        dtype: Dtype::Bfloat16,
        params,
    }
}

/// Rank-16 adapters on all seven projections of a 28-layer, hidden-1024 model.
fn lora() -> Set {
    let (h, inter, layers, r) = (1024, 3072, 28, 16);
    let mut params = Vec::new();
    for l in 0..layers {
        for (n, din, dout) in [
            ("self_attn.q_proj", h, 2048),
            ("self_attn.k_proj", h, 1024),
            ("self_attn.v_proj", h, 1024),
            ("self_attn.o_proj", 2048, h),
            ("mlp.gate_proj", h, inter),
            ("mlp.up_proj", h, inter),
            ("mlp.down_proj", inter, h),
        ] {
            let p = format!("model.layers.{l}.{n}");
            params.push((format!("{p}.lora_a"), vec![r, din]));
            params.push((format!("{p}.lora_b"), vec![dout, r]));
        }
    }
    Set {
        label: "LoRA r16, f32",
        dtype: Dtype::Float32,
        params,
    }
}

fn elements(set: &Set) -> usize {
    set.params
        .iter()
        .map(|(_, s)| s.iter().map(|&d| d as usize).product::<usize>())
        .sum()
}

/// How a step's parameters and optimizer state are waited for.
#[derive(Clone, Copy)]
enum Wait {
    /// Queue every array, then wait on each.
    Queued,
    /// One `eval` per array, each its own wait (what `compat::eval` did).
    PerArray,
    /// One `compat::eval` of the whole set.
    OneCall,
}

/// Median step time in milliseconds over `steps` steps after `warmup`.
fn time_steps(set: &Set, fused: bool, warmup: usize, steps: usize) -> f64 {
    time_steps_waiting(set, fused, warmup, steps, Wait::Queued)
}

fn time_steps_waiting(set: &Set, fused: bool, warmup: usize, steps: usize, wait: Wait) -> f64 {
    let dtype = set.dtype.as_i32();
    let mut params: Vec<InlineArray> = set
        .params
        .iter()
        .map(|(_, s)| InlineArray::random_normal(s, Dtype::Float32.as_i32()).as_dtype(dtype))
        .collect();
    let grads: Vec<InlineArray> = set
        .params
        .iter()
        .map(|(_, s)| InlineArray::random_normal(s, Dtype::Float32.as_i32()).as_dtype(dtype))
        .collect();
    for a in params.iter().chain(&grads) {
        a.eval();
    }
    let mut opt = AdamW::new(1e-4, 0.01).with_fused(fused);
    let mut times = Vec::new();
    for step in 0..warmup + steps {
        let started = Instant::now();
        opt.advance_step();
        for (((name, _), grad), param) in set.params.iter().zip(&grads).zip(&mut params) {
            opt.step_single(name, grad, param);
        }
        let all: Vec<&InlineArray> = params
            .iter()
            .chain(opt.states.values().flat_map(|s| s.arrays()))
            .collect();
        match wait {
            // Queue everything, then wait: one submission, not one sync per array.
            Wait::Queued => {
                for a in &all {
                    a.async_eval();
                }
                for a in &all {
                    a.eval();
                }
            }
            Wait::PerArray => {
                for a in &all {
                    a.eval();
                }
            }
            Wait::OneCall => {
                pmetal_runtime::compat::eval(all.iter().copied()).expect("eval");
            }
        }
        pmetal_runtime::check_unobserved_error().expect("no bridge error");
        if step >= warmup {
            times.push(started.elapsed().as_secs_f64() * 1e3);
        }
    }
    times.sort_by(|a, b| a.total_cmp(b));
    times[times.len() / 2]
}

#[test]
#[ignore = "benchmark"]
fn adamw_step_time() {
    println!("| set | params | ops (ms) | fused (ms) | speedup |");
    println!("|---|---|---|---|---|");
    for set in [lora(), dense_0_6b()] {
        let (warmup, steps) = if set.params.len() > 1000 || elements(&set) < 100_000_000 {
            (3, 15)
        } else {
            (2, 8)
        };
        let ops = time_steps(&set, false, warmup, steps);
        let fused = time_steps(&set, true, warmup, steps);
        println!(
            "| {} | {:.0}M | {ops:.1} | {fused:.1} | {:.2}x |",
            set.label,
            elements(&set) as f64 / 1e6,
            ops / fused
        );
    }
}

/// The optimizer step with its parameters and state waited for one array at a
/// time and in one call, alternated in this process so machine load cancels.
///
/// `cargo test --release -p pmetal-runtime --test adamw_bench eval_set_step_time -- --ignored --nocapture`
#[test]
#[ignore = "benchmark"]
fn eval_set_step_time() {
    println!("| set | arrays | per array (ms) | one call (ms) | speedup |");
    println!("|---|---|---|---|---|");
    for set in [lora(), dense_0_6b()] {
        let (warmup, steps) = (3, 12);
        let (mut per_array, mut one_call) = (Vec::new(), Vec::new());
        for _ in 0..3 {
            per_array.push(time_steps_waiting(
                &set,
                true,
                warmup,
                steps,
                Wait::PerArray,
            ));
            one_call.push(time_steps_waiting(&set, true, warmup, steps, Wait::OneCall));
        }
        per_array.sort_by(|a, b| a.total_cmp(b));
        one_call.sort_by(|a, b| a.total_cmp(b));
        println!(
            "| {} | {} | {:.1} | {:.1} | {:.2}x |",
            set.label,
            set.params.len() * 3,
            per_array[1],
            one_call[1],
            per_array[1] / one_call[1]
        );
    }
}
