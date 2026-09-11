//! Gradient checkpointing has to actually save memory.
//!
//! The correctness tests next door prove checkpointed gradients match retained
//! ones. That is necessary but not sufficient: a checkpoint that is correct and
//! saves nothing is a pure loss, one extra forward pass for no benefit.
//!
//! What gets saved is a block's *interior*, the tensors produced and consumed
//! inside it. A bare `Linear` has none, so checkpointing one costs a recompute
//! and returns nothing. The unit here is therefore shaped like the thing a
//! trunk is actually made of: a block whose middle is several times wider than
//! its input and output, which is where a transformer's activation memory
//! mostly lives (the MLP expansion and the attention score matrix).

use pmetal_bridge::compat::nn::LinearBuilder;
use pmetal_bridge::compat::{Array, Dtype, Module, checkpointed, layers::Linear, nn, ops};
use pmetal_bridge::inline_array::{get_peak_memory, reset_peak_memory, value_and_grad};

const LAYERS: usize = 16;
const HIDDEN: i32 = 512;
const EXPANSION: i32 = 4;
const BATCH: i32 = 4;
const SEQ: i32 = 256;
const RANK: i32 = 8;

/// An MLP block: widen, activate, narrow. The widened tensor is the interior
/// that checkpointing gets to throw away.
#[derive(Debug)]
struct Block {
    up: Linear,
    down: Linear,
}

pmetal_bridge::impl_module_params!(Block; up, down);

impl Block {
    fn new() -> Self {
        let mut up = LinearBuilder::new(HIDDEN, HIDDEN * EXPANSION)
            .bias(false)
            .build()
            .unwrap();
        let mut down = LinearBuilder::new(HIDDEN * EXPANSION, HIDDEN)
            .bias(false)
            .build()
            .unwrap();
        for layer in [&mut up, &mut down] {
            layer.attach_lora(RANK, 16.0, false).unwrap();
            // A fresh adapter is zero on the B side, which would leave the
            // adapters out of the graph entirely.
            let adapter = layer.adapter.as_mut().unwrap();
            let shape = adapter.b.shape().to_vec();
            adapter.b =
                Array::ones(&shape, Dtype::Float32.as_i32()).multiply(&Array::from_f32(0.01));
        }
        Self { up, down }
    }

    fn forward(&mut self, x: &Array) -> Array {
        let wide = Module::forward(&mut self.up, x).unwrap();
        let activated = nn::gelu(&wide);
        Module::forward(&mut self.down, &activated).unwrap()
    }

    fn adapters(&self) -> Vec<Array> {
        [&self.up, &self.down]
            .iter()
            .flat_map(|l| {
                let a = l.adapter.as_ref().unwrap();
                [a.a.clone(), a.b.clone()]
            })
            .collect()
    }

    fn set_adapters(&mut self, values: &[Array]) {
        for (i, layer) in [&mut self.up, &mut self.down].into_iter().enumerate() {
            let adapter = layer.adapter.as_mut().unwrap();
            adapter.a = values[2 * i].clone();
            adapter.b = values[2 * i + 1].clone();
        }
    }
}

const PARAMS_PER_BLOCK: usize = 4;

/// One training step: run the stack, sum to a scalar, take gradients.
fn step(stack: &mut [Block], x: &Array, checkpoint: bool) -> f32 {
    let params: Vec<Array> = stack.iter().flat_map(|b| b.adapters()).collect();
    let n_params = params.len();

    let (loss, _grads) = value_and_grad(
        |arrays| {
            for (i, block) in stack.iter_mut().enumerate() {
                block.set_adapters(&arrays[i * PARAMS_PER_BLOCK..(i + 1) * PARAMS_PER_BLOCK]);
            }

            let mut h = arrays[n_params].clone();
            for block in stack.iter_mut() {
                let out = if checkpoint {
                    // Safety: `stack` outlives this step, and the graph built
                    // here is differentiated once, inside this call.
                    unsafe { checkpointed(block, &[h], |b, ins| Ok(b.forward(&ins[0]))) }.unwrap()
                } else {
                    block.forward(&h)
                };
                h = ops::tanh(&out);
            }
            h.sum_all()
        },
        &params,
        std::slice::from_ref(x),
    );

    loss.item_f32()
}

fn peak_for(checkpoint: bool) -> (usize, f32) {
    let mut stack: Vec<Block> = (0..LAYERS).map(|_| Block::new()).collect();
    let x = Array::ones(&[BATCH, SEQ, HIDDEN], Dtype::Float32.as_i32())
        .multiply(&Array::from_f32(0.02));

    pmetal_bridge::inline_array::clear_cache();
    reset_peak_memory();
    let loss = step(&mut stack, &x, checkpoint);
    let peak = get_peak_memory();
    (peak, loss)
}

#[test]
fn checkpointing_lowers_the_peak() {
    let (plain_peak, plain_loss) = peak_for(false);
    let (ckpt_peak, ckpt_loss) = peak_for(true);

    // Same computation, so the same answer. If this drifts, the memory numbers
    // below are comparing two different models.
    assert!(
        (plain_loss - ckpt_loss).abs() / plain_loss.abs().max(1.0) < 1e-3,
        "checkpointed loss {ckpt_loss} differs from plain {plain_loss}"
    );

    println!(
        "peak over {LAYERS} blocks: plain {:.1} MiB, checkpointed {:.1} MiB ({:.2}x)",
        plain_peak as f64 / (1024.0 * 1024.0),
        ckpt_peak as f64 / (1024.0 * 1024.0),
        plain_peak as f64 / ckpt_peak.max(1) as f64,
    );

    assert!(
        ckpt_peak < plain_peak,
        "checkpointing did not lower the peak: plain {plain_peak} bytes, \
         checkpointed {ckpt_peak} bytes. A correct checkpoint that saves nothing \
         is one extra forward pass for no benefit."
    );
}
