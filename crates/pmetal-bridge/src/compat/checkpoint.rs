//! Gradient checkpointing for a module's forward pass.
//!
//! Activation memory, not weight memory, is what caps batch size and sequence
//! length on a Mac. A trunk holds every layer's activations alive until the
//! backward pass consumes them, so peak memory grows with depth. Checkpointing
//! throws those activations away as soon as a layer is done and recomputes them
//! when the backward pass asks, trading one extra forward for a peak that no
//! longer scales with the number of layers.
//!
//! [`checkpointed`] is the module-level form of
//! [`checkpoint_apply_unchecked`](crate::inline_array::checkpoint_apply_unchecked),
//! and it exists because the raw primitive has a trap: `checkpoint` is a
//! `custom_vjp`, and a `custom_vjp` differentiates only with respect to the
//! arrays handed to it. A weight the closure merely captures is a constant to
//! the tape, so its gradient comes back **zero** and training quietly stops
//! working. [`checkpointed`] threads the module's trainable parameters through
//! as explicit inputs and writes them back before running the forward, which is
//! what `mlx.nn.utils.checkpoint` does in Python for the same reason.

use std::cell::RefCell;
use std::rc::Rc;

use crate::compat::{Array, Exception, ModuleParametersExt};
use crate::inline_array::checkpoint_apply_unchecked;

/// Run `forward` on `module` under gradient checkpointing.
///
/// `inputs` are the activations flowing in (hidden states, a mask, whatever the
/// forward needs to differentiate through). `module`'s trainable parameters are
/// added behind them automatically, so `forward` sees the module already
/// carrying the right weights and can ignore the plumbing.
///
/// Frozen weights are deliberately *not* threaded through: they need no
/// gradient, and passing them would inflate the primal list for nothing. For a
/// LoRA run that means the traffic is a handful of small adapter matrices per
/// layer rather than the layer itself.
///
/// # Safety
///
/// The backward pass calls `forward` again, after this function has returned,
/// and MLX keeps the closure alive for as long as anything still references the
/// resulting graph. `module` is borrowed across that window, so the caller must
/// keep it alive until the step's graph has been differentiated and dropped.
/// A training loop that owns its model satisfies this; stashing an output array
/// somewhere longer-lived than the model does not.
pub unsafe fn checkpointed<M, F>(
    module: &mut M,
    inputs: &[Array],
    mut forward: F,
) -> Result<Array, Exception>
where
    M: ModuleParametersExt,
    F: FnMut(&mut M, &[Array]) -> Result<Array, Exception>,
{
    // Sorted so the order the parameters go in matches the order they come back
    // out. A HashMap's iteration order does not survive between the two walks.
    let mut names: Vec<String> = module
        .flatten_trainable_params()
        .keys()
        .map(|k| k.to_string())
        .collect();
    names.sort_unstable();

    if names.is_empty() {
        // Nothing to differentiate inside this module, so checkpointing it would
        // buy a recompute and no gradient. Run it straight.
        return forward(module, inputs);
    }

    let n_inputs = inputs.len();
    let mut all_inputs: Vec<Array> = inputs.to_vec();
    {
        let live = module.flatten_params_mut();
        for name in &names {
            match live.get(name.as_str()) {
                Some(value) => all_inputs.push((*value).clone()),
                None => {
                    return Err(Exception::from(format!(
                        "checkpointing: trainable parameter `{name}` is missing from the \
                         module's parameter tree"
                    )));
                }
            }
        }
    }

    // Errors cannot travel out of the FFI callback, so the closure parks one
    // here and signals failure by producing no outputs. This is deliberately a
    // shared handle rather than a captured local: the closure runs again during
    // the backward pass, by which time this function's frame is gone.
    let failure: Rc<RefCell<Option<Exception>>> = Rc::new(RefCell::new(None));
    let closure_failure = Rc::clone(&failure);

    // `move` matters. A borrowing closure would capture `n_inputs` and `names`
    // as pointers into this frame, and the backward recompute reads them after
    // the frame has returned. That does not fail loudly; it reads whatever is
    // on the stack by then. Everything the closure keeps is either owned here
    // or, for `module` and `forward`, covered by this function's safety
    // contract.
    let outputs = unsafe {
        checkpoint_apply_unchecked(&all_inputs, move |arrays| {
            // Point the module at the arrays MLX is tracing. Without this the
            // forward would rebuild the graph from the module's own copies,
            // which are not primals, and every gradient would be zero.
            {
                let mut live = module.flatten_params_mut();
                for (name, traced) in names.iter().zip(&arrays[n_inputs..]) {
                    if let Some(slot) = live.get_mut(name.as_str()) {
                        **slot = traced.clone();
                    }
                }
            }

            match forward(module, &arrays[..n_inputs]) {
                Ok(out) => vec![out],
                Err(e) => {
                    *closure_failure.borrow_mut() = Some(e);
                    Vec::new()
                }
            }
        })
    };

    if let Some(e) = failure.borrow_mut().take() {
        return Err(e);
    }
    match outputs.into_iter().next() {
        Some(out) => Ok(out),
        None => Err(crate::check_last_error()
            .err()
            .map(|e| Exception::from(e.to_string()))
            .unwrap_or_else(|| {
                Exception::from("checkpointing: the forward pass produced no output")
            })),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compat::layers::Linear;
    use crate::compat::nn::LinearBuilder;
    use crate::inline_array::value_and_grad;

    fn adapted_linear() -> Linear {
        let mut linear = LinearBuilder::new(4, 4).bias(false).build().unwrap();
        linear.weight.value = Array::from_slice(
            &[
                0.1f32, -0.2, 0.3, 0.4, 0.5, -0.6, 0.7, -0.8, 0.9, 1.0, -1.1, 1.2, -1.3, 1.4, 0.15,
                -0.25,
            ],
            &[4, 4],
        );
        linear.attach_lora(2, 4.0, false).unwrap();
        let adapter = linear.adapter.as_mut().unwrap();
        // `a` is randomly initialised, so pin it: two calls otherwise return two
        // different models and nothing downstream would be comparable.
        adapter.a = Array::from_slice(&[0.2f32, -0.3, 0.45, 0.1, -0.15, 0.35, 0.05, -0.4], &[2, 4]);
        // `b` is zero-initialised, which would make every gradient check below
        // pass for the wrong reason.
        adapter.b = Array::from_slice(&[0.3f32, -0.4, 0.5, 0.6, -0.7, 0.8, 0.2, -0.1], &[4, 2]);
        linear
    }

    fn input() -> Array {
        Array::from_slice(&[1.0f32, -2.0, 0.5, 0.25, 0.75, -1.5, 2.0, -0.5], &[2, 4])
    }

    #[test]
    fn a_checkpointed_module_gives_the_same_gradients() {
        let x = input();

        let mut plain = adapted_linear();
        let plain_a = plain.adapter.as_ref().unwrap().a.clone();
        let plain_b = plain.adapter.as_ref().unwrap().b.clone();
        let (plain_loss, plain_grads) = value_and_grad(
            |arrays| {
                plain.adapter.as_mut().unwrap().a = arrays[0].clone();
                plain.adapter.as_mut().unwrap().b = arrays[1].clone();
                crate::compat::Module::forward(&mut plain, &arrays[2])
                    .unwrap()
                    .sum_all()
            },
            &[plain_a, plain_b],
            &[x.clone()],
        );

        let mut checkpointed_module = adapted_linear();
        let ckpt_a = checkpointed_module.adapter.as_ref().unwrap().a.clone();
        let ckpt_b = checkpointed_module.adapter.as_ref().unwrap().b.clone();
        let (ckpt_loss, ckpt_grads) = value_and_grad(
            |arrays| {
                checkpointed_module.adapter.as_mut().unwrap().a = arrays[0].clone();
                checkpointed_module.adapter.as_mut().unwrap().b = arrays[1].clone();
                unsafe {
                    checkpointed(&mut checkpointed_module, &arrays[2..], |m, ins| {
                        crate::compat::Module::forward(m, &ins[0])
                    })
                }
                .unwrap()
                .sum_all()
            },
            &[ckpt_a, ckpt_b],
            &[x],
        );

        assert!(
            (plain_loss.item_f32() - ckpt_loss.item_f32()).abs() < 1e-5,
            "checkpointed loss {} differs from plain {}",
            ckpt_loss.item_f32(),
            plain_loss.item_f32()
        );

        for (i, (plain, ckpt)) in plain_grads.iter().zip(ckpt_grads.iter()).enumerate() {
            let diff = plain.subtract(ckpt).abs().max(None).item_f32();
            assert!(
                diff < 1e-5,
                "adapter gradient {i} differs by {diff} under checkpointing"
            );
        }
    }

    #[test]
    fn the_adapter_actually_receives_a_gradient() {
        // Guards the failure this module exists to prevent: if the parameters
        // were captured instead of threaded, everything above would still agree
        // at zero.
        let x = input();
        let mut module = adapted_linear();
        let a = module.adapter.as_ref().unwrap().a.clone();
        let b = module.adapter.as_ref().unwrap().b.clone();

        let (_, grads) = value_and_grad(
            |arrays| {
                module.adapter.as_mut().unwrap().a = arrays[0].clone();
                module.adapter.as_mut().unwrap().b = arrays[1].clone();
                unsafe {
                    checkpointed(&mut module, &arrays[2..], |m, ins| {
                        crate::compat::Module::forward(m, &ins[0])
                    })
                }
                .unwrap()
                .sum_all()
            },
            &[a, b],
            &[x],
        );

        for (i, g) in grads.iter().enumerate() {
            let magnitude = g.abs().max(None).item_f32();
            assert!(
                magnitude > 0.0,
                "adapter gradient {i} came back all zeros, so the parameter was \
                 captured rather than threaded through checkpoint()"
            );
        }
    }

    #[test]
    fn a_module_with_nothing_to_train_still_runs() {
        let x = input();
        let mut frozen = LinearBuilder::new(4, 4).bias(false).build().unwrap();
        let expected = crate::compat::Module::forward(&mut frozen, &x).unwrap();

        // No adapter attached, and `AdaptedModel` freezes the base, so the
        // trainable set can legitimately be empty. That has to run, not error.
        let mut frozen_view = frozen;
        let got = unsafe {
            checkpointed(&mut frozen_view, &[x], |m, ins| {
                crate::compat::Module::forward(m, &ins[0])
            })
        }
        .unwrap();

        let diff = expected.subtract(&got).abs().max(None).item_f32();
        assert_eq!(diff, 0.0, "checkpointing changed an unadapted forward");
    }
}
