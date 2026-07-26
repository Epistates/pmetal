// Re-export layer types so `use pmetal_bridge::compat::nn` works
// as a drop-in for `use mlx_rs::nn`.
pub use super::layers::{
    Conv1d, Conv1dBuilder, Conv2d, Conv2dBuilder, Embedding, GroupNorm, GroupNormBuilder,
    LayerNorm, LayerNormBuilder, Linear, LinearBuilder, RmsNorm, RmsNormBuilder, Rope, RopeBuilder,
    Sequential,
};

use super::{Array, Exception};

pub fn softplus(a: &Array) -> Array {
    a.softplus()
}
pub fn sigmoid(a: &Array) -> Array {
    a.sigmoid()
}
pub fn relu(a: &Array) -> Array {
    a.relu()
}
/// GELU "fast approximation" using `x * sigmoid(1.702 * x)`. This is
/// the same formula used by mlx-rs's `nn::gelu_fast_approx` and by the
/// `mlx_inline_gelu` kernel in `bridge_training.cpp`. This is NOT the
/// tanh approximation (`mx.fast.rms_norm(x, None, eps)` style) — for
/// that see `gelu_tanh_approximate` below.
pub fn gelu(a: &Array) -> Array {
    a.gelu()
}
/// Alias of [`gelu`] kept for source compatibility with mlx-rs's
/// `nn::gelu_approximate`, which was also the sigmoid fast-approx (and
/// not the tanh approximation, despite the name). Prefer calling
/// [`gelu_tanh_approximate`] when you need the tanh variant — see the
/// Gemma 4 parity investigation for the drift this discrepancy caused.
pub fn gelu_approximate(a: &Array) -> Array {
    a.gelu()
}
/// GELU tanh approximation matching mlx-lm's `nn.gelu_approx`:
///
/// ```text
///     0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
/// ```
///
/// Implemented in pure ops since there's no dedicated bridge kernel.
pub fn gelu_tanh_approximate(a: &Array) -> Array {
    use super::ops;
    let k = Array::from_f32(0.797_884_6); // sqrt(2/pi)
    let c = Array::from_f32(0.044_715);
    let half = Array::from_f32(0.5);
    let one = Array::from_f32(1.0);
    let x2 = a.multiply(a);
    let x3 = x2.multiply(a);
    let inner = a.add(&c.multiply(&x3));
    let scaled = k.multiply(&inner);
    let t = ops::tanh(&scaled);
    half.multiply(a).multiply(&one.add(&t))
}
/// **Exact** GELU — the erf definition, and the one HuggingFace's
/// `ACT2FN["gelu"]` (`GELUActivation`) computes:
///
/// ```text
///     0.5 * x * (1 + erf(x / √2))
/// ```
///
/// Use this whenever a reference config says `hidden_act: "gelu"`. Neither
/// [`gelu`] (sigmoid fast-approx, ~1e-2 off) nor [`gelu_tanh_approximate`]
/// (~1e-3 off) is a parity-grade substitute; those correspond to
/// `"gelu_pytorch_tanh"` / `"quick_gelu"`-style activations, which real configs
/// name explicitly.
pub fn gelu_erf(a: &Array) -> Array {
    let half = Array::from_f32(0.5);
    let one = Array::from_f32(1.0);
    let inv_sqrt2 = Array::from_f32(std::f32::consts::FRAC_1_SQRT_2);
    half.multiply(a)
        .multiply(&one.add(&a.multiply(&inv_sqrt2).erf()))
}
pub fn silu(a: &Array) -> Array {
    a.silu()
}
/// Log-sigmoid: `log(sigmoid(x)) = -softplus(-x)`.
pub fn log_sigmoid(a: &Array) -> Array {
    a.negative().softplus().negative()
}
pub fn log_softmax(a: &Array, axis: i32) -> Array {
    a.log_softmax(axis)
}
pub fn softmax(a: &Array, axis: i32) -> Array {
    a.softmax(axis)
}
pub fn leaky_relu(a: &Array, neg_slope: f32) -> Array {
    a.leaky_relu(neg_slope)
}
pub fn cross_entropy(logits: &Array, targets: &Array, axis: i32) -> Array {
    logits.cross_entropy(targets, axis)
}

/// Compute `(loss, gradients)` via callback-based autograd — explicit-array form.
///
/// `loss_fn` receives `[params..., inputs...]` as a flat slice and must
/// return a scalar loss array.  Gradients are computed w.r.t. the first
/// `params.len()` arrays.
///
/// This is a thin shim over the bridge `value_and_grad` function; the
/// `Result` wrapper is present only for API parity with `mlx_rs`.
pub fn value_and_grad_explicit<F>(
    loss_fn: F,
    params: &[Array],
    inputs: &[Array],
) -> Result<(Array, Vec<Array>), Exception>
where
    F: FnMut(&[Array]) -> Array,
{
    Ok(crate::inline_array::value_and_grad(loss_fn, params, inputs))
}

/// mlx-rs compatible closure-returning form of `value_and_grad`.
///
/// Mirrors the mlx-rs API:
/// ```ignore
/// let mut vag = nn::value_and_grad(loss_fn);
/// let (loss, grads) = vag(model, inputs)?;
/// ```
///
/// - `loss_fn` takes `(&mut M, T)` and returns `Result<Array, Exception>`.
/// - The returned closure accepts `(&mut M, T)` and returns
///   `Result<(Array, FlattenedModuleParam), Exception>`.
///
/// Trainable parameters are extracted from `M` before autograd, the
/// bridge computes gradients, and the result is re-keyed into a
/// `FlattenedModuleParam` with the same names.
pub fn value_and_grad<M, T, F>(
    mut loss_fn: F,
) -> impl FnMut(&mut M, T) -> Result<(Array, super::FlattenedModuleParam), super::Exception>
where
    M: super::ModuleParameters,
    F: FnMut(&mut M, T) -> Result<Array, super::Exception>,
{
    move |model: &mut M, inputs: T| {
        use super::FlattenedModuleParam;
        use std::rc::Rc;

        // 1. Snapshot trainable parameter values — stable key order.
        let flat: FlattenedModuleParam = {
            let tree = model.trainable_parameters();
            let mut out = FlattenedModuleParam::new();
            super::flatten_nested_ref_owned(&tree, "", &mut out);
            out
        };
        let keys: Vec<Rc<str>> = flat.keys().cloned().collect();
        let param_arrays: Vec<Array> = keys.iter().map(|k| flat[k].clone()).collect();
        let n_params = param_arrays.len();

        // 2. Wrap inputs in an Option so the inner closure can move them out
        //    exactly once (MLX calls the callback once per value_and_grad call).
        let mut inputs_slot: Option<T> = Some(inputs);

        // SAFETY: both `model` and `loss_fn` outlive the `flat_loss` closure —
        // they all live on the same call frame.  `flat_loss` is consumed
        // synchronously by `value_and_grad` before this function returns.
        let model_ptr: *mut M = model as *mut M;
        let loss_fn_ptr: *mut F = &mut loss_fn as *mut F;
        let keys_snap: Vec<Rc<str>> = keys.clone();

        let flat_loss = move |all_arrays: &[Array]| -> Array {
            let model_mut = unsafe { &mut *model_ptr };
            let loss_fn_mut = unsafe { &mut *loss_fn_ptr };

            // Update the model's trainable params with the autograd arrays.
            {
                let mut pm = model_mut.parameters_mut();
                for (key, arr) in keys_snap.iter().zip(all_arrays[..n_params].iter()) {
                    super::update_trainable_param(&mut pm, key, arr.clone());
                }
            }

            // Consume inputs from the slot (called exactly once by bridge).
            let inp = inputs_slot
                .take()
                .expect("value_and_grad callback called more than once");
            match loss_fn_mut(model_mut, inp) {
                Ok(loss) => loss,
                Err(_) => Array::from_f32(f32::NAN),
            }
        };

        // 3. Run bridge autograd (no extra "input" arrays — all captured).
        let (loss, grad_arrays) =
            crate::inline_array::value_and_grad(flat_loss, &param_arrays, &[]);

        // 4. Re-key gradients into FlattenedModuleParam.
        let grads: FlattenedModuleParam = keys.into_iter().zip(grad_arrays).collect();

        Ok((loss, grads))
    }
}

/// mlx-rs compatible `keyed_value_and_grad`.
///
/// Takes a closure `loss_fn(params: FlattenedModuleParam, inputs: T) -> Result<Vec<Array>>`
/// and returns a closure that computes `(values, grad_map)` via autograd over
/// the flattened param map.
///
/// The returned closure signature:
/// ```ignore
/// let mut vg = keyed_value_and_grad(loss_fn);
/// let (values, grads_map) = vg(params, inputs)?;
/// ```
pub fn keyed_value_and_grad<T, F>(
    mut loss_fn: F,
) -> impl FnMut(
    super::FlattenedModuleParam,
    T,
) -> Result<(Vec<super::Array>, super::FlattenedModuleParam), super::Exception>
where
    T: 'static,
    F: FnMut(super::FlattenedModuleParam, T) -> Result<Vec<super::Array>, super::Exception>,
{
    move |params: super::FlattenedModuleParam, inputs: T| {
        use super::FlattenedModuleParam;
        use std::rc::Rc;

        // Stable key order.
        let keys: Vec<Rc<str>> = params.keys().cloned().collect();
        let param_arrays: Vec<super::Array> = keys.iter().map(|k| params[k].clone()).collect();
        let n_params = param_arrays.len();

        let mut inputs_slot: Option<T> = Some(inputs);

        // SAFETY: params_ptr and loss_fn_ptr live for the duration of the
        // synchronous call to crate::inline_array::value_and_grad.
        let loss_fn_ptr: *mut F = &mut loss_fn as *mut F;
        let keys_snap: Vec<Rc<str>> = keys.clone();

        let flat_loss = move |all_arrays: &[super::Array]| -> super::Array {
            let loss_fn_mut = unsafe { &mut *loss_fn_ptr };

            // Re-build keyed param map from autograd arrays.
            let param_map: FlattenedModuleParam = keys_snap
                .iter()
                .cloned()
                .zip(all_arrays[..n_params].iter().cloned())
                .collect();

            let inp = inputs_slot
                .take()
                .expect("keyed_value_and_grad callback called more than once");

            match loss_fn_mut(param_map, inp) {
                Ok(mut vals) => {
                    // If there are multiple values, reduce to first (loss).
                    vals.drain(..)
                        .next()
                        .unwrap_or_else(|| super::Array::from_f32(0.0))
                }
                Err(_) => super::Array::from_f32(f32::NAN),
            }
        };

        // Bridge autograd: gradients w.r.t. param_arrays.
        let (loss_val, grad_arrays) =
            crate::inline_array::value_and_grad(flat_loss, &param_arrays, &[]);

        // Re-key gradients.
        let grads: FlattenedModuleParam = keys.into_iter().zip(grad_arrays).collect();

        Ok((vec![loss_val], grads))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `0.5·x·(1 + erf(x/√2))` at f64 precision, from
    /// `0.5*x*(1+math.erf(x/math.sqrt(2)))`.
    const GELU_ERF_TABLE: [(f32, f32); 10] = [
        (-4.0, -0.000_126_685),
        (-2.0, -0.045_500_264),
        (-1.0, -0.158_655_254),
        (-0.5, -0.154_268_769),
        (0.0, 0.0),
        (0.5, 0.345_731_231),
        (1.0, 0.841_344_746),
        (2.0, 1.954_499_736),
        (4.0, 3.999_873_315),
        (7.0, 7.0),
    ];

    #[test]
    fn gelu_erf_matches_the_exact_definition() {
        let xs: Vec<f32> = GELU_ERF_TABLE.iter().map(|&(x, _)| x).collect();
        let n = xs.len();
        let mut got = gelu_erf(&Array::from_f32_slice(&xs, &[n as i32]));
        got.eval();
        let got = got.to_f32_vec(n).expect("to_f32_vec");

        for (i, &(x, want)) in GELU_ERF_TABLE.iter().enumerate() {
            assert!(
                (got[i] - want).abs() < 1e-6,
                "gelu_erf({x}) = {}, want {want}",
                got[i]
            );
        }
    }

    /// The reason [`gelu_erf`] exists: the two approximations are *not*
    /// interchangeable with it at parity tolerances. `gelu` (sigmoid) is ~1.9e-2
    /// off at x = -2 and `gelu_tanh_approximate` ~1.5e-4 off at x = -1 — both
    /// orders of magnitude above the 1e-5 band a forward-pass parity test runs
    /// at. If this ever starts failing, the bridge's `gelu` changed meaning and
    /// every `hidden_act` mapping needs re-checking.
    #[test]
    fn the_gelu_approximations_are_not_parity_substitutes() {
        let xs: Vec<f32> = GELU_ERF_TABLE.iter().map(|&(x, _)| x).collect();
        let n = xs.len();
        let x = Array::from_f32_slice(&xs, &[n as i32]);

        for (name, mut approx, min_gap) in [
            ("gelu (sigmoid)", gelu(&x), 1e-2_f32),
            ("gelu_tanh_approximate", gelu_tanh_approximate(&x), 1e-4),
        ] {
            approx.eval();
            let approx = approx.to_f32_vec(n).expect("to_f32_vec");
            let worst = GELU_ERF_TABLE
                .iter()
                .zip(&approx)
                .map(|(&(_, want), &got)| (got - want).abs())
                .fold(0.0_f32, f32::max);
            assert!(
                worst > min_gap,
                "{name} is within {worst:.3e} of exact gelu — closer than the \
                 documented {min_gap:.0e}; the doc comments on these functions \
                 are now misleading"
            );
        }
    }
}
