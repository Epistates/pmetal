//! A bf16 model must stay bf16 through the compat layer: a scalar built as
//! an f32 array promotes whatever it touches, and MLX carries that promotion
//! through every op downstream, so one stray constant makes the rest of the
//! forward run in f32 without any error.

use pmetal_runtime::check_unobserved_error;
use pmetal_runtime::compat::{Array, Dtype, random};
use pmetal_runtime::compat::{Linear, Module};

fn bf16(shape: &[i32]) -> Array {
    random::uniform_range(-1.0, 1.0, shape, Dtype::Bfloat16)
}

fn eval(a: &Array) -> Dtype {
    a.eval();
    check_unobserved_error().expect("bridge error");
    a.dtype()
}

#[test]
fn an_adapted_bf16_layer_returns_bf16() {
    for dora in [false, true] {
        let mut layer = Linear::new(64, 32, true).expect("layer");
        layer.weight.value = layer.weight.value.as_dtype(Dtype::Bfloat16.as_i32());
        layer.bias.value = layer
            .bias
            .value
            .as_ref()
            .map(|b| b.as_dtype(Dtype::Bfloat16.as_i32()));
        let weight = layer.weight.value.clone();
        {
            let adapter = layer.attach_lora(8, 16.0, false).expect("attach");
            adapter.b = random::uniform_range(-0.1, 0.1, &[32, 8], Dtype::Float32);
            if dora {
                adapter.set_dora(&weight);
            }
        }
        let y = Module::forward(&mut layer, &bf16(&[2, 5, 64])).expect("forward");
        assert_eq!(eval(&y), Dtype::Bfloat16, "dora={dora}");
    }
}

#[test]
fn adapter_dropout_keeps_the_activations_dtype() {
    let mut layer = Linear::new(64, 32, false).expect("layer");
    layer.weight.value = layer.weight.value.as_dtype(Dtype::Bfloat16.as_i32());
    {
        let adapter = layer.attach_lora(8, 16.0, false).expect("attach");
        adapter.b = random::uniform_range(-0.1, 0.1, &[32, 8], Dtype::Float32);
        adapter.dropout = 0.1;
        adapter.training = true;
    }
    let y = Module::forward(&mut layer, &bf16(&[2, 5, 64])).expect("forward");
    assert_eq!(eval(&y), Dtype::Bfloat16);
}
