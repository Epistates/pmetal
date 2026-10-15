//! Custom Metal kernels (`MetalKernel`).

use pmetal_runtime::compat::Dtype;
use pmetal_runtime::{
    InlineArray, MetalKernel, TemplateArg, check_unobserved_error, expect_runtime_errors,
};

/// `out[i] = a[i] * SCALE + b[i]`, one thread per element.
const AXPY: &str = r#"
    uint i = thread_position_in_grid.x;
    if (i < a_shape[0]) {
        out[i] = static_cast<T>(static_cast<float>(a[i]) * SCALE + static_cast<float>(b[i]));
    }
"#;

fn axpy() -> MetalKernel {
    MetalKernel::new("runtime_test_axpy", &["a", "b"], &["out"], AXPY, "").unwrap()
}

#[test]
fn a_custom_kernel_computes_its_output_for_each_dtype_and_template_value() {
    let a = InlineArray::from_f32_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5]);
    let b = InlineArray::from_f32_slice(&[10.0, 20.0, 30.0, 40.0, 50.0], &[5]);
    for scale in [2, 3] {
        let out = axpy()
            .call(
                &[&a, &b],
                &[(&[5], Dtype::Float32)],
                [5, 1, 1],
                [5, 1, 1],
                &[
                    ("T", TemplateArg::Dtype(Dtype::Float32)),
                    ("SCALE", TemplateArg::Int(scale)),
                ],
            )
            .unwrap();
        let got = out[0].as_slice::<f32>().to_vec();
        let want: Vec<f32> = [1.0, 2.0, 3.0, 4.0, 5.0]
            .iter()
            .zip([10.0, 20.0, 30.0, 40.0, 50.0])
            .map(|(a, b)| a * scale as f32 + b)
            .collect();
        assert_eq!(got, want, "scale {scale}");
    }

    // Half-precision inputs and outputs go through the same source.
    let a16 = a.as_dtype(Dtype::Float16);
    let b16 = b.as_dtype(Dtype::Float16);
    let out = axpy()
        .call(
            &[&a16, &b16],
            &[(&[5], Dtype::Float16)],
            [5, 1, 1],
            [5, 1, 1],
            &[
                ("T", TemplateArg::Dtype(Dtype::Float16)),
                ("SCALE", TemplateArg::Int(2)),
            ],
        )
        .unwrap();
    assert_eq!(out[0].dtype(), Dtype::Float16);
    assert_eq!(
        out[0].as_dtype(Dtype::Float32).as_slice::<f32>(),
        [12.0, 24.0, 36.0, 48.0, 60.0]
    );
    check_unobserved_error().unwrap();
}

#[test]
fn a_kernel_is_lazy_and_composes_with_other_ops() {
    let a = InlineArray::from_f32_slice(&[1.0, 2.0, 3.0], &[3]);
    let b = InlineArray::from_f32_slice(&[0.0, 0.0, 0.0], &[3]);
    let out = axpy()
        .call(
            &[&a, &b],
            &[(&[3], Dtype::Float32)],
            [3, 1, 1],
            [3, 1, 1],
            &[
                ("T", TemplateArg::Dtype(Dtype::Float32)),
                ("SCALE", TemplateArg::Int(4)),
            ],
        )
        .unwrap();
    // The kernel's output feeds ordinary ops in the same graph.
    let total = out[0].add(&InlineArray::from_f32(1.0)).sum_all();
    assert_eq!(total.item_f32(), 4.0 * 6.0 + 3.0);
}

#[test]
fn a_kernel_with_a_syntax_error_reports_it() {
    let bad = MetalKernel::new(
        "runtime_test_bad",
        &["a"],
        &["out"],
        "this is not metal;",
        "",
    )
    .unwrap();
    let a = InlineArray::from_f32_slice(&[1.0], &[1]);
    expect_runtime_errors(|| {
        let outputs = bad
            .call(&[&a], &[(&[1], Dtype::Float32)], [1, 1, 1], [1, 1, 1], &[])
            .unwrap();
        // Compilation happens when the graph is evaluated.
        outputs[0].eval();
    });
    assert!(
        check_unobserved_error().is_err(),
        "a kernel that does not compile is an error"
    );
}

#[test]
fn argument_counts_are_checked() {
    let a = InlineArray::from_f32(1.0);
    let result = axpy().call(&[&a], &[(&[1], Dtype::Float32)], [1, 1, 1], [1, 1, 1], &[]);
    assert!(result.is_err());
}

#[test]
fn reusing_a_kernel_name_for_different_source_is_an_error() {
    let a = InlineArray::from_f32_slice(&[1.0], &[1]);
    let first = MetalKernel::new(
        "runtime_test_same_name",
        &["a"],
        &["out"],
        "out[0] = a[0];",
        "",
    )
    .unwrap();
    first
        .call(&[&a], &[(&[1], Dtype::Float32)], [1, 1, 1], [1, 1, 1], &[])
        .unwrap()[0]
        .eval();
    let second = MetalKernel::new(
        "runtime_test_same_name",
        &["a"],
        &["out"],
        "out[0] = a[0] + 1;",
        "",
    )
    .unwrap();
    let result = expect_runtime_errors(|| {
        second.call(&[&a], &[(&[1], Dtype::Float32)], [1, 1, 1], [1, 1, 1], &[])
    });
    assert!(result.is_err());
    let _ = check_unobserved_error();
}
