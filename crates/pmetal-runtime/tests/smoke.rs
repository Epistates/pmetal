use pmetal_runtime::InlineArray;

#[test]
fn smoke_test_inline_ops() {
    let a = InlineArray::from_f32(2.0);
    let b = InlineArray::from_f32(3.0);
    let c = a.add(&b);
    let c = c;
    c.eval();
    assert_eq!(c.item_f32(), 5.0);
}

#[test]
fn smoke_test_matmul() {
    let a = InlineArray::ones(&[2, 3], 10); // dtype 10 = float32
    let b = InlineArray::ones(&[3, 4], 10);
    let c = a.matmul(&b);
    let c = c;
    c.eval();
    assert_eq!(c.dim(0), 2);
    assert_eq!(c.dim(1), 4);
}

#[test]
fn smoke_test_rms_norm() {
    let x = InlineArray::ones(&[1, 1, 4], 10);
    let w = InlineArray::ones(&[4], 10);
    let out = x.rms_norm(Some(&w), 1e-6);
    let out = out;
    out.eval();
    assert_eq!(out.dim(2), 4);
}
