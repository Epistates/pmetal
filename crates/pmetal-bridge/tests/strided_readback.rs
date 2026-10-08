//! Reading an MLX array back to Rust returns its logical elements in row-major
//! order, whatever its memory layout.
//!
//! MLX's transposes, inner-axis slices and broadcasts are views: they share
//! the source buffer and only change strides (a broadcast also has fewer
//! elements in the buffer than it has logically). `as_slice`, `data_ptr` and
//! `to_f32_vec` used to walk `size()` elements from the data pointer, so they
//! returned the source's elements in storage order, or read past the end of
//! the buffer. Every case here builds the expected values by hand from the
//! row-major definition of the view, never by reading another MLX array.

use pmetal_bridge::check_last_error;
use pmetal_bridge::compat::{Array, Dtype, nn};

/// `[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]` as f32.
fn grid() -> Array {
    let data: Vec<f32> = (0..12).map(|v| v as f32).collect();
    Array::from_f32_slice(&data, &[3, 4])
}

/// All three readback paths, each checked against `want`.
fn assert_reads(view: &Array, want: &[f32], what: &str) {
    assert_eq!(view.as_slice::<f32>(), want, "as_slice of {what}");

    let ptr = view.data_ptr() as *const f32;
    let via_ptr = unsafe { std::slice::from_raw_parts(ptr, view.size()) };
    assert_eq!(via_ptr, want, "data_ptr of {what}");

    let mut owned = view.clone();
    assert_eq!(
        owned.to_f32_vec(want.len()).expect("to_f32_vec"),
        want,
        "to_f32_vec of {what}"
    );
    check_last_error().expect("bridge error");
}

#[test]
fn a_transposed_view_reads_back_transposed() {
    let t = grid().t();
    t.eval();
    assert_eq!(t.shape(), vec![4, 3]);
    assert_reads(
        &t,
        &[0., 4., 8., 1., 5., 9., 2., 6., 10., 3., 7., 11.],
        "a 2-D transpose",
    );
}

#[test]
fn a_permuted_3d_view_reads_back_permuted() {
    // [2, 3, 2] -> axes (2, 0, 1) -> [2, 2, 3]; out[i][j][k] = in[j][k][i].
    let data: Vec<f32> = (0..12).map(|v| v as f32).collect();
    let src = Array::from_f32_slice(&data, &[2, 3, 2]);
    let p = src.transpose_axes(&[2, 0, 1]);
    let mut want = Vec::new();
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..3 {
                want.push(data[j * 6 + k * 2 + i]);
            }
        }
    }
    assert_reads(&p, &want, "a 3-D permutation");
}

#[test]
fn an_inner_axis_slice_reads_back_only_its_columns() {
    // Columns 1..3 of every row: strides stay [4, 1] but the shape is [3, 2].
    let s = grid().slice(&[0, 1], &[3, 3]);
    assert_eq!(s.shape(), vec![3, 2]);
    assert_reads(&s, &[1., 2., 5., 6., 9., 10.], "an inner-axis slice");
}

#[test]
fn a_row_range_slice_stays_zero_copy_and_reads_correctly() {
    // Rows 1..3 are row-contiguous at an offset into the source buffer.
    let src = grid();
    src.eval();
    let rows = src.slice(&[1, 0], &[3, 4]);
    assert_reads(
        &rows,
        &[4., 5., 6., 7., 8., 9., 10., 11.],
        "a row-range slice",
    );
    // Zero-copy: the view reads straight out of the source's buffer.
    let src_ptr = src.data_ptr() as usize;
    assert_eq!(rows.data_ptr() as usize, src_ptr + 4 * size_of::<f32>());
}

#[test]
fn a_broadcast_reads_back_every_repeat() {
    // Broadcasting [4] to [3, 4] keeps a 4-element buffer with stride 0 on
    // axis 0; reading 12 elements from it ran past the end of the buffer.
    let row = Array::from_f32_slice(&[1., 2., 3., 4.], &[4]);
    let b = row.broadcast_to(&[3, 4]);
    assert_reads(
        &b,
        &[1., 2., 3., 4., 1., 2., 3., 4., 1., 2., 3., 4.],
        "a row broadcast",
    );

    let col = Array::from_f32_slice(&[1., 2., 3.], &[3, 1]);
    let b = col.broadcast_to(&[3, 4]);
    assert_reads(
        &b,
        &[1., 1., 1., 1., 2., 2., 2., 2., 3., 3., 3., 3.],
        "a column broadcast",
    );
}

#[test]
fn a_scalar_broadcast_reads_back_every_element() {
    let b = Array::from_f32(7.0).broadcast_to(&[2, 3]);
    assert_reads(&b, &[7.0; 6], "a scalar broadcast");
}

#[test]
fn integer_views_read_back_in_row_major_order() {
    let data: Vec<i32> = (0..6).collect();
    let t = Array::from_i32_slice_shaped(&data, &[2, 3]).t();
    assert_eq!(t.as_slice::<i32>(), &[0, 3, 1, 4, 2, 5]);
    assert_eq!(t.as_slice::<u32>(), &[0, 3, 1, 4, 2, 5]);
}

#[test]
fn a_view_of_a_non_f32_array_reads_back_through_to_f32_vec() {
    // The astype to f32 in to_f32_vec keeps the input's strides, so a bf16
    // view came back in storage order too.
    let mut t = grid().as_dtype(Dtype::Bfloat16.as_i32()).t();
    assert_eq!(
        t.to_f32_vec(12).expect("to_f32_vec"),
        vec![0., 4., 8., 1., 5., 9., 2., 6., 10., 3., 7., 11.]
    );
    check_last_error().expect("bridge error");
}

#[test]
fn the_gradient_of_a_transposed_weight_reads_back_in_the_weight_layout() {
    // loss = sum(x @ w.T * c). d loss / d w = c.T @ x, and MLX builds it as
    // the transpose of the cotangent of `w.T`, so the gradient is a strided
    // view of a [3, 2] buffer that has the [2, 3] shape of `w`.
    let x = Array::from_f32_slice(&[1., 2., 3., 4., 5., 6., 7., 8., 9.], &[3, 3]);
    let c = Array::from_f32_slice(&[1., -1., 2., 0.5, 3., -2.], &[3, 2]);
    let w = Array::from_f32_slice(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 3]);
    let (_, grads) = nn::value_and_grad_explicit(
        |a: &[Array]| a[1].matmul(&a[0].t()).multiply(&a[2]).sum_all(),
        &[w],
        &[x, c],
    )
    .expect("value_and_grad");
    let g = &grads[0];
    assert_eq!(g.shape(), vec![2, 3]);

    // c.T @ x by hand: row r of the result is sum_i c[i][r] * x[i].
    let x_rows: [[f32; 3]; 3] = [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]];
    let c_rows: [[f32; 2]; 3] = [[1., -1.], [2., 0.5], [3., -2.]];
    let want: Vec<f32> = (0..2)
        .flat_map(|r| {
            (0..3).map(move |k| {
                x_rows
                    .iter()
                    .zip(&c_rows)
                    .map(|(x, c)| c[r] * x[k])
                    .sum::<f32>()
            })
        })
        .collect();
    assert_reads(g, &want, "the gradient of a transposed weight");
}

#[test]
fn reading_a_view_leaves_its_source_and_other_handles_intact() {
    let src = grid();
    let t = src.t();
    let alias = t.clone();
    assert_eq!(t.as_slice::<f32>()[..3], [0., 4., 8.]);
    // The packed buffer is swapped into the shared array, so a clone reads
    // the same logical values, and the source keeps its own layout.
    assert_eq!(alias.as_slice::<f32>()[..3], [0., 4., 8.]);
    let want: Vec<f32> = (0..12).map(|v| v as f32).collect();
    assert_eq!(src.as_slice::<f32>(), want.as_slice());
    // And the view still computes correctly as an op input afterwards.
    let mut sum_rows = t.sum_axis(1, false);
    assert_eq!(
        sum_rows.to_f32_vec(4).expect("to_f32_vec"),
        vec![12., 15., 18., 21.]
    );
    check_last_error().expect("bridge error");
}

#[test]
fn an_unevaluated_array_is_evaluated_by_as_slice() {
    let lazy = grid().multiply(&Array::from_f32(2.0)).t();
    assert_eq!(lazy.as_slice::<f32>()[..4], [0., 8., 16., 2.]);
}

#[test]
fn an_empty_array_reads_back_as_an_empty_slice() {
    let empty = grid().slice(&[0, 0], &[0, 4]);
    assert_eq!(empty.size(), 0);
    assert!(empty.as_slice::<f32>().is_empty());
    check_last_error().expect("bridge error");
}

#[test]
#[should_panic(expected = "as_slice::<f32>() on a Bfloat16 array")]
fn as_slice_refuses_a_dtype_that_does_not_match() {
    // Reading a 2-byte dtype as f32 read twice the buffer's length.
    let bf = grid().as_dtype(Dtype::Bfloat16.as_i32());
    let _ = bf.as_slice::<f32>();
}
