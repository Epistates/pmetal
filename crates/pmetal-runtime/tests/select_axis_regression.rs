use pmetal_runtime::compat::{Array, ops};

#[test]
fn select_axis_scalar_index_returns_axis_removed_slice() {
    let data: Vec<i32> = (0..12).collect();
    let arr = Array::from_i32_slice_shaped(&data, &[3, 4]);

    let row = ops::select_axis(&arr, 1, 0);
    row.eval();
    assert_eq!(row.shape(), vec![4]);
    assert_eq!(row.as_slice::<i32>(), &[4, 5, 6, 7]);

    let col = ops::select_axis(&arr, 2, 1);
    col.eval();
    assert_eq!(col.shape(), vec![3]);
    assert_eq!(col.as_slice::<i32>(), &[2, 6, 10]);
}

#[test]
fn integer_index_uses_take_axis_without_invalid_squeeze() {
    let data: Vec<i32> = (0..12).collect();
    let arr = Array::from_i32_slice_shaped(&data, &[3, 4]);

    let row = arr.index(2_i32);
    row.eval();
    assert_eq!(row.shape(), vec![4]);
    assert_eq!(row.as_slice::<i32>(), &[8, 9, 10, 11]);
}
