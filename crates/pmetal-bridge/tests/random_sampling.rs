//! `compat::random` draws along the axis it is given.

use pmetal_bridge::check_unobserved_error;
use pmetal_bridge::compat::{Array, random};

/// Logits with the classes on axis 0, one row per column: every class but
/// `want[col]` is impossible, so any draw must return `want`.
fn classes_on_axis0(want: &[usize], n_classes: usize) -> Array {
    let cols = want.len();
    let mut v = vec![f32::NEG_INFINITY; n_classes * cols];
    for (col, &class) in want.iter().enumerate() {
        v[class * cols + col] = 0.0;
    }
    Array::from_f32_slice(&v, &[n_classes as i32, cols as i32])
}

fn indices(a: &Array) -> Vec<u32> {
    a.as_slice::<u32>().to_vec()
}

#[test]
fn categorical_reduces_the_axis_it_is_given() {
    let want = [2usize, 0, 3];
    let logits = classes_on_axis0(&want, 4);

    let drawn = random::categorical(&logits, 0);
    assert_eq!(drawn.shape(), &[3], "axis 0 of a [4, 3] array is reduced");
    assert_eq!(indices(&drawn), [2, 0, 3]);

    // The same logits transposed put the classes last; -1 and 1 agree.
    let last = logits.transpose_axes(&[1, 0]);
    assert_eq!(indices(&random::categorical(&last, -1)), [2, 0, 3]);
    assert_eq!(indices(&random::categorical(&last, 1)), [2, 0, 3]);
    check_unobserved_error().expect("no bridge op failed");
}
