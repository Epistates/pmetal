//! `compat::random` draws along the axis it is given, and a keyed draw is a
//! function of its key alone.

use pmetal_bridge::check_unobserved_error;
use pmetal_bridge::compat::{Array, random};
use pmetal_bridge::decode::{SamplingParams, sample_token};

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

    let drawn = random::categorical(&logits, 0, None);
    assert_eq!(drawn.shape(), &[3], "axis 0 of a [4, 3] array is reduced");
    assert_eq!(indices(&drawn), [2, 0, 3]);

    // The same logits transposed put the classes last; -1 and 1 agree.
    let last = logits.transpose_axes(&[1, 0]);
    assert_eq!(indices(&random::categorical(&last, -1, None)), [2, 0, 3]);
    assert_eq!(indices(&random::categorical(&last, 1, None)), [2, 0, 3]);
    check_unobserved_error().expect("no bridge op failed");
}

/// 64 rows of a uniform distribution over 1000 classes.
fn uniform_rows() -> Array {
    Array::from_f32_slice(&vec![0.0f32; 64 * 1000], &[64, 1000])
}

#[test]
fn a_keyed_draw_depends_on_the_key_and_nothing_else() {
    let logits = uniform_rows();
    let key = random::key(7);
    let first = indices(&random::categorical(&logits, -1, Some(&key)));
    // Unkeyed draws in between advance the thread's key, not this one.
    random::seed(123);
    let _ = indices(&random::categorical(&logits, -1, None));
    let again = indices(&random::categorical(&logits, -1, Some(&key)));
    assert_eq!(first, again, "the same key drew different classes");

    let other = indices(&random::categorical(&logits, -1, Some(&random::key(8))));
    assert_ne!(
        first, other,
        "two keys drew the same 64 classes out of 1000"
    );
    check_unobserved_error().expect("no bridge op failed");
}

/// Keyed uniform and normal draws are functions of their key alone, and
/// leave the thread's key where it was.
#[test]
fn keyed_uniform_and_normal_draws_depend_on_the_key_alone() {
    use pmetal_bridge::compat::Dtype;
    let floats = |a: Array| a.as_slice::<f32>().to_vec();
    for draw in [random::uniform_keyed, random::normal_keyed] {
        let key = random::key(7);
        random::seed(1);
        let first = floats(draw(&[256], Dtype::Float32, Some(&key)));
        let thread_after_first = floats(random::uniform(&[8], Dtype::Float32));
        random::seed(99);
        let _ = floats(random::uniform(&[8], Dtype::Float32));
        let again = floats(draw(&[256], Dtype::Float32, Some(&key)));
        assert_eq!(first, again, "the same key drew different values");

        // The keyed draw did not move the thread's key.
        random::seed(1);
        assert_eq!(
            floats(random::uniform(&[8], Dtype::Float32)),
            thread_after_first,
            "a keyed draw moved the thread's key"
        );

        let other = floats(draw(&[256], Dtype::Float32, Some(&random::key(8))));
        assert_ne!(first, other, "two keys drew the same values");
        // Without a key the draw takes the thread's key.
        random::seed(5);
        let unkeyed = floats(draw(&[256], Dtype::Float32, None));
        random::seed(5);
        assert_eq!(unkeyed, floats(draw(&[256], Dtype::Float32, None)));
    }
    check_unobserved_error().expect("no bridge op failed");
}

#[test]
fn a_seeded_decode_draw_is_the_same_on_any_thread() {
    let params = SamplingParams {
        seed: Some(42),
        ..SamplingParams::new(0.8)
    };
    let draws = |params: &SamplingParams| -> Vec<Vec<u32>> {
        let logits = uniform_rows();
        (0..4u64)
            .map(|draw| indices(&sample_token(&logits, params, draw)))
            .collect()
    };
    let here = draws(&params);
    let p = params.clone();
    let there = std::thread::spawn(move || {
        // A fresh thread starts from its own default key.
        random::seed(999);
        draws(&p)
    })
    .join()
    .unwrap();
    assert_eq!(here, there, "a seeded generation depends on its thread");
    assert_ne!(here[0], here[1], "draws 0 and 1 share a key");

    let other = draws(&SamplingParams {
        seed: Some(43),
        ..params.clone()
    });
    assert_ne!(here, other, "seeds 42 and 43 drew the same tokens");
    check_unobserved_error().expect("no bridge op failed");
}
