//! A float64 yardstick for the numerical tests: deterministic data, the
//! comparison of values, and the comparison of autograd's gradient with
//! central finite differences of a float64 evaluation.
//!
//! A kernel is checked against an independent float64 implementation of what
//! it computes (written in plain loops in the test), not against another f32
//! path that may share its mistakes.

use pmetal_runtime::compat::nn::value_and_grad_explicit;
use pmetal_runtime::compat::{Array, Dtype};

// Deterministic data

pub(crate) struct Rng(pub(crate) u64);

impl Rng {
    pub(crate) fn next_u64(&mut self) -> u64 {
        // splitmix64
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub(crate) fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    pub(crate) fn normal(&mut self) -> f64 {
        let (u1, u2) = (self.uniform().max(1e-12), self.uniform());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Round to f32 and back, so the f64 reference sees exactly what the f32 paths
/// are given.
pub(crate) fn round32(x: f64) -> f64 {
    x as f32 as f64
}

pub(crate) fn arr(data: &[f64], shape: &[usize]) -> Array {
    let f: Vec<f32> = data.iter().map(|&x| x as f32).collect();
    let shape: Vec<i32> = shape.iter().map(|&d| d as i32).collect();
    Array::from_f32_slice(&f, &shape)
}

pub(crate) fn vec_of(a: &Array) -> Vec<f64> {
    let mut a = a.as_dtype(Dtype::Float32.as_i32());
    a.eval();
    let n = a.size();
    a.to_f32_vec(n)
        .expect("readable")
        .into_iter()
        .map(f64::from)
        .collect()
}

pub(crate) fn drain() {
    if let Err(e) = pmetal_runtime::check_unobserved_error() {
        panic!("a runtime op threw: {e}");
    }
}

pub(crate) fn max_abs(x: &[f64]) -> f64 {
    x.iter().fold(0.0, |m, v| m.max(v.abs()))
}

pub(crate) fn assert_close(what: &str, got: &[f64], want: &[f64], rel: f64) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let scale = max_abs(want).max(1e-3);
    let worst = got
        .iter()
        .zip(want)
        .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
    assert!(
        worst <= rel * scale,
        "{what}: largest difference {worst:e} against a scale of {scale:e} (tolerance {rel:e} relative)"
    );
}

/// Autograd's gradient of `loss32` with respect to every input, at coordinates
/// sampled from each (the largest-gradient ones and random ones), against
/// central differences of the float64 `loss64`.
///
/// A coordinate passes when the two agree to `rel` of the input's largest
/// gradient entry. Every problem is reported, not the first, so a failure
/// shows which inputs are wrong.
pub(crate) fn check_gradient(
    what: &str,
    names: &[&str],
    bases: &[Vec<f64>],
    shapes: &[Vec<usize>],
    loss32: impl FnMut(&[Array]) -> Array,
    loss64: impl Fn(&[Vec<f64>]) -> f64,
    rel: f64,
) {
    const RANDOM_COORDS: usize = 12;
    const LARGEST_COORDS: usize = 4;
    let mut rng = Rng(0xD1CE);
    let inputs: Vec<Array> = bases.iter().zip(shapes).map(|(b, s)| arr(b, s)).collect();
    let (_, grads) = value_and_grad_explicit(loss32, &inputs, &[]).unwrap();
    drain();

    let mut problems = Vec::new();
    for (i, name) in names.iter().enumerate() {
        let g = vec_of(&grads[i]);
        let scale = max_abs(&g).max(1e-3);
        let mut coords: Vec<usize> = (0..RANDOM_COORDS)
            .map(|_| (rng.next_u64() % g.len() as u64) as usize)
            .collect();
        let mut by_size: Vec<usize> = (0..g.len()).collect();
        by_size.sort_by(|&a, &b| g[b].abs().total_cmp(&g[a].abs()));
        coords.extend(by_size.into_iter().take(LARGEST_COORDS));

        let mut worst = 0.0_f64;
        let mut worst_at = None;
        let mut probe = bases.to_vec();
        for j in coords {
            let base = probe[i][j];
            let h = 1e-6;
            probe[i][j] = base + h;
            let up = loss64(&probe);
            probe[i][j] = base - h;
            let down = loss64(&probe);
            probe[i][j] = base;
            let numeric = (up - down) / (2.0 * h);
            let gap = (g[j] - numeric).abs();
            if gap > worst {
                worst = gap;
                worst_at = Some((j, g[j], numeric));
            }
        }
        if let Some((j, got, want)) = worst_at
            && worst > rel * scale
        {
            problems.push(format!(
                "d/d{name}[{j}]: autograd {got:e} vs float64 finite difference {want:e} \
                 (off by {worst:e} against a gradient scale of {scale:e})"
            ));
        }
    }
    assert!(problems.is_empty(), "{what}:\n{}", problems.join("\n"));
}

pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
