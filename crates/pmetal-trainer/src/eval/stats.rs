//! The statistics an evaluation reports.

use serde::Serialize;

/// One reported number and the standard error of its mean.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Metric {
    /// The metric's value: the mean over the scored items.
    pub value: f64,
    /// The standard error of that mean: the sample standard deviation (the
    /// `n - 1` form) over `sqrt(n)`. `None` for a single item, which has none.
    pub stderr: Option<f64>,
}

impl Metric {
    /// The mean of `values` and its standard error.
    pub fn of(values: &[f64]) -> Self {
        Self {
            value: mean(values),
            stderr: mean_stderr(values),
        }
    }

    /// A metric with no standard error to report (a ratio of totals).
    pub fn exact(value: f64) -> Self {
        Self {
            value,
            stderr: None,
        }
    }
}

/// The arithmetic mean; zero for no values.
pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// The standard error of the mean of `values`: the sample standard deviation
/// over `sqrt(n)`. `None` below two values.
pub fn mean_stderr(values: &[f64]) -> Option<f64> {
    let n = values.len();
    if n < 2 {
        return None;
    }
    let m = mean(values);
    let variance = values.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (n - 1) as f64;
    Some((variance / n as f64).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_mean_and_its_standard_error() {
        let accuracy = [1.0, 0.0, 1.0, 1.0];
        let metric = Metric::of(&accuracy);
        assert_eq!(metric.value, 0.75);
        // Sample variance 0.25, over 4 items: sqrt(0.25 / 4).
        assert!((metric.stderr.unwrap() - 0.25).abs() < 1e-12);
        assert_eq!(Metric::of(&[1.0]).stderr, None);
        assert_eq!(Metric::of(&[]).value, 0.0);
        // Every item right: nothing varies.
        assert_eq!(Metric::of(&[1.0; 10]).stderr, Some(0.0));
    }
}
