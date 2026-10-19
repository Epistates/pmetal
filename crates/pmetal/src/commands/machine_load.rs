//! How busy the machine was while something was measured.
//!
//! A benchmark number is a statement about the machine as well as the code: a
//! compile, a model download or another run in the background takes the same
//! memory bandwidth and GPU time, and the result comes out slower with no
//! sign of why. The benchmark commands read the load average when they start
//! and when they end, write both next to the numbers, and call the numbers
//! provisional when the machine was not quiet, so a figure copied from a
//! report says how much to trust it.

use serde::{Deserialize, Serialize};

/// The one-minute load average: how many threads were runnable, on average,
/// over the last minute. `None` where the system does not report it.
fn one_minute_load() -> Option<f64> {
    let mut load = [0.0f64; 3];
    // SAFETY: `getloadavg` writes at most the three doubles it is given room
    // for, and returns how many it wrote or -1.
    #[allow(unsafe_code)]
    let written = unsafe { libc::getloadavg(load.as_mut_ptr(), 3) };
    (written >= 1).then_some(load[0])
}

/// The logical CPUs of the machine.
fn logical_cpus() -> usize {
    std::thread::available_parallelism().map_or(1, usize::from)
}

/// The load average above which a machine is not quiet: a quarter of its
/// logical CPUs, and never less than two (a desktop is rarely at zero).
pub(crate) fn quiet_threshold(logical_cpus: usize) -> f64 {
    (logical_cpus as f64 / 4.0).max(2.0)
}

/// The machine's load around a measurement.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct MeasurementConditions {
    /// The one-minute load average when the measurement began and when it
    /// ended; `None` where the system does not report it.
    #[serde(default)]
    pub(crate) load_before: Option<f64>,
    #[serde(default)]
    pub(crate) load_after: Option<f64>,
    #[serde(default)]
    pub(crate) logical_cpus: usize,
    /// The load above which the machine is not quiet.
    #[serde(default)]
    pub(crate) quiet_below: f64,
    /// Whether the numbers are provisional: the machine was busy, or its load
    /// is not known.
    #[serde(default = "unknown_is_provisional")]
    pub(crate) provisional: bool,
}

fn unknown_is_provisional() -> bool {
    true
}

/// Conditions nobody recorded: unknown, so provisional.
impl Default for MeasurementConditions {
    fn default() -> Self {
        Self {
            load_before: None,
            load_after: None,
            logical_cpus: 0,
            quiet_below: 0.0,
            provisional: true,
        }
    }
}

impl MeasurementConditions {
    /// The conditions for the two readings.
    pub(crate) fn from_loads(before: Option<f64>, after: Option<f64>, logical_cpus: usize) -> Self {
        let quiet_below = quiet_threshold(logical_cpus);
        let quiet = |load: Option<f64>| load.is_some_and(|l| l <= quiet_below);
        Self {
            load_before: before,
            load_after: after,
            logical_cpus,
            quiet_below,
            provisional: !(quiet(before) && quiet(after)),
        }
    }

    /// How to say it in a report.
    pub(crate) fn describe(&self) -> String {
        let load = |l: Option<f64>| l.map_or("unknown".to_string(), |l| format!("{l:.2}"));
        let verdict = if self.provisional {
            format!(
                "PROVISIONAL: the machine was not quiet (load average above {:.1}) or its load \
                 is unknown; repeat on an idle machine before quoting these numbers",
                self.quiet_below
            )
        } else {
            "quiet".to_string()
        };
        format!(
            "load average {} before, {} after ({} logical CPUs): {verdict}",
            load(self.load_before),
            load(self.load_after),
            self.logical_cpus
        )
    }
}

/// Reads the load when a measurement starts, and again when it ends.
pub(crate) struct ConditionsWatch {
    before: Option<f64>,
}

impl ConditionsWatch {
    pub(crate) fn start() -> Self {
        Self {
            before: one_minute_load(),
        }
    }

    pub(crate) fn finish(self) -> MeasurementConditions {
        MeasurementConditions::from_loads(self.before, one_minute_load(), logical_cpus())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_are_provisional_unless_the_machine_was_quiet_throughout() {
        // 8 CPUs: quiet means a load of 2 or less.
        assert_eq!(quiet_threshold(8), 2.0);
        assert_eq!(quiet_threshold(16), 4.0);
        assert_eq!(quiet_threshold(2), 2.0);

        let quiet = MeasurementConditions::from_loads(Some(0.4), Some(1.9), 8);
        assert!(!quiet.provisional);
        assert!(quiet.describe().ends_with("quiet"), "{}", quiet.describe());

        // Busy at the start, busy at the end, or not known at all.
        assert!(MeasurementConditions::from_loads(Some(3.5), Some(1.0), 8).provisional);
        assert!(MeasurementConditions::from_loads(Some(1.0), Some(3.5), 8).provisional);
        let unknown = MeasurementConditions::from_loads(None, Some(1.0), 8);
        assert!(unknown.provisional);
        assert!(unknown.describe().contains("PROVISIONAL"));
        assert!(unknown.describe().contains("unknown before"));
    }

    #[test]
    fn a_report_without_conditions_still_reads() {
        let c: MeasurementConditions = serde_json::from_str("{}").unwrap();
        assert_eq!(c, MeasurementConditions::default());
        assert!(c.provisional);
    }

    #[test]
    fn this_machine_reports_a_load() {
        // Every Mac does; a platform without it would make every number
        // provisional, which is the honest reading.
        assert!(one_minute_load().is_some_and(|l| l >= 0.0));
    }
}
