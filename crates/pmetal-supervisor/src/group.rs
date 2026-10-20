//! Signalling a job's process group.
//!
//! A job is started as the leader of its own group, so a signal to the group
//! reaches everything the job spawned (data loaders, helper processes) and not
//! only the process the supervisor holds. `rustix` makes the call without
//! `unsafe` and without a `kill(1)` child.

use std::path::PathBuf;

use rustix::process::{Pid, Signal, kill_process, kill_process_group};

/// Ask the group led by `pid` to finish.
pub(crate) fn terminate_group(pid: u32) -> bool {
    signal_group(pid, Signal::TERM)
}

/// End the group led by `pid` (SIGKILL) without asking. For a command the
/// caller runs itself; jobs are stopped through [`crate::Supervisor::stop`].
/// Whether anything was signalled.
pub fn kill_group(pid: u32) -> bool {
    signal_group(pid, Signal::KILL)
}

/// Send `signal` to the group led by `pid`; if there is no such group (the
/// child never formed one), to the process itself. Whether anything was
/// signalled.
fn signal_group(pid: u32, signal: Signal) -> bool {
    let Some(pid) = i32::try_from(pid).ok().and_then(Pid::from_raw) else {
        return false;
    };
    kill_process_group(pid, signal).is_ok() || kill_process(pid, signal).is_ok()
}

/// The `pmetal` binary jobs run.
///
/// Search order:
/// 1. `PMETAL_BIN` (explicit override)
/// 2. a `pmetal` beside the running executable
/// 3. `pmetal` on `PATH`
pub fn resolve_pmetal_binary() -> PathBuf {
    if let Some(bin) = std::env::var_os("PMETAL_BIN") {
        return PathBuf::from(bin);
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let candidate = dir.join("pmetal");
            if candidate.exists() {
                return candidate;
            }
        }
    }
    PathBuf::from("pmetal")
}
