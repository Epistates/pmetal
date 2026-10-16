//! Steering a running training job.
//!
//! A job that adapts its learning rate polls a control file in its output
//! directory. Whoever steers it (the TUI, the GUI, the MCP server) writes one
//! [`LrControlCommand`] there; the job claims the file by renaming it, applies
//! the command and deletes it. This module is the one definition of the command
//! and of how it is written.

use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// The control file's name, in a job's output directory.
pub const LR_CONTROL_FILE: &str = ".lr_control.json";

/// Where `output_dir`'s job looks for a control command.
pub fn lr_control_path(output_dir: &Path) -> PathBuf {
    output_dir.join(LR_CONTROL_FILE)
}

/// A command to a running training job.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum LrControlCommand {
    /// Set the learning rate to an absolute value.
    SetLr {
        /// The new learning rate.
        value: f64,
    },
    /// Multiply the learning rate by a factor.
    ReduceLr {
        /// The factor; the job clamps it to 0.01..=1.
        factor: f64,
    },
    /// Return to the scheduled learning rate, clearing every adjustment.
    ResetLr,
    /// Save a checkpoint now; training continues.
    SaveCheckpoint,
    /// Stop cleanly: restore the best weights, save a checkpoint, exit.
    GracefulStop,
}

/// Write `command` for the job whose output directory is `output_dir`.
///
/// The file appears whole or not at all, so the job never reads half a command.
/// A job reads one command at a time, so a command written while the last is
/// still unread would replace it; that is refused instead, with
/// [`io::ErrorKind::AlreadyExists`], and the caller tries again shortly.
pub fn write_lr_control(output_dir: &Path, command: &LrControlCommand) -> io::Result<()> {
    let json =
        serde_json::to_vec(command).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let target = lr_control_path(output_dir);
    // Written under a name of its own, then linked into place: the link fails if
    // the target exists, which is the "is the last command still unread" check.
    let staging = output_dir.join(format!(
        "{LR_CONTROL_FILE}.{}.{}.tmp",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos())
    ));
    std::fs::write(&staging, &json)?;
    let linked = std::fs::hard_link(&staging, &target);
    let _ = std::fs::remove_file(&staging);
    linked
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("pmetal-control-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn the_wire_format_is_the_tagged_snake_case_form_jobs_read() {
        for (command, json) in [
            (
                LrControlCommand::SetLr { value: 1e-5 },
                r#"{"action":"set_lr","value":1e-5}"#,
            ),
            (
                LrControlCommand::ReduceLr { factor: 0.5 },
                r#"{"action":"reduce_lr","factor":0.5}"#,
            ),
            (LrControlCommand::ResetLr, r#"{"action":"reset_lr"}"#),
            (
                LrControlCommand::SaveCheckpoint,
                r#"{"action":"save_checkpoint"}"#,
            ),
            (
                LrControlCommand::GracefulStop,
                r#"{"action":"graceful_stop"}"#,
            ),
        ] {
            assert_eq!(
                serde_json::from_str::<LrControlCommand>(json).unwrap(),
                command
            );
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(
                    &serde_json::to_string(&command).unwrap()
                )
                .unwrap(),
                serde_json::from_str::<serde_json::Value>(json).unwrap()
            );
        }
    }

    #[test]
    fn a_command_is_written_whole_and_leaves_nothing_behind() {
        let dir = scratch("whole");
        write_lr_control(&dir, &LrControlCommand::SetLr { value: 0.001 }).unwrap();
        let read: LrControlCommand =
            serde_json::from_slice(&std::fs::read(lr_control_path(&dir)).unwrap()).unwrap();
        assert_eq!(read, LrControlCommand::SetLr { value: 0.001 });
        let leftovers: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n != LR_CONTROL_FILE)
            .collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");
    }

    #[test]
    fn a_second_command_is_refused_until_the_first_is_read() {
        let dir = scratch("pending");
        write_lr_control(&dir, &LrControlCommand::SaveCheckpoint).unwrap();
        let err = write_lr_control(&dir, &LrControlCommand::GracefulStop).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::AlreadyExists);
        // The first command is untouched.
        let read: LrControlCommand =
            serde_json::from_slice(&std::fs::read(lr_control_path(&dir)).unwrap()).unwrap();
        assert_eq!(read, LrControlCommand::SaveCheckpoint);
        // The job claims it; the next one goes through.
        std::fs::remove_file(lr_control_path(&dir)).unwrap();
        write_lr_control(&dir, &LrControlCommand::GracefulStop).unwrap();
    }
}
