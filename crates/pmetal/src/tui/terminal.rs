//! The terminal, and giving it back.
//!
//! The interface switches the terminal into raw mode on the alternate screen
//! with mouse capture and bracketed paste. All of that has to be undone however
//! the interface ends: a normal exit, an error from the event loop, a panic, a
//! SIGTERM. [`TerminalSession`] undoes it when it is dropped, so an early
//! return or an unwinding panic cannot leave the shell raw, and a panic hook
//! (also installed here) covers the builds that abort instead of unwinding.
//! The hook also kills the jobs the interface started, which would otherwise
//! keep running with nobody watching.

use std::io::{self, Write};
use std::sync::Mutex;

use crossterm::event::{
    DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
};
use crossterm::execute;
use ratatui::DefaultTerminal;

use pmetal_supervisor::Supervisor;

/// The supervisor whose jobs the panic hook kills.
static LIVE_JOBS: Mutex<Option<Supervisor>> = Mutex::new(None);

/// Name the supervisor whose jobs must not outlive a panic.
pub fn guard_jobs(supervisor: Option<Supervisor>) {
    *LIVE_JOBS.lock().unwrap_or_else(|e| e.into_inner()) = supervisor;
}

fn kill_guarded_jobs() {
    if let Some(supervisor) = LIVE_JOBS.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
        supervisor.kill_all_now();
    }
}

/// Undo the terminal modes the interface turns on, to `out`.
pub fn restore_modes(out: &mut impl Write) {
    // Each step is attempted whatever the one before did: a terminal that
    // refuses to turn off mouse capture still has to leave raw mode.
    let _ = execute!(out, DisableMouseCapture, DisableBracketedPaste);
}

/// The terminal in interface mode. Dropping it restores the terminal.
pub struct TerminalSession {
    terminal: DefaultTerminal,
    mouse: bool,
}

impl TerminalSession {
    /// Enter raw mode and the alternate screen, with the panic hook that
    /// restores both. `mouse` turns on mouse capture.
    pub fn start(mouse: bool) -> io::Result<Self> {
        // Ours goes in before ratatui's, so ratatui restores the screen
        // first and ours runs on a usable terminal and then hands over to the
        // hook that prints the panic.
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            kill_guarded_jobs();
            restore_modes(&mut io::stdout());
            previous(info);
        }));
        let terminal = ratatui::try_init()?;
        let mut session = Self { terminal, mouse };
        let mut out = io::stdout();
        execute!(out, EnableBracketedPaste)?;
        if mouse {
            execute!(out, EnableMouseCapture)?;
        }
        // Reading the terminal's size can fail on the first draw of a closed
        // pty; surface that here rather than from the event loop.
        session.terminal.clear()?;
        Ok(session)
    }

    pub fn terminal(&mut self) -> &mut DefaultTerminal {
        &mut self.terminal
    }

    /// Turn mouse capture on or off (off lets the terminal select text).
    pub fn set_mouse(&mut self, on: bool) {
        if on == self.mouse {
            return;
        }
        self.mouse = on;
        let mut out = io::stdout();
        let _ = if on {
            execute!(out, EnableMouseCapture)
        } else {
            execute!(out, DisableMouseCapture)
        };
    }
}

impl Drop for TerminalSession {
    fn drop(&mut self) {
        restore_modes(&mut io::stdout());
        let _ = ratatui::try_restore();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_modes_the_interface_turns_on_are_turned_off() {
        let mut out = Vec::new();
        restore_modes(&mut out);
        let text = String::from_utf8_lossy(&out);
        // Mouse reporting (1000/1002/1003/1006 reset) and bracketed paste (2004 reset).
        assert!(text.contains("\u{1b}[?1000l"), "{text:?}");
        assert!(text.contains("\u{1b}[?2004l"), "{text:?}");
    }

    #[tokio::test]
    async fn a_panic_kills_the_jobs_the_interface_started() {
        use pmetal_supervisor::{JobRequest, Limits, ProcessStatus};
        let dir = tempfile::tempdir().unwrap();
        let script = dir.path().join("fake-pmetal");
        std::fs::write(&script, "#!/bin/sh\nsleep 300\n").unwrap();
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        let supervisor = Supervisor::new(&script, Limits::default());
        let id = supervisor
            .spawn(JobRequest {
                subcommand: "train".into(),
                ..JobRequest::default()
            })
            .unwrap();
        guard_jobs(Some(supervisor.clone()));

        // The hook the session installs, without a terminal to take over.
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| kill_guarded_jobs()));
        let result = std::panic::catch_unwind(|| panic!("the interface panicked"));
        std::panic::set_hook(previous);
        guard_jobs(None);

        assert!(result.is_err());
        for _ in 0..100 {
            if !supervisor.summary(&id).unwrap().status.is_active() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        }
        assert_eq!(
            supervisor.summary(&id).unwrap().status,
            ProcessStatus::Stopped
        );
    }
}
