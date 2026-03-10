//! Event handling for the PMetal TUI.
//!
//! Wraps crossterm events into application-level events with a dedicated
//! event polling thread for non-blocking UI updates.

use std::sync::mpsc::{self, SyncSender};
use std::thread;
use std::time::Duration;

use crossterm::event::{self, Event as CrosstermEvent, KeyEvent, MouseEvent};

/// Application-level events.
#[derive(Debug)]
#[allow(dead_code)]
pub enum Event {
    /// A key was pressed.
    Key(KeyEvent),
    /// A mouse event occurred.
    Mouse(MouseEvent),
    /// The terminal was resized.
    Resize(u16, u16),
    /// A tick event for periodic updates.
    Tick,
}

/// Event handler that polls crossterm events on a background thread.
pub struct EventHandler {
    rx: mpsc::Receiver<Event>,
    _handle: thread::JoinHandle<()>,
}

impl EventHandler {
    /// Create a new event handler with the given tick rate.
    pub fn new(tick_rate: Duration) -> Self {
        // Bounded channel prevents event backlog during UI-blocking scans
        let (tx, rx) = mpsc::sync_channel(32);

        let handle = thread::spawn(move || {
            Self::poll_loop(tx, tick_rate);
        });

        Self {
            rx,
            _handle: handle,
        }
    }

    /// Receive the next event (blocking).
    pub fn next(&self) -> Result<Event, mpsc::RecvError> {
        self.rx.recv()
    }

    fn poll_loop(tx: SyncSender<Event>, tick_rate: Duration) {
        loop {
            if event::poll(tick_rate).unwrap_or(false) {
                let event = match event::read() {
                    Ok(CrosstermEvent::Key(key)) => {
                        if key.kind == crossterm::event::KeyEventKind::Press {
                            Some(Event::Key(key))
                        } else {
                            None
                        }
                    }
                    Ok(CrosstermEvent::Mouse(mouse)) => Some(Event::Mouse(mouse)),
                    Ok(CrosstermEvent::Resize(w, h)) => Some(Event::Resize(w, h)),
                    _ => None,
                };
                if let Some(event) = event {
                    if tx.try_send(event).is_err() {
                        // Channel full or disconnected — exit
                        return;
                    }
                }
            } else {
                // Timeout — send tick (blocking is fine here since ticks are sparse)
                if tx.send(Event::Tick).is_err() {
                    return;
                }
            }
        }
    }
}
