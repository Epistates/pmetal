//! Serve tab: the OpenAI-compatible inference server.
//!
//! "Serving" means something answers on the address, not that the process
//! started: the app connects to the address the server announces and tells the
//! tab when it is reachable. A server that fails to bind (the port is taken)
//! is therefore never shown as serving.

use std::time::Instant;

use pmetal_core::jobs::ServeSpec;
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::text::{Line, Span};

use crate::tui::jobs::LaunchPlan;
use crate::tui::tabs::dashboard::Telemetry;
use crate::tui::tabs::panel::{FormTab, JobPanel, RunState};
use crate::tui::tabs::status_panel::render_standard;
use crate::tui::theme::THEME;
use crate::tui::widgets::{FormIssue, format_duration};

/// The line the server logs once it has built its router and is about to
/// bind (pmetal-serve, `run_server`).
const ANNOUNCEMENT: &str = "OpenAI-compatible API available at http://";

/// Whether a bind address is this machine only.
fn is_loopback(host: &str) -> bool {
    let host = host.trim().trim_start_matches('[').trim_end_matches(']');
    host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<std::net::IpAddr>()
            .is_ok_and(|ip| ip.is_loopback())
}

/// Serve tab state.
pub struct ServeTab {
    panel: JobPanel,
    /// The address the server announced, not yet checked.
    announced: Option<String>,
    /// The address, once something answers there.
    serving: Option<(String, Instant)>,
}

impl ServeTab {
    pub fn new() -> Self {
        Self {
            panel: JobPanel::new::<ServeSpec>("Serve"),
            announced: None,
            serving: None,
        }
    }

    /// Something answers on the server's address.
    #[cfg(test)]
    pub fn is_serving(&self) -> bool {
        self.serving.is_some()
    }

    /// Where the form says the server will listen.
    pub fn bind_url(&self) -> String {
        format!(
            "http://{}:{}",
            self.panel.form.value("Host"),
            self.panel.form.value("Port")
        )
    }
}

impl Default for ServeTab {
    fn default() -> Self {
        Self::new()
    }
}

impl FormTab for ServeTab {
    fn panel(&self) -> &JobPanel {
        &self.panel
    }

    fn panel_mut(&mut self) -> &mut JobPanel {
        &mut self.panel
    }

    fn plan(&self) -> Result<LaunchPlan, Vec<FormIssue>> {
        self.panel.plan_for::<ServeSpec>()
    }

    fn begin(&mut self) {
        self.announced = None;
        self.serving = None;
    }

    fn summary(&self) -> Vec<String> {
        let mut lines = self.panel.form.summary();
        if !is_loopback(&self.panel.form.value("Host")) {
            lines.push(String::new());
            lines.push("This listens on the network. Anyone who can reach the".to_string());
            lines.push("port can use the model unless PMETAL_API_KEY is set.".to_string());
        }
        lines
    }

    fn on_line(&mut self, line: &str) {
        if self.announced.is_none() && self.serving.is_none() {
            if let Some(at) = line.find(ANNOUNCEMENT) {
                let rest = &line[at + ANNOUNCEMENT.len()..];
                let address = rest.split(['/', ' ']).next().unwrap_or("");
                if !address.is_empty() {
                    self.announced = Some(address.to_string());
                }
            }
        }
    }

    fn take_announcement(&mut self) -> Option<String> {
        self.announced.take()
    }

    fn on_reachable(&mut self) {
        if self.panel.is_running() && self.serving.is_none() {
            self.serving = Some((self.bind_url(), Instant::now()));
        }
    }

    fn render(&mut self, area: Rect, buf: &mut Buffer, _: &Telemetry<'_>) {
        let mut extras = Vec::new();
        match (&self.panel.run, &self.serving) {
            (RunState::Running { .. }, Some((url, since))) => {
                extras.push(Line::from(vec![
                    Span::styled("  ✓ Serving   ", THEME.status_success),
                    Span::styled(url.clone(), THEME.kv_value),
                ]));
                extras.push(Line::from(vec![
                    Span::styled("  Up for      ", THEME.kv_key),
                    Span::styled(format_duration(since.elapsed()), THEME.kv_value),
                ]));
                extras.push(Line::from(Span::styled(
                    "  API: /v1/chat/completions, /v1/completions, /v1/models",
                    THEME.text_dim,
                )));
            }
            (RunState::Running { .. }, None) => {
                extras.push(Line::from(Span::styled(
                    format!("  Waiting for {} to answer...", self.bind_url()),
                    THEME.text_warning,
                )));
            }
            _ => {}
        }
        render_standard(&mut self.panel, area, buf, 9, extras, "Server Log");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::tabs::panel::Pick;

    fn tab_with_model() -> ServeTab {
        let mut tab = ServeTab::new();
        tab.on_pick(
            "Model",
            Pick::Model {
                id: "/models/m",
                architecture: None,
            },
        );
        tab
    }

    fn screen(tab: &mut ServeTab) -> String {
        let area = Rect::new(0, 0, 120, 30);
        let mut buf = Buffer::empty(area);
        tab.render(area, &mut buf, &Telemetry::none());
        (0..30)
            .map(|y| (0..120).map(|x| buf[(x, y)].symbol()).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn the_server_defaults_to_this_machine_only() {
        let tab = ServeTab::new();
        assert_eq!(tab.panel.form.value("Host"), "127.0.0.1");
        assert_eq!(tab.bind_url(), "http://127.0.0.1:8080");
    }

    #[test]
    fn a_server_that_listens_on_the_network_says_so_before_it_starts() {
        let mut tab = tab_with_model();
        assert!(!tab.summary().join("\n").contains("network"));
        tab.panel.form.set_value("Host", "0.0.0.0");
        let text = tab.summary().join("\n");
        assert!(
            text.contains("listens on the network") && text.contains("PMETAL_API_KEY"),
            "{text}"
        );
        for loopback in ["127.0.0.1", "localhost", "::1", "[::1]", "127.0.0.2"] {
            tab.panel.form.set_value("Host", loopback);
            assert!(!tab.summary().join("\n").contains("network"), "{loopback}");
        }
    }

    #[test]
    fn it_is_serving_when_something_answers_not_when_the_process_starts() {
        let mut tab = tab_with_model();
        tab.panel.begin("j".into(), "pmetal serve".into());
        tab.begin();
        assert!(screen(&mut tab).contains("Waiting for http://127.0.0.1:8080 to answer"));
        assert!(!screen(&mut tab).contains("Serving"));
        // The line pmetal-serve logs, as the tracing formatter writes it.
        tab.on_line("2026-10-20T10:00:00Z  INFO pmetal_serve::server: OpenAI-compatible API available at http://127.0.0.1:8080/v1");
        assert_eq!(tab.take_announcement().as_deref(), Some("127.0.0.1:8080"));
        assert_eq!(tab.take_announcement(), None, "announced once");
        assert!(!screen(&mut tab).contains("✓ Serving"));
        tab.on_reachable();
        let text = screen(&mut tab);
        assert!(
            text.contains("✓ Serving") && text.contains("http://127.0.0.1:8080"),
            "{text}"
        );
    }

    #[test]
    fn a_server_that_cannot_bind_never_looks_like_it_is_serving() {
        let mut tab = tab_with_model();
        tab.panel.begin("j".into(), "pmetal serve".into());
        tab.on_line(
            "INFO pmetal_serve: OpenAI-compatible API available at http://127.0.0.1:8080/v1",
        );
        tab.take_announcement();
        // The bind fails and the process ends; nothing ever answered.
        tab.panel.apply(&pmetal_supervisor::EventKind::Exited(
            pmetal_supervisor::ProcessStatus::Failed {
                exit_code: 1,
                error: "Address already in use".into(),
            },
        ));
        tab.on_reachable();
        let text = screen(&mut tab);
        assert!(!text.contains("✓ Serving"), "{text}");
        assert!(text.contains("Address already in use"));
    }

    #[test]
    fn a_new_run_forgets_the_last_servers_address() {
        let mut tab = tab_with_model();
        tab.panel.begin("j".into(), "c".into());
        tab.on_reachable();
        assert!(tab.serving.is_some());
        tab.begin();
        assert!(tab.serving.is_none());
    }
}
