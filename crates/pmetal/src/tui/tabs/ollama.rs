//! Modelfile export tab.
//!
//! PMetal keeps this as a file export workflow: generate a Modelfile from a
//! PMetal model or adapter and let the user decide whether to register it with
//! an external runtime later. `pmetal ollama modelfile` has no job spec, so the
//! fields are listed here; their defaults and the templates on offer are read
//! from the command's own definition.

use clap::CommandFactory;
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;

use crate::cli::Cli;
use crate::tui::jobs::LaunchPlan;
use crate::tui::tabs::dashboard::Telemetry;
use crate::tui::tabs::panel::{FormTab, JobPanel};
use crate::tui::tabs::status_panel::render_standard;
use crate::tui::widgets::{FieldKind, FormField, FormIssue, FormTabState};

/// The "use the architecture's template" choice, which is not a flag value.
const AUTO: &str = "auto";

/// What `pmetal ollama modelfile` says about one of its options.
fn modelfile_arg(id: &str) -> (String, Vec<String>) {
    let command = Cli::command();
    let arg = command
        .find_subcommand("ollama")
        .and_then(|c| c.find_subcommand("modelfile"))
        .and_then(|c| c.get_arguments().find(|a| a.get_long() == Some(id)));
    let default = arg
        .and_then(|a| a.get_default_values().first())
        .map(|v| v.to_string_lossy().into_owned())
        .unwrap_or_default();
    let values = arg
        .map(|a| {
            a.get_possible_values()
                .iter()
                .map(|v| v.get_name().to_string())
                .collect()
        })
        .unwrap_or_default();
    (default, values)
}

fn field(label: &str, value: &str, kind: FieldKind, section: &str, required: bool) -> FormField {
    let mut f = FormField::new(label, value, kind, section);
    f.required = required;
    f.optional = !required;
    f
}

fn form() -> FormTabState {
    let (output, _) = modelfile_arg("output");
    let (_, templates) = modelfile_arg("template");
    let mut template_options = vec![AUTO.to_string()];
    template_options.extend(templates);
    FormTabState::new(vec![
        field("Base Model", "", FieldKind::Text, "Model", true),
        field("LoRA Adapter", "", FieldKind::Path, "Model", false),
        field("Output Path", &output, FieldKind::Path, "Output", true),
        field(
            "Template",
            AUTO,
            FieldKind::Enum {
                options: template_options,
            },
            "Modelfile",
            false,
        ),
        field("System Prompt", "", FieldKind::Text, "Modelfile", false),
        field(
            "Temperature",
            "",
            FieldKind::Number { min: 0.0, max: 2.0 },
            "Modelfile",
            false,
        ),
        field(
            "Num Context",
            "",
            FieldKind::Integer {
                min: 1,
                max: i32::MAX as i64,
            },
            "Modelfile",
            false,
        ),
        field(
            "Top K",
            "",
            FieldKind::Integer {
                min: 0,
                max: i32::MAX as i64,
            },
            "Modelfile",
            false,
        ),
        field(
            "Top P",
            "",
            FieldKind::Number { min: 0.0, max: 1.0 },
            "Modelfile",
            false,
        ),
        field("License", "", FieldKind::Text, "Modelfile", false),
    ])
}

/// Modelfile tab state.
pub struct OllamaTab {
    panel: JobPanel,
}

impl OllamaTab {
    pub fn new() -> Self {
        Self {
            panel: JobPanel::with_form("Modelfile", form()),
        }
    }
}

impl Default for OllamaTab {
    fn default() -> Self {
        Self::new()
    }
}

impl FormTab for OllamaTab {
    fn panel(&self) -> &JobPanel {
        &self.panel
    }

    fn panel_mut(&mut self) -> &mut JobPanel {
        &mut self.panel
    }

    fn plan(&self) -> Result<LaunchPlan, Vec<FormIssue>> {
        let form = &self.panel.form;
        let mut issues = Vec::new();
        let base = form.value("Base Model");
        if base.trim().is_empty() {
            issues.push(FormIssue {
                field: "Base Model".into(),
                message: "required".into(),
            });
        }
        let output = form.value("Output Path");
        if output.trim().is_empty() {
            issues.push(FormIssue {
                field: "Output Path".into(),
                message: "required".into(),
            });
        }
        if !issues.is_empty() {
            return Err(issues);
        }
        let mut argv = vec![
            "modelfile".to_string(),
            "--base".to_string(),
            base.trim().to_string(),
            "--output".to_string(),
            output.trim().to_string(),
        ];
        let mut option = |flag: &str, label: &str| {
            let value = form.value(label);
            let value = value.trim();
            if !value.is_empty() {
                argv.extend([flag.to_string(), value.to_string()]);
            }
        };
        option("--lora", "LoRA Adapter");
        option("--system", "System Prompt");
        option("--temperature", "Temperature");
        option("--num-ctx", "Num Context");
        option("--top-k", "Top K");
        option("--top-p", "Top P");
        option("--license", "License");
        let template = form.value("Template");
        if template != AUTO && !template.is_empty() {
            argv.extend(["--template".to_string(), template]);
        }
        Ok(LaunchPlan {
            subcommand: "ollama".to_string(),
            argv,
            ..LaunchPlan::default()
        })
    }

    fn start_title(&self) -> String {
        "Generate Modelfile?".to_string()
    }

    fn render(&mut self, area: Rect, buf: &mut Buffer, _: &Telemetry<'_>) {
        render_standard(&mut self.panel, area, buf, 7, Vec::new(), "Export Log");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::cli_check::assert_parses;

    #[test]
    fn the_defaults_and_templates_are_the_commands_own() {
        let tab = OllamaTab::new();
        assert_eq!(tab.panel.form.value("Output Path"), "Modelfile");
        let template = tab
            .panel
            .form
            .fields
            .iter()
            .find(|f| f.label == "Template")
            .unwrap();
        let FieldKind::Enum { options } = &template.kind else {
            panic!("an enum")
        };
        assert_eq!(options[0], "auto");
        assert!(options.len() > 1, "{options:?}");
    }

    #[test]
    fn what_it_launches_is_what_the_cli_accepts() {
        let mut tab = OllamaTab::new();
        let issues = tab.plan().unwrap_err();
        assert_eq!(issues[0].field, "Base Model");
        tab.panel.form.set_value("Base Model", "/models/m.gguf");
        tab.panel.form.set_value("LoRA Adapter", "/out/adapter");
        tab.panel.form.set_value("System Prompt", "You are brief.");
        tab.panel.form.set_value("Temperature", "0.7");
        tab.panel.form.set_value("Num Context", "8192");
        tab.panel.form.set_value("Top K", "40");
        tab.panel.form.set_value("Top P", "0.9");
        tab.panel.form.set_value("License", "MIT");
        // Every template the command offers is accepted.
        let templates: Vec<String> = match &tab
            .panel
            .form
            .fields
            .iter()
            .find(|f| f.label == "Template")
            .unwrap()
            .kind
        {
            FieldKind::Enum { options } => options.clone(),
            _ => unreachable!(),
        };
        for template in templates {
            tab.panel.form.set_value("Template", template);
            let plan = tab.plan().unwrap();
            assert_eq!(plan.subcommand, "ollama");
            assert_parses(&plan);
        }
    }

    #[test]
    fn an_optional_entry_left_empty_is_not_passed() {
        let mut tab = OllamaTab::new();
        tab.panel.form.set_value("Base Model", "m");
        let plan = tab.plan().unwrap();
        assert_eq!(
            plan.argv,
            ["modelfile", "--base", "m", "--output", "Modelfile"]
        );
    }
}
