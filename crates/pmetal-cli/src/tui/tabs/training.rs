//! Training configuration and monitoring tab.

use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Widget, Wrap};

use crate::tui::theme::THEME;

/// Training config parameter for display.
#[derive(Debug, Clone)]
pub struct ConfigParam {
    pub key: String,
    pub value: String,
    pub section: ConfigSection,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigSection {
    Model,
    Training,
    LoRA,
    Data,
    Hardware,
}

impl std::fmt::Display for ConfigSection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigSection::Model => write!(f, "Model"),
            ConfigSection::Training => write!(f, "Training"),
            ConfigSection::LoRA => write!(f, "LoRA"),
            ConfigSection::Data => write!(f, "Data"),
            ConfigSection::Hardware => write!(f, "Hardware"),
        }
    }
}

/// Training run status.
#[derive(Debug, Clone)]
pub enum TrainingStatus {
    Idle,
    Running {
        step: usize,
        total_steps: usize,
        loss: f64,
    },
    Completed {
        final_loss: f64,
        total_steps: usize,
    },
    Failed(String),
}

/// Training tab state.
pub struct TrainingTab {
    pub config_params: Vec<ConfigParam>,
    pub list_state: ListState,
    pub status: TrainingStatus,
    pub active_section: ConfigSection,
}

impl TrainingTab {
    pub fn new() -> Self {
        Self {
            config_params: Self::default_params(),
            list_state: ListState::default().with_selected(Some(0)),
            status: TrainingStatus::Idle,
            active_section: ConfigSection::Model,
        }
    }

    fn default_params() -> Vec<ConfigParam> {
        vec![
            // Model
            ConfigParam {
                key: "Model".into(),
                value: "(not selected)".into(),
                section: ConfigSection::Model,
            },
            ConfigParam {
                key: "Architecture".into(),
                value: "-".into(),
                section: ConfigSection::Model,
            },
            // Training
            ConfigParam {
                key: "Learning Rate".into(),
                value: "2e-4".into(),
                section: ConfigSection::Training,
            },
            ConfigParam {
                key: "Batch Size".into(),
                value: "1".into(),
                section: ConfigSection::Training,
            },
            ConfigParam {
                key: "Epochs".into(),
                value: "1".into(),
                section: ConfigSection::Training,
            },
            ConfigParam {
                key: "Max Seq Len".into(),
                value: "2048".into(),
                section: ConfigSection::Training,
            },
            ConfigParam {
                key: "Grad Accum Steps".into(),
                value: "4".into(),
                section: ConfigSection::Training,
            },
            ConfigParam {
                key: "Max Grad Norm".into(),
                value: "1.0".into(),
                section: ConfigSection::Training,
            },
            // LoRA
            ConfigParam {
                key: "LoRA Rank".into(),
                value: "16".into(),
                section: ConfigSection::LoRA,
            },
            ConfigParam {
                key: "LoRA Alpha".into(),
                value: "32".into(),
                section: ConfigSection::LoRA,
            },
            ConfigParam {
                key: "Quantization".into(),
                value: "None".into(),
                section: ConfigSection::LoRA,
            },
            // Data
            ConfigParam {
                key: "Dataset".into(),
                value: "(not selected)".into(),
                section: ConfigSection::Data,
            },
            ConfigParam {
                key: "Eval Dataset".into(),
                value: "(none)".into(),
                section: ConfigSection::Data,
            },
            ConfigParam {
                key: "Sequence Packing".into(),
                value: "Enabled".into(),
                section: ConfigSection::Data,
            },
            // Hardware
            ConfigParam {
                key: "Flash Attention".into(),
                value: "Enabled".into(),
                section: ConfigSection::Hardware,
            },
            ConfigParam {
                key: "Fused Optimizer".into(),
                value: "Enabled".into(),
                section: ConfigSection::Hardware,
            },
            ConfigParam {
                key: "JIT Compilation".into(),
                value: "Enabled".into(),
                section: ConfigSection::Hardware,
            },
            ConfigParam {
                key: "ANE".into(),
                value: "Auto".into(),
                section: ConfigSection::Hardware,
            },
        ]
    }

    /// Compute the flat list index for a given param index (accounting for section headers).
    fn flat_index_for_param(&self, param_idx: usize) -> usize {
        let mut flat = 0;
        let mut current_section = None;
        for (i, param) in self.config_params.iter().enumerate() {
            if current_section != Some(param.section) {
                current_section = Some(param.section);
                flat += 1; // section header
            }
            if i == param_idx {
                return flat;
            }
            flat += 1; // param row
        }
        flat
    }

    pub fn next_param(&mut self) {
        let count = self.config_params.len();
        if count == 0 {
            return;
        }
        let param_idx = self.list_state.selected().map_or(0, |_| {
            // Find current param index from flat index, then advance
            self.selected_param_idx().map_or(0, |i| (i + 1) % count)
        });
        self.list_state.select(Some(self.flat_index_for_param(param_idx)));
    }

    pub fn prev_param(&mut self) {
        let count = self.config_params.len();
        if count == 0 {
            return;
        }
        let param_idx = self.list_state.selected().map_or(0, |_| {
            self.selected_param_idx()
                .map_or(0, |i| (i + count - 1) % count)
        });
        self.list_state.select(Some(self.flat_index_for_param(param_idx)));
    }

    /// Find which param index corresponds to the current flat list selection.
    fn selected_param_idx(&self) -> Option<usize> {
        let selected = self.list_state.selected()?;
        let mut flat = 0;
        let mut current_section = None;
        for (i, param) in self.config_params.iter().enumerate() {
            if current_section != Some(param.section) {
                current_section = Some(param.section);
                if flat == selected {
                    // Selected a header; snap to next param
                    return Some(i);
                }
                flat += 1;
            }
            if flat == selected {
                return Some(i);
            }
            flat += 1;
        }
        None
    }
}

impl Widget for &mut TrainingTab {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let [config_area, status_area] =
            Layout::horizontal([Constraint::Percentage(60), Constraint::Percentage(40)])
                .areas(area);

        self.render_config(config_area, buf);
        self.render_status(status_area, buf);
    }
}

impl TrainingTab {
    fn render_config(&mut self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title(" Configuration ")
            .title_style(THEME.block_title)
            .borders(Borders::ALL)
            .border_style(THEME.block);

        let mut current_section = None;
        let items: Vec<ListItem> = self
            .config_params
            .iter()
            .flat_map(|param| {
                let mut result = Vec::new();
                if current_section != Some(param.section) {
                    current_section = Some(param.section);
                    result.push(ListItem::new(Line::from(Span::styled(
                        format!("  --- {} ---", param.section),
                        THEME.text_muted,
                    ))));
                }
                result.push(ListItem::new(Line::from(vec![
                    Span::styled(format!("  {:>18}: ", param.key), THEME.kv_key),
                    Span::styled(&param.value, THEME.kv_value),
                ])));
                result
            })
            .collect();

        let list = List::new(items)
            .block(block)
            .highlight_style(THEME.table_selected);

        ratatui::widgets::StatefulWidget::render(list, area, buf, &mut self.list_state);
    }

    fn render_status(&self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title(" Status ")
            .title_style(THEME.block_title)
            .borders(Borders::ALL)
            .border_style(THEME.block);
        let inner = block.inner(area);
        block.render(area, buf);

        let lines: Vec<Line> = match &self.status {
            TrainingStatus::Idle => {
                vec![
                    Line::from(""),
                    Line::from(Span::styled("  Status: Idle", THEME.status_idle)),
                    Line::from(""),
                    Line::from(Span::styled(
                        "  Configure parameters and press",
                        THEME.text_dim,
                    )),
                    Line::from(Span::styled("  Enter to start training.", THEME.text_dim)),
                    Line::from(""),
                    Line::from(Span::styled(
                        "  Or run from CLI:",
                        THEME.text_dim,
                    )),
                    Line::from(Span::styled(
                        "  pmetal train -m <model> -d <dataset>",
                        THEME.text_muted,
                    )),
                ]
            }
            TrainingStatus::Running {
                step,
                total_steps,
                loss,
            } => {
                let progress = if *total_steps > 0 {
                    format!("{:.1}%", *step as f64 / *total_steps as f64 * 100.0)
                } else {
                    format!("step {step}")
                };
                vec![
                    Line::from(""),
                    Line::from(Span::styled("  Status: Running", THEME.status_running)),
                    Line::from(""),
                    Line::from(vec![
                        Span::styled("  Step:     ", THEME.kv_key),
                        Span::styled(
                            format!("{step}/{total_steps} ({progress})"),
                            THEME.kv_value,
                        ),
                    ]),
                    Line::from(vec![
                        Span::styled("  Loss:     ", THEME.kv_key),
                        Span::styled(format!("{loss:.4}"), THEME.kv_value),
                    ]),
                ]
            }
            TrainingStatus::Completed {
                final_loss,
                total_steps,
            } => {
                vec![
                    Line::from(""),
                    Line::from(Span::styled("  Status: Completed", THEME.status_success)),
                    Line::from(""),
                    Line::from(vec![
                        Span::styled("  Steps:      ", THEME.kv_key),
                        Span::styled(total_steps.to_string(), THEME.kv_value),
                    ]),
                    Line::from(vec![
                        Span::styled("  Final Loss: ", THEME.kv_key),
                        Span::styled(format!("{final_loss:.4}"), THEME.text_success),
                    ]),
                ]
            }
            TrainingStatus::Failed(msg) => {
                vec![
                    Line::from(""),
                    Line::from(Span::styled("  Status: Failed", THEME.status_error)),
                    Line::from(""),
                    Line::from(Span::styled(format!("  {msg}"), THEME.text_error)),
                ]
            }
        };

        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .render(inner, buf);
    }
}
