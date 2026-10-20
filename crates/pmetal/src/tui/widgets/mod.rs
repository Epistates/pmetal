//! Custom widgets for the PMetal TUI.

mod footer;
mod form_tab;
mod header;
pub mod input_field;
pub mod key_value;
mod log_view;
mod status;

pub use footer::Footer;
pub use form_tab::{FormAction, FormIssue, FormTabState, FormView};
pub use header::{Header, tab_at};
#[cfg(test)]
pub use header::{LOGO_WIDTH, tab_slots};
pub use input_field::{FieldKind, FormField};
pub use key_value::KeyValueList;
pub use log_view::{DEFAULT_CAP as DEFAULT_LOG_CAP, LogView};
pub use status::{StatusTone, format_duration, status_line};
