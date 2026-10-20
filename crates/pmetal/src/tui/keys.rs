//! The key bindings, written down once.
//!
//! The footer, the help overlay and the tests all read these tables, so what
//! the interface tells a person a key does is what the handlers in `app` do
//! (a test checks the tables against the handlers by pressing the keys).

use crate::tui::tabs::Tab;

/// One key binding.
#[derive(Debug, Clone, Copy)]
pub struct Binding {
    /// The key as shown (`j / k`, `Ctrl+P`).
    pub keys: &'static str,
    /// What it does, for the help overlay.
    pub help: &'static str,
    /// A word or two for the footer, if the footer has room for it.
    pub footer: Option<&'static str>,
}

const fn bind(keys: &'static str, help: &'static str, footer: Option<&'static str>) -> Binding {
    Binding { keys, help, footer }
}

/// Bindings that work on every tab (unless a text field has the keyboard).
pub const GLOBAL: &[Binding] = &[
    bind("?", "Show / hide this help", Some("help")),
    bind("Tab / Shift+Tab", "Next / previous tab", Some("switch")),
    bind("Ctrl+P", "Go to any tab (type to filter)", Some("tabs")),
    bind("Alt+1..9", "Jump to tab 1..9", None),
    bind(
        "Ctrl+C",
        "Stop this tab's running job; with none running, quit",
        None,
    ),
    bind(
        "q",
        "Quit (asks first when jobs are running; they are stopped)",
        Some("quit"),
    ),
    bind(
        "F2",
        "Toggle mouse capture (off lets you select text)",
        None,
    ),
    bind("Esc", "Close a dialog or cancel an edit", None),
];

/// The bindings of every tab that runs a job from a form.
const FORM: &[Binding] = &[
    bind("j / k", "Move between fields", Some("navigate")),
    bind(
        "Enter",
        "Edit, cycle or pick the field's value",
        Some("edit/pick"),
    ),
    bind("Left / Right", "Cycle an option backwards / forwards", None),
    bind("S", "Review and start the job", Some("start")),
    bind("x", "Stop the running job", Some("stop")),
    bind("PgUp / PgDn", "Scroll the log", None),
    bind("End", "Follow the log again", None),
];

const TRAINING: &[Binding] = &[
    bind("j / k", "Move between fields", Some("navigate")),
    bind(
        "Enter",
        "Edit, cycle or pick the field's value",
        Some("edit/pick"),
    ),
    bind("Left / Right", "Cycle an option backwards / forwards", None),
    bind("S", "Review and start the job", Some("start")),
    bind("x", "Stop the running job", Some("stop")),
    bind(
        "L",
        "Change the learning rate of the running job",
        Some("set LR"),
    ),
];

const DASHBOARD: &[Binding] = &[
    bind("p", "Pause / resume metric updates", Some("pause")),
    bind("r", "Clear the charts", Some("reset")),
];

const MODELS: &[Binding] = &[
    bind("j / k", "Move between models", Some("navigate")),
    bind("/", "Filter the list", Some("filter")),
    bind("R", "Rescan the model directories", Some("rescan")),
    bind("d", "Download a model by Hugging Face id", Some("download")),
    bind("S", "Search the Hugging Face Hub", Some("search")),
    bind("a", "Add a directory of models", None),
    bind("t", "Train with the selected model", Some("train")),
    bind("s", "Distill with the selected model as the student", None),
    bind("i", "Chat with the selected model", Some("infer")),
    bind("f", "Fuse a LoRA adapter into its base model", None),
];

const DATASETS: &[Binding] = &[
    bind("j / k", "Move between datasets", Some("navigate")),
    bind("c", "Convert to pmetal JSONL", Some("convert")),
    bind("a", "Add a directory of datasets", Some("add dir")),
    bind("R", "Rescan the dataset directories", Some("rescan")),
];

const INFERENCE: &[Binding] = &[
    bind("Enter", "Send the prompt", Some("send")),
    bind("Ctrl+P", "Pick a model", Some("model")),
    bind("Ctrl+S", "Focus the settings sidebar", Some("settings")),
    bind("Ctrl+H", "Show / hide the settings sidebar", None),
    bind("Ctrl+B", "Browse messages (y copies one)", Some("browse")),
    bind("Esc", "Stop the generation", Some("stop")),
    bind("PgUp / PgDn", "Scroll the conversation", None),
    bind("Ctrl+Q", "Quit", None),
];

const JOBS: &[Binding] = &[
    bind("j / k", "Move between jobs", Some("navigate")),
    bind("x", "Stop the selected job", Some("stop")),
    bind(
        "PgUp / PgDn",
        "Scroll the selected job's log",
        Some("scroll"),
    ),
    bind("End", "Follow the log again", None),
    bind("R", "Rescan finished runs", Some("rescan")),
];

const NONE: &[Binding] = &[];

/// The bindings specific to `tab`.
pub fn tab_bindings(tab: Tab) -> &'static [Binding] {
    match tab {
        Tab::Dashboard => DASHBOARD,
        Tab::Device => NONE,
        Tab::Models => MODELS,
        Tab::Datasets => DATASETS,
        Tab::Training | Tab::Pretrain | Tab::Distillation | Tab::Grpo => TRAINING,
        Tab::Inference => INFERENCE,
        Tab::Jobs => JOBS,
        Tab::Tokenize
        | Tab::EmbedTrain
        | Tab::Rlkd
        | Tab::Preference
        | Tab::Dflash
        | Tab::Serve
        | Tab::Quantize
        | Tab::Merge
        | Tab::Bench
        | Tab::Eval
        | Tab::Ollama => FORM,
    }
}

/// The footer's `(key, label)` pairs for `tab`: its own, then the global ones
/// that have a footer label. The inference tab owns `q` (it is typed text), so
/// quitting is its Ctrl+Q.
pub fn footer_items(tab: Tab) -> Vec<(&'static str, &'static str)> {
    let own = tab_bindings(tab).iter();
    let global = GLOBAL
        .iter()
        .filter(|b| !(tab == Tab::Inference && b.keys == "q"));
    own.chain(global)
        .filter_map(|b| b.footer.map(|label| (footer_key(b.keys), label)))
        .collect()
}

/// The short form of a binding's keys for the footer (`j / k` -> `jk`).
fn footer_key(keys: &'static str) -> &'static str {
    match keys {
        "j / k" => "jk",
        "Tab / Shift+Tab" => "Tab",
        other => other,
    }
}
