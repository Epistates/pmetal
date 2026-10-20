//! Datasets tab: browse local and Hugging Face cached datasets.
//!
//! Like the models scan, this one reads files (a line count, the first row), so
//! [`scan`] is a plain function for a blocking task. What a dataset's columns
//! say about its format is `pmetal_data`'s decision, the same one training
//! makes when it loads the file, so "Ready" here means the loader will accept
//! it.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use pmetal_data::DatasetFormat;
use pmetal_data::rows::RowReader;
use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, Borders, Cell, HighlightSpacing, Paragraph, Row, Scrollbar, ScrollbarOrientation,
    ScrollbarState, StatefulWidget, Table, TableState, Widget, Wrap,
};

use crate::tui::tabs::custom_dirs::CustomDirs;
use crate::tui::theme::THEME;

/// Whether the training loader can read a dataset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Readiness {
    /// The loader recognises the layout; the label names it.
    Ready(&'static str),
    /// There are columns, but no layout the loader knows.
    NeedsMapping,
    /// The file could not be read, or has no rows.
    Unreadable,
}

impl Readiness {
    /// Human-readable status label.
    pub fn label(&self) -> String {
        match self {
            Self::Ready(format) => format!("Ready ({format})"),
            Self::NeedsMapping => "Needs column mapping".to_string(),
            Self::Unreadable => "Unreadable or empty".to_string(),
        }
    }

    pub fn is_ready(&self) -> bool {
        matches!(self, Self::Ready(_))
    }
}

/// The name a [`DatasetFormat`] goes by on screen.
fn format_name(format: &DatasetFormat) -> &'static str {
    match format {
        DatasetFormat::Simple => "text",
        DatasetFormat::Alpaca => "Alpaca",
        DatasetFormat::ShareGpt => "ShareGPT",
        DatasetFormat::OpenAi => "messages",
        DatasetFormat::Reasoning => "reasoning",
        DatasetFormat::PromptCompletion => "prompt/completion",
        DatasetFormat::Auto | DatasetFormat::Custom { .. } => "custom",
    }
}

/// A dataset entry.
#[derive(Debug, Clone)]
pub struct DatasetEntry {
    pub name: String,
    pub path: PathBuf,
    pub format: String,
    pub size_bytes: u64,
    /// Rows, counted (JSONL) or estimated from a prefix of the file.
    pub sample_count: Option<usize>,
    /// The count is an extrapolation.
    pub sample_count_estimated: bool,
    pub preview: Option<String>,
    /// Where this dataset was found.
    pub source: DatasetSource,
    /// The first row's columns.
    pub columns: Vec<String>,
    /// Whether the loader can read it.
    pub readiness: Readiness,
}

/// Where a dataset was discovered from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DatasetSource {
    /// Local working directory.
    Local,
    /// HuggingFace datasets cache.
    HfCache,
    /// A user-added custom directory.
    Custom(PathBuf),
}

impl DatasetEntry {
    pub fn size_display(&self) -> String {
        let mb = self.size_bytes as f64 / (1024.0 * 1024.0);
        if mb >= 1024.0 {
            format!("{:.1} GB", mb / 1024.0)
        } else if mb >= 1.0 {
            format!("{:.1} MB", mb)
        } else {
            format!("{:.0} KB", self.size_bytes as f64 / 1024.0)
        }
    }
}

/// Where a scan looks.
#[derive(Debug, Clone)]
pub struct ScanPlan {
    /// Built-in local directories (`./`, `./data`, `./datasets`).
    pub local_dirs: Vec<PathBuf>,
    /// The Hugging Face hub cache.
    pub hub_cache: PathBuf,
    /// Directories the user added.
    pub custom_dirs: Vec<PathBuf>,
}

/// Find every dataset file the plan reaches. Blocking.
pub fn scan(plan: &ScanPlan) -> Vec<DatasetEntry> {
    let mut datasets = Vec::new();
    let mut seen: HashSet<PathBuf> = HashSet::new();

    for dir in &plan.local_dirs {
        scan_dir(dir, DatasetSource::Local, &mut seen, &mut datasets);
    }
    for cached in pmetal_hub::list_cached_datasets_in(&plan.hub_cache) {
        // Snapshots nest their files in any layout (root, data/, default/train/).
        for file in walkdir::WalkDir::new(&cached.snapshot)
            .max_depth(4)
            .into_iter()
            .flatten()
        {
            let path = file.path();
            if !path.is_file() || !is_dataset_file(path) {
                continue;
            }
            let rel = path.strip_prefix(&cached.snapshot).unwrap_or(path);
            let name = format!("{}/{}", cached.repo_id, rel.display());
            if let Some(entry) = inspect(path, name, DatasetSource::HfCache, &mut seen) {
                datasets.push(entry);
            }
        }
    }
    for dir in &plan.custom_dirs {
        scan_dir(
            dir,
            DatasetSource::Custom(dir.clone()),
            &mut seen,
            &mut datasets,
        );
    }

    datasets.sort_by(|a, b| a.name.cmp(&b.name));
    datasets
}

fn scan_dir(
    dir: &Path,
    source: DatasetSource,
    seen: &mut HashSet<PathBuf>,
    out: &mut Vec<DatasetEntry>,
) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() || !is_dataset_file(&path) {
            continue;
        }
        let name = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        if let Some(entry) = inspect(&path, name, source.clone(), seen) {
            out.push(entry);
        }
    }
}

fn container_of(path: &Path) -> Option<&'static str> {
    match path
        .extension()
        .and_then(|e| e.to_str())?
        .to_lowercase()
        .as_str()
    {
        "jsonl" => Some("JSONL"),
        "json" => Some("JSON"),
        "parquet" => Some("Parquet"),
        "csv" => Some("CSV"),
        _ => None,
    }
}

fn is_dataset_file(path: &Path) -> bool {
    container_of(path).is_some()
}

/// Read what a dataset file says about itself. `None` for a file already seen.
fn inspect(
    path: &Path,
    name: String,
    source: DatasetSource,
    seen: &mut HashSet<PathBuf>,
) -> Option<DatasetEntry> {
    let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    if !seen.insert(canonical) {
        return None;
    }
    let format = container_of(path)?;
    let size_bytes = path.metadata().map(|m| m.len()).unwrap_or(0);
    let (sample_count, sample_count_estimated) = if format == "JSONL" {
        count_lines(path).map_or((None, false), |(n, est)| (Some(n), est))
    } else {
        (None, false)
    };

    // The first row, read the way training reads it.
    let first = RowReader::open(path).ok().and_then(|mut rows| rows.next());
    let (columns, preview, readiness) = match first {
        Some(Ok(row)) => {
            let columns: Vec<String> = row.keys().cloned().collect();
            let readiness = match DatasetFormat::detect(&row) {
                Ok(f) => Readiness::Ready(format_name(&f)),
                Err(_) => Readiness::NeedsMapping,
            };
            let preview = serde_json::to_string_pretty(&row).ok();
            (columns, preview, readiness)
        }
        _ => (Vec::new(), None, Readiness::Unreadable),
    };

    Some(DatasetEntry {
        name,
        path: path.to_path_buf(),
        format: format.to_string(),
        size_bytes,
        sample_count,
        sample_count_estimated,
        preview,
        source,
        columns,
        readiness,
    })
}

/// Rows in a JSONL file: counted over at most the first 8 MiB, and scaled up
/// from that for a bigger file (`true` says it is an estimate).
fn count_lines(path: &Path) -> Option<(usize, bool)> {
    use std::io::Read;
    const MAX_SCAN_BYTES: u64 = 8 * 1024 * 1024;
    let file = std::fs::File::open(path).ok()?;
    let file_size = file.metadata().ok()?.len();
    let scan_size = file_size.min(MAX_SCAN_BYTES);

    let mut reader = file.take(scan_size);
    let mut buf = vec![0u8; 64 * 1024];
    let mut count = 0usize;
    let mut last = b'\n';
    loop {
        let n = reader.read(&mut buf).ok()?;
        if n == 0 {
            break;
        }
        count += buf[..n].iter().filter(|&&b| b == b'\n').count();
        last = buf[n - 1];
    }
    // A last line without a trailing newline is a line too.
    if scan_size > 0 && last != b'\n' {
        count += 1;
    }
    let estimated = file_size > scan_size;
    if estimated {
        count = (count as f64 * (file_size as f64 / scan_size as f64)) as usize;
    }
    Some((count, estimated))
}

/// What a sample of a dataset says about it, for sizing a training run's
/// context.
#[derive(Debug, Clone, Default)]
pub struct DatasetPeek {
    /// The first row's columns.
    pub columns: Vec<String>,
    /// Estimated tokens of the sampled rows, ascending.
    pub token_estimates: Vec<usize>,
}

impl DatasetPeek {
    /// Sample the first rows of `path`. Blocking.
    pub fn sample(path: &Path) -> Self {
        let columns = pmetal_data::peek_columns(path).unwrap_or_default();
        // A token is roughly four characters.
        let mut token_estimates: Vec<usize> = pmetal_data::sample_row_chars(path, 100)
            .unwrap_or_default()
            .into_iter()
            .map(|chars| chars / 4)
            .collect();
        token_estimates.sort_unstable();
        Self {
            columns,
            token_estimates,
        }
    }

    /// A line describing the sample, and a warning if `max_seq_len` does not
    /// suit it.
    pub fn describe(&self, max_seq_len: usize) -> (Option<String>, Option<String>) {
        let sorted = &self.token_estimates;
        if sorted.is_empty() {
            let info =
                (!self.columns.is_empty()).then(|| format!("Columns: {}", self.columns.join(", ")));
            return (info, None);
        }
        let avg = sorted.iter().sum::<usize>() / sorted.len();
        let max = sorted.last().copied().unwrap_or(0);
        let p95 = sorted
            .get(((sorted.len() as f64 * 0.95) as usize).min(sorted.len() - 1))
            .copied()
            .unwrap_or(avg);
        let suggested = if p95 > 0 { p95.div_ceil(64) * 64 } else { 2048 };
        let info = format!(
            "~{} samples | avg ~{avg} tok, max ~{max} tok | suggest seq_len {suggested}",
            sorted.len()
        );
        let warning = if max_seq_len == 0 {
            None
        } else if max_seq_len < avg {
            Some(format!(
                "max_seq_len {max_seq_len} < avg tokens {avg}: most samples will be truncated."
            ))
        } else if max_seq_len < suggested && max > max_seq_len {
            Some(format!(
                "Some samples may be truncated (max ~{max} tok). Suggest max_seq_len {suggested}."
            ))
        } else if max_seq_len > max * 2 && max > 0 {
            Some(format!(
                "max_seq_len {max_seq_len} >> data max ~{max} tok. Could reduce to {suggested} for speed."
            ))
        } else {
            None
        };
        (Some(info), warning)
    }
}

/// Datasets tab state.
pub struct DatasetsTab {
    pub datasets: Vec<DatasetEntry>,
    pub table_state: TableState,
    pub scrollbar_state: ScrollbarState,
    /// Built-in local scan directories (cwd, ./data, ./datasets).
    pub local_dirs: Vec<PathBuf>,
    /// User-configured custom directories.
    pub custom_dirs: CustomDirs,
    /// A scan is running.
    pub loading: bool,
}

impl DatasetsTab {
    pub fn new() -> Self {
        Self {
            datasets: Vec::new(),
            table_state: TableState::default(),
            scrollbar_state: ScrollbarState::default(),
            local_dirs: vec![
                PathBuf::from("./"),
                PathBuf::from("./data"),
                PathBuf::from("./datasets"),
            ],
            custom_dirs: CustomDirs::datasets(),
            loading: true,
        }
    }

    /// What a scan should look at now.
    pub fn scan_plan(&self) -> ScanPlan {
        ScanPlan {
            local_dirs: self.local_dirs.clone(),
            hub_cache: pmetal_hub::cache_dir(),
            custom_dirs: self.custom_dirs.dirs().to_vec(),
        }
    }

    pub fn begin_scan(&mut self) {
        self.loading = true;
    }

    /// A scan finished. The selection stays on the same file if it is still there.
    pub fn set_datasets(&mut self, datasets: Vec<DatasetEntry>) {
        let selected = self.selected_dataset().map(|d| d.path.clone());
        self.datasets = datasets;
        self.loading = false;
        let index = selected
            .and_then(|p| self.datasets.iter().position(|d| d.path == p))
            .or(if self.datasets.is_empty() {
                None
            } else {
                Some(0)
            });
        self.table_state.select(index);
        self.scrollbar_state =
            ScrollbarState::new(self.datasets.len()).position(index.unwrap_or(0));
    }

    /// Add a directory to scan. Returns whether it was new.
    pub fn add_custom_dir(&mut self, dir: PathBuf) -> bool {
        self.custom_dirs.add(dir)
    }

    pub fn next_row(&mut self) {
        let count = self.datasets.len();
        if count == 0 {
            return;
        }
        let i = self.table_state.selected().map_or(0, |i| (i + 1) % count);
        self.table_state.select(Some(i));
        self.scrollbar_state = self.scrollbar_state.position(i);
    }

    pub fn prev_row(&mut self) {
        let count = self.datasets.len();
        if count == 0 {
            return;
        }
        let i = self
            .table_state
            .selected()
            .map_or(0, |i| (i + count - 1) % count);
        self.table_state.select(Some(i));
        self.scrollbar_state = self.scrollbar_state.position(i);
    }

    pub fn selected_dataset(&self) -> Option<&DatasetEntry> {
        self.table_state
            .selected()
            .and_then(|i| self.datasets.get(i))
    }
}

impl Default for DatasetsTab {
    fn default() -> Self {
        Self::new()
    }
}

impl Widget for &mut DatasetsTab {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let [table_area, detail_area] =
            Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
                .areas(area);

        self.render_table(table_area, buf);
        self.render_detail(detail_area, buf);
    }
}

impl DatasetsTab {
    fn render_table(&mut self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title(format!(" Datasets ({}) ", self.datasets.len()))
            .title_style(THEME.block_title)
            .borders(Borders::ALL)
            .border_style(THEME.block);

        if self.loading && self.datasets.is_empty() {
            Paragraph::new("\n  Scanning the dataset directories...")
                .style(THEME.text_muted)
                .block(block)
                .render(area, buf);
            return;
        }
        if self.datasets.is_empty() {
            Paragraph::new("\n  No datasets found.\n\n  Scanned: ./, ./data/, ./datasets/, the HF cache\n  Add dir: press 'a' to add a custom dataset directory")
                .style(THEME.text_muted)
                .block(block)
                .render(area, buf);
            return;
        }

        let header = Row::new(vec!["Name", "Format", "Size", "Status"])
            .style(THEME.table_header)
            .height(1);

        let rows: Vec<Row<'_>> = self
            .datasets
            .iter()
            .enumerate()
            .map(|(i, ds)| {
                let style = if i % 2 == 0 {
                    THEME.table_row
                } else {
                    THEME.table_row_alt
                };
                let status_cell = match ds.readiness {
                    Readiness::Ready(_) => Cell::new("Ready").style(THEME.text_success),
                    Readiness::NeedsMapping => Cell::new("Map cols").style(THEME.text_warning),
                    Readiness::Unreadable => Cell::new("Unreadable").style(THEME.text_muted),
                };
                Row::new(vec![
                    Cell::new(ds.name.clone()),
                    Cell::new(ds.format.clone()),
                    Cell::new(ds.size_display()),
                    status_cell,
                ])
                .style(style)
            })
            .collect();

        let table = Table::new(
            rows,
            [
                Constraint::Fill(1),
                Constraint::Length(8),
                Constraint::Length(10),
                Constraint::Length(10),
            ],
        )
        .header(header)
        .block(block)
        .row_highlight_style(THEME.table_selected)
        .highlight_spacing(HighlightSpacing::Always);

        StatefulWidget::render(table, area, buf, &mut self.table_state);

        Scrollbar::new(ScrollbarOrientation::VerticalRight)
            .begin_symbol(None)
            .end_symbol(None)
            .render(area, buf, &mut self.scrollbar_state);
    }

    fn render_detail(&self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title(" Preview ")
            .title_style(THEME.block_title)
            .borders(Borders::ALL)
            .border_style(THEME.block);
        let inner = block.inner(area);
        block.render(area, buf);

        let Some(ds) = self.selected_dataset() else {
            Paragraph::new("Select a dataset to preview")
                .style(THEME.text_muted)
                .render(inner, buf);
            return;
        };

        let source_label = match &ds.source {
            DatasetSource::Local => "Local".to_string(),
            DatasetSource::HfCache => "HF Cache".to_string(),
            DatasetSource::Custom(p) => format!("Custom ({})", p.display()),
        };
        let kv = |key: &'static str, value: String, style| {
            Line::from(vec![
                Span::styled(format!("{key:<9}"), THEME.kv_key),
                Span::styled(value, style),
            ])
        };
        let mut lines = vec![
            kv("File:", ds.name.clone(), THEME.kv_value),
            kv("Format:", ds.format.clone(), THEME.kv_value),
            kv("Size:", ds.size_display(), THEME.kv_value),
            kv("Source:", source_label, THEME.text_dim),
            kv("Path:", ds.path.display().to_string(), THEME.text_dim),
        ];
        if let Some(count) = ds.sample_count {
            let shown = if ds.sample_count_estimated {
                format!("~{count}")
            } else {
                count.to_string()
            };
            lines.push(kv("Samples:", shown, THEME.kv_value));
        }
        let status_style = match ds.readiness {
            Readiness::Ready(_) => THEME.text_success,
            Readiness::NeedsMapping => THEME.text_warning,
            Readiness::Unreadable => THEME.text_muted,
        };
        lines.push(kv("Status:", ds.readiness.label(), status_style));

        if !ds.columns.is_empty() {
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled("Columns:", THEME.kv_key)));
            for col in &ds.columns {
                lines.push(Line::from(Span::styled(
                    format!("  - {col}"),
                    THEME.text_dim,
                )));
            }
        }
        if let Some(preview) = &ds.preview {
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled("First row:", THEME.text_dim)));
            for line in preview.lines().take(20) {
                lines.push(Line::from(Span::styled(line.to_string(), THEME.text_dim)));
            }
        }

        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .render(inner, buf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(path: &Path, text: &str) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }

    fn plan(root: &Path) -> ScanPlan {
        ScanPlan {
            local_dirs: vec![root.join("data")],
            hub_cache: root.join("hub"),
            custom_dirs: vec![root.join("mine")],
        }
    }

    #[test]
    fn readiness_is_the_loaders_own_decision() {
        let root = tempfile::tempdir().unwrap();
        let r = root.path();
        write(
            &r.join("data/chat.jsonl"),
            "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}\n",
        );
        write(
            &r.join("data/pairs.jsonl"),
            "{\"prompt\":\"q\",\"completion\":\"a\"}\n",
        );
        write(&r.join("data/odd.jsonl"), "{\"foo\":1,\"bar\":2}\n");
        write(&r.join("data/empty.jsonl"), "");
        write(&r.join("data/notes.txt"), "not a dataset");
        let by_name: std::collections::HashMap<String, DatasetEntry> = scan(&plan(r))
            .into_iter()
            .map(|d| (d.name.clone(), d))
            .collect();
        assert_eq!(by_name.len(), 4, "{:?}", by_name.keys());
        assert_eq!(
            by_name["chat.jsonl"].readiness,
            Readiness::Ready("messages")
        );
        // A layout the old hand-written check did not know.
        assert_eq!(
            by_name["pairs.jsonl"].readiness,
            Readiness::Ready("prompt/completion")
        );
        assert_eq!(by_name["odd.jsonl"].readiness, Readiness::NeedsMapping);
        assert_eq!(by_name["odd.jsonl"].columns, ["foo", "bar"]);
        assert_eq!(by_name["empty.jsonl"].readiness, Readiness::Unreadable);
        assert_eq!(by_name["chat.jsonl"].sample_count, Some(1));
    }

    #[test]
    fn the_scan_reaches_the_hub_cache_and_custom_directories_once_each() {
        let root = tempfile::tempdir().unwrap();
        let r = root.path();
        let sha = "c".repeat(40);
        write(
            &r.join(format!(
                "hub/datasets--org--corpus/snapshots/{sha}/data/train.jsonl"
            )),
            "{\"text\":\"hello\"}\n",
        );
        write(&r.join("hub/datasets--org--corpus/refs/main"), &sha);
        write(&r.join("mine/extra.jsonl"), "{\"text\":\"x\"}\n");
        // The same file reachable from two places is listed once.
        write(&r.join("data/shared.jsonl"), "{\"text\":\"s\"}\n");
        let mut p = plan(r);
        p.custom_dirs.push(r.join("data"));
        let names: Vec<String> = scan(&p).into_iter().map(|d| d.name).collect();
        assert_eq!(
            names,
            ["extra.jsonl", "org/corpus/data/train.jsonl", "shared.jsonl"]
        );
    }

    #[test]
    fn a_big_file_is_counted_from_a_prefix_and_marked_as_an_estimate() {
        let dir = tempfile::tempdir().unwrap();
        let small = dir.path().join("small.jsonl");
        std::fs::write(&small, "a\nb\nc").unwrap();
        assert_eq!(count_lines(&small), Some((3, false)));
        let big = dir.path().join("big.jsonl");
        // 20 MiB of 16-byte lines: far over the 8 MiB the count reads.
        let line = "0123456789abcde\n";
        std::fs::write(&big, line.repeat(20 * 1024 * 1024 / line.len())).unwrap();
        let (count, estimated) = count_lines(&big).unwrap();
        assert!(estimated);
        let truth = 20 * 1024 * 1024 / line.len();
        assert!(count.abs_diff(truth) * 100 <= truth, "{count} vs {truth}");
    }

    #[test]
    fn a_sample_suggests_a_context_length_and_warns_about_a_bad_one() {
        let peek = DatasetPeek {
            columns: vec!["text".into()],
            token_estimates: (1..=100).map(|n| n * 10).collect(), // 10..1000 tokens
        };
        let (info, warning) = peek.describe(1500);
        let info = info.unwrap();
        assert!(
            info.contains("~100 samples") && info.contains("suggest seq_len 960"),
            "{info}"
        );
        assert!(warning.is_none());
        // Below the average: most rows are cut.
        let (_, warning) = peek.describe(100);
        assert!(warning.unwrap().contains("most samples will be truncated"));
        // Between the average and the longest.
        let (_, warning) = peek.describe(700);
        assert!(warning.unwrap().contains("may be truncated"));
        // Far more than the data needs.
        let (_, warning) = peek.describe(8192);
        assert!(warning.unwrap().contains("Could reduce"));
        // A sample with no rows still names the columns.
        let (info, warning) = DatasetPeek {
            columns: vec!["a".into(), "b".into()],
            token_estimates: Vec::new(),
        }
        .describe(512);
        assert_eq!(info.as_deref(), Some("Columns: a, b"));
        assert!(warning.is_none());
    }

    #[test]
    fn sampling_reads_the_first_rows_the_way_training_does() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("d.jsonl");
        let rows: String = (0..150)
            .map(|n| format!("{{\"text\":\"{}\"}}\n", "w".repeat(40 + n)))
            .collect();
        std::fs::write(&path, rows).unwrap();
        let peek = DatasetPeek::sample(&path);
        assert_eq!(peek.columns, ["text"]);
        assert_eq!(
            peek.token_estimates.len(),
            100,
            "at most 100 rows are sampled"
        );
        assert!(peek.token_estimates.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn a_rescan_keeps_the_selection_on_the_same_file() {
        let entry = |name: &str| DatasetEntry {
            name: name.into(),
            path: PathBuf::from(name),
            format: "JSONL".into(),
            size_bytes: 10,
            sample_count: None,
            sample_count_estimated: false,
            preview: None,
            source: DatasetSource::Local,
            columns: Vec::new(),
            readiness: Readiness::Unreadable,
        };
        let mut tab = DatasetsTab::new();
        tab.set_datasets(vec![entry("a"), entry("b")]);
        tab.next_row();
        tab.set_datasets(vec![entry("0"), entry("a"), entry("b")]);
        assert_eq!(tab.selected_dataset().unwrap().name, "b");
        tab.set_datasets(Vec::new());
        assert!(tab.selected_dataset().is_none());
    }

    #[test]
    fn the_tab_draws_at_any_size() {
        let mut tab = DatasetsTab::new();
        for &(w, h) in &[(0, 0), (1, 1), (20, 5), (60, 12), (150, 40)] {
            let area = Rect::new(0, 0, w, h);
            let mut buf = Buffer::empty(area);
            (&mut tab).render(area, &mut buf);
        }
    }
}
