//! Datasets tab — browse local and HuggingFace cached datasets.

use std::collections::HashSet;
use std::path::PathBuf;

use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, Borders, Cell, HighlightSpacing, Paragraph, Row, Scrollbar, ScrollbarOrientation,
    ScrollbarState, StatefulWidget, Table, TableState, Widget, Wrap,
};

use crate::tui::theme::THEME;

/// Detected training data format compatibility.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrainingFormat {
    /// OpenAI chat format: {"messages": [{"role": ..., "content": ...}]}
    Messages,
    /// ShareGPT format: {"conversations": [{"from": ..., "value": ...}]}
    Conversations,
    /// Alpaca format: {"instruction": ..., "output": ...}
    Alpaca,
    /// Simple text: {"text": ...}
    Text,
    /// Has columns but doesn't match a known pmetal format — needs column mapping.
    NeedsMapping,
    /// Could not detect (binary, empty, or unrecognized).
    Unknown,
}

impl TrainingFormat {
    /// Human-readable status label.
    pub fn status_label(&self) -> &'static str {
        match self {
            TrainingFormat::Messages => "Ready (messages)",
            TrainingFormat::Conversations => "Ready (ShareGPT)",
            TrainingFormat::Alpaca => "Ready (Alpaca)",
            TrainingFormat::Text => "Ready (text)",
            TrainingFormat::NeedsMapping => "Needs column mapping",
            TrainingFormat::Unknown => "Unknown format",
        }
    }

    pub fn is_ready(&self) -> bool {
        matches!(
            self,
            TrainingFormat::Messages
                | TrainingFormat::Conversations
                | TrainingFormat::Alpaca
                | TrainingFormat::Text
        )
    }
}

/// A dataset entry.
#[derive(Debug, Clone)]
pub struct DatasetEntry {
    pub name: String,
    pub path: PathBuf,
    pub format: String,
    pub size_bytes: u64,
    pub sample_count: Option<usize>,
    pub preview: Option<String>,
    /// Where this dataset was found.
    pub source: DatasetSource,
    /// Detected top-level keys/columns.
    pub columns: Vec<String>,
    /// Detected training format compatibility.
    pub training_format: TrainingFormat,
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

/// Datasets tab state.
pub struct DatasetsTab {
    pub datasets: Vec<DatasetEntry>,
    pub table_state: TableState,
    pub scrollbar_state: ScrollbarState,
    /// Built-in local scan directories (cwd, ./data, ./datasets).
    pub local_dirs: Vec<PathBuf>,
    /// User-configured custom directories.
    pub custom_dirs: Vec<PathBuf>,
}

impl DatasetsTab {
    pub fn new() -> Self {
        let local_dirs = vec![
            PathBuf::from("./"),
            PathBuf::from("./data"),
            PathBuf::from("./datasets"),
        ];
        let custom_dirs = load_custom_dataset_dirs();
        let mut tab = Self {
            datasets: Vec::new(),
            table_state: TableState::default(),
            scrollbar_state: ScrollbarState::default(),
            local_dirs,
            custom_dirs,
        };
        tab.scan_datasets();
        tab
    }

    /// Add a custom directory to scan for datasets.
    pub fn add_custom_dir(&mut self, dir: PathBuf) {
        if !self.custom_dirs.contains(&dir) {
            self.custom_dirs.push(dir);
            save_custom_dataset_dirs(&self.custom_dirs);
            self.scan_datasets();
        }
    }

    /// Remove a custom directory.
    pub fn remove_custom_dir(&mut self, idx: usize) {
        if idx < self.custom_dirs.len() {
            self.custom_dirs.remove(idx);
            save_custom_dataset_dirs(&self.custom_dirs);
            self.scan_datasets();
        }
    }

    /// Scan all sources for dataset files.
    pub fn scan_datasets(&mut self) {
        self.datasets.clear();
        let mut seen_paths: HashSet<PathBuf> = HashSet::new();

        // 1. Local directories
        for dir in &self.local_dirs.clone() {
            self.scan_dir_for_files(dir, DatasetSource::Local, &mut seen_paths);
        }

        // 2. HuggingFace hub cache (datasets--* directories)
        let hf_hub_cache = pmetal_hub::cache_dir();
        self.scan_hf_datasets_cache(&hf_hub_cache, &mut seen_paths);

        // 3. Custom directories
        let custom_dirs: Vec<PathBuf> = self.custom_dirs.clone();
        for dir in &custom_dirs {
            self.scan_dir_for_files(dir, DatasetSource::Custom(dir.clone()), &mut seen_paths);
        }

        self.datasets.sort_by(|a, b| a.name.cmp(&b.name));
        if !self.datasets.is_empty() {
            self.table_state.select(Some(0));
        }
        self.scrollbar_state = ScrollbarState::new(self.datasets.len());
    }

    /// Scan a directory for dataset files (jsonl, json, parquet, csv).
    fn scan_dir_for_files(
        &mut self,
        dir: &std::path::Path,
        source: DatasetSource,
        seen: &mut HashSet<PathBuf>,
    ) {
        if !dir.exists() {
            return;
        }

        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return,
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }

            let canonical = path.canonicalize().unwrap_or_else(|_| path.clone());
            if seen.contains(&canonical) {
                continue;
            }

            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_lowercase();

            let format = match ext.as_str() {
                "jsonl" => "JSONL",
                "json" => "JSON",
                "parquet" => "Parquet",
                "csv" => "CSV",
                _ => continue,
            };

            let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
            let size_bytes = path.metadata().map(|m| m.len()).unwrap_or(0);

            let sample_count = if ext == "jsonl" {
                count_lines(&path)
            } else {
                None
            };

            let preview = read_first_line(&path);

            let columns = detect_columns(&path, format);
            let training_format = detect_training_format(&columns);

            seen.insert(canonical);
            self.datasets.push(DatasetEntry {
                name,
                path,
                format: format.to_string(),
                size_bytes,
                sample_count,
                preview,
                source: source.clone(),
                columns,
                training_format,
            });
        }
    }

    /// Scan the HuggingFace hub cache for dataset repositories.
    ///
    /// Structure: {hub_cache}/datasets--{org}--{name}/snapshots/{hash}/[nested/]
    /// Uses a recursive walkdir (max depth 4) inside each snapshot to find
    /// .jsonl, .json, .parquet, .csv files in any subdirectory layout.
    fn scan_hf_datasets_cache(
        &mut self,
        cache_dir: &std::path::Path,
        seen: &mut HashSet<PathBuf>,
    ) {
        if !cache_dir.exists() {
            return;
        }

        let entries = match std::fs::read_dir(cache_dir) {
            Ok(e) => e,
            Err(_) => return,
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let dir_name = path.file_name().unwrap_or_default().to_string_lossy();
            if !dir_name.starts_with("datasets--") {
                continue;
            }

            let dataset_id = dir_name
                .strip_prefix("datasets--")
                .unwrap_or(&dir_name)
                .replace("--", "/");

            // Find the latest snapshot directory
            let snapshots_dir = path.join("snapshots");
            let snapshot = if snapshots_dir.exists() {
                std::fs::read_dir(&snapshots_dir)
                    .ok()
                    .and_then(|entries| {
                        entries
                            .flatten()
                            .filter(|e| e.path().is_dir())
                            .max_by_key(|e| {
                                e.metadata()
                                    .and_then(|m| m.modified())
                                    .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
                            })
                            .map(|e| e.path())
                    })
                    .unwrap_or_else(|| path.clone())
            } else {
                path.clone()
            };

            // Recursively walk the snapshot (max depth 4) to find dataset files
            // in any nested layout (root, data/, default/train/, split/, etc.)
            for file_entry in walkdir::WalkDir::new(&snapshot)
                .max_depth(4)
                .into_iter()
                .flatten()
            {
                let file_path = file_entry.path();
                if !file_path.is_file() {
                    continue;
                }

                let canonical = file_path.canonicalize().unwrap_or_else(|_| file_path.to_path_buf());
                if seen.contains(&canonical) {
                    continue;
                }

                let ext = file_path
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("")
                    .to_lowercase();

                let format = match ext.as_str() {
                    "jsonl" => "JSONL",
                    "json" => "JSON",
                    "parquet" => "Parquet",
                    "csv" => "CSV",
                    _ => continue,
                };

                // Build a display name: dataset_id/relative_path_from_snapshot
                let rel = file_path
                    .strip_prefix(&snapshot)
                    .unwrap_or(file_path);
                let name = format!("{}/{}", dataset_id, rel.display());

                let size_bytes = file_path.metadata().map(|m| m.len()).unwrap_or(0);
                let sample_count = if ext == "jsonl" { count_lines(file_path) } else { None };
                let preview = read_first_line(file_path);

                let columns = detect_columns(file_path, format);
                let training_format = detect_training_format(&columns);

                seen.insert(canonical);
                self.datasets.push(DatasetEntry {
                    name,
                    path: file_path.to_path_buf(),
                    format: format.to_string(),
                    size_bytes,
                    sample_count,
                    preview,
                    source: DatasetSource::HfCache,
                    columns,
                    training_format,
                });
            }
        }
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
        let i = self.table_state.selected().map_or(0, |i| (i + count - 1) % count);
        self.table_state.select(Some(i));
        self.scrollbar_state = self.scrollbar_state.position(i);
    }

    pub fn selected_dataset(&self) -> Option<&DatasetEntry> {
        self.table_state.selected().and_then(|i| self.datasets.get(i))
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

        if self.datasets.is_empty() {
            Paragraph::new("\n  No datasets found.\n\n  Scanned: ./, ./data/, ./datasets/, HF cache\n  Add dir: press 'a' to add a custom dataset directory")
                .style(THEME.text_muted)
                .block(block)
                .render(area, buf);
            return;
        }

        let header = Row::new(vec!["Name", "Format", "Size", "Status"])
            .style(THEME.table_header)
            .height(1);

        let rows: Vec<Row> = self
            .datasets
            .iter()
            .enumerate()
            .map(|(i, ds)| {
                let style = if i % 2 == 0 {
                    THEME.table_row
                } else {
                    THEME.table_row_alt
                };
                let status_cell = if ds.training_format.is_ready() {
                    Cell::new("Ready").style(THEME.text_success)
                } else if ds.training_format == TrainingFormat::NeedsMapping {
                    Cell::new("Map cols").style(THEME.text_warning)
                } else {
                    Cell::new("Unknown").style(THEME.text_muted)
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

        let mut lines = vec![
            Line::from(vec![
                Span::styled("File:    ", THEME.kv_key),
                Span::styled(&ds.name, THEME.kv_value),
            ]),
            Line::from(vec![
                Span::styled("Format:  ", THEME.kv_key),
                Span::styled(&ds.format, THEME.kv_value),
            ]),
            Line::from(vec![
                Span::styled("Size:    ", THEME.kv_key),
                Span::styled(ds.size_display(), THEME.kv_value),
            ]),
            Line::from(vec![
                Span::styled("Source:  ", THEME.kv_key),
                Span::styled(source_label, THEME.text_dim),
            ]),
            Line::from(vec![
                Span::styled("Path:    ", THEME.kv_key),
                Span::styled(ds.path.display().to_string(), THEME.text_dim),
            ]),
        ];

        if let Some(count) = ds.sample_count {
            lines.push(Line::from(vec![
                Span::styled("Samples: ", THEME.kv_key),
                Span::styled(count.to_string(), THEME.kv_value),
            ]));
        }

        // Training readiness
        let status_style = if ds.training_format.is_ready() {
            THEME.text_success
        } else if ds.training_format == TrainingFormat::NeedsMapping {
            THEME.text_warning
        } else {
            THEME.text_muted
        };
        lines.push(Line::from(vec![
            Span::styled("Status:  ", THEME.kv_key),
            Span::styled(ds.training_format.status_label(), status_style),
        ]));

        // Detected columns
        if !ds.columns.is_empty() {
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                "Columns:",
                THEME.kv_key,
            )));
            for col in &ds.columns {
                lines.push(Line::from(Span::styled(
                    format!("  - {col}"),
                    THEME.text_dim,
                )));
            }
        }

        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "First sample:",
            THEME.text_dim,
        )));

        if let Some(preview) = &ds.preview {
            // Pretty-print JSON if possible
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(preview) {
                if let Ok(pretty) = serde_json::to_string_pretty(&json) {
                    for line in pretty.lines().take(20) {
                        lines.push(Line::from(Span::styled(line.to_string(), THEME.text_dim)));
                    }
                } else {
                    lines.push(Line::from(Span::styled(preview.clone(), THEME.text_dim)));
                }
            } else {
                lines.push(Line::from(Span::styled(preview.clone(), THEME.text_dim)));
            }
        }

        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .render(inner, buf);
    }
}

fn count_lines(path: &std::path::Path) -> Option<usize> {
    use std::io::Read;
    // Read in chunks to avoid loading entire file into memory.
    // Cap at 64 MB to avoid blocking the UI thread on huge files.
    const MAX_SCAN_BYTES: u64 = 64 * 1024 * 1024;
    let file = std::fs::File::open(path).ok()?;
    let file_size = file.metadata().ok()?.len();
    let scan_size = file_size.min(MAX_SCAN_BYTES);

    let mut reader = std::io::BufReader::with_capacity(64 * 1024, file.take(scan_size));
    let mut buf = [0u8; 64 * 1024];
    let mut count = 0usize;
    loop {
        let n = reader.read(&mut buf).ok()?;
        if n == 0 {
            break;
        }
        count += buf[..n].iter().filter(|&&b| b == b'\n').count();
    }
    // Account for final line without trailing newline
    if file_size > 0 && count == 0 {
        count = 1;
    }
    if file_size > scan_size {
        // Extrapolate from the scanned portion
        let ratio = file_size as f64 / scan_size as f64;
        count = (count as f64 * ratio) as usize;
    }
    Some(count)
}

fn read_first_line(path: &std::path::Path) -> Option<String> {
    use std::io::BufRead;
    let file = std::fs::File::open(path).ok()?;
    let mut reader = std::io::BufReader::new(file);
    let mut line = String::new();
    reader.read_line(&mut line).ok()?;
    if line.len() > 500 {
        line.truncate(500);
        line.push_str("...");
    }
    Some(line.trim().to_string())
}

/// Config file path for persisting custom dataset directories.
fn custom_dirs_config_path() -> PathBuf {
    pmetal_hub::pmetal_cache_dir().join("custom_dataset_dirs.json")
}

/// Load custom dataset directories from config.
fn load_custom_dataset_dirs() -> Vec<PathBuf> {
    let path = custom_dirs_config_path();
    if !path.exists() {
        return Vec::new();
    }
    let Ok(content) = std::fs::read_to_string(&path) else {
        return Vec::new();
    };
    serde_json::from_str::<Vec<PathBuf>>(&content).unwrap_or_default()
}

/// Persist custom dataset directories to config.
fn save_custom_dataset_dirs(dirs: &[PathBuf]) {
    let path = custom_dirs_config_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(json) = serde_json::to_string_pretty(dirs) {
        let _ = std::fs::write(&path, json);
    }
}

/// Detect columns/keys from the first line of a dataset file.
fn detect_columns(path: &std::path::Path, format: &str) -> Vec<String> {
    match format {
        "JSONL" | "JSON" => detect_jsonl_columns(path),
        "Parquet" => detect_parquet_columns(path),
        _ => Vec::new(),
    }
}

/// Read the first JSONL line and extract top-level keys.
fn detect_jsonl_columns(path: &std::path::Path) -> Vec<String> {
    use std::io::BufRead;
    let file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };
    let mut reader = std::io::BufReader::new(file);
    let mut line = String::new();
    if reader.read_line(&mut line).unwrap_or(0) == 0 {
        return Vec::new();
    }
    let Ok(json) = serde_json::from_str::<serde_json::Value>(line.trim()) else {
        return Vec::new();
    };
    match json.as_object() {
        Some(obj) => obj.keys().cloned().collect(),
        None => Vec::new(),
    }
}

/// Read parquet schema to extract column names.
/// Uses a lightweight approach — reads first few bytes to find schema metadata.
fn detect_parquet_columns(path: &std::path::Path) -> Vec<String> {
    // Try reading as a JSON sidecar first (some HF datasets have _metadata)
    // Otherwise, parse parquet footer for column names
    // For now, use the file extension heuristic + limited binary parsing
    use std::io::{Read, Seek, SeekFrom};

    let mut file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };

    // Parquet files start with "PAR1" magic and end with "PAR1" + 4-byte footer length
    let mut magic = [0u8; 4];
    if file.read_exact(&mut magic).is_err() || &magic != b"PAR1" {
        return Vec::new();
    }

    // Read footer length from last 8 bytes (4 bytes length + 4 bytes magic)
    if file.seek(SeekFrom::End(-8)).is_err() {
        return Vec::new();
    }
    let mut footer_bytes = [0u8; 8];
    if file.read_exact(&mut footer_bytes).is_err() {
        return Vec::new();
    }
    if &footer_bytes[4..8] != b"PAR1" {
        return Vec::new();
    }
    let footer_len = u32::from_le_bytes([
        footer_bytes[0],
        footer_bytes[1],
        footer_bytes[2],
        footer_bytes[3],
    ]) as u64;

    // Read footer (Thrift-encoded schema)
    if file.seek(SeekFrom::End(-(8 + footer_len as i64))).is_err() {
        return Vec::new();
    }
    let mut footer = vec![0u8; footer_len as usize];
    if file.read_exact(&mut footer).is_err() {
        return Vec::new();
    }

    // Extract column names from the Thrift-encoded footer
    // Column names appear as string fields in the schema elements
    extract_parquet_column_names(&footer)
}

/// Extract column names from a parquet Thrift footer.
/// This is a best-effort parser that finds string fields in the schema.
fn extract_parquet_column_names(footer: &[u8]) -> Vec<String> {
    // Parquet footer is Thrift Compact Protocol.
    // The schema is a list of SchemaElement, each with a name field.
    // We look for valid UTF-8 strings that appear to be column names.
    // This is a heuristic — we skip the root schema element.
    let mut names = Vec::new();
    let mut i = 0;

    while i < footer.len() {
        // Look for Thrift string pattern: length (varint) followed by UTF-8 bytes
        // In compact protocol, strings are preceded by their length as a varint
        if let Some((str_len, varint_size)) = decode_varint(&footer[i..]) {
            if str_len > 0
                && str_len < 256
                && i + varint_size + str_len <= footer.len()
            {
                let start = i + varint_size;
                let end = start + str_len;
                if let Ok(s) = std::str::from_utf8(&footer[start..end]) {
                    // Heuristic: valid column names are alphanumeric + underscores
                    if !s.is_empty()
                        && s.chars()
                            .all(|c| c.is_alphanumeric() || c == '_' || c == '-' || c == '.')
                        && s.chars().next().map_or(false, |c| c.is_alphabetic() || c == '_')
                        && !names.contains(&s.to_string())
                    {
                        names.push(s.to_string());
                    }
                }
            }
        }
        i += 1;
    }

    // Skip the first entry (usually the root schema name like "schema" or "arrow_schema")
    if names.len() > 1 {
        names.remove(0);
    }
    names
}

/// Decode a varint from a byte slice. Returns (value, bytes_consumed).
fn decode_varint(data: &[u8]) -> Option<(usize, usize)> {
    if data.is_empty() {
        return None;
    }
    let mut result: usize = 0;
    let mut shift = 0;
    for (i, &byte) in data.iter().enumerate() {
        if i >= 5 {
            return None; // Too long for a reasonable string length
        }
        result |= ((byte & 0x7F) as usize) << shift;
        if byte & 0x80 == 0 {
            return Some((result, i + 1));
        }
        shift += 7;
    }
    None
}

/// Determine training format from detected columns.
fn detect_training_format(columns: &[String]) -> TrainingFormat {
    if columns.is_empty() {
        return TrainingFormat::Unknown;
    }

    let has = |name: &str| columns.iter().any(|c| c == name);

    // OpenAI Messages format
    if has("messages") {
        return TrainingFormat::Messages;
    }
    // ShareGPT format
    if has("conversations") {
        return TrainingFormat::Conversations;
    }
    // Alpaca format (instruction + output)
    if has("instruction") && has("output") {
        return TrainingFormat::Alpaca;
    }
    // Simple text format
    if has("text") {
        return TrainingFormat::Text;
    }
    // Has columns but none match known formats
    TrainingFormat::NeedsMapping
}
