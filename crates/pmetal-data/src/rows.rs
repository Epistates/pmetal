//! Rows of a dataset, whatever file they sit in.
//!
//! A dataset is a file or a directory of files in JSONL, JSON, Parquet, Arrow
//! (IPC file or stream, the `datasets` cache format) or CSV/TSV. [`RowReader`]
//! reads any of them as a stream of JSON objects, one per row, holding one
//! record batch (Parquet, Arrow, CSV) or one line (JSONL) in memory at a time,
//! so every format downstream (chat, instruction, preference, ...) is written
//! once, against [`Row`], and not once per container.
//!
//! [`resolve_files`] decides which files of a directory make up the dataset,
//! the way a Hugging Face dataset repository lays them out: the `train` split,
//! every shard of it, in order.
//!
//! Nested columns arrive as nested JSON (a Parquet `messages` column of
//! `list<struct<role, content>>` is the array an OpenAI-style JSONL row holds),
//! and a null is an absent key. Columns that hold raw bytes (image data) are
//! left out: they have no JSON form and a text pipeline has no use for them.

use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use pmetal_core::{PMetalError, Result};
use serde_json::Value;

/// One row of a dataset: a JSON object, columns in file order.
pub type Row = serde_json::Map<String, Value>;

/// Rows decoded from a record batch at a time.
const BATCH_ROWS: usize = 2048;

/// The container format of a dataset file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FileKind {
    /// One JSON object per line (`.jsonl`, `.ndjson`).
    Jsonl,
    /// A JSON array of objects, or an object wrapping one (`.json`).
    Json,
    /// Apache Parquet (`.parquet`).
    Parquet,
    /// Arrow IPC, file or stream (`.arrow`, `.ipc`, `.feather`).
    Arrow,
    /// Comma-separated values with a header row (`.csv`).
    Csv,
    /// Tab-separated values with a header row (`.tsv`).
    Tsv,
}

impl FileKind {
    /// The kind a file name says, or `None` for an extension that is not data.
    pub fn from_path(path: &Path) -> Option<Self> {
        let ext = path.extension()?.to_str()?.to_ascii_lowercase();
        Some(match ext.as_str() {
            "jsonl" | "ndjson" => Self::Jsonl,
            "json" => Self::Json,
            "parquet" => Self::Parquet,
            "arrow" | "ipc" | "feather" => Self::Arrow,
            "csv" => Self::Csv,
            "tsv" => Self::Tsv,
            _ => return None,
        })
    }

    /// Which kind wins when a directory holds several for the same split:
    /// lower is preferred.
    fn priority(self) -> u8 {
        match self {
            Self::Jsonl => 0,
            Self::Json => 1,
            Self::Parquet => 2,
            Self::Arrow => 3,
            Self::Csv => 4,
            Self::Tsv => 5,
        }
    }
}

fn invalid_data(message: impl Into<String>) -> PMetalError {
    PMetalError::Io(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        message.into(),
    ))
}

fn io_error(path: &Path, what: &str, e: std::io::Error) -> PMetalError {
    PMetalError::Io(std::io::Error::new(
        e.kind(),
        format!("Failed to {what} '{}': {e}", path.display()),
    ))
}

// -- choosing the files -------------------------------------------------------

/// Which split of a dataset a path belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Split {
    Train,
    Validation,
    Test,
    /// A name that says nothing about a split (`data.jsonl`).
    Unsplit,
}

/// The split a dataset file's path names, by the words Hugging Face's own
/// file-pattern matching reads: `train-00000-of-00004.parquet`,
/// `data/validation.jsonl`, `test/0000.arrow`.
fn split_of(relative: &Path) -> Split {
    let text = relative.to_string_lossy().to_ascii_lowercase();
    let words: Vec<&str> = text
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect();
    let has = |names: &[&str]| words.iter().any(|w| names.contains(w));
    if has(&["train", "training"]) {
        Split::Train
    } else if has(&["validation", "valid", "val", "dev"]) {
        Split::Validation
    } else if has(&["test", "testing", "eval", "evaluation"]) {
        Split::Test
    } else {
        Split::Unsplit
    }
}

/// Files that sit in a dataset directory without being rows.
fn is_metadata(name: &str) -> bool {
    name.starts_with('.')
        || name.starts_with("cache-")
        || matches!(
            name,
            "dataset_info.json"
                | "dataset_infos.json"
                | "state.json"
                | "croissant.json"
                | "metadata.json"
                | "config.json"
        )
}

/// Every candidate data file under `dir`, to `depth` levels, following
/// symlinks (a Hugging Face cache snapshot is a tree of links into `blobs/`).
fn collect_candidates(dir: &Path, depth: usize, out: &mut Vec<(PathBuf, FileKind)>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') || name == "__pycache__" {
            continue;
        }
        let Ok(meta) = std::fs::metadata(&path) else {
            continue; // a dangling link
        };
        if meta.is_dir() {
            if depth > 0 {
                collect_candidates(&path, depth - 1, out);
            }
        } else if !is_metadata(&name) {
            if let Some(kind) = FileKind::from_path(&path) {
                out.push((path, kind));
            }
        }
    }
}

/// The directory a dataset directory's files are read from: a Hugging Face
/// cache entry (`datasets--org--name/snapshots/<revision>/`) is its most
/// recent snapshot, so two revisions never both load.
fn dataset_root(dir: &Path) -> PathBuf {
    let snapshots = dir.join("snapshots");
    let newest = std::fs::read_dir(&snapshots).ok().and_then(|rd| {
        rd.flatten()
            .filter(|e| e.path().is_dir())
            .max_by_key(|e| {
                e.metadata()
                    .and_then(|m| m.modified())
                    .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
            })
            .map(|e| e.path())
    });
    newest.unwrap_or_else(|| dir.to_path_buf())
}

/// The files that make up the dataset at `path`, in reading order.
///
/// A file is itself. For a directory, the files under it (to four levels, the
/// depth of `data/<config>/train-0000-of-0004.parquet`) are grouped by split
/// from the words in their paths. The `train` split is chosen, or the files
/// that name no split if there is no `train`; failing both, whatever is there,
/// with a warning. Of the chosen files only the most preferred container kind
/// is kept (JSONL, JSON, Parquet, Arrow, CSV, TSV), so a repository that
/// ships the same rows twice loads them once, and they are read in
/// lexicographic path order, which is shard order.
///
/// # Errors
/// If the path does not exist, or a directory holds no data file.
pub fn resolve_files(path: &Path) -> Result<Vec<PathBuf>> {
    let meta = std::fs::metadata(path).map_err(|e| io_error(path, "read", e))?;
    if !meta.is_dir() {
        return Ok(vec![path.to_path_buf()]);
    }

    let root = dataset_root(path);
    let mut candidates = Vec::new();
    collect_candidates(&root, 4, &mut candidates);
    if candidates.is_empty() {
        return Err(PMetalError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!(
                "'{}' is a directory with no recognized data files (.jsonl, .json, .parquet, \
                 .arrow, .csv, .tsv). Pass a file path directly instead.",
                path.display()
            ),
        )));
    }

    let relative = |p: &Path| p.strip_prefix(&root).unwrap_or(p).to_path_buf();
    let of_split = |split: Split| -> Vec<(PathBuf, FileKind)> {
        candidates
            .iter()
            .filter(|(p, _)| split_of(&relative(p)) == split)
            .cloned()
            .collect()
    };
    let mut chosen = of_split(Split::Train);
    if chosen.is_empty() {
        chosen = of_split(Split::Unsplit);
    }
    if chosen.is_empty() {
        chosen = of_split(Split::Validation);
        if chosen.is_empty() {
            chosen = of_split(Split::Test);
        }
        tracing::warn!(
            "'{}' has no train split; reading its other files",
            path.display()
        );
    }

    let kind = chosen
        .iter()
        .map(|(_, k)| *k)
        .min_by_key(|k| k.priority())
        .expect("chosen is not empty");
    let mut files: Vec<PathBuf> = chosen
        .into_iter()
        .filter(|(_, k)| *k == kind)
        .map(|(p, _)| p)
        .collect();
    files.sort_by_key(|p| relative(p));
    tracing::info!(
        "Dataset directory '{}': reading {} file(s), first '{}'",
        path.display(),
        files.len(),
        files[0].display()
    );
    Ok(files)
}

// -- reading rows -------------------------------------------------------------

type RowIter = Box<dyn Iterator<Item = Result<Row>> + Send>;

/// Streams the rows of a dataset file or directory; see the [module
/// docs](self).
///
/// Yields `Err` once, then ends, for a file that cannot be opened or a row
/// that cannot be read; the message names the file and, for text formats, the
/// line.
pub struct RowReader {
    files: VecDeque<PathBuf>,
    all_files: Vec<PathBuf>,
    current: Option<RowIter>,
    failed: bool,
}

impl RowReader {
    /// Read the dataset at `path`, a file or a directory (see
    /// [`resolve_files`]).
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self::from_files(resolve_files(path.as_ref())?))
    }

    /// Read exactly `files`, in order.
    pub fn from_files(files: Vec<PathBuf>) -> Self {
        Self {
            files: files.iter().cloned().collect(),
            all_files: files,
            current: None,
            failed: false,
        }
    }

    /// The files this reader reads, in order.
    pub fn files(&self) -> &[PathBuf] {
        &self.all_files
    }
}

impl Iterator for RowReader {
    type Item = Result<Row>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.failed {
            return None;
        }
        loop {
            if self.current.is_none() {
                let path = self.files.pop_front()?;
                match open_rows(&path) {
                    Ok(rows) => self.current = Some(rows),
                    Err(e) => {
                        self.failed = true;
                        return Some(Err(e));
                    }
                }
            }
            match self.current.as_mut()?.next() {
                Some(Ok(row)) => return Some(Ok(row)),
                Some(Err(e)) => {
                    self.failed = true;
                    return Some(Err(e));
                }
                None => self.current = None,
            }
        }
    }
}

/// The binary container a file's first bytes announce, for a file whose name
/// says nothing (a temp file, a blob in a hub cache).
fn sniff(path: &Path) -> Option<FileKind> {
    use std::io::Read;
    let mut magic = [0u8; 8];
    let n = File::open(path).ok()?.read(&mut magic).ok()?;
    let magic = &magic[..n];
    if magic.starts_with(b"PAR1") {
        Some(FileKind::Parquet)
    } else if magic.starts_with(b"ARROW1") || magic.starts_with(&[0xff; 4]) {
        // The IPC file format, or the stream format's continuation marker.
        Some(FileKind::Arrow)
    } else {
        None
    }
}

fn open_rows(path: &Path) -> Result<RowIter> {
    // A name that says nothing is read by its first bytes: Parquet and Arrow
    // announce themselves, and anything else is JSONL, as it always was.
    let kind = FileKind::from_path(path)
        .or_else(|| sniff(path))
        .unwrap_or(FileKind::Jsonl);
    match kind {
        FileKind::Jsonl => JsonlRows::open(path).map(|r| Box::new(r) as RowIter),
        FileKind::Json => json_rows(path),
        FileKind::Parquet => ParquetRows::open(path).map(|r| Box::new(r) as RowIter),
        FileKind::Arrow => arrow_rows(path),
        FileKind::Csv => csv_rows(path, b','),
        FileKind::Tsv => csv_rows(path, b'\t'),
    }
}

/// Parse one JSON value as a row.
fn to_row(value: Value, location: &str) -> Result<Row> {
    match value {
        Value::Object(row) => Ok(row),
        other => Err(invalid_data(format!(
            "{location}: expected a JSON object, found {}",
            match other {
                Value::Array(_) => "an array",
                Value::String(_) => "a string",
                Value::Number(_) => "a number",
                Value::Bool(_) => "a boolean",
                _ => "null",
            }
        ))),
    }
}

/// JSON Lines, read a line at a time.
struct JsonlRows {
    path: PathBuf,
    reader: Box<dyn BufRead + Send>,
    buf: String,
    line: usize,
}

impl JsonlRows {
    fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(|e| io_error(path, "open dataset file", e))?;
        Ok(Self::from_reader(path, Box::new(BufReader::new(file))))
    }

    fn from_reader(path: &Path, reader: Box<dyn BufRead + Send>) -> Self {
        Self {
            path: path.to_path_buf(),
            reader,
            buf: String::new(),
            line: 0,
        }
    }
}

impl Iterator for JsonlRows {
    type Item = Result<Row>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            self.buf.clear();
            match self.reader.read_line(&mut self.buf) {
                Ok(0) => return None,
                Ok(_) => {}
                Err(e) => {
                    return Some(Err(io_error(
                        &self.path,
                        &format!("read line {} of", self.line + 1),
                        e,
                    )));
                }
            }
            self.line += 1;
            let mut text = self.buf.trim();
            if self.line == 1 {
                text = text.trim_start_matches('\u{feff}');
            }
            if text.is_empty() {
                continue;
            }
            let location = format!("{}:{}", self.path.display(), self.line);
            return Some(
                serde_json::from_str::<Value>(text)
                    .map_err(|e| invalid_data(format!("{location}: invalid JSON: {e}")))
                    .and_then(|value| to_row(value, &location)),
            );
        }
    }
}

/// Keys under which an object that wraps a dataset keeps its rows.
const WRAPPER_KEYS: &[&str] = &["data", "rows", "train", "examples", "items", "instances"];

/// A `.json` file: an array of objects, an object wrapping one (`{"data":
/// [...]}`), or a single object. A file that is several objects in a row (JSON
/// Lines under a `.json` name) is read as such.
fn json_rows(path: &Path) -> Result<RowIter> {
    let text = std::fs::read_to_string(path).map_err(|e| io_error(path, "read", e))?;
    let text_ref = text.trim_start_matches('\u{feff}');
    let value = match serde_json::from_str::<Value>(text_ref) {
        Ok(value) => value,
        Err(_) => {
            // Not one document: JSON Lines.
            let reader = std::io::Cursor::new(text.into_bytes());
            return Ok(Box::new(JsonlRows::from_reader(path, Box::new(reader))));
        }
    };
    let location = path.display().to_string();
    let items = match value {
        Value::Array(items) => items,
        Value::Object(map) => {
            let wrapped = WRAPPER_KEYS
                .iter()
                .find_map(|key| match map.get(*key) {
                    Some(Value::Array(items)) => Some(items.clone()),
                    _ => None,
                })
                .or_else(|| {
                    // The only array-valued key.
                    let mut arrays = map.values().filter_map(|v| v.as_array());
                    match (arrays.next(), arrays.next()) {
                        (Some(items), None) if items.iter().all(Value::is_object) => {
                            Some(items.clone())
                        }
                        _ => None,
                    }
                });
            wrapped.unwrap_or_else(|| vec![Value::Object(map)])
        }
        other => vec![other],
    };
    Ok(Box::new(items.into_iter().enumerate().map(
        move |(i, item)| to_row(item, &format!("{location}: row {}", i + 1)),
    )))
}

/// Does a column of this type have a JSON form? Raw bytes do not.
fn is_json_representable(data_type: &DataType) -> bool {
    match data_type {
        DataType::Binary
        | DataType::LargeBinary
        | DataType::FixedSizeBinary(_)
        | DataType::BinaryView => false,
        DataType::List(f)
        | DataType::LargeList(f)
        | DataType::FixedSizeList(f, _)
        | DataType::ListView(f)
        | DataType::LargeListView(f)
        | DataType::Map(f, _) => is_json_representable(f.data_type()),
        DataType::Struct(fields) => fields.iter().all(|f| is_json_representable(f.data_type())),
        DataType::Dictionary(_, value) => is_json_representable(value),
        _ => true,
    }
}

/// Append the rows of `batch` to `out`, through Arrow's JSON writer (nulls are
/// left out, so a missing value is a missing key).
fn push_batch_rows(batch: &RecordBatch, location: &str, out: &mut VecDeque<Row>) -> Result<()> {
    let mut buf = Vec::new();
    {
        let mut writer = arrow::json::LineDelimitedWriter::new(&mut buf);
        writer
            .write(batch)
            .and_then(|()| writer.finish())
            .map_err(|e| invalid_data(format!("{location}: cannot convert to JSON: {e}")))?;
    }
    for line in buf.split(|b| *b == b'\n').filter(|l| !l.is_empty()) {
        let value: Value = serde_json::from_slice(line)
            .map_err(|e| invalid_data(format!("{location}: invalid row: {e}")))?;
        out.push_back(to_row(value, location)?);
    }
    Ok(())
}

/// Rows from record batches, decoded a batch at a time.
struct BatchRows {
    location: String,
    batches: Box<dyn Iterator<Item = std::result::Result<RecordBatch, String>> + Send>,
    pending: VecDeque<Row>,
}

impl BatchRows {
    fn new<I, E>(path: &Path, batches: I) -> Self
    where
        I: Iterator<Item = std::result::Result<RecordBatch, E>> + Send + 'static,
        E: std::fmt::Display,
    {
        Self {
            location: path.display().to_string(),
            batches: Box::new(batches.map(|b| b.map_err(|e| e.to_string()))),
            pending: VecDeque::new(),
        }
    }
}

impl Iterator for BatchRows {
    type Item = Result<Row>;

    fn next(&mut self) -> Option<Self::Item> {
        while self.pending.is_empty() {
            match self.batches.next()? {
                Ok(batch) => {
                    if let Err(e) = push_batch_rows(&batch, &self.location, &mut self.pending) {
                        return Some(Err(e));
                    }
                }
                Err(e) => return Some(Err(invalid_data(format!("{}: {e}", self.location)))),
            }
        }
        self.pending.pop_front().map(Ok)
    }
}

/// Parquet, a row group's worth of batches at a time.
struct ParquetRows;

impl ParquetRows {
    fn open(path: &Path) -> Result<BatchRows> {
        let file = File::open(path).map_err(|e| io_error(path, "open", e))?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| {
            invalid_data(format!(
                "{}: not a readable Parquet file: {e}",
                path.display()
            ))
        })?;
        let keep: Vec<usize> = builder
            .schema()
            .fields()
            .iter()
            .enumerate()
            .filter(|(_, f)| is_json_representable(f.data_type()))
            .map(|(i, _)| i)
            .collect();
        let mask = ProjectionMask::roots(builder.parquet_schema(), keep);
        let reader = builder
            .with_projection(mask)
            .with_batch_size(BATCH_ROWS)
            .build()
            .map_err(|e| invalid_data(format!("{}: {e}", path.display())))?;
        Ok(BatchRows::new(path, reader))
    }
}

/// Arrow IPC: the file format, or the stream format `datasets` caches in.
fn arrow_rows(path: &Path) -> Result<RowIter> {
    use arrow::ipc::reader::{FileReader, StreamReader};

    let open = || File::open(path).map_err(|e| io_error(path, "open", e));
    let project = |schema: &arrow::datatypes::Schema| -> Vec<usize> {
        schema
            .fields()
            .iter()
            .enumerate()
            .filter(|(_, f)| is_json_representable(f.data_type()))
            .map(|(i, _)| i)
            .collect()
    };

    if let Ok(probe) = FileReader::try_new(open()?, None) {
        let keep = project(&probe.schema());
        let reader = FileReader::try_new(open()?, Some(keep))
            .map_err(|e| invalid_data(format!("{}: {e}", path.display())))?;
        return Ok(Box::new(BatchRows::new(path, reader)));
    }
    let probe = StreamReader::try_new(BufReader::new(open()?), None)
        .map_err(|e| invalid_data(format!("{}: not an Arrow file: {e}", path.display())))?;
    let keep = project(&probe.schema());
    let reader = StreamReader::try_new(BufReader::new(open()?), Some(keep))
        .map_err(|e| invalid_data(format!("{}: {e}", path.display())))?;
    Ok(Box::new(BatchRows::new(path, reader)))
}

/// CSV or TSV with a header row; column types are inferred from the head of
/// the file.
fn csv_rows(path: &Path, delimiter: u8) -> Result<RowIter> {
    use arrow::csv::reader::{Format, ReaderBuilder};

    let format = Format::default()
        .with_header(true)
        .with_delimiter(delimiter);
    let file = File::open(path).map_err(|e| io_error(path, "open", e))?;
    let (schema, _) = format
        .infer_schema(BufReader::new(file), Some(1000))
        .map_err(|e| invalid_data(format!("{}: {e}", path.display())))?;
    let file = File::open(path).map_err(|e| io_error(path, "open", e))?;
    let reader = ReaderBuilder::new(Arc::new(schema))
        .with_format(format)
        .with_batch_size(BATCH_ROWS)
        .build(file)
        .map_err(|e| invalid_data(format!("{}: {e}", path.display())))?;
    Ok(Box::new(BatchRows::new(path, reader)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn write(dir: &Path, name: &str, contents: &str) -> PathBuf {
        let path = dir.join(name);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, contents).unwrap();
        path
    }

    fn rows_of(path: &Path) -> Vec<Row> {
        RowReader::open(path)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap()
    }

    fn names(files: &[PathBuf], root: &Path) -> Vec<String> {
        files
            .iter()
            .map(|p| p.strip_prefix(root).unwrap().to_string_lossy().into_owned())
            .collect()
    }

    #[test]
    fn jsonl_skips_blank_lines_and_reads_crlf_and_a_bom() {
        let dir = tempfile::tempdir().unwrap();
        let path = write(
            dir.path(),
            "a.jsonl",
            "\u{feff}{\"a\": 1}\r\n\n  \n{\"a\": 2}\r\n{\"a\": 3}",
        );
        let rows = rows_of(&path);
        assert_eq!(
            rows.iter()
                .map(|r| r["a"].as_i64().unwrap())
                .collect::<Vec<_>>(),
            [1, 2, 3]
        );
    }

    #[test]
    fn a_bad_line_is_reported_with_its_file_and_line() {
        let dir = tempfile::tempdir().unwrap();
        let path = write(dir.path(), "a.jsonl", "{\"a\": 1}\n\n{broken\n{\"a\": 3}\n");
        let mut reader = RowReader::open(&path).unwrap();
        assert!(reader.next().unwrap().is_ok());
        let err = reader.next().unwrap().unwrap_err().to_string();
        assert!(err.contains("a.jsonl:3"), "{err}");
        assert!(reader.next().is_none(), "a reader stops at its first error");

        let path = write(dir.path(), "b.jsonl", "[1, 2]\n");
        let err = RowReader::open(&path).unwrap().next().unwrap().unwrap_err();
        assert!(err.to_string().contains("b.jsonl:1"), "{err}");
        assert!(err.to_string().contains("array"), "{err}");
    }

    #[test]
    fn json_files_are_arrays_wrapped_arrays_objects_or_json_lines() {
        let dir = tempfile::tempdir().unwrap();
        let array = write(dir.path(), "a.json", r#"[{"x": 1}, {"x": 2}]"#);
        assert_eq!(rows_of(&array).len(), 2);
        let wrapped = write(
            dir.path(),
            "b.json",
            r#"{"version": 1, "data": [{"x": 1}]}"#,
        );
        assert_eq!(rows_of(&wrapped).len(), 1);
        let lone = write(dir.path(), "c.json", r#"{"x": {"y": [1]}}"#);
        assert_eq!(rows_of(&lone).len(), 1);
        let lines = write(dir.path(), "d.json", "{\"x\": 1}\n{\"x\": 2}\n{\"x\": 3}\n");
        assert_eq!(rows_of(&lines).len(), 3);
        let bad = write(dir.path(), "e.json", r#"[{"x": 1}, 7]"#);
        let err = RowReader::open(&bad).unwrap().nth(1).unwrap().unwrap_err();
        assert!(err.to_string().contains("row 2"), "{err}");
    }

    /// Write `rows` (as JSONL text) out as a record batch.
    fn batch_of(jsonl: &str) -> RecordBatch {
        use arrow::json::ReaderBuilder;
        use arrow::json::reader::infer_json_schema_from_seekable;
        let mut cursor = std::io::Cursor::new(jsonl.as_bytes().to_vec());
        let (schema, _) = infer_json_schema_from_seekable(&mut cursor, None).unwrap();
        ReaderBuilder::new(Arc::new(schema))
            .build(std::io::Cursor::new(jsonl.as_bytes().to_vec()))
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
    }

    const NESTED: &str = concat!(
        r#"{"id": 1, "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}], "note": "x"}"#,
        "\n",
        r#"{"id": 2, "messages": [{"role": "user", "content": "q"}]}"#,
        "\n",
    );

    #[test]
    fn parquet_rows_keep_nested_columns_and_leave_nulls_out() {
        let dir = tempfile::tempdir().unwrap();
        let batch = batch_of(NESTED);
        let path = dir.path().join("a.parquet");
        let mut writer = parquet::arrow::ArrowWriter::try_new(
            File::create(&path).unwrap(),
            batch.schema(),
            None,
        )
        .unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let rows = rows_of(&path);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0]["messages"][1]["content"], json!("yo"));
        assert_eq!(rows[0]["note"], json!("x"));
        assert!(!rows[1].contains_key("note"), "a null is an absent key");
    }

    #[test]
    fn a_parquet_binary_column_is_left_out() {
        use arrow::array::{ArrayRef, BinaryArray, Int64Array};
        use arrow::datatypes::{Field, Schema};
        let dir = tempfile::tempdir().unwrap();
        let schema = Arc::new(Schema::new(vec![
            Field::new("n", DataType::Int64, false),
            Field::new("image", DataType::Binary, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(vec![7])) as ArrayRef,
                Arc::new(BinaryArray::from(vec![&b"\x89PNG"[..]])) as ArrayRef,
            ],
        )
        .unwrap();
        let path = dir.path().join("a.parquet");
        let mut writer =
            parquet::arrow::ArrowWriter::try_new(File::create(&path).unwrap(), schema, None)
                .unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        let rows = rows_of(&path);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["n"], json!(7));
        assert!(!rows[0].contains_key("image"));
    }

    #[test]
    fn arrow_ipc_files_and_streams_are_read() {
        let dir = tempfile::tempdir().unwrap();
        let batch = batch_of(NESTED);

        let file_path = dir.path().join("file.arrow");
        let mut writer = arrow::ipc::writer::FileWriter::try_new(
            File::create(&file_path).unwrap(),
            &batch.schema(),
        )
        .unwrap();
        writer.write(&batch).unwrap();
        writer.finish().unwrap();

        let stream_path = dir.path().join("stream.arrow");
        let mut writer = arrow::ipc::writer::StreamWriter::try_new(
            File::create(&stream_path).unwrap(),
            &batch.schema(),
        )
        .unwrap();
        writer.write(&batch).unwrap();
        writer.finish().unwrap();

        for path in [file_path, stream_path] {
            let rows = rows_of(&path);
            assert_eq!(rows.len(), 2, "{}", path.display());
            assert_eq!(rows[0]["messages"][0]["role"], json!("user"));
        }
    }

    #[test]
    fn csv_and_tsv_have_a_header_and_typed_columns() {
        let dir = tempfile::tempdir().unwrap();
        let csv = write(
            dir.path(),
            "a.csv",
            "q,a,n\nwhy,\"because, so\",3\nhow,by hand,4\n",
        );
        let rows = rows_of(&csv);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0]["a"], json!("because, so"));
        assert_eq!(rows[1]["n"], json!(4));
        let tsv = write(dir.path(), "a.tsv", "q\ta\nwhy\tbecause, so\n");
        assert_eq!(rows_of(&tsv)[0]["a"], json!("because, so"));
    }

    #[test]
    fn a_reader_goes_through_every_file_in_order() {
        let dir = tempfile::tempdir().unwrap();
        let a = write(dir.path(), "a.jsonl", "{\"i\": 1}\n{\"i\": 2}\n");
        let b = write(dir.path(), "b.jsonl", "{\"i\": 3}\n");
        let reader = RowReader::from_files(vec![a, b]);
        assert_eq!(reader.files().len(), 2);
        let all: Vec<i64> = reader.map(|r| r.unwrap()["i"].as_i64().unwrap()).collect();
        assert_eq!(all, [1, 2, 3]);
    }

    #[test]
    fn a_missing_file_is_an_error_not_an_empty_dataset() {
        let dir = tempfile::tempdir().unwrap();
        assert!(RowReader::open(dir.path().join("nope.jsonl")).is_err());
        assert!(RowReader::open(dir.path()).is_err(), "an empty directory");
    }

    #[test]
    fn the_train_split_is_chosen_and_every_shard_of_it_is_read_in_order() {
        let dir = tempfile::tempdir().unwrap();
        for name in [
            "data/train-00001-of-00003.parquet",
            "data/train-00000-of-00003.parquet",
            "data/train-00002-of-00003.parquet",
            "data/test-00000-of-00001.parquet",
            "data/validation-00000-of-00001.parquet",
            "README.md",
            "dataset_info.json",
        ] {
            write(dir.path(), name, "x");
        }
        let files = resolve_files(dir.path()).unwrap();
        assert_eq!(
            names(&files, dir.path()),
            [
                "data/train-00000-of-00003.parquet",
                "data/train-00001-of-00003.parquet",
                "data/train-00002-of-00003.parquet"
            ]
        );
    }

    #[test]
    fn a_directory_without_a_split_name_is_all_train() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "data.jsonl", "{}");
        write(dir.path(), "extra/more.jsonl", "{}");
        write(dir.path(), "test.jsonl", "{}");
        let files = resolve_files(dir.path()).unwrap();
        assert_eq!(
            names(&files, dir.path()),
            ["data.jsonl", "extra/more.jsonl"]
        );
    }

    #[test]
    fn one_container_kind_is_chosen_per_dataset() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "train.parquet", "x");
        write(dir.path(), "train.jsonl", "{}");
        write(dir.path(), "train.csv", "a\n1\n");
        let files = resolve_files(dir.path()).unwrap();
        assert_eq!(names(&files, dir.path()), ["train.jsonl"]);
    }

    #[test]
    fn a_directory_of_only_other_splits_still_loads() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "test.jsonl", "{}");
        write(dir.path(), "validation.jsonl", "{}");
        let files = resolve_files(dir.path()).unwrap();
        assert_eq!(names(&files, dir.path()), ["validation.jsonl"]);
    }

    #[test]
    fn the_huggingface_cache_reads_its_newest_snapshot_through_links() {
        use std::os::unix::fs::symlink;
        let dir = tempfile::tempdir().unwrap();
        let blobs = dir.path().join("blobs");
        std::fs::create_dir_all(&blobs).unwrap();
        let old_blob = write(&blobs, "1111", "{\"v\": \"old\"}\n");
        let new_blob = write(&blobs, "2222", "{\"v\": \"new\"}\n");
        for (rev, blob) in [("aaa", old_blob), ("bbb", new_blob)] {
            let snap = dir.path().join("snapshots").join(rev).join("data");
            std::fs::create_dir_all(&snap).unwrap();
            symlink(&blob, snap.join("train-00000-of-00001.jsonl")).unwrap();
            // Distinct modification times for the two snapshot directories.
            std::thread::sleep(std::time::Duration::from_millis(1100));
        }
        let rows = rows_of(dir.path());
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["v"], json!("new"));
    }

    #[test]
    fn cache_files_and_hidden_files_are_not_data() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "cache-1234.arrow", "x");
        write(dir.path(), ".hidden/train.jsonl", "{}");
        write(dir.path(), "state.json", "{}");
        assert!(resolve_files(dir.path()).is_err());
    }
}
