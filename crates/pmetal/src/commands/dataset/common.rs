//! What the dataset commands share: the rows of a dataset in whatever file it
//! sits in, JSONL out, the model's tokenizer and chat template, and the
//! banners.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::Context;
use pmetal_data::chat_templates::{ChatTemplate, detect_chat_template};
use pmetal_data::rows::{Row, RowReader};
use pmetal_data::{DatasetFormat, DatasetSource, Tokenizer, resolve_dataset_source};

use crate::cli::dataset::TemplateName;

/// The rows of the dataset at `path`: a file in any format
/// [`pmetal_data::rows`] reads, or a directory (its train split).
pub(super) fn open_rows(path: &str) -> anyhow::Result<RowReader> {
    Ok(RowReader::open(path)?)
}

/// Every row of the dataset at `path`. The first row that cannot be read ends
/// the command, with the file and line it is on.
pub(super) fn read_all(path: &str) -> anyhow::Result<Vec<Row>> {
    open_rows(path)?
        .map(|row| row.map_err(anyhow::Error::from))
        .collect()
}

/// The rows of a dataset named by a local path, or by a Hugging Face ID, whose
/// `split` is fetched (as Parquet) into the Hub cache.
pub(super) async fn open_dataset(
    source: &str,
    split: &str,
    revision: Option<&str>,
) -> anyhow::Result<RowReader> {
    match resolve_dataset_source(source) {
        DatasetSource::Local(path) => Ok(RowReader::open(path)?),
        DatasetSource::HuggingFace(id) => open_hub_dataset(&id, split, revision).await,
    }
}

/// The rows of the `split` of the Hub dataset `id`.
pub(super) async fn open_hub_dataset(
    id: &str,
    split: &str,
    revision: Option<&str>,
) -> anyhow::Result<RowReader> {
    println!("Fetching the {split} split of {id} from the Hub...");
    let files = pmetal_hub::download_dataset_parquet(id, split, revision, None).await?;
    println!("Fetched {} file(s)", files.len());
    Ok(RowReader::from_files(files))
}

/// JSON Lines out: one row to a line.
pub(super) struct JsonlWriter<W: Write = BufWriter<File>> {
    out: W,
    rows: usize,
}

impl JsonlWriter {
    /// Create `path` (and the directory it is in), refusing a path that is
    /// not a place to write to. Done before the input is read, so a bad output
    /// path costs no work.
    pub(super) fn create(path: &str, context: &str) -> anyhow::Result<Self> {
        let path = crate::validate_output_path(path, context)?;
        let file = File::create(&path)
            .with_context(|| format!("could not create '{}'", path.display()))?;
        Ok(Self::new(BufWriter::new(file)))
    }
}

impl<W: Write> JsonlWriter<W> {
    pub(super) fn new(out: W) -> Self {
        Self { out, rows: 0 }
    }

    pub(super) fn write(&mut self, row: &Row) -> anyhow::Result<()> {
        serde_json::to_writer(&mut self.out, row)?;
        self.out.write_all(b"\n")?;
        self.rows += 1;
        Ok(())
    }

    /// The output, flushed.
    #[cfg(test)]
    pub(super) fn into_inner(mut self) -> anyhow::Result<W> {
        self.out.flush()?;
        Ok(self.out)
    }

    /// Flush, and say how many rows were written. A write error is reported
    /// here if it has not been already, not lost when the file is dropped.
    pub(super) fn finish(mut self) -> anyhow::Result<usize> {
        self.out.flush()?;
        Ok(self.rows)
    }
}

/// A row with one text column.
pub(super) fn text_row(text: String) -> Row {
    let mut row = Row::new();
    row.insert("text".into(), text.into());
    row
}

/// `--columns "target=source,target=source"` as the columns to rename:
/// source name to target name.
pub(super) fn parse_column_map(spec: &str) -> anyhow::Result<HashMap<String, String>> {
    let mut map = HashMap::new();
    for pair in spec.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        let (target, source) = pair
            .split_once('=')
            .map(|(t, s)| (t.trim(), s.trim()))
            .filter(|(t, s)| !t.is_empty() && !s.is_empty())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "--columns: '{pair}' is not target=source (e.g. \"text=content,prompt=question\")"
                )
            })?;
        if map.insert(source.to_string(), target.to_string()).is_some() {
            anyhow::bail!("--columns: the column '{source}' is renamed twice");
        }
    }
    Ok(map)
}

/// `row` with the columns of `map` (source to target) renamed, in place in the
/// column order.
pub(super) fn rename_columns(row: Row, map: &HashMap<String, String>) -> Row {
    row.into_iter()
        .map(|(name, value)| (map.get(&name).cloned().unwrap_or(name), value))
        .collect()
}

/// What a row's layout is called in a report.
pub(super) fn format_name(format: &DatasetFormat) -> &'static str {
    match format {
        DatasetFormat::Simple => "text",
        DatasetFormat::Alpaca => "alpaca",
        DatasetFormat::ShareGpt => "sharegpt",
        DatasetFormat::OpenAi => "messages",
        DatasetFormat::Reasoning => "reasoning",
        DatasetFormat::PromptCompletion => "prompt-completion",
        DatasetFormat::Auto => "auto",
        DatasetFormat::Custom { .. } => "custom",
    }
}

/// A model's tokenizer and chat template.
pub(super) struct ModelText {
    pub(super) tokenizer: Tokenizer,
    pub(super) template: ChatTemplate,
    /// Whether the tokenizer puts a BOS token in front of what it encodes.
    pub(super) adds_bos: bool,
}

impl ModelText {
    /// Read the tokenizer and chat template in the model directory `dir`;
    /// `name` is the model's name, which picks a built-in template for a
    /// model that has none of its own.
    pub(super) fn from_dir(dir: &Path, name: &str) -> anyhow::Result<Self> {
        let tokenizer = Tokenizer::from_model_dir(dir)
            .with_context(|| format!("could not load the tokenizer in '{}'", dir.display()))?;
        let template = detect_chat_template(dir, name);
        let adds_bos = tokenizer.bos_token_id().is_some_and(|bos| {
            tokenizer
                .encode_with_special_tokens("x")
                .is_ok_and(|ids| ids.first() == Some(&bos))
        });
        Ok(Self {
            tokenizer,
            template,
            adds_bos,
        })
    }

    /// Fetch (a Hub ID) or find (a directory) the model's tokenizer and chat
    /// template. Only those files are fetched, not the weights.
    pub(super) async fn load(model: &str) -> anyhow::Result<Self> {
        println!("Loading the tokenizer and chat template of {model}...");
        let dir = if Path::new(model).is_dir() {
            PathBuf::from(model)
        } else {
            pmetal_hub::download_tokenizer_with(model, &pmetal_hub::DownloadOptions::default())
                .await
                .map_err(|e| {
                    anyhow::anyhow!(
                        "'{model}' is not a model directory, and its tokenizer could not be \
                         fetched from the Hub: {e}"
                    )
                })?
        };
        Self::from_dir(&dir, model)
    }

    /// `rendered` (what `template` produced) as the text to store for a
    /// dataset that is trained on later.
    ///
    /// A text row is tokenized with the tokenizer's special tokens, so a BOS
    /// that the tokenizer adds must not also be written into the text, or the
    /// model would see two.
    pub(super) fn for_training(&self, rendered: &str, template: &ChatTemplate) -> String {
        if self.adds_bos {
            if let Some(bos) = template.bos_token.as_deref().filter(|b| !b.is_empty()) {
                if let Some(rest) = rendered.strip_prefix(bos) {
                    return rest.to_string();
                }
            }
        }
        rendered.to_string()
    }
}

/// The chat template a command formats with, and what to call it in output.
pub(super) struct TemplateChoice {
    /// `None` for no chat template: `Role: content` blocks.
    pub(super) template: Option<ChatTemplate>,
    pub(super) label: String,
}

/// The template `name` asks for: a built-in one, or the model's own. With no
/// name, the model's own, and chatml where there is no model.
pub(super) fn choose_template(
    name: Option<TemplateName>,
    model: Option<&ModelText>,
) -> anyhow::Result<TemplateChoice> {
    let own = |model: &ModelText| TemplateChoice {
        template: Some(model.template.clone()),
        label: if model.template.jinja_source.is_some() {
            "the model's own chat template".to_string()
        } else {
            format!(
                "{:?} (built in; the model has no chat template of its own)",
                model.template.template_type
            )
        },
    };
    let built_in = |template: ChatTemplate, name: &str| TemplateChoice {
        template: Some(template),
        label: format!("{name} (built in)"),
    };
    Ok(match (name, model) {
        (Some(TemplateName::Raw), _) => TemplateChoice {
            template: None,
            label: "none (Role: content blocks)".to_string(),
        },
        (Some(TemplateName::Auto) | None, Some(model)) => own(model),
        (Some(TemplateName::Auto), None) => {
            anyhow::bail!("--template auto uses the model's own chat template; give --model")
        }
        (None, None) => built_in(ChatTemplate::chatml(), "chatml"),
        (Some(TemplateName::Chatml), _) => built_in(ChatTemplate::chatml(), "chatml"),
        (Some(TemplateName::Llama3), _) => built_in(ChatTemplate::llama3(), "llama3"),
        (Some(TemplateName::Llama2), _) => built_in(ChatTemplate::llama2(), "llama2"),
        (Some(TemplateName::Mistral), _) => built_in(ChatTemplate::mistral(), "mistral"),
        (Some(TemplateName::Qwen), _) => built_in(ChatTemplate::qwen(), "qwen"),
        (Some(TemplateName::Phi), _) => built_in(ChatTemplate::phi3(), "phi"),
        (Some(TemplateName::Gemma), _) => built_in(ChatTemplate::gemma(), "gemma"),
    })
}

/// The part of `sorted` (ascending) below which a share `q` of it lies.
pub(super) fn percentile(sorted: &[usize], q: f64) -> usize {
    let at = ((sorted.len() as f64 * q) as usize).min(sorted.len().saturating_sub(1));
    sorted.get(at).copied().unwrap_or(0)
}

/// `part` as a percentage of `whole`; 0 for an empty whole.
pub(super) fn pct(part: usize, whole: usize) -> f64 {
    if whole == 0 {
        0.0
    } else {
        100.0 * part as f64 / whole as f64
    }
}

const RULE: &str = "========================================";

/// The heading of a command and what it was asked to do.
pub(super) fn banner(title: &str, fields: &[(&str, String)]) {
    println!("{RULE}\n  {title}\n{RULE}");
    for (name, value) in fields {
        println!("{:<13}{value}", format!("{name}:"));
    }
    println!("{RULE}\n");
}

/// The closing lines of a command: what it did.
pub(super) fn summary(title: &str, fields: &[(&str, String)]) {
    println!("\n{RULE}\n  {title}\n{RULE}");
    for (name, value) in fields {
        println!("{:<15}{value}", format!("{name}:"));
    }
    println!("{RULE}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_column_map_is_target_equals_source() {
        let map = parse_column_map("text = content, prompt=question").unwrap();
        assert_eq!(map["content"], "text");
        assert_eq!(map["question"], "prompt");
        assert_eq!(map.len(), 2);

        for bad in ["text", "=content", "text=", "a=b,a2=b"] {
            assert!(parse_column_map(bad).is_err(), "{bad}");
        }
        assert!(parse_column_map("").unwrap().is_empty());

        let row: Row = serde_json::from_str(r#"{"content": "x", "n": 1}"#).unwrap();
        let renamed = rename_columns(row, &map);
        assert_eq!(renamed.keys().collect::<Vec<_>>(), ["text", "n"]);
    }

    #[test]
    fn percentiles_never_leave_the_list() {
        assert_eq!(percentile(&[], 0.9), 0);
        assert_eq!(percentile(&[7], 0.99), 7);
        let v: Vec<usize> = (1..=100).collect();
        assert_eq!(percentile(&v, 0.5), 51);
        assert_eq!(percentile(&v, 0.99), 100);
        assert_eq!(pct(1, 0), 0.0);
    }

    #[test]
    fn a_template_is_the_one_asked_for_or_the_models() {
        assert!(choose_template(Some(TemplateName::Auto), None).is_err());
        let default = choose_template(None, None).unwrap();
        assert!(default.label.starts_with("chatml"));
        let raw = choose_template(Some(TemplateName::Raw), None).unwrap();
        assert!(raw.template.is_none());
    }
}
