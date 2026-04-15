#[cfg(feature = "trainer")]
pub(crate) mod bench;
pub(crate) mod dataset;
pub(crate) mod dflash;
#[cfg(feature = "trainer")]
pub(crate) mod distill;
#[cfg(feature = "trainer")]
pub(crate) mod embed_train;
#[cfg(feature = "lora")]
pub(crate) mod eval;
#[cfg(feature = "lora")]
pub(crate) mod fuse;
#[cfg(feature = "trainer")]
pub(crate) mod grpo;
pub(crate) mod infer;
pub(crate) mod info;
#[cfg(feature = "merge")]
pub(crate) mod merge;
pub(crate) mod ollama;
pub(crate) mod quantize;
#[cfg(feature = "trainer")]
pub(crate) mod rlkd;
pub(crate) mod search;
#[cfg(feature = "serve")]
pub(crate) mod serve;

use pmetal_data::DatasetColumnConfig;

/// Build an optional `DatasetColumnConfig` from the column-selection CLI flags.
///
/// Returns `None` when none of the flags were supplied (standard behavior),
/// and `Some(cfg)` when at least one column specifier is present.
pub(crate) fn build_column_config(
    text_column: Option<String>,
    text_columns: Option<Vec<String>>,
    column_separator: String,
    prompt_column: Option<String>,
    response_column: Option<String>,
) -> Option<DatasetColumnConfig> {
    if text_column.is_some()
        || text_columns.is_some()
        || prompt_column.is_some()
        || response_column.is_some()
    {
        Some(DatasetColumnConfig {
            text_column,
            text_columns,
            column_separator: Some(column_separator),
            prompt_column,
            response_column,
        })
    } else {
        None
    }
}
