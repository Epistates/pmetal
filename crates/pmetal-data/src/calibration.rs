//! Calibration chunks: text to the fixed-length token chunks an
//! importance-matrix run feeds a model.
//!
//! A calibration set is not a training set. Nothing is masked, nothing is
//! padded and no sample is a unit: the documents are tokenized, laid end to
//! end as one stream and cut into non-overlapping chunks of the context
//! length the run uses, so every chunk is full and the statistics are taken
//! over as many tokens as the data has. When the tokenizer puts a
//! beginning-of-sequence token in front of text, each chunk starts with one
//! (the model never sees a context that does not, and its first-token
//! attention sink is part of the activations being measured).
//!
//! The source is a plain text file (one document) or any dataset file
//! [`TrainingDataset`] reads. Rows of a dataset are rendered as they would be
//! for training: conversations through the chat template when one is given,
//! text columns as they are.

use std::path::Path;

use pmetal_core::{PMetalError, Result};

use crate::chat_templates::ChatTemplate;
use crate::dataset::{DatasetFormat, TrainingDataset};
use crate::tokenizer::Tokenizer;

/// How to cut a calibration set.
#[derive(Debug, Clone, Copy)]
pub struct CalibrationOptions<'a> {
    /// Tokens per chunk: the context length each forward runs at.
    pub chunk_size: usize,
    /// Use at most this many chunks (`None`: all the data makes).
    pub max_chunks: Option<usize>,
    /// Chat template for conversational rows (`None`: plain `Role: text`).
    pub template: Option<&'a ChatTemplate>,
}

/// A calibration set, cut.
#[derive(Debug, Clone)]
pub struct CalibrationSet {
    /// The chunks. All are `chunk_size` tokens long except possibly the last.
    pub chunks: Vec<Vec<u32>>,
    /// Documents (rows, or one for a text file) the data held.
    pub documents: usize,
    /// Tokens in the data before it was cut (and before `max_chunks`).
    pub total_tokens: usize,
}

/// Read, tokenize and cut the calibration data at `path`.
///
/// # Errors
///
/// The file cannot be read or parsed, the tokenizer fails, or the data has no
/// tokens.
pub fn load_calibration_chunks(
    path: &Path,
    tokenizer: &Tokenizer,
    options: &CalibrationOptions<'_>,
) -> Result<CalibrationSet> {
    if options.chunk_size < 2 {
        return Err(PMetalError::InvalidArgument(
            "a calibration chunk needs at least 2 tokens".into(),
        ));
    }
    let is_text = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| matches!(e.to_ascii_lowercase().as_str(), "txt" | "text" | "md"));
    let samples = if is_text {
        let text = std::fs::read_to_string(path).map_err(|e| {
            PMetalError::InvalidArgument(format!("cannot read {}: {e}", path.display()))
        })?;
        vec![crate::dataset::TextSample::document(text)]
    } else {
        TrainingDataset::load_jsonl_text(path, DatasetFormat::Auto, options.template)?
    };

    let documents = samples.len();
    let mut docs: Vec<Vec<u32>> = Vec::with_capacity(documents);
    for sample in &samples {
        // Text a chat template rendered carries its own special tokens;
        // anything else is encoded as `tokenizer(text)` would.
        let ids = if sample.templated {
            tokenizer.encode(&sample.text)?
        } else {
            tokenizer.encode_with_special_tokens(&sample.text)?
        };
        if !ids.is_empty() {
            docs.push(ids);
        }
    }
    let total_tokens = docs.iter().map(Vec::len).sum();

    // The model's own beginning-of-sequence token, if it expects one: plain
    // text came back with it in front.
    let bos = tokenizer
        .bos_token_id()
        .filter(|&b| docs.first().is_some_and(|d| d.first() == Some(&b)));
    let chunks = chunk_documents(docs, bos, options.chunk_size, options.max_chunks);
    if chunks.is_empty() {
        return Err(PMetalError::InvalidArgument(format!(
            "{} has no tokens to calibrate on",
            path.display()
        )));
    }
    Ok(CalibrationSet {
        chunks,
        documents,
        total_tokens,
    })
}

/// Lay `docs` end to end and cut the stream into chunks of `chunk_size`
/// tokens, at most `max_chunks` of them.
///
/// With `bos`, the first token of every chunk is that token (replacing the
/// stream's own token there, as the first token of a chunk is the one with no
/// left context anyway). A trailing partial chunk is kept when it is at least
/// half a chunk, or when it is all the data there is.
pub fn chunk_documents(
    docs: Vec<Vec<u32>>,
    bos: Option<u32>,
    chunk_size: usize,
    max_chunks: Option<usize>,
) -> Vec<Vec<u32>> {
    let stream: Vec<u32> = docs.into_iter().flatten().collect();
    let limit = max_chunks.unwrap_or(usize::MAX);
    let mut chunks: Vec<Vec<u32>> = Vec::new();
    for piece in stream.chunks(chunk_size.max(1)) {
        if chunks.len() >= limit {
            break;
        }
        let partial = piece.len() < chunk_size;
        if partial && !chunks.is_empty() && piece.len() * 2 < chunk_size {
            break;
        }
        let mut chunk = piece.to_vec();
        if let (Some(bos), Some(first)) = (bos, chunk.first_mut()) {
            *first = bos;
        }
        chunks.push(chunk);
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::testing::word_level;

    #[test]
    fn the_stream_is_cut_into_full_chunks() {
        let docs = vec![vec![1, 2, 3], vec![4, 5, 6, 7], vec![8, 9, 10]];
        let chunks = chunk_documents(docs, None, 4, None);
        // 10 tokens: two full chunks and a half chunk (2 of 4), which is kept.
        assert_eq!(
            chunks,
            vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8], vec![9, 10]]
        );
    }

    #[test]
    fn a_small_remainder_is_dropped_unless_it_is_all_there_is() {
        assert_eq!(
            chunk_documents(vec![vec![1, 2, 3, 4, 5]], None, 4, None),
            vec![vec![1, 2, 3, 4]]
        );
        assert_eq!(
            chunk_documents(vec![vec![1, 2, 3]], None, 8, None),
            vec![vec![1, 2, 3]]
        );
        assert!(chunk_documents(vec![], None, 8, None).is_empty());
    }

    #[test]
    fn every_chunk_starts_with_the_bos_and_the_count_is_capped() {
        let docs = vec![vec![100, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]];
        let chunks = chunk_documents(docs, Some(100), 4, Some(2));
        assert_eq!(chunks, vec![vec![100, 1, 2, 3], vec![100, 5, 6, 7]]);
    }

    fn dir_with(name: &str, contents: &str) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(name);
        std::fs::write(&path, contents).unwrap();
        (dir, path)
    }

    fn options(chunk_size: usize) -> CalibrationOptions<'static> {
        CalibrationOptions {
            chunk_size,
            max_chunks: None,
            template: None,
        }
    }

    #[test]
    fn a_text_file_is_one_document_and_a_jsonl_file_one_per_row() {
        let tokenizer = word_level(&["<unk>", "a", "b", "c", "d"], &[]);
        let (_d, txt) = dir_with("c.txt", "a b c d a b c d");
        let set = load_calibration_chunks(&txt, &tokenizer, &options(4)).unwrap();
        assert_eq!(set.documents, 1);
        assert_eq!(set.total_tokens, 8);
        assert_eq!(set.chunks, vec![vec![1, 2, 3, 4], vec![1, 2, 3, 4]]);

        let (_d, jsonl) = dir_with(
            "c.jsonl",
            "{\"text\": \"a b c\"}\n{\"text\": \"d a b c d\"}\n",
        );
        let set = load_calibration_chunks(&jsonl, &tokenizer, &options(4)).unwrap();
        assert_eq!(set.documents, 2);
        assert_eq!(set.total_tokens, 8);
        assert_eq!(set.chunks, vec![vec![1, 2, 3, 4], vec![1, 2, 3, 4]]);

        let mut capped = options(4);
        capped.max_chunks = Some(1);
        let set = load_calibration_chunks(&jsonl, &tokenizer, &capped).unwrap();
        assert_eq!(set.chunks.len(), 1);
    }

    #[test]
    fn a_tokenizer_that_adds_a_bos_gets_one_at_the_start_of_every_chunk() {
        use crate::tokenizer::testing::byte_level_bpe;
        let tokenizer = byte_level_bpe(&[], &["<s>"], Some("<s>"));
        let bos = tokenizer.bos_token_id().expect("a bos");
        let (_d, txt) = dir_with("c.txt", "abcdefghijklmnopqrstuvwxyz");
        let set = load_calibration_chunks(&txt, &tokenizer, &options(8)).unwrap();
        assert!(set.chunks.len() >= 3);
        for chunk in &set.chunks {
            assert_eq!(chunk[0], bos);
        }
        // The first chunk's first token was the BOS already: nothing was lost there.
        assert_eq!(set.chunks[0][1], u32::from(b'a'));
    }

    #[test]
    fn a_model_whose_template_writes_the_bos_gets_one_at_the_start_of_every_chunk() {
        use crate::tokenizer::testing::byte_level_bpe;
        // The Gemma 4 layout: the tokenizer's post-processor adds no BOS and
        // the chat template does.
        let bare = byte_level_bpe(&[], &["<bos>"], None);
        let model = tempfile::tempdir().unwrap();
        bare.inner()
            .save(model.path().join("tokenizer.json"), false)
            .unwrap();
        std::fs::write(
            model.path().join("tokenizer_config.json"),
            r#"{"bos_token": "<bos>"}"#,
        )
        .unwrap();
        std::fs::write(
            model.path().join("chat_template.jinja"),
            "{{- bos_token -}}{%- for m in messages -%}{{ m['content'] }}{%- endfor -%}",
        )
        .unwrap();
        let tokenizer = Tokenizer::from_model_dir(model.path()).unwrap();
        let bos = tokenizer.bos_token_id().expect("a bos");

        let (_d, txt) = dir_with("c.txt", "abcdefghijklmnopqrstuvwxyz");
        let set = load_calibration_chunks(&txt, &tokenizer, &options(8)).unwrap();
        assert!(set.chunks.len() >= 3);
        for chunk in &set.chunks {
            assert_eq!(chunk[0], bos);
        }
        assert_eq!(set.chunks[0][1], u32::from(b'a'));
    }

    #[test]
    fn empty_data_and_tiny_chunks_are_errors() {
        let tokenizer = word_level(&["<unk>", "a"], &[]);
        let (_d, empty) = dir_with("c.txt", "");
        assert!(load_calibration_chunks(&empty, &tokenizer, &options(4)).is_err());
        let (_d, txt) = dir_with("c.txt", "a a a");
        assert!(load_calibration_chunks(&txt, &tokenizer, &options(1)).is_err());
        assert!(
            load_calibration_chunks(Path::new("/nonexistent/c.txt"), &tokenizer, &options(4))
                .is_err()
        );
    }
}
