//! Streaming shard reader for pretraining-scale data.
// memmap2::Mmap::map is inherently unsafe (external file can be modified while
// mapped). This is an accepted trade-off for mmap-based I/O; all other code in
// this file is safe. The workspace `unsafe_code = "deny"` lint is overridden
// here following the established pattern in pmetal-mlx/src/bridge.rs.
#![allow(unsafe_code)]
//!
//! # Shard format
//!
//! Each shard file is a sequence of variable-length document records:
//!
//! ```text
//! [ u32 doc_len ][ u32 token_0 ][ u32 token_1 ] ... [ u32 token_(doc_len-1) ]
//! [ u32 doc_len ][ u32 token_0 ] ...
//! ```
//!
//! All integers are little-endian. A `doc_len` of zero is silently skipped.
//!
//! # Packing strategy
//!
//! Documents are concatenated into fixed-length sequences separated by
//! `eos_token_id`. When a doc is larger than the remaining space in a sequence
//! only the first `space` tokens are taken; the cursor is left pointing to the
//! remainder so the next sequence picks it up without re-reading the header.
//!
//! # Resume
//!
//! The iterator yields a [`StreamPosition`] alongside every batch. The position
//! encodes both the cursor byte offset and `tokens_remaining_in_doc`, which
//! together fully describe the reader state at a batch boundary. Resuming with
//! this position reproduces the identical token stream.
//!
//! # Example
//!
//! ```no_run
//! use pmetal_data::streaming::{StreamConfig, StreamingShardReader};
//! use std::path::PathBuf;
//!
//! let cfg = StreamConfig {
//!     shard_paths: vec![PathBuf::from("shard_000.bin")],
//!     seq_len: 2048,
//!     batch_size: 4,
//!     eos_token_id: 2,
//!     resume_from: None,
//! };
//!
//! let mut reader = StreamingShardReader::new(cfg).unwrap();
//! let (batch, pos) = reader.next().unwrap();
//! // batch: Vec<Vec<u32>> with shape [batch_size][seq_len]
//! // pos: StreamPosition for checkpointing
//! ```

use std::fs::File;
use std::io;
use std::path::PathBuf;

use memmap2::Mmap;

// ─── Public types ─────────────────────────────────────────────────────────────

/// Position within the streaming corpus, used for deterministic resume.
///
/// Capture this from each yielded batch item and persist it to disk.
/// Pass it back as [`StreamConfig::resume_from`] to resume exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct StreamPosition {
    /// Index of the shard currently being read.
    pub shard_idx: usize,
    /// Byte offset within `shard_idx`.  When `tokens_remaining_in_doc > 0`,
    /// this points to the first byte of the remaining tokens of the
    /// partially-consumed document (the header has already been parsed).
    /// When `tokens_remaining_in_doc == 0`, this points to the next record
    /// header (or EOF).
    pub byte_offset: usize,
    /// Number of tokens left in the currently-open document.  Zero means the
    /// cursor is at a record boundary.
    pub tokens_remaining_in_doc: usize,
}

/// Configuration for the streaming shard reader.
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Ordered list of shard file paths.
    pub shard_paths: Vec<PathBuf>,
    /// Target sequence length (tokens) for each packed sequence.
    pub seq_len: usize,
    /// Number of sequences per yielded batch.
    pub batch_size: usize,
    /// Token ID inserted between documents within a packed sequence.
    pub eos_token_id: u32,
    /// Optional resume position.
    pub resume_from: Option<StreamPosition>,
}

// ─── Errors ───────────────────────────────────────────────────────────────────

/// Errors produced by the streaming reader.
#[derive(Debug, thiserror::Error)]
pub enum StreamError {
    /// No shard paths were provided.
    #[error("shard_paths must not be empty")]
    NoShards,
    /// A shard file could not be opened or memory-mapped.
    #[error("failed to open shard {path}: {source}")]
    ShardOpen {
        /// Path that failed.
        path: PathBuf,
        /// Underlying OS error.
        source: io::Error,
    },
    /// The resume position points beyond the end of the shard.
    #[error("resume byte_offset {offset} exceeds shard size {size} for shard {idx}")]
    ResumeOutOfBounds {
        /// Shard index.
        idx: usize,
        /// Requested byte offset.
        offset: usize,
        /// Actual shard byte length.
        size: usize,
    },
    /// A shard ends inside a record: a tokenization run that was interrupted
    /// or a copy that was cut short. Reading on would silently drop the
    /// tail of the corpus.
    #[error(
        "shard {path} is truncated: the record at byte {offset} claims {claimed} tokens \
         but only {available} bytes remain"
    )]
    TruncatedRecord {
        /// Path of the shard.
        path: PathBuf,
        /// Byte offset of the record's length header.
        offset: usize,
        /// Tokens the header claims.
        claimed: usize,
        /// Bytes left in the file after the header.
        available: usize,
    },
}

/// One row of a packed batch: `seq_len` tokens, and where the documents in it
/// begin and end.
///
/// A row is documents laid end to end, an end-of-sequence token after each
/// document that is followed by another in the row. `doc_lengths` has one
/// entry per document, or per fragment of one at the start of a row (the rest
/// of a document the previous row ended inside) or its end; each counts its
/// own end-of-sequence token, so the entries sum to `tokens.len()`. They are
/// what a block-diagonal attention mask and per-document positions need to
/// keep documents from attending to each other.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedRow {
    /// The row's tokens.
    pub tokens: Vec<u32>,
    /// Length of each document (fragment) in the row, its separator included.
    pub doc_lengths: Vec<usize>,
}

// ─── Internal shard cursor ────────────────────────────────────────────────────

/// Memory-mapped shard with record-level and token-level read state.
///
/// `byte_offset` always points to the next byte to read. When
/// `tokens_in_current_doc > 0`, that byte is the first token of the
/// remaining portion of the open document. When `tokens_in_current_doc == 0`,
/// that byte is the first byte of a `doc_len` record header (or EOF).
struct ShardCursor {
    path: PathBuf,
    mmap: Option<Mmap>,
    byte_offset: usize,
    tokens_in_current_doc: usize,
}

impl ShardCursor {
    fn new(path: PathBuf, byte_offset: usize, tokens_in_current_doc: usize) -> Self {
        Self {
            path,
            mmap: None,
            byte_offset,
            tokens_in_current_doc,
        }
    }

    fn ensure_mapped(&mut self) -> Result<(), StreamError> {
        if self.mmap.is_some() {
            return Ok(());
        }
        let file = File::open(&self.path).map_err(|e| StreamError::ShardOpen {
            path: self.path.clone(),
            source: e,
        })?;
        // SAFETY: read-only mapping; file must not be truncated while live.
        let mmap = unsafe {
            Mmap::map(&file).map_err(|e| StreamError::ShardOpen {
                path: self.path.clone(),
                source: e,
            })?
        };
        self.mmap = Some(mmap);
        Ok(())
    }

    fn mapped_len(&self) -> usize {
        self.mmap.as_ref().map_or(0, |m| m.len())
    }

    fn is_at_eof(&self) -> bool {
        self.mmap
            .as_ref()
            .is_none_or(|m| self.byte_offset >= m.len())
    }

    /// Parse the next non-empty `doc_len` record header.
    ///
    /// On success, `tokens_in_current_doc` is set to the document length and
    /// `byte_offset` points to the first token.  Returns `false` at the end of
    /// the shard.
    ///
    /// # Errors
    /// [`StreamError::TruncatedRecord`] when the shard ends inside a record
    /// (a header cut short, or fewer tokens than it claims).
    fn open_next_doc(&mut self) -> Result<bool, StreamError> {
        let data = match self.mmap.as_ref() {
            Some(m) => m.as_ref(),
            None => return Ok(false),
        };
        loop {
            if self.byte_offset == data.len() {
                return Ok(false);
            }
            let header_at = self.byte_offset;
            let truncated = |claimed: usize, available: usize| StreamError::TruncatedRecord {
                path: self.path.clone(),
                offset: header_at,
                claimed,
                available,
            };
            let Some(header) = data.get(header_at..header_at + 4) else {
                return Err(truncated(0, data.len() - header_at));
            };
            let doc_len = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
            self.byte_offset += 4;

            if doc_len == 0 {
                continue; // skip zero-length records
            }
            let available = data.len() - self.byte_offset;
            if doc_len * 4 > available {
                return Err(truncated(doc_len, available));
            }

            self.tokens_in_current_doc = doc_len;
            return Ok(true);
        }
    }

    /// Read up to `max_tokens` from the current document into `out`.
    ///
    /// Returns the number actually written (may be less if the doc ends).
    fn read_tokens(&mut self, max_tokens: usize, out: &mut Vec<u32>) -> usize {
        let take = max_tokens.min(self.tokens_in_current_doc);
        if take == 0 {
            return 0;
        }
        let data = match self.mmap.as_ref() {
            Some(m) => m.as_ref(),
            None => return 0,
        };
        let end = self.byte_offset + take * 4;
        out.extend(
            data[self.byte_offset..end]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| u32::from_le_bytes(*c)),
        );
        self.byte_offset += take * 4;
        self.tokens_in_current_doc -= take;
        take
    }
}

// ─── Streaming iterator ───────────────────────────────────────────────────────

/// Iterator that yields `(batch, position)` pairs.
///
/// The batch is a `Vec<Vec<u32>>` of shape `[batch_size][seq_len]`.
/// The position is suitable for checkpointing and resume.
///
/// The iterator wraps at epoch boundaries and is effectively infinite unless
/// the corpus is empty, in which case it returns `None` immediately.
pub struct StreamingShardReader {
    config: StreamConfig,
    shard_idx: usize,
    cursor: ShardCursor,
    /// True when we're at the start of a new packed sequence (no EOS before
    /// the first document fragment).
    at_sequence_start: bool,
}

impl StreamingShardReader {
    /// Create a new reader.  Opens the resume shard for validation if needed.
    pub fn new(config: StreamConfig) -> Result<Self, StreamError> {
        if config.shard_paths.is_empty() {
            return Err(StreamError::NoShards);
        }

        let (shard_idx, byte_offset, tokens_remaining) = match config.resume_from {
            Some(ref pos) => (
                pos.shard_idx % config.shard_paths.len(),
                pos.byte_offset,
                pos.tokens_remaining_in_doc,
            ),
            None => (0, 0, 0),
        };

        let mut cursor = ShardCursor::new(
            config.shard_paths[shard_idx].clone(),
            byte_offset,
            tokens_remaining,
        );

        if byte_offset > 0 || tokens_remaining > 0 {
            cursor.ensure_mapped()?;
            let shard_size = cursor.mapped_len();
            if byte_offset > shard_size {
                return Err(StreamError::ResumeOutOfBounds {
                    idx: shard_idx,
                    offset: byte_offset,
                    size: shard_size,
                });
            }
        }

        Ok(Self {
            config,
            shard_idx,
            cursor,
            at_sequence_start: true,
        })
    }

    /// Advance to the next shard (wraps at end).
    fn advance_shard(&mut self) -> Result<(), StreamError> {
        self.shard_idx = (self.shard_idx + 1) % self.config.shard_paths.len();
        self.cursor = ShardCursor::new(self.config.shard_paths[self.shard_idx].clone(), 0, 0);
        self.cursor.ensure_mapped()
    }

    /// Ensure `cursor.tokens_in_current_doc > 0`.
    ///
    /// Opens the next doc header (possibly spanning shard boundaries) until
    /// one is found.  Returns `false` only if the entire corpus is empty.
    fn ensure_doc(&mut self) -> Result<bool, StreamError> {
        if self.cursor.tokens_in_current_doc > 0 {
            return Ok(true);
        }

        let start_shard = self.shard_idx;
        let mut wrapped = false;

        loop {
            self.cursor.ensure_mapped()?;

            if !self.cursor.is_at_eof() && self.cursor.open_next_doc()? {
                return Ok(true);
            }

            // Shard exhausted — move on.
            self.advance_shard()?;

            if self.shard_idx == start_shard {
                if wrapped {
                    return Ok(false);
                }
                wrapped = true;
            }
        }
    }

    /// Capture the reader state as a [`StreamPosition`].
    pub fn checkpoint(&self) -> StreamPosition {
        StreamPosition {
            shard_idx: self.shard_idx,
            byte_offset: self.cursor.byte_offset,
            tokens_remaining_in_doc: self.cursor.tokens_in_current_doc,
        }
    }

    /// Fill one `seq_len`-length row from the document stream.
    fn fill_sequence(&mut self) -> Result<Option<PackedRow>, StreamError> {
        let seq_len = self.config.seq_len;
        let eos = self.config.eos_token_id;
        let mut seq = Vec::with_capacity(seq_len);
        let mut doc_lengths = Vec::new();
        let mut doc_start = 0;

        while seq.len() < seq_len {
            // Check whether the previous doc was exhausted (tokens == 0)
            // before we try to open the next one.
            let prev_exhausted = self.cursor.tokens_in_current_doc == 0;

            if !self.ensure_doc()? {
                return Ok(None); // empty corpus
            }

            // Insert EOS separator when transitioning from one doc to the
            // next, but not at the very start of a new packed sequence. The
            // separator closes the document before it.
            if prev_exhausted && !self.at_sequence_start && seq.len() < seq_len {
                seq.push(eos);
                doc_lengths.push(seq.len() - doc_start);
                doc_start = seq.len();
            }
            self.at_sequence_start = false;

            let space = seq_len - seq.len();
            self.cursor.read_tokens(space, &mut seq);
            // If tokens_in_current_doc == 0 after the read, the doc was fully
            // consumed; the next loop iteration opens the next one.
        }
        if seq.len() > doc_start {
            doc_lengths.push(seq.len() - doc_start);
        }

        self.at_sequence_start = true;
        Ok(Some(PackedRow {
            tokens: seq,
            doc_lengths,
        }))
    }

    /// The next `batch_size` rows with their document boundaries, and the
    /// position to resume after them. `None` only for an empty corpus: the
    /// stream wraps around at the end of the last shard.
    ///
    /// # Errors
    /// A shard that cannot be opened, or that is truncated mid-record. The
    /// rows of the batch read before the error are lost and the reader sits
    /// at the fault, so it fails again on the next call: a shard in that state
    /// has to be repaired or left out.
    pub fn try_next_rows(&mut self) -> Result<Option<RowsBatch>, StreamError> {
        let mut rows = Vec::with_capacity(self.config.batch_size);
        while rows.len() < self.config.batch_size {
            match self.fill_sequence()? {
                Some(row) => rows.push(row),
                None => return Ok(None),
            }
        }
        Ok(Some((rows, self.checkpoint())))
    }

    /// Like [`try_next_rows`](Self::try_next_rows), without the boundaries:
    /// `batch_size` rows of `seq_len` tokens.
    pub fn try_next_batch(&mut self) -> Result<Option<TokenBatch>, StreamError> {
        Ok(self
            .try_next_rows()?
            .map(|(rows, position)| (rows.into_iter().map(|row| row.tokens).collect(), position)))
    }
}

/// A batch of `[batch_size][seq_len]` tokens and the position to resume after
/// it.
pub type TokenBatch = (Vec<Vec<u32>>, StreamPosition);

/// A batch of [`PackedRow`]s and the position to resume after it.
pub type RowsBatch = (Vec<PackedRow>, StreamPosition);

impl Iterator for StreamingShardReader {
    type Item = TokenBatch;

    /// A shard error ends the iteration after logging it, which is
    /// indistinguishable from an empty corpus: use
    /// [`try_next_batch`](StreamingShardReader::try_next_batch) to see it.
    fn next(&mut self) -> Option<Self::Item> {
        match self.try_next_batch() {
            Ok(batch) => batch,
            Err(e) => {
                tracing::error!("StreamingShardReader: {e}");
                None
            }
        }
    }
}

// ─── Shard writer helper ──────────────────────────────────────────────────────

/// What [`tokenize_corpus`] wrote.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CorpusStats {
    /// Documents tokenized into shards.
    pub documents: u64,
    /// Tokens in them.
    pub tokens: u64,
    /// Shard files written.
    pub shards: usize,
    /// Rows left out: unreadable, without the text column, or empty.
    pub skipped: u64,
}

/// Tokenize the text column of a dataset into shards a
/// [`StreamingShardReader`] reads.
///
/// `input` is anything [`crate::rows`] reads: JSONL, JSON, Parquet, Arrow,
/// CSV, or a directory of shards. Rows are read and tokenized (in parallel)
/// in bounded chunks and written out `docs_per_shard` documents at a time as
/// `shard_00000.bin`, `shard_00001.bin`, ..., so memory holds a shard's tokens
/// and a chunk of text whatever the corpus size. Tokens are encoded without
/// special tokens: a pretraining loop adds its separators when it packs.
///
/// A row that cannot be read, has no text in `text_column`, or tokenizes to
/// nothing is skipped and counted (a few bad lines are routine in a
/// web-scale corpus); a corpus where every row is bad is an error.
/// `progress` is called with the number of rows read so far after each chunk.
///
/// # Errors
/// An unreadable input, a tokenizer failure, a shard that cannot be written,
/// or a corpus with no usable document.
pub fn tokenize_corpus(
    input: impl AsRef<std::path::Path>,
    output_dir: &std::path::Path,
    tokenizer: &tokenizers::Tokenizer,
    text_column: &str,
    docs_per_shard: usize,
    mut progress: impl FnMut(u64),
) -> pmetal_core::Result<CorpusStats> {
    use crate::rows::RowReader;
    use pmetal_core::PMetalError;

    const CHUNK_ROWS: usize = 2048;
    let docs_per_shard = docs_per_shard.max(1);
    std::fs::create_dir_all(output_dir)?;

    let mut reader = RowReader::open(input)?;
    let mut stats = CorpusStats::default();
    let mut shard: Vec<Vec<u32>> = Vec::new();
    let mut rows_read = 0u64;
    let mut warned = 0usize;

    let flush = |shard: &mut Vec<Vec<u32>>, index: usize| -> pmetal_core::Result<()> {
        let docs: Vec<&[u32]> = shard.iter().map(Vec::as_slice).collect();
        let path = output_dir.join(format!("shard_{index:05}.bin"));
        write_shard(&path, &docs)?;
        shard.clear();
        Ok(())
    };

    loop {
        let mut texts: Vec<String> = Vec::with_capacity(CHUNK_ROWS);
        let mut chunk_rows = 0usize;
        for row in reader.by_ref().take(CHUNK_ROWS) {
            chunk_rows += 1;
            match row {
                Ok(row) => match row.get(text_column).and_then(|v| v.as_str()) {
                    Some(text) if !text.is_empty() => texts.push(text.to_owned()),
                    _ => stats.skipped += 1,
                },
                Err(e) => {
                    if warned < 5 {
                        tracing::warn!("Skipping a row: {e}");
                        warned += 1;
                    }
                    stats.skipped += 1;
                }
            }
        }
        if chunk_rows == 0 {
            break;
        }
        rows_read += chunk_rows as u64;

        let encodings = tokenizer
            .encode_batch(texts, false)
            .map_err(|e| PMetalError::Tokenizer(e.to_string()))?;
        for encoding in encodings {
            let ids = encoding.get_ids();
            if ids.is_empty() {
                stats.skipped += 1;
                continue;
            }
            stats.tokens += ids.len() as u64;
            stats.documents += 1;
            shard.push(ids.to_vec());
            if shard.len() >= docs_per_shard {
                flush(&mut shard, stats.shards)?;
                stats.shards += 1;
            }
        }
        progress(rows_read);
    }
    if !shard.is_empty() {
        flush(&mut shard, stats.shards)?;
        stats.shards += 1;
    }
    if stats.documents == 0 {
        return Err(PMetalError::InvalidArgument(format!(
            "no document to tokenize: {} row(s) read, none with text in column '{text_column}'",
            rows_read
        )));
    }
    Ok(stats)
}

/// Write a sequence of documents to a shard file.
///
/// Each document is encoded as `[u32 doc_len][u32 token_0 .. u32 token_N-1]`
/// in little-endian byte order.
///
/// The shard appears at `path` complete or not at all: it is written beside
/// it and renamed into place, so a run interrupted mid-write leaves no
/// truncated shard for a reader to trip over.
///
/// # Errors
/// An I/O error, or a document of more than `u32::MAX` tokens (its length
/// would not fit the record header).
pub fn write_shard(path: &std::path::Path, docs: &[&[u32]]) -> io::Result<()> {
    use std::io::Write;
    let mut partial = path.as_os_str().to_owned();
    partial.push(".partial");
    let partial = PathBuf::from(partial);
    let write = || -> io::Result<()> {
        let mut f = std::io::BufWriter::new(File::create(&partial)?);
        for doc in docs {
            let len = u32::try_from(doc.len()).map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "a document of {} tokens does not fit a shard record",
                        doc.len()
                    ),
                )
            })?;
            f.write_all(&len.to_le_bytes())?;
            for &tok in *doc {
                f.write_all(&tok.to_le_bytes())?;
            }
        }
        f.flush()?;
        f.get_ref().sync_all()
    };
    match write().and_then(|()| std::fs::rename(&partial, path)) {
        Ok(()) => Ok(()),
        Err(e) => {
            let _ = std::fs::remove_file(&partial);
            Err(e)
        }
    }
}
