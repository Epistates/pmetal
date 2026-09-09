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
    /// `byte_offset` points to the first token.  Returns `false` on EOF or
    /// truncation.
    fn open_next_doc(&mut self) -> bool {
        let data = match self.mmap.as_ref() {
            Some(m) => m.as_ref(),
            None => return false,
        };
        loop {
            if self.byte_offset + 4 > data.len() {
                return false;
            }
            let header: [u8; 4] = match data[self.byte_offset..self.byte_offset + 4].try_into() {
                Ok(b) => b,
                Err(_) => return false,
            };
            let doc_len = u32::from_le_bytes(header) as usize;
            self.byte_offset += 4;

            if doc_len == 0 {
                continue; // skip zero-length records
            }
            if self.byte_offset + doc_len * 4 > data.len() {
                return false; // truncated record → treat as EOF
            }

            self.tokens_in_current_doc = doc_len;
            return true;
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

            if !self.cursor.is_at_eof() && self.cursor.open_next_doc() {
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

    /// Fill one `seq_len`-length sequence from the document stream.
    fn fill_sequence(&mut self) -> Result<Option<Vec<u32>>, StreamError> {
        let seq_len = self.config.seq_len;
        let eos = self.config.eos_token_id;
        let mut seq = Vec::with_capacity(seq_len);

        while seq.len() < seq_len {
            // Check whether the previous doc was exhausted (tokens == 0)
            // before we try to open the next one.
            let prev_exhausted = self.cursor.tokens_in_current_doc == 0;

            if !self.ensure_doc()? {
                return Ok(None); // empty corpus
            }

            // Insert EOS separator when transitioning from one doc to the
            // next, but not at the very start of a new packed sequence.
            if prev_exhausted && !self.at_sequence_start && seq.len() < seq_len {
                seq.push(eos);
            }
            self.at_sequence_start = false;

            let space = seq_len - seq.len();
            self.cursor.read_tokens(space, &mut seq);
            // If tokens_in_current_doc == 0 after the read, the doc was fully
            // consumed; the next loop iteration opens the next one.
        }

        self.at_sequence_start = true;
        Ok(Some(seq))
    }
}

impl Iterator for StreamingShardReader {
    type Item = (Vec<Vec<u32>>, StreamPosition);

    fn next(&mut self) -> Option<Self::Item> {
        let mut batch = Vec::with_capacity(self.config.batch_size);
        while batch.len() < self.config.batch_size {
            match self.fill_sequence() {
                Ok(Some(seq)) => batch.push(seq),
                Ok(None) => return None,
                Err(e) => {
                    tracing::error!("StreamingShardReader: {e}");
                    return None;
                }
            }
        }
        let pos = self.checkpoint();
        Some((batch, pos))
    }
}

// ─── Shard writer helper ──────────────────────────────────────────────────────

/// Write a sequence of documents to a shard file.
///
/// Each document is encoded as `[u32 doc_len][u32 token_0 .. u32 token_N-1]`
/// in little-endian byte order.
pub fn write_shard(path: &std::path::Path, docs: &[&[u32]]) -> io::Result<()> {
    use std::io::Write;
    let mut f = std::io::BufWriter::new(File::create(path)?);
    for doc in docs {
        let len = doc.len() as u32;
        f.write_all(&len.to_le_bytes())?;
        for &tok in *doc {
            f.write_all(&tok.to_le_bytes())?;
        }
    }
    f.flush()
}
