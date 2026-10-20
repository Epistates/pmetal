//! Following a file another process appends lines to.

use std::path::{Path, PathBuf};

use tokio::io::{AsyncReadExt, AsyncSeekExt};

/// Most bytes read from the file in one poll.
const CHUNK: u64 = 256 * 1024;
/// A "line" longer than this has no newline coming and is not a record.
const MAX_PENDING: usize = crate::log::MAX_LINE_BYTES * 4;

/// A reader positioned in a JSONL file that is still being written.
///
/// Each [`poll`](Self::poll) returns the complete lines appended since the last
/// one. A file that does not exist yet is not an error (the writer may not have
/// started), and one that shrank was rewritten, so it is read again from its
/// start.
#[derive(Debug)]
pub struct Tail {
    path: PathBuf,
    offset: u64,
    pending: Vec<u8>,
}

impl Tail {
    /// Follow `path` from its start.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            offset: 0,
            pending: Vec::new(),
        }
    }

    /// The file being followed.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Start over from the beginning of the file, forgetting a half-read line.
    pub fn rewind(&mut self) {
        self.offset = 0;
        self.pending.clear();
    }

    /// The complete lines appended since the last call. With `last`, a final
    /// line the writer never ended with a newline is returned too.
    pub async fn poll(&mut self, last: bool) -> Vec<String> {
        let mut lines = Vec::new();
        // The file may not exist yet, or briefly; that is not an error.
        if let Ok(bytes) = self.read_new().await {
            self.pending.extend_from_slice(&bytes);
            while let Some(end) = self.pending.iter().position(|&b| b == b'\n') {
                let line: Vec<u8> = self.pending.drain(..=end).collect();
                lines.push(String::from_utf8_lossy(&line[..line.len() - 1]).into_owned());
            }
            if last && !self.pending.is_empty() {
                lines.push(String::from_utf8_lossy(&self.pending).into_owned());
                self.pending.clear();
            }
            if self.pending.len() > MAX_PENDING {
                self.pending.clear();
            }
        }
        lines
    }

    async fn read_new(&mut self) -> std::io::Result<Vec<u8>> {
        let mut file = tokio::fs::File::open(&self.path).await?;
        let len = file.metadata().await?.len();
        if len < self.offset {
            self.rewind();
        }
        if len == self.offset {
            return Ok(Vec::new());
        }
        file.seek(std::io::SeekFrom::Start(self.offset)).await?;
        let mut buf = Vec::new();
        let read = file.take(CHUNK).read_to_end(&mut buf).await?;
        self.offset += read as u64;
        Ok(buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[tokio::test]
    async fn complete_lines_are_returned_once_and_a_partial_one_waits() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.jsonl");
        let mut tail = Tail::new(&path);
        // Not there yet.
        assert!(tail.poll(false).await.is_empty());

        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(b"one\ntwo\nthr").unwrap();
        assert_eq!(tail.poll(false).await, ["one", "two"]);
        file.write_all(b"ee\nfour").unwrap();
        assert_eq!(tail.poll(false).await, ["three"]);
        assert!(tail.poll(false).await.is_empty());
        // At the end of the job the unterminated line counts.
        assert_eq!(tail.poll(true).await, ["four"]);
    }

    #[tokio::test]
    async fn a_rewritten_file_is_read_from_its_start() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.jsonl");
        std::fs::write(&path, "aaaa\nbbbb\n").unwrap();
        let mut tail = Tail::new(&path);
        assert_eq!(tail.poll(false).await, ["aaaa", "bbbb"]);
        std::fs::write(&path, "c\n").unwrap();
        assert_eq!(tail.poll(false).await, ["c"]);
    }

    #[tokio::test]
    async fn a_runaway_line_with_no_newline_does_not_grow_memory() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.jsonl");
        std::fs::write(&path, vec![b'x'; MAX_PENDING * 3]).unwrap();
        let mut tail = Tail::new(&path);
        for _ in 0..8 {
            assert!(tail.poll(false).await.is_empty());
            assert!(tail.pending.len() <= MAX_PENDING + CHUNK as usize);
        }
    }
}
