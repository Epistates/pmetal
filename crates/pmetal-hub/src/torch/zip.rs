//! A minimal reader for the ZIP container PyTorch writes `.pt` / `.bin` files in.
//!
//! `torch.save` stores each entry uncompressed, so all this needs is the
//! central directory (name -> offset, size) and the local header's variable
//! lengths to find where an entry's bytes start. Archives past 4 GiB (a
//! sharded checkpoint's 5-10 GB shards) use ZIP64 records, which are read too.

use std::collections::HashMap;

use pmetal_core::{PMetalError, Result};

fn corrupt(what: &str) -> PMetalError {
    PMetalError::Serialization(format!("not a valid PyTorch checkpoint archive: {what}"))
}

fn le16(b: &[u8], at: usize) -> Result<u64> {
    b.get(
        at..at
            .checked_add(2)
            .ok_or_else(|| corrupt("offset overflow"))?,
    )
    .map(|s| u64::from(u16::from_le_bytes([s[0], s[1]])))
    .ok_or_else(|| corrupt("truncated"))
}

fn le32(b: &[u8], at: usize) -> Result<u64> {
    b.get(
        at..at
            .checked_add(4)
            .ok_or_else(|| corrupt("offset overflow"))?,
    )
    .map(|s| u64::from(u32::from_le_bytes([s[0], s[1], s[2], s[3]])))
    .ok_or_else(|| corrupt("truncated"))
}

fn le64(b: &[u8], at: usize) -> Result<u64> {
    b.get(
        at..at
            .checked_add(8)
            .ok_or_else(|| corrupt("offset overflow"))?,
    )
    .map(|s| {
        let mut a = [0u8; 8];
        a.copy_from_slice(s);
        u64::from_le_bytes(a)
    })
    .ok_or_else(|| corrupt("truncated"))
}

fn usize_of(v: u64) -> Result<usize> {
    usize::try_from(v).map_err(|_| corrupt("a size does not fit in memory"))
}

/// Most entries an archive may hold: a billion-parameter model has a few
/// thousand tensors.
const MAX_ENTRIES: u64 = 4_000_000;

/// The entries of a stored (uncompressed) ZIP archive held in memory (or
/// mapped).
pub(crate) struct StoredZip<'a> {
    bytes: &'a [u8],
    entries: HashMap<String, (usize, usize)>,
}

impl<'a> StoredZip<'a> {
    /// Index the archive in `bytes`.
    pub(crate) fn parse(bytes: &'a [u8]) -> Result<Self> {
        const EOCD_SIG: [u8; 4] = [0x50, 0x4b, 0x05, 0x06];
        const EOCD64_LOCATOR_SIG: [u8; 4] = [0x50, 0x4b, 0x06, 0x07];
        const EOCD64_SIG: [u8; 4] = [0x50, 0x4b, 0x06, 0x06];
        const CD_SIG: [u8; 4] = [0x50, 0x4b, 0x01, 0x02];
        const LOCAL_SIG: [u8; 4] = [0x50, 0x4b, 0x03, 0x04];

        // The end-of-central-directory record is within 64 KiB of the end.
        let last = bytes
            .len()
            .checked_sub(22)
            .ok_or_else(|| corrupt("too short"))?;
        let search_from = last.saturating_sub(65535);
        let eocd = (search_from..=last)
            .rev()
            .find(|&i| bytes.get(i..i + 4) == Some(&EOCD_SIG))
            .ok_or_else(|| corrupt("no end-of-central-directory record"))?;
        let mut count = le16(bytes, eocd + 10)?;
        let mut cd_offset = le32(bytes, eocd + 16)?;

        // ZIP64: the 16/32-bit fields hold a sentinel and the real values are
        // in the ZIP64 end record, found through the locator before the EOCD.
        if count == 0xFFFF || cd_offset == 0xFFFF_FFFF || le32(bytes, eocd + 12)? == 0xFFFF_FFFF {
            let locator = eocd
                .checked_sub(20)
                .filter(|&l| bytes.get(l..l + 4) == Some(&EOCD64_LOCATOR_SIG))
                .ok_or_else(|| corrupt("ZIP64 archive without a ZIP64 locator"))?;
            let record = usize_of(le64(bytes, locator + 8)?)?;
            if bytes.get(record..record.saturating_add(4)) != Some(&EOCD64_SIG) {
                return Err(corrupt("bad ZIP64 end record"));
            }
            count = le64(bytes, record + 32)?;
            cd_offset = le64(bytes, record + 48)?;
        }
        if count > MAX_ENTRIES {
            return Err(corrupt("implausibly many entries"));
        }

        let mut entries = HashMap::new();
        let mut at = usize_of(cd_offset)?;
        for _ in 0..count {
            if bytes.get(at..at.saturating_add(4)) != Some(&CD_SIG) {
                return Err(corrupt("bad central directory entry"));
            }
            let method = le16(bytes, at + 10)?;
            let mut size = le32(bytes, at + 24)?;
            let mut compressed = le32(bytes, at + 20)?;
            let name_len = usize_of(le16(bytes, at + 28)?)?;
            let extra_len = usize_of(le16(bytes, at + 30)?)?;
            let comment_len = usize_of(le16(bytes, at + 32)?)?;
            let mut local = le32(bytes, at + 42)?;
            let name_end = at
                .checked_add(46 + name_len)
                .ok_or_else(|| corrupt("offset overflow"))?;
            let name = bytes
                .get(at + 46..name_end)
                .ok_or_else(|| corrupt("truncated name"))?;
            let name = String::from_utf8_lossy(name).into_owned();

            // ZIP64 extended information: only the fields whose 32-bit value
            // is the sentinel are present, in this order.
            if size == 0xFFFF_FFFF || compressed == 0xFFFF_FFFF || local == 0xFFFF_FFFF {
                let extra = bytes
                    .get(name_end..name_end + extra_len)
                    .ok_or_else(|| corrupt("truncated extra field"))?;
                let (mut s, mut c, mut l) = (size, compressed, local);
                let mut p = 0;
                while p + 4 <= extra.len() {
                    let id = le16(extra, p)?;
                    let len = usize_of(le16(extra, p + 2)?)?;
                    if id == 0x0001 {
                        let mut q = p + 4;
                        for field in [&mut s, &mut c, &mut l] {
                            if *field == 0xFFFF_FFFF {
                                *field = le64(extra, q)?;
                                q += 8;
                            }
                        }
                        break;
                    }
                    p += 4 + len;
                }
                (size, compressed, local) = (s, c, l);
            }
            if method != 0 {
                return Err(PMetalError::Serialization(format!(
                    "checkpoint entry `{name}` is compressed (method {method}); only stored entries are supported"
                )));
            }
            if size != compressed {
                return Err(corrupt("a stored entry has two different sizes"));
            }
            let local = usize_of(local)?;
            if bytes.get(local..local.saturating_add(4)) != Some(&LOCAL_SIG) {
                return Err(corrupt("bad local header"));
            }
            let data_start = local
                .checked_add(30)
                .and_then(|v| v.checked_add(usize::try_from(le16(bytes, local + 26).ok()?).ok()?))
                .and_then(|v| v.checked_add(usize::try_from(le16(bytes, local + 28).ok()?).ok()?))
                .ok_or_else(|| corrupt("bad local header"))?;
            let size = usize_of(size)?;
            if data_start
                .checked_add(size)
                .is_none_or(|end| end > bytes.len())
            {
                return Err(corrupt("entry runs past the end of the file"));
            }
            entries.insert(name, (data_start, size));
            at = name_end
                .checked_add(extra_len)
                .and_then(|v| v.checked_add(comment_len))
                .ok_or_else(|| corrupt("offset overflow"))?;
        }
        Ok(Self { bytes, entries })
    }

    /// The bytes of the entry named `name`.
    pub(crate) fn entry(&self, name: &str) -> Option<&'a [u8]> {
        self.entries
            .get(name)
            .map(|&(start, size)| &self.bytes[start..start + size])
    }

    /// Where the entry named `name` sits in the archive: `(offset, length)`.
    pub(crate) fn entry_range(&self, name: &str) -> Option<(usize, usize)> {
        self.entries.get(name).copied()
    }

    /// The entry whose name ends with `/suffix` (or equals `suffix`), with the
    /// archive's top-level directory (everything before the first `/`)
    /// returned alongside.
    pub(crate) fn find_suffix(&self, suffix: &str) -> Option<(&str, &'a [u8])> {
        self.entries.iter().find_map(|(name, &(start, size))| {
            let root = name.split('/').next()?;
            (name.strip_prefix(root)?.strip_prefix('/')? == suffix)
                .then(|| (root, &self.bytes[start..start + size]))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::torch::testing::stored_zip as archive;

    #[test]
    fn reads_stored_entries() {
        let bytes = archive(
            &[("archive/data.pkl", b"hello"), ("archive/data/0", b"xy")],
            false,
        );
        let zip = StoredZip::parse(&bytes).unwrap();
        assert_eq!(zip.entry("archive/data.pkl"), Some(&b"hello"[..]));
        assert_eq!(zip.entry("archive/data/0"), Some(&b"xy"[..]));
        let (root, data) = zip.find_suffix("data.pkl").unwrap();
        assert_eq!((root, data), ("archive", &b"hello"[..]));
        assert!(zip.entry("archive/missing").is_none());
    }

    #[test]
    fn reads_zip64_archives() {
        let bytes = archive(&[("a/data.pkl", b"hello"), ("a/data/0", b"world!")], true);
        let zip = StoredZip::parse(&bytes).unwrap();
        assert_eq!(zip.entry("a/data.pkl"), Some(&b"hello"[..]));
        assert_eq!(zip.entry("a/data/0"), Some(&b"world!"[..]));
    }

    #[test]
    fn compressed_entries_are_refused_with_their_name() {
        let mut bytes = archive(&[("a/data.pkl", b"hello")], false);
        // Method 8 (deflate) in the central directory entry.
        let cd = bytes
            .windows(4)
            .position(|w| w == [0x50, 0x4b, 0x01, 0x02])
            .unwrap();
        bytes[cd + 10] = 8;
        let err = StoredZip::parse(&bytes).err().unwrap().to_string();
        assert!(
            err.contains("compressed") && err.contains("a/data.pkl"),
            "{err}"
        );
    }

    #[test]
    fn garbage_and_truncation_are_errors_not_panics() {
        assert!(StoredZip::parse(b"not a zip file at all, not even close").is_err());
        assert!(StoredZip::parse(&[]).is_err());
        let bytes = archive(&[("a/data.pkl", b"hello world")], false);
        for cut in [1, 10, 30, bytes.len() - 5, bytes.len() - 1] {
            assert!(
                StoredZip::parse(&bytes[..bytes.len() - cut]).is_err(),
                "cut {cut}"
            );
        }
        // An offset that points past the file.
        let mut lie = bytes.clone();
        let cd = lie
            .windows(4)
            .position(|w| w == [0x50, 0x4b, 0x01, 0x02])
            .unwrap();
        lie[cd + 24..cd + 28].copy_from_slice(&u32::MAX.to_le_bytes());
        lie[cd + 20..cd + 24].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(StoredZip::parse(&lie).is_err());
    }
}
