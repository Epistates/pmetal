//! A minimal reader for the ZIP container PyTorch writes `.pt` files in.
//!
//! `torch.save` stores each entry uncompressed, so all this needs is the
//! central directory (name -> offset, size) and the local header's variable
//! lengths to find where an entry's bytes start.

use crate::error::{Result, VocoderError};
use std::collections::HashMap;

fn corrupt(what: &str) -> VocoderError {
    VocoderError::WeightLoad(format!("not a valid PyTorch checkpoint archive: {what}"))
}

fn le16(b: &[u8], at: usize) -> Result<usize> {
    b.get(at..at + 2)
        .map(|s| u16::from_le_bytes([s[0], s[1]]) as usize)
        .ok_or_else(|| corrupt("truncated"))
}

fn le32(b: &[u8], at: usize) -> Result<usize> {
    b.get(at..at + 4)
        .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]) as usize)
        .ok_or_else(|| corrupt("truncated"))
}

/// The entries of a stored (uncompressed) ZIP archive held in memory.
pub struct StoredZip<'a> {
    bytes: &'a [u8],
    entries: HashMap<String, (usize, usize)>,
}

impl<'a> StoredZip<'a> {
    /// Index the archive in `bytes`.
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        const EOCD_SIG: [u8; 4] = [0x50, 0x4b, 0x05, 0x06];
        const CD_SIG: [u8; 4] = [0x50, 0x4b, 0x01, 0x02];
        const LOCAL_SIG: [u8; 4] = [0x50, 0x4b, 0x03, 0x04];

        // The end-of-central-directory record is within 64 KiB of the end.
        let search_from = bytes.len().saturating_sub(22 + 65535);
        let eocd = (search_from..=bytes.len().saturating_sub(22))
            .rev()
            .find(|&i| bytes.get(i..i + 4) == Some(&EOCD_SIG))
            .ok_or_else(|| corrupt("no end-of-central-directory record"))?;
        let count = le16(bytes, eocd + 10)?;
        let cd_offset = le32(bytes, eocd + 16)?;
        if count == 0xFFFF || cd_offset == 0xFFFF_FFFF {
            return Err(VocoderError::WeightLoad(
                "ZIP64 checkpoint archives are not supported".into(),
            ));
        }

        let mut entries = HashMap::new();
        let mut at = cd_offset;
        for _ in 0..count {
            if bytes.get(at..at + 4) != Some(&CD_SIG) {
                return Err(corrupt("bad central directory entry"));
            }
            let method = le16(bytes, at + 10)?;
            let size = le32(bytes, at + 24)?;
            let name_len = le16(bytes, at + 28)?;
            let extra_len = le16(bytes, at + 30)?;
            let comment_len = le16(bytes, at + 32)?;
            let local = le32(bytes, at + 42)?;
            let name = bytes
                .get(at + 46..at + 46 + name_len)
                .ok_or_else(|| corrupt("truncated name"))?;
            let name = String::from_utf8_lossy(name).into_owned();
            if method != 0 {
                return Err(VocoderError::WeightLoad(format!(
                    "checkpoint entry `{name}` is compressed (method {method}); only stored entries are supported"
                )));
            }
            if bytes.get(local..local + 4) != Some(&LOCAL_SIG) {
                return Err(corrupt("bad local header"));
            }
            let data_start = local + 30 + le16(bytes, local + 26)? + le16(bytes, local + 28)?;
            if data_start + size > bytes.len() {
                return Err(corrupt("entry runs past the end of the file"));
            }
            entries.insert(name, (data_start, size));
            at += 46 + name_len + extra_len + comment_len;
        }
        Ok(Self { bytes, entries })
    }

    /// The bytes of the entry named `name`.
    pub fn entry(&self, name: &str) -> Option<&'a [u8]> {
        self.entries
            .get(name)
            .map(|&(start, size)| &self.bytes[start..start + size])
    }

    /// The entry whose name ends with `/suffix` (or equals `suffix`), with the
    /// archive's top-level directory (everything before the first `/`)
    /// returned alongside.
    pub fn find_suffix(&self, suffix: &str) -> Option<(&str, &'a [u8])> {
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

    /// A one-entry stored archive built by hand.
    fn archive(name: &str, data: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&[0x50, 0x4b, 0x03, 0x04]);
        out.extend_from_slice(&[20, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        out.extend_from_slice(&0u32.to_le_bytes()); // crc (unchecked)
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(data);
        let cd_offset = out.len();
        out.extend_from_slice(&[0x50, 0x4b, 0x01, 0x02]);
        out.extend_from_slice(&[20, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(&[0; 12]);
        out.extend_from_slice(&0u32.to_le_bytes()); // local header offset
        out.extend_from_slice(name.as_bytes());
        let cd_len = out.len() - cd_offset;
        out.extend_from_slice(&[0x50, 0x4b, 0x05, 0x06, 0, 0, 0, 0, 1, 0, 1, 0]);
        out.extend_from_slice(&(cd_len as u32).to_le_bytes());
        out.extend_from_slice(&(cd_offset as u32).to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out
    }

    #[test]
    fn reads_a_stored_entry() {
        let bytes = archive("archive/data.pkl", b"hello");
        let zip = StoredZip::parse(&bytes).unwrap();
        assert_eq!(zip.entry("archive/data.pkl"), Some(&b"hello"[..]));
        let (root, data) = zip.find_suffix("data.pkl").unwrap();
        assert_eq!((root, data), ("archive", &b"hello"[..]));
        assert!(zip.entry("archive/missing").is_none());
    }

    #[test]
    fn rejects_garbage() {
        assert!(StoredZip::parse(b"not a zip file at all, not even close").is_err());
        assert!(StoredZip::parse(&[]).is_err());
    }
}
