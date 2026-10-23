//! Reading a SentencePiece `tokenizer.model`.
//!
//! The Gemma and Llama 2 families tokenize with SentencePiece, and a GGUF
//! engine tokenizes them from the model's own pieces, scores and types. The
//! `tokenizer.json` Hugging Face derives from the same model keeps the pieces
//! but not the scores, so the protobuf is read for them. Only the two parts of
//! `ModelProto` that matter are decoded (the pieces, and the normalizer's
//! dummy-prefix flag); the rest is skipped by its wire type.

/// The kind a piece has in a GGUF `tokenizer.ggml.token_type`: SentencePiece's
/// own numbering, which the GGUF format adopted.
pub(super) const NORMAL: i32 = 1;

/// One piece of the vocabulary.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Piece {
    pub text: String,
    pub score: f32,
    pub kind: i32,
}

/// What the GGUF file needs of a SentencePiece model.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct SentencePieceModel {
    /// The pieces, by id.
    pub pieces: Vec<Piece>,
    /// Whether the tokenizer prepends a space to the text it encodes
    /// (`normalizer_spec.add_dummy_prefix`, which SentencePiece defaults to
    /// true).
    pub add_dummy_prefix: bool,
}

/// A cursor over protobuf bytes.
struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, at: 0 }
    }

    fn done(&self) -> bool {
        self.at >= self.bytes.len()
    }

    fn take(&mut self, n: usize) -> anyhow::Result<&'a [u8]> {
        let end = self
            .at
            .checked_add(n)
            .filter(|&end| end <= self.bytes.len())
            .ok_or_else(|| anyhow::anyhow!("tokenizer.model is truncated"))?;
        let slice = &self.bytes[self.at..end];
        self.at = end;
        Ok(slice)
    }

    fn varint(&mut self) -> anyhow::Result<u64> {
        let mut value = 0u64;
        for shift in (0..64).step_by(7) {
            let byte = self.take(1)?[0];
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        anyhow::bail!("tokenizer.model has an over-long varint")
    }

    /// The next field: its number and its payload.
    fn field(&mut self) -> anyhow::Result<(u64, Payload<'a>)> {
        let tag = self.varint()?;
        let payload = match tag & 7 {
            0 => Payload::Varint(self.varint()?),
            1 => Payload::Fixed(self.take(8)?),
            2 => {
                let len = self.varint()? as usize;
                Payload::Bytes(self.take(len)?)
            }
            5 => Payload::Fixed(self.take(4)?),
            wire => anyhow::bail!("tokenizer.model has a field of unknown wire type {wire}"),
        };
        Ok((tag >> 3, payload))
    }
}

enum Payload<'a> {
    Varint(u64),
    Fixed(&'a [u8]),
    Bytes(&'a [u8]),
}

fn parse_piece(bytes: &[u8]) -> anyhow::Result<Piece> {
    let mut piece = Piece {
        text: String::new(),
        score: 0.0,
        kind: NORMAL,
    };
    let mut reader = Reader::new(bytes);
    while !reader.done() {
        match reader.field()? {
            (1, Payload::Bytes(text)) => piece.text = String::from_utf8(text.to_vec())?,
            (2, Payload::Fixed(raw)) if raw.len() == 4 => {
                piece.score = f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
            }
            (3, Payload::Varint(kind)) => piece.kind = kind as i32,
            _ => {}
        }
    }
    Ok(piece)
}

impl SentencePieceModel {
    /// Decode a `tokenizer.model`.
    ///
    /// # Errors
    ///
    /// The bytes are not a protobuf `ModelProto`, or a piece is not UTF-8.
    pub(super) fn parse(bytes: &[u8]) -> anyhow::Result<Self> {
        let mut model = Self {
            pieces: Vec::new(),
            add_dummy_prefix: true,
        };
        let mut reader = Reader::new(bytes);
        while !reader.done() {
            match reader.field()? {
                (1, Payload::Bytes(piece)) => model.pieces.push(parse_piece(piece)?),
                (3, Payload::Bytes(normalizer)) => {
                    let mut inner = Reader::new(normalizer);
                    while !inner.done() {
                        if let (3, Payload::Varint(flag)) = inner.field()? {
                            model.add_dummy_prefix = flag != 0;
                        }
                    }
                }
                _ => {}
            }
        }
        if model.pieces.is_empty() {
            anyhow::bail!("tokenizer.model holds no pieces");
        }
        Ok(model)
    }
}

/// Writing a `tokenizer.model`, for the tests that need one.
#[cfg(test)]
pub(super) mod testing {
    pub(in crate::commands::gguf_export) fn varint(mut value: u64) -> Vec<u8> {
        let mut out = Vec::new();
        loop {
            let byte = (value & 0x7f) as u8;
            value >>= 7;
            if value == 0 {
                out.push(byte);
                return out;
            }
            out.push(byte | 0x80);
        }
    }

    pub(in crate::commands::gguf_export) fn bytes_field(number: u64, payload: &[u8]) -> Vec<u8> {
        let mut out = varint(number << 3 | 2);
        out.extend(varint(payload.len() as u64));
        out.extend(payload);
        out
    }

    pub(in crate::commands::gguf_export) fn piece(
        text: &str,
        score: f32,
        kind: Option<u64>,
    ) -> Vec<u8> {
        let mut body = bytes_field(1, text.as_bytes());
        body.extend(varint(2 << 3 | 5));
        body.extend(score.to_le_bytes());
        if let Some(kind) = kind {
            body.extend(varint(3 << 3));
            body.extend(varint(kind));
        }
        bytes_field(1, &body)
    }

    /// A model of `pieces` (text, score, kind) whose normalizer has the dummy
    /// prefix `add_dummy_prefix`.
    pub(in crate::commands::gguf_export) fn encode_model(
        pieces: &[(&str, f32, Option<u64>)],
        add_dummy_prefix: bool,
    ) -> Vec<u8> {
        let mut model = Vec::new();
        for &(text, score, kind) in pieces {
            model.extend(piece(text, score, kind));
        }
        model.extend(bytes_field(3, &[0x18, u8::from(add_dummy_prefix)]));
        model
    }
}

#[cfg(test)]
mod tests {
    use super::testing::*;
    use super::*;

    #[test]
    fn pieces_scores_and_types_are_read_in_order() {
        let mut model = piece("<unk>", 0.0, Some(2));
        model.extend(piece("▁the", -3.5, None));
        model.extend(piece("<0x0A>", 0.0, Some(6)));
        // A trainer spec and a normalizer with the dummy prefix off, in the
        // order a real file has them.
        model.extend(bytes_field(2, &[0x08, 0x01]));
        model.extend(bytes_field(3, &[0x18, 0x00]));

        let parsed = SentencePieceModel::parse(&model).unwrap();
        assert!(!parsed.add_dummy_prefix);
        assert_eq!(
            parsed.pieces,
            vec![
                Piece {
                    text: "<unk>".into(),
                    score: 0.0,
                    kind: 2
                },
                Piece {
                    text: "▁the".into(),
                    score: -3.5,
                    kind: NORMAL
                },
                Piece {
                    text: "<0x0A>".into(),
                    score: 0.0,
                    kind: 6
                },
            ]
        );
    }

    #[test]
    fn the_dummy_prefix_defaults_on() {
        let model = piece("a", 0.0, None);
        assert!(SentencePieceModel::parse(&model).unwrap().add_dummy_prefix);
    }

    #[test]
    fn a_truncated_or_empty_model_is_an_error() {
        let model = piece("abc", 0.0, None);
        assert!(SentencePieceModel::parse(&model[..model.len() - 2]).is_err());
        assert!(SentencePieceModel::parse(&[]).is_err());
    }
}
