//! The bytes each token of a tokenizer stands for.
//!
//! A tokenizer's vocabulary lists tokens as strings in the alphabet its model
//! was built with: a byte-level BPE (GPT-2, Llama 3, Qwen) writes each byte
//! as a printable stand-in character; a SentencePiece model writes a space as
//! `▁` and keeps raw bytes as `<0xNN>` tokens. [`ByteVocab`] undoes that,
//! once, so the server can say what bytes a token adds to the text: for
//! OpenAI's `logprobs` (`bytes`), and for the grammar engine that decides
//! which tokens may come next.

use pmetal_data::Tokenizer;
use tokenizers::DecoderWrapper;

/// How a vocabulary spells bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Alphabet {
    /// GPT-2's byte-to-character table.
    ByteLevel,
    /// SentencePiece: `▁` for a space, `<0xNN>` for a raw byte.
    ByteFallback { space: char },
}

/// The byte string of every token id of a tokenizer, and which ids are
/// special (they add no text of their own).
#[derive(Debug, Clone)]
pub struct ByteVocab {
    bytes: Vec<Vec<u8>>,
    special: Vec<bool>,
}

fn is_self_mapped(c: char) -> bool {
    matches!(c, '!'..='~' | '\u{00A1}'..='\u{00AC}' | '\u{00AE}'..='\u{00FF}')
}

/// GPT-2's table from a stand-in character to the byte it stands for.
fn byte_level_chars() -> std::collections::HashMap<char, u8> {
    let mut map = std::collections::HashMap::new();
    let mut next = 0x100u32;
    for byte in 0..=255u8 {
        let c = byte as char;
        if is_self_mapped(c) {
            map.insert(c, byte);
        } else {
            map.insert(char::from_u32(next).expect("a scalar value"), byte);
            next += 1;
        }
    }
    map
}

fn alphabet_of(decoder: Option<&DecoderWrapper>) -> Option<Alphabet> {
    fn scan(decoder: &DecoderWrapper, found: &mut Option<Alphabet>, space: &mut char) {
        match decoder {
            DecoderWrapper::ByteLevel(_) => *found = Some(Alphabet::ByteLevel),
            DecoderWrapper::ByteFallback(_) => {
                *found = Some(Alphabet::ByteFallback { space: '\u{2581}' });
            }
            DecoderWrapper::Replace(replace) if replace.content == " " => {
                // The pattern has no accessor; its serialization names it.
                if let Ok(value) = serde_json::to_value(replace) {
                    if let Some(pattern) = value["pattern"]["String"].as_str() {
                        let mut chars = pattern.chars();
                        if let (Some(c), None) = (chars.next(), chars.next()) {
                            *space = c;
                        }
                    }
                }
            }
            DecoderWrapper::Sequence(sequence) => {
                for inner in sequence.get_decoders() {
                    scan(inner, found, space);
                }
            }
            _ => {}
        }
    }
    let mut found = None;
    let mut space = '\u{2581}';
    scan(decoder?, &mut found, &mut space);
    match found {
        Some(Alphabet::ByteFallback { .. }) => Some(Alphabet::ByteFallback { space }),
        other => other,
    }
}

impl ByteVocab {
    /// The byte table of `tokenizer`, `None` when its decoder is neither
    /// byte-level nor SentencePiece-style (a word-level vocabulary has no
    /// byte spelling to recover).
    pub fn from_tokenizer(tokenizer: &Tokenizer) -> Option<Self> {
        let inner = tokenizer.inner();
        let alphabet = alphabet_of(inner.get_decoder())?;
        let size = inner.get_vocab_size(true);
        let mut bytes = vec![Vec::new(); size];
        let mut special = vec![false; size];
        let chars = byte_level_chars();

        for (id, added) in inner.get_added_tokens_decoder() {
            let id = id as usize;
            if id >= size {
                continue;
            }
            // Every added token of the form <...> is a control token; others
            // are text the tokenizer matches whole.
            if added.special || (added.content.starts_with('<') && added.content.ends_with('>')) {
                special[id] = true;
            } else {
                bytes[id] = added.content.clone().into_bytes();
            }
        }
        for id in 0..size {
            if special[id] {
                continue;
            }
            let Some(name) = inner.id_to_token(id as u32) else {
                special[id] = true;
                continue;
            };
            let spelled = match alphabet {
                Alphabet::ByteLevel => {
                    let spelled: Option<Vec<u8>> =
                        name.chars().map(|c| chars.get(&c).copied()).collect();
                    match spelled {
                        Some(spelled) => spelled,
                        // An added token the loop above already spelled.
                        None if !bytes[id].is_empty() => continue,
                        None => {
                            special[id] = true;
                            continue;
                        }
                    }
                }
                Alphabet::ByteFallback { space } => {
                    if name.len() == 6 && name.starts_with("<0x") && name.ends_with('>') {
                        match u8::from_str_radix(&name[3..5], 16) {
                            Ok(byte) => vec![byte],
                            Err(_) => name.into_bytes(),
                        }
                    } else {
                        name.replace(space, " ").into_bytes()
                    }
                }
            };
            bytes[id] = spelled;
        }
        Some(Self { bytes, special })
    }

    /// Number of tokens.
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// The bytes token `id` adds to the text; empty for a special token or
    /// an id past the vocabulary.
    pub fn bytes(&self, id: u32) -> &[u8] {
        self.bytes.get(id as usize).map_or(&[], Vec::as_slice)
    }

    /// Whether `id` is a control token.
    pub fn is_special(&self, id: u32) -> bool {
        self.special.get(id as usize).copied().unwrap_or(true)
    }

    /// Every token's bytes, by id.
    pub fn all_bytes(&self) -> &[Vec<u8>] {
        &self.bytes
    }

    /// Token `id` as text: lossy, as a `logprobs` entry shows it.
    pub fn text(&self, id: u32) -> String {
        String::from_utf8_lossy(self.bytes(id)).into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_data::tokenizer::testing::byte_level_bpe;

    #[test]
    fn byte_level_vocab_recovers_each_tokens_bytes() {
        let tok = byte_level_bpe(&["h i", "\u{120} t"], &["<|im_start|>"], None);
        let vocab = ByteVocab::from_tokenizer(&tok).unwrap();
        let id = |s: &str| tok.inner().token_to_id(s).unwrap();
        assert_eq!(vocab.bytes(id("hi")), b"hi");
        // "Ġt" is " t".
        assert_eq!(vocab.bytes(id("\u{120}t")), b" t");
        assert_eq!(vocab.bytes(id("a")), b"a");
        // A byte that is no printable character, and half of a character.
        assert_eq!(vocab.bytes(id("\u{10a}")), b"\n");
        assert!(vocab.is_special(id("<|im_start|>")));
        assert!(vocab.bytes(id("<|im_start|>")).is_empty());
        assert!(!vocab.is_special(id("hi")));
        // Every byte has a token of its own.
        let covered: std::collections::HashSet<u8> = (0..vocab.len() as u32)
            .filter(|&t| vocab.bytes(t).len() == 1)
            .map(|t| vocab.bytes(t)[0])
            .collect();
        assert_eq!(covered.len(), 256);
    }

    #[test]
    fn a_word_level_tokenizer_has_no_byte_spelling() {
        let tok = pmetal_data::tokenizer::testing::word_level(&["[UNK]", "a"], &[]);
        assert!(ByteVocab::from_tokenizer(&tok).is_none());
    }
}
