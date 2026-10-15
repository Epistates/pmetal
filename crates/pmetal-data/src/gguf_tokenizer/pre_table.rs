//! Pre-tokenizer patterns of GGUF byte-pair tokenizers, by `tokenizer.ggml.pre`.
//!
//! Each name selects the ordered list of patterns the text is split by before
//! byte-pair merging, whether the pieces are byte-encoded, and the defaults that
//! go with the name. The lists follow the format's reference tokenizer, except that
//! the o200k-style patterns (gpt-4o, Llama 4, Tekken, MiniMax) use the cased-letter
//! classes the model makers' own tokenizers use, where the reference approximates
//! them with lookaheads that mishandle combining marks.

/// How text is split before merging, for one `tokenizer.ggml.pre` value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PreSpec {
    /// Patterns applied in order, each splitting what the last left; matches
    /// and the text between them are pieces.
    pub patterns: &'static [&'static str],
    /// Whether pieces are mapped to the byte-level alphabet (all but the
    /// SentencePiece-style BPEs).
    pub byte_level: bool,
    /// Whether a piece that is itself a token is emitted whole, without merging.
    pub ignore_merges: bool,
    /// Whether the tokenizer adds a BOS unless the metadata says otherwise.
    pub add_bos: bool,
}

const PATTERNS_DEFAULT: &[&str] = &[
    r#"[\p{P}\$\+<=>\^~\|]+"#,
    r#"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"#,
    r#"\p{N}+"#,
    r#"[0-9][0-9][0-9]"#,
];

const PATTERNS_MINICPM5: &[&str] = &[
    r#"\p{N}{1,3}"#,
    r#"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}+| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#,
];

const PATTERNS_LLAMA3: &[&str] = &[
    r#"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#,
];

const PATTERNS_DEEPSEEK_LLM: &[&str] = &[
    r#"[\r\n]"#,
    r#"\s?[A-Za-zµÀ-ÖØ-öø-ƺƼ-ƿǄ-ʓʕ-ʯͰ-ͳͶͷͻ-ͽͿΆΈ-ΊΌΎ-ΡΣ-ϵϷ-ҁҊ-ԯԱ-ՖႠ-ჅᎠ-Ᏽᏸ-ᏽᲐ-ᲺᲽ-Ჿᴀ-ᴫᵫ-ᵷᵹ-ᶚḀ-ἕἘ-Ἕἠ-ὅὈ-Ὅὐ-ὗὙὛὝὟ-ώᾀ-ᾴᾶ-ᾼιῂ-ῄῆ-ῌῐ-ΐῖ-Ίῠ-Ῥῲ-ῴῶ-ῼℂℇℊ-ℓℕℙ-ℝℤΩℨK-ℭℯ-ℴℹℼ-ℿⅅ-ⅉⅎↃↄⰀ-ⱻⱾ-ⳤⳫ-ⳮⳲⳳꙀ-ꙭꚀ-ꚛꜢ-ꝯꝱ-ꞇꞋ-ꞎꭰ-ꮿﬀ-ﬆﬓ-ﬗＡ-Ｚａ-ｚ𐐀-𐑏𐒰-𐓓𐓘-𐓻𐲀-𐲲𐳀-𐳲𑢠-𑣟𞤀-𞥃]+"#,
    r#"\s?[!-/:-~！-／：-～‘-‟　-。]+"#,
    r#"\s+$"#,
    r#"[一-龥ࠀ-一가-퟿]+"#,
    r#"\p{N}+"#,
];

const PATTERNS_DEEPSEEK_CODER: &[&str] = &[
    r#"[\r\n]"#,
    r#"\s?\p{L}+"#,
    r#"\s?\p{P}+"#,
    r#"[一-龥ࠀ-一가-퟿]+"#,
    r#"\p{N}"#,
];

const PATTERNS_DEEPSEEK3_LLM: &[&str] = &[
    r#"\p{N}{1,3}"#,
    r#"[一-龥぀-ゟ゠-ヿ]+"#,
    r##"[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"##,
];

const PATTERNS_SPARK2_5: &[&str] = &[
    r#"\p{N}{1,3}"#,
    r#"[一-龥぀-ゟ゠-ヿ]+"#,
    r##"[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+|[\r\n]|\s+(?!\S)|\s+"##,
    r#"\p{N}"#,
];

const PATTERNS_YOUTU: &[&str] = &[
    r#"[가-힣ㄱ-ㆎ]+|[！…“”‘’—：；，、-〿︰-﹏]+|[ㄅ-ㄯ]+|[一-龥぀-ゟ゠-ヿ]+"#,
    r#"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"#,
];

const PATTERNS_FALCON: &[&str] = &[
    r#"[\p{P}\$\+<=>\^~\|`]+"#,
    r#"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"#,
    r#"[0-9][0-9][0-9]"#,
];

const PATTERNS_MPT: &[&str] =
    &[r#"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"#];

const PATTERNS_STARCODER: &[&str] = &[
    r#"\p{N}"#,
    r#"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"#,
];

const PATTERNS_GPT2: &[&str] =
    &[r#"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"#];

const PATTERNS_JAIS2: &[&str] = &[
    r#"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s{512}(?!\S)|\s{256}(?!\S)|\s{128}(?!\S)|\s{64}(?!\S)|\s{32}(?!\S)|\s{16}(?!\S)|\s{8}(?!\S)|\s{4}(?!\S)|\s{1,2}(?!\S)|\s{1}"#,
];

const PATTERNS_GEMMA4: &[&str] = &[r#"[^\n]+|[\n]+"#];

const PATTERNS_SARVAM_MOE: &[&str] = &[r#"[^\n]+|[\n]+"#];

const PATTERNS_REFACT: &[&str] = &[
    r#"\p{N}"#,
    r#"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"#,
];

const PATTERNS_COMMAND_R: &[&str] = &[
    r#"\p{N}"#,
    r#"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"#,
];

const PATTERNS_QWEN2: &[&str] = &[
    r#"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#,
];

const PATTERNS_QWEN35: &[&str] = &[
    r#"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#,
];

const PATTERNS_STABLELM2: &[&str] = &[
    r#"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#,
];

const PATTERNS_OLMO: &[&str] =
    &[r#"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"#];

const PATTERNS_DBRX: &[&str] = &[
    r#"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#,
];

const PATTERNS_SMAUG: &[&str] = &[
    r#"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#,
];

const PATTERNS_PORO: &[&str] = &[r#" ?[^(\s|.,!?…。，、।۔،)]+"#];

const PATTERNS_CHATGLM4: &[&str] = &[
    r#"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#,
];

const PATTERNS_VIKING: &[&str] = &[r#" ?[^(\s|.,!?…。，、।۔،)]+"#, r#"\p{N}"#];

const PATTERNS_JAIS: &[&str] =
    &[r#"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"#];

const PATTERNS_TEKKEN: &[&str] = &[
    r#"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"#,
];

const PATTERNS_SMOLLM: &[&str] = &[
    r#"\p{N}"#,
    r#"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"#,
];

const PATTERNS_CODESHELL: &[&str] = &[
    r#"\p{N}"#,
    r#"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"#,
];

const PATTERNS_BLOOM: &[&str] = &[r#" ?[^(\s|.,!?…。，、।۔،)]+"#];

const PATTERNS_GPT3_FINNISH: &[&str] = &[r#" ?[^(\s|.,!?…。，、।۔،)]+"#];

const PATTERNS_EXAONE: &[&str] = &[
    r#"\p{N}"#,
    r#"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"#,
];

const PATTERNS_EXAONE_MOE: &[&str] = &[
    r#"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?(?:\p{L}\p{M}*(?: \p{L}\p{M}*)*)+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]?|\s*[\r\n]|\s+(?!\S)|\s+"#,
];

const PATTERNS_CHAMELEON: &[&str] = &[
    r#"<sentinel:[0-9]+>"#,
    r#"(IMGIMG)((A|B|C|D|E|F|G|H|I){1,4})Z"#,
    r#"([\t\n]|    |  )"#,
    r#"\p{N}"#,
    r#"[\p{P}!-/:-@\[-`{-~]"#,
    r#"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"#,
];

const PATTERNS_MINERVA: &[&str] = &[
    r#"\p{N}"#,
    r#"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"#,
];

const PATTERNS_GPT4O: &[&str] = &[
    r#"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"#,
];

const PATTERNS_GRANITE_EMB_MULTI: &[&str] = &[
    r#"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"#,
];

const PATTERNS_TINY_AYA: &[&str] = &[
    r#"\d{1,3}(?=(?:\d{3})*\b)"#,
    r#"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"#,
];

const PATTERNS_SUPERBPE: &[&str] = &[r#"\p{N}+"#, r#"(?=(\d{3})+(?!\d))"#];

const PATTERNS_TRILLION: &[&str] =
    &[r#"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"#];

const PATTERNS_GRANITE_DOCLING: &[&str] =
    &[r#"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"#];

const PATTERNS_BAILINGMOE: &[&str] = &[
    r#"'(?:[sSdDmMtT]|[lL][lL]|[vV][eE]|[rR][eE])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"#,
];

const PATTERNS_SEED_CODER: &[&str] = &[
    r#"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1}| ?[^\s\p{L}\p{N}\r\n]+|\s*[\r\n]+|\s+(?!\S)|\s+"#,
];

const PATTERNS_HUNYUAN: &[&str] = &[
    r#"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#,
];

const PATTERNS_HUNYUAN_DENSE: &[&str] = &[
    r#"\p{N}{1,3}"#,
    r#"[一-龥぀-ゟ゠-ヿ]+"#,
    r##"[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"##,
];

const PATTERNS_HY_V4: &[&str] = &[
    r#"\p{N}{1,3}"#,
    r#"[一-龥぀-ゟ゠-ヿ]+"#,
    r##"[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"##,
];

const PATTERNS_JOYAI_LLM: &[&str] = &[
    r#"\p{N}{1,3}"#,
    r#"[一-龥぀-ゟ゠-ヿ]+"#,
    r##"[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"##,
];

const PATTERNS_UFAKZEKA: &[&str] =
    &[r#"[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#];

const PATTERNS_GROK_2: &[&str] = &[
    r#"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#,
];

const PATTERNS_LAGUNA: &[&str] = &[
    r#"[^\n]+|[\n]+"#,
    r#"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#,
];

const PATTERNS_MINIMAX_M2: &[&str] = &[
    r#"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"#,
];

const PATTERNS_SOLAR_OPEN: &[&str] = &[
    r#"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#,
];

const PATTERNS_MELLUM2: &[&str] = &[
    r#"\p{N}"#,
    r#"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"#,
];

const PATTERNS_K2_HORIZON: &[&str] = &[
    r#"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?(?:\p{L}|\p{M}|\u200C|\u200D)+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#,
];

/// The spec for a `tokenizer.ggml.pre` value, or `None` for a name this table
/// does not know. A missing name is the `default` spec.
pub(super) fn lookup(name: &str) -> Option<PreSpec> {
    let spec = |patterns, byte_level, ignore_merges, add_bos| {
        Some(PreSpec {
            patterns,
            byte_level,
            ignore_merges,
            add_bos,
        })
    };
    match name {
        "" | "default" => spec(PATTERNS_DEFAULT, true, false, false),
        "minicpm5" => spec(PATTERNS_MINICPM5, true, true, false),
        "llama3" | "llama-v3" | "llama-bpe" | "falcon3" | "falcon-h1" | "pixtral" | "midm-2.0"
        | "lfm2" | "jina-v5-nano" => spec(PATTERNS_LLAMA3, true, true, true),
        "deepseek-llm" => spec(PATTERNS_DEEPSEEK_LLM, true, false, false),
        "deepseek-coder" => spec(PATTERNS_DEEPSEEK_CODER, true, false, false),
        "deepseek-v3" => spec(PATTERNS_DEEPSEEK3_LLM, true, false, false),
        "spark2_5" => spec(PATTERNS_SPARK2_5, true, false, false),
        "youtu" => spec(PATTERNS_YOUTU, true, true, false),
        "falcon" => spec(PATTERNS_FALCON, true, false, false),
        "mpt" => spec(PATTERNS_MPT, true, false, false),
        "starcoder" => spec(PATTERNS_STARCODER, true, false, false),
        "gpt-2" | "phi-2" | "jina-es" | "jina-de" | "gigachat" | "jina-v2-es" | "jina-v2-de"
        | "a.x-4.0" | "mellum" | "modern-bert" => spec(PATTERNS_GPT2, true, false, false),
        "jais-2" => spec(PATTERNS_JAIS2, true, false, false),
        "gemma4" | "granite-embed-multi-311m" => spec(PATTERNS_GEMMA4, false, false, false),
        "sarvam-moe" => spec(PATTERNS_SARVAM_MOE, false, false, false),
        "jina-v1-en" | "jina-v2-code" | "roberta-bpe" => spec(PATTERNS_GPT2, true, false, false),
        "refact" => spec(PATTERNS_REFACT, true, false, false),
        "command-r" => spec(PATTERNS_COMMAND_R, true, false, false),
        "qwen2" | "deepseek-r1-qwen" | "kormo" | "f2llmv2" => {
            spec(PATTERNS_QWEN2, true, false, false)
        }
        "qwen35" => spec(PATTERNS_QWEN35, true, false, false),
        "stablelm2" => spec(PATTERNS_STABLELM2, true, false, false),
        "olmo" => spec(PATTERNS_OLMO, true, false, false),
        "dbrx" => spec(PATTERNS_DBRX, true, false, false),
        "smaug-bpe" => spec(PATTERNS_SMAUG, true, false, false),
        "poro-chat" => spec(PATTERNS_PORO, true, false, false),
        "chatglm-bpe" => spec(PATTERNS_CHATGLM4, true, false, false),
        "glm4" | "glm5" => spec(PATTERNS_CHATGLM4, true, true, false),
        "viking" => spec(PATTERNS_VIKING, true, false, false),
        "jais" => spec(PATTERNS_JAIS, true, false, false),
        "tekken" => spec(PATTERNS_TEKKEN, true, true, true),
        "smollm" => spec(PATTERNS_SMOLLM, true, false, false),
        "codeshell" => spec(PATTERNS_CODESHELL, true, false, false),
        "bloom" => spec(PATTERNS_BLOOM, true, false, false),
        "gpt3-finnish" => spec(PATTERNS_GPT3_FINNISH, true, false, false),
        "exaone" => spec(PATTERNS_EXAONE, true, false, false),
        "exaone4" => spec(PATTERNS_GPT2, true, false, false),
        "exaone-moe" => spec(PATTERNS_EXAONE_MOE, true, false, false),
        "chameleon" => spec(PATTERNS_CHAMELEON, true, false, true),
        "minerva-7b" => spec(PATTERNS_MINERVA, true, false, false),
        "megrez" => spec(PATTERNS_QWEN2, true, false, false),
        "gpt-4o" | "llama4" | "kanana2" | "talkie" => spec(PATTERNS_GPT4O, true, false, false),
        "granite-embed-multi-97m" => spec(PATTERNS_GRANITE_EMB_MULTI, true, true, false),
        "tiny_aya" | "cohere2moe" => spec(PATTERNS_TINY_AYA, true, false, false),
        "superbpe" => spec(PATTERNS_SUPERBPE, true, false, false),
        "trillion" => spec(PATTERNS_TRILLION, true, false, false),
        "granite-docling" => spec(PATTERNS_GRANITE_DOCLING, true, false, false),
        "bailingmoe" | "bailingmoe2" | "llada-moe" => spec(PATTERNS_BAILINGMOE, true, false, false),
        "seed-coder" => spec(PATTERNS_SEED_CODER, true, false, false),
        "hunyuan" => spec(PATTERNS_HUNYUAN, true, false, false),
        "hunyuan-dense" => spec(PATTERNS_HUNYUAN_DENSE, true, false, false),
        "hy_v4" => spec(PATTERNS_HY_V4, true, false, false),
        "joyai-llm" => spec(PATTERNS_JOYAI_LLM, true, false, false),
        "ufakzeka" => spec(PATTERNS_UFAKZEKA, true, false, false),
        "grok-2" => spec(PATTERNS_GROK_2, true, false, false),
        "laguna" => spec(PATTERNS_LAGUNA, true, false, false),
        "minimax-m2" => spec(PATTERNS_MINIMAX_M2, true, false, false),
        "solar-open" => spec(PATTERNS_SOLAR_OPEN, true, false, false),
        "mellum2" => spec(PATTERNS_MELLUM2, true, false, false),
        "k2-horizon" => spec(PATTERNS_K2_HORIZON, true, false, false),
        _ => None,
    }
}

/// Every name [`lookup`] knows.
#[cfg(test)]
pub(super) const NAMES: &[&str] = &[
    "default",
    "minicpm5",
    "llama3",
    "llama-v3",
    "llama-bpe",
    "falcon3",
    "falcon-h1",
    "pixtral",
    "midm-2.0",
    "lfm2",
    "jina-v5-nano",
    "deepseek-llm",
    "deepseek-coder",
    "deepseek-v3",
    "spark2_5",
    "youtu",
    "falcon",
    "mpt",
    "starcoder",
    "gpt-2",
    "phi-2",
    "jina-es",
    "jina-de",
    "gigachat",
    "jina-v2-es",
    "jina-v2-de",
    "a.x-4.0",
    "mellum",
    "modern-bert",
    "jais-2",
    "gemma4",
    "granite-embed-multi-311m",
    "sarvam-moe",
    "jina-v1-en",
    "jina-v2-code",
    "roberta-bpe",
    "refact",
    "command-r",
    "qwen2",
    "deepseek-r1-qwen",
    "kormo",
    "f2llmv2",
    "qwen35",
    "stablelm2",
    "olmo",
    "dbrx",
    "smaug-bpe",
    "poro-chat",
    "chatglm-bpe",
    "glm4",
    "glm5",
    "viking",
    "jais",
    "tekken",
    "smollm",
    "codeshell",
    "bloom",
    "gpt3-finnish",
    "exaone",
    "exaone4",
    "exaone-moe",
    "chameleon",
    "minerva-7b",
    "megrez",
    "gpt-4o",
    "llama4",
    "kanana2",
    "talkie",
    "granite-embed-multi-97m",
    "tiny_aya",
    "cohere2moe",
    "superbpe",
    "trillion",
    "granite-docling",
    "bailingmoe",
    "bailingmoe2",
    "llada-moe",
    "seed-coder",
    "hunyuan",
    "hunyuan-dense",
    "hy_v4",
    "joyai-llm",
    "ufakzeka",
    "grok-2",
    "laguna",
    "minimax-m2",
    "solar-open",
    "mellum2",
    "k2-horizon",
];
