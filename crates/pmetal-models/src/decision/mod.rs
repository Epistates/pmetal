//! Decision models: a language-model backbone read by a small joint schema
//! head, answering typed questions about a state in one forward pass.
//!
//! Supports the Clef releases (`Cloudflare/clef`, a Qwen3.8-27B backbone, and
//! `Cloudflare/clef-flash`, a Qwen3.5-9B-class backbone). A request names a
//! `state` (any string or JSON value) and a map of `questions`, each a
//! proposition (`noul`), a named choice (`choice`) or an ordered level
//! (`score`). The record is encoded into one prompt ([`encode`]).
//! [`systemone`] validates a `POST /v1/systemone` request body and builds the
//! matching response body.
//!
//! Text only for now: the Qwen3.5 vision encoder is not ported, so a request
//! with `images` or `videos` is refused rather than answered without them.

pub mod encode;
pub mod systemone;

use pmetal_bridge::compat::Exception;

pub use encode::{
    DEFAULT_MAX_LENGTH, EncodeOptions, EncodedQuestion, EncodedRecord, QuestionType, encode_record,
    python_json, render,
};

/// Why a decision could not be made.
#[derive(Debug, thiserror::Error)]
pub enum DecisionError {
    /// The request is malformed; the message is the client's to read.
    #[error("{0}")]
    Request(String),
    /// The request asks for something this port does not do yet.
    #[error("{0}")]
    Unsupported(String),
    /// The release directory could not be loaded.
    #[error("failed to load decision model: {0}")]
    Load(String),
    /// Tokenizing the prompt failed.
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
    /// The forward pass failed.
    #[error("model error: {0}")]
    Model(#[from] Exception),
}
