//! Embedding training data: pairs and triplets.

use pmetal_core::Result;

// ---------------------------------------------------------------------------
// Embedding training data types
// ---------------------------------------------------------------------------

/// A pair of texts for embedding training, with an optional similarity label.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EmbeddingPair {
    /// First text (anchor / query).
    pub text_a: String,
    /// Second text (positive or comparison).
    pub text_b: String,
    /// Similarity label in `[0, 1]`.  `1.0` = semantically identical.
    /// `None` means this is an implicit positive pair (treated as `1.0`).
    pub label: Option<f32>,
}

/// A triplet of texts for contrastive embedding training.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EmbeddingTriplet {
    /// Anchor text.
    pub anchor: String,
    /// Positive text (semantically similar to anchor).
    pub positive: String,
    /// Negative text (semantically different from anchor).
    pub negative: String,
}

/// Embedding training dataset — pairs or triplets.
///
/// Created via [`EmbeddingDataset::from_jsonl`], which auto-detects the format
/// by inspecting the first record.
///
/// **Pair format** (detected by `text_a`/`text_b`, `sentence1`/`sentence2`,
/// or `query`/`positive` keys):
/// ```json
/// {"text_a": "...", "text_b": "...", "label": 0.9}
/// {"sentence1": "...", "sentence2": "..."}
/// {"query": "...", "positive": "..."}
/// ```
///
/// **Triplet format** (detected by `anchor` key):
/// ```json
/// {"anchor": "...", "positive": "...", "negative": "..."}
/// ```
#[derive(Debug, Clone)]
pub enum EmbeddingDataset {
    /// Paired texts with optional similarity labels.
    Pairs(Vec<EmbeddingPair>),
    /// Triplets (anchor, positive, negative).
    Triplets(Vec<EmbeddingTriplet>),
}

impl EmbeddingDataset {
    /// Load from a JSONL file. Auto-detects format from the first non-empty record.
    ///
    /// Supported pair keys: `text_a`/`text_b`, `sentence1`/`sentence2`,
    /// `query`/`positive`. Score/label key: `label` or `score`.
    pub fn from_jsonl(path: impl AsRef<std::path::Path>) -> Result<Self> {
        use std::io::BufRead;

        let file = std::fs::File::open(path.as_ref()).map_err(pmetal_core::PMetalError::Io)?;
        let reader = std::io::BufReader::new(file);

        let mut lines: Vec<String> = Vec::new();
        for line in reader.lines() {
            let line = line.map_err(pmetal_core::PMetalError::Io)?;
            if !line.trim().is_empty() {
                lines.push(line);
            }
        }

        if lines.is_empty() {
            return Ok(Self::Pairs(Vec::new()));
        }

        // Detect format from first record
        let first: serde_json::Value = serde_json::from_str(&lines[0]).map_err(|e| {
            pmetal_core::PMetalError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e))
        })?;

        if first.get("anchor").is_some() {
            // Triplet format
            let mut triplets = Vec::with_capacity(lines.len());
            for (i, line) in lines.iter().enumerate() {
                let t: EmbeddingTriplet = serde_json::from_str(line).map_err(|e| {
                    pmetal_core::PMetalError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("line {}: {}", i + 1, e),
                    ))
                })?;
                triplets.push(t);
            }
            Ok(Self::Triplets(triplets))
        } else {
            // Pair format — flexible key mapping
            let mut pairs = Vec::with_capacity(lines.len());
            for (i, line) in lines.iter().enumerate() {
                let v: serde_json::Value = serde_json::from_str(line).map_err(|e| {
                    pmetal_core::PMetalError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("line {}: {}", i + 1, e),
                    ))
                })?;

                let text_a = v
                    .get("text_a")
                    .or_else(|| v.get("sentence1"))
                    .or_else(|| v.get("query"))
                    .or_else(|| v.get("premise"))
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| pmetal_core::PMetalError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!(
                            "Line {}: no recognized text_a key (expected text_a, sentence1, query, or premise)",
                            i + 1,
                        ),
                    )))?
                    .to_string();
                let text_b = v
                    .get("text_b")
                    .or_else(|| v.get("sentence2"))
                    .or_else(|| v.get("positive"))
                    .or_else(|| v.get("hypothesis"))
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| pmetal_core::PMetalError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!(
                            "Line {}: no recognized text_b key (expected text_b, sentence2, positive, or hypothesis)",
                            i + 1,
                        ),
                    )))?
                    .to_string();
                let label = v
                    .get("label")
                    .or_else(|| v.get("score"))
                    .and_then(|v| v.as_f64())
                    .map(|l| l as f32);

                pairs.push(EmbeddingPair {
                    text_a,
                    text_b,
                    label,
                });
            }
            Ok(Self::Pairs(pairs))
        }
    }

    /// Number of examples in this dataset.
    pub fn len(&self) -> usize {
        match self {
            Self::Pairs(p) => p.len(),
            Self::Triplets(t) => t.len(),
        }
    }

    /// Returns `true` if the dataset has no examples.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Shuffle in-place using a deterministic seed.
    pub fn shuffle(&mut self, seed: u64) {
        use rand::SeedableRng;
        use rand::seq::SliceRandom;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        match self {
            Self::Pairs(p) => p.shuffle(&mut rng),
            Self::Triplets(t) => t.shuffle(&mut rng),
        }
    }
}
