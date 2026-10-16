//! Pooling strategies for converting sequence representations to fixed-size embeddings.

use pmetal_runtime::compat::{Array, Dtype, Exception, ops};
/// Pooling modes for sentence embeddings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum PoolingMode {
    /// Mean of all non-padding tokens.
    #[default]
    Mean,
    /// First token (`[CLS]` for BERT, `<s>` for RoBERTa).
    Cls,
    /// Element-wise max across sequence dimension.
    Max,
    /// Last non-padding token (useful for causal models used as embedders).
    LastToken,
    /// Weighted mean with linearly increasing position weights.
    WeightedMean,
}

impl std::fmt::Display for PoolingMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Mean => write!(f, "mean"),
            Self::Cls => write!(f, "cls"),
            Self::Max => write!(f, "max"),
            Self::LastToken => write!(f, "last_token"),
            Self::WeightedMean => write!(f, "weighted_mean"),
        }
    }
}

/// How a sentence-transformers model turns encoder states into its embedding,
/// as `modules.json` and the pooling module's `config.json` describe it.
///
/// A sentence-transformers directory is a pipeline of modules: the encoder,
/// then a [`Pooling`](PoolingMode) module and usually a `Normalize`. The encoder
/// alone gives per-token states; which pooling the model was trained with is
/// written down only here, and pooling a model the wrong way (CLS for a
/// mean-pooled model) gives embeddings that are plausible and wrong.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SentenceTransformerConfig {
    /// Where the encoder's `config.json` and weights are, relative to the
    /// model directory (`""` for the directory itself, or `0_Transformer`).
    pub transformer_path: String,
    /// The pooling the model was trained with.
    pub pooling: PoolingMode,
    /// Whether the pooled vector is L2-normalized (a `Normalize` module).
    pub normalize: bool,
    /// The longest input the model was trained for, from
    /// `sentence_bert_config.json`.
    pub max_seq_length: Option<usize>,
}

impl SentenceTransformerConfig {
    /// Read `modules.json` (and what it points to) from `dir`; `None` for a
    /// directory that is not a sentence-transformers model.
    ///
    /// A pipeline this crate cannot reproduce is an error naming the module,
    /// not a partial reading: a `Dense` projection after the pooling, several
    /// poolings concatenated, or `mean_sqrt_len_tokens` pooling would each
    /// give embeddings that differ from the published model's.
    pub fn from_dir(dir: &std::path::Path) -> Result<Option<Self>, Exception> {
        let read = |path: std::path::PathBuf| -> Result<Option<serde_json::Value>, Exception> {
            match std::fs::read_to_string(&path) {
                Ok(text) => serde_json::from_str(&text)
                    .map(Some)
                    .map_err(|e| Exception::custom(format!("{}: {e}", path.display()))),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
                Err(e) => Err(Exception::custom(format!("{}: {e}", path.display()))),
            }
        };
        let Some(modules) = read(dir.join("modules.json"))? else {
            return Ok(None);
        };
        let modules = modules
            .as_array()
            .ok_or_else(|| Exception::custom("modules.json is not a list"))?;

        let mut transformer_path = None;
        let mut pooling = None;
        let mut normalize = false;
        for module in modules {
            let kind = module["type"].as_str().unwrap_or_default();
            let path = module["path"].as_str().unwrap_or_default();
            match kind.rsplit('.').next().unwrap_or_default() {
                "Transformer" => transformer_path = Some(path.to_string()),
                "Pooling" => {
                    let config = read(dir.join(path).join("config.json"))?.ok_or_else(|| {
                        Exception::custom(format!(
                            "modules.json names {path}/config.json, which is missing"
                        ))
                    })?;
                    pooling = Some(Self::pooling_mode(&config)?);
                }
                "Normalize" => normalize = true,
                other => {
                    return Err(Exception::custom(format!(
                        "sentence-transformers module {other:?} ({kind}) is not implemented; \
                         embeddings from this pipeline would differ from the published model's"
                    )));
                }
            }
        }
        let max_seq_length = read(dir.join("sentence_bert_config.json"))?
            .and_then(|config| config["max_seq_length"].as_u64())
            .map(|n| n as usize);
        Ok(Some(Self {
            transformer_path: transformer_path
                .ok_or_else(|| Exception::custom("modules.json has no Transformer module"))?,
            pooling: pooling
                .ok_or_else(|| Exception::custom("modules.json has no Pooling module"))?,
            normalize,
            max_seq_length,
        }))
    }

    /// The one pooling a pooling module's flags select.
    fn pooling_mode(config: &serde_json::Value) -> Result<PoolingMode, Exception> {
        let flag = |key: &str| config[key].as_bool().unwrap_or(false);
        if flag("pooling_mode_mean_sqrt_len_tokens") {
            return Err(Exception::custom(
                "sentence-transformers pooling `mean_sqrt_len_tokens` is not implemented",
            ));
        }
        let selected: Vec<PoolingMode> = [
            ("pooling_mode_cls_token", PoolingMode::Cls),
            ("pooling_mode_mean_tokens", PoolingMode::Mean),
            ("pooling_mode_max_tokens", PoolingMode::Max),
            ("pooling_mode_lasttoken", PoolingMode::LastToken),
            (
                "pooling_mode_weightedmean_tokens",
                PoolingMode::WeightedMean,
            ),
        ]
        .into_iter()
        .filter(|(key, _)| flag(key))
        .map(|(_, mode)| mode)
        .collect();
        match selected.as_slice() {
            [mode] => Ok(*mode),
            [] => Err(Exception::custom(
                "the pooling module selects no pooling mode",
            )),
            many => Err(Exception::custom(format!(
                "the pooling module concatenates {} poolings ({many:?}); only one is implemented",
                many.len()
            ))),
        }
    }
}

/// Pool sequence hidden states into a single embedding vector.
///
/// The arithmetic runs in f32 whatever the model's dtype and the result is cast
/// back to it: summing a long sequence, or weighting positions up to the
/// sequence length, loses real precision in bf16 (257 is not representable),
/// and an f32 scalar mixed into bf16 operands silently promotes the whole
/// result to f32.
///
/// # Arguments
/// * `hidden_states` - `[batch, seq_len, hidden_dim]`
/// * `attention_mask` - `[batch, seq_len]` (1 for real tokens, 0 for padding)
/// * `mode` - Pooling strategy
///
/// # Returns
/// Embedding tensor `[batch, hidden_dim]`, in the dtype of `hidden_states`.
pub fn pool(
    hidden_states: &Array,
    attention_mask: &Array,
    mode: PoolingMode,
) -> Result<Array, Exception> {
    let batch = hidden_states.dim(0);
    let seq_len = hidden_states.dim(1);
    let out_dtype = hidden_states.dtype().as_i32();
    let f32_dtype = Dtype::Float32.as_i32();
    let hidden = hidden_states.as_dtype(f32_dtype);
    // `[batch, seq_len, 1]`, 1.0 where a real token sits.
    let mask = attention_mask
        .reshape(&[batch, seq_len, 1])
        .as_dtype(f32_dtype);
    let tiny = Array::from_f32(1e-9);

    let pooled = match mode {
        PoolingMode::Mean => {
            let sum = hidden.multiply(&mask).sum_axes(&[1], false);
            let count = ops::maximum(&mask.sum_axes(&[1], false), &tiny);
            sum.divide(&count)
        }
        PoolingMode::Cls => ops::slice_axis(&hidden, 1, 0, 1).squeeze_axes(&[1]),
        PoolingMode::Max => {
            // Padding positions are pushed far below any real state.
            let padding = Array::from_f32(1.0).subtract(&mask);
            hidden
                .add(&padding.multiply(&Array::from_f32(-1e9)))
                .max_axis(1, false)
        }
        PoolingMode::LastToken => {
            // The last *attended* position, wherever the padding is: right
            // padding ends the sequence early, left padding (what last-token
            // embedders use) starts it late, and `count - 1` is right only for
            // the first. A row with nothing attended reads position 0.
            let positions = ops::arange(seq_len, Dtype::Int32).reshape(&[1, seq_len]);
            let attended = attention_mask
                .as_dtype(Dtype::Int32.as_i32())
                .gt(&Array::from_int(0));
            let last =
                ops::where_fn(&attended, &positions, &Array::from_int(-1)).max_axis(1, false);
            let last = ops::maximum(&last, &Array::from_int(0));
            let hidden_dim = hidden.dim(2);
            let indices = ops::broadcast_to(&last.reshape(&[batch, 1, 1]), &[batch, 1, hidden_dim]);
            hidden.take_along_axis(&indices, 1).squeeze_axes(&[1])
        }
        PoolingMode::WeightedMean => {
            // Linearly increasing weights: position i gets weight (i+1)
            let weights = ops::arange(seq_len, Dtype::Float32)
                .add(&Array::from_f32(1.0))
                .reshape(&[1, seq_len, 1]);
            let sum = hidden
                .multiply(&weights)
                .multiply(&mask)
                .sum_axes(&[1], false);
            let weight_sum = ops::maximum(&weights.multiply(&mask).sum_axes(&[1], false), &tiny);
            sum.divide(&weight_sum)
        }
    };
    Ok(pooled.as_dtype(out_dtype))
}

/// L2-normalize embeddings along the last axis.
///
/// Computed in f32 and returned in the dtype of `embeddings`.
///
/// # Arguments
/// * `embeddings` - `[batch, dim]`
///
/// # Returns
/// Unit-normalized embeddings `[batch, dim]`
pub fn normalize_embeddings(embeddings: &Array) -> Result<Array, Exception> {
    let wide = embeddings.as_dtype(Dtype::Float32.as_i32());
    let norm = wide.square().sum_axes(&[-1], true).sqrt();
    let norm = ops::maximum(&norm, &Array::from_f32(1e-12));
    Ok(wide.divide(&norm).as_dtype(embeddings.dtype().as_i32()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_hidden(batch: i32, seq: i32, dim: i32) -> Array {
        Array::from_slice(
            &vec![1.0f32; (batch * seq * dim) as usize],
            &[batch, seq, dim],
        )
    }

    #[test]
    fn test_mean_pooling_no_padding() {
        let hidden = make_hidden(2, 4, 8);
        let mask = Array::from_slice(&[1i32, 1, 1, 1, 1, 1, 1, 1], &[2, 4]);
        let out = pool(&hidden, &mask, PoolingMode::Mean).unwrap();
        out.eval().unwrap();
        assert_eq!(out.shape(), &[2, 8]);
        // Mean of all-ones is 1.0 — check first element
        let v: f32 = out.flatten(0, -1).index(0).item();
        assert!((v - 1.0).abs() < 1e-5, "expected 1.0, got {}", v);
    }

    #[test]
    fn test_cls_pooling() {
        let batch = 2i32;
        let seq = 4i32;
        let dim = 8i32;
        let mut data = vec![1.0f32; (batch * seq * dim) as usize];
        for b in 0..batch as usize {
            for d in 0..dim as usize {
                data[b * seq as usize * dim as usize + d] = 2.0;
            }
        }
        let hidden = Array::from_slice(&data, &[batch, seq, dim]);
        let mask = Array::from_slice(&[1i32, 1, 1, 1, 1, 1, 1, 1], &[2, 4]);
        let out = pool(&hidden, &mask, PoolingMode::Cls).unwrap();
        out.eval().unwrap();
        assert_eq!(out.shape(), &[2, 8]);
        let v: f32 = out.flatten(0, -1).index(0).item();
        assert!((v - 2.0).abs() < 1e-5, "expected 2.0, got {}", v);
    }

    /// `[batch, seq, 1]` hidden states whose value at position `p` is `p`, so a
    /// pooled value names the position it came from.
    fn positional(batch: i32, seq: i32) -> Array {
        let values: Vec<f32> = (0..batch * seq).map(|i| (i % seq) as f32).collect();
        Array::from_slice(&values, &[batch, seq, 1])
    }

    fn values(array: &Array) -> Vec<f32> {
        let n = array.size();
        let mut array = array.as_dtype(Dtype::Float32.as_i32());
        array.to_f32_vec(n).unwrap()
    }

    /// ⚠️ The last token is the last *attended* one, wherever the padding is.
    /// Qwen3-Embedding (and every other last-token embedder) pads on the left,
    /// where `count - 1` indexes a pad position instead of the final token.
    #[test]
    fn last_token_pooling_handles_left_and_right_padding() {
        let hidden = positional(3, 5);
        let mask = Array::from_slice(
            &[
                0i32, 0, 1, 1, 1, // left-padded: last real token is position 4
                1, 1, 1, 0, 0, // right-padded: position 2
                1, 1, 1, 1, 1, // unpadded: position 4
            ],
            &[3, 5],
        );
        let pooled = pool(&hidden, &mask, PoolingMode::LastToken).unwrap();
        assert_eq!(pooled.shape(), &[3, 1]);
        assert_eq!(values(&pooled), vec![4.0, 2.0, 4.0]);
        pmetal_runtime::check_unobserved_error().unwrap();
    }

    /// A row with nothing attended pools to position 0 rather than reading
    /// out of bounds.
    #[test]
    fn last_token_pooling_of_an_empty_row_stays_in_bounds() {
        let hidden = positional(1, 4);
        let mask = Array::from_slice(&[0i32, 0, 0, 0], &[1, 4]);
        let pooled = pool(&hidden, &mask, PoolingMode::LastToken).unwrap();
        assert_eq!(values(&pooled), vec![0.0]);
        pmetal_runtime::check_unobserved_error().unwrap();
    }

    /// Pooling a bf16 model's states must hand back bf16, not an f32 the
    /// scalars promoted it to.
    #[test]
    fn pooling_keeps_the_dtype_of_the_hidden_states() {
        let hidden = positional(2, 4).as_dtype(Dtype::Bfloat16.as_i32());
        let mask = Array::from_slice(&[1i32, 1, 1, 0, 1, 1, 1, 1], &[2, 4]);
        for mode in [
            PoolingMode::Mean,
            PoolingMode::Cls,
            PoolingMode::Max,
            PoolingMode::LastToken,
            PoolingMode::WeightedMean,
        ] {
            let pooled = pool(&hidden, &mask, mode).unwrap();
            assert_eq!(pooled.dtype(), Dtype::Bfloat16, "{mode}");
        }
        let normed = normalize_embeddings(&hidden.reshape(&[2, 4])).unwrap();
        assert_eq!(normed.dtype(), Dtype::Bfloat16);
        pmetal_runtime::check_unobserved_error().unwrap();
    }

    /// Mean and weighted-mean over a long bf16 sequence accumulate in f32: the
    /// position weights 1..=seq are not all representable in bf16 (257 is not),
    /// so weighting in the model dtype shifts the answer.
    #[test]
    fn weighted_mean_is_exact_over_a_long_bf16_sequence() {
        let seq = 600;
        let hidden = Array::from_slice(&vec![1.0f32; seq as usize], &[1, seq, 1])
            .as_dtype(Dtype::Bfloat16.as_i32());
        let mask = Array::from_slice(&vec![1i32; seq as usize], &[1, seq]);
        let pooled = pool(&hidden, &mask, PoolingMode::WeightedMean).unwrap();
        // Every state is 1, so any weighting of them is 1.
        assert_eq!(values(&pooled), vec![1.0]);
        let mean = pool(&hidden, &mask, PoolingMode::Mean).unwrap();
        assert_eq!(values(&mean), vec![1.0]);
    }

    #[test]
    fn mean_and_weighted_mean_skip_padding() {
        let hidden = positional(1, 4); // 0 1 2 3
        let mask = Array::from_slice(&[1i32, 1, 0, 0], &[1, 4]);
        let mean = values(&pool(&hidden, &mask, PoolingMode::Mean).unwrap());
        assert!((mean[0] - 0.5).abs() < 1e-6, "{mean:?}");
        // Weights 1 and 2 on states 0 and 1: (0*1 + 1*2) / (1 + 2).
        let weighted = values(&pool(&hidden, &mask, PoolingMode::WeightedMean).unwrap());
        assert!((weighted[0] - 2.0 / 3.0).abs() < 1e-6, "{weighted:?}");
    }

    #[test]
    fn max_pooling_ignores_padding() {
        let hidden = positional(1, 4);
        let mask = Array::from_slice(&[1i32, 1, 0, 0], &[1, 4]);
        let pooled = pool(&hidden, &mask, PoolingMode::Max).unwrap();
        assert_eq!(values(&pooled), vec![1.0]);
    }

    /// A sentence-transformers directory with the given module list, pooling
    /// flags and (optionally) max length.
    fn st_dir(modules: &str, pooling: &str, max_seq: Option<usize>) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("modules.json"), modules).unwrap();
        std::fs::create_dir(dir.path().join("1_Pooling")).unwrap();
        std::fs::write(dir.path().join("1_Pooling/config.json"), pooling).unwrap();
        if let Some(n) = max_seq {
            std::fs::write(
                dir.path().join("sentence_bert_config.json"),
                format!(r#"{{"max_seq_length": {n}, "do_lower_case": false}}"#),
            )
            .unwrap();
        }
        dir
    }

    const MINILM_MODULES: &str = r#"[
        {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
        {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
        {"idx": 2, "name": "2", "path": "2_Normalize", "type": "sentence_transformers.models.Normalize"}
    ]"#;

    /// The layout `sentence-transformers/all-MiniLM-L6-v2` ships: mean pooling,
    /// then normalization, a 256-token limit.
    #[test]
    fn a_sentence_transformers_directory_names_its_pooling() {
        let dir = st_dir(
            MINILM_MODULES,
            r#"{"word_embedding_dimension": 384, "pooling_mode_cls_token": false,
                "pooling_mode_mean_tokens": true, "pooling_mode_max_tokens": false,
                "pooling_mode_mean_sqrt_len_tokens": false}"#,
            Some(256),
        );
        let config = SentenceTransformerConfig::from_dir(dir.path())
            .unwrap()
            .unwrap();
        assert_eq!(
            config,
            SentenceTransformerConfig {
                transformer_path: String::new(),
                pooling: PoolingMode::Mean,
                normalize: true,
                max_seq_length: Some(256),
            }
        );
    }

    #[test]
    fn cls_pooling_without_a_normalize_module_and_a_transformer_subdirectory() {
        let dir = st_dir(
            r#"[{"idx": 0, "name": "0", "path": "0_Transformer",
                 "type": "sentence_transformers.models.Transformer"},
                {"idx": 1, "name": "1", "path": "1_Pooling",
                 "type": "sentence_transformers.models.Pooling"}]"#,
            r#"{"pooling_mode_cls_token": true}"#,
            None,
        );
        let config = SentenceTransformerConfig::from_dir(dir.path())
            .unwrap()
            .unwrap();
        assert_eq!(config.pooling, PoolingMode::Cls);
        assert!(!config.normalize);
        assert_eq!(config.transformer_path, "0_Transformer");
        assert_eq!(config.max_seq_length, None);
    }

    #[test]
    fn a_directory_without_modules_json_is_not_a_sentence_transformers_model() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(
            SentenceTransformerConfig::from_dir(dir.path()).unwrap(),
            None
        );
    }

    /// A pipeline this crate cannot reproduce is refused by name rather than
    /// read as if the module were not there.
    #[test]
    fn pipelines_that_cannot_be_reproduced_are_refused() {
        let dense = r#"[
            {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
            {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
            {"idx": 2, "name": "2", "path": "2_Dense", "type": "sentence_transformers.models.Dense"}
        ]"#;
        let dir = st_dir(dense, r#"{"pooling_mode_mean_tokens": true}"#, None);
        let err = SentenceTransformerConfig::from_dir(dir.path()).unwrap_err();
        assert!(err.to_string().contains("Dense"), "{err}");

        for (flags, needle) in [
            (
                r#"{"pooling_mode_mean_tokens": true, "pooling_mode_cls_token": true}"#,
                "concatenates",
            ),
            (
                r#"{"pooling_mode_mean_sqrt_len_tokens": true}"#,
                "mean_sqrt_len_tokens",
            ),
            (r#"{"pooling_mode_mean_tokens": false}"#, "no pooling mode"),
        ] {
            let dir = st_dir(MINILM_MODULES, flags, None);
            let err = SentenceTransformerConfig::from_dir(dir.path()).unwrap_err();
            assert!(err.to_string().contains(needle), "{flags}: {err}");
        }
    }

    #[test]
    fn last_token_and_weighted_mean_pooling_are_read() {
        for (flag, want) in [
            ("pooling_mode_lasttoken", PoolingMode::LastToken),
            (
                "pooling_mode_weightedmean_tokens",
                PoolingMode::WeightedMean,
            ),
            ("pooling_mode_max_tokens", PoolingMode::Max),
        ] {
            let dir = st_dir(MINILM_MODULES, &format!(r#"{{"{flag}": true}}"#), None);
            let config = SentenceTransformerConfig::from_dir(dir.path())
                .unwrap()
                .unwrap();
            assert_eq!(config.pooling, want, "{flag}");
        }
    }

    #[test]
    fn test_normalize_embeddings() {
        let emb = Array::from_slice(&[3.0f32, 4.0], &[1, 2]);
        let normed = normalize_embeddings(&emb).unwrap();
        normed.eval().unwrap();
        // norm = 5.0, so [0.6, 0.8]
        let flat = normed.flatten(0, -1);
        let v0: f32 = flat.index(0).item();
        let v1: f32 = flat.index(1).item();
        assert!((v0 - 0.6).abs() < 1e-5);
        assert!((v1 - 0.8).abs() < 1e-5);
    }
}
