//! Rotary position embeddings: the frequencies of every `rope_type`
//! transformers defines, and the one rotation that applies them.
//!
//! [`Rotary`] reads a config's `rope_scaling` / `rope_parameters` the way
//! transformers' `standardize_rope_params` and `ROPE_INIT_FUNCTIONS`
//! (`modeling_rope_utils.py`) read them, and computes the inverse frequencies
//! and the attention factor in f32, in the same order of operations.
//! [`RotaryEmbedding`] rotates queries and keys with them.
//!
//! Every engine builds its rotation here, the native bridges and the
//! `pmetal-models` architectures alike, so each scaling type has exactly one
//! implementation. A `rope_type` this module does not know is refused by name
//! rather than run as plain RoPE.
//!
//! | `rope_type`    | frequencies                                           | attention factor |
//! |----------------|-------------------------------------------------------|------------------|
//! | `default`      | `1 / theta^(2i/d)`                                     | 1                |
//! | `linear`       | default `/ factor`                                     | 1                |
//! | `dynamic`      | default at an NTK-raised `theta`, from the reach      | 1                |
//! | `yarn`         | interpolated/extrapolated blend over a ramp           | `get_mscale`     |
//! | `longrope`     | default `/ short_factor` or `/ long_factor` by reach  | LongRoPE's       |
//! | `llama3`       | three wavelength bands                                | 1                |
//! | `proportional` | exponent over the whole head, unrotated tail at 0     | 1                |
//!
//! "Reach" is one past the largest position a forward rotates, transformers'
//! `max(position_ids) + 1`; only `dynamic` and `longrope` depend on it.

use serde_json::{Map, Value};

use crate::compat::{Array, Dtype, ops};

/// The `rope_type` spellings [`RopeScaling::from_params`] accepts.
pub const SUPPORTED_ROPE_TYPES: &str = "\"default\", \"linear\", \"dynamic\", \"yarn\", \
     \"longrope\", \"llama3\", \"proportional\"";

/// transformers' YaRN `get_mscale`: `0.1 * mscale * ln(scale) + 1`, and 1 for
/// `scale <= 1`.
///
/// The paper's (arXiv 2309.00071, eq. 22) `sqrt(1/t) = 0.1 ln(s) + 1`.
/// DeepSeek's attention also squares it, at `mscale_all_dim`, into its softmax
/// scale.
pub fn yarn_get_mscale(scale: f64, mscale: f64) -> f64 {
    if scale <= 1.0 {
        1.0
    } else {
        0.1 * mscale * scale.ln() + 1.0
    }
}

/// Static YaRN (`rope_type: "yarn"`), resolved the way
/// `_compute_yarn_parameters` resolves it.
#[derive(Debug, Clone, PartialEq)]
pub struct Yarn {
    /// How far the context is stretched.
    pub factor: f64,
    /// The context length the model was pretrained for.
    pub original_max_position_embeddings: f64,
    /// Rotations past which a frequency is left as trained (default 32).
    pub beta_fast: f64,
    /// Rotations under which a frequency is fully interpolated (default 1).
    pub beta_slow: f64,
    /// Whether the correction range is rounded out to whole frequencies
    /// (default `true`; gpt-oss ships `false`).
    pub truncate: bool,
    /// Gain on the rotated channels of queries and keys (`cos` and `sin` both
    /// carry it).
    pub attention_factor: f64,
    /// `mscale_all_dim` as the config gave it. DeepSeek's attention squares
    /// `get_mscale(factor, mscale_all_dim)` into its softmax scale.
    pub mscale_all_dim: Option<f64>,
}

/// LongRoPE (`rope_type: "longrope"`, legacy `"su"`), resolved the way
/// `_compute_longrope_parameters` resolves it.
#[derive(Debug, Clone, PartialEq)]
pub struct LongRope {
    /// Per-frequency divisors while the reach stays within the pretraining
    /// length.
    pub short_factor: Vec<f64>,
    /// Per-frequency divisors once the reach passes it.
    pub long_factor: Vec<f64>,
    /// The pretraining length the two tables switch at.
    pub original_max_position_embeddings: f64,
    /// Gain on the rotated channels, `sqrt(1 + ln(factor) / ln(original))`
    /// unless the config names one. Fixed: it does not follow the table.
    pub attention_factor: f64,
}

/// Llama 3 frequency bands (`rope_type: "llama3"`).
#[derive(Debug, Clone, PartialEq)]
pub struct Llama3 {
    /// Divisor of the low-frequency band.
    pub factor: f64,
    /// `original / low_freq_factor` is the low-frequency wavelength boundary.
    pub low_freq_factor: f64,
    /// `original / high_freq_factor` is the high-frequency wavelength boundary.
    pub high_freq_factor: f64,
    /// The context length the model was pretrained for.
    pub original_max_position_embeddings: f64,
}

/// How a rotary embedding's frequencies are scaled: transformers'
/// `rope_type`, with its parameters resolved.
#[derive(Debug, Clone, PartialEq)]
pub enum RopeScaling {
    /// Plain RoPE.
    Default,
    /// Positions divided by `factor`.
    Linear {
        /// Divisor.
        factor: f64,
    },
    /// NTK-aware base raised from the reach once it passes
    /// `max_position_embeddings`.
    Dynamic {
        /// Scaling factor.
        factor: f64,
        /// The length past which the base starts to grow.
        max_position_embeddings: f64,
    },
    /// YaRN.
    Yarn(Yarn),
    /// LongRoPE.
    LongRope(LongRope),
    /// Llama 3 bands.
    Llama3(Llama3),
    /// Gemma 4's proportional RoPE: the exponent runs over the whole head and
    /// the frequencies past `partial_rotary_factor` are zero.
    Proportional {
        /// Divisor of every frequency (default 1).
        factor: f64,
    },
}

impl RopeScaling {
    /// The canonical `rope_type` spelling.
    pub fn rope_type(&self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Linear { .. } => "linear",
            Self::Dynamic { .. } => "dynamic",
            Self::Yarn(_) => "yarn",
            Self::LongRope(_) => "longrope",
            Self::Llama3(_) => "llama3",
            Self::Proportional { .. } => "proportional",
        }
    }

    /// Whether the frequencies depend on how far a forward reaches.
    pub fn depends_on_reach(&self) -> bool {
        matches!(self, Self::Dynamic { .. } | Self::LongRope(_))
    }

    /// Resolve a rope dict (`rope_scaling` or `rope_parameters`; `None` for
    /// plain RoPE).
    ///
    /// `config` supplies what transformers reads from outside the dict:
    /// `max_position_embeddings` (the default `original_max_position_embeddings`,
    /// the YaRN/LongRoPE default factor, the dynamic-NTK threshold) and a
    /// top-level `original_max_position_embeddings`, which wins over the
    /// dict's as it does for Phi-3. A missing required key or an unknown
    /// `rope_type` is an error naming it.
    pub fn from_params(
        params: Option<&Map<String, Value>>,
        config: &RopeConfig<'_>,
    ) -> Result<Self, String> {
        let Some(p) = params else {
            return Ok(Self::Default);
        };
        let rope_type = rope_type_of(p)?;
        let num = |key: &str| p.get(key).filter(|v| !v.is_null()).and_then(Value::as_f64);
        let required = |key: &str| {
            num(key).ok_or_else(|| format!("rope_type {rope_type:?} needs a numeric `{key}`"))
        };
        let original = || {
            config
                .original_max_position_embeddings
                .or_else(|| num("original_max_position_embeddings"))
                .or(config.max_position_embeddings)
                .ok_or_else(|| {
                    format!("rope_type {rope_type:?} needs `original_max_position_embeddings`")
                })
        };
        let at_least_one = |what: &str, factor: f64| {
            if factor.is_finite() && factor >= 1.0 {
                Ok(factor)
            } else {
                Err(format!(
                    "rope_type {rope_type:?}: {what} {factor} must be at least 1"
                ))
            }
        };
        Ok(match rope_type.as_str() {
            // Qwen2-VL's legacy "mrope" is the default rotation per axis.
            "default" | "mrope" => Self::Default,
            "linear" => Self::Linear {
                factor: at_least_one("factor", required("factor")?)?,
            },
            "dynamic" => Self::Dynamic {
                factor: at_least_one("factor", required("factor")?)?,
                max_position_embeddings: config
                    .max_position_embeddings
                    .ok_or("rope_type \"dynamic\" needs the config's `max_position_embeddings`")?,
            },
            "yarn" => {
                let original = original()?;
                // An absent or null factor is the ratio of the two lengths
                // (transformers' DeepSeek-V3 note).
                let factor = match num("factor") {
                    Some(f) => f,
                    None => {
                        config
                            .max_position_embeddings
                            .ok_or("rope_type \"yarn\" needs a `factor`")?
                            / original
                    }
                };
                let factor = at_least_one("factor", factor)?;
                let mscale = num("mscale");
                let mscale_all_dim = num("mscale_all_dim");
                let attention_factor = match num("attention_factor") {
                    Some(af) => af,
                    // `if mscale and mscale_all_dim`: both present and non-zero.
                    None => match (mscale, mscale_all_dim) {
                        (Some(m), Some(all)) if m != 0.0 && all != 0.0 => {
                            yarn_get_mscale(factor, m) / yarn_get_mscale(factor, all)
                        }
                        _ => yarn_get_mscale(factor, 1.0),
                    },
                };
                Self::Yarn(Yarn {
                    factor,
                    original_max_position_embeddings: original,
                    // `beta_fast or 32`: a zero falls back too.
                    beta_fast: num("beta_fast").filter(|&b| b != 0.0).unwrap_or(32.0),
                    beta_slow: num("beta_slow").filter(|&b| b != 0.0).unwrap_or(1.0),
                    truncate: p.get("truncate").and_then(Value::as_bool).unwrap_or(true),
                    attention_factor,
                    mscale_all_dim,
                })
            }
            "longrope" | "su" => {
                let factors = |key: &str| -> Result<Vec<f64>, String> {
                    p.get(key)
                        .and_then(Value::as_array)
                        .and_then(|a| a.iter().map(Value::as_f64).collect::<Option<Vec<_>>>())
                        .ok_or_else(|| {
                            format!("rope_type \"longrope\" needs `{key}`, a list of numbers")
                        })
                };
                let original = original()?;
                let factor = match num("factor") {
                    Some(f) => f,
                    None => config.max_position_embeddings.unwrap_or(original) / original,
                };
                let attention_factor = num("attention_factor").unwrap_or(if factor <= 1.0 {
                    1.0
                } else {
                    (1.0 + factor.ln() / original.ln()).sqrt()
                });
                Self::LongRope(LongRope {
                    short_factor: factors("short_factor")?,
                    long_factor: factors("long_factor")?,
                    original_max_position_embeddings: original,
                    attention_factor,
                })
            }
            "llama3" => Self::Llama3(Llama3 {
                factor: at_least_one("factor", required("factor")?)?,
                low_freq_factor: required("low_freq_factor")?,
                high_freq_factor: required("high_freq_factor")?,
                original_max_position_embeddings: original()?,
            }),
            "proportional" => Self::Proportional {
                factor: num("factor").unwrap_or(1.0),
            },
            other => {
                return Err(format!(
                    "rope_type {other:?} is not supported; supported: {SUPPORTED_ROPE_TYPES}"
                ));
            }
        })
    }
}

fn rope_type_of(p: &Map<String, Value>) -> Result<String, String> {
    match p
        .get("rope_type")
        .filter(|v| !v.is_null())
        .or_else(|| p.get("type"))
    {
        None | Some(Value::Null) => Ok("default".to_string()),
        Some(Value::String(s)) => Ok(s.clone()),
        Some(other) => Err(format!("rope_type must be a string, got {other}")),
    }
}

/// The parts of a model config the rotary embedding reads.
#[derive(Debug, Clone, Copy, Default)]
pub struct RopeConfig<'a> {
    /// The legacy `rope_scaling` object.
    pub rope_scaling: Option<&'a Value>,
    /// The `rope_parameters` object.
    pub rope_parameters: Option<&'a Value>,
    /// Top-level `rope_theta`.
    pub rope_theta: Option<f64>,
    /// Top-level `partial_rotary_factor`.
    pub partial_rotary_factor: Option<f64>,
    /// `max_position_embeddings`.
    pub max_position_embeddings: Option<f64>,
    /// A top-level `original_max_position_embeddings`.
    pub original_max_position_embeddings: Option<f64>,
}

impl<'a> RopeConfig<'a> {
    /// The keys of a flat (text) `config.json` object.
    pub fn from_json(config: &'a Value) -> Self {
        let num = |key: &str| {
            config
                .get(key)
                .filter(|v| !v.is_null())
                .and_then(Value::as_f64)
        };
        Self {
            rope_scaling: config.get("rope_scaling"),
            rope_parameters: config.get("rope_parameters"),
            rope_theta: num("rope_theta"),
            partial_rotary_factor: num("partial_rotary_factor"),
            max_position_embeddings: num("max_position_embeddings"),
            original_max_position_embeddings: num("original_max_position_embeddings"),
        }
    }

    /// The rope dict in force: a non-empty `rope_scaling` wins over
    /// `rope_parameters`, as in transformers' `convert_rope_params_to_dict`.
    pub fn params(&self) -> Result<Option<&'a Map<String, Value>>, String> {
        let object = |name: &str, v: Option<&'a Value>| match v {
            None | Some(Value::Null) => Ok(None),
            Some(Value::Object(m)) => Ok(Some(m)),
            Some(other) => Err(format!("{name} must be an object, got {other}")),
        };
        let scaling = object("rope_scaling", self.rope_scaling)?.filter(|m| !m.is_empty());
        match scaling {
            Some(m) => Ok(Some(m)),
            None => Ok(object("rope_parameters", self.rope_parameters)?.filter(|m| !m.is_empty())),
        }
    }
}

/// A model's rotary embedding: which channels rotate, at what base, and how
/// the frequencies are scaled.
#[derive(Debug, Clone, PartialEq)]
pub struct Rotary {
    /// Channels per attention head.
    pub head_dim: i32,
    /// Fraction of `head_dim` that rotates.
    pub partial_rotary_factor: f64,
    /// Channels the rotation spans: `int(head_dim * partial_rotary_factor)`,
    /// or the whole head for `proportional`, whose tail rotates at frequency 0.
    pub dims: i32,
    /// `rope_theta`.
    pub theta: f64,
    /// How the frequencies are scaled.
    pub scaling: RopeScaling,
}

impl Rotary {
    /// Assemble from resolved parts.
    pub fn new(
        head_dim: i32,
        partial_rotary_factor: f64,
        theta: f64,
        scaling: RopeScaling,
    ) -> Self {
        let dims = match scaling {
            RopeScaling::Proportional { .. } => head_dim,
            _ => (head_dim as f64 * partial_rotary_factor) as i32,
        };
        Self {
            head_dim,
            partial_rotary_factor,
            dims,
            theta,
            scaling,
        }
    }

    /// Plain RoPE over the first `dims` of `head_dim` channels.
    pub fn plain(head_dim: i32, dims: i32, theta: f64) -> Self {
        Self {
            head_dim,
            partial_rotary_factor: dims as f64 / head_dim as f64,
            dims,
            theta,
            scaling: RopeScaling::Default,
        }
    }

    /// Resolve a model config's rotary embedding.
    ///
    /// `rope_theta` and `partial_rotary_factor` inside the rope dict win over
    /// the top-level keys, which win over the architecture's defaults.
    /// Unsupported or malformed scaling is an error naming the key.
    pub fn from_config(
        head_dim: i32,
        config: RopeConfig<'_>,
        default_theta: f64,
        default_partial_rotary_factor: f64,
    ) -> Result<Self, String> {
        let params = config.params()?;
        let num = |key: &str| {
            params
                .and_then(|p| p.get(key))
                .filter(|v| !v.is_null())
                .and_then(Value::as_f64)
        };
        let theta = num("rope_theta")
            .or(config.rope_theta)
            .unwrap_or(default_theta);
        let partial = num("partial_rotary_factor")
            .or(config.partial_rotary_factor)
            .unwrap_or(default_partial_rotary_factor);
        if !(theta.is_finite() && theta > 0.0) {
            return Err(format!("rope_theta {theta} must be positive"));
        }
        if !(partial > 0.0 && partial <= 1.0) {
            return Err(format!("partial_rotary_factor {partial} is outside (0, 1]"));
        }
        let rotary = Self::new(
            head_dim,
            partial,
            theta,
            RopeScaling::from_params(params, &config)?,
        );
        rotary.validate()?;
        Ok(rotary)
    }

    /// Check the shape the frequencies need.
    pub fn validate(&self) -> Result<(), String> {
        if self.dims < 2 || self.dims % 2 != 0 || self.dims > self.head_dim {
            return Err(format!(
                "head_dim {} x partial_rotary_factor {} = {} rotary dims; expected a positive \
                 even number no larger than the head",
                self.head_dim, self.partial_rotary_factor, self.dims
            ));
        }
        if let RopeScaling::LongRope(l) = &self.scaling {
            let half = (self.dims / 2) as usize;
            for (name, f) in [
                ("short_factor", &l.short_factor),
                ("long_factor", &l.long_factor),
            ] {
                if f.len() != half {
                    return Err(format!(
                        "longrope {name} has {} entries; {} rotary dims need {half}",
                        f.len(),
                        self.dims
                    ));
                }
            }
        }
        Ok(())
    }

    /// The `(base, position scale)` pair the fused kernel takes, when that is
    /// all this rotation is: plain or linear RoPE.
    pub fn scalar(&self) -> Option<(f32, f32)> {
        match self.scaling {
            RopeScaling::Default => Some((self.theta as f32, 1.0)),
            RopeScaling::Linear { factor } => Some((self.theta as f32, (1.0 / factor) as f32)),
            _ => None,
        }
    }

    /// Gain on the rotated channels of queries and keys (`1.0` unless YaRN or
    /// LongRoPE).
    pub fn attention_factor(&self) -> f32 {
        match &self.scaling {
            RopeScaling::Yarn(y) => y.attention_factor as f32,
            RopeScaling::LongRope(l) => l.attention_factor as f32,
            _ => 1.0,
        }
    }

    /// `rope_theta` for a forward reaching `reach` positions: raised by
    /// dynamic NTK past `max_position_embeddings`, unchanged otherwise.
    ///
    /// transformers computes the raised base in f32 (its `seq_len` is a
    /// tensor), so this does too.
    pub fn base_at(&self, reach: i64) -> f32 {
        let theta = self.theta as f32;
        let RopeScaling::Dynamic {
            factor,
            max_position_embeddings,
        } = self.scaling
        else {
            return theta;
        };
        let max_pos = max_position_embeddings as f32;
        let seq_len = (reach as f32).max(max_pos);
        let factor = factor as f32;
        let dim = self.dims as f32;
        theta * ((factor * seq_len / max_pos) - (factor - 1.0)).powf(dim / (dim - 2.0))
    }

    /// The `[dims / 2]` inverse frequencies for a forward reaching `reach`
    /// positions (`max(position_ids) + 1`), computed in f32 as transformers
    /// computes them.
    pub fn inverse_frequencies(&self, reach: i64) -> Vec<f32> {
        let dims = self.dims as usize;
        let half = dims / 2;
        let exponent = |i: usize| (2 * i) as f32 / dims as f32;
        let theta = self.base_at(reach);
        let pos_freqs = || -> Vec<f32> { (0..half).map(|i| theta.powf(exponent(i))).collect() };
        match &self.scaling {
            RopeScaling::Default | RopeScaling::Dynamic { .. } => {
                pos_freqs().iter().map(|p| 1.0 / p).collect()
            }
            RopeScaling::Linear { factor } => {
                let factor = *factor as f32;
                pos_freqs().iter().map(|p| (1.0 / p) / factor).collect()
            }
            RopeScaling::Yarn(yarn) => yarn_inverse_frequencies(yarn, self.dims, self.theta),
            RopeScaling::LongRope(l) => {
                let ext = if reach > l.original_max_position_embeddings as i64 {
                    &l.long_factor
                } else {
                    &l.short_factor
                };
                pos_freqs()
                    .iter()
                    .zip(ext)
                    .map(|(p, &e)| 1.0 / (e as f32 * p))
                    .collect()
            }
            RopeScaling::Llama3(l) => llama3_inverse_frequencies(l, &pos_freqs()),
            RopeScaling::Proportional { factor } => {
                // `rope_angles = int(partial * head_dim // 2)` frequencies at
                // `theta^(2i / head_dim)`, then zeros.
                let angles = (self.partial_rotary_factor * self.head_dim as f64 / 2.0) as usize;
                let factor = *factor as f32;
                (0..half)
                    .map(|i| {
                        if i < angles {
                            (1.0 / theta.powf(exponent(i))) / factor
                        } else {
                            0.0
                        }
                    })
                    .collect()
            }
        }
    }

    /// The reciprocals of [`inverse_frequencies`](Self::inverse_frequencies),
    /// the period table the fused kernel's `freqs` argument takes. A zero
    /// frequency becomes `+inf`, which the kernel rotates by nothing.
    pub fn periods(&self, reach: i64) -> Vec<f32> {
        self.inverse_frequencies(reach)
            .iter()
            .map(|&f| if f == 0.0 { f32::INFINITY } else { 1.0 / f })
            .collect()
    }
}

/// `_compute_yarn_parameters`' frequencies: the trained ones blended with the
/// interpolated `1 / (factor * base^(2i/dims))` by a linear ramp over the
/// correction range, in f32.
fn yarn_inverse_frequencies(yarn: &Yarn, dims: i32, theta: f64) -> Vec<f32> {
    let half = (dims / 2) as usize;
    let base = theta as f32;
    let pos_freqs: Vec<f32> = (0..half)
        .map(|i| base.powf((2 * i) as f32 / dims as f32))
        .collect();
    let correction_dim = |rotations: f64| {
        (dims as f64
            * (yarn.original_max_position_embeddings / (rotations * 2.0 * std::f64::consts::PI))
                .ln())
            / (2.0 * theta.ln())
    };
    let (mut low, mut high) = (
        correction_dim(yarn.beta_fast),
        correction_dim(yarn.beta_slow),
    );
    if yarn.truncate {
        low = low.floor();
        high = high.ceil();
    }
    let low = low.max(0.0);
    let mut high = high.min((dims - 1) as f64);
    if low == high {
        high += 0.001; // transformers' guard against a zero-width ramp
    }
    let factor = yarn.factor as f32;
    let (low, width) = (low as f32, (high - low) as f32);
    (0..half)
        .map(|i| {
            let ramp = ((i as f32 - low) / width).clamp(0.0, 1.0);
            let extrapolation = 1.0 - ramp;
            let inv_extrapolation = 1.0 / pos_freqs[i];
            let inv_interpolation = 1.0 / (factor * pos_freqs[i]);
            inv_interpolation * (1.0 - extrapolation) + inv_extrapolation * extrapolation
        })
        .collect()
}

/// `_compute_llama3_parameters`, in f32.
fn llama3_inverse_frequencies(l: &Llama3, pos_freqs: &[f32]) -> Vec<f32> {
    let factor = l.factor as f32;
    let old = l.original_max_position_embeddings as f32;
    let low_wavelen = (l.original_max_position_embeddings / l.low_freq_factor) as f32;
    let high_wavelen = (l.original_max_position_embeddings / l.high_freq_factor) as f32;
    let (low, width) = (
        l.low_freq_factor as f32,
        (l.high_freq_factor - l.low_freq_factor) as f32,
    );
    pos_freqs
        .iter()
        .map(|p| {
            let inv = 1.0 / p;
            let wavelen = (2.0 * std::f64::consts::PI) as f32 / inv;
            let banded = if wavelen > low_wavelen {
                inv / factor
            } else {
                inv
            };
            // Wavelengths are finite, so the reference's `~(w < high) & ~(w > low)`
            // is the closed band.
            let medium = (high_wavelen..=low_wavelen).contains(&wavelen);
            if medium && width != 0.0 {
                let smooth = (old / wavelen - low) / width;
                (1.0 - smooth) * banded / factor + smooth * banded
            } else {
                banded
            }
        })
        .collect()
}

/// A [`Rotary`] ready to rotate `[B, heads, L, head_dim]` queries and keys.
///
/// Contiguous positions run the fused kernel: from `(base, scale)` for plain,
/// linear and dynamic RoPE, from an explicit period table otherwise.
/// Explicit positions build the `cos`/`sin` tables from the same frequencies.
/// Either way the attention factor scales the rotated channels only, as
/// transformers' `cos * attention_scaling` does, and the result keeps the
/// input's dtype.
#[derive(Clone)]
pub struct RotaryEmbedding {
    rotary: Rotary,
    traditional: bool,
    /// The period table for a reach within the pretraining length (every
    /// reach for a static table); `None` when the kernel's scalars suffice.
    periods: Option<Array>,
    /// LongRoPE's table past the pretraining length.
    long_periods: Option<Array>,
}

impl std::fmt::Debug for RotaryEmbedding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RotaryEmbedding")
            .field("rotary", &self.rotary)
            .field("traditional", &self.traditional)
            .finish()
    }
}

impl RotaryEmbedding {
    /// `traditional`: interleaved pairs `(x[2i], x[2i+1])` instead of the
    /// split halves `(x[i], x[i + dims/2])`.
    pub fn new(rotary: Rotary, traditional: bool) -> Self {
        let table = |reach: i64| {
            let periods = rotary.periods(reach);
            let n = periods.len() as i32;
            Array::from_f32_slice(&periods, &[n])
        };
        let (periods, long_periods) = match &rotary.scaling {
            RopeScaling::Default | RopeScaling::Linear { .. } | RopeScaling::Dynamic { .. } => {
                (None, None)
            }
            RopeScaling::LongRope(l) => {
                let original = l.original_max_position_embeddings as i64;
                (Some(table(original)), Some(table(original + 1)))
            }
            _ => (Some(table(0)), None),
        };
        Self {
            rotary,
            traditional,
            periods,
            long_periods,
        }
    }

    /// Plain RoPE over the first `dims` of `head_dim` channels.
    pub fn plain(head_dim: i32, dims: i32, theta: f64, traditional: bool) -> Self {
        Self::new(Rotary::plain(head_dim, dims, theta), traditional)
    }

    /// [`Rotary::from_config`], ready to rotate.
    pub fn from_config(
        head_dim: i32,
        config: RopeConfig<'_>,
        default_theta: f64,
        default_partial_rotary_factor: f64,
        traditional: bool,
    ) -> Result<Self, String> {
        Rotary::from_config(
            head_dim,
            config,
            default_theta,
            default_partial_rotary_factor,
        )
        .map(|rotary| Self::new(rotary, traditional))
    }

    /// The rotary embedding this applies.
    pub fn rotary(&self) -> &Rotary {
        &self.rotary
    }

    /// Channels the rotation spans.
    pub fn dims(&self) -> i32 {
        self.rotary.dims
    }

    /// Interleaved (`true`) or split-half pairs.
    pub fn traditional(&self) -> bool {
        self.traditional
    }

    /// See [`Rotary::scalar`].
    pub fn scalar(&self) -> Option<(f32, f32)> {
        self.rotary.scalar()
    }

    /// See [`Rotary::attention_factor`].
    pub fn attention_factor(&self) -> f32 {
        self.rotary.attention_factor()
    }

    /// See [`Rotary::inverse_frequencies`].
    pub fn inverse_frequencies(&self, reach: i64) -> Vec<f32> {
        self.rotary.inverse_frequencies(reach)
    }

    /// Rotate `x` (`[B, heads, L, head_dim]`) at positions
    /// `offset..offset + L`.
    pub fn apply(&self, x: &Array, offset: i32) -> Array {
        let reach = offset as i64 + x.dim(x.ndim() - 2) as i64;
        let dims = self.rotary.dims;
        let rotated = match self.table_for(reach) {
            Some(periods) => x.rope_with_freqs(dims, self.traditional, 1.0, offset, periods),
            None => {
                let scale = match self.rotary.scaling {
                    RopeScaling::Linear { factor } => (1.0 / factor) as f32,
                    _ => 1.0,
                };
                x.rope(
                    dims,
                    self.traditional,
                    self.rotary.base_at(reach),
                    scale,
                    offset,
                )
            }
        };
        self.with_gain(rotated, x)
    }

    /// Rotate `x` (`[B, heads, L, head_dim]`) at one explicit position per
    /// token (`positions`, `[L]` integers). The reach is read from the
    /// positions only when the frequencies depend on it.
    pub fn apply_at(&self, x: &Array, positions: &Array) -> Array {
        let reach = if self.rotary.scaling.depends_on_reach() {
            positions
                .as_dtype(Dtype::Float32.as_i32())
                .max(None)
                .item_f32() as i64
                + 1
        } else {
            0
        };
        let inv_freq = self.rotary.inverse_frequencies(reach);
        let half = inv_freq.len() as i32;
        let inv_freq = Array::from_f32_slice(&inv_freq, &[1, half]);
        let angles = positions
            .as_dtype(Dtype::Float32.as_i32())
            .reshape(&[-1, 1])
            .multiply(&inv_freq);
        let (mut cos, mut sin) = (ops::cos(&angles), ops::sin(&angles));
        let gain = self.rotary.attention_factor();
        if gain != 1.0 {
            let gain = Array::from_f32(gain);
            cos = cos.multiply(&gain);
            sin = sin.multiply(&gain);
        }
        let shape = [1, 1, -1, half];
        rotate_with_cos_sin(
            &x.as_dtype(Dtype::Float32.as_i32()),
            &cos.reshape(&shape),
            &sin.reshape(&shape),
            self.rotary.dims,
            self.traditional,
        )
        .as_dtype(x.dtype().as_i32())
    }

    fn table_for(&self, reach: i64) -> Option<&Array> {
        match (&self.rotary.scaling, &self.long_periods) {
            (RopeScaling::LongRope(l), Some(long))
                if reach > l.original_max_position_embeddings as i64 =>
            {
                Some(long)
            }
            _ => self.periods.as_ref(),
        }
    }

    /// The attention factor on the rotated channels of a fused-kernel result.
    fn with_gain(&self, rotated: Array, x: &Array) -> Array {
        let gain = self.rotary.attention_factor();
        if gain == 1.0 {
            return rotated;
        }
        let head_dim = x.dim(x.ndim() - 1);
        let gain = if self.rotary.dims >= head_dim {
            Array::from_f32(gain)
        } else {
            let channels: Vec<f32> = (0..head_dim)
                .map(|c| if c < self.rotary.dims { gain } else { 1.0 })
                .collect();
            Array::from_f32_slice(&channels, &[head_dim])
        };
        rotated
            .as_dtype(Dtype::Float32.as_i32())
            .multiply(&gain)
            .as_dtype(x.dtype().as_i32())
    }
}

/// Rotate the first `dims` channels of `x` (`[B, heads, L, head_dim]`) by
/// precomputed `cos`/`sin` tables broadcastable to `[B, heads, L, dims / 2]`,
/// leaving the rest untouched. `traditional` pairs `(x[2i], x[2i+1])`,
/// otherwise `(x[i], x[i + dims/2])`.
///
/// Computes in the promoted dtype of `x` and the tables; callers that need
/// `x`'s dtype back cast the result.
pub fn rotate_with_cos_sin(
    x: &Array,
    cos_theta: &Array,
    sin_theta: &Array,
    dims: i32,
    traditional: bool,
) -> Array {
    let head_dim = x.dim(x.ndim() - 1);
    let half_dims = dims / 2;

    if traditional {
        let x_rope = if dims < head_dim {
            x.split(&[dims], -1).remove(0)
        } else {
            x.clone()
        };
        let shape = x_rope.shape();
        let (batch, heads, seq_len) = (shape[0], shape[1], shape[2]);
        let pairs = x_rope.reshape(&[batch, heads, seq_len, half_dims, 2]);
        let even = pairs
            .slice(&[0, 0, 0, 0, 0], &[batch, heads, seq_len, half_dims, 1])
            .squeeze(-1);
        let odd = pairs
            .slice(&[0, 0, 0, 0, 1], &[batch, heads, seq_len, half_dims, 2])
            .squeeze(-1);
        let r_even = even.multiply(cos_theta).subtract(&odd.multiply(sin_theta));
        let r_odd = even.multiply(sin_theta).add(&odd.multiply(cos_theta));
        let rotated = ops::stack_axis(&[r_even, r_odd], -1).reshape(&[batch, heads, seq_len, dims]);
        if dims < head_dim {
            let parts = x.split(&[dims], -1);
            ops::concatenate_axis(&[&rotated, &parts[1]], -1)
        } else {
            rotated
        }
    } else {
        let parts = if dims == head_dim {
            x.split(&[half_dims], -1)
        } else {
            x.split(&[half_dims, dims], -1)
        };
        let (x1, x2) = (&parts[0], &parts[1]);
        let r1 = x1.multiply(cos_theta).subtract(&x2.multiply(sin_theta));
        let r2 = x1.multiply(sin_theta).add(&x2.multiply(cos_theta));
        let rotated = ops::concatenate_axis(&[&r1, &r2], -1);
        if dims < head_dim && parts.len() > 2 {
            ops::concatenate_axis(&[&rotated, &parts[2]], -1)
        } else {
            rotated
        }
    }
}

#[cfg(test)]
mod tests;
