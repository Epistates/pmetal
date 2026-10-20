//! `GenerationConfig` and `ModelArchitecture`.
//!
//! Training and job settings are the job specs (see `spec.rs`); what is left
//! here is generation, which a loaded `Model` takes per call.

use std::collections::HashMap;

use pmetal_models::GenerationConfig;
use pyo3::prelude::*;

/// Sampling settings for `Model.generate`.
///
/// Every field is optional, and a field left unset keeps the model's own
/// default (its `generation_config.json`, else the library's): a config says
/// only what the caller wants different. Temperature 0 or less is greedy.
#[pyclass(name = "GenerationConfig", module = "pmetal", frozen, from_py_object)]
#[derive(Clone, Default)]
pub struct PyGenerationConfig {
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_k: Option<usize>,
    top_p: Option<f32>,
    min_p: Option<f32>,
    repetition_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    seed: Option<u64>,
    stop_tokens: Option<Vec<u32>>,
    logit_bias: Option<Vec<(u32, f32)>>,
}

impl PyGenerationConfig {
    /// `config` with any keyword that is given replacing its field.
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        config: Option<&PyGenerationConfig>,
        max_tokens: Option<usize>,
        temperature: Option<f32>,
        top_k: Option<usize>,
        top_p: Option<f32>,
        min_p: Option<f32>,
        repetition_penalty: Option<f32>,
        frequency_penalty: Option<f32>,
        presence_penalty: Option<f32>,
        seed: Option<u64>,
        stop_tokens: Option<Vec<u32>>,
        logit_bias: Option<HashMap<u32, f32>>,
    ) -> Self {
        let base = config.cloned().unwrap_or_default();
        Self {
            max_tokens: max_tokens.or(base.max_tokens),
            temperature: temperature.or(base.temperature),
            top_k: top_k.or(base.top_k),
            top_p: top_p.or(base.top_p),
            min_p: min_p.or(base.min_p),
            repetition_penalty: repetition_penalty.or(base.repetition_penalty),
            frequency_penalty: frequency_penalty.or(base.frequency_penalty),
            presence_penalty: presence_penalty.or(base.presence_penalty),
            seed: seed.or(base.seed),
            stop_tokens: stop_tokens.or(base.stop_tokens),
            logit_bias: logit_bias.map(sorted_bias).or(base.logit_bias),
        }
    }

    /// `base` with every field this config sets applied over it.
    pub fn resolve(&self, mut base: GenerationConfig) -> GenerationConfig {
        if let Some(n) = self.max_tokens {
            base.max_new_tokens = n;
        }
        if let Some(t) = self.temperature {
            base.temperature = t;
            // 0 or less is greedy, and a positive temperature samples even
            // when the model's defaults say otherwise.
            base.do_sample = t > 0.0;
        }
        if let Some(k) = self.top_k {
            base.top_k = k;
        }
        if let Some(p) = self.top_p {
            base.top_p = p;
        }
        if let Some(p) = self.min_p {
            base.min_p = p;
        }
        if let Some(p) = self.repetition_penalty {
            base.repetition_penalty = p;
        }
        if let Some(p) = self.frequency_penalty {
            base.frequency_penalty = p;
        }
        if let Some(p) = self.presence_penalty {
            base.presence_penalty = p;
        }
        if let Some(seed) = self.seed {
            base.seed = Some(seed);
        }
        if let Some(stop) = &self.stop_tokens {
            for token in stop {
                if !base.stop_tokens.contains(token) {
                    base.stop_tokens.push(*token);
                }
            }
        }
        if let Some(bias) = &self.logit_bias {
            base.logit_bias.clone_from(bias);
        }
        base
    }
}

fn sorted_bias(bias: HashMap<u32, f32>) -> Vec<(u32, f32)> {
    let mut pairs: Vec<_> = bias.into_iter().collect();
    pairs.sort_by_key(|(token, _)| *token);
    pairs
}

#[pymethods]
impl PyGenerationConfig {
    #[new]
    #[pyo3(signature = (*, max_tokens=None, temperature=None, top_k=None, top_p=None,
        min_p=None, repetition_penalty=None, frequency_penalty=None, presence_penalty=None,
        seed=None, stop_tokens=None, logit_bias=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        max_tokens: Option<usize>,
        temperature: Option<f32>,
        top_k: Option<usize>,
        top_p: Option<f32>,
        min_p: Option<f32>,
        repetition_penalty: Option<f32>,
        frequency_penalty: Option<f32>,
        presence_penalty: Option<f32>,
        seed: Option<u64>,
        stop_tokens: Option<Vec<u32>>,
        logit_bias: Option<HashMap<u32, f32>>,
    ) -> Self {
        Self::from_parts(
            None,
            max_tokens,
            temperature,
            top_k,
            top_p,
            min_p,
            repetition_penalty,
            frequency_penalty,
            presence_penalty,
            seed,
            stop_tokens,
            logit_bias,
        )
    }

    /// Greedy decoding: the likeliest token every step.
    #[staticmethod]
    #[pyo3(signature = (max_tokens=None))]
    fn greedy(max_tokens: Option<usize>) -> Self {
        Self {
            max_tokens,
            temperature: Some(0.0),
            ..Self::default()
        }
    }

    /// Maximum new tokens, or None for the model's default.
    #[getter]
    fn max_tokens(&self) -> Option<usize> {
        self.max_tokens
    }
    /// Sampling temperature, or None for the model's default.
    #[getter]
    fn temperature(&self) -> Option<f32> {
        self.temperature
    }
    /// Top-k, or None for the model's default.
    #[getter]
    fn top_k(&self) -> Option<usize> {
        self.top_k
    }
    /// Top-p, or None for the model's default.
    #[getter]
    fn top_p(&self) -> Option<f32> {
        self.top_p
    }
    /// Min-p, or None for the model's default.
    #[getter]
    fn min_p(&self) -> Option<f32> {
        self.min_p
    }
    /// Repetition penalty, or None for the model's default.
    #[getter]
    fn repetition_penalty(&self) -> Option<f32> {
        self.repetition_penalty
    }
    /// Frequency penalty, or None for the model's default.
    #[getter]
    fn frequency_penalty(&self) -> Option<f32> {
        self.frequency_penalty
    }
    /// Presence penalty, or None for the model's default.
    #[getter]
    fn presence_penalty(&self) -> Option<f32> {
        self.presence_penalty
    }
    /// Seed, or None for a different draw every call.
    #[getter]
    fn seed(&self) -> Option<u64> {
        self.seed
    }
    /// Extra stop token ids.
    #[getter]
    fn stop_tokens(&self) -> Option<Vec<u32>> {
        self.stop_tokens.clone()
    }
    /// Logit biases, `{token_id: bias}`.
    #[getter]
    fn logit_bias(&self) -> Option<HashMap<u32, f32>> {
        self.logit_bias
            .as_ref()
            .map(|pairs| pairs.iter().copied().collect())
    }

    fn __repr__(&self) -> String {
        let mut parts = Vec::new();
        macro_rules! field {
            ($name:ident) => {
                if let Some(v) = &self.$name {
                    parts.push(format!("{}={:?}", stringify!($name), v));
                }
            };
        }
        field!(max_tokens);
        field!(temperature);
        field!(top_k);
        field!(top_p);
        field!(min_p);
        field!(repetition_penalty);
        field!(frequency_penalty);
        field!(presence_penalty);
        field!(seed);
        field!(stop_tokens);
        field!(logit_bias);
        format!("GenerationConfig({})", parts.join(", "))
    }
}

/// A model architecture pmetal implements.
#[pyclass(
    name = "ModelArchitecture",
    module = "pmetal",
    eq,
    eq_int,
    frozen,
    from_py_object
)]
#[derive(Clone, Debug, PartialEq)]
pub enum PyModelArchitecture {
    Llama,
    Llama4,
    Qwen2,
    Qwen3,
    Qwen3MoE,
    Gemma,
    Mistral,
    Phi,
    Phi4,
    DeepSeek,
    Cohere,
    Granite,
    NemotronH,
    Qwen3Next,
    GptOss,
    Gemma4,
    Bert,
    Flux,
    DiffusionGemma,
    Mllama,
    Qwen4Exp,
}

impl From<pmetal_models::ModelArchitecture> for PyModelArchitecture {
    fn from(a: pmetal_models::ModelArchitecture) -> Self {
        use pmetal_models::ModelArchitecture as A;
        match a {
            A::Llama => Self::Llama,
            A::Llama4 => Self::Llama4,
            A::Qwen2 => Self::Qwen2,
            A::Qwen3 => Self::Qwen3,
            A::Qwen3MoE => Self::Qwen3MoE,
            A::Gemma => Self::Gemma,
            A::Mistral => Self::Mistral,
            A::Phi => Self::Phi,
            A::Phi4 => Self::Phi4,
            A::DeepSeek => Self::DeepSeek,
            A::Cohere => Self::Cohere,
            A::Granite => Self::Granite,
            A::NemotronH => Self::NemotronH,
            A::Qwen3Next => Self::Qwen3Next,
            A::GptOss => Self::GptOss,
            A::Gemma4 => Self::Gemma4,
            A::Bert => Self::Bert,
            A::Flux => Self::Flux,
            A::DiffusionGemma => Self::DiffusionGemma,
            A::Mllama => Self::Mllama,
            A::Qwen4Exp => Self::Qwen4Exp,
        }
    }
}
