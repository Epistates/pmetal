//! Python wrappers for PMetal configuration types.

use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// LoRA Configuration
// ---------------------------------------------------------------------------

#[pyclass(from_py_object,name = "LoraConfig")]
#[derive(Clone)]
pub struct PyLoraConfig(pub(crate) pmetal_core::LoraConfig);

#[pymethods]
impl PyLoraConfig {
    #[new]
    #[pyo3(signature = (r=16, alpha=32.0, dropout=0.0, use_rslora=false, use_dora=false))]
    fn new(r: usize, alpha: f32, dropout: f32, use_rslora: bool, use_dora: bool) -> Self {
        Self(pmetal_core::LoraConfig {
            r,
            alpha,
            dropout,
            use_rslora,
            use_dora,
            ..Default::default()
        })
    }

    #[getter]
    fn r(&self) -> usize {
        self.0.r
    }
    #[getter]
    fn alpha(&self) -> f32 {
        self.0.alpha
    }
    #[getter]
    fn dropout(&self) -> f32 {
        self.0.dropout
    }
    #[getter]
    fn use_rslora(&self) -> bool {
        self.0.use_rslora
    }
    #[getter]
    fn use_dora(&self) -> bool {
        self.0.use_dora
    }
    #[getter]
    fn scaling(&self) -> f32 {
        self.0.scaling()
    }

    fn __repr__(&self) -> String {
        format!(
            "LoraConfig(r={}, alpha={}, dropout={}, use_rslora={}, use_dora={}, scaling={:.4})",
            self.0.r,
            self.0.alpha,
            self.0.dropout,
            self.0.use_rslora,
            self.0.use_dora,
            self.0.scaling(),
        )
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string_pretty(&self.0).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Serialization error: {e}"))
        })
    }

    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        let config: pmetal_core::LoraConfig = serde_json::from_str(json).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Deserialization error: {e}"))
        })?;
        Ok(Self(config))
    }
}

// ---------------------------------------------------------------------------
// Training Configuration
// ---------------------------------------------------------------------------

#[pyclass(from_py_object,name = "TrainingConfig")]
#[derive(Clone)]
pub struct PyTrainingConfig(pub(crate) pmetal_core::TrainingConfig);

#[pymethods]
impl PyTrainingConfig {
    #[new]
    #[pyo3(signature = (
        learning_rate=2e-4,
        batch_size=4,
        num_epochs=3,
        max_seq_len=2048,
        warmup_steps=100,
        weight_decay=0.01,
        max_grad_norm=1.0,
        use_packing=true,
        output_dir="./output",
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        learning_rate: f64,
        batch_size: usize,
        num_epochs: usize,
        max_seq_len: usize,
        warmup_steps: usize,
        weight_decay: f64,
        max_grad_norm: f64,
        use_packing: bool,
        output_dir: &str,
    ) -> Self {
        Self(pmetal_core::TrainingConfig {
            learning_rate,
            batch_size,
            num_epochs,
            max_seq_len,
            warmup_steps,
            weight_decay,
            max_grad_norm,
            use_packing,
            output_dir: output_dir.to_string(),
            ..Default::default()
        })
    }

    #[getter]
    fn learning_rate(&self) -> f64 {
        self.0.learning_rate
    }
    #[getter]
    fn batch_size(&self) -> usize {
        self.0.batch_size
    }
    #[getter]
    fn num_epochs(&self) -> usize {
        self.0.num_epochs
    }
    #[getter]
    fn max_seq_len(&self) -> usize {
        self.0.max_seq_len
    }
    #[getter]
    fn warmup_steps(&self) -> usize {
        self.0.warmup_steps
    }
    #[getter]
    fn weight_decay(&self) -> f64 {
        self.0.weight_decay
    }
    #[getter]
    fn max_grad_norm(&self) -> f64 {
        self.0.max_grad_norm
    }
    #[getter]
    fn use_packing(&self) -> bool {
        self.0.use_packing
    }
    #[getter]
    fn output_dir(&self) -> &str {
        &self.0.output_dir
    }

    fn __repr__(&self) -> String {
        format!(
            "TrainingConfig(lr={:.2e}, batch_size={}, epochs={}, max_seq_len={}, warmup={}, \
             weight_decay={}, max_grad_norm={}, packing={}, output_dir='{}')",
            self.0.learning_rate,
            self.0.batch_size,
            self.0.num_epochs,
            self.0.max_seq_len,
            self.0.warmup_steps,
            self.0.weight_decay,
            self.0.max_grad_norm,
            self.0.use_packing,
            self.0.output_dir,
        )
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string_pretty(&self.0).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Serialization error: {e}"))
        })
    }

    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        let config: pmetal_core::TrainingConfig = serde_json::from_str(json).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Deserialization error: {e}"))
        })?;
        Ok(Self(config))
    }
}

// ---------------------------------------------------------------------------
// Generation Configuration
// ---------------------------------------------------------------------------

#[pyclass(from_py_object,name = "GenerationConfig")]
#[derive(Clone)]
pub struct PyGenerationConfig(pub(crate) pmetal_models::GenerationConfig);

#[pymethods]
impl PyGenerationConfig {
    #[new]
    #[pyo3(signature = (max_tokens=256, temperature=0.7, top_k=50, top_p=0.9, min_p=0.05, seed=None))]
    fn new(
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        min_p: f32,
        seed: Option<u64>,
    ) -> Self {
        let mut config = pmetal_models::GenerationConfig::sampling(max_tokens, temperature)
            .with_top_k(top_k)
            .with_top_p(top_p)
            .with_min_p(min_p);
        if let Some(s) = seed {
            config = config.with_seed(s);
        }
        Self(config)
    }

    #[staticmethod]
    fn greedy(max_tokens: usize) -> Self {
        Self(pmetal_models::GenerationConfig::greedy(max_tokens))
    }

    #[staticmethod]
    #[pyo3(signature = (max_tokens=256, temperature=0.7))]
    fn sampling(max_tokens: usize, temperature: f32) -> Self {
        Self(pmetal_models::GenerationConfig::sampling(
            max_tokens,
            temperature,
        ))
    }

    #[getter]
    fn max_tokens(&self) -> usize {
        self.0.max_new_tokens
    }
    #[getter]
    fn temperature(&self) -> f32 {
        self.0.temperature
    }
    #[getter]
    fn top_k(&self) -> usize {
        self.0.top_k
    }
    #[getter]
    fn top_p(&self) -> f32 {
        self.0.top_p
    }
    #[getter]
    fn min_p(&self) -> f32 {
        self.0.min_p
    }
    #[getter]
    fn seed(&self) -> Option<u64> {
        self.0.seed
    }

    fn __repr__(&self) -> String {
        format!(
            "GenerationConfig(max_tokens={}, temp={}, top_k={}, top_p={}, min_p={})",
            self.0.max_new_tokens, self.0.temperature, self.0.top_k, self.0.top_p, self.0.min_p,
        )
    }
}

// ---------------------------------------------------------------------------
// DataLoader Configuration
// ---------------------------------------------------------------------------

#[pyclass(from_py_object,name = "DataLoaderConfig")]
#[derive(Clone)]
pub struct PyDataLoaderConfig(pub(crate) pmetal_data::DataLoaderConfig);

#[pymethods]
impl PyDataLoaderConfig {
    #[new]
    #[pyo3(signature = (batch_size=4, max_seq_len=2048, shuffle=true, seed=42, pad_token_id=0, drop_last=false))]
    fn new(
        batch_size: usize,
        max_seq_len: usize,
        shuffle: bool,
        seed: u64,
        pad_token_id: u32,
        drop_last: bool,
    ) -> Self {
        Self(pmetal_data::DataLoaderConfig {
            batch_size,
            max_seq_len,
            shuffle,
            seed,
            pad_token_id,
            drop_last,
        })
    }

    #[getter]
    fn batch_size(&self) -> usize {
        self.0.batch_size
    }
    #[getter]
    fn max_seq_len(&self) -> usize {
        self.0.max_seq_len
    }
    #[getter]
    fn shuffle(&self) -> bool {
        self.0.shuffle
    }
    #[getter]
    fn seed(&self) -> u64 {
        self.0.seed
    }
    #[getter]
    fn pad_token_id(&self) -> u32 {
        self.0.pad_token_id
    }
    #[getter]
    fn drop_last(&self) -> bool {
        self.0.drop_last
    }

    fn __repr__(&self) -> String {
        format!(
            "DataLoaderConfig(batch_size={}, max_seq_len={}, shuffle={}, pad_token_id={})",
            self.0.batch_size, self.0.max_seq_len, self.0.shuffle, self.0.pad_token_id,
        )
    }
}

// ---------------------------------------------------------------------------
// Enum wrappers
// ---------------------------------------------------------------------------

#[pyclass(from_py_object,name = "Dtype", eq)]
#[derive(Clone, PartialEq)]
pub enum PyDtype {
    Float32,
    Float16,
    BFloat16,
    Float8E4M3,
    Float8E5M2,
    Int32,
    Int64,
    UInt8,
    Bool,
}

impl From<PyDtype> for pmetal_core::Dtype {
    fn from(d: PyDtype) -> Self {
        match d {
            PyDtype::Float32 => Self::Float32,
            PyDtype::Float16 => Self::Float16,
            PyDtype::BFloat16 => Self::BFloat16,
            PyDtype::Float8E4M3 => Self::Float8E4M3,
            PyDtype::Float8E5M2 => Self::Float8E5M2,
            PyDtype::Int32 => Self::Int32,
            PyDtype::Int64 => Self::Int64,
            PyDtype::UInt8 => Self::UInt8,
            PyDtype::Bool => Self::Bool,
        }
    }
}

impl From<pmetal_core::Dtype> for PyDtype {
    fn from(d: pmetal_core::Dtype) -> Self {
        match d {
            pmetal_core::Dtype::Float32 => Self::Float32,
            pmetal_core::Dtype::Float16 => Self::Float16,
            pmetal_core::Dtype::BFloat16 => Self::BFloat16,
            pmetal_core::Dtype::Float8E4M3 => Self::Float8E4M3,
            pmetal_core::Dtype::Float8E5M2 => Self::Float8E5M2,
            pmetal_core::Dtype::Int32 => Self::Int32,
            pmetal_core::Dtype::Int64 => Self::Int64,
            pmetal_core::Dtype::UInt8 => Self::UInt8,
            pmetal_core::Dtype::Bool => Self::Bool,
        }
    }
}

#[pyclass(from_py_object,name = "Quantization", eq)]
#[derive(Clone, PartialEq)]
pub enum PyQuantization {
    None,
    NF4,
    FP4,
    Int8,
    FP8,
}

impl From<PyQuantization> for pmetal_core::Quantization {
    fn from(q: PyQuantization) -> Self {
        match q {
            PyQuantization::None => Self::None,
            PyQuantization::NF4 => Self::NF4,
            PyQuantization::FP4 => Self::FP4,
            PyQuantization::Int8 => Self::Int8,
            PyQuantization::FP8 => Self::FP8,
        }
    }
}

#[pyclass(from_py_object,name = "LoraBias", eq)]
#[derive(Clone, PartialEq)]
pub enum PyLoraBias {
    None,
    All,
    LoraOnly,
}

#[pyclass(from_py_object,name = "LrSchedulerType", eq)]
#[derive(Clone, PartialEq)]
pub enum PyLrSchedulerType {
    Constant,
    Linear,
    Cosine,
    CosineWithRestarts,
    Polynomial,
}

#[pyclass(from_py_object,name = "OptimizerType", eq)]
#[derive(Clone, PartialEq)]
pub enum PyOptimizerType {
    AdamW,
    Sgd,
    Adafactor,
    Lion,
}

#[pyclass(from_py_object,name = "DatasetFormat", eq)]
#[derive(Clone, PartialEq)]
pub enum PyDatasetFormat {
    Simple,
    Alpaca,
    ShareGpt,
    OpenAi,
    Auto,
}

impl From<PyDatasetFormat> for pmetal_data::DatasetFormat {
    fn from(f: PyDatasetFormat) -> Self {
        match f {
            PyDatasetFormat::Simple => Self::Simple,
            PyDatasetFormat::Alpaca => Self::Alpaca,
            PyDatasetFormat::ShareGpt => Self::ShareGpt,
            PyDatasetFormat::OpenAi => Self::OpenAi,
            PyDatasetFormat::Auto => Self::Auto,
        }
    }
}

#[pyclass(from_py_object,name = "ModelArchitecture", eq)]
#[derive(Clone, PartialEq)]
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
    StarCoder2,
    RecurrentGemma,
    Jamba,
    Flux,
}

impl From<pmetal_models::ModelArchitecture> for PyModelArchitecture {
    fn from(a: pmetal_models::ModelArchitecture) -> Self {
        match a {
            pmetal_models::ModelArchitecture::Llama => Self::Llama,
            pmetal_models::ModelArchitecture::Llama4 => Self::Llama4,
            pmetal_models::ModelArchitecture::Qwen2 => Self::Qwen2,
            pmetal_models::ModelArchitecture::Qwen3 => Self::Qwen3,
            pmetal_models::ModelArchitecture::Qwen3MoE => Self::Qwen3MoE,
            pmetal_models::ModelArchitecture::Gemma => Self::Gemma,
            pmetal_models::ModelArchitecture::Mistral => Self::Mistral,
            pmetal_models::ModelArchitecture::Phi => Self::Phi,
            pmetal_models::ModelArchitecture::Phi4 => Self::Phi4,
            pmetal_models::ModelArchitecture::DeepSeek => Self::DeepSeek,
            pmetal_models::ModelArchitecture::Cohere => Self::Cohere,
            pmetal_models::ModelArchitecture::Granite => Self::Granite,
            pmetal_models::ModelArchitecture::NemotronH => Self::NemotronH,
            pmetal_models::ModelArchitecture::Qwen3Next => Self::Qwen3Next,
            pmetal_models::ModelArchitecture::StarCoder2 => Self::StarCoder2,
            pmetal_models::ModelArchitecture::RecurrentGemma => Self::RecurrentGemma,
            pmetal_models::ModelArchitecture::Jamba => Self::Jamba,
            pmetal_models::ModelArchitecture::Flux => Self::Flux,
        }
    }
}
