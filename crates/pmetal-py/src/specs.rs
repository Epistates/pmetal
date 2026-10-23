//! One Python class per `pmetal` subcommand's job spec.
//!
//! The classes are all the same wrapper (`spec::py_spec!`) over a different
//! `pmetal_core::jobs` type; this file only names them.

use pmetal_core::jobs::{
    BenchSpec, DecideSpec, DflashSpec, DistillSpec, EmbedTrainSpec, EvalSpec, FuseSpec, GrpoSpec,
    ImatrixSpec, InferSpec, MergeSpec, PackExpertsSpec, PreferenceSpec, PretrainSpec, QuantizeSpec,
    RlkdSpec, ServeSpec, TokenizeSpec, TrainDiffusionSpec, TrainDraftSpec, TrainMtpSpec, TrainSpec,
    VocodeSpec,
};
use pyo3::prelude::*;

use crate::spec::py_spec;

py_spec!(
    PyTrainSpec,
    "TrainSpec",
    TrainSpec,
    "Supervised fine-tuning: LoRA, QLoRA or the Neural Engine (`pmetal train`).",
    {
        /// The adapter configuration the spec asks for (`LoraConfig`), as a dict.
        fn lora_config<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
            crate::trainer::config_dict(py, &self.inner.lora_config())
        }

        /// The optimisation settings the spec asks for (`TrainingConfig`), as a dict.
        /// Raises `ConfigError` for an optimizer or schedule name the trainer does not know.
        fn training_config<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
            let config = self
                .inner
                .training_config()
                .map_err(crate::error::config_err)?;
            crate::trainer::config_dict(py, &config)
        }
    }
);
py_spec!(
    PyInferSpec,
    "InferSpec",
    InferSpec,
    "Text generation (`pmetal infer`)."
);
py_spec!(
    PyServeSpec,
    "ServeSpec",
    ServeSpec,
    "An OpenAI-compatible inference server (`pmetal serve`)."
);
py_spec!(
    PyGrpoSpec,
    "GrpoSpec",
    GrpoSpec,
    "GRPO reinforcement learning (`pmetal grpo`)."
);
py_spec!(
    PyPreferenceSpec,
    "PreferenceSpec",
    PreferenceSpec,
    "Preference optimization: DPO, SimPO, ORPO, KTO (`pmetal preference`)."
);
py_spec!(
    PyDistillSpec,
    "DistillSpec",
    DistillSpec,
    "Knowledge distillation (`pmetal distill`)."
);
py_spec!(
    PyRlkdSpec,
    "RlkdSpec",
    RlkdSpec,
    "On-policy distillation (`pmetal rlkd`)."
);
py_spec!(
    PyEmbedTrainSpec,
    "EmbedTrainSpec",
    EmbedTrainSpec,
    "Embedding-model training (`pmetal embed-train`)."
);
py_spec!(
    PyPretrainSpec,
    "PretrainSpec",
    PretrainSpec,
    "Pretraining from scratch (`pmetal pretrain`)."
);
py_spec!(
    PyQuantizeSpec,
    "QuantizeSpec",
    QuantizeSpec,
    "Model quantization (`pmetal quantize`)."
);
py_spec!(
    PyImatrixSpec,
    "ImatrixSpec",
    ImatrixSpec,
    "Importance-matrix measurement on calibration data (`pmetal imatrix`)."
);
py_spec!(
    PyMergeSpec,
    "MergeSpec",
    MergeSpec,
    "Model merging (`pmetal merge`)."
);
py_spec!(
    PyFuseSpec,
    "FuseSpec",
    FuseSpec,
    "Fusing a LoRA adapter into its base model (`pmetal fuse`)."
);
py_spec!(
    PyEvalSpec,
    "EvalSpec",
    EvalSpec,
    "Evaluation (`pmetal eval`)."
);
py_spec!(
    PyBenchSpec,
    "BenchSpec",
    BenchSpec,
    "Training benchmarks (`pmetal bench`)."
);
py_spec!(
    PyDflashSpec,
    "DflashSpec",
    DflashSpec,
    "DFlash block-diffusion speculative decoding (`pmetal dflash`)."
);
py_spec!(
    PyTrainMtpSpec,
    "TrainMtpSpec",
    TrainMtpSpec,
    "Training an MTP predictor or Gemma 4 assistant (`pmetal train-mtp`)."
);
py_spec!(
    PyTrainDraftSpec,
    "TrainDraftSpec",
    TrainDraftSpec,
    "Training a DFlash draft model (`pmetal train-draft`)."
);
py_spec!(
    PyTrainDiffusionSpec,
    "TrainDiffusionSpec",
    TrainDiffusionSpec,
    "Fine-tuning a block-diffusion model (`pmetal train-diffusion`)."
);
py_spec!(
    PyDecideSpec,
    "DecideSpec",
    DecideSpec,
    "Typed decisions with a decision model (`pmetal decide`)."
);
py_spec!(
    PyTokenizeSpec,
    "TokenizeSpec",
    TokenizeSpec,
    "Tokenizing a dataset into shards (`pmetal tokenize`)."
);
py_spec!(
    PyVocodeSpec,
    "VocodeSpec",
    VocodeSpec,
    "Neural vocoding (`pmetal vocode`)."
);
py_spec!(
    PyPackExpertsSpec,
    "PackExpertsSpec",
    PackExpertsSpec,
    "Packing MoE experts for SSD offload (`pmetal pack-experts`)."
);

/// Add every spec class to the module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<crate::spec::PyFieldInfo>()?;
    m.add_class::<PyTrainSpec>()?;
    m.add_class::<PyInferSpec>()?;
    m.add_class::<PyServeSpec>()?;
    m.add_class::<PyGrpoSpec>()?;
    m.add_class::<PyPreferenceSpec>()?;
    m.add_class::<PyDistillSpec>()?;
    m.add_class::<PyRlkdSpec>()?;
    m.add_class::<PyEmbedTrainSpec>()?;
    m.add_class::<PyPretrainSpec>()?;
    m.add_class::<PyQuantizeSpec>()?;
    m.add_class::<PyImatrixSpec>()?;
    m.add_class::<PyMergeSpec>()?;
    m.add_class::<PyFuseSpec>()?;
    m.add_class::<PyEvalSpec>()?;
    m.add_class::<PyBenchSpec>()?;
    m.add_class::<PyDflashSpec>()?;
    m.add_class::<PyTrainMtpSpec>()?;
    m.add_class::<PyTrainDraftSpec>()?;
    m.add_class::<PyTrainDiffusionSpec>()?;
    m.add_class::<PyDecideSpec>()?;
    m.add_class::<PyTokenizeSpec>()?;
    m.add_class::<PyVocodeSpec>()?;
    m.add_class::<PyPackExpertsSpec>()?;
    Ok(())
}
