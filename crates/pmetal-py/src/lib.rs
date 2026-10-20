//! Python bindings for PMetal.
//!
//! The `pmetal.pmetal` extension module, re-exported by the `pmetal` package
//! (see `py_src/pmetal`). Its parts:
//!
//! - `error`: the exception hierarchy, and Rust errors mapped onto it.
//! - `spec`, `specs`: one class per `pmetal` subcommand, wrapping the job
//!   specs of `pmetal-core`, which define every field, default and check.
//! - `trainer`, `callbacks`: a `TrainSpec` run in this process.
//! - `model`, `dflash`, `infer`, `tokenizer`: inference.
//! - `hub`: the Hugging Face Hub.
//! - `runtime`: the shared async runtime, and how a Python thread waits on
//!   work with the GIL released and Ctrl-C honoured.

use pyo3::prelude::*;
use pyo3::types::PyTuple;

mod callbacks;
mod config;
mod dflash;
mod error;
mod hub;
mod infer;
mod json;
mod metallib;
mod model;
mod runtime;
mod spec;
mod specs;
mod tokenizer;
mod trainer;

/// PMetal: fine-tuning and inference for Apple Silicon.
///
/// `gil_used = false` declares the module safe without the GIL: its state is
/// either immutable, guarded by PyO3's per-object borrow checking, or owned
/// by a thread of its own (`Model`, `DFlashGenerator`).
#[pymodule(gil_used = false)]
fn pmetal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    m.add_function(wrap_pyfunction!(metallib::configure_metallib, m)?)?;
    error::register(m)?;
    specs::register(m)?;

    // Training
    m.add_class::<trainer::PyTrainer>()?;
    m.add_class::<trainer::PyTrainResult>()?;
    m.add_class::<callbacks::PyProgressCallback>()?;
    m.add_class::<callbacks::PyLoggingCallback>()?;
    m.add_class::<callbacks::PyMetricsJsonCallback>()?;

    // Inference
    m.add_class::<model::PyModel>()?;
    m.add_class::<model::PyTextStream>()?;
    m.add_class::<dflash::PyDFlashGenerator>()?;
    m.add_class::<tokenizer::PyTokenizer>()?;
    m.add_class::<config::PyGenerationConfig>()?;
    m.add_class::<config::PyModelArchitecture>()?;
    m.add_class::<infer::PyInferResult>()?;
    m.add(
        "INFER_COMMAND_LINE_ONLY",
        PyTuple::new(m.py(), infer::COMMAND_LINE_ONLY)?,
    )?;
    m.add_function(wrap_pyfunction!(infer::infer, m)?)?;
    m.add_function(wrap_pyfunction!(infer::infer_with_metrics, m)?)?;

    // Hub
    m.add_class::<hub::PyDownloadPlan>()?;
    m.add_class::<hub::PyModelReport>()?;
    m.add_class::<hub::PyUploadReport>()?;
    m.add_function(wrap_pyfunction!(hub::download_model, m)?)?;
    m.add_function(wrap_pyfunction!(hub::download_file, m)?)?;
    m.add_function(wrap_pyfunction!(hub::plan_download, m)?)?;
    m.add_function(wrap_pyfunction!(hub::verify_model, m)?)?;
    m.add_function(wrap_pyfunction!(hub::search_models, m)?)?;
    m.add_function(wrap_pyfunction!(hub::estimate_fit, m)?)?;
    m.add_function(wrap_pyfunction!(hub::upload_model, m)?)?;

    Ok(())
}
