//! Python bindings for PMetal.
//!
//! This crate provides a `cdylib` Python extension module via PyO3,
//! exposing PMetal's training, inference, and model management APIs
//! to Python.

use pyo3::prelude::*;

mod array_bridge;
mod callbacks;
mod config;
mod easy;
pub(crate) mod error;
pub(crate) mod hub;
mod model;
mod tokenizer;
mod trainer;

/// PMetal Python module.
#[pymodule]
fn pmetal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Version
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // Config types
    m.add_class::<config::PyLoraConfig>()?;
    m.add_class::<config::PyTrainingConfig>()?;
    m.add_class::<config::PyGenerationConfig>()?;
    m.add_class::<config::PyDataLoaderConfig>()?;

    // Enums
    m.add_class::<config::PyDtype>()?;
    m.add_class::<config::PyQuantization>()?;
    m.add_class::<config::PyLoraBias>()?;
    m.add_class::<config::PyLrSchedulerType>()?;
    m.add_class::<config::PyOptimizerType>()?;
    m.add_class::<config::PyDatasetFormat>()?;
    m.add_class::<config::PyModelArchitecture>()?;

    // Hub
    m.add_function(wrap_pyfunction!(hub::download_model, m)?)?;
    m.add_function(wrap_pyfunction!(hub::download_file, m)?)?;

    // Model
    m.add_class::<model::PyModel>()?;

    // Tokenizer
    m.add_class::<tokenizer::PyTokenizer>()?;

    // Trainer
    m.add_class::<trainer::PyTrainer>()?;

    // Callbacks
    m.add_class::<callbacks::PyProgressCallback>()?;
    m.add_class::<callbacks::PyLoggingCallback>()?;
    m.add_class::<callbacks::PyMetricsJsonCallback>()?;

    // Easy API
    m.add_function(wrap_pyfunction!(easy::finetune, m)?)?;
    m.add_function(wrap_pyfunction!(easy::infer, m)?)?;

    Ok(())
}
