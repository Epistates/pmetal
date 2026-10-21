//! Python objects to and from `serde_json::Value`.
//!
//! Specs cross the boundary as JSON values: a Python value becomes the value
//! of one spec field, and serde (the spec's own `Deserialize`) decides whether
//! the spec accepts it. Nothing here knows a field's name or type.

use pyo3::exceptions::{PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
use serde_json::{Map, Number, Value};

/// The JSON value of a serializable `T`.
///
/// Through text, not `serde_json::to_value`: `to_value` widens an `f32` to the
/// `f64` nearest it (`2e-5` becomes `1.9999999494757503e-05`), where the text
/// of an `f32` is its shortest decimal (`2e-05`), the number the user wrote.
pub fn value_of<T: serde::Serialize>(value: &T) -> Result<Value, serde_json::Error> {
    serde_json::from_slice(&serde_json::to_vec(value)?)
}

/// A JSON value for a Python object.
///
/// Accepts `None`, `bool`, `int`, `float`, `str`, `list`, `tuple`, `dict` with
/// `str` keys, and `os.PathLike` (as its `os.fspath`). Anything else is a
/// `TypeError`; a NaN or infinity a `ValueError`, since JSON has neither.
pub fn to_value(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if obj.is_none() {
        return Ok(Value::Null);
    }
    // `bool` is an `int`, so it goes first.
    if obj.is_instance_of::<PyBool>() {
        return Ok(Value::Bool(obj.extract::<bool>()?));
    }
    if obj.is_instance_of::<PyInt>() {
        return obj
            .extract::<i64>()
            .map(Value::from)
            .or_else(|_| obj.extract::<u64>().map(Value::from))
            .map_err(|_| PyOverflowError::new_err(format!("{obj} does not fit a 64-bit integer")));
    }
    if obj.is_instance_of::<PyFloat>() {
        let f = obj.extract::<f64>()?;
        return Number::from_f64(f)
            .map(Value::Number)
            .ok_or_else(|| PyValueError::new_err(format!("{f} cannot be represented")));
    }
    if obj.is_instance_of::<PyString>() {
        return Ok(Value::String(obj.extract::<String>()?));
    }
    if let Ok(list) = obj.cast::<PyList>() {
        return list.iter().map(|item| to_value(&item)).collect();
    }
    if let Ok(tuple) = obj.cast::<PyTuple>() {
        return tuple.iter().map(|item| to_value(&item)).collect();
    }
    if let Ok(dict) = obj.cast::<PyDict>() {
        let mut map = Map::with_capacity(dict.len());
        for (key, value) in dict.iter() {
            let key = key.extract::<String>().map_err(|_| {
                PyTypeError::new_err(format!("dict keys must be str, got {}", type_name(&key)))
            })?;
            map.insert(key, to_value(&value)?);
        }
        return Ok(Value::Object(map));
    }
    if obj.hasattr("__fspath__")? {
        let path = obj.py().import("os")?.call_method1("fspath", (obj,))?;
        return to_value(&path);
    }
    Err(PyTypeError::new_err(format!(
        "cannot use a {} here",
        type_name(obj)
    )))
}

fn type_name(obj: &Bound<'_, PyAny>) -> String {
    obj.get_type()
        .name()
        .map_or_else(|_| "object".to_string(), |n| n.to_string())
}

/// The Python object for a JSON value.
pub fn to_py<'py>(py: Python<'py>, value: &Value) -> PyResult<Bound<'py, PyAny>> {
    Ok(match value {
        Value::Null => py.None().into_bound(py),
        Value::Bool(b) => PyBool::new(py, *b).to_owned().into_any(),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.into_pyobject(py)?.into_any()
            } else if let Some(u) = n.as_u64() {
                u.into_pyobject(py)?.into_any()
            } else {
                n.as_f64().unwrap_or(f64::NAN).into_pyobject(py)?.into_any()
            }
        }
        Value::String(s) => PyString::new(py, s).into_any(),
        Value::Array(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(to_py(py, item)?)?;
            }
            list.into_any()
        }
        Value::Object(map) => {
            let dict = PyDict::new(py);
            for (key, item) in map {
                dict.set_item(key, to_py(py, item)?)?;
            }
            dict.into_any()
        }
    })
}
