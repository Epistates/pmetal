//! The job specs, as Python classes.
//!
//! `pmetal-core`'s [`jobs`](pmetal_core::jobs) are the one definition of what
//! each job takes: its fields, their defaults, their bounds, how they are
//! checked (`normalize`) and how they become a command line (`to_argv`). The
//! classes here wrap them without restating any of that. A class has no field
//! list of its own: the keyword arguments, attributes, defaults and validation
//! all come from the spec's serde representation and its field descriptors, so
//! a field added to a spec reaches Python with no change in this crate (the
//! type stubs are generated from the same data, see `stub.py`).

use pmetal_core::{FieldDescriptor, FieldError, FieldKind, JobFields};
use pyo3::exceptions::{PyAttributeError, PyKeyError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Map, Value};

use crate::error::{config_err, validation_err};
use crate::json::{to_py, to_value, value_of};

/// What the Python class needs of a job spec.
pub trait SpecType:
    JobFields + Serialize + DeserializeOwned + Default + Clone + Send + Sync + 'static
{
    /// The class name.
    const NAME: &'static str;
    /// The spec's own `normalize`.
    fn normalize_spec(&mut self) -> Result<(), Vec<FieldError>>;
}

/// The spec's fields as a JSON object.
fn object_of<S: SpecType>(spec: &S) -> PyResult<Map<String, Value>> {
    match value_of(spec) {
        Ok(Value::Object(map)) => Ok(map),
        Ok(other) => Err(config_err(format!(
            "{} serialized to {other}, not an object",
            S::NAME
        ))),
        Err(e) => Err(config_err(format!("{}: {e}", S::NAME))),
    }
}

fn from_object<S: SpecType>(map: Map<String, Value>) -> Result<S, serde_json::Error> {
    serde_json::from_value(Value::Object(map))
}

/// The edit distance between two names, for "did you mean".
fn distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut diagonal = row[0];
        row[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let above = row[j + 1];
            row[j + 1] = (diagonal + usize::from(ca != *cb))
                .min(row[j] + 1)
                .min(above + 1);
            diagonal = above;
        }
    }
    row[b.len()]
}

fn suggestion<'a>(name: &str, known: impl Iterator<Item = &'a String>) -> String {
    known
        .map(|k| (distance(name, k), k))
        .filter(|(d, k)| *d <= 2.max(k.len() / 4))
        .min_by_key(|(d, _)| *d)
        .map_or_else(String::new, |(_, k)| format!(" Did you mean '{k}'?"))
}

/// Set `updates` over the defaults (or over `base`) and read the result back as a spec.
fn apply<S: SpecType>(
    mut object: Map<String, Value>,
    updates: Vec<(String, Value)>,
    unknown: impl Fn(&str, String) -> PyErr,
) -> PyResult<S> {
    for (key, value) in &updates {
        if !object.contains_key(key) {
            return Err(unknown(key, suggestion(key, object.keys())));
        }
        object.insert(key.clone(), value.clone());
    }
    from_object::<S>(object).map_err(|whole| {
        // serde names no field; find the one that does not take its value.
        let culprit = updates.iter().find_map(|(key, value)| {
            let mut one = object_of(&S::default()).ok()?;
            one.insert(key.clone(), value.clone());
            from_object::<S>(one)
                .err()
                .map(|e| format!("{}.{key}: {e}", S::NAME))
        });
        PyTypeError::new_err(culprit.unwrap_or_else(|| format!("{}: {whole}", S::NAME)))
    })
}

fn updates_of(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Vec<(String, Value)>> {
    let Some(kwargs) = kwargs else {
        return Ok(Vec::new());
    };
    kwargs
        .iter()
        .map(|(key, value)| Ok((key.extract::<String>()?, to_value(&value)?)))
        .collect()
}

pub fn new<S: SpecType>(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<S> {
    apply::<S>(
        object_of(&S::default())?,
        updates_of(kwargs)?,
        |key, hint| {
            PyTypeError::new_err(format!(
                "{}() got an unexpected keyword argument '{key}'.{hint}",
                S::NAME
            ))
        },
    )
}

pub fn replaced<S: SpecType>(spec: &S, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<S> {
    apply::<S>(object_of(spec)?, updates_of(kwargs)?, |key, hint| {
        PyTypeError::new_err(format!(
            "{}.copy() got an unexpected keyword argument '{key}'.{hint}",
            S::NAME
        ))
    })
}

pub fn get_field<'py, S: SpecType>(
    py: Python<'py>,
    spec: &S,
    name: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let object = object_of(spec)?;
    match object.get(name) {
        Some(value) => to_py(py, value),
        None => Err(PyAttributeError::new_err(format!(
            "'{}' object has no attribute '{name}'.{}",
            S::NAME,
            suggestion(name, object.keys())
        ))),
    }
}

pub fn set_field<S: SpecType>(
    spec: &mut S,
    name: &str,
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    let Some(value) = value else {
        return Err(PyAttributeError::new_err(format!(
            "cannot delete '{name}': a spec field always has a value (set it to None or its default)"
        )));
    };
    let updates = vec![(name.to_string(), to_value(value)?)];
    *spec = apply::<S>(object_of(spec)?, updates, |key, hint| {
        PyAttributeError::new_err(format!("'{}' object has no field '{key}'.{hint}", S::NAME))
    })?;
    Ok(())
}

pub fn normalize<S: SpecType>(spec: &mut S) -> PyResult<()> {
    spec.normalize_spec()
        .map_err(|errors| validation_err(S::NAME, &errors))
}

pub fn validation_errors<S: SpecType>(spec: &S) -> Vec<(String, String)> {
    let mut copy = spec.clone();
    copy.normalize_spec()
        .err()
        .unwrap_or_default()
        .into_iter()
        .map(|e| (e.field, e.message))
        .collect()
}

pub fn to_dict<'py, S: SpecType>(
    py: Python<'py>,
    spec: &S,
    exclude_defaults: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let defaults = object_of(&S::default())?;
    let dict = PyDict::new(py);
    for (key, value) in object_of(spec)? {
        if exclude_defaults && defaults.get(&key) == Some(&value) {
            continue;
        }
        dict.set_item(key, to_py(py, &value)?)?;
    }
    Ok(dict)
}

pub fn from_dict<S: SpecType>(data: &Bound<'_, PyDict>) -> PyResult<S> {
    new::<S>(Some(data))
}

pub fn to_json<S: SpecType>(spec: &S) -> PyResult<String> {
    serde_json::to_string_pretty(spec).map_err(config_err)
}

pub fn from_json<S: SpecType>(json: &str) -> PyResult<S> {
    let value: Value = serde_json::from_str(json).map_err(config_err)?;
    let Value::Object(map) = value else {
        return Err(PyTypeError::new_err(format!(
            "{}.from_json needs a JSON object",
            S::NAME
        )));
    };
    apply::<S>(
        object_of(&S::default())?,
        map.into_iter().collect(),
        |key, hint| PyKeyError::new_err(format!("{} has no field '{key}'.{hint}", S::NAME)),
    )
}

pub fn repr<S: SpecType>(py: Python<'_>, spec: &S) -> PyResult<String> {
    let defaults = object_of(&S::default())?;
    let mut parts = Vec::new();
    for (key, value) in object_of(spec)? {
        if defaults.get(&key) == Some(&value) {
            continue;
        }
        parts.push(format!("{key}={}", to_py(py, &value)?.repr()?));
    }
    Ok(format!("{}({})", S::NAME, parts.join(", ")))
}

pub fn dir<S: SpecType>(spec: &S) -> PyResult<Vec<String>> {
    Ok(object_of(spec)?.keys().cloned().collect())
}

pub fn equal<S: SpecType>(a: &S, b: &S) -> PyResult<bool> {
    Ok(object_of(a)? == object_of(b)?)
}

// ---------------------------------------------------------------------------
// Field metadata
// ---------------------------------------------------------------------------

/// One field of a spec: what a form, a help page or a type stub needs.
#[pyclass(frozen, name = "FieldInfo", module = "pmetal", skip_from_py_object)]
pub struct PyFieldInfo {
    /// The attribute and keyword name.
    #[pyo3(get)]
    name: String,
    /// A short human label.
    #[pyo3(get)]
    label: String,
    /// What the field does, when the spec says.
    #[pyo3(get)]
    help: Option<String>,
    /// The form section it belongs to.
    #[pyo3(get)]
    group: String,
    /// `text`, `number`, `integer`, `enum`, `toggle`, `path`, `model`, `dataset` or `readonly`.
    #[pyo3(get)]
    kind: String,
    /// The Python type, as an annotation (`"float"`, `"str | None"`, `"Literal['a', 'b']"`).
    #[pyo3(get)]
    annotation: String,
    /// The value the field has when it is not given.
    #[pyo3(get)]
    default: Py<PyAny>,
    /// Whether `normalize()` rejects the field when it is empty.
    #[pyo3(get)]
    required: bool,
    /// The `pmetal` command-line flag the field becomes, when it becomes one.
    #[pyo3(get)]
    flag: Option<String>,
    /// For an enum field, its allowed values.
    #[pyo3(get)]
    options: Option<Vec<String>>,
    /// The lowest value a number field takes.
    #[pyo3(get)]
    minimum: Option<f64>,
    /// The highest value a number field takes.
    #[pyo3(get)]
    maximum: Option<f64>,
}

#[pymethods]
impl PyFieldInfo {
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "FieldInfo(name={:?}, annotation={:?}, default={})",
            self.name,
            self.annotation,
            self.default.bind(py).repr()?
        ))
    }
}

fn kind_name(kind: &FieldKind) -> &'static str {
    match kind {
        FieldKind::Text => "text",
        FieldKind::Number { .. } => "number",
        FieldKind::Integer { .. } => "integer",
        FieldKind::Enum { .. } => "enum",
        FieldKind::Toggle => "toggle",
        FieldKind::ModelPicker => "model",
        FieldKind::DatasetPicker => "dataset",
        FieldKind::ReadOnly => "readonly",
        FieldKind::Path => "path",
    }
}

fn literal(options: &[&str]) -> String {
    let quoted: Vec<String> = options
        .iter()
        .map(|o| format!("'{}'", o.replace('\\', "\\\\").replace('\'', "\\'")))
        .collect();
    format!("Literal[{}]", quoted.join(", "))
}

/// The annotation of a field: what its default is, or for an empty (`None`)
/// default, what its descriptor says it holds.
fn annotation(default: &Value, descriptor: Option<&FieldDescriptor>) -> String {
    let kind = descriptor.map(|d| &d.kind);
    let base = match default {
        Value::Bool(_) => "bool".to_string(),
        Value::Number(n) if n.is_f64() => "float".to_string(),
        Value::Number(_) => "int".to_string(),
        Value::String(_) | Value::Null | Value::Object(_) => match kind {
            Some(FieldKind::Enum { options }) => literal(options),
            Some(FieldKind::Number { .. }) => "float".to_string(),
            Some(FieldKind::Integer { .. }) => "int".to_string(),
            Some(FieldKind::Toggle) => "bool".to_string(),
            _ => "str".to_string(),
        },
        Value::Array(_) => "list[str]".to_string(),
    };
    if default.is_null() {
        format!("{base} | None")
    } else {
        base
    }
}

pub fn field_infos<S: SpecType>(py: Python<'_>) -> PyResult<Vec<PyFieldInfo>> {
    let defaults = object_of(&S::default())?;
    let descriptors = S::field_descriptors();
    let mut infos = Vec::with_capacity(defaults.len());
    for (name, default) in &defaults {
        let descriptor = descriptors.iter().find(|d| d.name == name);
        let (options, minimum, maximum) = match descriptor.map(|d| &d.kind) {
            Some(FieldKind::Enum { options }) => (
                Some(options.iter().map(|o| (*o).to_string()).collect()),
                None,
                None,
            ),
            Some(FieldKind::Number { min, max }) => (None, Some(*min), Some(*max)),
            #[allow(clippy::cast_precision_loss)]
            Some(FieldKind::Integer { min, max }) => (None, Some(*min as f64), Some(*max as f64)),
            _ => (None, None, None),
        };
        infos.push(PyFieldInfo {
            name: name.clone(),
            label: descriptor.map_or_else(|| name.clone(), |d| d.label.to_string()),
            help: descriptor.and_then(|d| d.help).map(str::to_string),
            group: descriptor.map_or("Advanced", |d| d.group).to_string(),
            kind: descriptor
                .map_or("text", |d| kind_name(&d.kind))
                .to_string(),
            annotation: annotation(default, descriptor),
            default: to_py(py, default)?.unbind(),
            required: descriptor.is_some_and(|d| d.required),
            flag: descriptor.and_then(|d| d.argv).map(str::to_string),
            options,
            minimum,
            maximum,
        });
    }
    Ok(infos)
}

pub fn fields_list<'py, S: SpecType>(py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
    let infos = field_infos::<S>(py)?
        .into_iter()
        .map(|info| Py::new(py, info))
        .collect::<PyResult<Vec<_>>>()?;
    PyList::new(py, infos)
}

pub fn argv<S: SpecType>(spec: &S) -> Vec<String> {
    spec.to_argv()
}

pub fn reduce<'py, S: SpecType>(
    py: Python<'py>,
    class: &Bound<'py, PyAny>,
    spec: &S,
) -> PyResult<Bound<'py, PyTuple>> {
    let from_dict = class.getattr("from_dict")?;
    PyTuple::new(
        py,
        [
            from_dict,
            PyTuple::new(py, [to_dict(py, spec, false)?])?.into_any(),
        ],
    )
}

// ---------------------------------------------------------------------------
// The classes
// ---------------------------------------------------------------------------

/// Declare the Python class of a job spec.
macro_rules! py_spec {
    ($py:ident, $name:literal, $spec:ty, $doc:literal $(, { $($extra:tt)* })?) => {
        #[doc = $doc]
        #[pyclass(name = $name, module = "pmetal", skip_from_py_object)]
        #[derive(Clone)]
        pub struct $py {
            pub(crate) inner: $spec,
        }

        impl $crate::spec::SpecType for $spec {
            const NAME: &'static str = $name;
            fn normalize_spec(&mut self) -> Result<(), Vec<::pmetal_core::FieldError>> {
                self.normalize()
            }
        }

        const _: () = {
        use $crate::spec::{
            argv, dir, equal, fields_list, from_dict, from_json, get_field, new, normalize, reduce,
            repr, replaced, set_field, to_dict, to_json, validation_errors,
        };
        use ::pyo3::prelude::*;
        use ::pyo3::types::{PyDict, PyList, PyTuple, PyType};

        #[pymethods]
        impl $py {
            #[new]
            #[pyo3(signature = (**kwargs))]
            fn new(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
                Ok(Self { inner: new::<$spec>(kwargs)? })
            }

            /// The `pmetal` subcommand this spec runs.
            #[classattr]
            fn subcommand() -> &'static str {
                <$spec as ::pmetal_core::JobFields>::subcommand()
            }

            /// Every field with its type, default, bounds and help.
            #[staticmethod]
            fn fields(py: Python<'_>) -> PyResult<Bound<'_, PyList>> {
                fields_list::<$spec>(py)
            }

            fn __getattr__<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
                get_field(py, &self.inner, name)
            }

            fn __setattr__(&mut self, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
                set_field(&mut self.inner, name, Some(value))
            }

            fn __delattr__(&mut self, name: &str) -> PyResult<()> {
                set_field(&mut self.inner, name, None)
            }

            fn __dir__(&self) -> PyResult<Vec<String>> {
                let mut names = dir(&self.inner)?;
                names.extend(
                    ["fields", "normalize", "to_argv", "to_dict", "from_dict", "to_json",
                     "from_json", "copy", "validation_errors", "subcommand"]
                        .map(String::from),
                );
                Ok(names)
            }

            fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
                repr(py, &self.inner)
            }

            fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
                match other.extract::<PyRef<'_, Self>>() {
                    Ok(other) => equal(&self.inner, &other.inner),
                    Err(_) => Ok(false),
                }
            }

            #[classattr]
            const __hash__: Option<Py<PyAny>> = None;

            fn __copy__(&self) -> Self {
                self.clone()
            }

            fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
                self.clone()
            }

            fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
                reduce(slf.py(), &slf.get_type().into_any(), &slf.borrow().inner)
            }

            /// Check the spec; raise `SpecValidationError` listing every problem.
            /// Fills in whatever the spec derives from other fields.
            fn normalize(&mut self) -> PyResult<()> {
                normalize(&mut self.inner)
            }

            /// Every problem `normalize()` would raise, as `(field, message)`; empty when valid.
            fn validation_errors(&self) -> Vec<(String, String)> {
                validation_errors(&self.inner)
            }

            /// The arguments that run this spec: `pmetal <subcommand> <to_argv()>`.
            fn to_argv(&self) -> Vec<String> {
                argv(&self.inner)
            }

            /// The fields as a dict; with `exclude_defaults`, only those that differ from the defaults.
            #[pyo3(signature = (*, exclude_defaults=false))]
            fn to_dict<'py>(&self, py: Python<'py>, exclude_defaults: bool) -> PyResult<Bound<'py, PyDict>> {
                to_dict(py, &self.inner, exclude_defaults)
            }

            /// A spec from a dict of fields.
            #[classmethod]
            fn from_dict(_cls: &Bound<'_, PyType>, data: &Bound<'_, PyDict>) -> PyResult<Self> {
                Ok(Self { inner: from_dict::<$spec>(data)? })
            }

            /// The fields as JSON.
            fn to_json(&self) -> PyResult<String> {
                to_json(&self.inner)
            }

            /// A spec from JSON text.
            #[classmethod]
            fn from_json(_cls: &Bound<'_, PyType>, json: &str) -> PyResult<Self> {
                Ok(Self { inner: from_json::<$spec>(json)? })
            }

            /// A copy with the given fields replaced.
            #[pyo3(signature = (**changes))]
            fn copy(&self, changes: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
                Ok(Self { inner: replaced(&self.inner, changes)? })
            }

            $($($extra)*)?
        }
        };
    };
}

pub(crate) use py_spec;

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_core::jobs::TrainSpec;
    use serde_json::json;

    #[test]
    fn edit_distance_is_the_usual_one() {
        assert_eq!(distance("epochs", "epochs"), 0);
        assert_eq!(distance("epochz", "epochs"), 1);
        assert_eq!(distance("modle", "model"), 2);
        assert_eq!(distance("", "abc"), 3);
        assert_eq!(distance("kitten", "sitting"), 3);
    }

    #[test]
    fn a_near_miss_is_suggested_and_a_far_one_is_not() {
        let known = ["epochs".to_string(), "batch_size".to_string()];
        assert_eq!(suggestion("epochz", known.iter()), " Did you mean 'epochs'?");
        assert_eq!(suggestion("batch_siez", known.iter()), " Did you mean 'batch_size'?");
        assert_eq!(suggestion("xylophone", known.iter()), "");
    }

    #[test]
    fn an_annotation_follows_the_default_and_the_descriptor() {
        assert_eq!(annotation(&json!(true), None), "bool");
        assert_eq!(annotation(&json!(3), None), "int");
        assert_eq!(annotation(&json!(0.5), None), "float");
        assert_eq!(annotation(&json!("x"), None), "str");
        assert_eq!(annotation(&json!([]), None), "list[str]");
        assert_eq!(annotation(&Value::Null, None), "str | None");
        assert_eq!(literal(&["a", "b'c"]), "Literal['a', 'b\\'c']");
    }

    #[test]
    fn every_trainspec_field_gets_an_annotation_from_its_descriptor_when_unset() {
        // `Option` fields default to null, so their type can only come from the descriptor.
        let descriptors = <TrainSpec as JobFields>::field_descriptors();
        let min_lr = descriptors.iter().find(|d| d.name == "min_lr");
        assert_eq!(annotation(&Value::Null, min_lr), "float | None");
        let style = descriptors.iter().find(|d| d.name == "wsd_decay_style");
        let text = annotation(&Value::Null, style);
        assert!(text.starts_with("Literal['") && text.ends_with("| None"), "{text}");
    }

    #[test]
    fn f32_fields_serialize_as_their_shortest_decimal() {
        let spec = TrainSpec {
            lora_dropout: 0.05,
            ..TrainSpec::default()
        };
        let object = object_of_for_test(&spec);
        assert_eq!(object["lora_dropout"], json!(0.05));
        // `to_value` is what widened it.
        assert_ne!(serde_json::to_value(&spec).unwrap()["lora_dropout"], json!(0.05));
    }

    fn object_of_for_test(spec: &TrainSpec) -> Map<String, Value> {
        match value_of(spec).unwrap() {
            Value::Object(map) => map,
            other => panic!("{other}"),
        }
    }
}
