//! A form is a spec in text: the default text of every spec builds the spec it
//! came from, and what a person types builds the spec the CLI would build.

use pmetal_core::jobs::*;
use pmetal_core::{DefaultValue, FieldKind, FieldShape, JobFields, default_text, spec_from_text};
use serde::{Serialize, de::DeserializeOwned};

/// The text of every field of `S::default()`, handed back to `spec_from_text`.
fn through_the_form<S>() -> S
where
    S: JobFields + Default + Serialize + DeserializeOwned,
{
    let texts = default_text::<S>();
    spec_from_text::<S>(|d| {
        texts
            .iter()
            .find(|(name, _)| *name == d.name)
            .map(|(_, text)| text.clone())
    })
    .unwrap_or_else(|errors| {
        panic!(
            "{}: its own default text was refused: {errors:?}",
            S::subcommand()
        )
    })
}

fn check_roundtrip<S>()
where
    S: JobFields + Default + Serialize + DeserializeOwned,
{
    let built = through_the_form::<S>();
    let default = S::default();
    assert_eq!(
        serde_json::to_value(&built).unwrap(),
        serde_json::to_value(&default).unwrap(),
        "{}: the default form builds a different spec than `Default`",
        S::subcommand()
    );
    assert_eq!(built.to_argv(), default.to_argv(), "{}", S::subcommand());
}

/// What the descriptor says the default is, against what the spec's `Default`
/// is. A form reads the latter; the former is what other surfaces display.
fn check_descriptor_defaults<S>() -> Vec<String>
where
    S: JobFields + Default + Serialize + DeserializeOwned,
{
    let texts = default_text::<S>();
    let mut drift = Vec::new();
    for d in S::field_descriptors() {
        let actual = &texts.iter().find(|(n, _)| *n == d.name).unwrap().1;
        let declared = match d.default {
            DefaultValue::None => continue,
            other => other.display(),
        };
        let same = match d.kind {
            FieldKind::Number { .. } => declared
                .parse::<f64>()
                .ok()
                .zip(actual.parse::<f64>().ok())
                // An f32 field shows its f64 default widened.
                .is_some_and(|(a, b)| (a - b).abs() <= 1e-6 * a.abs().max(b.abs()).max(1e-12)),
            FieldKind::Integer { .. } => declared == *actual,
            _ => declared == *actual,
        };
        if !same {
            drift.push(format!(
                "{}.{}: descriptor says {declared:?}, Default is {actual:?}",
                S::subcommand(),
                d.name
            ));
        }
    }
    drift
}

macro_rules! for_every_spec {
    ($callback:ident) => {{
        $callback!(BenchSpec);
        $callback!(DecideSpec);
        $callback!(DflashSpec);
        $callback!(DistillSpec);
        $callback!(EmbedTrainSpec);
        $callback!(EvalSpec);
        $callback!(FuseSpec);
        $callback!(GrpoSpec);
        $callback!(InferSpec);
        $callback!(MergeSpec);
        $callback!(PackExpertsSpec);
        $callback!(PreferenceSpec);
        $callback!(PretrainSpec);
        $callback!(QuantizeSpec);
        $callback!(RlkdSpec);
        $callback!(ServeSpec);
        $callback!(TokenizeSpec);
        $callback!(TrainSpec);
        $callback!(TrainDiffusionSpec);
        $callback!(TrainDraftSpec);
        $callback!(TrainMtpSpec);
        $callback!(VocodeSpec);
    }};
}

#[test]
fn the_default_form_of_every_spec_builds_that_spec() {
    macro_rules! check {
        ($spec:ty) => {
            check_roundtrip::<$spec>()
        };
    }
    for_every_spec!(check);
}

#[test]
fn the_descriptors_defaults_are_the_specs_defaults() {
    let mut drift = Vec::new();
    macro_rules! check {
        ($spec:ty) => {
            drift.extend(check_descriptor_defaults::<$spec>())
        };
    }
    for_every_spec!(check);
    assert!(drift.is_empty(), "{}", drift.join("\n"));
}

#[test]
fn every_numeric_and_toggle_field_has_a_default_text_a_form_can_show() {
    // An empty number is refused by the form, so a spec whose default text is
    // empty for one could never be built from its own untouched form.
    macro_rules! check {
        ($spec:ty) => {{
            let texts = default_text::<$spec>();
            for d in <$spec>::field_descriptors() {
                let numeric = matches!(
                    d.kind,
                    FieldKind::Number { .. } | FieldKind::Integer { .. } | FieldKind::Toggle
                );
                if numeric && d.shape == FieldShape::Scalar {
                    let text = &texts.iter().find(|(n, _)| *n == d.name).unwrap().1;
                    assert!(!text.is_empty(), "{}.{}", <$spec>::subcommand(), d.name);
                }
            }
        }};
    }
    for_every_spec!(check);
}

#[test]
fn typed_text_builds_the_spec_the_cli_would() {
    let mut texts = default_text::<TrainSpec>();
    for (name, text) in &mut texts {
        match *name {
            "model" => *text = "Qwen/Qwen3-0.6B".into(),
            "dataset" => *text = "data.jsonl".into(),
            "learning_rate" => *text = "1e-5".into(),
            "batch_size" => *text = "4".into(),
            "cut_cross_entropy" => *text = "Enabled".into(),
            _ => {}
        }
    }
    let spec = spec_from_text::<TrainSpec>(|d| {
        texts
            .iter()
            .find(|(n, _)| *n == d.name)
            .map(|(_, t)| t.clone())
    })
    .unwrap();
    assert_eq!(spec.model, "Qwen/Qwen3-0.6B");
    assert_eq!(spec.learning_rate, 1e-5);
    assert_eq!(spec.batch_size, 4);
    assert!(spec.cut_cross_entropy);
    let argv = spec.to_argv();
    let at = argv.iter().position(|a| a == "--learning-rate").unwrap();
    assert_eq!(argv[at + 1], "0.00001");
}

#[test]
fn text_that_is_not_a_value_is_refused_naming_the_field() {
    let errors = spec_from_text::<TrainSpec>(|d| match d.name {
        "learning_rate" => Some("fast".to_string()),
        "batch_size" => Some("4.5".to_string()),
        "cut_cross_entropy" => Some("maybe".to_string()),
        "epochs" => Some(String::new()),
        _ => None,
    })
    .unwrap_err();
    let mut named: Vec<&str> = errors.iter().map(|e| e.field.as_str()).collect();
    named.sort_unstable();
    assert_eq!(
        named,
        ["batch_size", "cut_cross_entropy", "epochs", "learning_rate"]
    );
    let lr = errors.iter().find(|e| e.field == "learning_rate").unwrap();
    assert!(lr.message.contains("'fast'"), "{}", lr.message);
}

#[test]
fn a_value_the_field_cannot_hold_is_refused_naming_the_field() {
    // 300 does not fit the `u8` the field is.
    let errors =
        spec_from_text::<InferSpec>(|d| (d.name == "kv_quant").then(|| "300".into())).unwrap_err();
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].field, "kv_quant");
}

#[test]
fn an_empty_optional_is_unset_and_a_picker_placeholder_is_empty() {
    let spec = spec_from_text::<InferSpec>(|d| match d.name {
        "temperature" => Some(String::new()),
        "lora" => Some("(none)".to_string()),
        "model" => Some("(not selected)".to_string()),
        _ => None,
    })
    .unwrap();
    assert_eq!(spec.temperature, None);
    assert_eq!(spec.lora, None);
    assert_eq!(spec.model, InferSpec::default().model);
}

#[test]
fn a_list_is_comma_separated() {
    let spec = spec_from_text::<InferSpec>(|d| {
        (d.name == "image").then(|| " a.png, b.png ,, c.png".to_string())
    })
    .unwrap();
    assert_eq!(spec.image, ["a.png", "b.png", "c.png"]);
    let spec = spec_from_text::<InferSpec>(|d| (d.name == "image").then(String::new)).unwrap();
    assert!(spec.image.is_empty());
}

#[test]
fn a_field_the_form_does_not_show_keeps_the_specs_default() {
    let spec = spec_from_text::<TrainSpec>(|_| None).unwrap();
    assert_eq!(spec.to_argv(), TrainSpec::default().to_argv());
}
