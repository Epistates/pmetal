//! Behavioural tests for the code `#[derive(JobSpec)]` generates.
//!
//! Every spec here is derived in a crate that denies all warnings, so the
//! generated impl also has to be lint-clean in user code. Compile-fail cases
//! live in `tests/ui`.

#![deny(warnings)]
// The specs below exist to exercise the generated impls; some fields are only
// declared, never read by the test body.
#![allow(dead_code)]

use pmetal_core::{DefaultValue, FieldKind, JobFields, JobKind};
use pmetal_core_derive::JobSpec;
use std::marker::PhantomData;

// ---------------------------------------------------------------------------
// Generics, where clauses, empty types, raw identifiers
// ---------------------------------------------------------------------------

#[derive(JobSpec)]
#[spec(kind = "Train", subcommand = "generic")]
struct Generic<T, const N: usize>
where
    T: Clone + Default,
{
    #[job(label = "Name", group = "G", argv = "--name")]
    name: String,
    #[job(skip_descriptor, skip_argv)]
    _marker: PhantomData<T>,
    #[job(skip_descriptor, skip_argv)]
    _arr: [u8; N],
}

#[test]
fn derive_supports_generics_and_where_clauses() {
    let spec = Generic::<u8, 2> {
        name: "x".into(),
        _marker: PhantomData,
        _arr: [0; 2],
    };
    assert_eq!(spec.to_argv(), vec!["--name", "x"]);
    assert_eq!(Generic::<u8, 2>::subcommand(), "generic");
    assert_eq!(Generic::<u8, 2>::field_descriptors().len(), 1);
}

#[derive(JobSpec)]
#[spec(kind = "Eval", subcommand = "empty")]
struct Empty {}

#[test]
fn derive_on_an_empty_struct_is_lint_clean() {
    let spec = Empty {};
    assert!(spec.to_argv().is_empty());
    assert!(spec.validate_descriptors().is_empty());
    assert!(Empty::field_descriptors().is_empty());
    assert!(matches!(Empty::job_kind(), JobKind::Eval));
}

#[derive(JobSpec)]
#[spec(kind = "Train", subcommand = "raw")]
struct RawIdent {
    #[job(label = "Type", group = "G", argv = "--type")]
    r#type: String,
}

#[test]
fn raw_identifier_fields_are_named_without_the_prefix() {
    assert_eq!(RawIdent::field_descriptors()[0].name, "type");
    let spec = RawIdent { r#type: "a".into() };
    assert_eq!(spec.to_argv(), vec!["--type", "a"]);
    let errs = RawIdent { r#type: "a".into() }.validate_descriptors();
    assert!(errs.is_empty());
}

// ---------------------------------------------------------------------------
// Validation semantics
// ---------------------------------------------------------------------------

#[derive(JobSpec)]
#[spec(kind = "Train", subcommand = "ranges")]
struct Ranges {
    #[job(label = "LR", group = "G", argv = "--lr", min = 0.0, max = 1.0)]
    lr: f64,
    #[job(label = "Free", group = "G", argv = "--free")]
    free: f32,
    #[job(label = "Steps", group = "G", argv = "--steps", min = 1, max = 10)]
    steps: u32,
    #[job(label = "Maybe", group = "G", argv = "--maybe", min = 2, max = 4)]
    maybe: Option<i16>,
    #[job(
        label = "Mode",
        group = "G",
        argv = "--mode",
        kind = "enum",
        enum_options = ["a", "b"]
    )]
    mode: String,
    #[job(label = "Name", group = "G", argv = "--name", required)]
    name: Option<String>,
}

fn ranges() -> Ranges {
    Ranges {
        lr: 0.5,
        free: 1.0,
        steps: 5,
        maybe: None,
        mode: "a".into(),
        name: Some("n".into()),
    }
}

fn error_fields(spec: &Ranges) -> Vec<String> {
    spec.validate_descriptors()
        .into_iter()
        .map(|e| e.field)
        .collect()
}

#[test]
fn in_range_spec_validates_clean() {
    assert!(error_fields(&ranges()).is_empty());
}

#[test]
fn out_of_range_values_are_reported_per_field() {
    let mut s = ranges();
    s.lr = 1.5;
    s.steps = 0;
    s.maybe = Some(9);
    s.mode = "zzz".into();
    s.name = Some(String::new());
    assert_eq!(
        error_fields(&s),
        vec!["lr", "steps", "maybe", "mode", "name"]
    );
}

#[test]
fn non_finite_floats_are_rejected() {
    // `NaN < min` and `NaN > max` are both false, so without an explicit
    // finiteness check a NaN learning rate sails through a bounded field.
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut s = ranges();
        s.lr = bad;
        assert_eq!(error_fields(&s), vec!["lr"], "lr = {bad}");
    }
    let mut s = ranges();
    s.free = f32::NAN;
    assert_eq!(error_fields(&s), vec!["free"]);
}

#[test]
fn descriptors_carry_declared_bounds_and_options() {
    let d = Ranges::field_descriptors();
    assert!(matches!(d[0].kind, FieldKind::Number { min, max } if min == 0.0 && max == 1.0));
    assert!(matches!(d[2].kind, FieldKind::Integer { min: 1, max: 10 }));
    assert!(matches!(d[4].kind, FieldKind::Enum { options } if options == ["a", "b"]));
    assert!(d[5].required && d[5].argv_optional);
    assert!(matches!(d[0].default, DefaultValue::None));
}

// ---------------------------------------------------------------------------
// argv emission
// ---------------------------------------------------------------------------

#[derive(JobSpec)]
#[spec(kind = "Train", subcommand = "argv")]
struct Argv {
    #[job(label = "A", group = "G", argv = "--a", flag)]
    a: bool,
    #[job(label = "B", group = "G", argv = "--no-b", flag, invert)]
    b: bool,
    #[job(label = "C", group = "G", argv = "--c", csv)]
    c: Vec<String>,
    #[job(label = "D", group = "G", argv = "--d")]
    d: Option<f32>,
    #[job(label = "E", group = "G", argv = "--e")]
    e: i64,
    #[job(label = "F", group = "G", argv = "--f", skip_argv)]
    f: String,
    #[job(label = "H", group = "G")]
    h: String,
    #[job(label = "V", group = "G", argv = "--v", valued, default_bool = true)]
    v: bool,
    #[job(skip_descriptor, argv = "--cli-only", flag)]
    cli_only: bool,
}

#[test]
fn argv_follows_flag_csv_option_and_skip_rules() {
    let s = Argv {
        a: true,
        b: false,
        c: vec!["x".into(), "y".into()],
        d: Some(0.5),
        e: -3,
        f: "ignored".into(),
        h: "form-only".into(),
        v: false,
        cli_only: true,
    };
    assert_eq!(
        s.to_argv(),
        [
            "--a",
            "--no-b",
            "--c",
            "x,y",
            "--d",
            "0.5",
            "--e",
            "-3",
            "--v",
            "false",
            "--cli-only"
        ]
    );
    let s = Argv {
        a: false,
        b: true,
        c: vec![],
        d: None,
        e: 0,
        f: String::new(),
        h: String::new(),
        v: true,
        cli_only: false,
    };
    assert_eq!(s.to_argv(), ["--e", "0", "--v", "true"]);
}

/// A descriptor says whether a field holds one value, one that may be absent,
/// or a list: a form needs it to know what an empty entry means.
#[test]
fn a_descriptor_knows_the_shape_of_its_field() {
    use pmetal_core::FieldShape;
    let shapes: Vec<(&str, FieldShape)> = Argv::field_descriptors()
        .iter()
        .map(|d| (d.name, d.shape))
        .collect();
    assert_eq!(
        shapes,
        [
            ("a", FieldShape::Scalar),
            ("b", FieldShape::Scalar),
            ("c", FieldShape::List),
            ("d", FieldShape::Optional),
            ("e", FieldShape::Scalar),
            ("f", FieldShape::Scalar),
            ("h", FieldShape::Scalar),
            ("v", FieldShape::Scalar),
        ]
    );
}

/// `argv_flags` lists every flag `to_argv` can spell, the CLI-only ones
/// included and the form-only and `skip_argv` fields not.
#[test]
fn argv_flags_are_every_flag_the_spec_can_spell() {
    assert_eq!(
        Argv::argv_flags(),
        ["--a", "--no-b", "--c", "--d", "--e", "--v", "--cli-only"]
    );
}
