//! `#[derive(JobSpec)]`: derives `pmetal_core::JobFields` for a spec struct.
//!
//! Use it through `pmetal_core::JobSpec`, which re-exports it; the generated
//! code needs `pmetal-core` in scope. The specs in `pmetal_core::jobs` are
//! worked examples, and this crate's README shows a minimal one.
//!
//! ## Spec-level attributes (`#[spec(...)]` on the struct)
//!
//! - `kind = "Train"` — required; the `pmetal_core::JobKind` variant.
//! - `subcommand = "train"` — required; the CLI subcommand string.
//! - `events` — optional; the command writes a `--log-events` file
//!   (`JobEvent` lines a supervisor follows). It becomes
//!   `JobFields::REPORTS_EVENTS`, the one place that fact is stated: the CLI
//!   refuses the flag for other commands and every surface that launches jobs
//!   asks for the file only where the command writes it.
//!
//! ## Field-level attributes (`#[job(...)]` on each field)
//!
//! - `label = "Learning Rate"` — form label (required unless `skip_descriptor`).
//! - `group = "Training"` — form section (required unless `skip_descriptor`).
//! - `help = "..."` — optional extended help.
//! - `argv = "--learning-rate"` — CLI flag this field maps to (omit for form-only).
//! - `default = "2e-4"` / `default_int = 1` / `default_float = 1.0` /
//!   `default_bool = true` — typed default for descriptor display; at most one.
//! - `kind = "text" | "toggle" | "number" | "integer" | "model_picker" |
//!   "dataset_picker" | "path" | "read_only" | "enum"` — overrides the
//!   type-inferred `pmetal_core::FieldKind`. `number` needs a float field and
//!   `integer` an integer field.
//! - `enum_options = ["none", "nf4", "fp4"]` — required by, and only valid
//!   with, `kind = "enum"` on a `String` / `Option<String>` field.
//! - `min = 0.0`, `max = 1.0` — inclusive bounds for numeric fields (defaults:
//!   the numeric type's full range). Integer fields take whole numbers and keep
//!   full `i64` precision; float bounds must be finite; `min <= max`.
//! - `flag` — `bool` field; emit `--flag` when `true`, omit when `false`.
//! - `invert` — pair with `flag`; emit `--flag` when `false`, omit when `true`
//!   (for `--no-foo` semantics).
//! - `valued` — `bool` field whose flag takes a value, `--flag true|false`;
//!   emitted unless it is the default. For a default of `true` that a switch
//!   could not turn off.
//! - `always` — emit the flag even at the default, for a setting that must be
//!   stated and never left to the command (a bind address). Valid on plain
//!   numeric, `String` and `valued` fields.
//! - `csv` — `Vec<String>` field; emit `--flag a,b,c` joined with commas.
//! - `repeat` — `Vec<String>` field; emit `--flag a --flag b`, one per item.
//! - `required` — non-empty / non-default validation. Valid on `String`,
//!   `Option<_>` and `Vec<String>` fields.
//! - `skip_descriptor` — exclude from `field_descriptors()` and from
//!   `validate_descriptors()` (CLI-only field).
//! - `skip_argv` — exclude from `to_argv()` (form-only field).
//!
//! Every key may appear at most once per field, and every combination that
//! would silently do nothing (an `invert` without `flag`, a `min` on a string,
//! an `argv` flag on a bool without `flag`, ...) is a compile error pointing at
//! the offending key. All problems in a struct are reported in one pass.
//!
//! Float fields additionally reject `NaN` and infinities in
//! `validate_descriptors()`, bounded or not: `NaN` compares false against
//! every bound, so a range check alone would wave it through.
//!
//! The generated impl also names the fields that `--output` and `--log-metrics`
//! map to (`JobFields::OUTPUT_FIELD`, `JobFields::LOG_METRICS_FIELD`), so a
//! launcher can find where a job writes without scanning descriptors (which a
//! `skip_descriptor` field does not have).
//!
//! The generated impl carries the struct's generics and `where` clause, names
//! everything through absolute paths (`::pmetal_core`, `::std`) and is marked
//! `#[automatically_derived]`.

use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::quote;
use syn::{
    Attribute, Data, DeriveInput, Expr, ExprLit, Field, Fields, Ident, Lit, LitStr, Token, Type,
    ext::IdentExt, meta::ParseNestedMeta, parse_macro_input, punctuated::Punctuated,
    spanned::Spanned,
};

#[proc_macro_derive(JobSpec, attributes(job, spec))]
pub fn derive_job_spec(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match expand(&input) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn expand(input: &DeriveInput) -> syn::Result<TokenStream2> {
    let ident = &input.ident;

    let spec_meta = parse_spec_meta(&input.attrs, ident.span());

    let fields = match &input.data {
        Data::Struct(s) => match &s.fields {
            Fields::Named(named) => named.named.iter().collect::<Vec<_>>(),
            _ => {
                return Err(syn::Error::new(
                    ident.span(),
                    "JobSpec only supports structs with named fields",
                ));
            }
        },
        _ => {
            return Err(syn::Error::new(
                ident.span(),
                "JobSpec only supports structs",
            ));
        }
    };

    // Collect every problem instead of stopping at the first, so one compile
    // reports the whole struct.
    let mut errors: Option<syn::Error> = None;
    let mut fail = |e: syn::Error| match &mut errors {
        Some(all) => all.combine(e),
        None => errors = Some(e),
    };

    let spec_meta = match spec_meta {
        Ok(m) => Some(m),
        Err(e) => {
            fail(e);
            None
        }
    };

    let mut descriptor_items = Vec::new();
    let mut argv_items = Vec::new();
    let mut argv_flags = Vec::new();
    let mut validate_items = Vec::new();
    let mut output_field: Option<String> = None;
    let mut log_metrics_field: Option<String> = None;

    for field in fields {
        let parsed = match parse_field(field) {
            Ok(p) => p,
            Err(e) => {
                fail(e);
                continue;
            }
        };
        if !parsed.attrs.skip_descriptor {
            match emit_descriptor(&parsed) {
                Ok(t) => descriptor_items.push(t),
                Err(e) => fail(e),
            }
            validate_items.push(emit_validate(&parsed));
        }
        if !parsed.attrs.skip_argv && parsed.attrs.argv.is_some() {
            argv_items.push(emit_argv(&parsed));
            if let Some(flag) = &parsed.attrs.argv {
                argv_flags.push(flag.clone());
                match flag.as_str() {
                    "--output" => output_field = Some(parsed.name_str.clone()),
                    "--log-metrics" => log_metrics_field = Some(parsed.name_str.clone()),
                    _ => {}
                }
            }
        }
    }

    if let Some(e) = errors {
        return Err(e);
    }
    let spec_meta = spec_meta.expect("a missing spec_meta was recorded as an error above");

    let kind_ident = &spec_meta.kind;
    let subcommand = &spec_meta.subcommand;
    let events = spec_meta.events;
    let option_str = |name: &Option<String>| match name {
        Some(name) => quote! { ::std::option::Option::Some(#name) },
        None => quote! { ::std::option::Option::None },
    };
    let output_field = option_str(&output_field);
    let log_metrics_field = option_str(&log_metrics_field);
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    Ok(quote! {
        #[automatically_derived]
        impl #impl_generics ::pmetal_core::JobFields for #ident #ty_generics #where_clause {
            const REPORTS_EVENTS: bool = #events;
            const OUTPUT_FIELD: ::std::option::Option<&'static str> = #output_field;
            const LOG_METRICS_FIELD: ::std::option::Option<&'static str> = #log_metrics_field;

            fn field_descriptors() -> &'static [::pmetal_core::FieldDescriptor] {
                static DESCRIPTORS: &[::pmetal_core::FieldDescriptor] = &[
                    #(#descriptor_items),*
                ];
                DESCRIPTORS
            }

            fn to_argv(&self) -> ::std::vec::Vec<::std::string::String> {
                #[allow(unused_mut)]
                let mut args: ::std::vec::Vec<::std::string::String> = ::std::vec::Vec::new();
                // A setting at its default is left to the command's own: a
                // flag passed with the default's value would beat a config
                // file that names another.
                #[allow(unused_variables)]
                let default = <Self as ::std::default::Default>::default();
                #(#argv_items)*
                args
            }

            fn argv_flags() -> &'static [&'static str] {
                &[#(#argv_flags),*]
            }

            fn validate_descriptors(&self) -> ::std::vec::Vec<::pmetal_core::FieldError> {
                #[allow(unused_mut)]
                let mut errs: ::std::vec::Vec<::pmetal_core::FieldError> = ::std::vec::Vec::new();
                #(#validate_items)*
                errs
            }

            fn subcommand() -> &'static str {
                #subcommand
            }

            fn job_kind() -> ::pmetal_core::JobKind {
                ::pmetal_core::JobKind::#kind_ident
            }
        }
    })
}

// ---------------------------------------------------------------------------
// Spec-level (#[spec(...)]) attribute parsing
// ---------------------------------------------------------------------------

struct SpecMeta {
    kind: Ident,
    subcommand: LitStr,
    events: bool,
}

fn parse_spec_meta(attrs: &[Attribute], span: Span) -> syn::Result<SpecMeta> {
    let mut kind: Option<Ident> = None;
    let mut subcommand: Option<LitStr> = None;
    let mut events = false;

    for attr in attrs.iter().filter(|a| a.path().is_ident("spec")) {
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("kind") {
                reject_duplicate(&meta, kind.is_some())?;
                let lit: LitStr = meta.value()?.parse()?;
                // `format_ident!` would panic on text that is not an
                // identifier; parsing reports it at the literal instead.
                kind = Some(lit.parse::<Ident>().map_err(|_| {
                    syn::Error::new(
                        lit.span(),
                        format!("`{}` is not a `JobKind` variant name", lit.value()),
                    )
                })?);
                Ok(())
            } else if meta.path.is_ident("subcommand") {
                reject_duplicate(&meta, subcommand.is_some())?;
                subcommand = Some(meta.value()?.parse()?);
                Ok(())
            } else if meta.path.is_ident("events") {
                reject_duplicate(&meta, events)?;
                events = true;
                Ok(())
            } else {
                Err(meta
                    .error("unknown #[spec(...)] key — expected `kind`, `subcommand` or `events`"))
            }
        })?;
    }

    let kind = kind.ok_or_else(|| {
        syn::Error::new(
            span,
            "JobSpec requires #[spec(kind = \"<JobKind variant>\", subcommand = \"<cli>\")]",
        )
    })?;
    let subcommand = subcommand
        .ok_or_else(|| syn::Error::new(span, "JobSpec requires #[spec(subcommand = \"...\")]"))?;
    Ok(SpecMeta {
        kind,
        subcommand,
        events,
    })
}

fn reject_duplicate(meta: &ParseNestedMeta<'_>, already_set: bool) -> syn::Result<()> {
    if already_set {
        let key = meta
            .path
            .get_ident()
            .map(|i| i.to_string())
            .unwrap_or_default();
        return Err(meta.error(format!("`{key}` is set more than once")));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Field-level parsing
// ---------------------------------------------------------------------------

/// A `min` / `max` literal, kept exact: integer fields must not round-trip
/// their bounds through `f64`.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Bound {
    Int(i128),
    Float(f64),
}

impl Bound {
    fn as_f64(self) -> f64 {
        match self {
            Self::Int(i) => i as f64,
            Self::Float(f) => f,
        }
    }

    /// The bound as a whole number, if it is one that fits `i64` (the
    /// descriptor's integer type).
    fn as_whole_i64(self) -> Option<i64> {
        match self {
            Self::Int(i) => i64::try_from(i).ok(),
            Self::Float(f) if f.fract() == 0.0 && f.abs() < 9.0e18 => Some(f as i64),
            Self::Float(_) => None,
        }
    }
}

#[derive(Default, Clone)]
struct FieldAttrs {
    label: Option<String>,
    help: Option<String>,
    group: Option<String>,
    argv: Option<String>,

    default_str: Option<String>,
    default_int: Option<i64>,
    default_float: Option<f64>,
    default_bool: Option<bool>,

    // Kind override.
    kind_override: Option<(String, Span)>,
    enum_options: Option<Vec<String>>,
    min: Option<(Bound, Span)>,
    max: Option<(Bound, Span)>,

    // Argv style.
    flag: Option<Span>,
    invert: Option<Span>,
    csv: Option<Span>,
    repeat: Option<Span>,
    valued: Option<Span>,
    always: Option<Span>,

    required: Option<Span>,
    skip_descriptor: bool,
    skip_argv: bool,
}

struct ParsedField {
    name: Ident,
    /// The field's name as written, without any `r#` prefix.
    name_str: String,
    ty: FieldType,
    attrs: FieldAttrs,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FieldType {
    String,
    Bool,
    F32,
    F64,
    U8,
    U16,
    U32,
    U64,
    Usize,
    I8,
    I16,
    I32,
    I64,
    /// Wrapped in `Option<T>` — emit only when `Some`.
    OptionString,
    OptionF32,
    OptionF64,
    OptionU8,
    OptionU16,
    OptionU32,
    OptionU64,
    OptionUsize,
    OptionI8,
    OptionI16,
    OptionI32,
    OptionI64,
    OptionBool,
    /// `Vec<String>` — pair with `csv` for argv emission.
    VecString,
    /// Anything else; argv emission must use `Display` and descriptor must
    /// supply an explicit `kind = "..."` override.
    Other,
}

impl FieldType {
    fn is_option(&self) -> bool {
        matches!(
            self,
            Self::OptionString
                | Self::OptionF32
                | Self::OptionF64
                | Self::OptionU8
                | Self::OptionU16
                | Self::OptionU32
                | Self::OptionU64
                | Self::OptionUsize
                | Self::OptionI8
                | Self::OptionI16
                | Self::OptionI32
                | Self::OptionI64
                | Self::OptionBool
        )
    }

    fn is_string_like(&self) -> bool {
        matches!(self, Self::String | Self::OptionString)
    }

    fn is_bool(&self) -> bool {
        matches!(self, Self::Bool | Self::OptionBool)
    }

    fn is_int(&self) -> bool {
        matches!(
            self,
            Self::U8
                | Self::U16
                | Self::U32
                | Self::U64
                | Self::Usize
                | Self::I8
                | Self::I16
                | Self::I32
                | Self::I64
                | Self::OptionU8
                | Self::OptionU16
                | Self::OptionU32
                | Self::OptionU64
                | Self::OptionUsize
                | Self::OptionI8
                | Self::OptionI16
                | Self::OptionI32
                | Self::OptionI64
        )
    }

    fn is_float(&self) -> bool {
        matches!(
            self,
            Self::F32 | Self::F64 | Self::OptionF32 | Self::OptionF64
        )
    }
}

fn parse_field(field: &Field) -> syn::Result<ParsedField> {
    let name = field
        .ident
        .clone()
        .ok_or_else(|| syn::Error::new(field.span(), "JobSpec fields must be named"))?;
    let name_str = name.unraw().to_string();
    let ty = classify_type(&field.ty);
    let attrs = parse_field_attrs(&field.attrs)?;
    let parsed = ParsedField {
        name,
        name_str,
        ty,
        attrs,
    };
    check_attr_consistency(&parsed)?;
    Ok(parsed)
}

fn classify_type(ty: &Type) -> FieldType {
    let last = type_path_tail(ty);
    let inner = type_option_inner(ty).and_then(|t| type_path_tail(&t));
    let vec_inner = type_generic_inner_tail(ty, "Vec");

    match (last.as_deref(), inner.as_deref(), vec_inner.as_deref()) {
        (Some("String"), _, _) => FieldType::String,
        (Some("bool"), _, _) => FieldType::Bool,
        (Some("f32"), _, _) => FieldType::F32,
        (Some("f64"), _, _) => FieldType::F64,
        (Some("u8"), _, _) => FieldType::U8,
        (Some("u16"), _, _) => FieldType::U16,
        (Some("u32"), _, _) => FieldType::U32,
        (Some("u64"), _, _) => FieldType::U64,
        (Some("usize"), _, _) => FieldType::Usize,
        (Some("i8"), _, _) => FieldType::I8,
        (Some("i16"), _, _) => FieldType::I16,
        (Some("i32"), _, _) => FieldType::I32,
        (Some("i64"), _, _) => FieldType::I64,
        (Some("Vec"), _, Some("String")) => FieldType::VecString,
        (Some("Option"), Some("String"), _) => FieldType::OptionString,
        (Some("Option"), Some("f32"), _) => FieldType::OptionF32,
        (Some("Option"), Some("f64"), _) => FieldType::OptionF64,
        (Some("Option"), Some("u8"), _) => FieldType::OptionU8,
        (Some("Option"), Some("u16"), _) => FieldType::OptionU16,
        (Some("Option"), Some("u32"), _) => FieldType::OptionU32,
        (Some("Option"), Some("u64"), _) => FieldType::OptionU64,
        (Some("Option"), Some("usize"), _) => FieldType::OptionUsize,
        (Some("Option"), Some("i8"), _) => FieldType::OptionI8,
        (Some("Option"), Some("i16"), _) => FieldType::OptionI16,
        (Some("Option"), Some("i32"), _) => FieldType::OptionI32,
        (Some("Option"), Some("i64"), _) => FieldType::OptionI64,
        (Some("Option"), Some("bool"), _) => FieldType::OptionBool,
        _ => FieldType::Other,
    }
}

fn type_path_tail(ty: &Type) -> Option<String> {
    if let Type::Path(p) = ty {
        p.path.segments.last().map(|s| s.ident.to_string())
    } else {
        None
    }
}

fn type_option_inner(ty: &Type) -> Option<Type> {
    type_generic_inner(ty, "Option")
}

fn type_generic_inner_tail(ty: &Type, wrapper: &str) -> Option<String> {
    type_generic_inner(ty, wrapper).and_then(|t| type_path_tail(&t))
}

fn type_generic_inner(ty: &Type, wrapper: &str) -> Option<Type> {
    let Type::Path(p) = ty else { return None };
    let last = p.path.segments.last()?;
    if last.ident != wrapper {
        return None;
    }
    let syn::PathArguments::AngleBracketed(ab) = &last.arguments else {
        return None;
    };
    for arg in &ab.args {
        if let syn::GenericArgument::Type(t) = arg {
            return Some(t.clone());
        }
    }
    None
}

fn parse_field_attrs(attrs: &[Attribute]) -> syn::Result<FieldAttrs> {
    let mut out = FieldAttrs::default();

    for attr in attrs.iter().filter(|a| a.path().is_ident("job")) {
        attr.parse_nested_meta(|meta| {
            let path = &meta.path;
            if path.is_ident("label") {
                reject_duplicate(&meta, out.label.is_some())?;
                out.label = Some(parse_str_value(&meta)?);
            } else if path.is_ident("help") {
                reject_duplicate(&meta, out.help.is_some())?;
                out.help = Some(parse_str_value(&meta)?);
            } else if path.is_ident("group") {
                reject_duplicate(&meta, out.group.is_some())?;
                out.group = Some(parse_str_value(&meta)?);
            } else if path.is_ident("argv") {
                reject_duplicate(&meta, out.argv.is_some())?;
                out.argv = Some(parse_str_value(&meta)?);
            } else if path.is_ident("default") {
                reject_duplicate(&meta, out.default_str.is_some())?;
                out.default_str = Some(parse_str_value(&meta)?);
            } else if path.is_ident("default_int") {
                reject_duplicate(&meta, out.default_int.is_some())?;
                out.default_int = Some(parse_int_value(&meta)?);
            } else if path.is_ident("default_float") {
                reject_duplicate(&meta, out.default_float.is_some())?;
                out.default_float = Some(parse_float_value(&meta)?);
            } else if path.is_ident("default_bool") {
                reject_duplicate(&meta, out.default_bool.is_some())?;
                out.default_bool = Some(parse_bool_value(&meta)?);
            } else if path.is_ident("kind") {
                reject_duplicate(&meta, out.kind_override.is_some())?;
                let lit: LitStr = meta.value()?.parse()?;
                out.kind_override = Some((lit.value(), lit.span()));
            } else if path.is_ident("enum_options") {
                reject_duplicate(&meta, out.enum_options.is_some())?;
                let value = meta.value()?;
                let content;
                syn::bracketed!(content in value);
                let opts: Punctuated<LitStr, Token![,]> = Punctuated::parse_terminated(&content)?;
                out.enum_options = Some(opts.iter().map(|s| s.value()).collect());
            } else if path.is_ident("min") {
                reject_duplicate(&meta, out.min.is_some())?;
                out.min = Some(parse_bound_value(&meta)?);
            } else if path.is_ident("max") {
                reject_duplicate(&meta, out.max.is_some())?;
                out.max = Some(parse_bound_value(&meta)?);
            } else if path.is_ident("flag") {
                reject_duplicate(&meta, out.flag.is_some())?;
                out.flag = Some(path.span());
            } else if path.is_ident("invert") {
                reject_duplicate(&meta, out.invert.is_some())?;
                out.invert = Some(path.span());
            } else if path.is_ident("repeat") {
                reject_duplicate(&meta, out.repeat.is_some())?;
                out.repeat = Some(path.span());
            } else if path.is_ident("valued") {
                reject_duplicate(&meta, out.valued.is_some())?;
                out.valued = Some(path.span());
            } else if path.is_ident("always") {
                reject_duplicate(&meta, out.always.is_some())?;
                out.always = Some(path.span());
            } else if path.is_ident("csv") {
                reject_duplicate(&meta, out.csv.is_some())?;
                out.csv = Some(path.span());
            } else if path.is_ident("required") {
                reject_duplicate(&meta, out.required.is_some())?;
                out.required = Some(path.span());
            } else if path.is_ident("skip_descriptor") {
                reject_duplicate(&meta, out.skip_descriptor)?;
                out.skip_descriptor = true;
            } else if path.is_ident("skip_argv") {
                reject_duplicate(&meta, out.skip_argv)?;
                out.skip_argv = true;
            } else {
                return Err(meta.error("unknown #[job(...)] key"));
            }
            Ok(())
        })?;
    }

    Ok(out)
}

fn parse_str_value(meta: &ParseNestedMeta<'_>) -> syn::Result<String> {
    let lit: LitStr = meta.value()?.parse()?;
    Ok(lit.value())
}

fn parse_int_value(meta: &ParseNestedMeta<'_>) -> syn::Result<i64> {
    let expr: Expr = meta.value()?.parse()?;
    match expr_to_bound(&expr)? {
        Bound::Int(i) => i64::try_from(i)
            .map_err(|_| syn::Error::new(expr.span(), "integer literal does not fit `i64`")),
        Bound::Float(_) => Err(syn::Error::new(expr.span(), "expected integer literal")),
    }
}

fn parse_float_value(meta: &ParseNestedMeta<'_>) -> syn::Result<f64> {
    let expr: Expr = meta.value()?.parse()?;
    Ok(expr_to_bound(&expr)?.as_f64())
}

fn parse_bound_value(meta: &ParseNestedMeta<'_>) -> syn::Result<(Bound, Span)> {
    let expr: Expr = meta.value()?.parse()?;
    Ok((expr_to_bound(&expr)?, expr.span()))
}

fn parse_bool_value(meta: &ParseNestedMeta<'_>) -> syn::Result<bool> {
    let expr: Expr = meta.value()?.parse()?;
    if let Expr::Lit(ExprLit {
        lit: Lit::Bool(b), ..
    }) = &expr
    {
        Ok(b.value)
    } else {
        Err(syn::Error::new(expr.span(), "expected `true` or `false`"))
    }
}

/// A numeric literal, optionally negated. Float literals must be finite: a
/// literal like `1e999` overflows to infinity, which cannot be re-emitted as a
/// token.
fn expr_to_bound(expr: &Expr) -> syn::Result<Bound> {
    match expr {
        Expr::Lit(ExprLit {
            lit: Lit::Int(i), ..
        }) => Ok(Bound::Int(i.base10_parse::<i128>()?)),
        Expr::Lit(ExprLit {
            lit: Lit::Float(f), ..
        }) => {
            let v = f.base10_parse::<f64>()?;
            if v.is_finite() {
                Ok(Bound::Float(v))
            } else {
                Err(syn::Error::new(f.span(), "float literal must be finite"))
            }
        }
        Expr::Unary(syn::ExprUnary {
            op: syn::UnOp::Neg(_),
            expr: inner,
            ..
        }) => match expr_to_bound(inner)? {
            Bound::Int(i) => Ok(Bound::Int(-i)),
            Bound::Float(f) => Ok(Bound::Float(-f)),
        },
        _ => Err(syn::Error::new(expr.span(), "expected numeric literal")),
    }
}

// ---------------------------------------------------------------------------
// Attribute / type consistency
// ---------------------------------------------------------------------------

/// Reject attribute combinations that would otherwise be accepted and then
/// silently do nothing.
fn check_attr_consistency(field: &ParsedField) -> syn::Result<()> {
    let a = &field.attrs;
    let ty = field.ty;
    let here = field.name.span();

    let defaults = [
        a.default_str.is_some(),
        a.default_int.is_some(),
        a.default_float.is_some(),
        a.default_bool.is_some(),
    ];
    if defaults.iter().filter(|d| **d).count() > 1 {
        return Err(syn::Error::new(
            here,
            "at most one of `default`, `default_int`, `default_float`, `default_bool` may be set",
        ));
    }

    // Argv style keys.
    if let Some(span) = a.flag {
        if ty != FieldType::Bool {
            return Err(syn::Error::new(
                span,
                "`flag` may only be applied to `bool` fields",
            ));
        }
        if a.argv.is_none() {
            return Err(syn::Error::new(span, "`flag` needs `argv = \"...\"`"));
        }
        if a.csv.is_some() {
            return Err(syn::Error::new(span, "`flag` and `csv` cannot be combined"));
        }
    }
    if let Some(span) = a.repeat {
        if ty != FieldType::VecString {
            return Err(syn::Error::new(
                span,
                "`repeat` may only be applied to `Vec<String>` fields",
            ));
        }
        if a.argv.is_none() {
            return Err(syn::Error::new(span, "`repeat` needs `argv = \"...\"`"));
        }
        if a.csv.is_some() {
            return Err(syn::Error::new(
                span,
                "`repeat` and `csv` cannot be combined",
            ));
        }
    }
    if let Some(span) = a.valued {
        if ty != FieldType::Bool {
            return Err(syn::Error::new(
                span,
                "`valued` may only be applied to `bool` fields",
            ));
        }
        if a.argv.is_none() {
            return Err(syn::Error::new(span, "`valued` needs `argv = \"...\"`"));
        }
        if a.flag.is_some() || a.invert.is_some() || a.csv.is_some() {
            return Err(syn::Error::new(
                span,
                "`valued` cannot be combined with `flag`, `invert` or `csv`",
            ));
        }
    }
    if let Some(span) = a.always {
        if a.argv.is_none() {
            return Err(syn::Error::new(span, "`always` needs `argv = \"...\"`"));
        }
        if a.flag.is_some() || a.csv.is_some() || a.repeat.is_some() || ty.is_option() {
            return Err(syn::Error::new(
                span,
                "`always` cannot be combined with `flag`, `csv` or `repeat`, or applied to an \
                 `Option` field",
            ));
        }
    }
    if let Some(span) = a.invert
        && a.flag.is_none()
    {
        return Err(syn::Error::new(
            span,
            "`invert` only applies together with `flag`",
        ));
    }
    if let Some(span) = a.csv {
        if ty != FieldType::VecString {
            return Err(syn::Error::new(
                span,
                "`csv` may only be applied to `Vec<String>` fields",
            ));
        }
        if a.argv.is_none() {
            return Err(syn::Error::new(span, "`csv` needs `argv = \"...\"`"));
        }
    }
    if a.argv.is_some() && !a.skip_argv && a.flag.is_none() && a.valued.is_none() {
        if ty == FieldType::VecString && a.csv.is_none() && a.repeat.is_none() {
            return Err(syn::Error::new(
                here,
                "`Vec<String>` fields with `argv` must use `#[job(csv)]` or `#[job(repeat)]`",
            ));
        }
        if ty == FieldType::Bool {
            return Err(syn::Error::new(
                here,
                "bool fields with `argv` must use `#[job(flag)]`, `#[job(flag, invert)]` or \
                 `#[job(valued)]`",
            ));
        }
    }

    // Numeric bounds.
    for (key, bound) in [("min", a.min), ("max", a.max)] {
        let Some((bound, span)) = bound else { continue };
        if ty.is_float() {
            continue;
        }
        if ty.is_int() {
            if bound.as_whole_i64().is_none() {
                return Err(syn::Error::new(
                    span,
                    format!("`{key}` on an integer field must be a whole number that fits `i64`"),
                ));
            }
        } else {
            return Err(syn::Error::new(
                span,
                format!("`{key}` only applies to numeric fields"),
            ));
        }
    }
    if let (Some((lo, _)), Some((hi, span))) = (a.min, a.max)
        && lo.as_f64() > hi.as_f64()
    {
        return Err(syn::Error::new(span, "`max` is below `min`"));
    }

    // Kind override.
    if let Some((kind, span)) = &a.kind_override {
        match kind.as_str() {
            "text" | "toggle" | "model_picker" | "dataset_picker" | "path" | "read_only" => {}
            "number" if ty.is_float() => {}
            "number" => {
                return Err(syn::Error::new(
                    *span,
                    "kind = \"number\" needs an `f32` / `f64` field (or `Option` of one)",
                ));
            }
            "integer" if ty.is_int() => {}
            "integer" => {
                return Err(syn::Error::new(
                    *span,
                    "kind = \"integer\" needs an integer field (or `Option` of one)",
                ));
            }
            "enum" => {
                if a.enum_options.is_none() {
                    return Err(syn::Error::new(
                        *span,
                        "kind = \"enum\" requires `enum_options = [...]`",
                    ));
                }
                if !ty.is_string_like() {
                    return Err(syn::Error::new(
                        *span,
                        "kind = \"enum\" needs a `String` or `Option<String>` field",
                    ));
                }
            }
            other => {
                return Err(syn::Error::new(
                    *span,
                    format!(
                        "unknown kind override `{other}`; expected one of text, toggle, number, \
                         integer, model_picker, dataset_picker, path, read_only, enum"
                    ),
                ));
            }
        }
    }
    if a.enum_options.is_some() && a.kind_override.as_ref().map(|(k, _)| k.as_str()) != Some("enum")
    {
        return Err(syn::Error::new(
            here,
            "`enum_options` only applies together with `kind = \"enum\"`",
        ));
    }

    // `required`.
    if let Some(span) = a.required
        && !(ty.is_string_like() || ty.is_option() || ty == FieldType::VecString)
    {
        return Err(syn::Error::new(
            span,
            "`required` is only supported on `String`, `Vec<String>` and `Option<_>` fields",
        ));
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Descriptor emission
// ---------------------------------------------------------------------------

fn emit_descriptor(field: &ParsedField) -> syn::Result<TokenStream2> {
    let attrs = &field.attrs;
    let name_str = &field.name_str;

    let label = attrs.label.clone().ok_or_else(|| {
        syn::Error::new(
            field.name.span(),
            format!("field `{name_str}` needs `#[job(label = \"...\")]`"),
        )
    })?;
    let group = attrs.group.clone().ok_or_else(|| {
        syn::Error::new(
            field.name.span(),
            format!("field `{name_str}` needs `#[job(group = \"...\")]`"),
        )
    })?;

    let help_lit = match &attrs.help {
        Some(s) => quote! { ::std::option::Option::Some(#s) },
        None => quote! { ::std::option::Option::None },
    };
    let argv_lit = match &attrs.argv {
        Some(s) => quote! { ::std::option::Option::Some(#s) },
        None => quote! { ::std::option::Option::None },
    };

    let kind_tokens = emit_field_kind(field)?;
    let default_tokens = emit_default_value(field);
    let argv_optional = field.ty.is_option() || attrs.flag.is_some();
    let required = attrs.required.is_some();
    let shape = if field.ty == FieldType::VecString {
        quote! { ::pmetal_core::FieldShape::List }
    } else if field.ty.is_option() {
        quote! { ::pmetal_core::FieldShape::Optional }
    } else {
        quote! { ::pmetal_core::FieldShape::Scalar }
    };

    Ok(quote! {
        ::pmetal_core::FieldDescriptor {
            name: #name_str,
            label: #label,
            help: #help_lit,
            group: #group,
            kind: #kind_tokens,
            default: #default_tokens,
            required: #required,
            argv: #argv_lit,
            argv_optional: #argv_optional,
            shape: #shape,
        }
    })
}

fn emit_field_kind(field: &ParsedField) -> syn::Result<TokenStream2> {
    let attrs = &field.attrs;
    if let Some((k, _)) = &attrs.kind_override {
        // `check_attr_consistency` has already rejected unknown names and
        // type mismatches.
        return Ok(match k.as_str() {
            "text" => quote! { ::pmetal_core::FieldKind::Text },
            "toggle" => quote! { ::pmetal_core::FieldKind::Toggle },
            "model_picker" => quote! { ::pmetal_core::FieldKind::ModelPicker },
            "dataset_picker" => quote! { ::pmetal_core::FieldKind::DatasetPicker },
            "path" => quote! { ::pmetal_core::FieldKind::Path },
            "read_only" => quote! { ::pmetal_core::FieldKind::ReadOnly },
            "enum" => {
                let opts = attrs.enum_options.iter().flatten();
                quote! {
                    ::pmetal_core::FieldKind::Enum {
                        options: &[#(#opts),*],
                    }
                }
            }
            "number" => emit_number_kind(field),
            "integer" => emit_integer_kind(field),
            other => unreachable!("kind override `{other}` passed validation"),
        });
    }

    if field.ty.is_string_like() || field.ty == FieldType::VecString {
        Ok(quote! { ::pmetal_core::FieldKind::Text })
    } else if field.ty.is_bool() {
        Ok(quote! { ::pmetal_core::FieldKind::Toggle })
    } else if field.ty.is_float() {
        Ok(emit_number_kind(field))
    } else if field.ty.is_int() {
        Ok(emit_integer_kind(field))
    } else {
        Err(syn::Error::new(
            field.name.span(),
            "could not infer FieldKind; add `#[job(kind = \"...\")]` override",
        ))
    }
}

fn emit_number_kind(field: &ParsedField) -> TokenStream2 {
    let min = field.attrs.min.map_or(f64::MIN, |(b, _)| b.as_f64());
    let max = field.attrs.max.map_or(f64::MAX, |(b, _)| b.as_f64());
    quote! {
        ::pmetal_core::FieldKind::Number { min: #min, max: #max }
    }
}

fn emit_integer_kind(field: &ParsedField) -> TokenStream2 {
    let (default_min, default_max) = int_descriptor_bounds(field.ty);
    let min = int_bound(field.attrs.min).unwrap_or(default_min);
    let max = int_bound(field.attrs.max).unwrap_or(default_max);
    quote! {
        ::pmetal_core::FieldKind::Integer { min: #min, max: #max }
    }
}

/// A validated integer bound; `check_attr_consistency` guarantees it is whole
/// and fits `i64` for every integer field.
fn int_bound(bound: Option<(Bound, Span)>) -> Option<i64> {
    bound.and_then(|(b, _)| b.as_whole_i64())
}

fn int_descriptor_bounds(ty: FieldType) -> (i64, i64) {
    match ty {
        FieldType::I8 | FieldType::OptionI8 => (i8::MIN as i64, i8::MAX as i64),
        FieldType::I16 | FieldType::OptionI16 => (i16::MIN as i64, i16::MAX as i64),
        FieldType::I32 | FieldType::OptionI32 => (i32::MIN as i64, i32::MAX as i64),
        FieldType::I64 | FieldType::OptionI64 => (i64::MIN, i64::MAX),
        FieldType::U8 | FieldType::OptionU8 => (0, u8::MAX as i64),
        FieldType::U16 | FieldType::OptionU16 => (0, u16::MAX as i64),
        FieldType::U32 | FieldType::OptionU32 => (0, u32::MAX as i64),
        FieldType::U64 | FieldType::OptionU64 | FieldType::Usize | FieldType::OptionUsize => {
            (0, i64::MAX)
        }
        _ => (0, i64::MAX),
    }
}

fn emit_default_value(field: &ParsedField) -> TokenStream2 {
    let attrs = &field.attrs;
    if let Some(s) = &attrs.default_str {
        quote! { ::pmetal_core::DefaultValue::Str(#s) }
    } else if let Some(i) = attrs.default_int {
        quote! { ::pmetal_core::DefaultValue::Int(#i) }
    } else if let Some(f) = attrs.default_float {
        quote! { ::pmetal_core::DefaultValue::Float(#f) }
    } else if let Some(b) = attrs.default_bool {
        quote! { ::pmetal_core::DefaultValue::Bool(#b) }
    } else {
        quote! { ::pmetal_core::DefaultValue::None }
    }
}

// ---------------------------------------------------------------------------
// to_argv emission
// ---------------------------------------------------------------------------

/// Emit the `to_argv` statements for a field that has an `argv` flag. The
/// attribute/type combinations were validated by `check_attr_consistency`.
fn emit_argv(field: &ParsedField) -> TokenStream2 {
    let attrs = &field.attrs;
    let name = &field.name;
    let argv = attrs
        .argv
        .as_ref()
        .expect("argv presence is filtered by caller");

    // Bool flags.
    if attrs.flag.is_some() {
        let test = if attrs.invert.is_some() {
            quote! { !self.#name }
        } else {
            quote! { self.#name }
        };
        return quote! {
            if #test {
                args.push(::std::string::ToString::to_string(#argv));
            }
        };
    }

    // A setting is passed when it differs from the default, or always if the
    // field says it must be stated.
    let differs = if attrs.always.is_some() {
        quote! { true }
    } else {
        quote! { self.#name != default.#name }
    };

    // Bool that takes `true` / `false`, so a default of true can be turned off.
    if attrs.valued.is_some() {
        return quote! {
            if #differs {
                args.push(::std::string::ToString::to_string(#argv));
                args.push(::std::string::ToString::to_string(&self.#name));
            }
        };
    }

    // Repeated flag, one occurrence per item.
    if attrs.repeat.is_some() {
        return quote! {
            for item in &self.#name {
                args.push(::std::string::ToString::to_string(#argv));
                args.push(::std::clone::Clone::clone(item));
            }
        };
    }

    // Csv vec
    if attrs.csv.is_some() {
        return quote! {
            if !self.#name.is_empty() {
                args.push(::std::string::ToString::to_string(#argv));
                args.push(self.#name.join(","));
            }
        };
    }

    // Option<T>
    if field.ty.is_option() {
        let push_value = if matches!(field.ty, FieldType::OptionString) {
            quote! { ::std::clone::Clone::clone(v) }
        } else {
            quote! { ::std::string::ToString::to_string(v) }
        };
        return quote! {
            if let ::std::option::Option::Some(v) = &self.#name {
                args.push(::std::string::ToString::to_string(#argv));
                args.push(#push_value);
            }
        };
    }

    // Plain string
    if field.ty.is_string_like() {
        return quote! {
            if !self.#name.is_empty() && #differs {
                args.push(::std::string::ToString::to_string(#argv));
                args.push(::std::clone::Clone::clone(&self.#name));
            }
        };
    }

    // Plain numeric (or any `Display` type with an explicit kind override).
    quote! {
        if #differs {
            args.push(::std::string::ToString::to_string(#argv));
            args.push(::std::string::ToString::to_string(&self.#name));
        }
    }
}

// ---------------------------------------------------------------------------
// validate emission
// ---------------------------------------------------------------------------

fn field_error(name_str: &str, message: TokenStream2) -> TokenStream2 {
    quote! {
        errs.push(::pmetal_core::FieldError::new(#name_str, #message));
    }
}

fn emit_validate(field: &ParsedField) -> TokenStream2 {
    let attrs = &field.attrs;
    let name = &field.name;
    let name_str = &field.name_str;
    let mut checks = Vec::<TokenStream2>::new();

    if attrs.required.is_some() {
        let missing = match field.ty {
            FieldType::String | FieldType::VecString => quote! { self.#name.is_empty() },
            FieldType::OptionString => {
                quote! { self.#name.as_ref().is_none_or(|s| s.is_empty()) }
            }
            _ => quote! { self.#name.is_none() },
        };
        let err = field_error(name_str, quote! { "required" });
        checks.push(quote! {
            if #missing {
                #err
            }
        });
    }

    // Numeric range validation
    if field.ty.is_float() {
        let value_expr = if field.ty.is_option() {
            quote! { self.#name }
        } else {
            quote! { ::std::option::Option::Some(self.#name) }
        };
        let not_finite = field_error(name_str, quote! { "must be a finite number" });
        let min_check = attrs.min.map(|(b, _)| {
            let m = b.as_f64();
            let err = field_error(name_str, quote! { ::std::format!("must be ≥ {}", #m) });
            quote! { if (v as f64) < #m { #err } }
        });
        let max_check = attrs.max.map(|(b, _)| {
            let m = b.as_f64();
            let err = field_error(name_str, quote! { ::std::format!("must be ≤ {}", #m) });
            quote! { if (v as f64) > #m { #err } }
        });
        checks.push(quote! {
            if let ::std::option::Option::Some(v) = #value_expr {
                if !(v as f64).is_finite() {
                    #not_finite
                } else {
                    #min_check
                    #max_check
                }
            }
        });
    } else if field.ty.is_int() && (attrs.min.is_some() || attrs.max.is_some()) {
        let value_expr = if field.ty.is_option() {
            quote! { self.#name }
        } else {
            quote! { ::std::option::Option::Some(self.#name) }
        };
        let min_check = int_bound(attrs.min).map(|m| {
            let m = i128::from(m);
            let err = field_error(name_str, quote! { ::std::format!("must be ≥ {}", #m) });
            quote! { if (v as i128) < #m { #err } }
        });
        let max_check = int_bound(attrs.max).map(|m| {
            let m = i128::from(m);
            let err = field_error(name_str, quote! { ::std::format!("must be ≤ {}", #m) });
            quote! { if (v as i128) > #m { #err } }
        });
        checks.push(quote! {
            if let ::std::option::Option::Some(v) = #value_expr {
                #min_check
                #max_check
            }
        });
    }

    // Enum option check
    if attrs.kind_override.as_ref().map(|(k, _)| k.as_str()) == Some("enum")
        && let Some(opts) = &attrs.enum_options
    {
        let opts_arr = opts.iter();
        let value_expr = if matches!(field.ty, FieldType::OptionString) {
            quote! { self.#name.as_deref() }
        } else {
            quote! { ::std::option::Option::Some(self.#name.as_str()) }
        };
        let opts_join = opts.join(", ");
        let err = field_error(
            name_str,
            quote! { ::std::format!("must be one of: {}", #opts_join) },
        );
        checks.push(quote! {
            const ALLOWED: &[&str] = &[#(#opts_arr),*];
            if let ::std::option::Option::Some(v) = #value_expr {
                if !v.is_empty() && !ALLOWED.contains(&v) {
                    #err
                }
            }
        });
    }

    // Block-scope each field's checks so locals (`ALLOWED`, `v`) don't collide.
    quote! {
        {
            #(#checks)*
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ty(src: &str) -> Type {
        syn::parse_str(src).expect("valid type")
    }

    /// Expand a struct source: the expansion text, or all error messages.
    fn expand_src(src: &str) -> Result<String, String> {
        let input: DeriveInput = syn::parse_str(src).expect("valid item");
        expand(&input).map(|t| t.to_string()).map_err(|e| {
            e.into_iter()
                .map(|e| e.to_string())
                .collect::<Vec<_>>()
                .join(" | ")
        })
    }

    fn expect_err(src: &str, needle: &str) {
        match expand_src(src) {
            Ok(_) => panic!("expected an error containing {needle:?}, but it expanded"),
            Err(msg) => assert!(msg.contains(needle), "error {msg:?} lacks {needle:?}"),
        }
    }

    const SPEC: &str = r#"#[spec(kind = "Train", subcommand = "t")]"#;
    const LG: &str = r#"label = "L", group = "G""#;

    fn with_field(attrs: &str, ty: &str) -> String {
        format!("{SPEC} struct S {{ #[job({attrs})] f: {ty} }}")
    }

    #[test]
    fn the_event_log_and_the_output_fields_are_part_of_the_impl() {
        let plain = expand_src(&format!(
            "{SPEC} struct S {{ #[job(skip_descriptor)] f: u8 }}"
        ))
        .expect("expands");
        assert!(plain.contains("REPORTS_EVENTS : bool = false"), "{plain}");
        assert!(
            plain.contains("OUTPUT_FIELD : :: std :: option :: Option < & 'static str > = :: std :: option :: Option :: None"),
            "{plain}"
        );

        let src = r#"#[spec(kind = "Train", subcommand = "t", events)]
            struct S {
                #[job(label = "O", group = "G", argv = "--output")] out: String,
                #[job(skip_descriptor, argv = "--log-metrics")] m: Option<String>,
            }"#;
        let out = expand_src(src).expect("expands");
        assert!(out.contains("REPORTS_EVENTS : bool = true"), "{out}");
        assert!(out.contains("Option :: Some (\"out\")"), "{out}");
        assert!(out.contains("Option :: Some (\"m\")"), "{out}");

        expect_err(
            r#"#[spec(kind = "Train", subcommand = "t", events, events)] struct S { #[job(skip_descriptor)] f: u8 }"#,
            "more than once",
        );
    }

    #[test]
    fn vec_classification_requires_string_inner_type() {
        assert_eq!(classify_type(&ty("Vec<String>")), FieldType::VecString);
        assert_eq!(
            classify_type(&ty("std::vec::Vec<String>")),
            FieldType::VecString
        );
        assert_eq!(classify_type(&ty("Vec<u32>")), FieldType::Other);
    }

    #[test]
    fn signed_integer_descriptors_keep_signed_minimums() {
        assert_eq!(
            int_descriptor_bounds(FieldType::I32),
            (i32::MIN as i64, i32::MAX as i64)
        );
        assert_eq!(
            int_descriptor_bounds(FieldType::OptionI64),
            (i64::MIN, i64::MAX)
        );
        assert_eq!(int_descriptor_bounds(FieldType::U16), (0, u16::MAX as i64));
    }

    #[test]
    fn invalid_job_kind_text_is_an_error_not_a_panic() {
        // `format_ident!` panics on text that is not an identifier.
        for bad in ["Train Job", "", "1Train", "Tr-ain"] {
            let src = format!(
                r#"#[spec(kind = "{bad}", subcommand = "t")] struct S {{ #[job({LG})] f: String }}"#
            );
            expect_err(&src, "is not a `JobKind` variant name");
        }
    }

    #[test]
    fn non_finite_float_literals_are_rejected() {
        expect_err(&with_field(&format!("{LG}, max = 1e999"), "f64"), "finite");
    }

    #[test]
    fn duplicate_keys_are_rejected() {
        expect_err(
            &with_field(r#"label = "A", label = "B", group = "G""#, "String"),
            "`label` is set more than once",
        );
        expect_err(
            &format!(
                r#"#[spec(kind = "Train", kind = "Eval", subcommand = "t")] struct S {{ #[job({LG})] f: String }}"#
            ),
            "`kind` is set more than once",
        );
    }

    #[test]
    fn silently_inert_attribute_combinations_are_rejected() {
        let l = LG;
        expect_err(
            &with_field(&format!(r#"{l}, argv = "--f", invert"#), "bool"),
            "`invert` only applies together with `flag`",
        );
        expect_err(
            &with_field(&format!("{l}, flag"), "bool"),
            "`flag` needs `argv",
        );
        expect_err(
            &with_field(&format!(r#"{l}, argv = "--f", flag"#), "Option<bool>"),
            "`flag` may only be applied to `bool`",
        );
        expect_err(
            &with_field(&format!(r#"{l}, argv = "--f""#), "Vec<String>"),
            "must use `#[job(csv)]`",
        );
        expect_err(&with_field(&format!("{l}, min = 1"), "String"), "numeric");
        expect_err(
            &with_field(&format!(r#"{l}, enum_options = ["a"]"#), "String"),
            "only applies together with `kind = \"enum\"`",
        );
        expect_err(
            &with_field(
                &format!(r#"{l}, kind = "enum", enum_options = ["a"]"#),
                "u32",
            ),
            "needs a `String`",
        );
        expect_err(
            &with_field(&format!(r#"{l}, default = "a", default_int = 1"#), "String"),
            "at most one of",
        );
        expect_err(&with_field(&format!("{l}, required"), "u32"), "`required`");
        expect_err(
            &with_field(&format!("{l}, min = 3, max = 1"), "u32"),
            "`max` is below `min`",
        );
        expect_err(
            &with_field(&format!("{l}, min = 0.5"), "u32"),
            "whole number",
        );
        expect_err(
            &with_field(&format!(r#"{l}, kind = "number""#), "String"),
            "needs an `f32`",
        );
        expect_err(
            &with_field(&format!(r#"{l}, kind = "nope""#), "String"),
            "unknown kind override",
        );
    }

    #[test]
    fn every_problem_in_a_struct_is_reported_at_once() {
        let src = format!(
            r#"{SPEC} struct S {{
                #[job({LG}, min = 1)] a: String,
                #[job({LG}, invert)] b: bool,
            }}"#
        );
        let msg = expand_src(&src).unwrap_err();
        assert!(msg.contains("numeric") && msg.contains("`invert`"), "{msg}");
    }

    #[test]
    fn integer_bounds_keep_full_i64_precision() {
        // 2^53 + 1 is not representable as f64.
        let out = expand_src(&with_field(&format!("{LG}, max = 9007199254740993"), "i64"))
            .expect("expands");
        assert!(out.contains("9007199254740993"), "{out}");
    }

    #[test]
    fn rejects_non_struct_and_tuple_inputs() {
        expect_err(&format!("{SPEC} enum E {{ A }}"), "only supports structs");
        expect_err(&format!("{SPEC} struct T(String);"), "named fields");
    }

    #[test]
    fn generics_are_carried_onto_the_impl() {
        let src = format!("{SPEC} struct G<T> where T: Clone {{ #[job(skip_descriptor)] f: T }}");
        let out = expand_src(&src).expect("expands");
        assert!(
            out.contains("impl < T > :: pmetal_core :: JobFields for G < T > where T : Clone"),
            "{out}"
        );
    }
}
