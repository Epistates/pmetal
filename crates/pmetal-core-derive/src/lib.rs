//! `#[derive(JobSpec)]` — derives `pmetal_core::JobFields` for a spec struct.
//!
//! See `pmetal_core::jobs::*` for usage examples.
//!
//! ## Spec-level attributes (`#[spec(...)]` on the struct)
//!
//! - `kind = "Train"` — required; the `pmetal_core::JobKind` variant.
//! - `subcommand = "train"` — required; the CLI subcommand string.
//!
//! ## Field-level attributes (`#[job(...)]` on each field)
//!
//! - `label = "Learning Rate"` — form label (required unless `skip_descriptor`).
//! - `group = "Training"` — form section (required unless `skip_descriptor`).
//! - `help = "..."` — optional extended help.
//! - `argv = "--learning-rate"` — CLI flag this field maps to (omit for form-only).
//! - `default = "2e-4"` / `default_int = 1` / `default_float = 1.0` /
//!   `default_bool = true` — typed defaults for descriptor display.
//!   For complex constants, use `default_const = path::TO::CONST` (verbatim path
//!   reference with no display value — descriptor `DefaultValue::None`).
//! - `kind = "model_picker" | "dataset_picker" | "path" | "read_only" | "enum"`
//!   — overrides type-inferred `pmetal_core::FieldKind`.
//! - `enum_options = ["none", "nf4", "fp4"]` — required when `kind = "enum"`.
//! - `min = 0.0`, `max = 1.0` — bounds for `Number`/`Integer` kinds
//!   (defaults: numeric type's full range).
//! - `flag` — `bool` field; emit `--flag` when `true`, omit when `false`.
//! - `invert` — pair with `flag`; emit `--flag` when `false`, omit when `true`
//!   (for `--no-foo` semantics).
//! - `csv` — `Vec<String>` field; emit `--flag a,b,c` joined with commas.
//! - `required` — non-empty / non-default validation.
//! - `skip_descriptor` — exclude from `field_descriptors()` (CLI-only field).
//! - `skip_argv` — exclude from `to_argv()` (form-only field).

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{
    Attribute, Data, DeriveInput, Expr, ExprLit, Field, Fields, Ident, Lit, LitStr, Token, Type,
    parse_macro_input, punctuated::Punctuated, spanned::Spanned,
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

    let spec_meta = parse_spec_meta(&input.attrs, input.span())?;

    let fields = match &input.data {
        Data::Struct(s) => match &s.fields {
            Fields::Named(named) => named.named.iter().collect::<Vec<_>>(),
            _ => {
                return Err(syn::Error::new(
                    input.span(),
                    "JobSpec only supports structs with named fields",
                ));
            }
        },
        _ => {
            return Err(syn::Error::new(
                input.span(),
                "JobSpec only supports structs",
            ));
        }
    };

    let parsed: Vec<ParsedField> = fields
        .iter()
        .map(|f| parse_field(f))
        .collect::<syn::Result<Vec<_>>>()?;

    let descriptor_items = parsed
        .iter()
        .filter(|f| !f.attrs.skip_descriptor)
        .map(emit_descriptor)
        .collect::<syn::Result<Vec<_>>>()?;

    let argv_items = parsed
        .iter()
        .filter(|f| !f.attrs.skip_argv && f.attrs.argv.is_some())
        .map(emit_argv)
        .collect::<syn::Result<Vec<_>>>()?;

    let validate_items = parsed
        .iter()
        .filter(|f| !f.attrs.skip_descriptor)
        .map(emit_validate)
        .collect::<syn::Result<Vec<_>>>()?;

    let kind_ident = &spec_meta.kind;
    let subcommand = &spec_meta.subcommand;

    Ok(quote! {
        impl ::pmetal_core::JobFields for #ident {
            fn field_descriptors() -> &'static [::pmetal_core::FieldDescriptor] {
                static DESCRIPTORS: &[::pmetal_core::FieldDescriptor] = &[
                    #(#descriptor_items),*
                ];
                DESCRIPTORS
            }

            fn to_argv(&self) -> ::std::vec::Vec<::std::string::String> {
                let mut args: ::std::vec::Vec<::std::string::String> = ::std::vec::Vec::new();
                #(#argv_items)*
                args
            }

            fn validate_descriptors(&self) -> ::std::vec::Vec<::pmetal_core::FieldError> {
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
}

fn parse_spec_meta(attrs: &[Attribute], span: proc_macro2::Span) -> syn::Result<SpecMeta> {
    let mut kind: Option<Ident> = None;
    let mut subcommand: Option<LitStr> = None;

    for attr in attrs.iter().filter(|a| a.path().is_ident("spec")) {
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("kind") {
                let value = meta.value()?;
                let lit: LitStr = value.parse()?;
                kind = Some(format_ident!("{}", lit.value(), span = lit.span()));
                Ok(())
            } else if meta.path.is_ident("subcommand") {
                let value = meta.value()?;
                let lit: LitStr = value.parse()?;
                subcommand = Some(lit);
                Ok(())
            } else {
                Err(meta.error("unknown #[spec(...)] key — expected `kind` or `subcommand`"))
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
    Ok(SpecMeta { kind, subcommand })
}

// ---------------------------------------------------------------------------
// Field-level parsing
// ---------------------------------------------------------------------------

#[derive(Default, Clone)]
struct FieldAttrs {
    label: Option<String>,
    help: Option<String>,
    group: Option<String>,
    argv: Option<String>,

    // Default-value variants (mutually exclusive at the descriptor level — first one wins).
    default_str: Option<String>,
    default_int: Option<i64>,
    default_float: Option<f64>,
    default_bool: Option<bool>,
    /// `default_const = path::TO::CONST` — the value is referenced verbatim
    /// (no descriptor display). Used by `Default::default()` only via the
    /// surrounding `#[serde(default = "...")]` or `Spec::default()`.
    default_const_path: Option<syn::Path>,

    // Kind override.
    kind_override: Option<String>,
    enum_options: Option<Vec<String>>,
    min: Option<f64>,
    max: Option<f64>,

    // Argv style.
    flag: bool,
    invert: bool,
    csv: bool,

    required: bool,
    skip_descriptor: bool,
    skip_argv: bool,
}

struct ParsedField {
    name: Ident,
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
    let name_str = name.to_string();
    let ty = classify_type(&field.ty);
    let attrs = parse_field_attrs(&field.attrs, field.span())?;
    Ok(ParsedField {
        name,
        name_str,
        ty,
        attrs,
    })
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

fn parse_field_attrs(attrs: &[Attribute], _span: proc_macro2::Span) -> syn::Result<FieldAttrs> {
    let mut out = FieldAttrs::default();

    for attr in attrs.iter().filter(|a| a.path().is_ident("job")) {
        attr.parse_nested_meta(|meta| {
            let path = &meta.path;
            if path.is_ident("label") {
                out.label = Some(parse_str_value(&meta)?);
            } else if path.is_ident("help") {
                out.help = Some(parse_str_value(&meta)?);
            } else if path.is_ident("group") {
                out.group = Some(parse_str_value(&meta)?);
            } else if path.is_ident("argv") {
                out.argv = Some(parse_str_value(&meta)?);
            } else if path.is_ident("default") {
                out.default_str = Some(parse_str_value(&meta)?);
            } else if path.is_ident("default_int") {
                out.default_int = Some(parse_int_value(&meta)?);
            } else if path.is_ident("default_float") {
                out.default_float = Some(parse_float_value(&meta)?);
            } else if path.is_ident("default_bool") {
                out.default_bool = Some(parse_bool_value(&meta)?);
            } else if path.is_ident("default_const") {
                let value = meta.value()?;
                out.default_const_path = Some(value.parse()?);
            } else if path.is_ident("kind") {
                out.kind_override = Some(parse_str_value(&meta)?);
            } else if path.is_ident("enum_options") {
                let value = meta.value()?;
                let content;
                syn::bracketed!(content in value);
                let opts: Punctuated<LitStr, Token![,]> = Punctuated::parse_terminated(&content)?;
                out.enum_options = Some(opts.iter().map(|s| s.value()).collect());
            } else if path.is_ident("min") {
                out.min = Some(parse_float_value(&meta)?);
            } else if path.is_ident("max") {
                out.max = Some(parse_float_value(&meta)?);
            } else if path.is_ident("flag") {
                out.flag = true;
            } else if path.is_ident("invert") {
                out.invert = true;
            } else if path.is_ident("csv") {
                out.csv = true;
            } else if path.is_ident("required") {
                out.required = true;
            } else if path.is_ident("skip_descriptor") {
                out.skip_descriptor = true;
            } else if path.is_ident("skip_argv") {
                out.skip_argv = true;
            } else {
                return Err(meta.error("unknown #[job(...)] key"));
            }
            Ok(())
        })?;
    }

    Ok(out)
}

fn parse_str_value(meta: &syn::meta::ParseNestedMeta<'_>) -> syn::Result<String> {
    let value = meta.value()?;
    let lit: LitStr = value.parse()?;
    Ok(lit.value())
}

fn parse_int_value(meta: &syn::meta::ParseNestedMeta<'_>) -> syn::Result<i64> {
    let value = meta.value()?;
    let expr: Expr = value.parse()?;
    expr_to_i64(&expr)
}

fn parse_float_value(meta: &syn::meta::ParseNestedMeta<'_>) -> syn::Result<f64> {
    let value = meta.value()?;
    let expr: Expr = value.parse()?;
    expr_to_f64(&expr)
}

fn parse_bool_value(meta: &syn::meta::ParseNestedMeta<'_>) -> syn::Result<bool> {
    let value = meta.value()?;
    let expr: Expr = value.parse()?;
    if let Expr::Lit(ExprLit {
        lit: Lit::Bool(b), ..
    }) = &expr
    {
        Ok(b.value)
    } else {
        Err(syn::Error::new(expr.span(), "expected `true` or `false`"))
    }
}

fn expr_to_i64(expr: &Expr) -> syn::Result<i64> {
    if let Expr::Lit(ExprLit {
        lit: Lit::Int(int), ..
    }) = expr
    {
        return int.base10_parse::<i64>();
    }
    // Allow `-N` literal.
    if let Expr::Unary(syn::ExprUnary {
        op: syn::UnOp::Neg(_),
        expr: inner,
        ..
    }) = expr
        && let Expr::Lit(ExprLit {
            lit: Lit::Int(int), ..
        }) = inner.as_ref()
    {
        let n: i64 = int.base10_parse()?;
        return Ok(-n);
    }
    Err(syn::Error::new(expr.span(), "expected integer literal"))
}

fn expr_to_f64(expr: &Expr) -> syn::Result<f64> {
    match expr {
        Expr::Lit(ExprLit {
            lit: Lit::Float(f), ..
        }) => f.base10_parse::<f64>(),
        Expr::Lit(ExprLit {
            lit: Lit::Int(i), ..
        }) => i.base10_parse::<i64>().map(|n| n as f64),
        Expr::Unary(syn::ExprUnary {
            op: syn::UnOp::Neg(_),
            expr: inner,
            ..
        }) => expr_to_f64(inner).map(|v| -v),
        _ => Err(syn::Error::new(expr.span(), "expected numeric literal")),
    }
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
    let argv_optional = field.ty.is_option() || attrs.flag;
    let required = attrs.required;

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
        }
    })
}

fn emit_field_kind(field: &ParsedField) -> syn::Result<TokenStream2> {
    let attrs = &field.attrs;
    if let Some(k) = &attrs.kind_override {
        return match k.as_str() {
            "text" => Ok(quote! { ::pmetal_core::FieldKind::Text }),
            "toggle" => Ok(quote! { ::pmetal_core::FieldKind::Toggle }),
            "model_picker" => Ok(quote! { ::pmetal_core::FieldKind::ModelPicker }),
            "dataset_picker" => Ok(quote! { ::pmetal_core::FieldKind::DatasetPicker }),
            "path" => Ok(quote! { ::pmetal_core::FieldKind::Path }),
            "read_only" => Ok(quote! { ::pmetal_core::FieldKind::ReadOnly }),
            "enum" => {
                let opts = attrs.enum_options.as_ref().ok_or_else(|| {
                    syn::Error::new(
                        field.name.span(),
                        "kind = \"enum\" requires `enum_options = [...]`",
                    )
                })?;
                let opts_iter = opts.iter().map(|s| quote! { #s });
                Ok(quote! {
                    ::pmetal_core::FieldKind::Enum {
                        options: &[#(#opts_iter),*],
                    }
                })
            }
            "number" => Ok(emit_number_kind(field)),
            "integer" => Ok(emit_integer_kind(field)),
            other => Err(syn::Error::new(
                field.name.span(),
                format!("unknown kind override `{other}`"),
            )),
        };
    }

    if field.ty.is_string_like() {
        Ok(quote! { ::pmetal_core::FieldKind::Text })
    } else if field.ty.is_bool() {
        Ok(quote! { ::pmetal_core::FieldKind::Toggle })
    } else if field.ty.is_float() {
        Ok(emit_number_kind(field))
    } else if field.ty.is_int() {
        Ok(emit_integer_kind(field))
    } else if matches!(field.ty, FieldType::VecString) {
        Ok(quote! { ::pmetal_core::FieldKind::Text })
    } else {
        Err(syn::Error::new(
            field.name.span(),
            "could not infer FieldKind; add `#[job(kind = \"...\")]` override",
        ))
    }
}

fn emit_number_kind(field: &ParsedField) -> TokenStream2 {
    let min = field.attrs.min.unwrap_or(f64::MIN);
    let max = field.attrs.max.unwrap_or(f64::MAX);
    quote! {
        ::pmetal_core::FieldKind::Number { min: #min, max: #max }
    }
}

fn emit_integer_kind(field: &ParsedField) -> TokenStream2 {
    let (default_min, default_max) = int_descriptor_bounds(field.ty);
    let min = field.attrs.min.map(|m| m as i64).unwrap_or(default_min);
    let max = field.attrs.max.map(|m| m as i64).unwrap_or(default_max);
    quote! {
        ::pmetal_core::FieldKind::Integer { min: #min, max: #max }
    }
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

fn emit_argv(field: &ParsedField) -> syn::Result<TokenStream2> {
    let attrs = &field.attrs;
    let name = &field.name;
    let argv = attrs
        .argv
        .as_ref()
        .expect("argv presence is filtered by caller");

    // Bool flags
    if attrs.flag {
        if !field.ty.is_bool() {
            return Err(syn::Error::new(
                name.span(),
                "`flag` may only be applied to `bool` fields",
            ));
        }
        let body = if attrs.invert {
            quote! {
                if !self.#name {
                    args.push(#argv.to_string());
                }
            }
        } else {
            quote! {
                if self.#name {
                    args.push(#argv.to_string());
                }
            }
        };
        return Ok(body);
    }

    // Csv vec
    if attrs.csv {
        if !matches!(field.ty, FieldType::VecString) {
            return Err(syn::Error::new(
                name.span(),
                "`csv` may only be applied to `Vec<String>` fields",
            ));
        }
        return Ok(quote! {
            if !self.#name.is_empty() {
                args.push(#argv.to_string());
                args.push(self.#name.join(","));
            }
        });
    }

    // Option<T>
    if field.ty.is_option() {
        let push_value = if matches!(field.ty, FieldType::OptionString) {
            quote! { v.clone() }
        } else {
            quote! { v.to_string() }
        };
        return Ok(quote! {
            if let ::std::option::Option::Some(v) = &self.#name {
                args.push(#argv.to_string());
                args.push(#push_value);
            }
        });
    }

    // Plain string
    if field.ty.is_string_like() {
        return Ok(quote! {
            if !self.#name.is_empty() {
                args.push(#argv.to_string());
                args.push(self.#name.clone());
            }
        });
    }

    // Bool without `flag` — skip (must be explicit)
    if field.ty.is_bool() {
        return Err(syn::Error::new(
            name.span(),
            "bool fields with `argv` must use `#[job(flag)]` or `#[job(flag, invert)]`",
        ));
    }

    // Plain numeric
    Ok(quote! {
        args.push(#argv.to_string());
        args.push(self.#name.to_string());
    })
}

// ---------------------------------------------------------------------------
// validate emission
// ---------------------------------------------------------------------------

fn emit_validate(field: &ParsedField) -> syn::Result<TokenStream2> {
    let attrs = &field.attrs;
    let name = &field.name;
    let name_str = &field.name_str;
    let mut checks = Vec::<TokenStream2>::new();

    if attrs.required && field.ty.is_string_like() {
        if matches!(field.ty, FieldType::OptionString) {
            checks.push(quote! {
                if self.#name.as_ref().map_or(true, |s| s.is_empty()) {
                    errs.push(::pmetal_core::FieldError::new(
                        #name_str,
                        "required",
                    ));
                }
            });
        } else {
            checks.push(quote! {
                if self.#name.is_empty() {
                    errs.push(::pmetal_core::FieldError::new(
                        #name_str,
                        "required",
                    ));
                }
            });
        }
    }

    // Numeric range validation
    if field.ty.is_float() {
        let value_expr = if field.ty.is_option() {
            quote! { self.#name }
        } else {
            quote! { ::std::option::Option::Some(self.#name) }
        };
        let min = attrs.min;
        let max = attrs.max;
        if min.is_some() || max.is_some() {
            let min_check = min
                .map(|m| {
                    quote! {
                        if (v as f64) < #m {
                            errs.push(::pmetal_core::FieldError::new(
                                #name_str,
                                format!("must be ≥ {}", #m),
                            ));
                        }
                    }
                })
                .unwrap_or_default();
            let max_check = max
                .map(|m| {
                    quote! {
                        if (v as f64) > #m {
                            errs.push(::pmetal_core::FieldError::new(
                                #name_str,
                                format!("must be ≤ {}", #m),
                            ));
                        }
                    }
                })
                .unwrap_or_default();
            checks.push(quote! {
                if let ::std::option::Option::Some(v) = #value_expr {
                    #min_check
                    #max_check
                }
            });
        }
    } else if field.ty.is_int() {
        let value_expr = if field.ty.is_option() {
            quote! { self.#name }
        } else {
            quote! { ::std::option::Option::Some(self.#name) }
        };
        let min = attrs.min.map(|m| m as i128);
        let max = attrs.max.map(|m| m as i128);
        if min.is_some() || max.is_some() {
            let min_check = min
                .map(|m| {
                    quote! {
                        if (v as i128) < #m {
                            errs.push(::pmetal_core::FieldError::new(
                                #name_str,
                                format!("must be ≥ {}", #m),
                            ));
                        }
                    }
                })
                .unwrap_or_default();
            let max_check = max
                .map(|m| {
                    quote! {
                        if (v as i128) > #m {
                            errs.push(::pmetal_core::FieldError::new(
                                #name_str,
                                format!("must be ≤ {}", #m),
                            ));
                        }
                    }
                })
                .unwrap_or_default();
            checks.push(quote! {
                if let ::std::option::Option::Some(v) = #value_expr {
                    #min_check
                    #max_check
                }
            });
        }
    }

    // Enum option check
    if field.attrs.kind_override.as_deref() == Some("enum")
        && let Some(opts) = &field.attrs.enum_options
        && field.ty.is_string_like()
    {
        let opts_arr = opts.iter().map(|s| quote! { #s });
        let value_expr = if matches!(field.ty, FieldType::OptionString) {
            quote! { self.#name.as_deref() }
        } else {
            quote! { ::std::option::Option::Some(self.#name.as_str()) }
        };
        let opts_join = opts.join(", ");
        checks.push(quote! {
            const ALLOWED: &[&str] = &[#(#opts_arr),*];
            if let ::std::option::Option::Some(v) = #value_expr {
                if !v.is_empty() && !ALLOWED.contains(&v) {
                    errs.push(::pmetal_core::FieldError::new(
                        #name_str,
                        format!("must be one of: {}", #opts_join),
                    ));
                }
            }
        });
    }

    // Block-scope each field's checks so locals (`ALLOWED`, `v`) don't collide.
    Ok(quote! {
        {
            #(#checks)*
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ty(src: &str) -> Type {
        syn::parse_str(src).expect("valid type")
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
}
