//! Checking arguments against the input schema a tool advertised.
//!
//! A client may or may not validate before it calls, and a model-written call is
//! often off by a little: a string for a number, an option that does not exist, a
//! value out of range. The server holds every call to the schema it published,
//! and the error names the parameter and what was wrong so the model can fix it.
//!
//! This covers the keywords the server's schemas use: `type` (one name or a
//! list), `enum`, `const`, `anyOf`, `oneOf`, `minimum`, `maximum`, `items`,
//! `minItems`, and for an object `properties`, `required` and
//! `additionalProperties: false`.

use serde_json::{Map, Value};

/// Check `args` against the object schema `schema`.
pub fn check_arguments(tool: &str, schema: &Value, args: &Value) -> Result<(), String> {
    let empty = Map::new();
    let args = match args {
        Value::Null => &empty,
        Value::Object(map) => map,
        other => {
            return Err(format!(
                "the arguments must be a JSON object, got {}",
                describe(other)
            ));
        }
    };
    let properties = schema
        .get("properties")
        .and_then(Value::as_object)
        .unwrap_or(&empty);
    if schema.get("additionalProperties") == Some(&Value::Bool(false)) {
        for key in args.keys() {
            if !properties.contains_key(key) {
                return Err(format!(
                    "unknown parameter `{key}`; {tool} takes: {}",
                    properties.keys().cloned().collect::<Vec<_>>().join(", ")
                ));
            }
        }
    }
    let missing: Vec<&str> = schema
        .get("required")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .filter(|name| args.get(*name).is_none_or(Value::is_null))
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "missing required parameter{}: {}",
            if missing.len() == 1 { "" } else { "s" },
            missing.join(", ")
        ));
    }
    for (name, value) in args {
        // A null is an argument left out.
        if value.is_null() {
            continue;
        }
        if let Some(property) = properties.get(name) {
            check_value(name, value, property)?;
        }
    }
    Ok(())
}

/// Check one value against its schema.
pub fn check_value(name: &str, value: &Value, schema: &Value) -> Result<(), String> {
    if let Some(forms) = schema
        .get("anyOf")
        .or_else(|| schema.get("oneOf"))
        .and_then(Value::as_array)
    {
        if forms
            .iter()
            .any(|form| check_value(name, value, form).is_ok())
        {
            return Ok(());
        }
        let mut options = Vec::new();
        for form in forms {
            constants(form, &mut options);
        }
        return Err(if options.is_empty() {
            format!("`{name}` is not any of the forms allowed: {value}")
        } else {
            format!(
                "`{name}` must be one of {}, got {value}",
                options.join(", ")
            )
        });
    }
    if let Some(expected) = schema.get("const") {
        if expected != value {
            return Err(format!("`{name}` must be {expected}, got {value}"));
        }
    }

    if let Some(types) = schema.get("type") {
        let names: Vec<&str> = match types {
            Value::String(s) => vec![s.as_str()],
            Value::Array(a) => a.iter().filter_map(Value::as_str).collect(),
            _ => Vec::new(),
        };
        if !names.is_empty() && !names.iter().any(|t| has_type(value, t)) {
            let want = names
                .iter()
                .filter(|t| **t != "null")
                .map(|t| article(t))
                .collect::<Vec<_>>()
                .join(" or ");
            return Err(format!("`{name}` must be {want}, got {}", describe(value)));
        }
    }
    if let Some(options) = schema.get("enum").and_then(Value::as_array) {
        if !options.contains(value) {
            return Err(format!(
                "`{name}` must be one of {}, got {value}",
                options
                    .iter()
                    .map(|o| o.as_str().map_or_else(|| o.to_string(), str::to_owned))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
    }
    if value.is_number() {
        if let Some(number) = value.as_f64() {
            if let Some(min) = schema.get("minimum").and_then(Value::as_f64) {
                if number < min {
                    return Err(format!("`{name}` must be at least {min}, got {value}"));
                }
            }
            if let Some(max) = schema.get("maximum").and_then(Value::as_f64) {
                if number > max {
                    return Err(format!("`{name}` must be at most {max}, got {value}"));
                }
            }
        }
    }
    if let (Some(object), Some(properties)) = (
        value.as_object(),
        schema.get("properties").and_then(Value::as_object),
    ) {
        for required in schema
            .get("required")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
        {
            if !object.contains_key(required) {
                return Err(format!("`{name}` is missing `{required}`"));
            }
        }
        for (key, member) in object {
            if let Some(member_schema) = properties.get(key) {
                check_value(&format!("{name}.{key}"), member, member_schema)?;
            }
        }
    }
    if let Some(items) = value.as_array() {
        if let Some(min) = schema.get("minItems").and_then(Value::as_u64) {
            if (items.len() as u64) < min {
                return Err(format!(
                    "`{name}` needs at least {min} item{}",
                    if min == 1 { "" } else { "s" }
                ));
            }
        }
        if let Some(item_schema) = schema.get("items") {
            for (i, item) in items.iter().enumerate() {
                check_value(&format!("{name}[{i}]"), item, item_schema)?;
            }
        }
    }
    Ok(())
}

/// The `const` values a schema allows, looking through `anyOf` and `oneOf`.
fn constants(schema: &Value, out: &mut Vec<String>) {
    if let Some(c) = schema.get("const") {
        out.push(c.as_str().map_or_else(|| c.to_string(), str::to_owned));
    }
    for key in ["anyOf", "oneOf"] {
        for form in schema
            .get(key)
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            constants(form, out);
        }
    }
}

fn has_type(value: &Value, name: &str) -> bool {
    match name {
        "string" => value.is_string(),
        "boolean" => value.is_boolean(),
        "integer" => value.is_i64() || value.is_u64(),
        "number" => value.is_number(),
        "array" => value.is_array(),
        "object" => value.is_object(),
        "null" => value.is_null(),
        _ => true,
    }
}

fn article(name: &str) -> &'static str {
    match name {
        "string" => "a string",
        "boolean" => "a boolean",
        "integer" => "an integer",
        "number" => "a number",
        "array" => "an array",
        "object" => "an object",
        _ => "a value",
    }
}

fn describe(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn schema() -> Value {
        json!({
            "type": "object",
            "additionalProperties": false,
            "required": ["model"],
            "properties": {
                "model": {"type": "string"},
                "limit": {"type": ["integer", "null"], "minimum": 1, "maximum": 100},
                "ratio": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "sort": {"anyOf": [
                    {"oneOf": [{"type": "string", "const": "downloads"}, {"type": "string", "const": "likes"}]},
                    {"type": "null"}
                ]},
                "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1}
            }
        })
    }

    #[test]
    fn valid_arguments_pass_and_nulls_count_as_left_out() {
        let ok = json!({"model": "m", "limit": 5, "ratio": 1, "sort": "likes", "tags": ["a"]});
        assert!(check_arguments("t", &schema(), &ok).is_ok());
        assert!(
            check_arguments(
                "t",
                &schema(),
                &json!({"model": "m", "limit": null, "sort": null})
            )
            .is_ok()
        );
    }

    #[test]
    fn each_kind_of_mistake_names_the_parameter() {
        let s = schema();
        for (args, needle) in [
            (json!({}), "missing required parameter: model"),
            (json!({"model": null}), "model"),
            (
                json!({"model": "m", "extra": 1}),
                "unknown parameter `extra`; t takes: ",
            ),
            (
                json!({"model": 5}),
                "`model` must be a string, got a number",
            ),
            (
                json!({"model": "m", "limit": "7"}),
                "`limit` must be an integer, got a string",
            ),
            (
                json!({"model": "m", "limit": 0}),
                "`limit` must be at least 1",
            ),
            (
                json!({"model": "m", "limit": 101}),
                "`limit` must be at most 100",
            ),
            (
                json!({"model": "m", "limit": 1.5}),
                "`limit` must be an integer",
            ),
            (
                json!({"model": "m", "ratio": 1.5}),
                "`ratio` must be at most 1",
            ),
            (
                json!({"model": "m", "sort": "newest"}),
                "`sort` must be one of downloads, likes, got \"newest\"",
            ),
            (
                json!({"model": "m", "tags": []}),
                "`tags` needs at least 1 item",
            ),
            (
                json!({"model": "m", "tags": ["a", 2]}),
                "`tags[1]` must be a string",
            ),
            (json!([1]), "must be a JSON object"),
        ] {
            let error = check_arguments("t", &s, &args).unwrap_err();
            assert!(error.contains(needle), "{args}: {error}");
        }
    }
}
