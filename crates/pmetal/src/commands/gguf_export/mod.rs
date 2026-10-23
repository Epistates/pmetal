//! What a GGUF file holds besides the tensor data: the architecture and how
//! its tensors are named and laid out, the model config and the tokenizer,
//! written from the Hugging Face files of a model.

mod arch;
mod config;
#[cfg(test)]
mod family_tests;
mod sentencepiece;
mod tokenizer;

pub(crate) use arch::ExportArch;
pub(crate) use config::add_hf_config_metadata;
pub(crate) use tokenizer::add_tokenizer_metadata;

/// The model's name for `general.name`: the hub repository's name, or the
/// directory's.
///
/// `model` is what the user gave (a hub id or a path), `resolved` the
/// directory it came to.
pub(crate) fn model_name(model: &str, resolved: &std::path::Path) -> String {
    let file_name = |path: &std::path::Path| {
        path.file_name()
            .and_then(|name| name.to_str())
            .map(str::to_string)
    };
    // A hub cache is `models--<org>--<name>/snapshots/<revision>`.
    let repository = resolved
        .parent()
        .filter(|snapshots| {
            snapshots
                .file_name()
                .is_some_and(|name| name == "snapshots")
        })
        .and_then(|snapshots| snapshots.parent())
        .and_then(file_name);
    if let Some(name) = repository
        .as_deref()
        .and_then(|repo| repo.strip_prefix("models--"))
        .and_then(|repo| repo.rsplit("--").next())
    {
        return name.to_string();
    }
    if !std::path::Path::new(model).exists()
        && let Some((_, name)) = model.rsplit_once('/')
    {
        return name.to_string();
    }
    file_name(resolved).unwrap_or_else(|| "model".to_string())
}

fn json_i64(json: &serde_json::Value, key: &str) -> Option<i64> {
    json.get(key).and_then(|value| value.as_i64())
}

fn json_u32(json: &serde_json::Value, key: &str) -> Option<u32> {
    json_i64(json, key).and_then(|value| u32::try_from(value).ok())
}

pub(crate) fn json_usize(json: &serde_json::Value, key: &str) -> Option<usize> {
    json_i64(json, key).and_then(|value| usize::try_from(value).ok())
}

fn json_f64(json: &serde_json::Value, key: &str) -> Option<f64> {
    json.get(key).and_then(|value| value.as_f64())
}

fn json_bool(json: &serde_json::Value, key: &str) -> Option<bool> {
    json.get(key).and_then(|value| value.as_bool())
}

fn json_str<'a>(json: &'a serde_json::Value, key: &str) -> Option<&'a str> {
    json.get(key).and_then(|value| value.as_str())
}

#[cfg(test)]
mod tests {
    use super::model_name;
    use std::path::Path;

    #[test]
    fn a_model_is_named_for_its_repository_or_its_directory() {
        // A hub id that was downloaded to the cache.
        assert_eq!(
            model_name(
                "Qwen/Qwen3-0.6B",
                Path::new("/cache/models--Qwen--Qwen3-0.6B/snapshots/c1899de2")
            ),
            "Qwen3-0.6B"
        );
        // A hub id that resolved to a plain directory.
        assert_eq!(
            model_name("org/some-model", Path::new("/nowhere/at/all/abc")),
            "some-model"
        );
        // A path.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("my-model");
        std::fs::create_dir(&path).unwrap();
        assert_eq!(model_name(path.to_str().unwrap(), &path), "my-model");
    }
}
