//! What a GGUF file holds besides the tensors: the architecture, the model
//! config and the tokenizer, written from the Hugging Face files of a model.

mod config;
mod tokenizer;

pub(crate) use config::add_hf_config_metadata;
pub(crate) use tokenizer::add_tokenizer_metadata;

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
