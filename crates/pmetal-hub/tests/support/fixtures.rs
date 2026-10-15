//! Valid model files for the fake Hub to serve, so that what a download
//! produces passes the same checks a real model does.

use serde_json::json;

/// Detach this test process from the developer's own Hub setup: no implicit
/// token (a real one must never be sent to a loopback server), no offline
/// switch, a cache and home of their own. Call at the top of every test.
#[allow(unsafe_code)]
pub fn isolate_env() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let home = tempfile::tempdir().unwrap().keep();
        // SAFETY: runs once, before this test binary builds a Hub client; the
        // tests in this binary only read these variables through that.
        unsafe {
            std::env::set_var("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1");
            std::env::set_var("HF_HOME", &home);
            std::env::remove_var("HF_HUB_CACHE");
            std::env::remove_var("HUGGINGFACE_HUB_CACHE");
            std::env::remove_var("HF_TOKEN");
            std::env::remove_var("HF_TOKEN_PATH");
            std::env::remove_var("HF_HUB_OFFLINE");
            std::env::remove_var("HF_ENDPOINT");
        }
    });
}

/// A valid safetensors file holding one F16 tensor named `tag` of `elements`
/// values (`elements * 2` bytes of data). Two files with different tags
/// differ in content even at the same size.
pub fn safetensors_file(tag: &str, elements: u64) -> Vec<u8> {
    let bytes = elements * 2;
    let header = serde_json::to_vec(&json!({
        tag: { "dtype": "F16", "shape": [elements], "data_offsets": [0, bytes] }
    }))
    .unwrap();
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend(header);
    out.extend((0..bytes).map(|i| (i as u8).wrapping_add(tag.len() as u8)));
    out
}

/// A valid GGUF file. `split` is `(index, count, total tensors)` for a shard
/// of a split model.
pub fn gguf_file(split: Option<(u16, u16, i32)>, tensors: &[(&str, f32)]) -> Vec<u8> {
    use pmetal_gguf::{GgufBuilder, MetadataValue};
    let mut builder = GgufBuilder::with_model("llama", "m");
    if let Some((no, count, total)) = split {
        builder.add_metadata("split.no", MetadataValue::Uint16(no));
        builder.add_metadata("split.count", MetadataValue::Uint16(count));
        builder.add_metadata("split.tensors.count", MetadataValue::Int32(total));
    }
    for (name, value) in tensors {
        builder.add_f32_tensor(*name, vec![4], vec![*value; 4]);
    }
    builder.build_to_bytes().unwrap()
}
