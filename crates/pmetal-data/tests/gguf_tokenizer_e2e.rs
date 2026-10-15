//! End-to-end GGUF tokenizer reconstruction.
//!
//! Writes a synthetic GGUF (BPE tokenizer metadata) with `GgufBuilder`, then
//! loads it through `Tokenizer::from_model_path` — the same entry point the
//! inference runner uses — and checks the round-trip plus special-token id
//! resolution straight from `tokenizer.ggml.*` metadata.

use pmetal_data::tokenizer::Tokenizer;
use pmetal_gguf::{GgufBuilder, MetadataValue};

fn write_bpe_gguf(path: &std::path::Path) {
    let mut b = GgufBuilder::with_model("llama", "tok-test");

    // Minimal decoder metadata so the file is a plausible model, though the
    // tokenizer path only reads tokenizer.ggml.*.
    b.add_u32("llama.embedding_length", 8);
    b.add_u32("llama.block_count", 1);
    b.add_u32("llama.attention.head_count", 1);
    b.add_u32("llama.feed_forward_length", 8);

    b.add_string("tokenizer.ggml.model", "gpt2");
    b.add_string_array(
        "tokenizer.ggml.tokens",
        vec!["a".into(), "b".into(), "ab".into(), "<s>".into()],
    );
    b.add_metadata(
        "tokenizer.ggml.token_type",
        // 1=NORMAL, 3=CONTROL
        MetadataValue::Array(vec![
            MetadataValue::Int32(1),
            MetadataValue::Int32(1),
            MetadataValue::Int32(1),
            MetadataValue::Int32(3),
        ]),
    );
    b.add_string_array("tokenizer.ggml.merges", vec!["a b".into()]);
    b.add_u32("tokenizer.ggml.bos_token_id", 3);
    b.add_u32("tokenizer.ggml.eos_token_id", 3);
    b.add_u32("tokenizer.ggml.eot_token_id", 2);
    b.add_u32("tokenizer.ggml.eom_token_id", 1);
    b.add_bool("tokenizer.ggml.add_space_prefix", false);
    b.add_string(
        "tokenizer.chat_template",
        "{{ bos_token }}{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}",
    );

    std::fs::write(path, b.build_to_bytes().expect("build gguf")).expect("write gguf");
}

#[test]
fn from_model_path_reconstructs_gguf_tokenizer() {
    let dir = tempfile::tempdir().expect("tempdir");
    let gguf = dir.path().join("model.gguf");
    write_bpe_gguf(&gguf);

    // File path.
    let tok = Tokenizer::from_model_path(&gguf).expect("load tokenizer from gguf file");
    assert_eq!(tok.encode("ab").expect("encode"), vec![2]);
    assert_eq!(tok.decode(&[2]).expect("decode"), "ab");
    assert_eq!(tok.bos_token_id(), Some(3));
    assert_eq!(tok.eos_token_id(), Some(3));

    // Directory containing the single .gguf resolves identically.
    let tok_dir = Tokenizer::from_model_path(dir.path()).expect("load tokenizer from gguf dir");
    assert_eq!(tok_dir.encode("ab").expect("encode"), vec![2]);
}

/// A GGUF-only checkpoint carries its chat template in the metadata, and the
/// template renders with the BOS the file names.
#[test]
fn the_chat_template_comes_from_the_gguf() {
    use pmetal_data::chat_templates::{ChatTemplateKwargs, Message, detect_chat_template};

    let dir = tempfile::tempdir().expect("tempdir");
    let gguf = dir.path().join("model.gguf");
    write_bpe_gguf(&gguf);
    for path in [gguf.as_path(), dir.path()] {
        let template = detect_chat_template(path, "tok-test");
        assert_eq!(template.bos_token.as_deref(), Some("<s>"));
        let text = template
            .render(
                &[Message::user("hi")],
                None,
                &ChatTemplateKwargs::new(),
                false,
            )
            .expect("render");
        assert_eq!(text, "<s><|im_start|>user\nhi<|im_end|>\n");
    }
}

/// The metadata the tokenizer is built from is read whole: the end-of-turn
/// and end-of-message ids and the space-prefix flag come through.
#[test]
fn the_tokenizer_metadata_is_read_whole() {
    let dir = tempfile::tempdir().expect("tempdir");
    let gguf = dir.path().join("model.gguf");
    write_bpe_gguf(&gguf);
    let content = pmetal_gguf::GgufContent::from_file(&gguf).expect("read gguf");
    let data = content.tokenizer_data().expect("tokenizer data");
    assert_eq!(data.eot_id, Some(2));
    assert_eq!(data.eom_id, Some(1));
    assert_eq!(data.add_space_prefix, Some(false));
    assert_eq!(data.bos_id, Some(3));
}
