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
