//! DFlash on the native engine decodes what plain decoding of the same target
//! does: the same tokens, greedy or drawn from a seed, whether the target is
//! a pure-attention model (a rejected tail is trimmed off its KV cache) or a
//! GDN hybrid (it is replayed out of the recurrent state too), and when a
//! tree of candidates is verified instead of one block.
//!
//! The targets are the parity suite's tiny fp32 checkpoints (Qwen3 with YaRN
//! rope, and the Qwen3.5 GDN hybrid); the draft is a randomly initialized
//! first-generation DFlash model, which is wrong far more often than right,
//! so nearly every round rolls the target's state back. Whatever the draft
//! says, the tokens must not change.

mod common;

use common::fixture_path;
use pmetal_models::GenerationConfig;
use pmetal_models::architectures::dflash_draft::{
    DFlashDraftConfig, DFlashDraftModel, DFlashExtras,
};
use pmetal_models::dflash_decoder::{DFlashConfig, DFlashDecoder, DFlashTarget};
use pmetal_models::dflash_native_target::NativeQwen3Target;
use serial_test::serial;

const PROMPT: [u32; 6] = [3, 11, 7, 2, 9, 40];
const NEW_TOKENS: usize = 36;

fn checkpoint(prefix: &str) -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    for (from, to) in [
        (format!("{prefix}_config.json"), "config.json"),
        (format!("{prefix}_weights.safetensors"), "model.safetensors"),
    ] {
        std::fs::copy(fixture_path(&from), dir.path().join(to)).expect("copy fixture");
    }
    dir
}

/// A random draft for a target `hidden` wide with `vocab` tokens, tapping
/// layers 0 and 1, drafting blocks of `block_size`.
fn random_draft(hidden: i32, vocab: i32, block_size: i32) -> DFlashDraftModel {
    let config = DFlashDraftConfig {
        model_type: "dflash_qwen3".to_string(),
        hidden_size: hidden,
        num_hidden_layers: 2,
        intermediate_size: 96,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        rms_norm_eps: 1e-6,
        vocab_size: vocab,
        max_position_embeddings: 256,
        rope_theta: Some(10_000.0),
        head_dim: 32,
        tie_word_embeddings: true,
        block_size,
        dflash_config: DFlashExtras {
            target_layer_ids: vec![0, 1],
            mask_token_id: 5,
            ..Default::default()
        },
        ..Default::default()
    };
    DFlashDraftModel::new(config).unwrap()
}

fn drain(what: &str) {
    pmetal_runtime::check_unobserved_error().unwrap_or_else(|e| panic!("{what}: {e}"));
}

fn configs() -> Vec<GenerationConfig> {
    vec![
        GenerationConfig::greedy(NEW_TOKENS),
        GenerationConfig::sampling(NEW_TOKENS, 0.9)
            .with_top_k(20)
            .with_top_p(0.95)
            .with_min_p(0.0)
            .with_repetition_penalty(1.0)
            .with_seed(3),
        GenerationConfig::greedy(NEW_TOKENS)
            .with_repetition_penalty(1.2)
            .with_frequency_penalty(0.2),
    ]
}

/// DFlash against plain decoding on `prefix`'s checkpoint, over every config
/// and a few block sizes.
fn check_linear(prefix: &str) {
    let dir = checkpoint(prefix);
    let mut accepted = 0;
    let mut drafted = 0;
    for block_size in [2, 4, 5] {
        let target = NativeQwen3Target::load(dir.path()).expect("native target loads");
        let mut decoder = DFlashDecoder::new(target, random_draft(64, 80, block_size));
        for (i, generation) in configs().into_iter().enumerate() {
            let config = DFlashConfig::new(generation);
            let plain = decoder.generate_plain(&PROMPT, &config).expect("plain");
            let speculative = decoder.generate(&PROMPT, &config).expect("dflash");
            drain("decode");
            assert_eq!(plain.metrics.num_generated, NEW_TOKENS);
            assert_eq!(
                speculative.tokens, plain.tokens,
                "{prefix}, block {block_size}, config {i}"
            );
            accepted += speculative.metrics.total_accepted;
            drafted += speculative.metrics.total_drafted;
        }
    }
    // Some guesses were kept and most were not, so both the rollback of a
    // rejected tail and the continuation after an accepted one ran.
    assert!(
        accepted > 0 && accepted < drafted,
        "{accepted} of {drafted}"
    );
}

#[test]
#[serial]
fn dflash_on_a_pure_attention_target_decodes_what_plain_decoding_does() {
    check_linear("qwen3_yarn");
}

#[test]
#[serial]
fn dflash_on_a_gdn_hybrid_target_decodes_what_plain_decoding_does() {
    check_linear("qwen3_5_swish");
}

/// The tree's mask is built in the dtype the target attends in, which the
/// runtime's attention insists on.
#[test]
#[serial]
fn the_tree_mask_matches_the_targets_dtype() {
    let dir = checkpoint("qwen3_yarn");
    let target = NativeQwen3Target::load(dir.path()).expect("native target loads");
    assert_eq!(
        target.tree_mask_dtype(),
        pmetal_runtime::compat::Dtype::Float32.as_i32(),
        "an fp32 checkpoint attends in fp32"
    );
}

/// A tree of candidates verified in one forward, with the draft's per-position
/// alternatives as siblings, ends where plain greedy decoding does.
#[test]
#[serial]
fn tree_verification_decodes_what_greedy_decoding_does() {
    let dir = checkpoint("qwen3_yarn");
    for (block_size, budget) in [(3, 4), (5, 8)] {
        let target = NativeQwen3Target::load(dir.path()).expect("native target loads");
        assert!(target.supports_tree_verify());
        let mut decoder = DFlashDecoder::new(target, random_draft(64, 80, block_size));
        let config = DFlashConfig::new(GenerationConfig::greedy(NEW_TOKENS));
        let plain = decoder.generate_plain(&PROMPT, &config).expect("plain");
        let tree = decoder
            .generate_ddtree(&PROMPT, &config, budget)
            .expect("tree");
        drain("tree decode");
        assert_eq!(
            tree.tokens, plain.tokens,
            "block {block_size}, budget {budget}"
        );
        assert!(tree.metrics.total_drafted > 0);
        assert!(
            tree.metrics.total_accepted > 0,
            "no node of any tree was accepted: block {block_size}, budget {budget}"
        );

        // Sampling has no per-node argmax to walk: the tree decoder runs the
        // linear loop and still draws the seeded plain tokens.
        let sampled = DFlashConfig::new(
            GenerationConfig::sampling(NEW_TOKENS, 1.0)
                .with_top_k(0)
                .with_top_p(1.0)
                .with_min_p(0.0)
                .with_repetition_penalty(1.0)
                .with_seed(9),
        );
        let plain = decoder.generate_plain(&PROMPT, &sampled).expect("plain");
        let tree = decoder
            .generate_ddtree(&PROMPT, &sampled, budget)
            .expect("tree");
        assert_eq!(tree.tokens, plain.tokens, "sampled, block {block_size}");
    }
}
