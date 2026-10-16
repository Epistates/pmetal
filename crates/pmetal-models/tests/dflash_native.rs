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

use std::path::PathBuf;

use pmetal_models::GenerationConfig;
use pmetal_models::architectures::dflash_draft::{
    DFlashDraftConfig, DFlashDraftModel, DFlashExtras,
};
use pmetal_models::dflash_decoder::{DFlashConfig, DFlashDecoder, DFlashTarget};
use pmetal_models::dflash_native_target::NativeQwen3Target;
use serial_test::serial;

const PROMPT: [u32; 6] = [3, 11, 7, 2, 9, 40];
const NEW_TOKENS: usize = 36;

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

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
/// layers 0 and 1, drafting blocks of `block_size`. Seeded: the tests also
/// require that some guesses are kept, which an unlucky draw can miss.
fn random_draft(hidden: i32, vocab: i32, block_size: i32) -> DFlashDraftModel {
    pmetal_runtime::compat::random::seed(17 + block_size as u64);
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

/// Every node of a verified tree sees what a plain forward over its own
/// path sees: the logits the tree forward gives at a node (tree attention
/// mask, the node's own rope position, siblings hidden) are those of running
/// the prompt, the root and the node's ancestors straight through.
#[test]
#[serial]
fn a_tree_forward_gives_each_node_the_logits_of_its_own_path() {
    use pmetal_cache::speculative::SpecCapture;
    use pmetal_models::ddtree::{build_tree, compile_tree};
    use pmetal_runtime::compat::Array;

    let dir = checkpoint("qwen3_yarn");
    let mut target = NativeQwen3Target::load(dir.path()).expect("native target loads");

    // A draft distribution over three positions, wide enough that the tree
    // branches: the nodes at one depth are siblings with different paths.
    // The first position is soft and the later ones sharp, so a likely
    // child outranks a second sibling and the tree goes deep as well as wide.
    let rows: Vec<f32> = (0..3 * 80)
        .map(|i| match (i / 80, i % 80) {
            (0, _) => (((i * 29 + 5) % 23) as f32) * 0.1,
            (1, 17) | (2, 40) => 12.0,
            _ => 0.0,
        })
        .collect();
    let draft_logits = Array::from_slice(&rows, &[3, 80]);
    let tree = build_tree(&draft_logits, 9);
    assert!(
        tree.node_depths.contains(&2) && tree.parents.iter().filter(|&&p| p == 0).count() > 1,
        "the tree should branch and go deep: {:?}",
        tree.node_depths
    );

    let seed = 17u32;
    let mut capture = SpecCapture::with_layers(Vec::new());
    target.reset_state();
    let mut kv = target.make_kv_cache(1);
    target
        .prefill_with_capture(&array_row(&PROMPT), Some(&mut kv), None, &mut capture)
        .expect("prefill");
    let past = PROMPT.len() as i32;
    let compiled = compile_tree(&tree, seed as i32, past, past, target.tree_mask_dtype());
    let tree_logits = target
        .forward_tree_verify(
            &compiled.verify_input_ids,
            &compiled.verify_position_ids,
            &compiled.attention_mask,
            &mut capture,
        )
        .expect("tree verify");
    drain("tree verify");
    let nodes = compiled.current_length;
    assert_eq!(tree_logits.dim(1) as usize, nodes);

    for node in 0..nodes {
        // The tokens on the path from the root to this node.
        let mut path = Vec::new();
        let mut at = node;
        while at != 0 {
            path.push(tree.node_token_ids[at - 1] as u32);
            at = tree.parents[at] as usize;
        }
        path.reverse();
        let mut sequence = PROMPT.to_vec();
        sequence.push(seed);
        sequence.extend(path);

        target.reset_state();
        let mut fresh = target.make_kv_cache(1);
        let mut capture = SpecCapture::with_layers(Vec::new());
        let straight = target
            .forward_with_capture(
                &array_row(&sequence),
                None,
                Some(&mut fresh),
                None,
                &mut capture,
            )
            .expect("straight forward");
        let last = straight.dim(1);
        let want = straight.slice(&[0, last - 1, 0], &[1, last, 80]);
        let got = tree_logits.slice(&[0, node as i32, 0], &[1, node as i32 + 1, 80]);
        let diff = pmetal_cache::test_utils::max_abs_diff(&got, &want);
        drain("straight");
        assert!(
            diff < 1e-4,
            "node {node} differs from its straight path by {diff:.3e}"
        );
    }
}

fn array_row(tokens: &[u32]) -> pmetal_runtime::compat::Array {
    let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
    pmetal_runtime::compat::Array::from_slice(&ids, &[1, ids.len() as i32])
}

/// After a tree round the cache keeps the accepted path and nothing else:
/// the next token's logits are those of the prompt, the root and the path run
/// straight through.
#[test]
#[serial]
fn a_compacted_tree_cache_continues_like_the_straight_path() {
    use pmetal_cache::speculative::SpecCapture;
    use pmetal_models::ddtree::{build_tree, compile_tree};
    use pmetal_runtime::compat::Array;

    let dir = checkpoint("qwen3_yarn");
    let mut target = NativeQwen3Target::load(dir.path()).expect("native target loads");
    let rows: Vec<f32> = (0..3 * 80)
        .map(|i| match (i / 80, i % 80) {
            (0, _) => (((i * 29 + 5) % 23) as f32) * 0.1,
            (1, 17) | (2, 40) => 12.0,
            _ => 0.0,
        })
        .collect();
    let tree = build_tree(&Array::from_slice(&rows, &[3, 80]), 9);
    // A depth-2 node, its parent, and the root: the accepted path.
    let deep = tree
        .node_depths
        .iter()
        .position(|&d| d == 2)
        .expect("a depth-2 node")
        + 1;
    let parent = tree.parents[deep] as usize;
    let accepted = [0usize, parent, deep];

    let seed = 17u32;
    let mut capture = SpecCapture::with_layers(Vec::new());
    target.reset_state();
    let mut kv = target.make_kv_cache(1);
    target
        .prefill_with_capture(&array_row(&PROMPT), Some(&mut kv), None, &mut capture)
        .expect("prefill");
    let past = PROMPT.len();
    let compiled = compile_tree(
        &tree,
        seed as i32,
        past as i32,
        past as i32,
        target.tree_mask_dtype(),
    );
    target
        .forward_tree_verify(
            &compiled.verify_input_ids,
            &compiled.verify_position_ids,
            &compiled.attention_mask,
            &mut capture,
        )
        .expect("tree verify");
    target.compact_tree_cache(past, compiled.current_length, &accepted);
    let next = 33u32;
    let continued = target
        .forward_with_capture(&array_row(&[next]), None, Some(&mut kv), None, &mut capture)
        .expect("continue");
    drain("compacted decode");

    let mut sequence = PROMPT.to_vec();
    sequence.push(seed);
    sequence.push(tree.node_token_ids[parent - 1] as u32);
    sequence.push(tree.node_token_ids[deep - 1] as u32);
    sequence.push(next);
    target.reset_state();
    let mut fresh = target.make_kv_cache(1);
    let mut capture = SpecCapture::with_layers(Vec::new());
    let straight = target
        .forward_with_capture(
            &array_row(&sequence),
            None,
            Some(&mut fresh),
            None,
            &mut capture,
        )
        .expect("straight");
    let last = straight.dim(1);
    let want = straight.slice(&[0, last - 1, 0], &[1, last, 80]);
    let diff = pmetal_cache::test_utils::max_abs_diff(&continued, &want);
    drain("straight");
    assert!(
        diff < 1e-4,
        "the compacted cache continues {diff:.3e} off the straight path"
    );
}
