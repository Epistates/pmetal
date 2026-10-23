//! [`Speculator`]: every drafter decodes what plain decoding of the same
//! model decodes, through the same sampler, and streams it as it goes.
//!
//! Plain decoding here is the model's own cached decode with the sampler the
//! speculator is given a copy of. Qwen 3.5 runs on the `qwen3_5_swish` parity
//! checkpoint (fp32, a bundled `mtp.*` head); Gemma 4 and DFlash on tiny
//! random models.

use pmetal_cache::kv_cache::KVCache;
use pmetal_models::DynamicModel;
use pmetal_models::architectures::dflash_draft::{
    DFlashDraftConfig, DFlashDraftModel, DFlashExtras,
};
use pmetal_models::architectures::qwen3::{Qwen3Config, Qwen3ForCausalLM};
use pmetal_models::architectures::{
    Gemma4AssistantConfig, Gemma4AssistantForCausalLM, Gemma4Config, Gemma4ForCausalLM,
};
use pmetal_models::dflash_drafts::DFlashDraft;
use pmetal_models::generation::GenerationConfig;
use pmetal_models::speculator::{Speculator, SpeculatorSpec};
use pmetal_runtime::compat::random;
use pmetal_runtime::decode::{DecodeSampler, last_token_logits, prompt_tokens_to_input};
use serial_test::serial;

const PROMPT: [u32; 5] = [3, 17, 42, 8, 61];

fn drain(what: &str) {
    pmetal_runtime::check_unobserved_error().unwrap_or_else(|e| panic!("{what}: {e}"));
}

/// Plain decoding of `model`: the cached decode, one token a step, drawn by
/// `sampler`. A stop token ends it and is left out.
fn plain(
    model: &mut DynamicModel,
    prompt: &[u32],
    config: &GenerationConfig,
    mut sampler: DecodeSampler,
) -> Vec<u32> {
    let mut kv: KVCache = model.create_cache(256);
    let mut mamba = model.create_mamba_cache();
    let mut out = Vec::new();
    let mut input = prompt_tokens_to_input(prompt);
    while out.len() < config.max_new_tokens {
        let logits = model
            .forward_with_hybrid_cache(&input, None, Some(&mut kv), mamba.as_mut())
            .expect("forward");
        let token = sampler.sample(&last_token_logits(&logits));
        token.eval();
        let token = token.item_u32();
        if config.is_stop_token(token) {
            break;
        }
        out.push(token);
        input = prompt_tokens_to_input(&[token]);
    }
    drain("plain decode");
    out
}

/// What the speculator emits, and how.
struct Spec {
    tokens: Vec<u32>,
    run: pmetal_models::speculator::SpeculativeRun,
}

fn speculate(
    speculator: &mut Speculator,
    model: &mut DynamicModel,
    prompt: &[u32],
    config: &GenerationConfig,
    sampler: DecodeSampler,
) -> Spec {
    let mut tokens = Vec::new();
    let run = speculator
        .generate(model, prompt, config, sampler, |t| {
            tokens.push(t);
            true
        })
        .expect("speculative decode");
    drain("speculative decode");
    Spec { tokens, run }
}

fn configs() -> Vec<GenerationConfig> {
    vec![
        GenerationConfig::greedy(24),
        GenerationConfig::greedy(24)
            .with_repetition_penalty(1.3)
            .with_presence_penalty(0.4)
            .with_logit_bias(3, 2.0),
    ]
}

// ---------------------------------------------------------------------------
// Qwen 3.5 and its bundled MTP head
// ---------------------------------------------------------------------------

fn qwen_dir() -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    for (from, to) in [
        ("qwen3_5_swish_config.json", "config.json"),
        ("qwen3_5_swish_weights.safetensors", "model.safetensors"),
    ] {
        let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(from);
        std::fs::copy(fixture, dir.path().join(to)).expect("copy fixture");
    }
    dir
}

#[test]
#[serial]
fn a_qwen_mtp_speculator_decodes_what_plain_decoding_does() {
    let dir = qwen_dir();
    let mut model = DynamicModel::load(dir.path()).expect("checkpoint loads");
    let mut speculator = Speculator::load(
        &model,
        &SpeculatorSpec::QwenMtp {
            checkpoint: dir.path().to_path_buf(),
            draft_tokens: 3,
        },
        256,
    )
    .expect("the bundled head loads");
    assert_eq!(speculator.kind(), "qwen-mtp");
    assert_eq!(speculator.draft_tokens(), 3);

    let mut drafted = 0;
    for (i, config) in configs().iter().enumerate() {
        let want = plain(&mut model, &PROMPT, config, config.sampler(&PROMPT));
        // Twice: a generation leaves nothing behind for the next.
        for round in 0..2 {
            let got = speculate(
                &mut speculator,
                &mut model,
                &PROMPT,
                config,
                config.sampler(&PROMPT),
            );
            assert_eq!(got.tokens, want, "config {i}, round {round}");
            assert_eq!(got.run.num_generated, want.len());
            assert!(got.run.accepted_tokens <= got.run.drafted_tokens);
            drafted += got.run.drafted_tokens;
        }
    }
    assert!(drafted > 0, "the head drafted something");
}

#[test]
#[serial]
fn a_stop_token_ends_a_speculative_decode_and_is_left_out() {
    let dir = qwen_dir();
    let mut model = DynamicModel::load(dir.path()).expect("checkpoint loads");
    let mut speculator = Speculator::load(
        &model,
        &SpeculatorSpec::QwenMtp {
            checkpoint: dir.path().to_path_buf(),
            draft_tokens: 3,
        },
        256,
    )
    .unwrap();
    let free = GenerationConfig::greedy(12);
    let made = plain(&mut model, &PROMPT, &free, free.sampler(&PROMPT));
    let stop = made[5];
    let at = made.iter().position(|&t| t == stop).unwrap();
    let config = GenerationConfig::greedy(40).with_stop_tokens(vec![stop]);
    let got = speculate(
        &mut speculator,
        &mut model,
        &PROMPT,
        &config,
        config.sampler(&PROMPT),
    );
    assert_eq!(got.tokens, made[..at]);
    assert!(got.run.stopped_by_token);
}

#[test]
#[serial]
fn on_token_can_end_a_speculative_decode_early() {
    let dir = qwen_dir();
    let mut model = DynamicModel::load(dir.path()).expect("checkpoint loads");
    let mut speculator = Speculator::load(
        &model,
        &SpeculatorSpec::QwenMtp {
            checkpoint: dir.path().to_path_buf(),
            draft_tokens: 3,
        },
        256,
    )
    .unwrap();
    let config = GenerationConfig::greedy(24);
    let want = plain(&mut model, &PROMPT, &config, config.sampler(&PROMPT));
    let mut seen = Vec::new();
    let run = speculator
        .generate(&mut model, &PROMPT, &config, config.sampler(&PROMPT), |t| {
            seen.push(t);
            seen.len() < 5
        })
        .unwrap();
    drain("early stop");
    assert_eq!(seen, want[..5]);
    assert_eq!(run.num_generated, 5);
    assert!(!run.stopped_by_token);
}

#[test]
#[serial]
fn a_drafter_for_another_family_is_refused_at_load() {
    random::seed(1);
    let llama = DynamicModel::Llama(
        pmetal_models::architectures::llama::LlamaForCausalLM::new(
            pmetal_models::architectures::llama::LlamaConfig {
                vocab_size: 64,
                hidden_size: 32,
                intermediate_size: 64,
                num_hidden_layers: 2,
                num_attention_heads: 4,
                num_key_value_heads: Some(2),
                head_dim: Some(8),
                max_position_embeddings: 64,
                ..Default::default()
            },
        )
        .unwrap(),
    );
    let dir = qwen_dir();
    let Err(err) = Speculator::load(
        &llama,
        &SpeculatorSpec::QwenMtp {
            checkpoint: dir.path().to_path_buf(),
            draft_tokens: 3,
        },
        64,
    ) else {
        panic!("an MTP head cannot draft for Llama");
    };
    assert!(err.to_string().contains("Qwen 3.5"), "{err}");
    // A directory that is neither a DFlash draft nor (for a Llama) usable.
    let Err(err) = Speculator::load(
        &llama,
        &SpeculatorSpec::Draft {
            path: dir.path().to_path_buf(),
            tokens: None,
        },
        64,
    ) else {
        panic!("not a draft model");
    };
    assert!(err.to_string().contains("Gemma 4"), "{err}");
}

// ---------------------------------------------------------------------------
// DFlash
// ---------------------------------------------------------------------------

const DFLASH_VOCAB: i32 = 9;

fn dflash_target() -> DynamicModel {
    random::seed(21);
    DynamicModel::Qwen3(
        Qwen3ForCausalLM::new(Qwen3Config {
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 4,
            num_attention_heads: 2,
            num_key_value_heads: Some(1),
            head_dim: 16,
            vocab_size: DFLASH_VOCAB,
            max_position_embeddings: 256,
            ..Default::default()
        })
        .unwrap(),
    )
}

fn dflash_draft(block_size: i32) -> DFlashDraft {
    random::seed(22);
    DFlashDraft::V1(
        DFlashDraftModel::new(DFlashDraftConfig {
            model_type: "dflash_qwen3".to_string(),
            hidden_size: 32,
            num_hidden_layers: 2,
            intermediate_size: 64,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            rms_norm_eps: 1e-6,
            vocab_size: DFLASH_VOCAB,
            max_position_embeddings: 64,
            rope_theta: Some(10_000.0),
            head_dim: 16,
            tie_word_embeddings: true,
            block_size,
            dflash_config: DFlashExtras {
                target_layer_ids: vec![2, 3],
                mask_token_id: 5,
                ..Default::default()
            },
            ..Default::default()
        })
        .unwrap(),
    )
}

#[test]
#[serial]
fn a_dflash_speculator_decodes_what_plain_decoding_does() {
    let mut model = dflash_target();
    let prompt = [1u32, 2, 3, 4, 5];
    let configs = [
        GenerationConfig::greedy(30),
        GenerationConfig::greedy(30)
            .with_repetition_penalty(1.3)
            .with_logit_bias(3, 2.0),
        GenerationConfig::sampling(30, 0.9)
            .with_top_k(5)
            .with_top_p(0.9)
            .with_seed(13),
    ];
    let mut accepted = 0;
    for block in [2, 3, 5] {
        let mut speculator = Speculator::from_dflash(&model, dflash_draft(5), Some(block), 256)
            .expect("the draft fits the target");
        assert_eq!(speculator.kind(), "dflash");
        assert_eq!(speculator.draft_tokens(), block);
        for (i, config) in configs.iter().enumerate() {
            let want = plain(&mut model, &prompt, config, config.sampler(&prompt));
            let got = speculate(
                &mut speculator,
                &mut model,
                &prompt,
                config,
                config.sampler(&prompt),
            );
            assert_eq!(got.tokens, want, "block {block}, config {i}");
            accepted += got.run.accepted_tokens;
        }
    }
    assert!(accepted > 0, "no draft was ever accepted");
}

#[test]
#[serial]
fn a_dflash_speculator_for_a_model_of_another_size_is_refused() {
    let model = dflash_target();
    random::seed(1);
    let other = DynamicModel::Qwen3(
        Qwen3ForCausalLM::new(Qwen3Config {
            hidden_size: 48,
            intermediate_size: 64,
            num_hidden_layers: 4,
            num_attention_heads: 3,
            num_key_value_heads: Some(1),
            head_dim: 16,
            vocab_size: DFLASH_VOCAB,
            ..Default::default()
        })
        .unwrap(),
    );
    assert!(Speculator::from_dflash(&model, dflash_draft(4), None, 64).is_ok());
    let Err(err) = Speculator::from_dflash(&other, dflash_draft(4), None, 64) else {
        panic!("a hidden size mismatch must be refused");
    };
    assert!(err.to_string().contains("hidden size"), "{err}");
}

// ---------------------------------------------------------------------------
// Gemma 4
// ---------------------------------------------------------------------------

const GEMMA_VOCAB: i32 = 64;

fn gemma_config(layer_types: &[&str], hidden: i32, shared: i32) -> Gemma4Config {
    Gemma4Config {
        model_type: "gemma4_text".to_string(),
        vocab_size: GEMMA_VOCAB,
        hidden_size: hidden,
        intermediate_size: 2 * hidden,
        num_hidden_layers: layer_types.len() as i32,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 8,
        global_head_dim: Some(8),
        num_global_key_value_heads: Some(1),
        max_position_embeddings: 128,
        rms_norm_eps: 1e-6,
        attention_k_eq_v: false,
        tie_word_embeddings: true,
        sliding_window: 16,
        final_logit_softcapping: None,
        layer_types: layer_types.iter().map(|t| t.to_string()).collect(),
        rope_parameters: None,
        _raw_rope_parameters: None,
        hidden_size_per_layer_input: None,
        vocab_size_per_layer_input: None,
        hidden_activation: None,
        num_kv_shared_layers: Some(shared),
        use_double_wide_mlp: Some(false),
        enable_moe_block: Some(false),
        num_experts: None,
        top_k_experts: None,
        moe_intermediate_size: None,
    }
}

#[test]
#[serial]
fn a_gemma_assistant_speculator_decodes_what_plain_decoding_does() {
    random::seed(11);
    let mut model = DynamicModel::Gemma4(
        Gemma4ForCausalLM::new(gemma_config(
            &[
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
            ],
            32,
            2,
        ))
        .unwrap(),
    );
    random::seed(12);
    let assistant = Gemma4AssistantForCausalLM::new(Gemma4AssistantConfig {
        model_type: "gemma4_assistant".to_string(),
        text_config: gemma_config(&["sliding_attention", "full_attention"], 16, 2),
        backbone_hidden_size: 32,
        use_ordered_embeddings: false,
        num_centroids: 2048,
        centroid_intermediate_top_k: 32,
        tie_word_embeddings: true,
    })
    .unwrap();
    let mut speculator = Speculator::from_gemma4_assistant(&model, assistant, 3, 256)
        .expect("the assistant fits the target");
    assert_eq!(speculator.kind(), "gemma4-assistant");
    assert_eq!(speculator.draft_tokens(), 3);
    for (i, config) in configs().iter().enumerate() {
        let want = plain(&mut model, &PROMPT, config, config.sampler(&PROMPT));
        let got = speculate(
            &mut speculator,
            &mut model,
            &PROMPT,
            config,
            config.sampler(&PROMPT),
        );
        assert_eq!(got.tokens, want, "config {i}");
    }
}
