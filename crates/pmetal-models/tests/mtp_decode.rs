//! Multi-token-prediction speculative decoding, against every target engine
//! and draft head it runs on.
//!
//! Greedy speculative decoding is exact: whatever the head drafts, the output
//! is token for token what plain greedy decoding of the target gives. That is
//! checked against the target recomputed from scratch at every step, so a
//! cache the speculative loop rewinds wrongly shows up as a different token.
//!
//! Exactness says nothing about the drafts, which only decide how many tokens
//! a round keeps. Two more checks cover them:
//!
//! * An oracle head that drafts the target's own continuation must have every
//!   draft accepted, and every hidden state the loop hands it must be the
//!   target's state that predicted the token it comes with (`h[i]` with
//!   `x[i + 1]`, the pairing MTP heads are trained on).
//! * The real heads' incremental state (the Qwen predictor's own KV cache)
//!   must draft what the head drafts when recomputed from scratch over the
//!   committed sequence.
//!
//! Qwen runs on the `qwen3_5_swish` parity checkpoint (fp32; GDN and gated
//! attention layers, a bundled `mtp.*` predictor), on both the dynamic model
//! and the native engine. Gemma 4 runs on a tiny random target and assistant.

use pmetal_bridge::compat::{Array, Exception, ops::concatenate_axis, random};
use pmetal_bridge::decode::{greedy_token, last_token_hidden, last_token_logits};
use pmetal_mlx::kv_cache::{KVCache, KVCacheConfig, MambaCache};
use pmetal_mlx::test_utils::max_abs_diff;
use pmetal_models::architectures::{
    Gemma4AssistantConfig, Gemma4AssistantForCausalLM, Gemma4AssistantSharedKvStates, Gemma4Config,
    Gemma4ForCausalLM, Qwen3NextForCausalLM, Qwen3NextMtpForCausalLM, load_qwen3_next_mtp_from_dir,
};
use pmetal_models::generation::{GenerationConfig, GenerationOutput};
use pmetal_models::mtp::{MtpDrafter, MtpTarget, generate_mtp_streaming};
use pmetal_models::{
    DynamicModel, Gemma4MtpDrafter, Gemma4MtpVerifier, NativeQwen3MtpVerifier, Qwen3NextMtpDrafter,
    Qwen3NextMtpVerifier,
};
use serial_test::serial;

const NEW_TOKENS: usize = 24;
const PROMPT: [u32; 5] = [3, 17, 42, 8, 61];

/// A hidden state computed by a cached forward against the same one
/// recomputed from scratch, both fp32.
const HIDDEN_TOL: f32 = 1e-3;

fn drain(what: &str) {
    pmetal_bridge::check_unobserved_error().unwrap_or_else(|e| panic!("{what}: {e}"));
}

/// Greedy decoding with no penalties: the bare argmax.
fn greedy_config() -> GenerationConfig {
    GenerationConfig::greedy(NEW_TOKENS)
}

/// Run `tokens` through `target` from an empty state.
fn from_scratch<T: MtpTarget>(target: &mut T, tokens: &[u32]) -> (Array, Array) {
    target.reset().expect("reset");
    target.forward(tokens).expect("forward")
}

/// Plain greedy decoding of `target`, recomputed from scratch every step.
fn plain_greedy<T: MtpTarget>(target: &mut T, prompt: &[u32], n: usize) -> Vec<u32> {
    let mut seq = prompt.to_vec();
    for _ in 0..n {
        let (_, logits) = from_scratch(target, &seq);
        seq.push(greedy_token(&last_token_logits(&logits)));
    }
    drain("plain greedy");
    seq
}

fn decode<T: MtpTarget, D: MtpDrafter<T>>(
    target: &mut T,
    drafter: &mut D,
    budget: usize,
) -> GenerationOutput {
    let out = generate_mtp_streaming(target, drafter, &PROMPT, &greedy_config(), budget, |_| true)
        .expect("MTP decode");
    drain("MTP decode");
    out
}

fn assert_exact(what: &str, out: &GenerationOutput, want: &[u32]) {
    assert_eq!(out.num_generated, NEW_TOKENS, "{what}: tokens generated");
    assert_eq!(
        out.token_ids, want,
        "{what}: speculative greedy decoding left the target's own greedy tokens"
    );
}

fn assert_close(what: &str, got: &Array, want: &Array) {
    assert_eq!(got.shape(), want.shape(), "{what}: shape");
    let diff = max_abs_diff(got, want);
    assert!(diff <= HIDDEN_TOL, "{what}: off by {diff:.3e}");
}

fn one_hot_logits(token: u32, vocab: i32) -> Array {
    let mut row = vec![0.0f32; vocab as usize];
    row[token as usize] = 1.0;
    Array::from_f32_slice(&row, &[1, vocab])
}

/// A head that drafts the target's own greedy continuation, computed from
/// scratch on a second copy of the target, except every `wrong_every`-th
/// draft, which it gets wrong on purpose. It checks every hidden state the
/// loop hands it against that copy.
struct OracleDrafter<O: MtpTarget> {
    oracle: O,
    vocab: i32,
    wrong_every: Option<usize>,
    committed: Vec<u32>,
    round: Vec<u32>,
    drafted: usize,
    seeds_checked: usize,
}

impl<O: MtpTarget> OracleDrafter<O> {
    fn new(oracle: O, vocab: i32, wrong_every: Option<usize>) -> Self {
        Self {
            oracle,
            vocab,
            wrong_every,
            committed: Vec::new(),
            round: Vec::new(),
            drafted: 0,
            seeds_checked: 0,
        }
    }
}

impl<T: MtpTarget, O: MtpTarget> MtpDrafter<T> for OracleDrafter<O> {
    fn prefill(&mut self, _: &mut T, prompt: &[u32], hidden: &Array) -> Result<(), Exception> {
        let (want, _) = from_scratch(&mut self.oracle, prompt);
        assert_close("prefill hidden states", hidden, &want);
        self.committed = prompt.to_vec();
        Ok(())
    }

    fn step(
        &mut self,
        _: &mut T,
        token: u32,
        hidden: &Array,
        step: usize,
        position: usize,
    ) -> Result<(Array, Array), Exception> {
        if step == 0 {
            // `token` is the pending one; the state that predicted it is the
            // target's at the last committed position.
            assert_eq!(position, self.committed.len(), "pending token position");
            let (want, _) = from_scratch(&mut self.oracle, &self.committed);
            assert_close(
                &format!("seed of the round at position {position}"),
                hidden,
                &last_token_hidden(&want),
            );
            self.seeds_checked += 1;
            self.round = self.committed.clone();
        }
        self.round.push(token);
        let (_, logits) = from_scratch(&mut self.oracle, &self.round);
        let mut next = greedy_token(&last_token_logits(&logits));
        self.drafted += 1;
        if self.wrong_every.is_some_and(|n| self.drafted % n == 0) {
            next = (next + 1) % self.vocab as u32;
        }
        Ok((hidden.clone(), one_hot_logits(next, self.vocab)))
    }

    fn commit(
        &mut self,
        _: &mut T,
        pending: u32,
        _: &Array,
        drafts: &[u32],
        accepted: usize,
        _: &Array,
    ) -> Result<(), Exception> {
        self.committed.push(pending);
        self.committed.extend_from_slice(&drafts[..accepted]);
        Ok(())
    }
}

/// Every draft of an oracle that never errs is accepted; one that errs on
/// purpose gets partial rounds, which rewind the target. Both are exact.
fn check_oracle<T: MtpTarget, O: MtpTarget>(
    what: &str,
    target: &mut T,
    oracle: O,
    vocab: i32,
    want: &[u32],
) -> O {
    let mut oracle = Some(oracle);
    for (budget, wrong_every) in [(3, None), (4, Some(3)), (2, Some(2))] {
        let mut drafter = OracleDrafter::new(oracle.take().unwrap(), vocab, wrong_every);
        let out = decode(target, &mut drafter, budget);
        let what = format!("{what}, oracle drafting {budget}, wrong every {wrong_every:?}");
        assert_exact(&what, &out, want);
        let m = out.speculative_metrics.expect("metrics");
        assert!(m.drafted_tokens > 0, "{what}: nothing drafted");
        assert!(drafter.seeds_checked > 1, "{what}: no round seed checked");
        match wrong_every {
            None => {
                assert_eq!(
                    m.accepted_draft_tokens, m.drafted_tokens,
                    "{what}: the target rejected its own tokens"
                );
                assert!(m.avg_accepted_per_verify() > 1.0, "{what}: {m:?}");
            }
            Some(_) => {
                assert!(
                    m.partial_accept_steps + m.zero_accept_steps > 0,
                    "{what}: {m:?}"
                );
                assert!(m.accepted_draft_tokens > 0, "{what}: {m:?}");
            }
        }
        oracle = Some(drafter.oracle);
    }
    oracle.unwrap()
}

// ---------------------------------------------------------------------------
// Qwen3.5 (the `qwen3_5_swish` checkpoint)
// ---------------------------------------------------------------------------

struct Checkpoint {
    dir: tempfile::TempDir,
}

fn qwen_checkpoint() -> Checkpoint {
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
    Checkpoint { dir }
}

impl Checkpoint {
    fn dynamic(&self) -> Qwen3NextForCausalLM {
        let DynamicModel::Qwen3Next(model) = DynamicModel::load(self.dir.path()).expect("load")
        else {
            panic!("a Qwen3.5 checkpoint loads as Qwen3Next");
        };
        model
    }

    fn mtp(&self, target: &Qwen3NextForCausalLM) -> Qwen3NextMtpForCausalLM {
        load_qwen3_next_mtp_from_dir(self.dir.path(), &target.config).expect("MTP head")
    }

    fn native(&self) -> NativeQwen3MtpVerifier {
        NativeQwen3MtpVerifier::load(self.dir.path()).expect("native engine")
    }
}

struct QwenCaches {
    kv: KVCache,
    mamba: MambaCache,
}

impl QwenCaches {
    fn new(model: &Qwen3NextForCausalLM) -> Self {
        let c = &model.config;
        Self {
            kv: KVCache::new(KVCacheConfig::new(
                c.num_hidden_layers as usize,
                128,
                c.get_num_kv_heads() as usize,
                c.get_head_dim() as usize,
            )),
            mamba: MambaCache::new(c.num_hidden_layers as usize),
        }
    }

    fn verifier<'a>(&'a mut self, model: &'a mut Qwen3NextForCausalLM) -> Qwen3NextMtpVerifier<'a> {
        Qwen3NextMtpVerifier::new(model, &mut self.kv, &mut self.mamba)
    }
}

/// The predictor recomputed from scratch: the target's states over the
/// committed tokens, each paired with the token after it, then the pending
/// token with the last state. Its logits are what the incremental head must
/// draft first in a round.
struct FromScratchQwenHead {
    target: Qwen3NextForCausalLM,
    caches: QwenCaches,
    mtp: Qwen3NextMtpForCausalLM,
}

impl FromScratchQwenHead {
    fn first_draft_logits(&mut self, committed: &[u32], pending: u32) -> Array {
        let (hidden, _) = from_scratch(&mut self.caches.verifier(&mut self.target), committed);
        let mut tokens = committed[1..].to_vec();
        tokens.push(pending);
        let input = pmetal_bridge::decode::prompt_tokens_to_input(&tokens);
        let (_, logits) = self
            .mtp
            .forward_logits(&input, &hidden, None, None, 0)
            .expect("MTP from scratch");
        last_token_logits(&logits)
    }

    /// The target's greedy token after `tokens`.
    fn target_greedy(&mut self, tokens: &[u32]) -> u32 {
        let (_, logits) = from_scratch(&mut self.caches.verifier(&mut self.target), tokens);
        greedy_token(&last_token_logits(&logits))
    }
}

/// The Qwen predictor, checked at each round's first step against itself
/// recomputed from scratch. Steered, it drafts the target's own tokens
/// instead of its own (still running every step), so rounds keep drafts and
/// its commit has the target's states for them to take in.
struct CheckedQwenHead<'a> {
    inner: Qwen3NextMtpDrafter<'a>,
    reference: FromScratchQwenHead,
    steer: bool,
    committed: Vec<u32>,
    round: Vec<u32>,
    rounds_checked: usize,
}

impl<T: MtpTarget> MtpDrafter<T> for CheckedQwenHead<'_> {
    fn prefill(&mut self, target: &mut T, prompt: &[u32], hidden: &Array) -> Result<(), Exception> {
        self.committed = prompt.to_vec();
        self.inner.prefill(target, prompt, hidden)
    }

    fn step(
        &mut self,
        target: &mut T,
        token: u32,
        hidden: &Array,
        step: usize,
        position: usize,
    ) -> Result<(Array, Array), Exception> {
        let (next, logits) = self.inner.step(target, token, hidden, step, position)?;
        if step == 0 {
            let want = self.reference.first_draft_logits(&self.committed, token);
            assert_close(
                &format!("first draft logits at position {position}"),
                &logits,
                &want,
            );
            self.rounds_checked += 1;
            self.round = self.committed.clone();
        }
        self.round.push(token);
        if self.steer {
            let vocab = logits.dim(1);
            let token = self.reference.target_greedy(&self.round);
            return Ok((next, one_hot_logits(token, vocab)));
        }
        Ok((next, logits))
    }

    fn commit(
        &mut self,
        target: &mut T,
        pending: u32,
        seed: &Array,
        drafts: &[u32],
        accepted: usize,
        verify_hidden: &Array,
    ) -> Result<(), Exception> {
        self.committed.push(pending);
        self.committed.extend_from_slice(&drafts[..accepted]);
        self.inner
            .commit(target, pending, seed, drafts, accepted, verify_hidden)
    }
}

fn check_qwen_head<T: MtpTarget>(
    what: &str,
    target: &mut T,
    ck: &Checkpoint,
    head_model: &mut Qwen3NextMtpForCausalLM,
    want: &[u32],
) {
    let reference_target = ck.dynamic();
    let reference = FromScratchQwenHead {
        caches: QwenCaches::new(&reference_target),
        mtp: ck.mtp(&reference_target),
        target: reference_target,
    };
    let mut cache = head_model.create_cache(128);
    let mut head = CheckedQwenHead {
        inner: Qwen3NextMtpDrafter::new(head_model, &mut cache),
        reference,
        steer: false,
        committed: Vec::new(),
        round: Vec::new(),
        rounds_checked: 0,
    };
    for (budget, steer) in [(1, false), (3, false), (3, true)] {
        head.steer = steer;
        let out = decode(target, &mut head, budget);
        let what = format!("{what}, drafting {budget}, steered {steer}");
        assert_exact(&what, &out, want);
        if steer {
            let m = out.speculative_metrics.expect("metrics");
            assert!(m.accepted_draft_tokens > NEW_TOKENS / 2, "{what}: {m:?}");
        }
    }
    assert!(head.rounds_checked > 2, "{what}: no round checked");
}

#[test]
#[serial]
fn qwen_mtp_on_the_dynamic_model() {
    let ck = qwen_checkpoint();
    let mut model = ck.dynamic();
    let vocab = model.config.vocab_size;
    let mut head = ck.mtp(&model);
    let mut caches = QwenCaches::new(&model);
    let mut target = caches.verifier(&mut model);
    let want = plain_greedy(&mut target, &PROMPT, NEW_TOKENS);

    check_qwen_head("dynamic Qwen", &mut target, &ck, &mut head, &want);

    let mut oracle_model = ck.dynamic();
    let mut oracle_caches = QwenCaches::new(&oracle_model);
    check_oracle(
        "dynamic Qwen",
        &mut target,
        oracle_caches.verifier(&mut oracle_model),
        vocab,
        &want,
    );
}

#[test]
#[serial]
fn qwen_mtp_on_the_native_engine() {
    let ck = qwen_checkpoint();
    let config_model = ck.dynamic();
    let vocab = config_model.config.vocab_size;
    let mut head = ck.mtp(&config_model);
    let mut target = ck.native();
    let want = plain_greedy(&mut target, &PROMPT, NEW_TOKENS);

    check_qwen_head("native Qwen", &mut target, &ck, &mut head, &want);
    check_oracle("native Qwen", &mut target, ck.native(), vocab, &want);
}

/// The production entry point, as `pmetal infer --mtp` calls it.
#[test]
#[serial]
fn qwen_mtp_entry_point_is_exact() {
    let ck = qwen_checkpoint();
    let mut model = ck.dynamic();
    let mut head = ck.mtp(&model);
    let mut caches = QwenCaches::new(&model);
    let want = plain_greedy(&mut caches.verifier(&mut model), &PROMPT, NEW_TOKENS);
    let mut mtp_cache = head.create_cache(128);
    let out = pmetal_models::generate_qwen3_next_mtp_streaming(
        &mut model,
        &mut head,
        &PROMPT,
        greedy_config(),
        &mut caches.kv,
        &mut caches.mamba,
        &mut mtp_cache,
        pmetal_models::Qwen3NextMtpConfig {
            num_draft_tokens: 3,
        },
        |_| true,
    )
    .expect("decode");
    drain("decode");
    assert_exact("generate_qwen3_next_mtp_streaming", &out, &want);

    let out = pmetal_models::generate_qwen3_native_mtp_streaming(
        ck.dir.path(),
        &mut head,
        &mut mtp_cache,
        &PROMPT,
        greedy_config(),
        pmetal_models::Qwen3NextMtpConfig {
            num_draft_tokens: 3,
        },
        |_| true,
    )
    .expect("decode");
    drain("native decode");
    let want_native = plain_greedy(&mut ck.native(), &PROMPT, NEW_TOKENS);
    assert_exact("generate_qwen3_native_mtp_streaming", &out, &want_native);
}

// ---------------------------------------------------------------------------
// Gemma 4 (tiny random target and assistant)
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

const GEMMA_LAYERS: [&str; 4] = [
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "full_attention",
];

/// The same random target every call.
fn gemma_target() -> Gemma4ForCausalLM {
    random::seed(11);
    Gemma4ForCausalLM::new(gemma_config(&GEMMA_LAYERS, 32, 2)).expect("target")
}

fn gemma_cache(model: &Gemma4ForCausalLM) -> KVCache {
    KVCache::new(KVCacheConfig::new(
        model.config.num_hidden_layers as usize,
        128,
        model.config.num_key_value_heads as usize,
        model.config.head_dim as usize,
    ))
}

fn gemma_assistant() -> Gemma4AssistantForCausalLM {
    random::seed(12);
    Gemma4AssistantForCausalLM::new(Gemma4AssistantConfig {
        model_type: "gemma4_assistant".to_string(),
        text_config: gemma_config(&["sliding_attention", "full_attention"], 16, 2),
        backbone_hidden_size: 32,
        use_ordered_embeddings: false,
        num_centroids: 2048,
        centroid_intermediate_top_k: 32,
        tie_word_embeddings: true,
    })
    .expect("assistant")
}

#[test]
#[serial]
fn gemma4_mtp_is_exact() {
    let mut model = gemma_target();
    let mut cache = gemma_cache(&model);
    let mut assistant = gemma_assistant();
    let want = plain_greedy(
        &mut Gemma4MtpVerifier::new(&mut model, &mut cache),
        &PROMPT,
        NEW_TOKENS,
    );

    for budget in [1, 3, 6] {
        let out = pmetal_models::generate_gemma4_mtp_streaming(
            &mut model,
            &mut assistant,
            &PROMPT,
            greedy_config(),
            &mut cache,
            pmetal_models::Gemma4MtpConfig {
                num_assistant_tokens: budget,
            },
            |_| true,
        )
        .expect("decode");
        drain("decode");
        assert_exact(&format!("Gemma 4, drafting {budget}"), &out, &want);
    }

    let reference_model = gemma_target();
    let mut head = CheckedGemmaHead {
        reference_cache: gemma_cache(&reference_model),
        reference_model,
        reference_assistant: gemma_assistant(),
        inner: Gemma4MtpDrafter::new(&model, &mut assistant).expect("drafter"),
        steer: false,
        committed: Vec::new(),
        round: Vec::new(),
        rounds_checked: 0,
    };
    let mut target = Gemma4MtpVerifier::new(&mut model, &mut cache);
    for (budget, steer) in [(3, false), (4, true)] {
        head.steer = steer;
        let out = decode(&mut target, &mut head, budget);
        let what = format!("Gemma 4, drafting {budget}, steered {steer}");
        assert_exact(&what, &out, &want);
        if steer {
            let m = out.speculative_metrics.expect("metrics");
            assert!(m.accepted_draft_tokens > NEW_TOKENS / 2, "{what}: {m:?}");
        }
    }
    assert!(head.rounds_checked > 2, "Gemma 4: no round checked");

    let mut oracle_model = gemma_target();
    let mut oracle_cache = gemma_cache(&oracle_model);
    check_oracle(
        "Gemma 4",
        &mut target,
        Gemma4MtpVerifier::new(&mut oracle_model, &mut oracle_cache),
        GEMMA_VOCAB,
        &want,
    );
}

/// The Gemma 4 assistant, checked at each round's first step against a
/// second copy run by hand the way the reference candidate generator runs
/// it: the target's scaled embedding of the pending token beside the target
/// state that predicted it, attending to a second target's cache over the
/// committed tokens (the last non-shared sliding and full layers, 0 and 1
/// here), at the pending token's position. Steered, it drafts the target's
/// own tokens, so rounds keep drafts.
struct CheckedGemmaHead<'a> {
    inner: Gemma4MtpDrafter<'a>,
    reference_model: Gemma4ForCausalLM,
    reference_cache: KVCache,
    reference_assistant: Gemma4AssistantForCausalLM,
    steer: bool,
    committed: Vec<u32>,
    round: Vec<u32>,
    rounds_checked: usize,
}

impl CheckedGemmaHead<'_> {
    fn first_draft_logits(&mut self, pending: u32) -> Array {
        let mut reference =
            Gemma4MtpVerifier::new(&mut self.reference_model, &mut self.reference_cache);
        let (states, _) = from_scratch(&mut reference, &self.committed);
        let seed = last_token_hidden(&states);
        let shared = Gemma4AssistantSharedKvStates {
            sliding_attention: self.reference_cache.get(0),
            full_attention: self.reference_cache.get(1),
        };
        let input = pmetal_bridge::decode::prompt_tokens_to_input(&[pending]);
        let embedding = self
            .reference_model
            .model
            .embed_tokens
            .forward(&input)
            .multiply(&Array::from_f32(self.reference_model.model.embed_scale));
        let (_, logits) = self
            .reference_assistant
            .forward_logits(
                &concatenate_axis(&[&embedding, &seed], -1),
                &shared,
                self.committed.len() as i32,
            )
            .expect("assistant");
        logits
    }
}

impl<'v> MtpDrafter<Gemma4MtpVerifier<'v>> for CheckedGemmaHead<'_> {
    fn prefill(
        &mut self,
        target: &mut Gemma4MtpVerifier<'v>,
        prompt: &[u32],
        hidden: &Array,
    ) -> Result<(), Exception> {
        self.committed = prompt.to_vec();
        self.inner.prefill(target, prompt, hidden)
    }

    fn step(
        &mut self,
        target: &mut Gemma4MtpVerifier<'v>,
        token: u32,
        hidden: &Array,
        step: usize,
        position: usize,
    ) -> Result<(Array, Array), Exception> {
        let (next, logits) = self.inner.step(target, token, hidden, step, position)?;
        if step == 0 {
            let want = self.first_draft_logits(token);
            assert_close(
                &format!("first draft logits at position {position}"),
                &logits,
                &want,
            );
            self.rounds_checked += 1;
            self.round = self.committed.clone();
        }
        self.round.push(token);
        if self.steer {
            let mut reference =
                Gemma4MtpVerifier::new(&mut self.reference_model, &mut self.reference_cache);
            let (_, target_logits) = from_scratch(&mut reference, &self.round);
            let token = greedy_token(&last_token_logits(&target_logits));
            return Ok((next, one_hot_logits(token, GEMMA_VOCAB)));
        }
        Ok((next, logits))
    }

    fn commit(
        &mut self,
        target: &mut Gemma4MtpVerifier<'v>,
        pending: u32,
        seed: &Array,
        drafts: &[u32],
        accepted: usize,
        verify_hidden: &Array,
    ) -> Result<(), Exception> {
        self.committed.push(pending);
        self.committed.extend_from_slice(&drafts[..accepted]);
        self.inner
            .commit(target, pending, seed, drafts, accepted, verify_hidden)
    }
}

// ---------------------------------------------------------------------------
// Penalties
// ---------------------------------------------------------------------------

/// Greedy decoding with the repetition, frequency and presence penalties on,
/// as `GenerationConfig::default()` and model-card presets turn them on.
fn penalized_config() -> GenerationConfig {
    GenerationConfig {
        repetition_penalty: 1.3,
        frequency_penalty: 0.5,
        presence_penalty: 1.5,
        ..greedy_config()
    }
}

/// Plain greedy decoding through the `Sampler`, which applies the penalties
/// before its argmax, recomputed from scratch every step.
fn plain_penalized<T: MtpTarget>(target: &mut T, prompt: &[u32], n: usize) -> Vec<u32> {
    let mut sampler = pmetal_models::generation::Sampler::new(penalized_config());
    let mut seq = prompt.to_vec();
    for _ in 0..n {
        let (_, logits) = from_scratch(target, &seq);
        let token = sampler
            .sample(&last_token_logits(&logits), &seq)
            .expect("sample");
        seq.push(token);
    }
    drain("plain penalized greedy");
    seq
}

/// The penalties change what greedy decoding picks, and greedy MTP decoding
/// picks what plain greedy decoding with the same penalties picks, on every
/// target and head.
#[test]
#[serial]
fn penalized_greedy_mtp_matches_penalized_plain_greedy() {
    let config = penalized_config();
    let ck = qwen_checkpoint();

    let mut model = ck.dynamic();
    let mut head = ck.mtp(&model);
    let mut caches = QwenCaches::new(&model);
    let want = plain_penalized(&mut caches.verifier(&mut model), &PROMPT, NEW_TOKENS);
    let unpenalized = plain_greedy(&mut caches.verifier(&mut model), &PROMPT, NEW_TOKENS);
    assert_ne!(
        want, unpenalized,
        "the penalties must change the greedy tokens"
    );
    let mut mtp_cache = head.create_cache(128);
    for budget in [1, 3] {
        let mut target = caches.verifier(&mut model);
        let mut drafter = Qwen3NextMtpDrafter::new(&mut head, &mut mtp_cache);
        let out =
            generate_mtp_streaming(&mut target, &mut drafter, &PROMPT, &config, budget, |_| {
                true
            })
            .expect("decode");
        drain("decode");
        let what = format!("dynamic Qwen, penalized, drafting {budget}");
        assert_exact(&what, &out, &want);
    }

    let mut native = ck.native();
    let want_native = plain_penalized(&mut native, &PROMPT, NEW_TOKENS);
    let mut drafter = Qwen3NextMtpDrafter::new(&mut head, &mut mtp_cache);
    let out = generate_mtp_streaming(&mut native, &mut drafter, &PROMPT, &config, 3, |_| true)
        .expect("decode");
    drain("native decode");
    assert_exact("native Qwen, penalized", &out, &want_native);

    let mut model = gemma_target();
    let mut cache = gemma_cache(&model);
    let mut assistant = gemma_assistant();
    let want = plain_penalized(
        &mut Gemma4MtpVerifier::new(&mut model, &mut cache),
        &PROMPT,
        NEW_TOKENS,
    );
    let out = pmetal_models::generate_gemma4_mtp_streaming(
        &mut model,
        &mut assistant,
        &PROMPT,
        config,
        &mut cache,
        pmetal_models::Gemma4MtpConfig {
            num_assistant_tokens: 3,
        },
        |_| true,
    )
    .expect("decode");
    drain("Gemma decode");
    assert_exact("Gemma 4, penalized", &out, &want);
}
