//! Speculative decoding of a loaded [`DynamicModel`]: the one place a target
//! model meets a drafter.
//!
//! Three kinds of drafter speed a decode up without changing what it says:
//!
//! * a Qwen 3.5-family model's own multi-token-prediction head
//!   ([`Qwen3NextMtpForCausalLM`], bundled in the checkpoint as `mtp.*` or
//!   shipped on its own), run by [`crate::mtp`];
//! * a Gemma 4 assistant, which drafts from the target's KV cache, also run by
//!   [`crate::mtp`];
//! * a DFlash draft model, which drafts a block in one pass from the target's
//!   hidden states, run by [`crate::dflash_decoder`].
//!
//! Whatever the drafter, the target checks every token, so the output is the
//! target's own: greedy decoding gives the tokens plain greedy decoding gives
//! and sampling draws from the target's distribution (rejection sampling for
//! the MTP heads, the target's own seeded draws for DFlash), through the same
//! [`DecodeSampler`] a plain decode uses: temperature, top-k, top-p, min-p,
//! the penalties, a logit bias and a grammar.
//!
//! A [`Speculator`] owns its drafter and is built once, beside the model, from
//! a [`SpeculatorSpec`]; each [`generate`](Speculator::generate) call decodes
//! one request on the model it is handed. The target runs on the plain
//! float KV cache a rewind needs, whatever cache mode the model's plain decode
//! would use.

use std::path::PathBuf;

use pmetal_cache::kv_cache::KVCache;
use pmetal_runtime::compat::Exception;
use pmetal_runtime::decode::DecodeSampler;

use crate::architectures::{
    Gemma4AssistantForCausalLM, Qwen3NextMtpForCausalLM, load_gemma4_assistant_from_dir,
    load_qwen3_next_mtp_from_dir,
};
use crate::dflash_decoder::{DFlashConfig, DFlashDecoder};
use crate::dflash_drafter::is_dflash_draft;
use crate::dflash_drafts::{DFlashDraft, DFlashDraftQuant, load_dflash_draft};
use crate::dispatcher::DynamicModel;
use crate::gemma4_mtp::{
    Gemma4MtpConfig, Gemma4MtpDrafter, Gemma4MtpVerifier, validate_gemma4_mtp_pair,
};
use crate::generation::GenerationConfig;
use crate::mtp::generate_mtp_streaming_with;
use crate::qwen3_next_mtp::{
    Qwen3NextMtpDrafter, Qwen3NextMtpVerifier, validate_qwen3_next_mtp_pair,
};

/// Which drafter to load for a target.
#[derive(Debug, Clone)]
pub enum SpeculatorSpec {
    /// A Qwen 3.5-family model's MTP head, read from the checkpoint directory
    /// `checkpoint`: the target's own, whose `mtp.*` tensors are the head, or a
    /// head exported on its own. `draft_tokens` is how many tokens a round
    /// drafts.
    QwenMtp {
        checkpoint: PathBuf,
        draft_tokens: usize,
    },
    /// The draft model in `path`: a DFlash draft model, told by its
    /// `dflash_config`, or else a Gemma 4 assistant. `tokens` overrides how
    /// many tokens a round drafts (DFlash: the block size, within what the
    /// draft was trained for; the assistant: its own default otherwise).
    Draft {
        path: PathBuf,
        tokens: Option<usize>,
    },
}

/// What a speculative generation did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SpeculativeRun {
    /// Tokens emitted to the caller.
    pub num_generated: usize,
    /// Whether a stop token ended the generation.
    pub stopped_by_token: bool,
    /// Tokens the drafter proposed.
    pub drafted_tokens: usize,
    /// Proposed tokens the target accepted.
    pub accepted_tokens: usize,
}

impl SpeculativeRun {
    /// Fraction of the drafted tokens the target accepted.
    pub fn acceptance_rate(&self) -> f32 {
        if self.drafted_tokens == 0 {
            0.0
        } else {
            self.accepted_tokens as f32 / self.drafted_tokens as f32
        }
    }
}

enum Drafter {
    QwenMtp {
        head: Qwen3NextMtpForCausalLM,
        /// The head's own KV cache, reset by each generation.
        cache: KVCache,
        draft_tokens: usize,
    },
    Gemma4 {
        assistant: Gemma4AssistantForCausalLM,
        config: Gemma4MtpConfig,
    },
    /// `None` only while a generation has it lent to a decoder.
    DFlash {
        draft: Option<DFlashDraft>,
        block: Option<usize>,
    },
}

/// A drafter loaded for one target model. See the module docs.
pub struct Speculator {
    drafter: Drafter,
    /// Longest sequence (prompt and output) a generation holds.
    max_seq_len: usize,
}

fn mismatch(what: &str, target: &DynamicModel) -> Exception {
    Exception::custom(format!(
        "{what}, but this model is {:?}",
        target.architecture()
    ))
}

impl Speculator {
    /// Load the drafter `spec` names for `target`, checked against it. A
    /// generation holds at most `max_seq_len` tokens.
    ///
    /// # Errors
    ///
    /// The drafter is missing or unreadable, or does not fit `target`: an MTP
    /// head for another model family, a hidden or vocabulary size that
    /// differs.
    pub fn load(
        target: &DynamicModel,
        spec: &SpeculatorSpec,
        max_seq_len: usize,
    ) -> Result<Self, Exception> {
        let drafter = match spec {
            SpeculatorSpec::QwenMtp {
                checkpoint,
                draft_tokens,
            } => {
                let DynamicModel::Qwen3Next(model) = target else {
                    return Err(mismatch(
                        "MTP drafting is for Qwen 3.5-family models (Qwen3.5, 3.6, 3.8)",
                        target,
                    ));
                };
                let head = load_qwen3_next_mtp_from_dir(checkpoint, &model.config)?;
                validate_qwen3_next_mtp_pair(&model.config, &head)?;
                let cache = head.create_cache(max_seq_len);
                Drafter::QwenMtp {
                    head,
                    cache,
                    draft_tokens: (*draft_tokens).max(1),
                }
            }
            SpeculatorSpec::Draft { path, tokens } => {
                if is_dflash_draft(path) {
                    let draft = load_dflash_draft(path, DFlashDraftQuant::None)?;
                    Self::check_dflash_pair(target, &draft)?;
                    Drafter::DFlash {
                        draft: Some(draft),
                        block: *tokens,
                    }
                } else {
                    let DynamicModel::Gemma4(model) = target else {
                        return Err(mismatch(
                            "a draft model is a DFlash draft or a Gemma 4 assistant; \
                             the one in this directory is no DFlash draft, and an assistant \
                             drafts for Gemma 4 models",
                            target,
                        ));
                    };
                    let (assistant, generation) = load_gemma4_assistant_from_dir(path)?;
                    validate_gemma4_mtp_pair(model, &assistant)?;
                    let config = Gemma4MtpConfig {
                        num_assistant_tokens: tokens
                            .unwrap_or(generation.num_assistant_tokens)
                            .max(1),
                    };
                    Drafter::Gemma4 { assistant, config }
                }
            }
        };
        Ok(Self {
            drafter,
            max_seq_len,
        })
    }

    /// A Qwen 3.5-family MTP head the caller has loaded (see
    /// [`load_qwen3_next_mtp_from_dir`]), checked against `target`.
    pub fn from_qwen_mtp(
        target: &DynamicModel,
        head: Qwen3NextMtpForCausalLM,
        draft_tokens: usize,
        max_seq_len: usize,
    ) -> Result<Self, Exception> {
        let DynamicModel::Qwen3Next(model) = target else {
            return Err(mismatch(
                "MTP drafting is for Qwen 3.5-family models (Qwen3.5, 3.6, 3.8)",
                target,
            ));
        };
        validate_qwen3_next_mtp_pair(&model.config, &head)?;
        let cache = head.create_cache(max_seq_len);
        Ok(Self {
            drafter: Drafter::QwenMtp {
                head,
                cache,
                draft_tokens: draft_tokens.max(1),
            },
            max_seq_len,
        })
    }

    /// A Gemma 4 assistant the caller has loaded (see
    /// [`load_gemma4_assistant_from_dir`]), drafting `draft_tokens` a round,
    /// checked against `target`.
    pub fn from_gemma4_assistant(
        target: &DynamicModel,
        assistant: Gemma4AssistantForCausalLM,
        draft_tokens: usize,
        max_seq_len: usize,
    ) -> Result<Self, Exception> {
        let DynamicModel::Gemma4(model) = target else {
            return Err(mismatch(
                "a Gemma 4 assistant drafts for Gemma 4 models",
                target,
            ));
        };
        validate_gemma4_mtp_pair(model, &assistant)?;
        Ok(Self {
            drafter: Drafter::Gemma4 {
                assistant,
                config: Gemma4MtpConfig {
                    num_assistant_tokens: draft_tokens.max(1),
                },
            },
            max_seq_len,
        })
    }

    /// A DFlash draft model the caller has loaded, checked against `target`.
    pub fn from_dflash(
        target: &DynamicModel,
        draft: DFlashDraft,
        block: Option<usize>,
        max_seq_len: usize,
    ) -> Result<Self, Exception> {
        Self::check_dflash_pair(target, &draft)?;
        Ok(Self {
            drafter: Drafter::DFlash {
                draft: Some(draft),
                block,
            },
            max_seq_len,
        })
    }

    fn check_dflash_pair(target: &DynamicModel, draft: &DFlashDraft) -> Result<(), Exception> {
        let config = draft.config();
        if config.hidden_size != target.hidden_size() {
            return Err(Exception::custom(format!(
                "the DFlash draft's hidden size ({}) is not the target's ({})",
                config.hidden_size,
                target.hidden_size()
            )));
        }
        if config.vocab_size != target.vocab_size() {
            return Err(Exception::custom(format!(
                "the DFlash draft's vocabulary ({}) is not the target's ({})",
                config.vocab_size,
                target.vocab_size()
            )));
        }
        // Every architecture the draft can tap is one the target implements
        // the hooks for; the rest say so on their first forward, too late.
        match target {
            DynamicModel::Llama4(_)
            | DynamicModel::Cohere(_)
            | DynamicModel::Granite(_)
            | DynamicModel::NemotronH(_)
            | DynamicModel::Qwen4Exp(_)
            | DynamicModel::Mllama(_)
            | DynamicModel::Flux(_)
            | DynamicModel::Bert(_)
            | DynamicModel::DiffusionGemma(_) => Err(mismatch(
                "DFlash drafts for Qwen, Llama, Mistral, Gemma, Phi, DeepSeek and GPT-OSS targets",
                target,
            )),
            _ => Ok(()),
        }
    }

    /// What drafts: `"qwen-mtp"`, `"gemma4-assistant"` or `"dflash"`.
    pub fn kind(&self) -> &'static str {
        match self.drafter {
            Drafter::QwenMtp { .. } => "qwen-mtp",
            Drafter::Gemma4 { .. } => "gemma4-assistant",
            Drafter::DFlash { .. } => "dflash",
        }
    }

    /// Tokens a round drafts (DFlash: the block it verifies, its own and the
    /// drafts).
    pub fn draft_tokens(&self) -> usize {
        match &self.drafter {
            Drafter::QwenMtp { draft_tokens, .. } => *draft_tokens,
            Drafter::Gemma4 { config, .. } => config.num_assistant_tokens,
            Drafter::DFlash { draft, block } => block
                .or_else(|| draft.as_ref().map(DFlashDraft::block_size))
                .unwrap_or(0),
        }
    }

    /// Decode `input_ids` on `target`, drawing through `sampler` (which must
    /// continue `input_ids`; see [`GenerationConfig::sampler`]) and calling
    /// `on_token` with every token as it is decided, a round's together;
    /// `false` ends the generation. `gen_config` sets the length and the stop
    /// tokens, which are not emitted.
    ///
    /// # Errors
    ///
    /// `target` is not the model this was loaded for, or a forward failed.
    pub fn generate(
        &mut self,
        target: &mut DynamicModel,
        input_ids: &[u32],
        gen_config: &GenerationConfig,
        sampler: DecodeSampler,
        mut on_token: impl FnMut(u32) -> bool,
    ) -> Result<SpeculativeRun, Exception> {
        let max_seq_len = self.max_seq_len;
        match &mut self.drafter {
            Drafter::QwenMtp {
                head,
                cache,
                draft_tokens,
            } => {
                let mut kv = target.create_cache(max_seq_len);
                let mut mamba = target.create_mamba_cache().ok_or_else(|| {
                    Exception::custom("Qwen MTP needs the target's GDN state; it has none")
                })?;
                let DynamicModel::Qwen3Next(model) = target else {
                    return Err(mismatch("this speculator drafts for Qwen 3.5", target));
                };
                let mut verifier = Qwen3NextMtpVerifier::new(model, &mut kv, &mut mamba);
                let mut drafter = Qwen3NextMtpDrafter::new(head, cache);
                let out = generate_mtp_streaming_with(
                    &mut verifier,
                    &mut drafter,
                    input_ids,
                    gen_config,
                    sampler,
                    *draft_tokens,
                    &mut on_token,
                )?;
                Ok(Self::mtp_run(&out))
            }
            Drafter::Gemma4 { assistant, config } => {
                let mut kv = target.create_cache(max_seq_len);
                let DynamicModel::Gemma4(model) = target else {
                    return Err(mismatch("this speculator drafts for Gemma 4", target));
                };
                let mut drafter = Gemma4MtpDrafter::new(model, assistant)?;
                let mut verifier = Gemma4MtpVerifier::new(model, &mut kv);
                let out = generate_mtp_streaming_with(
                    &mut verifier,
                    &mut drafter,
                    input_ids,
                    gen_config,
                    sampler,
                    config.num_assistant_tokens,
                    &mut on_token,
                )?;
                Ok(Self::mtp_run(&out))
            }
            Drafter::DFlash { draft, block } => {
                let lent = draft
                    .take()
                    .ok_or_else(|| Exception::custom("the DFlash draft is lent out"))?;
                let mut decoder = DFlashDecoder::new(&mut *target, lent);
                let config = DFlashConfig {
                    speculative_tokens: *block,
                    ..DFlashConfig::new(gen_config.clone())
                };
                let result = decoder.generate_with(input_ids, &config, sampler, &mut on_token);
                // The draft goes back whatever the decode did.
                *draft = Some(decoder.into_draft());
                let out = result?;
                Ok(SpeculativeRun {
                    num_generated: out.metrics.num_generated,
                    stopped_by_token: out.stopped_by_token,
                    drafted_tokens: out.metrics.total_drafted,
                    accepted_tokens: out.metrics.total_accepted,
                })
            }
        }
    }

    fn mtp_run(out: &crate::generation::GenerationOutput) -> SpeculativeRun {
        let metrics = out.speculative_metrics.clone().unwrap_or_default();
        SpeculativeRun {
            num_generated: out.num_generated,
            stopped_by_token: out.stopped_by_token,
            drafted_tokens: metrics.drafted_tokens,
            accepted_tokens: metrics.accepted_draft_tokens,
        }
    }
}
