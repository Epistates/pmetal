//! `pmetal dflash` — block-diffusion speculative decoding.
//!
//! This is a dedicated top-level command rather than a flag on `pmetal
//! infer` because the DFlash loop owns two models (target + draft) and has
//! its own verify/accept/rollback pipeline that doesn't plug into the
//! standard per-token generation loop. Both generations of draft run here,
//! told apart by the checkpoint's declared architecture.

use anyhow::{Context, Result};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use pmetal_data::chat_templates::{Message, detect_chat_template};
use pmetal_mlx::Array;
use pmetal_models::DynamicModel;
use pmetal_models::dflash_decoder::{
    DFlashConfig, DFlashDecoder, DFlashDraftQuant, DFlashOutput, DFlashTarget,
};
use pmetal_models::dflash_drafts::load_dflash_draft;
use pmetal_models::dflash_native_target::NativeQwen3Target;

/// Run DFlash speculative decoding against a Qwen3 target.
#[allow(clippy::too_many_arguments)]
pub async fn run_dflash(
    target_model: &str,
    draft_model: &str,
    prompt: &str,
    max_new_tokens: usize,
    temperature: f32,
    speculative_tokens: Option<usize>,
    draft_fp8: bool,
    json: bool,
    no_chat: bool,
    tree_budget: usize,
    compare_greedy: bool,
) -> Result<()> {
    let target_path = resolve_model_path(target_model, /*need_tokenizer*/ true).await?;
    let draft_path = resolve_model_path(draft_model, /*need_tokenizer*/ false).await?;

    // Detect architecture up-front so we can route Qwen3 through the
    // fused native bridge. The dynamic DynamicModel::load path is still
    // used for non-Qwen3 targets, and as a fallback if the native load
    // fails (e.g., quantized checkpoints whose loader path hasn't been
    // wired into qwen3_native yet).
    let arch = pmetal_models::dispatcher::ModelArchitecture::detect(&target_path)
        .map_err(|e| anyhow::anyhow!("detect architecture: {e}"))?;
    eprintln!("[dflash] target architecture: {arch:?}");
    let wants_native = matches!(
        arch,
        pmetal_models::dispatcher::ModelArchitecture::Qwen3
            | pmetal_models::dispatcher::ModelArchitecture::Qwen3Next
    );

    let draft_quant = if draft_fp8 {
        DFlashDraftQuant::Fp8
    } else {
        DFlashDraftQuant::None
    };
    let draft =
        load_dflash_draft(&draft_path, draft_quant).context("loading DFlash draft model")?;
    eprintln!(
        "[dflash] draft loaded: DFlash {}, {} layers, target_layer_ids={:?}, block_size={}, mask_token_id={}",
        if draft.is_dflash2() { 2 } else { 1 },
        draft.num_layers(),
        draft.target_layer_ids(),
        draft.block_size(),
        draft.mask_token_id(),
    );
    if temperature > 0.0 {
        eprintln!(
            "[dflash] note: the draft/verify loop decodes greedily; --temperature is ignored"
        );
    }

    let tokenizer = pmetal_data::Tokenizer::from_model_dir(&target_path)
        .context("loading tokenizer from target model dir")?;

    // DFlash drafts are trained against chat-templated targets. Running
    // without a template leaves the draft cross-attending to target
    // hidden states that are out-of-distribution, which collapses
    // acceptance to ~0. Always
    // apply the model's chat template unless the caller explicitly
    // opts out with --no-chat.
    let prompt_ids: Vec<i32> = if no_chat {
        tokenizer
            .encode(prompt)
            .context("encoding prompt")?
            .into_iter()
            .map(|t| t as i32)
            .collect()
    } else {
        let template = detect_chat_template(&target_path, &target_path.to_string_lossy());
        let messages = [Message::user(prompt)];
        // `enable_thinking=false` / `no_thinking=true`, so the rendered
        // conversation ends at the assistant prompt with no internal
        // reasoning blocks.
        let rendered = template.apply_inference(&messages, true, None).text;
        if std::env::var_os("PMETAL_DFLASH_DEBUG_PROMPT").is_some() {
            eprintln!("[dflash debug] rendered prompt: {:?}", rendered);
        }
        tokenizer
            .encode_with_special_tokens(&rendered)
            .map_err(|e| anyhow::anyhow!("encoding chat-templated prompt: {e}"))?
            .into_iter()
            .map(|t| t as i32)
            .collect()
    };
    if prompt_ids.is_empty() {
        anyhow::bail!("tokenizer returned 0 tokens for prompt");
    }
    eprintln!(
        "[dflash] prompt tokens: {} ({})",
        prompt_ids.len(),
        if no_chat { "raw" } else { "chat-templated" }
    );

    let stop_tokens: Vec<i32> = tokenizer
        .eos_token_id()
        .into_iter()
        .map(|t| t as i32)
        .collect();

    let prompt_arr = Array::from_slice(prompt_ids.as_slice(), &[1, prompt_ids.len() as i32]);
    let config = DFlashConfig {
        max_new_tokens,
        temperature,
        stop_tokens,
        speculative_tokens,
    };

    let use_tree = tree_budget > 0;
    if use_tree && !wants_native {
        eprintln!(
            "[dflash] --tree-budget requested but target is not on the native bridge; \
             falling back to linear DFlash"
        );
    }
    if use_tree && draft.is_dflash2() {
        eprintln!(
            "[dflash] --tree-budget: a DFlash 2 draft proposes one path, so it verifies linearly"
        );
    }
    eprintln!(
        "[dflash] mode: {}",
        if use_tree && wants_native && !draft.is_dflash2() {
            format!("tree-verify (budget={tree_budget})")
        } else {
            "linear".to_string()
        }
    );

    let run = Run {
        prompt: &prompt_arr,
        config: &config,
        tree_budget,
        compare_greedy,
    };
    let decoded = if wants_native {
        // Fused native-bridge target: the parallel-replay verify forward
        // runs on the fused kernels. Falls back to the dynamic path if
        // the native loader rejects the checkpoint (e.g., quantized or
        // unsupported variant).
        match NativeQwen3Target::load(&target_path) {
            Ok(target) => {
                eprintln!("[dflash] target path: native bridge (qwen3_native)");
                run.decode(DFlashDecoder::new(target, draft))?
            }
            Err(native_err) => {
                eprintln!(
                    "[dflash] native bridge load failed ({native_err}); falling back to dynamic path"
                );
                let target = DynamicModel::load(&target_path)
                    .map_err(|e| anyhow::anyhow!("load {}: {e}", target_path.display()))?;
                run.decode(DFlashDecoder::new(target, draft))?
            }
        }
    } else {
        eprintln!("[dflash] target path: dynamic (pmetal-models)");
        let target = DynamicModel::load(&target_path)
            .map_err(|e| anyhow::anyhow!("load {}: {e}", target_path.display()))?;
        run.decode(DFlashDecoder::new(target, draft))?
    };
    let (output, elapsed) = (&decoded.dflash.0, decoded.dflash.1);

    let prompt_len = prompt_ids.len();
    let generated: Vec<u32> = output.tokens[prompt_len..]
        .iter()
        .map(|&i| i as u32)
        .collect();
    let text = tokenizer
        .decode(&generated)
        .context("decoding generated tokens")?;
    let tok_per_sec = rate(output, elapsed);

    // Plain greedy decoding is what DFlash must reproduce, token for token.
    let greedy = decoded.greedy.as_ref().map(|(greedy, secs)| Comparison {
        tok_per_sec: rate(greedy, *secs),
        identical: greedy.tokens == output.tokens,
        divergence: decoded.divergence,
    });

    if json {
        let mut obj = serde_json::json!({
            "prompt": prompt,
            "output": text,
            "num_generated": output.metrics.num_generated,
            "total_drafted": output.metrics.total_drafted,
            "total_accepted": output.metrics.total_accepted,
            "avg_acceptance_length": output.metrics.avg_acceptance_length(),
            "acceptance_rate": output.metrics.acceptance_rate(),
            "acceptance_lengths": output.metrics.acceptance_lengths,
            "elapsed_s": elapsed.as_secs_f32(),
            "tok_per_sec": tok_per_sec,
        });
        if let Some(greedy) = &greedy {
            obj["greedy"] = serde_json::json!({
                "tok_per_sec": greedy.tok_per_sec,
                "speedup": greedy.speedup(tok_per_sec),
                "identical": greedy.identical,
                "first_difference": greedy.divergence.map(|d| serde_json::json!({
                    "at": d.at,
                    "dflash_token": d.dflash_token,
                    "greedy_token": d.greedy_token,
                    "top_two": d.top_two,
                })),
            });
        }
        println!("{}", serde_json::to_string_pretty(&obj)?);
    } else {
        println!("{text}");
        eprintln!(
            "[dflash] {:.1} tok/s · {} drafted · {} accepted · avg accept len {:.2}",
            tok_per_sec,
            output.metrics.total_drafted,
            output.metrics.total_accepted,
            output.metrics.avg_acceptance_length()
        );
        if let Some(greedy) = &greedy {
            let verdict = match greedy.divergence {
                _ if greedy.identical => "identical tokens".to_string(),
                Some(d) => format!(
                    "first differs at generated token {} ({} vs greedy's {}), where the \
                     target's top two are {} at {:.4} and {} at {:.4}",
                    d.at,
                    d.dflash_token,
                    d.greedy_token,
                    d.top_two[0].0,
                    d.top_two[0].1,
                    d.top_two[1].0,
                    d.top_two[1].1
                ),
                None => "same tokens, different length".to_string(),
            };
            eprintln!(
                "[dflash] greedy: {:.1} tok/s, DFlash {:.2}x · {verdict}",
                greedy.tok_per_sec,
                greedy.speedup(tok_per_sec),
            );
        }
    }

    Ok(())
}

/// Generated tokens per second of wall time, prefill included.
fn rate(output: &DFlashOutput, elapsed: Duration) -> f32 {
    let secs = elapsed.as_secs_f32();
    if secs > 0.0 {
        output.metrics.num_generated as f32 / secs
    } else {
        0.0
    }
}

/// How DFlash's run compares with plain greedy decoding's.
struct Comparison {
    tok_per_sec: f32,
    identical: bool,
    divergence: Option<Divergence>,
}

/// Where DFlash's tokens and greedy decoding's first part ways.
#[derive(Clone, Copy)]
struct Divergence {
    /// Generated position.
    at: usize,
    dflash_token: i32,
    greedy_token: i32,
    /// The target's two likeliest tokens there and their logits, from one
    /// forward over the shared prefix: a near tie means the two runs'
    /// differently shaped forwards rounded it differently.
    top_two: [(i32, f32); 2],
}

impl Comparison {
    fn speedup(&self, dflash_tok_per_sec: f32) -> f32 {
        if self.tok_per_sec > 0.0 {
            dflash_tok_per_sec / self.tok_per_sec
        } else {
            0.0
        }
    }
}

/// One prompt to decode, and how.
struct Run<'a> {
    prompt: &'a Array,
    config: &'a DFlashConfig,
    tree_budget: usize,
    compare_greedy: bool,
}

/// The DFlash run, and plain greedy decoding's when compared, each with its
/// wall time.
struct Decoded {
    dflash: (DFlashOutput, Duration),
    greedy: Option<(DFlashOutput, Duration)>,
    divergence: Option<Divergence>,
}

impl Run<'_> {
    fn decode<T: DFlashTarget>(&self, mut decoder: DFlashDecoder<T>) -> Result<Decoded> {
        let dflash = |decoder: &mut DFlashDecoder<T>, config: &DFlashConfig| {
            if self.tree_budget > 0 {
                decoder.generate_ddtree(self.prompt, config, self.tree_budget)
            } else {
                decoder.generate(self.prompt, config)
            }
            .map_err(|e| anyhow::anyhow!("dflash generate: {e}"))
        };
        let greedy = |decoder: &mut DFlashDecoder<T>, config: &DFlashConfig| {
            decoder
                .generate_greedy(self.prompt, config)
                .map_err(|e| anyhow::anyhow!("greedy decode: {e}"))
        };
        if self.compare_greedy {
            // Both build kernels on first use; warm them so neither timed
            // run pays for that.
            let warm = DFlashConfig {
                max_new_tokens: 16,
                stop_tokens: Vec::new(),
                ..self.config.clone()
            };
            dflash(&mut decoder, &warm)?;
            greedy(&mut decoder, &warm)?;
        }
        let start = Instant::now();
        let output = dflash(&mut decoder, self.config)?;
        let dflash = (output, start.elapsed());
        let greedy = if self.compare_greedy {
            let start = Instant::now();
            let output = greedy(&mut decoder, self.config)?;
            Some((output, start.elapsed()))
        } else {
            None
        };
        let divergence = match &greedy {
            Some((greedy, _)) => self.divergence(&mut decoder, &dflash.0, greedy)?,
            None => None,
        };
        Ok(Decoded {
            dflash,
            greedy,
            divergence,
        })
    }

    fn divergence<T: DFlashTarget>(
        &self,
        decoder: &mut DFlashDecoder<T>,
        dflash: &DFlashOutput,
        greedy: &DFlashOutput,
    ) -> Result<Option<Divergence>> {
        let prompt_len = self.prompt.dim(1) as usize;
        let Some(at) = dflash.tokens[prompt_len..]
            .iter()
            .zip(&greedy.tokens[prompt_len..])
            .position(|(a, b)| a != b)
        else {
            return Ok(None);
        };
        let end = prompt_len + at;
        let prefix = Array::from_slice(&greedy.tokens[..end], &[1, end as i32]);
        let top_two = decoder
            .next_token_top_two(&prefix)
            .map_err(|e| anyhow::anyhow!("top two: {e}"))?;
        Ok(Some(Divergence {
            at,
            dflash_token: dflash.tokens[end],
            greedy_token: greedy.tokens[end],
            top_two,
        }))
    }
}

/// Download or locate a model on disk. Pulls extra tokenizer files for the
/// target model so the speculative decoder can use them at runtime.
async fn resolve_model_path(path_or_id: &str, need_tokenizer: bool) -> Result<PathBuf> {
    let path = pmetal_hub::resolve_model_path(path_or_id, None, None)
        .await
        .map_err(|e| anyhow::anyhow!("resolve_model_path {path_or_id}: {e}"))?;
    if need_tokenizer && pmetal_hub::is_hf_id(path_or_id) {
        let _ = pmetal_hub::download_file(path_or_id, "tokenizer.json", None, None).await;
        let _ = pmetal_hub::download_file(path_or_id, "tokenizer_config.json", None, None).await;
    }
    Ok(path)
}
