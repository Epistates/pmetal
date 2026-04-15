//! `pmetal dflash` — block-diffusion speculative decoding.
//!
//! This is a dedicated top-level command rather than a flag on `pmetal
//! infer` because the DFlash loop owns two models (target + draft) and has
//! its own verify/accept/rollback pipeline that doesn't plug into the
//! standard per-token generation loop.

use anyhow::{Context, Result};
use std::path::PathBuf;

use pmetal_data::chat_templates::{Message, detect_chat_template};
use pmetal_mlx::Array;
use pmetal_models::DynamicModel;
use pmetal_models::dflash_decoder::{
    DFlashConfig, DFlashDecoder, DFlashDraftQuant, load_dflash_draft_from_dir_quantized,
};
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
    let (draft, report) = load_dflash_draft_from_dir_quantized(&draft_path, draft_quant)
        .context("loading DFlash draft model")?;
    eprintln!(
        "[dflash] draft loaded: {} params, {} unused keys, target_layer_ids={:?}, block_size={}, mask_token_id={}",
        report.loaded,
        report.skipped.len(),
        draft.config.dflash_config.target_layer_ids,
        draft.config.block_size,
        draft.config.dflash_config.mask_token_id,
    );
    if !report.skipped.is_empty() {
        for skipped in &report.skipped {
            eprintln!("[dflash]   unused: {skipped}");
        }
    }

    let tokenizer = pmetal_data::Tokenizer::from_model_dir(&target_path)
        .context("loading tokenizer from target model dir")?;

    // DFlash drafts are trained against chat-templated targets. Running
    // without a template leaves the draft cross-attending to target
    // hidden states that are out-of-distribution, which collapses
    // acceptance to ~0. Match upstream dflash-mlx's behavior: always
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
        // `enable_thinking=false` / `no_thinking=true` matches upstream
        // dflash-mlx's adapter (which always passes enable_thinking=False)
        // so the rendered conversation ends at the assistant prompt with
        // no internal reasoning blocks.
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
        ..Default::default()
    };

    let use_tree = tree_budget > 0;
    if use_tree && !wants_native {
        eprintln!(
            "[dflash] --tree-budget requested but target is not on the native bridge; \
             falling back to linear DFlash"
        );
    }
    eprintln!(
        "[dflash] mode: {}",
        if use_tree && wants_native {
            format!("tree-verify (budget={tree_budget})")
        } else {
            "linear".to_string()
        }
    );

    let start = std::time::Instant::now();
    let output = if wants_native {
        // Fused native-bridge target: matches mlx-lm's parallel-replay
        // forward kernel-for-kernel, so the target forward is at parity
        // with upstream dflash-mlx. Falls back to the dynamic path if
        // the native loader rejects the checkpoint (e.g., quantized or
        // unsupported variant).
        match NativeQwen3Target::load(&target_path) {
            Ok(target) => {
                eprintln!("[dflash] target path: native bridge (qwen3_native)");
                let mut decoder = DFlashDecoder::new(target, draft);
                if use_tree {
                    decoder
                        .generate_ddtree(&prompt_arr, &config, tree_budget)
                        .map_err(|e| {
                            anyhow::anyhow!("dflash generate_ddtree (native): {e}")
                        })?
                } else {
                    decoder
                        .generate(&prompt_arr, &config)
                        .map_err(|e| anyhow::anyhow!("dflash generate (native): {e}"))?
                }
            }
            Err(native_err) => {
                eprintln!(
                    "[dflash] native bridge load failed ({native_err}); falling back to dynamic path"
                );
                let target = DynamicModel::load(&target_path)
                    .map_err(|e| anyhow::anyhow!("load {}: {e}", target_path.display()))?;
                let mut decoder = DFlashDecoder::new(target, draft);
                decoder
                    .generate(&prompt_arr, &config)
                    .map_err(|e| anyhow::anyhow!("dflash generate (dynamic): {e}"))?
            }
        }
    } else {
        eprintln!("[dflash] target path: dynamic (pmetal-models)");
        let target = DynamicModel::load(&target_path)
            .map_err(|e| anyhow::anyhow!("load {}: {e}", target_path.display()))?;
        let mut decoder = DFlashDecoder::new(target, draft);
        decoder
            .generate(&prompt_arr, &config)
            .map_err(|e| anyhow::anyhow!("dflash generate (dynamic): {e}"))?
    };
    let elapsed = start.elapsed();

    let prompt_len = prompt_ids.len();
    let generated = &output.tokens[prompt_len..];
    let generated_u32: Vec<u32> = generated.iter().map(|&i| i as u32).collect();
    let decoded = tokenizer
        .decode(&generated_u32)
        .context("decoding generated tokens")?;

    let tok_per_sec = if elapsed.as_secs_f32() > 0.0 {
        output.metrics.num_generated as f32 / elapsed.as_secs_f32()
    } else {
        0.0
    };

    if json {
        let obj = serde_json::json!({
            "prompt": prompt,
            "output": decoded,
            "num_generated": output.metrics.num_generated,
            "total_drafted": output.metrics.total_drafted,
            "total_accepted": output.metrics.total_accepted,
            "avg_acceptance_length": output.metrics.avg_acceptance_length(),
            "acceptance_rate": output.metrics.acceptance_rate(),
            "acceptance_lengths": output.metrics.acceptance_lengths,
            "elapsed_s": elapsed.as_secs_f32(),
            "tok_per_sec": tok_per_sec,
        });
        println!("{}", serde_json::to_string_pretty(&obj)?);
    } else {
        println!("{decoded}");
        eprintln!(
            "[dflash] {:.1} tok/s · {} drafted · {} accepted · avg accept len {:.2}",
            tok_per_sec,
            output.metrics.total_drafted,
            output.metrics.total_accepted,
            output.metrics.avg_acceptance_length()
        );
    }

    Ok(())
}

/// Download or locate a model on disk.
async fn resolve_model_path(path_or_id: &str, need_tokenizer: bool) -> Result<PathBuf> {
    if is_hf_like(path_or_id) {
        let path = pmetal_hub::download_model(path_or_id, None, None)
            .await
            .map_err(|e| anyhow::anyhow!("download_model {path_or_id}: {e}"))?;
        if need_tokenizer {
            let _ = pmetal_hub::download_file(path_or_id, "tokenizer.json", None, None).await;
            let _ =
                pmetal_hub::download_file(path_or_id, "tokenizer_config.json", None, None).await;
        }
        Ok(path)
    } else {
        Ok(PathBuf::from(path_or_id))
    }
}

fn is_hf_like(s: &str) -> bool {
    // HF model ids have the shape `owner/repo`. Local paths tend to contain
    // `/` too but also have a leading `.` or absolute prefix; err on the
    // side of treating anything starting with `.` or `/` as a local path.
    !(s.starts_with('.') || s.starts_with('/')) && s.contains('/')
}

