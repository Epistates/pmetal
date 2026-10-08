//! Text generation on the ANE, built on [`super::extend`].
//!
//! The model's layers run as a few multi-layer extend programs over a KV cache
//! held in IOSurfaces, the final norm and logits as one more program, and the
//! token embedding is a lookup in an fp16 table on the CPU. Weights come
//! straight from a Hugging Face checkpoint, int8 by default, and compile once:
//! Apple's ANE service caches the programs for later runs.
//!
//! Every pass carries up to `width` tokens for about the cost of one, so a
//! prompt goes through `width` tokens at a time.

use std::path::Path;
use std::time::Instant;

use crate::ane::checkpoint::{self, Checkpoint};
use crate::ane::extend::{
    ExtendConfig, ExtendModel, LayerWeights, WeightFormat, gen_lm_head, lm_head_pieces,
};
use crate::ane::iosurface::IoSurface;
use crate::ane::runtime::{AneModel, AneRuntime};
use crate::error::{MetalError, Result};

/// How to build an [`AneLm`].
#[derive(Debug, Clone)]
pub struct AneLmOptions {
    /// Tokens per pass: a prompt goes through this many at a time, and a
    /// decode pass verifies up to this many less one guesses. Attention costs
    /// in proportion, so 16 (a DFlash block) decodes 15-25% faster than 32 on
    /// Qwen3-4B, while a prompt takes twice the passes.
    pub width: usize,
    /// KV cache slots: the longest prompt plus output the model can run.
    /// Attention reads every slot, so a smaller cache is faster.
    pub capacity: usize,
    /// Projection weight storage.
    pub weights: WeightFormat,
    /// Weight bytes per layer program, which decides how many layers each
    /// holds. About 1 GB works well.
    pub max_program_bytes: usize,
    /// Layers whose output a [`Drafter`] reads ([`Drafter::taps`]),
    /// ascending; empty for none.
    pub taps: Vec<usize>,
}

impl Default for AneLmOptions {
    fn default() -> Self {
        Self {
            width: 16,
            capacity: 2048,
            weights: WeightFormat::Int8,
            max_program_bytes: 1 << 30,
            taps: Vec::new(),
        }
    }
}

/// Whether [`AneLm`] implements the model `config` (a `config.json`)
/// describes: Qwen3-shaped dense models.
pub fn check_supported(config: &serde_json::Value) -> std::result::Result<(), String> {
    checkpoint::check_model_type(config, &["qwen3"], "inference")?;
    checkpoint::check_architecture_limits(config, "inference")
}

/// How [`AneLm::generate`] decodes.
#[derive(Debug, Clone)]
pub struct GenerateOptions {
    /// Most tokens to generate.
    pub max_new: usize,
    /// Sampling temperature; 0 is greedy.
    pub temperature: f32,
    /// Sample among the `top_k` most likely tokens (0 for all).
    pub top_k: usize,
    /// Tokens that end generation (and are returned).
    pub stop: Vec<u32>,
}

/// Guesses at what follows a context, for [`AneLm::generate`] to verify.
pub trait Drafter {
    /// Layers whose output (the residual stream after them) the drafter
    /// reads, ascending. The model must be built with them in
    /// [`AneLmOptions::taps`].
    fn taps(&self) -> &[usize] {
        &[]
    }

    /// Forget everything observed: a new generation starts.
    fn reset(&mut self) {}

    /// The tapped layers' output for the next `n` tokens of context, as they
    /// are decided: `[n, taps * dim]`, token-major, the taps in order. Every
    /// context token is observed once, in order, before a draft follows it,
    /// except the last, which [`draft`](Self::draft) is asked to continue.
    fn observe(&mut self, hidden: &[f32], n: usize) -> Result<()> {
        let _ = (hidden, n);
        Ok(())
    }

    /// Up to `max` tokens likely to follow `context`.
    fn draft(&mut self, context: &[u32], max: usize) -> Result<Vec<u32>>;
}

/// No guesses: one token per pass.
pub struct NoDraft;

impl Drafter for NoDraft {
    fn draft(&mut self, _: &[u32], _: usize) -> Result<Vec<u32>> {
        Ok(Vec::new())
    }
}

/// Prompt lookup: find the last `n` tokens earlier in the context (the
/// longest `n` from `max_ngram` down to `min_ngram` that matches) and guess
/// the tokens that followed them. No draft model; it pays off wherever
/// output repeats its input, as in code edits, quoting and structured text.
pub struct PromptLookup {
    /// Longest n-gram to match.
    pub max_ngram: usize,
    /// Shortest n-gram to match.
    pub min_ngram: usize,
}

impl Default for PromptLookup {
    fn default() -> Self {
        Self {
            max_ngram: 4,
            min_ngram: 2,
        }
    }
}

impl Drafter for PromptLookup {
    fn draft(&mut self, context: &[u32], max: usize) -> Result<Vec<u32>> {
        for n in (self.min_ngram..=self.max_ngram).rev() {
            if max == 0 || context.len() <= n {
                continue;
            }
            let tail = &context[context.len() - n..];
            // The most recent earlier occurrence.
            if let Some(at) = (0..context.len() - n)
                .rev()
                .find(|&i| &context[i..i + n] == tail)
            {
                let from = at + n;
                return Ok(context[from..(from + max).min(context.len())].to_vec());
            }
        }
        Ok(Vec::new())
    }
}

/// Statistics for one [`AneLm::generate`] call.
#[derive(Debug, Clone, Default)]
pub struct GenerationStats {
    /// Prompt tokens.
    pub prompt_tokens: usize,
    /// Tokens generated.
    pub generated_tokens: usize,
    /// Seconds spent on the prompt.
    pub prefill_secs: f64,
    /// Seconds spent generating.
    pub decode_secs: f64,
    /// Of `decode_secs`, the seconds the drafter took (drafting and
    /// observing).
    pub draft_secs: f64,
    /// Decode passes run.
    pub passes: usize,
    /// Guesses verified.
    pub drafted_tokens: usize,
    /// Guesses kept.
    pub accepted_tokens: usize,
}

/// A language model compiled for the ANE.
pub struct AneLm {
    cfg: ExtendConfig,
    vocab: usize,
    layers: ExtendModel,
    head: AneModel,
    head_in: IoSurface,
    /// The logits, one surface per LM head piece, each `[rows, W]`.
    head_out: Vec<IoSurface>,
    /// `[vocab, dim]` fp16.
    embed: Vec<half::f16>,
    taps: Vec<usize>,
}

impl AneLm {
    /// Load and compile the model in `dir`.
    pub fn load(dir: &Path, opts: &AneLmOptions) -> Result<Self> {
        let config: serde_json::Value = {
            let text = std::fs::read_to_string(dir.join("config.json"))
                .map_err(|e| MetalError::InvalidConfig(format!("config.json: {e}")))?;
            serde_json::from_str(&text)
                .map_err(|e| MetalError::InvalidConfig(format!("config.json: {e}")))?
        };
        check_supported(&config).map_err(MetalError::InvalidConfig)?;
        let get = |key: &str| -> Result<usize> {
            config
                .get(key)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .ok_or_else(|| MetalError::InvalidConfig(format!("config.json has no {key}")))
        };
        let (dim, n_heads) = (get("hidden_size")?, get("num_attention_heads")?);
        let cfg = ExtendConfig {
            dim,
            hidden_dim: get("intermediate_size")?,
            n_heads,
            n_kv_heads: get("num_key_value_heads").unwrap_or(n_heads),
            head_dim: get("head_dim").unwrap_or(dim / n_heads),
            width: opts.width,
            capacity: opts.capacity,
            rms_norm_eps: config["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
            weights: opts.weights,
        };
        let n_layers = get("num_hidden_layers")?;
        let vocab = get("vocab_size")?;
        let rope_theta = config["rope_theta"].as_f64().unwrap_or(1e6) as f32;
        let tied = config["tie_word_embeddings"].as_bool().unwrap_or(true);

        let ckpt = Checkpoint::open(dir)?;
        let (d, h, qd, kvd, hd) = (dim, cfg.hidden_dim, cfg.q_dim(), cfg.kv_dim(), cfg.head_dim);

        // Embedding table, final norm and LM head.
        let embed_f32 = ckpt.f32("model.embed_tokens.weight", vocab * d)?;
        let embed: Vec<half::f16> = embed_f32.iter().map(|&v| half::f16::from_f32(v)).collect();
        let head_weight = if tied || !ckpt.contains("lm_head.weight") {
            embed_f32
        } else {
            drop(embed_f32);
            ckpt.f32("lm_head.weight", vocab * d)?
        };
        let final_norm = ckpt.f32("model.norm.weight", d)?;
        let rt = AneRuntime::global()?;
        let head_kernel = gen_lm_head(&cfg, &final_norm, &head_weight, vocab)?;
        drop(head_weight);
        let head = rt.compile(head_kernel.mil_text.as_bytes(), Some(&head_kernel.weights))?;
        drop(head_kernel);

        // Layers, a program's worth at a time.
        let per_weight = match opts.weights {
            WeightFormat::Fp16 => 2,
            WeightFormat::Int8 => 1,
        };
        let layer_bytes = (2 * qd * d + 2 * kvd * d + 3 * h * d) * per_weight;
        let per_program = (opts.max_program_bytes / layer_bytes).clamp(1, n_layers);
        let started = Instant::now();
        let layers = ExtendModel::compile(
            cfg.clone(),
            rope_theta,
            n_layers,
            per_program,
            &opts.taps,
            |i| {
                let t = |name: &str, n: usize| ckpt.f32(&format!("model.layers.{i}.{name}"), n);
                Ok(LayerWeights {
                    rms_att: t("input_layernorm.weight", d)?,
                    wq: t("self_attn.q_proj.weight", qd * d)?,
                    wk: t("self_attn.k_proj.weight", kvd * d)?,
                    wv: t("self_attn.v_proj.weight", kvd * d)?,
                    wo: t("self_attn.o_proj.weight", d * qd)?,
                    q_norm: t("self_attn.q_norm.weight", hd)?,
                    k_norm: t("self_attn.k_norm.weight", hd)?,
                    rms_ffn: t("post_attention_layernorm.weight", d)?,
                    w_gate: t("mlp.gate_proj.weight", h * d)?,
                    w_up: t("mlp.up_proj.weight", h * d)?,
                    w_down: t("mlp.down_proj.weight", d * h)?,
                })
            },
        )?;
        tracing::info!(
            layers = n_layers,
            per_program,
            secs = format!("{:.1}", started.elapsed().as_secs_f64()),
            "ANE layers ready"
        );

        Ok(Self {
            head_in: IoSurface::for_tensor(d, opts.width)?,
            head_out: lm_head_pieces(vocab, d)
                .into_iter()
                .map(|(_, rows)| IoSurface::for_tensor(rows, opts.width))
                .collect::<Result<_>>()?,
            cfg,
            vocab,
            layers,
            head,
            embed,
            taps: opts.taps.clone(),
        })
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab
    }

    /// Tokens per pass.
    pub fn width(&self) -> usize {
        self.cfg.width
    }

    /// KV cache slots.
    pub fn capacity(&self) -> usize {
        self.cfg.capacity
    }

    /// Tokens in the KV cache.
    pub fn position(&self) -> usize {
        self.layers.position()
    }

    /// Forget the cache from `pos` on.
    pub fn truncate(&mut self, pos: usize) {
        self.layers.truncate(pos);
    }

    /// Run `tokens` (at most [`width`](Self::width)) at the next positions
    /// and return their logits.
    pub fn step(&mut self, tokens: &[u32]) -> Result<Logits<'_>> {
        let (d, n) = (self.cfg.dim, tokens.len());
        let mut x = vec![0.0f32; d * n];
        for (t, &token) in tokens.iter().enumerate() {
            let row = self
                .embed
                .get(token as usize * d..(token as usize + 1) * d)
                .ok_or_else(|| {
                    MetalError::InvalidConfig(format!(
                        "token {token} is past the vocabulary ({})",
                        self.vocab
                    ))
                })?;
            for (c, v) in row.iter().enumerate() {
                x[c * n + t] = v.to_f32();
            }
        }
        let hidden = self.layers.extend(&x, n)?;
        let w = self.cfg.width;
        let mut padded = vec![0.0f32; d * w];
        for c in 0..d {
            padded[c * w..c * w + n].copy_from_slice(&hidden[c * n..(c + 1) * n]);
        }
        self.head_in.write_f32_as_fp16(&padded, d, w);
        let outputs: Vec<_> = self.head_out.iter().map(IoSurface::as_ptr).collect();
        self.head.evaluate(&[self.head_in.as_ptr()], &outputs)?;
        Ok(Logits {
            pieces: &self.head_out,
            dim: self.cfg.dim,
            vocab: self.vocab,
            width: w,
            n,
        })
    }

    /// Generate after `prompt`, starting from an empty cache, until
    /// `opts.max_new` tokens, a token in `opts.stop`, or `on_token` returning
    /// false. Returns the generated tokens.
    ///
    /// Each pass verifies the drafter's guesses at what follows alongside the
    /// next token, for about the cost of the next token alone, and keeps the
    /// guesses the model agrees with. The output is what decoding one token
    /// at a time would produce, in distribution when sampling (a guess is
    /// kept only when the token sampled at its position equals it).
    pub fn generate(
        &mut self,
        prompt: &[u32],
        opts: &GenerateOptions,
        drafter: &mut dyn Drafter,
        mut on_token: impl FnMut(u32) -> bool,
    ) -> Result<(Vec<u32>, GenerationStats)> {
        if prompt.is_empty() {
            return Err(MetalError::InvalidConfig("empty prompt".into()));
        }
        drafter.reset();
        // The drafter reads tapped layers only if the model was built with
        // exactly those.
        let tapping = !drafter.taps().is_empty();
        if tapping && drafter.taps() != self.taps.as_slice() {
            return Err(MetalError::InvalidConfig(format!(
                "the drafter reads layers {:?}; the model was built tapping {:?}",
                drafter.taps(),
                self.taps
            )));
        }
        let max_new = opts
            .max_new
            .min(self.capacity().saturating_sub(prompt.len()));
        let pick = |row: Vec<f32>| sample(&row, opts.temperature, opts.top_k);
        self.truncate(0);
        let mut stats = GenerationStats {
            prompt_tokens: prompt.len(),
            ..Default::default()
        };

        let started = Instant::now();
        let chunks: Vec<&[u32]> = prompt.chunks(self.width()).collect();
        let mut next = 0;
        for (i, chunk) in chunks.iter().enumerate() {
            let logits = self.step(chunk)?;
            if i + 1 == chunks.len() {
                next = pick(logits.row(chunk.len() - 1));
            }
            if tapping {
                drafter.observe(&self.layers.tapped(0..chunk.len()), chunk.len())?;
            }
        }
        stats.prefill_secs = started.elapsed().as_secs_f64();

        let started = Instant::now();
        let mut context = prompt.to_vec();
        let mut out = Vec::with_capacity(max_new);
        'decode: loop {
            // `next` is decided; emit it.
            out.push(next);
            context.push(next);
            if opts.stop.contains(&next) || !on_token(next) || out.len() >= max_new {
                break;
            }

            // Verify `next` and up to width - 1 guesses in one pass. The
            // cache has room for every token fed, so a guess can't push the
            // output past max_new.
            let room = (max_new - out.len()).min(self.capacity() - self.position() - 1);
            let mut feed = vec![next];
            let drafting = Instant::now();
            feed.extend(drafter.draft(&context, room.min(self.width() - 1))?);
            stats.draft_secs += drafting.elapsed().as_secs_f64();
            let start = self.position();
            let logits = self.step(&feed)?;
            stats.passes += 1;
            stats.drafted_tokens += feed.len() - 1;
            let mut kept = 0;
            next = pick(logits.row(0));
            while kept + 1 < feed.len() && next == feed[kept + 1] {
                kept += 1;
                next = pick(logits.row(kept));
            }
            self.truncate(start + 1 + kept);
            stats.accepted_tokens += kept;
            if tapping {
                let observing = Instant::now();
                drafter.observe(&self.layers.tapped(0..kept + 1), kept + 1)?;
                stats.draft_secs += observing.elapsed().as_secs_f64();
            }
            for &token in &feed[1..=kept] {
                out.push(token);
                context.push(token);
                if opts.stop.contains(&token) || !on_token(token) || out.len() >= max_new {
                    break 'decode;
                }
            }
        }
        stats.decode_secs = started.elapsed().as_secs_f64();
        stats.generated_tokens = out.len();
        Ok((out, stats))
    }
}

/// Sample a token: greedy at `temperature` 0.
fn sample(logits: &[f32], temperature: f32, top_k: usize) -> u32 {
    crate::ane::inference::sample(logits, temperature, top_k)
}

/// The logits from one [`AneLm::step`].
pub struct Logits<'a> {
    pieces: &'a [IoSurface],
    dim: usize,
    vocab: usize,
    width: usize,
    n: usize,
}

impl Logits<'_> {
    /// Positions with logits.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether there are none.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Position `t`'s logits, `[vocab]`.
    pub fn row(&self, t: usize) -> Vec<f32> {
        assert!(t < self.n, "position {t} of {}", self.n);
        let w = self.width;
        let mut row = Vec::with_capacity(self.vocab);
        for (piece, (_, rows)) in self.pieces.iter().zip(lm_head_pieces(self.vocab, self.dim)) {
            piece.with_fp16(|bits| {
                row.extend((0..rows).map(|v| half::f16::from_bits(bits[v * w + t]).to_f32()));
            });
        }
        row
    }

    /// Position `t`'s most likely token.
    pub fn argmax(&self, t: usize) -> u32 {
        let row = self.row(t);
        let mut best = 0;
        for (i, v) in row.iter().enumerate() {
            if *v > row[best] {
                best = i;
            }
        }
        best as u32
    }
}
