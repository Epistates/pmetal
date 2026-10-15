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
    /// Tokens that end generation (and are returned).
    pub stop: Vec<u32>,
}

/// Decides each generated token from the logits at its position.
///
/// A generation asks for its tokens in order, one call per position: the
/// tokens returned before a call are exactly the tokens that precede its
/// position. So a sampler whose decision reads the tokens before it (a
/// repetition penalty) records each token it returns, and a speculative
/// decoder never shows it a position past a guess it would reject. Only the
/// last call's token may go unused, when generation stops.
pub trait TokenSampler {
    /// The token at the next position, from its `[vocab]` logits.
    fn sample(&mut self, logits: &[f32]) -> Result<u32>;
}

/// The most likely token, the first on a tie.
pub struct Greedy;

impl TokenSampler for Greedy {
    fn sample(&mut self, logits: &[f32]) -> Result<u32> {
        Ok(argmax(logits))
    }
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
    /// Positions the last [`step`](AneLm::step) ran, whose logits the head
    /// surfaces hold.
    last_n: usize,
}

/// The tensors one layer reads, under `model.layers.{i}.`.
const LAYER_TENSORS: [&str; 11] = [
    "input_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
];

/// What a `config.json` says about the model, checked against what the
/// engine implements and the options it's built with.
#[derive(Debug)]
struct ModelShape {
    cfg: ExtendConfig,
    n_layers: usize,
    vocab: usize,
    rope_theta: f32,
    /// Whether the logits come from the embedding table.
    tied: bool,
}

impl ModelShape {
    fn from_config(config: &serde_json::Value, opts: &AneLmOptions) -> Result<Self> {
        let invalid = |why: String| MetalError::InvalidConfig(why);
        check_supported(config).map_err(invalid)?;
        let get = |key: &str| -> Result<usize> {
            config
                .get(key)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .ok_or_else(|| invalid(format!("config.json has no {key}")))
        };
        let (dim, n_heads) = (get("hidden_size")?, get("num_attention_heads")?);
        if n_heads == 0 {
            return Err(invalid("config.json: num_attention_heads is 0".into()));
        }
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
        if !opts.taps.is_sorted_by(|a, b| a < b) || opts.taps.last().is_some_and(|&t| t >= n_layers)
        {
            return Err(invalid(format!(
                "taps {:?} must ascend within the model's {n_layers} layers",
                opts.taps
            )));
        }
        // A cache longer than the context the model was trained for would
        // attend to positions it has never seen.
        if let Some(trained) = config
            .get("max_position_embeddings")
            .and_then(|v| v.as_u64())
        {
            if opts.capacity as u64 > trained {
                return Err(invalid(format!(
                    "KV capacity {} is past the model's {trained}-token context \
                     (max_position_embeddings)",
                    opts.capacity
                )));
            }
        }
        Ok(Self {
            cfg,
            n_layers,
            vocab: get("vocab_size")?,
            // The Hugging Face default for Qwen3 when a config names none.
            rope_theta: checkpoint::rope_theta(config).unwrap_or(1e6) as f32,
            // Hugging Face defaults tie_word_embeddings to true.
            tied: config
                .get("tie_word_embeddings")
                .and_then(|v| v.as_bool())
                .unwrap_or(true),
        })
    }
}

/// The tensor the logits are computed from, which a checkpoint must have.
fn head_tensor(tied: bool, has_lm_head: bool) -> Result<&'static str> {
    match (tied, has_lm_head) {
        (true, _) => Ok("model.embed_tokens.weight"),
        (false, true) => Ok("lm_head.weight"),
        // Falling back to the embedding table would run a different model.
        (false, false) => Err(MetalError::InvalidConfig(
            "config.json says tie_word_embeddings is false, but the checkpoint has no \
             lm_head.weight"
                .into(),
        )),
    }
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
        let ModelShape {
            cfg,
            n_layers,
            vocab,
            rope_theta,
            tied,
        } = ModelShape::from_config(&config, opts)?;

        let ckpt = Checkpoint::open(dir)?;
        let (d, h, qd, kvd, hd) = (
            cfg.dim,
            cfg.hidden_dim,
            cfg.q_dim(),
            cfg.kv_dim(),
            cfg.head_dim,
        );
        // Every tensor, before the first (slow) compile.
        let head_name = head_tensor(tied, ckpt.contains("lm_head.weight"))?;
        ckpt.require(
            ["model.embed_tokens.weight", "model.norm.weight", head_name]
                .map(String::from)
                .into_iter()
                .chain(checkpoint::layer_tensors(n_layers, &LAYER_TENSORS)),
        )?;

        // Embedding table, final norm and LM head.
        let embed_f32 = ckpt.f32("model.embed_tokens.weight", vocab * d)?;
        let embed: Vec<half::f16> = embed_f32.iter().map(|&v| half::f16::from_f32(v)).collect();
        let head_weight = if tied {
            embed_f32
        } else {
            drop(embed_f32);
            ckpt.f32(head_name, vocab * d)?
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
            last_n: 0,
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
        self.last_n = n;
        Ok(self.logits())
    }

    /// The logits of the last [`step`](Self::step).
    fn logits(&self) -> Logits<'_> {
        Logits {
            pieces: &self.head_out,
            dim: self.cfg.dim,
            vocab: self.vocab,
            width: self.cfg.width,
            n: self.last_n,
        }
    }

    /// Generate after `prompt`, starting from an empty cache, until
    /// `opts.max_new` tokens, a token in `opts.stop`, or `on_token` returning
    /// false, each token from `sampler`. Returns the generated tokens.
    ///
    /// Each pass verifies the drafter's guesses at what follows alongside the
    /// next token, for about the cost of the next token alone, and keeps the
    /// guesses the model agrees with. The output is what decoding one token
    /// at a time would produce, in distribution when sampling: the sampler
    /// decides each position in turn, and a guess is kept only when the
    /// token it decides there equals it, so it never sees a position past a
    /// rejected guess ([`TokenSampler`]).
    pub fn generate(
        &mut self,
        prompt: &[u32],
        opts: &GenerateOptions,
        sampler: &mut dyn TokenSampler,
        drafter: &mut dyn Drafter,
        on_token: impl FnMut(u32) -> bool,
    ) -> Result<(Vec<u32>, GenerationStats)> {
        generate_loop(self, prompt, opts, sampler, drafter, on_token)
    }
}

/// What the generation loop asks of a model: [`AneLm`], or a stand-in the
/// loop's tests can run on the CPU.
trait Pass {
    /// Tokens per pass.
    fn width(&self) -> usize;
    /// KV cache slots.
    fn capacity(&self) -> usize;
    /// Tokens in the KV cache.
    fn position(&self) -> usize;
    /// Forget the cache from `pos` on.
    fn truncate(&mut self, pos: usize);
    /// The layers whose output the model keeps for a drafter.
    fn taps(&self) -> &[usize];
    /// Run `tokens` at the next positions, keeping their logits for
    /// [`row`](Self::row) and tapped output for [`tapped`](Self::tapped).
    fn run(&mut self, tokens: &[u32]) -> Result<()>;
    /// Position `t`'s logits from the last [`run`](Self::run), `[vocab]`.
    fn row(&self, t: usize) -> Vec<f32>;
    /// The tapped output for `tokens` of the last run, token-major.
    fn tapped(&self, tokens: std::ops::Range<usize>) -> Vec<f32>;
}

impl Pass for AneLm {
    fn width(&self) -> usize {
        self.cfg.width
    }
    fn capacity(&self) -> usize {
        self.cfg.capacity
    }
    fn position(&self) -> usize {
        self.layers.position()
    }
    fn truncate(&mut self, pos: usize) {
        self.layers.truncate(pos);
    }
    fn taps(&self) -> &[usize] {
        &self.taps
    }
    fn run(&mut self, tokens: &[u32]) -> Result<()> {
        self.step(tokens).map(|_| ())
    }
    fn row(&self, t: usize) -> Vec<f32> {
        self.logits().row(t)
    }
    fn tapped(&self, tokens: std::ops::Range<usize>) -> Vec<f32> {
        self.layers.tapped(tokens)
    }
}

/// [`AneLm::generate`] over any [`Pass`].
fn generate_loop<P: Pass>(
    pass: &mut P,
    prompt: &[u32],
    opts: &GenerateOptions,
    sampler: &mut dyn TokenSampler,
    drafter: &mut dyn Drafter,
    mut on_token: impl FnMut(u32) -> bool,
) -> Result<(Vec<u32>, GenerationStats)> {
    if prompt.is_empty() {
        return Err(MetalError::InvalidConfig("empty prompt".into()));
    }
    if prompt.len() > pass.capacity() {
        return Err(MetalError::InvalidConfig(format!(
            "the prompt is {} tokens; the KV cache holds {}",
            prompt.len(),
            pass.capacity()
        )));
    }
    drafter.reset();
    // The drafter reads tapped layers only if the model was built with
    // exactly those.
    let tapping = !drafter.taps().is_empty();
    if tapping && drafter.taps() != pass.taps() {
        return Err(MetalError::InvalidConfig(format!(
            "the drafter reads layers {:?}; the model was built tapping {:?}",
            drafter.taps(),
            pass.taps()
        )));
    }
    let mut stats = GenerationStats {
        prompt_tokens: prompt.len(),
        ..Default::default()
    };
    let max_new = opts.max_new.min(pass.capacity() - prompt.len());
    pass.truncate(0);
    if max_new == 0 {
        return Ok((Vec::new(), stats));
    }

    let started = Instant::now();
    let chunks: Vec<&[u32]> = prompt.chunks(pass.width()).collect();
    let mut next = 0;
    for (i, chunk) in chunks.iter().enumerate() {
        pass.run(chunk)?;
        if i + 1 == chunks.len() {
            next = sampler.sample(&pass.row(chunk.len() - 1))?;
        }
        if tapping {
            drafter.observe(&pass.tapped(0..chunk.len()), chunk.len())?;
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
        let room = (max_new - out.len()).min(pass.capacity() - pass.position() - 1);
        let mut feed = vec![next];
        let drafting = Instant::now();
        let max_guesses = room.min(pass.width() - 1);
        // A drafter asked for `max_guesses` may return more; the pass takes
        // no more than that.
        feed.extend(
            drafter
                .draft(&context, max_guesses)?
                .into_iter()
                .take(max_guesses),
        );
        stats.draft_secs += drafting.elapsed().as_secs_f64();
        let start = pass.position();
        pass.run(&feed)?;
        stats.passes += 1;
        stats.drafted_tokens += feed.len() - 1;
        // The sampler decides the position after `next`, then, while it
        // agrees with the guess there, the one after that guess.
        let mut kept = 0;
        next = sampler.sample(&pass.row(0))?;
        while kept + 1 < feed.len() && next == feed[kept + 1] {
            kept += 1;
            next = sampler.sample(&pass.row(kept))?;
        }
        pass.truncate(start + 1 + kept);
        stats.accepted_tokens += kept;
        if tapping {
            let observing = Instant::now();
            drafter.observe(&pass.tapped(0..kept + 1), kept + 1)?;
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

/// Index of the largest value; the first one on a tie.
fn argmax(values: &[f32]) -> u32 {
    let mut best = 0;
    for (i, v) in values.iter().enumerate() {
        if *v > values[best] {
            best = i;
        }
    }
    best as u32
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
        argmax(&self.row(t))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_the_first_max() {
        let mut greedy = Greedy;
        assert_eq!(greedy.sample(&[0.1, 0.5, 0.3, 0.9, 0.2]).unwrap(), 3);
        assert_eq!(
            greedy.sample(&[-1.0, -2.0, 5.0, -3.0, 4.0, 0.0]).unwrap(),
            2
        );
        assert_eq!(greedy.sample(&[0.0; 100]).unwrap(), 0);
    }

    // ---- The generation loop, on a model that runs on the CPU ----------

    const VOCAB: u32 = 50;

    /// The token a context leads to: a function of its last two tokens, so a
    /// model that has seen a different token before a position reads a
    /// different answer there.
    fn successor(context: &[u32]) -> u32 {
        let last = context[context.len() - 1];
        let before = context.len().checked_sub(2).map_or(0, |i| context[i]);
        (last * 31 + before * 7 + 3) % VOCAB
    }

    /// A [`Pass`] whose logits peak at [`successor`] of the tokens in its
    /// cache and fall off with distance from it, and whose tapped output for
    /// a token is its position.
    struct FakeLm {
        width: usize,
        capacity: usize,
        taps: Vec<usize>,
        cache: Vec<u32>,
        rows: Vec<Vec<f32>>,
        first_new: usize,
    }

    impl FakeLm {
        fn new(width: usize, capacity: usize, taps: Vec<usize>) -> Self {
            Self {
                width,
                capacity,
                taps,
                cache: Vec::new(),
                rows: Vec::new(),
                first_new: 0,
            }
        }
    }

    impl Pass for FakeLm {
        fn width(&self) -> usize {
            self.width
        }
        fn capacity(&self) -> usize {
            self.capacity
        }
        fn position(&self) -> usize {
            self.cache.len()
        }
        fn truncate(&mut self, pos: usize) {
            self.cache.truncate(pos);
        }
        fn taps(&self) -> &[usize] {
            &self.taps
        }
        fn run(&mut self, tokens: &[u32]) -> Result<()> {
            assert!(!tokens.is_empty() && tokens.len() <= self.width);
            if self.cache.len() + tokens.len() > self.capacity {
                return Err(MetalError::InvalidConfig("KV cache full".into()));
            }
            self.first_new = self.cache.len();
            self.cache.extend_from_slice(tokens);
            self.rows = (0..tokens.len())
                .map(|i| {
                    let best = successor(&self.cache[..self.first_new + i + 1]);
                    (0..VOCAB)
                        .map(|j| -(j as f32 - best as f32).abs())
                        .collect()
                })
                .collect();
            Ok(())
        }
        fn row(&self, t: usize) -> Vec<f32> {
            self.rows[t].clone()
        }
        fn tapped(&self, tokens: std::ops::Range<usize>) -> Vec<f32> {
            tokens.map(|t| (self.first_new + t) as f32).collect()
        }
    }

    /// Decides by the logits, avoiding the tokens it returned last: a
    /// sampler whose decision reads its own history, as a repetition
    /// penalty does. It also checks the contract that the tokens it has
    /// returned are exactly the tokens before the position it is asked
    /// about, which the model's logits encode.
    struct Avoiding {
        prompt: Vec<u32>,
        returned: Vec<u32>,
        window: usize,
    }

    impl Avoiding {
        fn new(prompt: &[u32], window: usize) -> Self {
            Self {
                prompt: prompt.to_vec(),
                returned: Vec::new(),
                window,
            }
        }
    }

    impl TokenSampler for Avoiding {
        fn sample(&mut self, logits: &[f32]) -> Result<u32> {
            let context: Vec<u32> = self.prompt.iter().chain(&self.returned).copied().collect();
            let model_saw = argmax(logits);
            assert_eq!(
                model_saw,
                successor(&context),
                "the sampler was shown a position that doesn't follow its own tokens"
            );
            let recent = &self.returned[self.returned.len().saturating_sub(self.window)..];
            let mut best: Option<(f32, u32)> = None;
            for (token, &logit) in logits.iter().enumerate() {
                let token = token as u32;
                if recent.contains(&token) {
                    continue;
                }
                if best.is_none_or(|(b, _)| logit > b) {
                    best = Some((logit, token));
                }
            }
            let token = best.unwrap().1;
            self.returned.push(token);
            Ok(token)
        }
    }

    /// Guesses from a closure `(context, max)`.
    struct Guesses<F: FnMut(&[u32], usize) -> Vec<u32>>(F);

    impl<F: FnMut(&[u32], usize) -> Vec<u32>> Drafter for Guesses<F> {
        fn draft(&mut self, context: &[u32], max: usize) -> Result<Vec<u32>> {
            Ok((self.0)(context, max))
        }
    }

    /// A drafter reading tap 0 (each token's position) that guesses via
    /// `inner` and records what it was shown.
    struct Watching<D: Drafter> {
        inner: D,
        seen: Vec<f32>,
        resets: usize,
    }

    impl<D: Drafter> Drafter for Watching<D> {
        fn taps(&self) -> &[usize] {
            &[0]
        }
        fn reset(&mut self) {
            self.resets += 1;
            self.seen.clear();
        }
        fn observe(&mut self, hidden: &[f32], n: usize) -> Result<()> {
            assert_eq!(hidden.len(), n);
            self.seen.extend_from_slice(hidden);
            Ok(())
        }
        fn draft(&mut self, context: &[u32], max: usize) -> Result<Vec<u32>> {
            self.inner.draft(context, max)
        }
    }

    /// The continuation the model would produce, one token at a time.
    fn walk(prompt: &[u32], n: usize) -> Vec<u32> {
        let mut context = prompt.to_vec();
        for _ in 0..n {
            context.push(successor(&context));
        }
        context[prompt.len()..].to_vec()
    }

    fn settings(max_new: usize) -> GenerateOptions {
        GenerateOptions {
            max_new,
            stop: Vec::new(),
        }
    }

    const PROMPT: [u32; 5] = [4, 17, 9, 33, 21];

    fn run_with(
        lm: &mut FakeLm,
        opts: &GenerateOptions,
        drafter: &mut dyn Drafter,
    ) -> (Vec<u32>, GenerationStats) {
        generate_loop(lm, &PROMPT, opts, &mut Greedy, drafter, |_| true).unwrap()
    }

    /// Guesses that are right, wrong, partly right, and (against the trait's
    /// word) more than asked for.
    fn drafters() -> Vec<(&'static str, Box<dyn Drafter>)> {
        let oracle = |context: &[u32], max: usize| walk(context, max);
        let wrong = |context: &[u32], max: usize| {
            walk(context, max)
                .into_iter()
                .map(|t| (t + 1) % VOCAB)
                .collect()
        };
        let partly = |context: &[u32], max: usize| {
            let mut guesses = walk(context, max);
            if let Some(g) = guesses.get_mut(2) {
                *g = (*g + 1) % VOCAB;
            }
            guesses
        };
        let too_many = |context: &[u32], max: usize| walk(context, max + 40);
        vec![
            ("none", Box::new(NoDraft)),
            ("prompt lookup", Box::new(PromptLookup::default())),
            ("oracle", Box::new(Guesses(oracle))),
            ("wrong", Box::new(Guesses(wrong))),
            ("partly right", Box::new(Guesses(partly))),
            ("too many", Box::new(Guesses(too_many))),
        ]
    }

    #[test]
    fn guesses_change_the_speed_not_the_tokens() {
        for width in [1, 2, 4, 16] {
            for max_new in [0, 1, 2, 3, 5, 15, 16, 17, 40] {
                let want = walk(&PROMPT, max_new);
                for (name, mut drafter) in drafters() {
                    let mut lm = FakeLm::new(width, 128, Vec::new());
                    let (got, stats) = run_with(&mut lm, &settings(max_new), drafter.as_mut());
                    assert_eq!(got, want, "width {width}, max_new {max_new}, {name}");
                    assert_eq!(stats.generated_tokens, want.len());
                    assert!(stats.accepted_tokens <= stats.drafted_tokens);
                }
            }
        }
    }

    #[test]
    fn right_guesses_cut_the_passes_and_wrong_ones_cost_nothing_but_the_pass() {
        let mut lm = FakeLm::new(8, 256, Vec::new());
        let mut oracle = Guesses(|context: &[u32], max| walk(context, max));
        let (out, stats) = run_with(&mut lm, &settings(64), &mut oracle);
        assert_eq!(out.len(), 64);
        assert_eq!(stats.accepted_tokens, stats.drafted_tokens);
        // 1 token + 7 guesses per pass, after the one from the prompt.
        assert!(stats.passes <= 9, "{} passes", stats.passes);

        let mut lm = FakeLm::new(8, 256, Vec::new());
        let mut wrong = Guesses(|context: &[u32], max| {
            walk(context, max)
                .into_iter()
                .map(|t| (t + 1) % VOCAB)
                .collect()
        });
        let (_, stats) = run_with(&mut lm, &settings(64), &mut wrong);
        assert_eq!(stats.accepted_tokens, 0);
        assert_eq!(stats.passes, 63, "one pass per token");
    }

    #[test]
    fn a_stateful_sampler_sees_only_positions_that_follow_its_own_tokens() {
        let reference = {
            let mut lm = FakeLm::new(1, 128, Vec::new());
            let mut sampler = Avoiding::new(&PROMPT, 3);
            generate_loop(
                &mut lm,
                &PROMPT,
                &settings(40),
                &mut sampler,
                &mut NoDraft,
                |_| true,
            )
            .unwrap()
            .0
        };
        assert_ne!(
            reference,
            walk(&PROMPT, 40),
            "the sampler changes the tokens"
        );
        for width in [2, 4, 16] {
            for (name, mut drafter) in drafters() {
                let mut lm = FakeLm::new(width, 128, Vec::new());
                let mut sampler = Avoiding::new(&PROMPT, 3);
                let (got, _) = generate_loop(
                    &mut lm,
                    &PROMPT,
                    &settings(40),
                    &mut sampler,
                    drafter.as_mut(),
                    |_| true,
                )
                .unwrap();
                assert_eq!(got, reference, "width {width}, {name}");
            }
        }
    }

    #[test]
    fn generation_stops_at_a_stop_token_and_when_the_caller_says() {
        let all = walk(&PROMPT, 40);
        let stop_at = 9;
        // A token that first appears at `stop_at`, so nothing earlier ends it.
        let stop = all[stop_at];
        let first = all.iter().position(|&t| t == stop).unwrap();
        for (name, mut drafter) in drafters() {
            let mut lm = FakeLm::new(8, 128, Vec::new());
            let opts = GenerateOptions {
                max_new: 40,
                stop: vec![stop],
            };
            let (got, stats) = run_with(&mut lm, &opts, drafter.as_mut());
            assert_eq!(got, all[..=first], "stop token, {name}");
            assert_eq!(stats.generated_tokens, first + 1);

            let mut lm = FakeLm::new(8, 128, Vec::new());
            let mut asked = 0;
            let (got, _) = generate_loop(
                &mut lm,
                &PROMPT,
                &settings(40),
                &mut Greedy,
                drafter.as_mut(),
                |_| {
                    asked += 1;
                    asked < 6
                },
            )
            .unwrap();
            assert_eq!(got, all[..6], "on_token returning false, {name}");
        }
    }

    #[test]
    fn the_cache_bounds_the_output_and_a_prompt_that_does_not_fit_is_refused() {
        let room = 10;
        for (name, mut drafter) in drafters() {
            let mut lm = FakeLm::new(8, PROMPT.len() + room, Vec::new());
            let (got, _) = run_with(&mut lm, &settings(1000), drafter.as_mut());
            assert_eq!(got, walk(&PROMPT, room), "{name}");
            assert!(lm.position() <= lm.capacity());
        }
        // A cache the prompt fills exactly leaves nothing to generate.
        let mut lm = FakeLm::new(8, PROMPT.len(), Vec::new());
        let (got, stats) = run_with(&mut lm, &settings(8), &mut NoDraft);
        assert!(got.is_empty());
        assert_eq!(stats.generated_tokens, 0);

        // And one that doesn't fit is refused before any pass runs.
        let mut lm = FakeLm::new(8, PROMPT.len() - 1, Vec::new());
        let err = generate_loop(
            &mut lm,
            &PROMPT,
            &settings(8),
            &mut Greedy,
            &mut NoDraft,
            |_| true,
        )
        .unwrap_err();
        assert!(err.to_string().contains("KV cache holds"), "{err}");
        assert_eq!(lm.position(), 0, "nothing ran");
    }

    #[test]
    fn a_tapping_drafter_sees_every_fed_token_once_in_order() {
        for width in [1, 3, 16] {
            for max_new in [1, 2, 7, 33] {
                for (name, inner) in drafters() {
                    let mut lm = FakeLm::new(width, 128, vec![0]);
                    let mut watching = Watching {
                        inner: BoxedDrafter(inner),
                        seen: Vec::new(),
                        resets: 0,
                    };
                    let (out, _) = run_with(&mut lm, &settings(max_new), &mut watching);
                    let in_order: Vec<f32> = (0..watching.seen.len()).map(|i| i as f32).collect();
                    assert_eq!(
                        watching.seen, in_order,
                        "width {width}, max_new {max_new}, {name}: positions seen"
                    );
                    // Every context token but the last one emitted.
                    assert!(
                        watching.seen.len() >= PROMPT.len() + out.len() - 1,
                        "width {width}, max_new {max_new}, {name}: only {} seen",
                        watching.seen.len()
                    );
                    assert_eq!(watching.resets, 1);
                }
            }
        }
    }

    /// A boxed drafter, so the list above can be wrapped.
    struct BoxedDrafter(Box<dyn Drafter>);

    impl Drafter for BoxedDrafter {
        fn draft(&mut self, context: &[u32], max: usize) -> Result<Vec<u32>> {
            self.0.draft(context, max)
        }
    }

    #[test]
    fn a_drafter_reading_other_layers_than_the_model_taps_is_refused() {
        let mut lm = FakeLm::new(4, 64, vec![1, 2]);
        let mut watching = Watching {
            inner: NoDraft,
            seen: Vec::new(),
            resets: 0,
        };
        let err = generate_loop(
            &mut lm,
            &PROMPT,
            &settings(4),
            &mut Greedy,
            &mut watching,
            |_| true,
        )
        .unwrap_err();
        assert!(err.to_string().contains("tapping"), "{err}");
    }

    #[test]
    fn prompt_lookup_guesses_what_followed_the_last_occurrence() {
        let mut lookup = PromptLookup::default();
        // "5 6 7 8 | 1 5 6 7": the tail "5 6 7" occurred before, followed by 8.
        let context = [5, 6, 7, 8, 1, 5, 6, 7];
        assert_eq!(lookup.draft(&context, 3).unwrap(), vec![8, 1, 5]);
        assert_eq!(lookup.draft(&context, 1).unwrap(), vec![8]);
        assert!(lookup.draft(&context, 0).unwrap().is_empty());
        // Nothing earlier matches a 2-gram of fresh tokens.
        assert!(lookup.draft(&[1, 2, 3, 4, 5, 6], 4).unwrap().is_empty());
        // The most recent earlier occurrence wins.
        let context = [1, 2, 9, 1, 2, 8, 1, 2];
        assert_eq!(lookup.draft(&context, 1).unwrap(), vec![8]);
    }

    // ---- config.json ----------------------------------------------------

    fn qwen3_config() -> serde_json::Value {
        serde_json::json!({
            "model_type": "qwen3",
            "hidden_size": 64,
            "intermediate_size": 192,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "num_hidden_layers": 6,
            "vocab_size": 1000,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000,
            "max_position_embeddings": 4096,
            "tie_word_embeddings": true,
        })
    }

    #[test]
    fn the_shape_comes_from_config_json() {
        let opts = AneLmOptions {
            capacity: 256,
            ..AneLmOptions::default()
        };
        let shape = ModelShape::from_config(&qwen3_config(), &opts).unwrap();
        assert_eq!((shape.n_layers, shape.vocab), (6, 1000));
        assert_eq!(
            (
                shape.cfg.dim,
                shape.cfg.n_heads,
                shape.cfg.n_kv_heads,
                shape.cfg.head_dim
            ),
            (64, 4, 2, 32)
        );
        assert_eq!((shape.cfg.width, shape.cfg.capacity), (16, 256));
        assert_eq!(shape.rope_theta, 1e6);
        assert!(shape.tied);

        // transformers' newer layout, and an untied head.
        let mut config = qwen3_config();
        config["rope_theta"] = serde_json::Value::Null;
        config["rope_parameters"] = serde_json::json!({"rope_type": "default", "rope_theta": 5e6});
        config["tie_word_embeddings"] = serde_json::json!(false);
        let shape = ModelShape::from_config(&config, &opts).unwrap();
        assert_eq!(shape.rope_theta, 5e6);
        assert!(!shape.tied);
    }

    #[test]
    fn a_config_the_engine_would_misread_is_refused_up_front() {
        let opts = AneLmOptions {
            capacity: 256,
            ..AneLmOptions::default()
        };
        let refused = |edit: &dyn Fn(&mut serde_json::Value), opts: &AneLmOptions| {
            let mut config = qwen3_config();
            edit(&mut config);
            ModelShape::from_config(&config, opts)
                .unwrap_err()
                .to_string()
        };
        assert!(refused(&|c| c["model_type"] = "llama".into(), &opts).contains("llama"));
        assert!(
            refused(
                &|c| c["rope_parameters"] = serde_json::json!({"rope_type": "yarn", "factor": 4.0}),
                &opts
            )
            .contains("yarn")
        );
        assert!(
            refused(
                &|c| {
                    c.as_object_mut().unwrap().remove("hidden_size");
                },
                &opts
            )
            .contains("hidden_size")
        );
        assert!(
            refused(&|c| c["num_attention_heads"] = 0.into(), &opts)
                .contains("num_attention_heads")
        );
        let past = AneLmOptions {
            capacity: 8192,
            ..AneLmOptions::default()
        };
        assert!(refused(&|_| {}, &past).contains("max_position_embeddings"));
        let taps = AneLmOptions {
            capacity: 256,
            taps: vec![2, 6],
            ..AneLmOptions::default()
        };
        assert!(refused(&|_| {}, &taps).contains("taps"));
        let unsorted = AneLmOptions {
            taps: vec![3, 1],
            ..taps
        };
        assert!(refused(&|_| {}, &unsorted).contains("taps"));
    }

    #[test]
    fn an_untied_head_must_be_in_the_checkpoint() {
        assert_eq!(
            head_tensor(true, false).unwrap(),
            "model.embed_tokens.weight"
        );
        assert_eq!(
            head_tensor(true, true).unwrap(),
            "model.embed_tokens.weight"
        );
        assert_eq!(head_tensor(false, true).unwrap(), "lm_head.weight");
        let err = head_tensor(false, false).unwrap_err().to_string();
        assert!(err.contains("lm_head.weight"), "{err}");
    }
}
