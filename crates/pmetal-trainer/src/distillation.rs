//! Knowledge Distillation Trainer.
//!
//! Trains a student against a teacher's next-token distributions with the
//! losses of `pmetal-distill`, from a live teacher (online, progressive) or a
//! precomputed logit cache (offline), and optionally on the student's own
//! samples (on-policy, Agarwal et al., *On-Policy Distillation of Language
//! Models*, ICLR 2024).
//!
//! # What a position means
//!
//! The logits at position `t` predict the token at `t + 1`. Both the hard
//! labels and the token weights that mask the prompt and padding therefore
//! belong to the *next* position's label, which [`shifted_targets`] builds:
//! a student that puts all its mass on each next token has a hard loss of 0.
//!
//! # Teachers
//!
//! A live teacher gives dense `[batch, seq, vocab]` logits. An offline cache
//! is read in the form it was stored in and never densified when it need not
//! be: a top-k cache feeds the sparse forward KL
//! ([`Distiller::compute_loss_sparse`]) and a sampled one the Random-Sampling
//! estimate ([`Distiller::compute_loss_sampled`]); only the dense caches
//! (`none`, `int8`, `int4`) are loaded as `[batch, seq, vocab]`.

use pmetal_data::{DataLoader, TrainingBatch, TrainingDataset};
use pmetal_distill::{DistillLossOutput, Distiller, LogitCache, SparseLogits, SparseTargets};
use pmetal_lora::TrainableModel;
use pmetal_runtime::compat::{Array, Dtype, Exception, nn, ops, optimizers::Optimizer};
use pmetal_runtime::nn::with_training_mode;

use crate::{
    AdaptiveAction, CheckpointManager, CheckpointMetadata, Result, SftError, StepStats,
    TrainingLoop, TrainingLoopConfig, rlkd::TeacherModel,
};

fn config_error(message: impl Into<String>) -> SftError {
    SftError::Config(message.into())
}

/// On-policy distillation (GKD): with probability `probability` a batch's
/// responses are sampled from the student itself and scored by the teacher,
/// so the student's own mistakes are corrected, instead of read from the
/// corpus.
///
/// The prompt of each row is its tokens before the first trained one (the
/// dataset must mask its prompts); the student completes it, up to
/// `max_new_tokens` or a stop token, and the divergence is taken over the
/// completion alone. A batch with a row that has no prompt is trained
/// off-policy.
#[derive(Debug, Clone)]
pub struct OnPolicyConfig {
    /// The share of batches that are on-policy, in `[0, 1]` (the paper's `λ`).
    pub probability: f32,
    /// Longest completion sampled.
    pub max_new_tokens: usize,
    /// Sampling temperature of the student (the paper samples at 1).
    pub temperature: f32,
    /// Tokens that end a completion; empty samples to `max_new_tokens`.
    pub stop_tokens: Vec<u32>,
}

impl OnPolicyConfig {
    fn validate(&self) -> Result<()> {
        if !(self.probability.is_finite() && (0.0..=1.0).contains(&self.probability)) {
            return Err(config_error(format!(
                "the on-policy probability must be in [0, 1], got {}",
                self.probability
            )));
        }
        if self.max_new_tokens == 0 {
            return Err(config_error("on-policy max_new_tokens must be at least 1"));
        }
        if !(self.temperature.is_finite() && self.temperature > 0.0) {
            return Err(config_error(format!(
                "the on-policy temperature must be positive, got {}",
                self.temperature
            )));
        }
        Ok(())
    }
}

/// The teacher's distribution for a batch, in the form it was held in.
enum TeacherSignal {
    /// `[batch, seq, vocab]` logits.
    Dense(Array),
    /// A top-k cache's kept logits, `[batch, seq, k]`.
    Sparse(SparseLogits),
    /// A sampled cache's draws, `[batch, seq, slots]`.
    Sampled(SparseTargets),
}

/// Where a batch's teacher distribution comes from.
trait TeacherSource {
    fn signal(&mut self, batch: &TrainingBatch) -> Result<TeacherSignal>;
    /// Whether the teacher can score sequences the dataset does not hold
    /// (the student's own samples).
    fn scores_new_sequences(&self) -> bool;
}

/// A live teacher model.
struct LiveTeacher<'a, T: TeacherModel>(&'a mut T);

impl<T: TeacherModel> TeacherSource for LiveTeacher<'_, T> {
    fn signal(&mut self, batch: &TrainingBatch) -> Result<TeacherSignal> {
        // Outside the autodiff scope: no gradient, and the graph is not kept.
        let logits = self
            .0
            .forward_teacher(&batch.input_ids)
            .map_err(SftError::Mlx)?;
        logits.eval();
        Ok(TeacherSignal::Dense(logits))
    }

    fn scores_new_sequences(&self) -> bool {
        true
    }
}

/// A precomputed logit cache, one sequence per dataset sample.
struct CachedTeacher<'a>(&'a LogitCache);

impl TeacherSource for CachedTeacher<'_> {
    fn signal(&mut self, batch: &TrainingBatch) -> Result<TeacherSignal> {
        let cache = self.0;
        let to_err = |e: pmetal_distill::DistillError| config_error(e.to_string());
        match cache.metadata().compression.as_str() {
            "top_k" => {
                let rows = batch
                    .sample_indices
                    .iter()
                    .map(|&i| cache.load_sequence_sparse(i))
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .map_err(to_err)?;
                Ok(TeacherSignal::Sparse(
                    SparseLogits::stack(&rows, batch.seq_len).map_err(to_err)?,
                ))
            }
            "sampled" => {
                let rows = batch
                    .sample_indices
                    .iter()
                    .map(|&i| cache.load_sequence_sampled(i))
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .map_err(to_err)?;
                Ok(TeacherSignal::Sampled(
                    SparseTargets::stack(&rows, batch.seq_len).map_err(to_err)?,
                ))
            }
            _ => Ok(TeacherSignal::Dense(dense_cached_logits(cache, batch)?)),
        }
    }

    fn scores_new_sequences(&self) -> bool {
        false
    }
}

/// The labels `[B, S]` and token weights `[B, S]` that go with logits
/// `[B, S, V]`: position `t` predicts the token at `t + 1`, so both come from
/// the *next* position's label, and the last position, which predicts
/// nothing, is ignored. The weights are 1 where the predicted token is
/// trained and 0 over the prompt and padding.
pub fn shifted_targets(labels: &Array, ignore_index: i32) -> (Array, Array) {
    let (batch, seq) = (labels.dim(0), labels.dim(1));
    let next = labels.slice(&[0, 1], &[batch, seq]);
    let dtype = next.dtype_raw();
    let pad =
        Array::full(&[batch, 1], ignore_index as f32, Dtype::Float32.as_i32()).as_dtype(dtype);
    let shifted = ops::concatenate_axis(&[&next, &pad], 1);
    let ignore = Array::from_int(ignore_index).as_dtype(dtype);
    let weights = shifted.ne(&ignore).as_dtype(Dtype::Float32.as_i32());
    (shifted, weights)
}

/// Dense `[batch, seq, vocab]` teacher logits from a cache that stores them
/// dense. Unfilled positions are a constant row, which the token weights mask.
fn dense_cached_logits(cache: &LogitCache, batch: &TrainingBatch) -> Result<Array> {
    let vocab_size = cache.metadata().vocab_size;
    if vocab_size == 0 {
        return Err(config_error(
            "offline distillation cache metadata is missing vocab_size",
        ));
    }

    let mut batch_logits = vec![-1.0e10_f32; batch.batch_size * batch.seq_len * vocab_size];

    for (batch_row, &sample_index) in batch.sample_indices.iter().enumerate() {
        let sequence = cache
            .load_sequence(sample_index)
            .map_err(|e| config_error(e.to_string()))?;
        let shape = sequence.shape().to_vec();
        let (seq_len, cached_vocab) = match shape.as_slice() {
            [1, seq_len, vocab] => (*seq_len as usize, *vocab as usize),
            [seq_len, vocab] => (*seq_len as usize, *vocab as usize),
            other => {
                return Err(config_error(format!(
                    "cached teacher logits for sample {sample_index} have unsupported shape {other:?}"
                )));
            }
        };

        if cached_vocab != vocab_size {
            return Err(config_error(format!(
                "cached teacher logits for sample {sample_index} use vocab {cached_vocab}, expected {vocab_size}"
            )));
        }

        let numel: usize = shape.iter().map(|&dim| dim as usize).product();
        let flat = sequence.clone().to_f32_vec(numel).ok_or_else(|| {
            config_error(format!(
                "failed to materialize cached teacher logits for sample {sample_index}"
            ))
        })?;

        let copy_seq_len = seq_len.min(batch.seq_len);
        for token_idx in 0..copy_seq_len {
            let src_offset = token_idx * vocab_size;
            let dst_offset = (batch_row * batch.seq_len + token_idx) * vocab_size;
            batch_logits[dst_offset..dst_offset + vocab_size]
                .copy_from_slice(&flat[src_offset..src_offset + vocab_size]);
        }
    }

    Ok(Array::from_f32_slice(
        &batch_logits,
        &[
            batch.batch_size as i32,
            batch.seq_len as i32,
            vocab_size as i32,
        ],
    ))
}

/// Trainer for Knowledge Distillation.
pub struct DistillationTrainer {
    /// The distillation engine (holds config and loss logic).
    distiller: Distiller,
    /// Underlying training loop state.
    loop_state: TrainingLoop,
    /// On-policy distillation, if configured.
    on_policy: Option<OnPolicyConfig>,
    /// Batches seen: the index the on-policy coin and rollouts are drawn from.
    batches_seen: u64,
}

impl DistillationTrainer {
    /// Create a new DistillationTrainer.
    ///
    /// The configuration is validated, and a distillation the trainer cannot
    /// carry out is an error rather than a term that silently never runs:
    /// hidden-state and attention transfer need every layer's states from both
    /// models, which the trainable-model interface does not expose, and
    /// outcome supervision needs per-sample correctness labels the dataset
    /// does not carry.
    pub fn new(distiller: Distiller, config: TrainingLoopConfig) -> Result<Self> {
        use pmetal_core::ConfigValidator;
        config
            .training
            .validate()
            .map_err(|e| config_error(e.to_string()))?;
        distiller
            .config()
            .validate()
            .map_err(|e| config_error(e.to_string()))?;
        let loss = &distiller.config().loss;
        if loss.hidden_state.is_some() || loss.attention.is_some() {
            return Err(config_error(
                "hidden-state and attention distillation are not supported by the trainer: \
                 they need every layer's hidden states or attention maps from both models, \
                 which the trainable-model interface does not expose",
            ));
        }
        if loss.outcome_supervised {
            return Err(config_error(
                "outcome-supervised distillation needs per-sample correctness labels, which \
                 the dataset does not carry",
            ));
        }
        Ok(Self {
            distiller,
            loop_state: TrainingLoop::new(config)?,
            on_policy: None,
            batches_seen: 0,
        })
    }

    /// Distill on the student's own samples for a share of the batches (GKD);
    /// see [`OnPolicyConfig`]. Online and progressive runs only: a cache holds
    /// the teacher's distribution over the corpus, not over new samples.
    pub fn with_on_policy(mut self, config: OnPolicyConfig) -> Result<Self> {
        config.validate()?;
        self.on_policy = Some(config);
        Ok(self)
    }

    /// Add a training callback for metrics logging or dashboard integration.
    pub fn add_callback(&mut self, callback: Box<dyn pmetal_core::TrainingCallback>) {
        self.loop_state.add_callback(callback);
    }

    /// Resume at micro-batch `step` of epoch `epoch`.
    pub fn set_progress(&mut self, step: usize, epoch: usize) {
        self.loop_state.set_step(step);
        self.loop_state.set_epoch(epoch);
    }

    /// Enable adaptive LR with control file for TUI communication.
    pub fn enable_adaptive_lr_with_control(
        &mut self,
        config: crate::adaptive_lr::AdaptiveLrConfig,
        control_file: std::path::PathBuf,
    ) {
        self.loop_state
            .enable_adaptive_lr_with_control(config, control_file);
    }

    /// Size the learning-rate schedule (`gradient_accumulation_steps`
    /// micro-batches per optimizer step) and put the optimizer on its first
    /// rate. Returns the run's length in micro-batches and in optimizer steps.
    fn plan_schedule(
        &mut self,
        train_dataset: &TrainingDataset,
        optimizer: &mut crate::ParamGroupOptimizer,
    ) -> (usize, usize) {
        let micro_batches = self.loop_state.micro_batches_per_epoch(train_dataset);
        let accum = self
            .loop_state
            .config
            .training
            .gradient_accumulation_steps
            .max(1);
        let total_micro = self.loop_state.plan_schedule(micro_batches, accum);
        optimizer.set_learning_rate(self.loop_state.get_learning_rate());
        (total_micro, total_micro / accum)
    }

    /// The distillation loss of `student_logits` against `signal`.
    ///
    /// `labels` are the next-token labels ([`shifted_targets`]), `None` for a
    /// batch with no ground truth (the student's own samples).
    fn compute_distillation_output(
        &self,
        signal: &TeacherSignal,
        student_logits: &Array,
        labels: Option<&Array>,
        weights: &Array,
        current_step: usize,
        total_steps: usize,
    ) -> Result<DistillLossOutput> {
        let (step, total) = (current_step, total_steps.max(1));
        let out = match signal {
            TeacherSignal::Dense(teacher) => self.distiller.compute_loss(
                teacher,
                student_logits,
                labels,
                Some(weights),
                step,
                total,
            ),
            TeacherSignal::Sparse(teacher) => self.distiller.compute_loss_sparse(
                teacher,
                student_logits,
                labels,
                Some(weights),
                step,
                total,
            ),
            TeacherSignal::Sampled(teacher) => self.distiller.compute_loss_sampled(
                teacher,
                student_logits,
                labels,
                Some(weights),
                step,
                total,
            ),
        };
        out.map_err(|e| config_error(e.to_string()))
    }

    /// The batch with the student's own completions in place of the corpus's:
    /// each row's prompt completed by the student. `None` when a row has no
    /// prompt to complete (the whole text is trained).
    fn on_policy_batch<S: TrainableModel>(
        &mut self,
        student: &mut S,
        batch: &TrainingBatch,
        policy: &OnPolicyConfig,
    ) -> Result<Option<TrainingBatch>> {
        let ignore = self.distiller.config().training.ignore_index;
        let ids = batch.input_ids.as_dtype(Dtype::Int32.as_i32());
        let labels = batch.labels.as_dtype(Dtype::Int32.as_i32());
        let mask = batch.attention_mask.as_dtype(Dtype::Int32.as_i32());
        for a in [&ids, &labels, &mask] {
            a.eval();
        }
        let (rows, seq) = (batch.batch_size, batch.seq_len);
        let (ids, labels, mask) = (
            ids.as_slice::<i32>(),
            labels.as_slice::<i32>(),
            mask.as_slice::<i32>(),
        );

        let mut prompts: Vec<Vec<u32>> = Vec::with_capacity(rows);
        for r in 0..rows {
            let len = mask[r * seq..(r + 1) * seq]
                .iter()
                .filter(|&&m| m != 0)
                .count();
            let row_labels = &labels[r * seq..r * seq + len];
            let Some(start) = row_labels.iter().position(|&l| l != ignore) else {
                return Ok(None);
            };
            if start == 0 {
                return Ok(None);
            }
            prompts.push(
                ids[r * seq..r * seq + start]
                    .iter()
                    .map(|&t| t as u32)
                    .collect(),
            );
        }

        let seed = self
            .loop_state
            .config
            .dataloader
            .seed
            .wrapping_add(0x6b43_a9b5);
        let mut sequences: Vec<(Vec<u32>, usize)> = Vec::with_capacity(rows);
        for prompt in &prompts {
            let out = crate::rollout::sample_completions(
                student,
                prompt,
                &crate::rollout::RolloutConfig {
                    num_generations: 1,
                    max_new_tokens: policy.max_new_tokens,
                    temperature: policy.temperature,
                    top_p: 1.0,
                    top_k: 0,
                    min_p: 0.0,
                    stop_tokens: policy.stop_tokens.clone(),
                    seed: Some(pmetal_runtime::decode::draw_bits(seed, self.batches_seen)),
                    max_len: prompt.len() + policy.max_new_tokens,
                    kv_cache_bits: None,
                },
            )
            .map_err(SftError::Mlx)?;
            self.batches_seen += 1;
            let tokens = out.token_ids.into_iter().next().unwrap_or_default();
            sequences.push((tokens, prompt.len()));
        }

        let width = sequences.iter().map(|(t, _)| t.len()).max().unwrap_or(0);
        if width < 2 {
            return Ok(None);
        }
        let pad = self.loop_state.config.dataloader.pad_token_id as i32;
        let mut new_ids = vec![pad; rows * width];
        let mut new_labels = vec![ignore; rows * width];
        let mut new_mask = vec![0i32; rows * width];
        for (r, (tokens, prompt_len)) in sequences.iter().enumerate() {
            for (t, &token) in tokens.iter().enumerate() {
                new_ids[r * width + t] = token as i32;
                new_mask[r * width + t] = 1;
                if t >= *prompt_len {
                    new_labels[r * width + t] = token as i32;
                }
            }
        }
        let shape = [rows as i32, width as i32];
        Ok(Some(TrainingBatch {
            input_ids: Array::from_i32_slice(&new_ids).reshape(&shape),
            labels: Array::from_i32_slice(&new_labels).reshape(&shape),
            attention_mask: Array::from_i32_slice(&new_mask).reshape(&shape),
            pixel_values: None,
            batch_size: rows,
            seq_len: width,
            sample_indices: batch.sample_indices.clone(),
        }))
    }

    /// Perform a single distillation micro-batch against a live teacher.
    ///
    /// The gradient is accumulated, and applied through `optimizer` once
    /// `gradient_accumulation_steps` micro-batches have been seen (clipped to
    /// `max_grad_norm`).
    ///
    /// # Arguments
    /// * `student` - The student model (trainable).
    /// * `teacher` - The teacher model (frozen/inference).
    /// * `batch` - Training batch.
    /// * `optimizer` - Optimizer for student.
    /// * `total_steps` - Optimizer steps in the run (for the progressive schedule).
    pub fn train_step<S, T, O>(
        &mut self,
        student: &mut S,
        teacher: &mut T,
        batch: &pmetal_data::TrainingBatch,
        optimizer: &mut O,
        total_steps: usize,
    ) -> Result<StepStats>
    where
        S: TrainableModel,
        T: TeacherModel,
        O: Optimizer,
    {
        let signal = LiveTeacher(teacher).signal(batch)?;
        Ok(self
            .micro_step(student, batch, &signal, optimizer, total_steps, true)?
            .0)
    }

    /// One micro-batch: forward and backward against `signal`, accumulate,
    /// and apply when the accumulation is complete. Returns the stats and
    /// whether an optimizer step was taken.
    fn micro_step<S, O>(
        &mut self,
        student: &mut S,
        batch: &TrainingBatch,
        signal: &TeacherSignal,
        optimizer: &mut O,
        total_steps: usize,
        hard_labels: bool,
    ) -> Result<(StepStats, bool)>
    where
        S: TrainableModel,
        O: Optimizer,
    {
        let start_time = std::time::Instant::now();
        let batch_tokens = batch.batch_size.saturating_mul(batch.seq_len);
        let ignore = self.distiller.config().training.ignore_index;
        let (next_labels, weights) = shifted_targets(&batch.labels, ignore);
        let labels = hard_labels.then_some(&next_labels);
        let current_step = self.loop_state.optimizer_steps_taken();
        let (loss, grads) = {
            let this = &*self;
            let loss_fn =
                |student: &mut S, input_ids: &Array| -> std::result::Result<Array, Exception> {
                    let student_logits = student
                        .forward(input_ids, None)
                        .map_err(|e| Exception::custom(e.to_string()))?;
                    let output = this
                        .compute_distillation_output(
                            signal,
                            &student_logits,
                            labels,
                            &weights,
                            current_step,
                            total_steps,
                        )
                        .map_err(|e| Exception::custom(e.to_string()))?;
                    Ok(output.total)
                };
            let mut loss_and_grad_fn = nn::value_and_grad(loss_fn);
            with_training_mode(|| loss_and_grad_fn(student, &batch.input_ids))?
        };

        let loss_val = crate::step_check::check_step(self.loop_state.step + 1, loss.item_f32())?;
        self.loop_state.accumulate_gradients(grads)?;
        let applied = self.loop_state.should_apply_gradients();
        let mut grad_norm = None;
        if applied {
            let mut grads = self
                .loop_state
                .take_accumulated_gradients()
                .expect("an accumulation is complete");
            if let Some(norm) = self.loop_state.clip_gradients_gpu(&mut grads)? {
                norm.eval();
                grad_norm = Some(crate::step_check::check_grad_norm(
                    self.loop_state.step + 1,
                    norm.item_f32(),
                )?);
            }
            optimizer.update(student, grads)?;
        }

        Ok((
            StepStats {
                step: self.loop_state.step,
                loss: loss_val,
                learning_rate: self.loop_state.get_learning_rate(),
                tokens: batch_tokens,
                grad_norm,
                step_time_ms: start_time.elapsed().as_millis() as u64,
            },
            applied,
        ))
    }

    /// Mean distillation loss of the student over `dataset`, without gradients.
    fn evaluate<S: TrainableModel>(
        &mut self,
        student: &mut S,
        source: &mut dyn TeacherSource,
        dataset: &TrainingDataset,
        total_steps: usize,
    ) -> Result<f64> {
        let mut eval_config = self.loop_state.config.dataloader.clone();
        eval_config.shuffle = false;
        eval_config.drop_last = false;
        let ignore = self.distiller.config().training.ignore_index;

        let mut dataloader = DataLoader::new(dataset.clone(), eval_config, None);
        let mut total_loss = 0.0_f64;
        let mut num_batches = 0_usize;

        while let Some(batch) = dataloader
            .try_next_batch()
            .map_err(|e| config_error(e.to_string()))?
        {
            let signal = source.signal(&batch)?;
            let student_logits = student
                .forward(&batch.input_ids, None)
                .map_err(|e| config_error(e.to_string()))?;
            let (labels, weights) = shifted_targets(&batch.labels, ignore);
            let output = self.compute_distillation_output(
                &signal,
                &student_logits,
                Some(&labels),
                &weights,
                self.loop_state.optimizer_steps_taken().min(total_steps),
                total_steps,
            )?;
            output.total.eval();
            total_loss += f64::from(output.total.item_f32());
            num_batches += 1;
        }

        Ok(if num_batches > 0 {
            total_loss / num_batches as f64
        } else {
            0.0
        })
    }

    /// Run the distillation loop against a live teacher: online, progressive
    /// and (with [`with_on_policy`](Self::with_on_policy)) on-policy.
    pub fn run<S, T>(
        &mut self,
        student: &mut S,
        teacher: &mut T,
        train_dataset: TrainingDataset,
        eval_dataset: Option<TrainingDataset>,
        checkpoint_manager: Option<&CheckpointManager>,
    ) -> Result<()>
    where
        S: TrainableModel,
        T: TeacherModel,
    {
        tracing::info!("Starting distillation...");
        self.run_with(
            student,
            &mut LiveTeacher(teacher),
            train_dataset,
            eval_dataset,
            checkpoint_manager,
        )
    }

    /// Run offline distillation using a precomputed teacher-logit cache, in
    /// whatever form it holds the logits ([the module docs](self)).
    pub fn run_offline<S>(
        &mut self,
        student: &mut S,
        cache: &LogitCache,
        train_dataset: TrainingDataset,
        eval_dataset: Option<TrainingDataset>,
        checkpoint_manager: Option<&CheckpointManager>,
    ) -> Result<()>
    where
        S: TrainableModel,
    {
        if cache.metadata().num_sequences < train_dataset.len() {
            return Err(config_error(format!(
                "offline distillation cache is incomplete: expected at least {} sequences, found {}",
                train_dataset.len(),
                cache.metadata().num_sequences
            )));
        }
        if self.on_policy.is_some() {
            tracing::warn!(
                "on-policy distillation needs the live teacher; the offline run trains on the corpus"
            );
        }
        tracing::info!(
            "Starting offline distillation from a {} cache...",
            cache.metadata().compression
        );
        self.run_with(
            student,
            &mut CachedTeacher(cache),
            train_dataset,
            eval_dataset,
            checkpoint_manager,
        )
    }

    /// The loop online and offline runs share.
    fn run_with<S>(
        &mut self,
        student: &mut S,
        source: &mut dyn TeacherSource,
        train_dataset: TrainingDataset,
        eval_dataset: Option<TrainingDataset>,
        checkpoint_manager: Option<&CheckpointManager>,
    ) -> Result<()>
    where
        S: TrainableModel,
    {
        let mut optimizer = self
            .loop_state
            .build_optimizer(student.output_projection_path());
        self.loop_state
            .apply_gradient_checkpointing(student, "distillation");

        let num_epochs = self.loop_state.config.training.num_epochs;
        let checkpoint_every = self.loop_state.config.checkpoint_every;

        // Track the best eval (or train) loss seen so far for "is_best" tagging.
        let mut best_loss = f64::MAX;

        let (total_micro, total_steps) = self.plan_schedule(&train_dataset, &mut optimizer);
        let on_policy = self
            .on_policy
            .clone()
            .filter(|_| source.scores_new_sequences());
        let seed = self.loop_state.config.dataloader.seed;

        for cb in &mut self.loop_state.callbacks {
            cb.on_train_start();
        }

        tracing::info!(
            "Distilling for {total_steps} optimizer steps ({total_micro} micro-batches)"
        );

        for epoch in 0..num_epochs {
            self.loop_state.epoch = epoch;

            let mut dataloader = DataLoader::new(
                train_dataset.clone(),
                self.loop_state.config.dataloader.clone(),
                None,
            );

            if epoch > 0 {
                dataloader.reset(Some(seed + epoch as u64));
            }

            while let Some(corpus_batch) = dataloader
                .try_next_batch()
                .map_err(|e| config_error(e.to_string()))?
            {
                let step_start = std::time::Instant::now();

                // The student's own samples for a share of the batches.
                let mut batch = corpus_batch;
                let mut hard_labels = true;
                if let Some(policy) = &on_policy {
                    let draw = (pmetal_runtime::decode::draw_bits(
                        seed.wrapping_add(0x51ed),
                        self.loop_state.step as u64,
                    ) >> 11) as f64
                        / (1u64 << 53) as f64;
                    if draw < f64::from(policy.probability) {
                        match self.on_policy_batch(student, &batch, policy)? {
                            Some(sampled) => {
                                batch = sampled;
                                hard_labels = false;
                            }
                            None => tracing::debug!(
                                "on-policy batch skipped: a row has no prompt to complete"
                            ),
                        }
                    }
                }

                let signal = source.signal(&batch)?;
                let (stats, applied) = self.micro_step(
                    student,
                    &batch,
                    &signal,
                    &mut optimizer,
                    total_steps,
                    hard_labels,
                )?;
                let step_ms = step_start.elapsed().as_secs_f64() * 1000.0;

                self.loop_state.step += 1;

                // Feed loss to adaptive LR controller for next step
                let action = self.loop_state.apply_adaptive_lr(stats.loss as f64);

                if action == AdaptiveAction::Continue && self.loop_state.should_snapshot_best() {
                    self.loop_state.snapshot_best_weights(student);
                }
                if action == AdaptiveAction::Rollback {
                    self.loop_state.restore_best_weights(student);
                }
                if action == AdaptiveAction::EarlyStop {
                    self.loop_state.restore_best_weights(student);
                    tracing::info!("Early stopping distillation — best checkpoint restored.");
                    for cb in &mut self.loop_state.callbacks {
                        cb.on_train_end();
                    }
                    return Ok(());
                }

                // The learning rate of the next optimizer step.
                let next_lr = self.loop_state.get_learning_rate();
                optimizer.set_learning_rate(next_lr);

                if self.loop_state.step % self.loop_state.config.log_every.max(1) == 0 {
                    tracing::info!(
                        "Step {}: loss={:.4}, lr={:.2e}{}",
                        stats.step,
                        stats.loss,
                        next_lr,
                        if hard_labels { "" } else { " (on-policy)" },
                    );
                }

                // Emit metrics to callbacks
                let tok_sec = if step_ms > 0.0 {
                    stats.tokens as f64 / (step_ms / 1000.0)
                } else {
                    0.0
                };
                let metrics = pmetal_core::StepMetrics {
                    step: self.loop_state.step,
                    epoch,
                    total_epochs: num_epochs,
                    total_steps: total_micro,
                    loss: stats.loss as f64,
                    lr: next_lr as f64,
                    tok_sec,
                    total_ms: step_ms,
                    tokens: stats.tokens,
                    grad_norm: stats.grad_norm.map(f64::from),
                    ..Default::default()
                };
                for cb in &mut self.loop_state.callbacks {
                    cb.on_step_end_with_metrics(&metrics);
                }
                if self.loop_state.callbacks.iter().any(|cb| cb.should_stop()) {
                    for cb in &mut self.loop_state.callbacks {
                        cb.on_train_end();
                    }
                    return Err(SftError::Cancelled);
                }

                // Checkpointing and evaluation, on optimizer-step boundaries.
                let optimizer_steps = self.loop_state.optimizer_steps_taken();
                if applied
                    && checkpoint_every > 0
                    && optimizer_steps > 0
                    && optimizer_steps % checkpoint_every == 0
                {
                    let eval_loss: Option<f64> = if let Some(ref eval_ds) = eval_dataset {
                        let loss = self.evaluate(student, source, eval_ds, total_steps)?;
                        tracing::info!("Step {}: eval_loss={:.4}", self.loop_state.step, loss);
                        Some(loss)
                    } else {
                        None
                    };

                    // Use eval loss when available, otherwise fall back to train loss.
                    let reference_loss = eval_loss.unwrap_or(stats.loss as f64);
                    let is_best = reference_loss < best_loss;
                    if is_best {
                        best_loss = reference_loss;
                    }

                    if let Some(manager) = checkpoint_manager {
                        let lora_params = student.lora_parameters();
                        let mut metadata = CheckpointMetadata::new(
                            self.loop_state.step,
                            self.loop_state.epoch,
                            stats.loss as f64,
                            next_lr as f64,
                        );
                        if let Some(loss) = eval_loss {
                            metadata = metadata.with_best_val_loss(loss);
                        }
                        manager.save_checkpoint(&lora_params, &metadata, is_best)?;
                    }
                }
            }
        }

        for cb in &mut self.loop_state.callbacks {
            cb.on_train_end();
        }

        Ok(())
    }
}

/// Generate or resume an offline teacher-logit cache for distillation.
pub fn generate_teacher_logit_cache<T>(
    teacher: &mut T,
    dataset: &TrainingDataset,
    cache: &mut LogitCache,
    teacher_id: &str,
    max_seq_len: usize,
) -> Result<()>
where
    T: TeacherModel,
{
    let mut vocab_size = cache.metadata().vocab_size;

    for (sample_index, sample) in dataset.samples().iter().enumerate() {
        if cache.has_sequence(sample_index) {
            continue;
        }

        let seq_len = sample.input_ids.len().min(max_seq_len);
        let input_ids: Vec<i32> = sample
            .input_ids
            .iter()
            .take(seq_len)
            .map(|&token| token as i32)
            .collect();
        let input = Array::from_i32_slice(&input_ids).reshape(&[1, seq_len as i32]);
        let logits = teacher.forward_teacher(&input).map_err(SftError::Mlx)?;
        logits.eval();

        if vocab_size == 0 {
            vocab_size = logits.dim(-1) as usize;
        }

        cache
            .cache_sequence(sample_index, &logits)
            .map_err(|e| config_error(e.to_string()))?;
    }

    if vocab_size == 0 {
        return Err(config_error(
            "cannot generate teacher-logit cache for an empty dataset",
        ));
    }

    cache.set_metadata(teacher_id.to_string(), vocab_size, max_seq_len);
    cache
        .save_metadata()
        .map_err(|e| config_error(e.to_string()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_data::DataLoaderConfig;
    use pmetal_distill::{CompressionMethod, DistillConfig};
    use tempfile::tempdir;

    fn distiller_with(loss: pmetal_distill::LossConfig) -> Distiller {
        Distiller::new(DistillConfig {
            teacher: "teacher".to_string(),
            student: "student".to_string(),
            method: pmetal_distill::DistillMethod::Online,
            loss,
            offline: None,
            output_path: None,
            training: pmetal_distill::TrainingConfig::default(),
        })
        .unwrap()
    }

    fn test_trainer() -> DistillationTrainer {
        test_trainer_with(pmetal_distill::LossConfig::default())
    }

    fn test_trainer_with(loss: pmetal_distill::LossConfig) -> DistillationTrainer {
        DistillationTrainer::new(
            distiller_with(loss),
            TrainingLoopConfig {
                dataloader: DataLoaderConfig {
                    batch_size: 2,
                    max_seq_len: 8,
                    shuffle: false,
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .unwrap()
    }

    fn batch_of(ids: &[i32], labels: &[i32], rows: usize) -> TrainingBatch {
        let seq = ids.len() / rows;
        TrainingBatch {
            input_ids: Array::from_i32_slice(ids).reshape(&[rows as i32, seq as i32]),
            labels: Array::from_i32_slice(labels).reshape(&[rows as i32, seq as i32]),
            attention_mask: Array::from_i32_slice(&vec![1; ids.len()])
                .reshape(&[rows as i32, seq as i32]),
            pixel_values: None,
            batch_size: rows,
            seq_len: seq,
            sample_indices: (0..rows).collect(),
        }
    }

    fn values(a: &Array) -> Vec<f32> {
        a.eval();
        pmetal_runtime::check_unobserved_error().expect("bridge op failed");
        a.as_slice::<f32>().to_vec()
    }

    #[test]
    fn dense_cached_logits_respect_batch_indices_and_padding() {
        let tempdir = tempdir().unwrap();
        let mut cache = LogitCache::new(tempdir.path(), CompressionMethod::None, 128).unwrap();
        cache.set_metadata("teacher".to_string(), 3, 8);
        cache
            .cache_sequence(
                0,
                &Array::from_f32_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 3]),
            )
            .unwrap();
        cache
            .cache_sequence(
                1,
                &Array::from_f32_slice(&[7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0], &[1, 2, 3]),
            )
            .unwrap();
        cache.save_metadata().unwrap();

        let mut batch = batch_of(&[1, 2, 3, 4, 5, 0], &[1, 2, 3, 4, 5, -100], 2);
        batch.sample_indices = vec![1, 0];

        let logits = dense_cached_logits(&cache, &batch).unwrap();
        let values = values(&logits);

        assert_eq!(&values[0..6], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        assert_eq!(&values[9..15], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(values[6..9].iter().all(|v| *v < -1.0e9));
        assert!(values[15..18].iter().all(|v| *v < -1.0e9));
    }

    /// Position `t`'s logits predict the token at `t + 1`, so both the hard
    /// label and the token weights belong to the *next* position's label. A
    /// student that puts all its mass on each next token has a hard loss of 0;
    /// the loss was computed against the same position's label instead, which
    /// trains a model to copy its input.
    #[test]
    fn the_hard_label_and_weights_are_the_next_tokens() {
        // Prompt of two tokens (labels masked), completion 5 6 7.
        let batch = batch_of(&[1, 2, 5, 6, 7], &[-100, -100, 5, 6, 7], 1);
        // The student's logits at position t are a spike on input_ids[t + 1].
        let vocab = 10usize;
        let next = [2usize, 5, 6, 7, 0];
        let mut logits = vec![0.0f32; 5 * vocab];
        for (t, &token) in next.iter().enumerate() {
            logits[t * vocab + token] = 30.0;
        }
        let student = Array::from_f32_slice(&logits, &[1, 5, vocab as i32]);
        let signal = TeacherSignal::Dense(student.clone());

        // Pure hard-label loss.
        let trainer = test_trainer_with(pmetal_distill::LossConfig {
            alpha: 0.0,
            ..Default::default()
        });
        let (labels, weights) = shifted_targets(&batch.labels, -100);
        assert_eq!(values(&weights), vec![0.0, 1.0, 1.0, 1.0, 0.0]);
        let out = trainer
            .compute_distillation_output(&signal, &student, Some(&labels), &weights, 0, 1)
            .unwrap();
        let hard = out.hard.expect("a hard loss").item_f32();
        assert!(
            hard < 1e-3,
            "hard loss {hard}: not scored against the next token"
        );
    }

    /// The soft loss is masked by the same shifted
    /// weights: a teacher that disagrees with the student only at positions
    /// that predict prompt tokens or nothing leaves
    /// the loss at zero.
    #[test]
    fn the_soft_loss_ignores_positions_that_predict_prompt_tokens() {
        let batch = batch_of(&[1, 2, 5, 6], &[-100, -100, 5, 6], 1);
        let vocab = 6usize;
        let spike = |tokens: &[usize]| {
            let mut v = vec![0.0f32; tokens.len() * vocab];
            for (t, &k) in tokens.iter().enumerate() {
                v[t * vocab + k] = 10.0;
            }
            Array::from_f32_slice(&v, &[1, tokens.len() as i32, vocab as i32])
        };
        // Position 0 predicts the prompt's second token and position 3
        // predicts nothing: the teacher and the student disagree there.
        // Positions 1 and 2 agree.
        let student = spike(&[0, 5, 5, 0]);
        let teacher = spike(&[1, 5, 5, 3]);
        let trainer = test_trainer_with(pmetal_distill::LossConfig {
            alpha: 1.0,
            ..Default::default()
        });
        let (labels, weights) = shifted_targets(&batch.labels, -100);
        // Weights keep positions 1 and 2 only (labels at t+1 are 5 and 6).
        assert_eq!(values(&weights), vec![0.0, 1.0, 1.0, 0.0]);
        let out = trainer
            .compute_distillation_output(
                &TeacherSignal::Dense(teacher),
                &student,
                Some(&labels),
                &weights,
                0,
                1,
            )
            .unwrap();
        // Positions 1 (teacher 5, student 5) agree; position 2 (teacher 5,
        // student 5) agrees: nothing left to distill.
        assert!(
            out.soft.item_f32().abs() < 1e-4,
            "soft {}",
            out.soft.item_f32()
        );
    }

    /// Every adapter value, in parameter-name order.
    fn adapter_values(model: &impl TrainableModel) -> Vec<f32> {
        let mut params: Vec<_> = model.lora_parameters().into_iter().collect();
        params.sort_by(|a, b| a.0.cmp(&b.0));
        params
            .into_iter()
            .flat_map(|(_, mut p)| {
                let n = p.size();
                p.to_f32_vec(n).unwrap()
            })
            .collect()
    }

    const CONFIG: &str = r#"{
        "model_type": "llama", "vocab_size": 64, "hidden_size": 32,
        "intermediate_size": 64, "num_hidden_layers": 1,
        "num_attention_heads": 2, "num_key_value_heads": 1,
        "max_position_embeddings": 64, "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0
    }"#;

    fn student_model() -> pmetal_lora::AdaptedModel {
        pmetal_lora::AdaptedModel::attach(
            pmetal_models::DynamicModel::from_config(CONFIG).unwrap(),
            pmetal_core::LoraConfig {
                r: 4,
                alpha: 8.0,
                target_modules: vec!["q_proj".into(), "v_proj".into()],
                ..Default::default()
            },
        )
        .unwrap()
    }

    /// The teacher is a plain loaded model, frozen and without adapters, as
    /// `pmetal distill` loads it, and a step moves the student's adapters.
    #[test]
    fn frozen_teacher_trains_the_student() {
        use pmetal_runtime::compat::optimizers::AdamW;

        let mut teacher = pmetal_models::DynamicModel::from_config(CONFIG).unwrap();
        let mut student = student_model();
        let before = adapter_values(&student);

        let batch = batch_of(&[1, 2, 3, 4, 5, 6], &[1, 2, 3, 4, 5, 6], 2);
        let mut optimizer = AdamW::new(1e-2, 0.0);
        // One micro-batch is one step when the accumulation is 1.
        let mut trainer = test_trainer();
        trainer
            .loop_state
            .config
            .training
            .gradient_accumulation_steps = 1;
        let stats = trainer
            .train_step(&mut student, &mut teacher, &batch, &mut optimizer, 1)
            .unwrap();
        pmetal_runtime::check_unobserved_error().unwrap();
        assert!(stats.loss.is_finite(), "loss {}", stats.loss);
        assert!(stats.grad_norm.is_some_and(f32::is_finite));

        let after = adapter_values(&student);
        assert_ne!(
            before, after,
            "the step left the student's adapters as they were"
        );
    }

    /// The step clips the gradient to `max_grad_norm` and reports the norm it
    /// clipped from: a norm over the limit is reported whole, and the update
    /// that follows is bounded by the limit (the step of plain SGD is
    /// `lr · min(1, max_norm / norm)` of the gradient).
    #[test]
    fn the_gradient_is_clipped_to_the_configured_norm() {
        use pmetal_runtime::compat::optimizers::AdamW;
        let mut teacher = pmetal_models::DynamicModel::from_config(CONFIG).unwrap();
        let batch = batch_of(&[1, 2, 3, 4, 5, 6], &[1, 2, 3, 4, 5, 6], 2);

        let mut norms = Vec::new();
        for max_norm in [1e-6, 0.0] {
            pmetal_runtime::compat::random::seed(5);
            let mut student = student_model();
            let mut trainer = test_trainer();
            trainer
                .loop_state
                .config
                .training
                .gradient_accumulation_steps = 1;
            trainer.loop_state.config.training.max_grad_norm = max_norm;
            let mut optimizer = AdamW::new(1e-2, 0.0);
            let stats = trainer
                .train_step(&mut student, &mut teacher, &batch, &mut optimizer, 1)
                .unwrap();
            norms.push(stats.grad_norm);
        }
        // Clipping on reports the (unclipped) norm, which is far above 1e-6;
        // clipping off (0) reports none.
        assert!(norms[0].is_some_and(|n| n > 1e-6), "{:?}", norms[0]);
        assert_eq!(norms[1], None);
    }

    /// Distillation from a top-k cache trains against the kept logits
    /// without ever forming `[batch, seq, vocab]` teacher logits, and agrees
    /// with distilling from the same teacher with every logit kept.
    #[test]
    fn a_top_k_cache_distills_without_densifying() {
        let mut teacher = pmetal_models::DynamicModel::from_config(CONFIG).unwrap();
        let dir = tempdir().unwrap();
        let mut cache = LogitCache::new(dir.path(), CompressionMethod::TopK, 8).unwrap();
        let dataset = TrainingDataset::from_samples(vec![
            pmetal_data::Sample::new(vec![1, 2, 3, 4, 5, 6]),
            pmetal_data::Sample::new(vec![7, 8, 9, 10, 11, 12]),
        ]);
        generate_teacher_logit_cache(&mut teacher, &dataset, &mut cache, "teacher", 6).unwrap();
        assert_eq!(cache.metadata().compression, "top_k");

        let mut source = CachedTeacher(&cache);
        let batch = batch_of(
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            2,
        );
        let signal = source.signal(&batch).unwrap();
        let TeacherSignal::Sparse(sparse) = &signal else {
            panic!("a top-k cache must give the sparse signal");
        };
        assert_eq!(sparse.shape(), &[2, 6, 8]);

        // The same positions from the live teacher, restricted to the kept
        // logits, give the same loss.
        let student = student_model();
        let mut student = student;
        let logits = student.forward(&batch.input_ids, None).unwrap();
        let trainer = test_trainer();
        let (labels, weights) = shifted_targets(&batch.labels, -100);
        let sparse_loss = trainer
            .compute_distillation_output(&signal, &logits, Some(&labels), &weights, 0, 1)
            .unwrap()
            .soft
            .item_f32();
        let dense_signal = TeacherSignal::Dense(sparse.to_dense());
        let dense_loss = trainer
            .compute_distillation_output(&dense_signal, &logits, Some(&labels), &weights, 0, 1)
            .unwrap()
            .soft
            .item_f32();
        assert!(sparse_loss.is_finite() && sparse_loss > 0.0);
        assert!(
            (sparse_loss - dense_loss).abs() < 1e-4 * dense_loss.abs().max(1.0),
            "sparse {sparse_loss} vs dense-of-the-kept {dense_loss}"
        );
    }

    /// A corpus of prompt (masked) and completion rows.
    fn corpus() -> TrainingDataset {
        let rows: Vec<(Vec<u32>, usize)> = vec![
            (vec![1, 2, 3, 20, 21, 22, 23, 24], 3),
            (vec![4, 5, 30, 31, 32, 33, 34, 35], 2),
            (vec![6, 7, 8, 9, 40, 41, 42, 43], 4),
            (vec![10, 11, 50, 51, 52, 53, 54, 55], 2),
        ];
        TrainingDataset::from_samples(
            rows.into_iter()
                .map(|(ids, prompt)| {
                    let labels = ids
                        .iter()
                        .enumerate()
                        .map(|(i, &t)| if i < prompt { -100 } else { i64::from(t) })
                        .collect();
                    pmetal_data::Sample::with_labels(ids, labels)
                })
                .collect(),
        )
    }

    fn distill_config(loss: pmetal_distill::LossConfig) -> DistillationTrainer {
        DistillationTrainer::new(
            distiller_with(loss),
            TrainingLoopConfig {
                training: pmetal_core::TrainingConfig {
                    learning_rate: 0.03,
                    num_epochs: 25,
                    warmup_steps: 0,
                    lr_scheduler: pmetal_core::LrSchedulerType::Constant,
                    weight_decay: 0.0,
                    gradient_accumulation_steps: 1,
                    ..Default::default()
                },
                dataloader: DataLoaderConfig {
                    batch_size: 2,
                    max_seq_len: 8,
                    shuffle: false,
                    ..Default::default()
                },
                checkpoint_every: 0,
                eval_every: 0,
                log_every: 1000,
                ..Default::default()
            },
        )
        .unwrap()
    }

    /// The student's distillation loss on the corpus, against the live
    /// teacher, without gradients.
    fn corpus_loss(
        trainer: &mut DistillationTrainer,
        student: &mut pmetal_lora::AdaptedModel,
        teacher: &mut pmetal_models::DynamicModel,
    ) -> f64 {
        trainer
            .evaluate(student, &mut LiveTeacher(teacher), &corpus(), 1)
            .unwrap()
    }

    /// Distilling from a live teacher pulls the student's distribution toward
    /// the teacher's, off-policy (the corpus) and on-policy (the student's own
    /// samples), measured on the same corpus tokens; the objective improves,
    /// the adapters move, and a run with nothing to learn from does not.
    #[test]
    fn distillation_pulls_the_student_toward_the_teacher() {
        let loss = pmetal_distill::LossConfig {
            // Soft loss only: the corpus tokens are not the teacher's choices.
            alpha: 1.0,
            temperature: 1.0,
            ..Default::default()
        };
        for on_policy in [None, Some(1.0f32)] {
            pmetal_runtime::compat::random::seed(21);
            let mut teacher = pmetal_models::DynamicModel::from_config(CONFIG).unwrap();
            pmetal_runtime::compat::random::seed(22);
            let mut student = student_model();
            let mut trainer = distill_config(loss.clone());
            if let Some(probability) = on_policy {
                trainer = trainer
                    .with_on_policy(OnPolicyConfig {
                        probability,
                        max_new_tokens: 3,
                        temperature: 1.0,
                        stop_tokens: vec![],
                    })
                    .unwrap();
            }
            let before = corpus_loss(&mut trainer, &mut student, &mut teacher);
            let adapters = adapter_values(&student);
            trainer
                .run(&mut student, &mut teacher, corpus(), None, None)
                .unwrap();
            pmetal_runtime::check_unobserved_error().unwrap();
            let after = corpus_loss(&mut trainer, &mut student, &mut teacher);
            assert_ne!(adapters, adapter_values(&student));
            assert!(
                // Trained on its own samples the student improves on the corpus less
                // directly than on the corpus itself.
                after < if on_policy.is_some() { 0.85 } else { 0.6 } * before,
                "on_policy {on_policy:?}: KL {before:.4} -> {after:.4}"
            );
        }
    }

    /// The same run from a precomputed top-k cache, never forming the dense
    /// teacher.
    #[test]
    fn offline_distillation_from_a_top_k_cache_learns_too() {
        let loss = pmetal_distill::LossConfig {
            alpha: 1.0,
            temperature: 1.0,
            ..Default::default()
        };
        pmetal_runtime::compat::random::seed(21);
        let mut teacher = pmetal_models::DynamicModel::from_config(CONFIG).unwrap();
        pmetal_runtime::compat::random::seed(22);
        let mut student = student_model();
        let dir = tempdir().unwrap();
        let mut cache = LogitCache::new(dir.path(), CompressionMethod::TopK, 16).unwrap();
        generate_teacher_logit_cache(&mut teacher, &corpus(), &mut cache, "teacher", 8).unwrap();

        let mut trainer = distill_config(loss);
        let before = corpus_loss(&mut trainer, &mut student, &mut teacher);
        trainer
            .run_offline(&mut student, &cache, corpus(), None, None)
            .unwrap();
        pmetal_runtime::check_unobserved_error().unwrap();
        let after = corpus_loss(&mut trainer, &mut student, &mut teacher);
        assert!(after < 0.6 * before, "KL {before:.4} -> {after:.4}");
    }

    #[test]
    fn a_distillation_the_trainer_cannot_run_is_refused() {
        let with = |loss: pmetal_distill::LossConfig| {
            DistillationTrainer::new(distiller_with(loss), TrainingLoopConfig::default())
        };
        assert!(with(pmetal_distill::LossConfig::default()).is_ok());
        assert!(
            with(pmetal_distill::LossConfig {
                hidden_state: Some(pmetal_distill::HiddenStateConfig {
                    layer_mapping: vec![(1, 1)],
                    loss_type: Default::default(),
                    weight: 0.1,
                    use_projection: false,
                }),
                ..Default::default()
            })
            .is_err()
        );
        assert!(
            with(pmetal_distill::LossConfig {
                outcome_supervised: true,
                ..Default::default()
            })
            .is_err()
        );
        assert!(
            test_trainer()
                .with_on_policy(OnPolicyConfig {
                    probability: 1.5,
                    max_new_tokens: 4,
                    temperature: 1.0,
                    stop_tokens: vec![],
                })
                .is_err()
        );
    }
}
