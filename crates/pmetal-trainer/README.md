# pmetal-trainer

Training loops and optimization strategies for LLM fine-tuning.

## Overview

This crate provides the training infrastructure for PMetal, including various training methods, learning rate scheduling, checkpointing, and callback systems.

## Training Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| **SFT** | Supervised Fine-Tuning | General instruction tuning |
| **LoRA** | Low-Rank Adaptation | Parameter-efficient fine-tuning |
| **Full fine-tune** | Every weight trains; saves a whole model | Memory-checked up front (`full_finetune`) |
| **DPO / Robust DPO / IPO / hinge** | Preference optimization against a reference | Pairwise preference data |
| **GRPO** | Group Relative Policy Optimization | Efficient PPO alternative |
| **DAPO** | Decoupled Clip and Dynamic Sampling PO | GRPO mode (`GrpoConfig::for_dapo`) |
| **Dr. GRPO** | GRPO without length or difficulty bias | GRPO mode (`GrpoConfig::for_dr_grpo`) |
| **GSPO** | Group Sequence Policy Optimization | GRPO mode (`GrpoConfig::for_gspo`), sequence-level ratio |
| **ORPO** | Odds Ratio Preference Optimization | Reference-free, SFT and alignment in one stage |
| **SimPO** | Simple Preference Optimization | Reference-free, length-normalized |
| **KTO** | Kahneman-Tversky Optimization | Unpaired good/bad feedback |
| **Distillation** | Knowledge distillation | Teacher→student transfer |
| **ANE** | Apple Neural Engine training | Power-efficient on-device training |
| **RLKD** | RL with Knowledge Distillation | GRPO + teacher distillation |
| **Embedding** | Sentence-transformer training | InfoNCE, Triplet, CoSENT contrastive |

## Usage

### Basic Training Loop

`TrainingLoop::new` takes a single `TrainingLoopConfig` and checks it: a configuration that cannot
work (a zero batch size, a label smoothing outside `[0, 1)`) is an error here, not a failure
partway through a run. The model and dataset go to `run`, which trains the model in place.

```rust,no_run
use pmetal_core::TrainingConfig;
use pmetal_data::{DataLoaderConfig, TrainingDataset};
use pmetal_lora::DynamicLoraModel;
use pmetal_trainer::{ProgressCallback, TrainingLoop, TrainingLoopConfig};

fn train(
    mut model: DynamicLoraModel,
    train_dataset: TrainingDataset,
) -> anyhow::Result<DynamicLoraModel> {
    let loop_config = TrainingLoopConfig {
        training: TrainingConfig {
            batch_size: 4,
            gradient_accumulation_steps: 4,
            learning_rate: 2e-4,
            num_epochs: 1,
            max_grad_norm: 1.0,
            ..Default::default()
        },
        dataloader: DataLoaderConfig { batch_size: 4, ..Default::default() },
        use_sequence_packing: true,
        ..Default::default()
    };

    let mut training_loop = TrainingLoop::new(loop_config)?;
    training_loop.add_callback(Box::new(ProgressCallback::new(1_000)));

    training_loop.run(&mut model, train_dataset, None, None)?;
    Ok(model)
}
```

`run` packs short sequences when `use_sequence_packing` is set and the model allows it, trains on
this rank's share of the data under data parallelism, takes an optimizer step every
`gradient_accumulation_steps` micro-batches (the mean gradient over all their tokens), and
evaluates once per micro-batch. Drive the iteration yourself with
`train_step(&mut model, &batch, &mut optimizer)` if you need control over batching.

### With Checkpointing

```rust,no_run
use pmetal_data::TrainingDataset;
use pmetal_lora::DynamicLoraModel;
use pmetal_trainer::{CheckpointManager, TrainingLoop};

fn train_with_checkpoints(
    mut model: DynamicLoraModel,
    train_dataset: TrainingDataset,
    training_loop: &mut TrainingLoop,
) -> anyhow::Result<DynamicLoraModel> {
    let checkpoints = CheckpointManager::new("output/checkpoints")?
        .with_max_checkpoints(3)
        .with_save_best(true);

    // Resume from the newest checkpoint if there is one: its weights, the optimizer's
    // state and the data loader's position, so the run takes the steps the interrupted
    // one would have.
    if training_loop.resume_from(&mut model, &checkpoints)? {
        println!("resumed at step {}", training_loop.current_step());
    }

    // Hand the manager to `run` and it saves on the configured cadence.
    training_loop.run(&mut model, train_dataset, None, Some(&checkpoints))?;
    Ok(model)
}
```

### Full-Parameter Fine-Tuning

`DynamicLoraModel::from_pretrained_full` is the inference path's own model with no adapters and every
parameter trainable, so it runs through the same `TrainingLoop` (optimizers, gradient checkpointing,
packing). `full_finetune` estimates the memory a run needs, so it can be refused before anything
loads, and saves the result as a whole Hugging Face-layout model.

```rust,no_run
use std::path::Path;

use pmetal_data::TrainingDataset;
use pmetal_lora::{DynamicLoraModel, FullFineTune};
use pmetal_runtime::compat::ModuleParametersExt;
use pmetal_trainer::TrainingLoop;
use pmetal_trainer::full_finetune::{plan_for_model, save_full_model};

fn full_finetune(
    source: &Path,
    output: &Path,
    dataset: TrainingDataset,
    mut training_loop: TrainingLoop,
) -> anyhow::Result<()> {
    let mut model = DynamicLoraModel::from_pretrained_full(
        source,
        FullFineTune { freeze_embeddings: false },
    )?;
    // Every parameter must land under a tensor name the source checkpoint has;
    // find out now, not after the run.
    let plan = plan_for_model(source, &model.flatten_params())?;

    training_loop.run(&mut model, dataset, None, None)?;

    // Sharded safetensors under the checkpoint's own names, with its config,
    // tokenizer and chat template.
    save_full_model(&model.flatten_params(), &plan, source, output)?;
    Ok(())
}
```

### GRPO Rewards

A `CombinedReward` is a weighted sum of `RewardFunction`s. Besides the built-in ones, an
`ExternalCommandReward` keeps one process of yours for the run and scores each batch over JSON lines
(prompts, completions and the dataset row's other columns in; `{"rewards": [...]}` out), with a
timeout, a process-group kill when it is dropped, and an error, never a zero, for a completion it
cannot score. `RegexReward` and `JsonReward` (optionally against a JSON Schema) check the text. The
whole job a `GrpoSpec` describes is `grpo::run_grpo_job`, which takes rewards of the caller's own in
`GrpoJobOptions::extra_rewards`.

```rust,no_run
use std::time::Duration;

use pmetal_trainer::{
    CombinedReward, ExternalCommandReward, ExternalRewardConfig, GrpoError, JsonReward, RegexReward,
};

fn rewards() -> Result<CombinedReward, GrpoError> {
    let judge = ExternalCommandReward::spawn(
        ExternalRewardConfig::from_command_line("python3 judge.py --strict")?
            .with_timeout(Duration::from_secs(60)),
    )?;
    Ok(CombinedReward::new()
        .add(Box::new(judge), 2.0)
        .add(Box::new(RegexReward::new(r"^\d+$")?), 0.1)
        .add(Box::new(JsonReward::any()), 0.5))
}
```

### Task Evaluation

`eval::evaluate` scores a benchmark task (GSM8K, MMLU, ARC-Challenge, HellaSwag, a subset of IFEval,
or your own JSONL) against anything that implements `EvalModel` (the log-likelihood of a
continuation, and a greedy generation); `LoadedModel` is the inference path's own model. A
`TaskReport` holds each metric with its standard error.

```rust,no_run
use pmetal_data::Tokenizer;
use pmetal_models::DynamicModel;
use pmetal_trainer::eval::{EvalError, LoadedModel, Task, TaskData, TaskOptions, evaluate};

fn score(model: DynamicModel, tokenizer: &Tokenizer, data: &TaskData) -> Result<(), EvalError> {
    let mut model = LoadedModel::base(model).with_batch_size(16);
    let options = TaskOptions { limit: 50, ..TaskOptions::default() };
    let report = evaluate(&mut model, tokenizer, Task::Gsm8k, data, &options, &mut |_, _| {})?;
    println!("{:?}", report.metrics["exact_match,strict-match"]);
    Ok(())
}
```

## Optimizers

| Optimizer | Description |
|-----------|-------------|
| **AdamW Groups** | AdamW with per-parameter-group learning rates |
| **SGD, Lion, Adafactor** | Momentum SGD, sign-momentum Lion, factored-moment Adafactor |
| **Muon** | Newton-Schulz-orthogonalized momentum on hidden matrices, AdamW on the rest |
| **Schedule-free AdamW** | Iterate averaging instead of a decaying schedule |
| **8-bit AdamW** | Blockwise-quantized moments, a quarter of the optimizer memory |

## Learning Rate Schedulers

| Scheduler | Description |
|-----------|-------------|
| Constant | Fixed learning rate |
| Linear | Linear warmup and decay |
| Cosine | Cosine annealing |
| Cosine with Restarts | Cosine with periodic warm restarts |
| Polynomial | Polynomial decay |
| WSD | Warmup-Stable-Decay schedule |

## Modules

| Module | Description |
|--------|-------------|
| `training_loop` | Main training orchestration |
| `sft` | Supervised fine-tuning trainer |
| `preference` | DPO, Robust DPO, IPO, hinge, SimPO, ORPO (`PreferenceTrainer`) and KTO (`KtoTrainer`) |
| `grpo` | Group Relative Policy Optimization |
| `ane_training` | ANE training loop (feature-gated: `ane`) |
| `distillation` | Knowledge distillation orchestration |
| `rlkd` | Reinforcement Learning with Knowledge Distillation |
| `embedding_trainer` | Sentence-transformer fine-tuning |
| `contrastive_loss` | InfoNCE, Triplet, CoSENT loss functions |
| `orchestrator` | Unified training pipeline (shared across CLI/GUI/TUI/easy) |
| `optimizer` | AdamW, SGD with momentum, Lion, Adafactor, Muon, schedule-free AdamW and 8-bit AdamW, in parameter groups |
| `mlx_metal_optimizer` | Metal-accelerated AdamW updates |
| `adaptive_lr` | EMA-based adaptive learning rate control |
| `checkpoint` | Checkpoint save/load |
| `scheduler` | Learning rate schedulers |
| `callbacks` | Training callbacks (`MetricsJsonCallback`, `StepMetrics`) |
| `param_groups` | Per-layer learning rates |
| `distributed_bridge` | Distributed training sync (feature-gated: `distributed`) |

## Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `batch_size` | Micro-batch size | 4 |
| `gradient_accumulation_steps` | Accumulation steps | 1 |
| `learning_rate` | Initial learning rate | 2e-4 |
| `max_grad_norm` | Gradient clipping | 1.0 |
| `warmup_steps` | LR warmup steps | 0 |
| `weight_decay` | L2 regularization | 0.0 |

## License

MIT OR Apache-2.0
