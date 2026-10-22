# pmetal distill

Knowledge distillation: transfer knowledge from a teacher model to a smaller student.

The student is trained with LoRA against the teacher's outputs and saved as an adapter (`adapter_model.safetensors` and `adapter_config.json`) in the output directory. Teacher and student share a tokenizer, except with `--loss-type universal_logit`.

Ctrl-C (SIGINT) or SIGTERM finishes the step in flight, saves the student's adapter and exits with status 130. `--log-events` (a global option) writes the run's events as JSONL.

## Usage

```bash
pmetal distill \
  --teacher <TEACHER_MODEL> \
  --student <STUDENT_MODEL> \
  --dataset <DATASET> \
  [OPTIONS]
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `-t`, `--teacher` | required | Teacher model ID or path |
| `-s`, `--student` | required | Student model ID or path |
| `-d`, `--dataset` | required | Training dataset: a file, a directory of shards or a Hub dataset ID |
| `--eval-dataset` | none | Evaluation dataset (same forms as `--dataset`) |
| `-o`, `--output` | `./output/distilled` | Output directory for the distilled student |
| `--method` | `online` | `online`, `offline` or `progressive` (see Methods) |
| `--offline` | `false` | Shortcut for `--method offline` |
| `--offline-cache` | `<output>/teacher_logits` | Directory that stores or supplies the cached teacher logits |
| `--offline-generate` | `false` | Generate teacher logits that are missing from the cache |
| `--offline-compression` | `top_k` | Compression of a newly created cache: `top_k`, `none`, `int8`, `int4` or `sampled` |
| `--offline-top-k` | `128` | Top-k width of a newly created cache |
| `--loss-type` | `kl_divergence` | See Loss Functions |
| `--skew-alpha` | per loss | Mixing weight of `jsd_skewed` (default 0.5) and of `skew_kl` / `skew_reverse_kl` (default 0.1) |
| `--universal-logit-top-k` | whole vocabulary | Entries `universal_logit` keeps before sorting |
| `--temperature` | `2.0` | Softmax temperature |
| `--alpha` | `0.5` | Blend of soft and hard targets (0 to 1) |
| `--rationale` | `false` | Reasoning-aware (rationale) distillation |
| `--rationale-weight` | `1.0` | Weight of reasoning tokens with `--rationale` |
| `--on-policy` | `0.0` | Share of batches (0 to 1) whose responses the student samples and the teacher scores; online and progressive runs only |
| `--on-policy-max-tokens` | `256` | Longest response sampled for an on-policy batch |
| `--lora-r` | `16` | Student LoRA rank |
| `--lora-alpha` | `32.0` | Student LoRA alpha |
| `--learning-rate` | `2e-5` | Learning rate |
| `--optimizer` | `adamw` | `adamw`, `sgd`, `lion`, `adafactor`, `muon`, `adamw_schedule_free` or `adamw_8bit` |
| `--batch-size` | `1` | Batch size |
| `--epochs` | `1` | Number of epochs |
| `--max-seq-len` | `1024` | Maximum sequence length |
| `--seed` | `42` | Random seed for shuffling and initialization |
| `--checkpoint-every` | `100` | Save a checkpoint every N steps; `0` disables |
| `--resume` | `false` | Resume from the latest checkpoint in the output directory |
| `--text-column` | none | JSONL column holding the training text |
| `--text-columns` | none | Comma-separated columns to concatenate as the text |
| `--column-separator` | blank line | Separator between columns with `--text-columns` |
| `--prompt-column` | none | Column holding the prompt (enables SFT loss masking) |
| `--response-column` | none | Column holding the response |
| `--log-metrics` | none | Write per-step metrics as JSONL to this path |

## Methods

| Method | Description |
|--------|-------------|
| `online` | The teacher runs beside the student: highest quality, slowest |
| `offline` | The teacher's logits are cached on disk first, then reused: faster, uses disk |
| `progressive` | The teacher's influence is reduced over the run |

## Loss Functions

`--loss-type` takes one of these:

| Loss | What it is |
|------|------------|
| `kl_divergence` (default) | Forward KL, teacher to student |
| `reverse_kl` | Reverse KL, student to teacher |
| `jensen_shannon` | Jensen-Shannon divergence |
| `soft_cross_entropy` | Cross-entropy against the teacher's soft targets |
| `mse_loss` | Mean squared error on the logits |
| `jsd_skewed` | The generalized JSD of GKD (Agarwal et al., 2024), mixing weight `--skew-alpha` (default 0.5, the symmetric JSD) |
| `skew_kl` / `skew_reverse_kl` | DistiLLM's skew KL (Ko et al., 2024), mixing weight `--skew-alpha` (default 0.1) |
| `universal_logit` | Universal logit distillation (Boizard et al., 2024) for a teacher with another tokenizer; `--universal-logit-top-k` truncates before sorting |

GKD is `--on-policy 0.5` with `--loss-type jsd_skewed`: that share of the batches are the student's
own samples, scored by the teacher.

## Examples

```bash
# Online distillation (live teacher inference)
pmetal distill \
  --teacher Qwen/Qwen3-4B \
  --student Qwen/Qwen3-0.6B \
  --dataset train.jsonl

# Offline distillation (cached logits)
pmetal distill \
  --teacher Qwen/Qwen3-4B \
  --student Qwen/Qwen3-0.6B \
  --dataset train.jsonl \
  --offline \
  --offline-cache ./output/distilled/teacher_logits

# GKD: half of the batches are the student's own samples
pmetal distill \
  --teacher Qwen/Qwen3-4B \
  --student Qwen/Qwen3-0.6B \
  --dataset train.jsonl \
  --loss-type jsd_skewed --on-policy 0.5

# Use the distilled student
pmetal infer --model Qwen/Qwen3-0.6B --lora ./output/distilled --prompt "Hello"
```

## See Also

- [Training Overview](/training/overview/): All training methods
- [TAID Distillation](/training/distillation/): Advanced TAID method (library only)
- [pmetal grpo](/cli/grpo/) and [pmetal rlkd](/cli/rlkd/): Reinforcement learning with and without a teacher
