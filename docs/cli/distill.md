# pmetal distill

Knowledge distillation — transfer knowledge from a teacher model to a smaller student.

Distill knowledge from a larger teacher model into a smaller student model. Supports online, offline, and progressive methods.

## Usage

```bash
pmetal distill \
  --teacher <TEACHER_MODEL> \
  --student <STUDENT_MODEL> \
  --dataset <DATASET> \
  [OPTIONS]
```

## Examples

```bash
# Online distillation (live teacher inference)
pmetal distill \
  --teacher Qwen/Qwen3-4B \
  --student Qwen/Qwen3.5-0.8B-Base \
  --dataset train.jsonl

# Offline distillation (cached logits)
pmetal distill \
  --teacher Qwen/Qwen3-4B \
  --student Qwen/Qwen3.5-0.8B-Base \
  --dataset train.jsonl \
  --offline \
  --offline-cache ./output/distilled/teacher_logits
```

## Methods

| Method | Description |
|--------|-------------|
| Online | Live teacher inference during training — highest quality, slowest |
| Offline | Pre-cached logits with compression — faster, uses more disk |
| Progressive | Gradually increase distillation difficulty |

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

Other options: `--temperature`, `--alpha` (the blend of soft and hard targets), `--rationale` and
`--rationale-weight` (reasoning-aware weighting), `--eval-dataset`, `--checkpoint-every` and
`--resume`. `--dataset` and `--eval-dataset` are a file, a directory of shards or a Hub dataset ID.

## See Also

- [Training Overview](/training/overview/) — All training methods
- [TAID Distillation](/training/distillation/) — Advanced TAID method (library only)
