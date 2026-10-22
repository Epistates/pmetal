# Knowledge Distillation

Transfer knowledge from large teacher models to smaller students: online, offline and progressive, with a library of losses.

`pmetal distill` trains a LoRA adapter on a student model against a teacher. The method is chosen with `--method` (or the `--offline` shortcut) and the objective with `--loss-type`.

## Methods

### Online Distillation
Live teacher inference during training. Highest quality but slowest: both teacher and student must fit in memory.

```bash
pmetal distill \
  --teacher Qwen/Qwen3-4B \
  --student Qwen/Qwen3.5-0.8B-Base \
  --dataset train.jsonl
```

### Offline Distillation
Pre-cache teacher logits to disk with compression, then train from the cache. Faster training at the cost of disk space, and the teacher does not have to be in memory during training.

```bash
pmetal distill \
  --teacher Qwen/Qwen3-4B \
  --student Qwen/Qwen3-0.6B \
  --dataset train.jsonl \
  --offline \
  --offline-cache ./output/distilled/teacher_logits
```

`--offline-compression` picks how the cache is stored: `top_k` (the default, `--offline-top-k` entries per position, read back sparse and never densified), `none`, `int8`, `int4`, or `sampled` (Random-Sampling KD, an unbiased estimate of the teacher where a top-k cache is biased toward its head). `--offline-generate` writes the cache before training.

### Progressive Distillation
Gradually reduces the teacher's influence over the run.

### On-Policy Distillation
`--on-policy <share>` (0 to 1) makes that share of the batches use responses the student samples itself, scored by the teacher (GKD); pair it with `--loss-type jsd_skewed`. It applies to online and progressive runs.

### TAID (Temporally Adaptive Interpolated Distillation)
ICLR 2025 method: a logit-space intermediate target whose interpolation adapts to the student's progress. Available as `TaidDistiller` in the library; there is no command for it yet.

## Loss Functions

`--loss-type` takes:

| Value | Description |
|-------|-------------|
| `kl_divergence` (default) | Forward KL at `--temperature` |
| `reverse_kl` | Reverse KL |
| `jensen_shannon` | Symmetric divergence |
| `soft_cross_entropy` | Temperature-scaled cross-entropy |
| `mse_loss` | Mean squared error on the logits |
| `jsd_skewed` | Generalized JSD of GKD; `--skew-alpha` (default 0.5, the symmetric JSD) |
| `skew_kl`, `skew_reverse_kl` | DistiLLM's skew KL and its reverse; `--skew-alpha` (default 0.1) |
| `universal_logit` | Universal Logit Distillation for a teacher and student with different tokenizers; `--universal-logit-top-k` |

The library adds TVD, hinge and logistic ranking losses, hidden-state alignment (MSE, cosine, L1), attention transfer, MiniLLM's policy-gradient objective and a sparse KL against an offline top-k cache; see [`pmetal-distill`](https://github.com/Epistates/pmetal/tree/main/crates/pmetal-distill).

## Reasoning-Aware Distillation

For reasoning models, `--rationale` (with `--rationale-weight`) weights the reasoning span of a response so the thinking process is distilled, not just the final answer.

## See Also

- [pmetal distill](/cli/distill/): CLI distillation
- [Training Overview](/training/overview/): All training methods
