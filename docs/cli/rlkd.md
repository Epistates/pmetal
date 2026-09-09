# pmetal rlkd

Reinforcement Learning with Knowledge Distillation — combines GRPO with teacher distillation.

Combines GRPO policy gradient optimization with knowledge distillation from a frozen teacher model. The loss formula is `L = (1 - alpha) * L_grpo + alpha * L_distill`.

## Usage

```bash
pmetal rlkd \
  --model <STUDENT_MODEL> \
  --teacher-model <TEACHER_MODEL> \
  --dataset <DATASET> \
  --output <OUTPUT_DIR> \
  [OPTIONS]
```

## Examples

```bash
# Basic RLKD with teacher distillation
pmetal rlkd \
  --model Qwen/Qwen3-0.6B \
  --teacher-model Qwen/Qwen3-4B \
  --dataset reasoning.jsonl \
  --output ./output/rlkd

# With annealing alpha (reduce distillation over time)
pmetal rlkd \
  --model Qwen/Qwen3-0.6B \
  --teacher-model Qwen/Qwen3-4B \
  --dataset reasoning.jsonl \
  --distill-alpha 0.5 --final-alpha 0.1 --anneal-alpha
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | *required* | Student/policy model ID or local path |
| `--teacher-model` | *required* | Teacher model ID or local path |
| `--dataset` | *required* | Training dataset (JSONL) |
| `--output` | `./output/rlkd` | Output directory |
| `--distill-alpha` | `0.5` | Distillation weight (0 = pure GRPO, 1 = pure distillation) |
| `--final-alpha` | — | Final alpha for annealing schedule |
| `--anneal-alpha` | `false` | Enable alpha annealing over training |
| `--lora-r` | `16` | LoRA rank |
| `--learning-rate` | `2e-4` | Learning rate |
| `--reasoning-rewards` | `false` | Enable reasoning-aware reward functions |

All standard LoRA/training arguments from `pmetal train` are also accepted.

## See Also

- [Training Overview](/training/overview/) — All training methods
- [Knowledge Distillation](/training/distillation/) — Distillation methods
- [GRPO Training](/cli/grpo/) — GRPO without distillation
