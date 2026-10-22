# pmetal rlkd

Reinforcement Learning with Knowledge Distillation: GRPO with a teacher's distillation loss.

Combines GRPO policy gradient optimization with knowledge distillation from a frozen teacher model in one training loop. The loss is `L = (1 - alpha) * L_grpo + alpha * L_distill`. The policy is trained with LoRA and saved as an adapter in the output directory.

Ctrl-C (SIGINT) or SIGTERM finishes the step in flight, saves the policy's adapter and exits with status 130. `--log-events` (a global option) writes the run's events as JSONL.

## Usage

```bash
pmetal rlkd \
  --model <STUDENT_MODEL> \
  --teacher-model <TEACHER_MODEL> \
  --dataset <DATASET> \
  [OPTIONS]
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `-m`, `--model` | required | Policy (student) model ID or local path |
| `--teacher-model` | required | Teacher model ID or local path (frozen, provides soft targets) |
| `-d`, `--dataset` | required | Prompt dataset, in the formats [`pmetal grpo`](/cli/grpo/) reads |
| `-o`, `--output` | `./output/rlkd` | Output directory for the LoRA adapter |
| `--distill-alpha` | `0.3` | Distillation blend: `0` is pure RL, `1` is pure distillation |
| `--final-alpha` | `0.05` | Alpha the annealing schedule ends at |
| `--anneal-alpha` | `false` | Anneal alpha linearly from `--distill-alpha` to `--final-alpha` over the run |
| `--distill-temperature` | `2.0` | Temperature of the distillation soft targets |
| `--num-generations` | `8` | Completions sampled per prompt (the GRPO group size) |
| `--beta` | `0.0` | KL penalty coefficient against the GRPO reference model |
| `--learning-rate` | `5e-6` | Learning rate |
| `--epochs` | `1` | Training epochs |
| `--lora-r` | `16` | Policy LoRA rank |
| `--lora-alpha` | `32.0` | Policy LoRA alpha |
| `--max-seq-len` | `512` | Maximum sequence length (prompt and completion) |
| `--max-completion-length` | `512` | Maximum completion length per generation |
| `--seed` | `42` | Random seed |
| `--reasoning-rewards` | `false` | Reasoning-aware rewards (format and length signals) |
| `--prompt-column` | none | Column holding the prompt |
| `--response-column` | none | Column holding the reference answer |
| `--log-metrics` | none | Write per-step metrics as JSONL to this path |

## Examples

```bash
# Basic RLKD with teacher distillation
pmetal rlkd \
  --model Qwen/Qwen3-0.6B \
  --teacher-model Qwen/Qwen3-4B \
  --dataset reasoning.jsonl \
  --output ./output/rlkd

# Anneal alpha: distill early, mostly RL by the end
pmetal rlkd \
  --model Qwen/Qwen3-0.6B \
  --teacher-model Qwen/Qwen3-4B \
  --dataset reasoning.jsonl \
  --distill-alpha 0.5 --final-alpha 0.1 --anneal-alpha
```

## See Also

- [Training Overview](/training/overview/): All training methods
- [Knowledge Distillation](/training/distillation/): Distillation methods
- [pmetal grpo](/cli/grpo/): GRPO without distillation
- [pmetal distill](/cli/distill/): Distillation without reinforcement learning
