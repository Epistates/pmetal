# pmetal grpo

GRPO and DAPO reasoning training with reward functions and sampling.

Train models for reasoning tasks using Group Relative Policy Optimization (GRPO) or Decoupled Alignment with Policy Optimization (DAPO).

## Usage

```bash
pmetal grpo \
  --model <MODEL> \
  --dataset <DATASET> \
  [OPTIONS]
```

## Examples

```bash
# GRPO with reasoning rewards
pmetal grpo \
  --model Qwen/Qwen3-0.6B \
  --dataset reasoning.jsonl \
  --reasoning-rewards

# DAPO variant
pmetal grpo \
  --model Qwen/Qwen3-0.6B \
  --dataset reasoning.jsonl \
  --dapo

# With speculative decoding (2-4× faster rollouts)
pmetal grpo \
  --model Qwen/Qwen3-0.6B \
  --dataset reasoning.jsonl \
  --speculative --speculative-draft-tokens 3

# VLM mode with image inputs
pmetal grpo \
  --model Qwen/Qwen2-VL-2B \
  --dataset vlm_reasoning.jsonl \
  --vlm --max-image-size 512

# ML reward model scoring
pmetal grpo \
  --model Qwen/Qwen3-0.6B \
  --dataset reasoning.jsonl \
  --reward-model reward-model-path \
  --reward-model-weight 0.5 --async-rewards
```

## Dataset Format

GRPO expects a reasoning dataset:

```json
{"problem": "What is 15 × 23?", "thinking": "15 × 23 = 15 × 20 + 15 × 3 = 300 + 45 = 345", "solution": "345"}
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | *required* | Model ID or local path |
| `--dataset` | *required* | Reasoning dataset (JSONL) |
| `--dapo` | `false` | Use DAPO variant |
| `--optimizer` | `adamw` | `adamw`, `sgd` (momentum 0.9), `lion`, `adafactor`. Lion wants a 3-10x smaller learning rate and 3-10x larger weight decay than AdamW |
| `--reasoning-rewards` | `false` | Enable reasoning-aware rewards |
| `--speculative` | `false` | Speculative decoding for faster rollouts |
| `--speculative-draft-tokens` | `3` | Draft tokens per speculative step |
| `--vlm` | `false` | Vision-Language Model mode |
| `--max-image-size` | — | Max image dimension for VLM |
| `--reward-model` | — | Pretrained reward model path/ID |
| `--reward-model-weight` | — | Weight for ML reward model scores |
| `--async-rewards` | `false` | Background reward scoring |

## Methods

| Method | Description |
|--------|-------------|
| GRPO | Group Relative Policy Optimization — samples multiple completions and optimizes relative to group reward |
| DAPO | Decoupled GRPO — separates alignment and policy optimization for more stable training |

## See Also

- [Training Methods](/training/methods/) — All training method details
- [pmetal train](/cli/train/) — SFT/LoRA training
