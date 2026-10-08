# pmetal grpo

GRPO and DAPO reasoning training with reward functions and sampling.

Train models for reasoning tasks with Group Relative Policy Optimization (GRPO) and its variants: DAPO, Dr. GRPO and GSPO.

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

# GSPO: a sequence-level ratio, clipped over several updates per batch
pmetal grpo \
  --model Qwen/Qwen3-0.6B \
  --dataset reasoning.jsonl \
  --loss-type gspo --num-iterations 2 --beta 0

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
| `--dapo` | `false` | DAPO recipe: token-level loss, no KL, clip-higher (0.2, 0.28), dynamic sampling, overlong penalty, at least 16 completions |
| `--loss-type` | `dapo` | `dapo` (token-level), `grpo` (each completion averaged over its length), `dr_grpo` (constant normalizer, no std scaling), `gspo` (sequence-level ratio, clip 3e-4/4e-4) |
| `--num-iterations` | `1` | Optimizer updates per generation batch (μ). The ratio against the generating policy is 1 on the first, so clipping and GSPO only act from the second |
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
| GRPO | Group Relative Policy Optimization (arXiv 2402.03300): samples a group of completions per prompt and pushes toward the ones that score above the group mean, with a clipped per-token ratio |
| DAPO | Decoupled Clip and Dynamic sAmpling Policy Optimization (arXiv 2503.14476): token-level loss, clip-higher, dynamic sampling and an overlong penalty (`--dapo`) |
| Dr. GRPO | *Understanding R1-Zero-Like Training* (arXiv 2503.20783): normalizes by a constant and drops the std scaling of advantages, removing GRPO's length and difficulty biases (`--loss-type dr_grpo`) |
| GSPO | Group Sequence Policy Optimization (arXiv 2507.18071): clips one length-normalized likelihood ratio per completion instead of one per token (`--loss-type gspo --num-iterations 2`) |

## See Also

- [Training Methods](/training/methods/) — All training method details
- [pmetal train](/cli/train/) — SFT/LoRA training
