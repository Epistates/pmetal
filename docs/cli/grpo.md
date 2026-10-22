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

# Your own reward: a command that scores each batch (repeat --reward-cmd for several)
pmetal grpo \
  --model Qwen/Qwen3-0.6B \
  --dataset prompts.jsonl \
  --reward-cmd "python3 judge.py --strict" --reward-weight 2 \
  --reward-cmd ./run_tests.sh --reward-weight 0.5 \
  --reward-regex '^\d+$' --reward-regex-weight 0.1

# Completions that must be a JSON document of a given shape
pmetal grpo \
  --model Qwen/Qwen3-0.6B \
  --dataset prompts.jsonl \
  --reward-json-schema schema.json
```

## Rewards

Every reward scores each completion with a number; a completion's reward is the weighted sum of the
rewards that apply, and its advantage is that reward relative to the group's. A reward that cannot
score a completion fails the run with a message naming it: it is never a silent `0`, which would
train the policy away from a completion that was simply never judged.

| Reward | Turned on by | Weight |
|--------|--------------|--------|
| Accuracy of the final answer (`<answer>…</answer>`, the last `\boxed{…}`, or the last line) against the row's `answer` | every row has an `answer` column | 1 |
| Reasoning format (`<think>`/`<answer>` tags) and length | `--reasoning-rewards` | 1 and 0.2 |
| A reward command | `--reward-cmd`, repeatable | `--reward-weight`, in the order of the commands (1 without one) |
| A pattern the completion matches anywhere (1, else 0) | `--reward-regex` | `--reward-regex-weight` |
| The completion is a JSON document, alone or in a fenced block (1, else 0) | `--reward-json` | `--reward-json-weight` |
| The same, and valid against a JSON Schema file | `--reward-json-schema` | `--reward-json-weight` |
| A pretrained reward model | `--reward-model` | `--reward-model-weight` |

Python callables join them from `pmetal.grpo(..., reward_funcs=[...], reward_weights=[...])` (see
[Python: full control](/python/full-control/#grpo-with-python-reward-functions)).

### Reward commands

A reward command is started once, when the run begins, and kept for the whole run, so whatever it
loads (a sandbox, a judge model, a test suite) is paid for once. It is split into arguments as a
shell would split it, but run without one. Each batch of completions (the group for one prompt) is one
JSON line on its standard input, and it answers with one JSON line on its standard output:

```text
-> {"prompts": ["<prompt text>", "<prompt text>"], "completions": ["...", "..."], "answer": ["42", "42"], "difficulty": [3, 3]}
<- {"rewards": [1.0, 0.0]}
```

`prompts` is the prompt as the model saw it (chat template applied) and `completions` the decoded
completions. Every other column of the dataset row (`answer`, `tests`, an id, anything) is sent as a
list with one entry per completion, as the dataset has it; the prompt column and images are not.
The command answers `{"rewards": [...]}` with one finite number per completion, or
`{"error": "why"}` for a batch it cannot score. Anything it writes to standard error is passed on to
yours, and its last lines are in the error if it fails.

The run fails, with the command's name and what it last said, when the command exits, does not
answer within `--reward-timeout` seconds (120; the first batch included, so a command that loads a
model on start has that long), answers anything but the protocol, gives the wrong number of rewards,
or gives one that is not a finite number. A command should exit when its standard input ends, which
is how the trainer tells it the run is over; its whole process group is killed after two seconds
either way (and when the run is stopped), so nothing it started outlives the run.

```python
#!/usr/bin/env python3
import json, sys

for line in sys.stdin:
    batch = json.loads(line)
    rewards = [float(c.strip() == a) for c, a in zip(batch["completions"], batch["answer"])]
    print(json.dumps({"rewards": rewards}), flush=True)
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
| `--optimizer` | `adamw` | `adamw`, `sgd` (momentum 0.9), `lion`, `adafactor`, `muon`, `adamw_schedule_free`, `adamw_8bit`. Lion wants a 3-10x smaller learning rate and 3-10x larger weight decay than AdamW; schedule-free AdamW wants a constant rate |
| `--reasoning-rewards` | `false` | Enable reasoning-aware rewards |
| `--speculative` | `false` | Speculative decoding for faster rollouts |
| `--speculative-draft-tokens` | `3` | Draft tokens per speculative step |
| `--vlm` | `false` | Vision-Language Model mode |
| `--max-image-size` | — | Max image dimension for VLM |
| `--reward-model` | — | Pretrained reward model path/ID |
| `--reward-model-weight` | — | Weight for ML reward model scores |
| `--async-rewards` | `false` | Background reward scoring |
| `--reward-cmd` | — | A command that scores completions over JSON lines (see [Reward commands](#reward-commands)); repeatable |
| `--reward-weight` | `1` | The weight of each `--reward-cmd`, in order; a negative weight penalises |
| `--reward-timeout` | `120` | Seconds a reward command has to answer one batch |
| `--reward-regex` | — | A pattern the completion should match anywhere (1, else 0) |
| `--reward-regex-weight` | `1` | Weight of `--reward-regex` |
| `--reward-json` | `false` | Reward a completion that is a JSON document |
| `--reward-json-schema` | — | A JSON Schema file the JSON document must be valid against (includes `--reward-json`) |
| `--reward-json-weight` | `1` | Weight of `--reward-json` and `--reward-json-schema` |

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
