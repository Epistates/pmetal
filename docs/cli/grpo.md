# pmetal grpo

GRPO and DAPO reasoning training with reward functions and sampling.

Train models for reasoning tasks with Group Relative Policy Optimization (GRPO) and its variants: DAPO, Dr. GRPO and GSPO. The policy is trained with LoRA and saved as an adapter (`adapter_model.safetensors` and `adapter_config.json`) in the output directory.

GRPO samples its own completions, so a run needs something to score them by: `--reasoning-rewards`, an `answer` in every dataset row (rewards correct final answers), or `--reward-model`. A run with none of these stops with an error instead of training on nothing.

Ctrl-C (SIGINT) or SIGTERM finishes the step in flight, saves the policy's adapter and exits with status 130. `--log-events` (a global option) writes the run's events as JSONL.

## Usage

```bash
pmetal grpo \
  --model <MODEL> \
  --dataset <DATASET> \
  [OPTIONS]
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `-m`, `--model` | required | Model ID or local path |
| `-d`, `--dataset` | required | Prompt dataset (see Dataset Format) |
| `-o`, `--output` | `./output/grpo` | Output directory |
| `--num-generations` | `8` | Completions sampled per prompt (the group size) |
| `--beta` | `0.0` | KL penalty coefficient; the frozen reference model is loaded only when this is above 0 |
| `--learning-rate` | `5e-6` | Learning rate |
| `--optimizer` | `adamw` | `adamw`, `sgd` (momentum 0.9), `lion`, `adafactor`, `muon`, `adamw_schedule_free` or `adamw_8bit`. Lion wants a 3-10x smaller learning rate and 3-10x larger weight decay than AdamW; schedule-free AdamW wants a constant rate |
| `--epochs` | `1` | Training epochs |
| `--lora-r` | `16` | Policy LoRA rank |
| `--lora-alpha` | `32.0` | Policy LoRA alpha |
| `--max-seq-len` | `512` | Maximum prompt length |
| `--max-completion-length` | `512` | Maximum completion length per generation |
| `--seed` | `42` | Random seed |
| `--checkpoint-every` | `50` | Save a LoRA checkpoint every N policy updates; `0` disables |
| `--resume` | `false` | Resume the LoRA weights from the latest checkpoint in the output directory |
| `--dapo` | `false` | DAPO recipe: token-level loss, no KL, clip-higher (0.2, 0.28), dynamic sampling, overlong penalty, at least 16 completions |
| `--loss-type` | `dapo` | `dapo` (token-level), `grpo` (each completion averaged over its length), `dr_grpo` (constant normalizer, no std scaling), `gspo` (sequence-level ratio, clip 3e-4/4e-4; needs `--num-iterations` above 1) |
| `--num-iterations` | `1` | Optimizer updates per generation batch (μ). The ratio against the generating policy is 1 on the first, so clipping and GSPO only act from the second |
| `--reasoning-rewards` | `false` | Reasoning-aware rewards (length, formatting, and the accuracy of the final answer when every row has an `answer`) |
| `--vlm` | `false` | Vision-language model mode, with image inputs |
| `--max-image-size` | `336` | Maximum image size in pixels per side for VLM preprocessing |
| `--reward-model` | none | Pretrained reward model (path or Hub ID) that scores completions |
| `--reward-model-weight` | `1.0` | Weight of the reward model in the combined reward |
| `--reward-model-max-length` | `2048` | Maximum input length for the reward model, in tokens |
| `--reward-model-template` | none | Chat template for the reward model |
| `--async-rewards` | `false` | Score rewards in a background thread, pipelined with the training step |
| `--prompt-column` | none | Column holding the prompt |
| `--response-column` | none | Column holding the reference answer |
| `--grpo-kv-bits` | none | KV cache quantization bits for rollout generation (2, 4 or 8) |
| `--log-metrics` | none | Write per-step metrics as JSONL to this path |

## Dataset Format

A row contributes a prompt: a `prompt` (also `question`, `instruction`, `input` or `query`) given as a string or a list of `{"role", "content"}` messages, with an optional `system`; or a `messages` list, whose last assistant turn, if any, is the reference answer. The reference answer is the first of `answer`, `solution`, `reference`, `ground_truth`, `target`, `completion`, `response` and `output` that the row has; a GSM8K-style `... #### 18` keeps what follows the last `####`. `--prompt-column` and `--response-column` name other columns.

```json
{"prompt": "What is 15 × 23?", "answer": "345"}
```

JSONL, JSON, Parquet, Arrow and CSV files are read.

## Methods

| Method | Description |
|--------|-------------|
| GRPO | Group Relative Policy Optimization (arXiv 2402.03300): samples a group of completions per prompt and pushes toward the ones that score above the group mean, with a clipped per-token ratio |
| DAPO | Decoupled Clip and Dynamic sAmpling Policy Optimization (arXiv 2503.14476): token-level loss, clip-higher, dynamic sampling and an overlong penalty (`--dapo`) |
| Dr. GRPO | *Understanding R1-Zero-Like Training* (arXiv 2503.20783): normalizes by a constant and drops the std scaling of advantages, removing GRPO's length and difficulty biases (`--loss-type dr_grpo`) |
| GSPO | Group Sequence Policy Optimization (arXiv 2507.18071): clips one length-normalized likelihood ratio per completion instead of one per token (`--loss-type gspo --num-iterations 2`) |

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

# VLM mode with image inputs
pmetal grpo \
  --model Qwen/Qwen2-VL-2B-Instruct \
  --dataset vlm_reasoning.jsonl \
  --reasoning-rewards --vlm --max-image-size 512

# ML reward model scoring
pmetal grpo \
  --model Qwen/Qwen3-0.6B \
  --dataset reasoning.jsonl \
  --reward-model /path/to/reward-model \
  --reward-model-weight 0.5 --async-rewards
```

## See Also

- [Training Methods](/training/methods/): All training method details
- [pmetal train](/cli/train/): SFT/LoRA training
- [pmetal rlkd](/cli/rlkd/): GRPO with a teacher's distillation loss
