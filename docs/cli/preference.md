# pmetal preference

Preference optimization with LoRA: DPO, IPO, hinge, SimPO and ORPO on prompt/chosen/rejected pairs, or KTO on completions labelled good or bad. `pmetal dpo` is the same command.

The reference model costs no memory. DPO, IPO, hinge and KTO compare the policy with the model as it was before training, and a fresh LoRA adapter starts at zero, so the policy is the reference until its first update. Every example is scored once before training and the log-probabilities are kept: one forward pass per sequence and no second copy of the model.

## Usage

```bash
pmetal preference \
  --model <MODEL> \
  --dataset <DATASET> \
  [--loss dpo|ipo|hinge|simpo|orpo|kto] \
  [OPTIONS]
```

## Examples

```bash
# DPO on a local preference file
pmetal preference --model Qwen/Qwen3-0.6B --dataset pairs.jsonl

# SimPO (reference-free, length-normalized)
pmetal preference --model Qwen/Qwen3-0.6B --dataset pairs.jsonl --loss simpo

# Robust DPO for noisy labels (10% assumed flipped)
pmetal dpo --model Qwen/Qwen3-0.6B --dataset pairs.jsonl --label-smoothing 0.1

# KTO on thumbs-up / thumbs-down data
pmetal preference --model Qwen/Qwen3-0.6B --dataset feedback.jsonl --loss kto --batch-size 4

# Use the trained adapter
pmetal infer --model Qwen/Qwen3-0.6B --lora ./output/preference/lora_weights.safetensors
```

## Objectives

| `--loss` | Data | Reference | Log-probs | Loss |
|----------|------|-----------|-----------|------|
| `dpo` (default) | pairs | yes | summed | `−log σ(β·h)`; with `--label-smoothing ε`, Robust DPO `((1−ε)·softplus(−βh) − ε·softplus(βh)) / (1−2ε)` |
| `ipo` | pairs | yes | averaged | `(h − 1/(2β))²` |
| `hinge` | pairs | yes | summed | `max(0, 1 − β·h)` |
| `simpo` | pairs | no | averaged | `−log σ(β·(Δ − γ/β))` |
| `orpo` | pairs | no | averaged | `NLL(chosen) − β·log σ(log odds(chosen) − log odds(rejected))` |
| `kto` | labelled completions | yes | summed | `w·(1 − σ(β·(r − z)))` desirable, `w·(1 − σ(β·(z − r)))` undesirable |

`h` is the chosen log-ratio minus the rejected log-ratio against the reference, `Δ` the difference of averaged log-probs, `r` a completion's log-ratio, and `z` KTO's KL estimate: the mean log-ratio of mismatched completions in the batch, clamped at zero and kept out of the gradient.

## Data

Rows come from JSONL, a JSON array, Parquet, or a Hugging Face dataset ID.

```json
{"prompt": "What is 2 + 2?", "chosen": "4.", "rejected": "5."}
{"prompt": [{"role": "user", "content": "What is 2 + 2?"}], "chosen": [{"role": "assistant", "content": "4."}], "rejected": [{"role": "assistant", "content": "5."}]}
{"chosen": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}], "rejected": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Go away."}]}
{"prompt": "What is 2 + 2?", "completion": "4.", "label": true}
```

The last shape is KTO's. KTO also reads paired rows, taking the chosen completion as desirable and the rejected one as undesirable. Prompts go through the model's chat template, as `pmetal train` renders them, so the completion is the assistant's turn including its end-of-turn token. Prompts longer than `--max-prompt-length` keep their last tokens; sequences longer than `--max-length` are cut at the end.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | *required* | Model ID or local path |
| `--dataset` | *required* | JSONL / JSON / Parquet file or Hugging Face dataset ID |
| `--output` | `./output/preference` | Where `lora_weights.safetensors` and `adapter_config.json` go |
| `--loss` | `dpo` | `dpo`, `ipo`, `hinge`, `simpo`, `orpo` or `kto` |
| `--beta` | 2.5 for SimPO, 0.1 otherwise | β |
| `--simpo-gamma-ratio` | `0.5` | SimPO's target margin over β (γ/β) |
| `--label-smoothing` | `0.0` | DPO: share of labels assumed flipped; above 0 trains Robust DPO |
| `--desirable-weight` | `1.0` | KTO weight of desirable examples |
| `--undesirable-weight` | `1.0` | KTO weight of undesirable examples |
| `--learning-rate` | `1e-5` | Peak learning rate (cosine decay after warmup) |
| `--batch-size` | `2` | Pairs or KTO examples per micro-batch |
| `--gradient-accumulation-steps` | `8` | Micro-batches per optimizer step |
| `--epochs` | `1` | Passes over the dataset |
| `--max-steps` | — | Stop after this many optimizer steps |
| `--warmup-ratio` | `0.1` | Share of steps spent warming up |
| `--max-grad-norm` | `1.0` | Global gradient-norm clip (0 disables) |
| `--weight-decay` | `0.0` | Weight decay |
| `--optimizer` | `adamw` | `adamw`, `sgd` (momentum 0.9), `lion`, `adafactor`. Lion wants a 3-10x smaller learning rate and 3-10x larger weight decay than AdamW |
| `--lora-r` | `16` | LoRA rank |
| `--lora-alpha` | `32` | LoRA alpha |
| `--max-prompt-length` | `512` | Prompt tokens kept |
| `--max-length` | `1024` | Prompt plus completion tokens kept |
| `--seed` | `42` | Adapter initialization and shuffling |
| `--log-metrics` | — | Per-step metrics as JSONL |

Each step logs the loss, the mean implicit reward of chosen and rejected completions, their margin, the share of pairs ranked correctly, and for KTO the KL estimate.

## Choosing settings

- The defaults follow the reference implementations: β = 0.1 for DPO, IPO, hinge, ORPO and KTO; SimPO's β in the range its authors report (2 to 2.5, sometimes 10) with γ/β = 0.5 as their suggested starting point.
- The learning rate is the setting that matters most. `1e-5` suits LoRA adapters; full fine-tunes in the papers use 5e-7 to 1e-6. Lower it for reasoning-heavy data.
- KTO estimates its reference point from each micro-batch, so give it `--batch-size 4` or more, and an effective batch (batch size times accumulation) of 16 to 128. If one kind of example outnumbers the other, weight them so that `desirable-weight × desirable / (undesirable-weight × undesirable)` is between 1 and 4/3.

## See also

- [Training methods](../training/methods.md)
- [`pmetal grpo`](grpo.md) for online reinforcement learning with reward functions
