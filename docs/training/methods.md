# Training Methods

Detailed guide to each training method — SFT, LoRA, DPO, SimPO, ORPO, KTO, GRPO, and more.

## Supervised Fine-Tuning (SFT)

Standard fine-tuning on instruction/response pairs. Used via `pmetal train` or `easy::finetune()`.

### LoRA
Low-Rank Adaptation — trains small adapter matrices instead of full weights. Parameters:
- **rank** (`--lora-r`): Adapter rank (default: 16)
- **alpha** (`--lora-alpha`): Scaling factor (default: 2× rank)

### QLoRA
4-bit quantized LoRA. Loads base model in NF4/FP4/INT8, trains adapters in full precision.
```bash
pmetal train --model Qwen/Qwen3-0.6B --dataset train.jsonl --quantization nf4
```

### DoRA
Weight-Decomposed LoRA — decomposes weight updates into magnitude and direction for better training stability.
```bash
pmetal train --model Qwen/Qwen3-0.6B --dataset train.jsonl --dora
```

## Preference Optimization

### DPO (Direct Preference Optimization)
Trains on preference pairs (chosen/rejected) without a reward model.
```rust
easy::dpo("model", "preferences.jsonl")
    .dpo_beta(0.1)
    .reference_model("model")
    .run().await?;
```

### SimPO (Simple Preference Optimization)
Simplified DPO without a reference model.

### ORPO (Odds-Ratio Preference Optimization)
Combines SFT and preference optimization in a single stage.

### KTO (Kahneman-Tversky Optimization)
Preference optimization using prospect theory — works with binary feedback (good/bad) instead of pairwise comparisons.

## Reasoning Training

### GRPO (Group Relative Policy Optimization)
Samples multiple completions per prompt, scores them with reward functions, and optimizes policy relative to group performance.
```bash
pmetal grpo --model Qwen/Qwen3-0.6B --dataset reasoning.jsonl --reasoning-rewards
```

**Advanced GRPO features** (added in v0.3.9):
- **VLM mode** (`--vlm`): Vision-Language Model support with image inputs
- **ML reward model** (`--reward-model`): Pretrained reward model scoring alongside heuristic rewards
- **Speculative decoding** (`--speculative`): Draft/verify rollout generation for 2-4× throughput
- **Async reward pipelining** (`--async-rewards`): Background reward scoring concurrent with GPU training

### DAPO (Decoupled Alignment with Policy Optimization)
Decouples the alignment and policy optimization steps for more stable reasoning training.

### RLKD (Reinforcement Learning with Knowledge Distillation)
Combines GRPO policy gradient optimization with distillation from a frozen teacher model. Loss: `L = (1-alpha) * L_grpo + alpha * L_distill`.
```bash
pmetal rlkd --model Qwen/Qwen3-0.6B --teacher Qwen/Qwen3-4B --dataset reasoning.jsonl
```

## Embedding Training

Sentence-transformer fine-tuning for BERT/encoder models with contrastive learning objectives: InfoNCE, Triplet, and CoSENT. Supports pair and triplet datasets with configurable pooling (CLS, Mean, LastToken) and L2 normalization.
```bash
pmetal embed-train --model BAAI/bge-small-en-v1.5 --dataset pairs.jsonl --loss infonce
```

## MTP And Draft Training

`pmetal train-mtp` trains Gemma 4 assistant checkpoints and Qwen3Next/Qwen3.6 MTP predictor checkpoints for exact speculative decoding. Qwen checkpoints exported by this command can be loaded with `pmetal infer --mtp --mtp-model <dir>`.

`pmetal train-draft` trains DFlash block-diffusion draft checkpoints from tokenized shards and a frozen Qwen3 target for use with `pmetal dflash`.

## ANE Training
Automatic Apple Neural Engine training when available. Uses the ANE for forward passes with CPU-based gradient computation. Activated automatically on supported models.

## See Also

- [Training Overview](/training/overview/) — Method availability matrix
- [Distillation](/training/distillation/) — Knowledge distillation methods
