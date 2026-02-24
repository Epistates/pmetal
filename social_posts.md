# PMetal Social Media Announcements

## Reddit Post (r/rust, r/MachineLearning, r/LocalLLaMA)

---

### Title: PMetal - LLM fine-tuning framework for Apple Silicon, written in Rust with custom Metal GPU kernels

### Body:

Hey everyone, open-sourcing PMetal today -- a Rust framework for fine-tuning LLMs natively on Apple Silicon using custom Metal compute shaders.

**What it is:** A 15-crate workspace (~112K lines of Rust, ~9K lines of Metal shaders) that covers the full training pipeline: LoRA/QLoRA adapters, RLHF alignment (DPO, GRPO, DAPO, GSPO, KTO, SimPO, ORPO, PPO), knowledge distillation (TAID + reasoning-aware), and model merging (TIES, DARE, Model Stock, and more).

**Why Rust?** Zero-copy safetensor loading, compile-time architecture validation, fearless concurrency for async data pipelines, and `#[repr(C)]` interop with Metal shaders. The type system catches misconfigurations that Python would only surface at runtime mid-training.

**Custom Metal kernels:** This is the core differentiator. Hand-written `.metal` compute shaders for:

- Fused RMSNorm + LoRA forward (single kernel dispatch instead of 5+ ops)
- Fused cross-entropy loss (logits never materialize the full vocab distribution)
- Fused SwiGLU activation
- FlashAttention for training (forward + backward)
- Fused RoPE embeddings
- Grouped GEMM for MoE routing
- FP8 training kernels
- Fused distillation kernels

Each kernel includes an auto-tuner (`pmetal-metal/tuna`) that profiles tile sizes and threadgroup configurations per-device, so M1 through M4 Ultra all get tuned dispatch parameters.

**Supported model families:** Llama (3.x, 4), Qwen (2, 2-VL, 3, 3-MoE), DeepSeek, Mistral, Gemma, Phi, Granite, Cohere, Nemotron-H, Pixtral, MLlama (vision), Whisper.

**Training features:**

- Custom autograd for LoRA that only stores `x` and `x @ A^T` per layer (rank << hidden), cutting memory ~6x per LoRA layer vs standard autodiff
- Sequence packing with cross-attention masking
- 8-bit Adam, schedule-free optimizers, parameter groups with per-layer LR
- JIT compilation of training steps via MLX
- Streaming checkpoint save/resume
- HuggingFace Hub integration (download + upload)

**What it's not:** This doesn't replace PyTorch for multi-GPU cluster training. It's specifically for the Apple Silicon niche -- M-series Macs and potentially future Apple hardware. If you have an NVIDIA setup, use Unsloth/axolotl/TRL.

Built on top of [mlx-rs](https://github.com/oxideai/mlx-rs) (Rust bindings to Apple's MLX framework). We've been contributing fixes upstream as we go.

Dual-licensed MIT/Apache-2.0.

GitHub: [link]

Happy to answer questions about the Metal kernel design, the custom autograd approach, or anything else.

---

## X/Twitter Thread

---

**Tweet 1 (Hook):**

Releasing PMetal -- an LLM fine-tuning framework for Apple Silicon, written entirely in Rust with hand-tuned Metal GPU kernels.

15 crates. 112K lines of Rust. 9K lines of Metal shaders. LoRA, RLHF, distillation, model merging.

Open source (MIT/Apache-2.0). Thread below.

#rustlang

**Tweet 2 (Metal kernels):**

Custom Metal compute shaders are the core:

- Fused RMSNorm+LoRA (1 dispatch vs 5+)
- Fused cross-entropy (no full vocab materialization)
- FlashAttention (fwd+bwd)
- Fused SwiGLU, RoPE, distillation
- Grouped GEMM for MoE
- FP8 training

Auto-tuner profiles tile sizes per Apple GPU family (M1-M4 Ultra).

**Tweet 3 (Training methods):**

Full training pipeline:
- LoRA / QLoRA with custom autograd (~6x memory savings per adapter layer)
- RLHF: DPO, GRPO, DAPO, GSPO, KTO, SimPO, ORPO, PPO
- Knowledge distillation: TAID + reasoning-aware
- Model merging: TIES, DARE, Model Stock, +more
- 8-bit Adam, schedule-free optimizers, sequence packing

**Tweet 4 (Models + CTA):**

Supports 12+ architectures: Llama 3/4, Qwen 2/3/MoE, DeepSeek, Mistral, Gemma, Phi, Granite, Cohere, Nemotron-H, plus vision (Pixtral, MLlama) and audio (Whisper).

Built on @oxideai's mlx-rs bindings to Apple's MLX.

GitHub: [link]

#rustlang #machinelearning

---

## Posting Notes

**Reddit:**
- Post to r/rust first (primary audience for Rust projects)
- Cross-post to r/LocalLLaMA (Apple Silicon fine-tuning audience) and r/MachineLearning (broader ML)
- Best time: Tuesday-Thursday, 9-11 AM ET
- Respond to comments within the first 2 hours
- If posting to r/MachineLearning, use the [P] tag for project posts

**X/Twitter:**
- Post thread Tuesday-Wednesday, 7-10 AM ET
- Add a reply with a screenshot/terminal output showing a training run if possible
- Pin the thread
- Engage with replies same-day
