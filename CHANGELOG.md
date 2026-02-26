# Changelog

All notable changes to PMetal will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-26

Initial public release.

### Core Framework

- **pmetal-core**: Foundation types, configuration system, and shared traits for the workspace
- **pmetal-cli**: Command-line interface with `train`, `infer`, and `bench` subcommands

### Model Support

- **pmetal-models**: Dynamic architecture loading with support for:
  - Llama (2, 3, 3.1, 3.2, 3.3, 4)
  - Qwen (2, 2.5, 3, 3-MoE)
  - DeepSeek (V3, V3.2, V3.2-Speciale)
  - Mistral (7B, 8x7B)
  - Gemma (2, 3), Phi (3, 4), Granite (3.0, 3.1), Cohere (Command R), GPT-OSS, Nemotron-H
  - Vision: Pixtral 12B, Qwen2-VL, MLlama 3.2-Vision

### Training

- **pmetal-trainer**: SFT, DPO, and GRPO training loops with learning rate schedulers and gradient checkpointing
- **pmetal-lora**: LoRA and QLoRA with configurable rank, alpha, and target modules
- **pmetal-data**: Dataset loading for ShareGPT, Alpaca, Messages, and raw text formats with sequence packing (99.7% efficiency)
- **pmetal-distill**: Knowledge distillation with KL divergence, Jensen-Shannon, soft cross-entropy, hidden state alignment, and offline logit caching

### GPU Acceleration

- **pmetal-metal**: Custom Metal compute kernels:
  - FlashAttention with O(n) memory
  - Fused LoRA forward pass
  - Fused cross-entropy (Unsloth-style chunked loss)
  - Fused RoPE
  - Fused sampler with JIT compilation
  - Fused DoRA kernels

### Model Operations

- **pmetal-merge**: Model merging via Linear, SLERP, TIES, DARE, DELLA, NearSwap, and Model Stock methods
- **pmetal-gguf**: GGUF format reading, writing, dequantization, and imatrix quantization
- **pmetal-hub**: HuggingFace Hub downloading, caching, and upload support

### Experimental

- **pmetal-mhc**: Manifold-Constrained Hyper-Connections (Sinkhorn-Knopp doubly stochastic projections) with Metal GPU acceleration
- **pmetal-distributed**: Peer-to-peer distributed training with mDNS auto-discovery, ring all-reduce, and gradient compression
- **pmetal-vocoder**: BigVGAN neural vocoder for text-to-speech synthesis
- **pmetal-mlx**: MLX backend integration with KV cache management, quantization, speculative decoding, and NEFTune

### Infrastructure

- Rust edition 2024, minimum supported Rust version 1.85
- Continuous fuzzing for GGUF reader via `cargo-fuzz`
- CI with clippy, fmt, test, and fuzz workflows
- Dual licensed under MIT and Apache-2.0
