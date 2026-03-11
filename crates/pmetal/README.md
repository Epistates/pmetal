# pmetal

**Powdered Metal** — High-performance LLM fine-tuning framework for Apple Silicon, written in Rust.

[![Crates.io](https://img.shields.io/crates/v/pmetal.svg)](https://crates.io/crates/pmetal)
[![docs.rs](https://docs.rs/pmetal/badge.svg)](https://docs.rs/pmetal)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](../../LICENSE)

This is the umbrella crate that re-exports all PMetal sub-crates behind feature flags. Add a single dependency to access the full framework:

```toml
[dependencies]
pmetal = "0.3"                                      # default features
pmetal = { version = "0.3", features = ["full"] }   # everything
```

## Quick Start

### Fine-tune a model

```rust,no_run
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let result = pmetal::easy::finetune("Qwen/Qwen3-0.6B", "data.jsonl")
        .lora(16, 32.0)
        .epochs(3)
        .learning_rate(2e-4)
        .output("./output")
        .run()
        .await?;

    println!("Final loss: {:.4}", result.final_loss);
    Ok(())
}
```

### Run inference

```rust,no_run
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let result = pmetal::easy::infer("Qwen/Qwen3-0.6B")
        .lora("./output/lora_weights.safetensors")
        .temperature(0.7)
        .max_tokens(256)
        .generate("What is 2+2?")
        .await?;

    println!("{}", result.text);
    Ok(())
}
```

### Query device info

```rust,no_run
fn main() {
    println!("{}", pmetal::version::device_info());
}
```

## Feature Flags

| Feature | Crate | Default | Description |
|---------|-------|---------|-------------|
| `core` | `pmetal-core` | yes | Foundation types, configs, traits |
| `gguf` | `pmetal-gguf` | yes | GGUF format with imatrix quantization |
| `metal` | `pmetal-metal` | yes | Custom Metal GPU kernels + ANE runtime |
| `hub` | `pmetal-hub` | yes | HuggingFace Hub integration |
| `mlx` | `pmetal-mlx` | yes | MLX backend (KV cache, RoPE, ops) |
| `models` | `pmetal-models` | yes | LLM architectures (Llama, Qwen, DeepSeek, ...) |
| `lora` | `pmetal-lora` | yes | LoRA/QLoRA training |
| `trainer` | `pmetal-trainer` | yes | Training loops (SFT, DPO, GRPO, DAPO) |
| `easy` | all training/inference | yes | High-level builder API |
| `ane` | ANE integration | yes | Apple Neural Engine direct programming |
| `data` | `pmetal-data` | no | Dataset loading and preprocessing |
| `distill` | `pmetal-distill` | no | Knowledge distillation (cross-vocab) |
| `merge` | `pmetal-merge` | no | Model merging (SLERP, TIES, DARE, ModelStock) |
| `vocoder` | `pmetal-vocoder` | no | BigVGAN neural vocoder |
| `distributed` | `pmetal-distributed` | no | Distributed training (mDNS, Ring All-Reduce) |
| `mhc` | `pmetal-mhc` | no | Manifold-Constrained Hyper-Connections |
| `lora-metal-fused` | — | no | Fused Metal kernels for ~2x LoRA speedup |
| `full` | all of the above | no | Everything |

## Hardware Support

PMetal auto-detects Apple Silicon capabilities and tunes kernel parameters per device:

- **M1–M5** families (Base, Pro, Max, Ultra)
- **NAX** (Neural Accelerators in GPU) on M5/Apple10
- **ANE** (Apple Neural Engine) with CPU RMSNorm workaround for fp16 stability
- **UltraFusion** multi-die topology detection
- **Tier-based tuning**: FlashAttention block sizes, GEMM tile sizes, threadgroup sizes, batch multipliers

## Examples

```sh
# Device info
cargo run -p pmetal --example device_info

# Fine-tuning (easy API)
cargo run -p pmetal --example finetune_easy --features easy -- \
    --model Qwen/Qwen3-0.6B --dataset data.jsonl

# Inference (easy API)
cargo run -p pmetal --example inference_easy --features easy -- \
    --model Qwen/Qwen3-0.6B --prompt "What is 2+2?"

# Manual fine-tuning (lower-level control)
cargo run -p pmetal --example finetune_manual --features data,lora,trainer
```

## Re-exports

All sub-crates are available as modules:

```rust
use pmetal::core;       // pmetal-core
use pmetal::metal;      // pmetal-metal
use pmetal::mlx;        // pmetal-mlx
use pmetal::models;     // pmetal-models
use pmetal::lora;       // pmetal-lora
use pmetal::trainer;    // pmetal-trainer
use pmetal::hub;        // pmetal-hub
use pmetal::gguf;       // pmetal-gguf
use pmetal::prelude::*; // commonly used types from all crates
```

## License

Licensed under either of [MIT](../../LICENSE-MIT) or [Apache-2.0](../../LICENSE-APACHE).
