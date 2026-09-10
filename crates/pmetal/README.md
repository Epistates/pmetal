# pmetal

**Powdered Metal** — High-performance LLM fine-tuning framework for Apple Silicon, written in Rust.

[![Crates.io](https://img.shields.io/crates/v/pmetal.svg)](https://crates.io/crates/pmetal)
[![docs.rs](https://docs.rs/pmetal/badge.svg)](https://docs.rs/pmetal)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](../../LICENSE)

This is the umbrella crate that re-exports all PMetal sub-crates behind feature flags. Add a single dependency to access the full framework:

```toml
[dependencies]
pmetal = "0.6"                                      # default features
pmetal = { version = "0.6", features = ["full"] }   # every sub-crate
```

## Quick Start

### Fine-tune a model

`orchestrator::run_training` is the one-call entry point the CLI itself uses. It resolves the
model, builds the dataset, picks the training path from `TrainingJobConfig`, and writes the
adapter.

```rust,no_run
use pmetal::trainer::orchestrator::{TrainingJobConfig, run_training};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = TrainingJobConfig {
        model_id: "Qwen/Qwen3-0.6B".to_string(),
        dataset: "data.jsonl".to_string(),
        output_dir: "./output".to_string(),
        ..Default::default()
    };

    // No phase callback, no training callbacks.
    let result = run_training(config, None, Vec::new()).await?;
    println!("Final loss: {:.4}", result.final_loss);
    Ok(())
}
```

For step-by-step control over the loop, see
[`examples/finetune_manual.rs`](examples/finetune_manual.rs).

### Run inference

Note that `pmetal::prelude::*` brings in `pmetal_core::Result<T>`, a one-parameter alias. Spell the
two-parameter form out when you want `std`'s.

```rust,no_run
use pmetal::models::generate;
use pmetal::prelude::*;

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let model_dir = pmetal::hub::resolve_model_path("Qwen/Qwen3-0.6B", None, None).await?;

    let mut model = DynamicModel::load(&model_dir)?;
    let tokenizer = Tokenizer::from_model_dir(&model_dir)?;

    let input_ids = tokenizer.encode_with_special_tokens("What is 2+2?")?;
    let config = GenerationConfig::sampling(256, 0.7);

    let output = generate(|input| model.forward(input, None), &input_ids, config)?;
    println!("{}", tokenizer.decode(&output.token_ids[input_ids.len()..])?);
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

The default set is `cli`, `dashboard`, `trainer`, `lora`, `merge`, `ane` and `distributed`; the
rest come in transitively.

| Feature | Crate | Default | Description |
|---------|-------|---------|-------------|
| `cli` | — | yes | The `pmetal` binary and its CLI-only dependencies |
| `core` | `pmetal-core` | yes* | Foundation types, configs, traits |
| `gguf` | `pmetal-gguf` | yes* | GGUF format with imatrix quantization |
| `metal` | `pmetal-metal` | yes* | Custom Metal GPU kernels + ANE runtime |
| `hub` | `pmetal-hub` | yes* | HuggingFace Hub integration |
| `mlx` | `pmetal-mlx` | yes* | MLX backend (KV cache, RoPE, ops) |
| `models` | `pmetal-models` | yes* | LLM architectures (Llama, Qwen, DeepSeek, ...) |
| `lora` | `pmetal-lora` | yes | LoRA/QLoRA training |
| `trainer` | `pmetal-trainer` | yes | Training loops (SFT, DPO, SimPO, ORPO, KTO, GRPO, DAPO, RLKD, Embedding, PPO, GSPO, Online DPO, Diffusion) — enables `data` + `distill` |
| `data` | `pmetal-data` | yes* | Dataset loading and preprocessing (*via `cli` and `trainer`) |
| `distill` | `pmetal-distill` | yes* | Knowledge distillation incl. TAID (*via `trainer`) |
| `merge` | `pmetal-merge` | yes | Model merging (15 strategies: Linear, SLERP, TIES, DARE, DELLA, ModelStock, etc.) |
| `distributed` | `pmetal-distributed` | yes | Distributed training (mDNS, Ring All-Reduce) and the `cluster` subcommand |
| `ane` | `pmetal-metal` | yes | Apple Neural Engine direct programming |
| `dashboard` | — | yes | TUI control center |
| `native-only` | `pmetal-bridge` | no | Bridge-only build with no mlx-rs/mlx-sys |
| `vocoder` | `pmetal-vocoder` | no | BigVGAN neural vocoder |
| `mhc` | `pmetal-mhc` | no | Manifold-Constrained Hyper-Connections |
| `serve` | `pmetal-serve` | no | OpenAI-compatible inference server (`pmetal serve`) |
| `mcp` | `pmetal-mcp` | no | MCP server for Claude Desktop (`pmetal mcp`) |
| `lora-metal-fused` | — | no | Fused Metal kernels for ~2x LoRA speedup |
| `full` | sub-crates | no | Every sub-crate feature (not `cli`, `serve` or `mcp`) |

`serve` and `mcp` stay opt-in so library consumers don't inherit axum and rmcp. The prebuilt
release binary and the Homebrew formula both enable them.

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

# Inference
cargo run -p pmetal --example inference -- \
    --model ./path/to/model --prompt "What is 2+2?"

# Fine-tuning with direct sub-crate orchestration
cargo run -p pmetal --example finetune_manual -- \
    --model ./path/to/model --dataset data.jsonl

# Fine-tuning on the Apple Neural Engine
cargo run -p pmetal --example finetune_ane -- \
    --model ./path/to/model --dataset data.jsonl
```

## Re-exports

All sub-crates are available as modules:

```rust
use pmetal::core; // pmetal-core
use pmetal::gguf; // pmetal-gguf
use pmetal::hub; // pmetal-hub
use pmetal::lora; // pmetal-lora
use pmetal::metal; // pmetal-metal
use pmetal::mlx; // pmetal-mlx
use pmetal::models; // pmetal-models
use pmetal::prelude::*; // commonly used types from all crates
use pmetal::trainer; // pmetal-trainer
```

## License

Licensed under either of [MIT](../../LICENSE-MIT) or [Apache-2.0](../../LICENSE-APACHE).
