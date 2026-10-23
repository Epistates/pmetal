# pmetal-models

LLM architecture implementations with dynamic dispatch.

## Overview

This crate provides implementations of popular LLM architectures optimized for Apple Silicon. It includes a dynamic dispatch system that automatically detects and loads models based on their configuration.

It sits between the runtime and the training and serving crates: it builds on `pmetal-runtime`, `pmetal-cache`, `pmetal-data`, `pmetal-gguf`, `pmetal-hub` and `pmetal-metal`, and `pmetal-lora`, `pmetal-trainer`, `pmetal-serve`, the `pmetal` CLI and the Python bindings load and run models through it. The fastest decode paths for Qwen 3 / 3.5, Gemma 4, DeepSeek, GPT-OSS and Llama 4 are standalone engines in `pmetal-runtime`, which the `pmetal` CLI selects where they apply; the architectures here are the ones training, adapters and the general path run on.

## Supported Architectures

### Dispatched Models (via `DynamicModel`)

These are the `ModelArchitecture` variants. Each is detected from `config.json` by an exact lookup of its `model_type`, or failing that of a whole `architectures` class name, and every one but `Flux` loads as a `DynamicModel`. A family not listed is refused with an `UnsupportedModel` error that names the `model_type` and lists these, never run as the nearest one:

| Architecture | `model_type` | Models |
|-------------|--------------|--------|
| `Llama` | `llama`, `llama3` | Llama 2, 3, 3.1, 3.2, 3.3 |
| `Llama4` | `llama4`, `llama4_text` | Llama 4 Scout, Maverick |
| `Mllama` | `mllama` | Llama 3.2 Vision |
| `Qwen2` | `qwen2`, `qwen2_5` | Qwen 2, 2.5 |
| `Qwen3` | `qwen3` | Qwen 3 |
| `Qwen3MoE` | `qwen3_moe` | Qwen 3 MoE |
| `Qwen3Next` | `qwen3_next`, `qwen3_5`, `qwen3_5_moe`, `qwen3_6`, `qwen3_6_moe` | Qwen 3.5, 3.6, 3.8 (hybrid Gated DeltaNet and attention) |
| `Qwen4Exp` | `qwen4_exp` | Qwen3.8-Flash-Next |
| `Gemma` | `gemma`, `gemma2`, `gemma3` | Gemma, Gemma 2, Gemma 3 |
| `Gemma4` | `gemma4`, `gemma4_text`, `gemma4_unified` | Gemma 4, dense and MoE |
| `DiffusionGemma` | `diffusion_gemma` | DiffusionGemma (block-autoregressive diffusion) |
| `Mistral` | `mistral` | Mistral 7B and the other dense Mistral decoders |
| `Phi` | `phi3` | Phi 3, 3.5 |
| `Phi4` | `phi4` | Phi 4 |
| `DeepSeek` | `deepseek_v2`, `deepseek_v3` | DeepSeek V2, V3, R1 |
| `Cohere` | `cohere`, `cohere2`, `command_r` | Command R |
| `Granite` | `granite`, `granitemoe`, `granitemoeshared`, `granitemoehybrid` | Granite 3.x dense and MoE, 4.0 / 4.0-H (Mamba-2 hybrid), 4.1, 4.2 |
| `NemotronH` | `nemotron_h` | Nemotron-H (hybrid Mamba and attention) |
| `GptOss` | `gpt_oss` | GPT-OSS 20B, 120B (MoE) |
| `Bert` | `bert`, `roberta`, `xlm-roberta`, `camembert`, `distilbert`, `modernbert` | BERT, RoBERTa, XLM-RoBERTa, CamemBERT, DistilBERT, ModernBERT (encoders; hidden states for pooling) |
| `Flux` | `flux` | Flux.1 (diffusion): detected, but built with `pipelines::FluxPipeline` rather than `DynamicModel` |

### Pipeline Components (Not Dispatched)

These are loaded by the Flux pipeline rather than by `DynamicModel`:

| Module | Family | Notes |
|--------|--------|-------|
| `clip` | CLIP | Text encoder |
| `t5` | T5 | Text encoder |
| `vae` | VAE (`AutoencoderKL`) | Image encoder and latent decoder |

## Features

- **Dynamic Model Loading**: Auto-detect architecture from `config.json`; safetensors, PyTorch and GGUF checkpoints, MLX-quantized and NVIDIA ModelOpt (NVFP4, FP8) weights
- **Unified Generation API**: `generate`, `generate_cached` and pipelined streaming decode for every `CausalLMModel`
- **Sampling**: temperature, top-k, top-p, min-p, repetition, frequency and presence penalties, logit bias, seeded draws
- **Speculative decoding**: DFlash block-diffusion drafters (both generations), tree verification (`ddtree`), and multi-token-prediction heads (Qwen 3.5's bundled predictor, Gemma 4's assistant)
- **Decision models**: Clef and Clef-flash answer typed questions about a state in one forward pass (`decision`)
- **SSD-offloaded experts**: routed experts of a sparse layer read from packed per-layer files, for MoE models larger than memory
- **ModelThread**: one thread that owns a model and its arrays, for callers (a server) that must not touch them from several threads
- **Diffusion**: DiffusionGemma sampling, and the Flux pipeline with CLIP, T5 and VAE
- **Vision**: Llama 3.2 Vision, Gemma 4 and Qwen 3.5 vision towers
- **Ollama Modelfile** generation and parsing, for exporting a trained model

## Installation

```toml
[dependencies]
pmetal-models = "0.7"
```

## Usage

A generation loop needs only a function from token ids to logits, so it runs without weights.
This toy "model" always predicts the next id:

```rust
use pmetal_models::{GenerationConfig, generate};
use pmetal_runtime::compat::{Array, Exception};

fn main() -> Result<(), Exception> {
    const VOCAB: usize = 8;
    let counting = |input: &Array| -> Result<Array, Exception> {
        let seq = input.dim(1) as usize; // input is [1, seq]
        let mut logits = vec![0.0f32; seq * VOCAB];
        for t in 0..seq {
            logits[t * VOCAB + (t + 1) % VOCAB] = 10.0;
        }
        Ok(Array::from_f32_slice(&logits, &[1, seq as i32, VOCAB as i32]))
    };

    let output = generate(counting, &[0, 1], GenerationConfig::greedy(4))?;
    // The prompt, then the four generated ids.
    assert_eq!(output.token_ids, [0, 1, 2, 3, 4, 5]);
    Ok(())
}
```

With a real model, `DynamicModel::load` detects the architecture from `config.json`. It takes a
**local directory**; resolve HuggingFace ids with `pmetal_hub::download_model` first.

```rust,no_run
use pmetal_runtime::compat::Exception;
use pmetal_models::{DynamicModel, GenerationConfig, generate};

fn run(model_dir: &str, input_tokens: &[u32]) -> Result<Vec<u32>, Exception> {
    let mut model = DynamicModel::load(model_dir)?;

    let config = GenerationConfig::sampling(256, 0.7)
        .with_top_k(40)
        .with_top_p(0.95);

    // `token_ids` is the prompt followed by what was generated.
    let output = generate(|input| model.forward(input, None), input_tokens, config)?;
    Ok(output.token_ids)
}
```

## Loading Checkpoints

`DynamicModel::load` reads safetensors (single file or sharded), PyTorch pickles
(`pytorch_model.bin`, through the restricted reader in `pmetal_hub::torch`) and GGUF, and
unpacks quantized checkpoints to dense tensors as it reads them: MLX affine and float modes
(mxfp4, nvfp4, mxfp8, per-module widths included), NVIDIA ModelOpt and block-scaled FP8.
For inference, `DynamicModel::load_with_options` with `keep_quantized` instead leaves an
MLX-quantized checkpoint's projections, embedding tables and stacked experts packed on their
layers, where MLX's quantized (gather) matmul reads them: a quarter of the memory and of the
bytes a decode step reads. An architecture that cannot (Mllama, Qwen4-Exp, the encoder and
diffusion models) unpacks and logs what it unpacked.

A load must account for the whole model. It fails, naming the tensors and parameters, when a
checkpoint tensor reaches no parameter or when a parameter is never filled; the only
exceptions are named in `ModelArchitecture::load_policy` with the reason each is safe
(buffers the runtime computes itself, a tied head's spare copy, Cohere's bias-free
LayerNorm, Llama 4's parameter-free QK norm). A checkpoint quantized with a method pmetal
cannot read (GPTQ, AWQ, bitsandbytes and similar) is reported as that, not as noise.

## Architecture Detection

The `DynamicModel` automatically detects model architecture:

```rust,no_run
use pmetal_runtime::compat::Exception;
use pmetal_models::ModelArchitecture;

fn detect(model_dir: &str) -> Result<ModelArchitecture, Exception> {
    // Reads config.json: Llama, Qwen3, Mistral, Gemma, Phi, and so on.
    ModelArchitecture::detect(model_dir)
}
```

## Generation Configuration

`GenerationConfig::default()` samples; `greedy(n)`, `sampling(n, temperature)`, `creative(n)`,
`precise(n)` and the Qwen 3 presets start from other defaults, and
`GenerationConfig::from_model_dir` reads the model's own `generation_config.json`.

| Field | Description | Default |
|-------|-------------|---------|
| `max_new_tokens` | Maximum tokens to generate | 256 |
| `temperature` | Sampling temperature (0 = greedy) | 0.7 |
| `top_k` | Top-k sampling (0 = disabled) | 40 |
| `top_p` | Nucleus sampling threshold (1.0 = disabled) | 0.95 |
| `min_p` | Drop tokens below `min_p` times the top token's probability (0 = disabled) | 0.05 |
| `repetition_penalty` | Penalty for repeated tokens (1.0 = disabled) | 1.1 |
| `frequency_penalty`, `presence_penalty` | OpenAI-style penalties (0 = disabled) | 0.0 |
| `logit_bias` | `(token, bias)` pairs added to the logits | none |
| `stop_tokens` | Tokens that stop generation | none |
| `seed` | Seed for reproducible draws | unseeded |

## Modules

| Module | Description |
|--------|-------------|
| `architectures/` | Model implementations (Llama, Qwen, etc.) |
| `decision` | Decision models (Clef): record encoding, the joint schema head, `/v1/systemone` |
| `dispatcher` | Dynamic model loading and dispatch |
| `generation` | Token generation with sampling |
| `loader` | Reading checkpoints, unpacking quantization, assigning and auditing weights |
| `pooling` | Sentence-embedding pooling over hidden states |
| `weight_format` | GGUF weight loading and tensor-name mapping |
| `sampling/` | Diffusion sampling strategies |
| `traits` | `CausalLMModel` and `ModelConfig` |
| `dflash_*`, `ddtree`, `mtp`, `speculative` | Speculative decoding: DFlash drafters, tree verification, MTP heads, the shared rejection-sampling correction |
| `expert_layout`, `expert_io`, `expert_offload` | SSD-offloaded routed experts |
| `model_thread` | A thread that owns a model and its arrays |
| `rl_generation` | Batched group sampling for GRPO-style training |
| `shard` | Splitting a forward pass into stages for pipeline inference |
| `ollama` | Ollama Modelfile generation and parsing |
| `pipelines` | The Flux diffusion pipeline |

## Feature flags

| Feature | Default | Description |
|---------|---------|-------------|
| `ane` | yes | Apple Neural Engine inference (`generate_cached_ane_streaming`, the ANE worker thread); turns on `pmetal-metal/ane` |

## Requirements

- Rust 1.95 or later
- macOS 14 or later on Apple Silicon

## License

MIT OR Apache-2.0
