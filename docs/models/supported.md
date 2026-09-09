# Supported Models

Model-family support status for inference, embeddings, LoRA/QLoRA training, and direct architecture modules in PMetal.

PMetal supports a wide range of model architectures. Models are loaded from HuggingFace Hub or local safetensors with automatic architecture detection.

## Inference Support

All causal language models below work with the CLI (`pmetal infer`), TUI, GUI, and SDK.

| Family | Architecture | Variants | `model_type` values |
|--------|-------------|----------|-------------------|
| Llama | `Llama` | 2, 3, 3.1, 3.2, 3.3 | `llama`, `llama3` |
| Llama 4 | `Llama4` | Scout, Maverick | `llama4` |
| Qwen 2 | `Qwen2` | 2, 2.5 | `qwen2`, `qwen2_5` |
| Qwen 3 | `Qwen3` | 3 | `qwen3` |
| Qwen 3 MoE | `Qwen3MoE` | 3-MoE | `qwen3_moe` |
| Qwen 3.5 / 3.6 | `Qwen3Next` | 3.5 (Next), 3.6 | `qwen3_next`, `qwen3_5`, `qwen3_6` |
| DeepSeek | `DeepSeek` | V3, V3.2, V3.2-Speciale | `deepseek`, `deepseek_v3` |
| Mistral | `Mistral` | 7B, Mixtral 8×7B | `mistral`, `mixtral` |
| Gemma | `Gemma` | 2, 3 | `gemma`, `gemma2`, `gemma3` |
| Phi 3 | `Phi` | 3, 3.5 | `phi`, `phi3` |
| Phi 4 | `Phi4` | 4 | `phi4` |
| Cohere | `Cohere` | Command R | `cohere`, `command_r` |
| Granite | `Granite` | 3.0, 3.1, Hybrid MoE | `granite`, `granitehybrid` |
| NemotronH | `NemotronH` | Hybrid (Mamba+Attention) | `nemotron_h` |
| GPT-OSS | `GptOss` | 20B, 120B | `gpt_oss`, `gpt-oss` |
| Gemma 4 | `Gemma4` | 4 | `gemma4`, `gemma4_text` |
| Llama 3.2 Vision | `Mllama` | 11B, 90B | `mllama`, `mllama_text_model` |
| DiffusionGemma | `DiffusionGemma` | dense, MoE | `diffusion_gemma`, `diffusion_gemma_text` |

Gemma 4 MTP assistant checkpoints (`model_type = "gemma4_assistant"`) are supported as
draft assistants via `pmetal infer --draft-model <assistant>`. Greedy and sampling
generation both use exact speculative verification.

Qwen3Next/Qwen3.6 checkpoints with bundled `mtp.*` weights can enable exact
speculative decoding with `pmetal infer --mtp --mtp-draft-tokens 3`. Bundled MTP
supports multiple predictor layers, FP8 target/MTP weights, packed expert offload, and
LoRA-merged Qwen3Next targets. LoRA and packed expert offload are separate modes; fuse the
adapter first if you need both. MTP inference reports draft acceptance, average accepted
draft tokens per verify step, and target bonus/correction token counts.

Custom MTP and draft checkpoint creation is available from the CLI. Use `pmetal train-mtp`
to export Gemma 4 assistant checkpoints or Qwen `mtp.*` predictor checkpoints, then load
Qwen predictors with `pmetal infer --mtp --mtp-model ./qwen-mtp`. Use `pmetal train-draft`
to export DFlash draft checkpoints for the dedicated `pmetal dflash` runtime.

## Embedding / Encoder Models

| Family | Architecture | Variants | `model_type` values |
|--------|-------------|----------|-------------------|
| BERT | `Bert` | BERT, RoBERTa, DistilBERT, XLM-RoBERTa | `bert`, `roberta`, `distilbert`, `xlm-roberta`, `xlm_roberta` |

## LoRA / QLoRA Training Support

| Architecture | LoRA | QLoRA | Notes |
|-------------|------|-------|-------|
| Llama | Yes | Yes | Covers Llama 2–3.3. Gradient checkpointing supported. |
| Llama 4 | Yes | Yes | Scout/Maverick support via `DynamicLoraModel`. |
| Qwen 2 | Yes | Yes | Uses Qwen3 LoRA implementation internally. |
| Qwen 3 | Yes | Yes | Gradient checkpointing supported. |
| Qwen 3 MoE | Yes | Yes | Sparse MoE support. |
| Qwen 3.5 / 3.6 (Next) | Yes | Yes | Hybrid architecture with nested `text_config` and bundled MTP inference. |
| Gemma | Yes | Yes | GeGLU activation, special RMSNorm. |
| Gemma 4 | Yes | Yes | Multimodal-era Gemma text path with MTP assistant inference support. |
| Mistral | Yes | Yes | Sliding window attention support. |
| Phi 3/4 | Yes | Yes | Partial RoPE, fused gate_up projection. |
| DeepSeek | Yes | Yes | V3-family support. |
| Cohere | Yes | Yes | Command R support. |
| Granite | Yes | Yes | Dense and hybrid variants. |
| NemotronH | Yes | Yes | Hybrid architecture support. |
| GPT-OSS | Yes | Yes | MoE variants. |
| DiffusionGemma | Yes | Yes | Block-diffusion; trained via `pmetal train-diffusion` (add `--qlora`). |

## Architecture Modules (Not Yet in Dispatcher)

These have implementations in `pmetal-models` but are not in the `DynamicModel` dispatcher:

| Family | Module | Notes |
|--------|--------|-------|
| Pixtral | `pixtral` | 12B vision-language |
| Qwen2-VL | `qwen2_vl` | 2B, 7B vision-language |
| CLIP | `clip` | ViT-L/14 vision encoder |
| Whisper | `whisper` | Base–Large speech models |
| T5 | `t5` | Encoder-decoder architecture |

These can be used directly via their Rust types (e.g., `pmetal_models::architectures::pixtral::Pixtral`).

## Diffusion Models

| Family | Variants | Status |
|--------|----------|--------|
| Flux | 1-dev, 1-schnell | Dispatcher + pipeline implemented |

## See Also

- [Model Merging](/models/merging/) — Merge strategies
- [Quantization](/models/quantization/) — GGUF quantization
