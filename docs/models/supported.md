# Supported Models

Model-family support status for inference, embeddings, LoRA/QLoRA training, and direct architecture modules in PMetal.

PMetal supports a wide range of model architectures. Models are loaded from HuggingFace Hub or local safetensors with automatic architecture detection.

## Language Models

Loaded from the Hugging Face Hub or a local directory; the architecture is detected from `config.json`. All of them work with the CLI (`pmetal infer`), TUI, GUI, server and SDK.

| Family | Architecture | Variants | `model_type` values |
|--------|-------------|----------|-------------------|
| Llama | `Llama` | 2, 3, 3.1, 3.2, 3.3 | `llama`, `llama3` |
| Llama 4 | `Llama4` | Scout, Maverick | `llama4`, `llama4_text` |
| Llama 3.2 Vision | `Mllama` | 11B, 90B (image input) | `mllama`, `mllama_text_model` |
| Qwen 2 | `Qwen2` | 2, 2.5 | `qwen2`, `qwen2_5` |
| Qwen 3 | `Qwen3` | dense | `qwen3` |
| Qwen 3 MoE | `Qwen3MoE` | e.g. Qwen3-30B-A3B | `qwen3_moe` |
| Qwen 3.5 / 3.6 / 3.8 | `Qwen3Next` | hybrid Gated DeltaNet, dense and MoE; image and video input | `qwen3_next`, `qwen3_5`, `qwen3_5_moe`, `qwen3_6`, `qwen3_6_moe` (and their `_text` forms) |
| Qwen 3.8 Flash Next | `Qwen4Exp` | text tower | `qwen4_exp`, `qwen4_exp_text` |
| DeepSeek | `DeepSeek` | V2, V3, R1 (dense multi-head latent attention) | `deepseek_v2`, `deepseek_v3`, `deepseek2` |
| Mistral | `Mistral` | 7B and the other dense Mistral decoders | `mistral` |
| Gemma | `Gemma` | 1, 2, 3 | `gemma`, `gemma2`, `gemma3`, `gemma3_text` |
| Gemma 4 | `Gemma4` | dense and MoE | `gemma4`, `gemma4_text` |
| DiffusionGemma | `DiffusionGemma` | dense, MoE | `diffusion_gemma`, `diffusion_gemma_text` |
| Phi 3 | `Phi` | 3, 3.5 | `phi3` |
| Phi 4 | `Phi4` | 4 | `phi4` |
| Cohere | `Cohere` | Command R, R+, R7B, Command A | `cohere`, `cohere2`, `command_r` |
| Granite | `Granite` | 3.x (including MoE), 4.0, 4.0-H | `granite`, `granitemoe`, `granitemoeshared`, `granitemoehybrid` |
| Nemotron-H, Nemotron 3 | `NemotronH` | hybrid Mamba-2 and attention (Nemotron 3 Nano and Super, Nemotron-H 4B to 12B) | `nemotron_h` |
| GPT-OSS | `GptOss` | 20B, 120B | `gpt_oss`, `gpt-oss` |

The Clef decision models (`pmetal decide`, `POST /v1/systemone`) run on Qwen3.5-family backbones.

### Not supported yet

A `model_type` maps to exactly one architecture above or to none, so a family that is not listed is refused by name, with an error that names the `model_type`, says it is not supported yet and lists the supported families. It is never run as the nearest family it resembles. The checkpoints that look closest to a supported family, and are declined:

| Family | `model_type` values | Why |
|--------|--------------------|-----|
| Qwen3-VL, Qwen2-VL, Qwen2.5-VL, Qwen Omni | `qwen3_vl`, `qwen3_vl_moe`, `qwen2_vl`, `qwen2_5_vl`, `qwen3_omni_moe` | vision and audio towers are not implemented (the Qwen3.5 family runs its own image and video input) |
| Qwen2-MoE | `qwen2_moe` | the shared-expert layout differs from Qwen3-MoE |
| Mistral3, Ministral, Mistral 4, Pixtral, LLaVA | `mistral3`, `ministral3`, `mistral4`, `pixtral`, `llava` | a vision wrapper around the decoder |
| Mixtral | `mixtral` | the sparse expert layers are not implemented |
| DeepSeek V3.2 | `deepseek_v32`, and `deepseek_v3` with `index_topk` | the sparse-attention indexer is not implemented |
| DeepSeek V4, V4.1 | `deepseek_v4`, `deepseek_v41` | not implemented |
| DeepSeek V1 | `deepseek` | plain attention, not multi-head latent attention |
| GLM | `glm4`, `glm4_moe`, `glm4_moe_lite`, `glm_moe_dsa`, `glm5_next` | not implemented |
| Kimi | `kimi_k2`, `kimi_k25`, `kimi_k3`, `kimi_linear`, `kimi_vl` | not implemented |
| MiniMax | `minimax_m2`, `minimax_m3` | not implemented |
| Gemma 3n, PaliGemma, embedding Gemma | `gemma3n`, `paligemma`, `embedding_gemma2` | different architectures |
| Phi-1, Phi-2, Phi-3.5-MoE, Phi multimodal | `phi`, `phimoe`, `phi4mm` | different architectures |
| Granite variants | `granite_swa`, `granite_switch`, `granite_speech`, `granite4_vision` | sliding-window, speech and vision variants |
| Nemotron-H Omni and Puzzle, Nemotron | `nemotron_h_omni`, `nemotron_h_puzzle`, `nemotron` | different architectures |

The same is true of the rest of the Hub's trending families (LFM2, MiMo, Hunyuan, Ling, OLMo, SmolLM3, Falcon and others): a `model_type` outside the tables above is refused, not guessed.

Quantized checkpoints load packed: MLX-quantized checkpoints (affine, mxfp4, nvfp4, mxfp8), NVIDIA ModelOpt NVFP4 and FP8, and gpt-oss MXFP4.

Gemma 4 MTP assistant checkpoints (`model_type = "gemma4_assistant"`) are supported as
draft assistants via `pmetal infer --draft-model <assistant>`. Greedy and sampling
generation both use exact speculative verification, and the standard sampling controls
(`--temperature`, `--top-k`, `--top-p`, `--min-p`, the penalties) carry through it.

Qwen3Next/Qwen3.6 checkpoints with bundled `mtp.*` weights can enable exact
speculative decoding with `pmetal infer --mtp --mtp-draft-tokens 3`. Bundled MTP
supports multiple predictor layers, FP8 target/MTP weights, packed expert offload, and
LoRA-merged Qwen3Next targets. LoRA and packed expert offload are separate modes; fuse the
adapter first if you need both. MTP inference reports draft acceptance, average accepted
draft tokens per verify step, and target bonus/correction token counts.

Both MTP heads draft the tokens after the target's own next token: the head reads the
target hidden state that predicted that token beside the token's embedding, as the heads
are trained to. One target forward then checks the drafts and emits the accepted ones plus
one token of the target's, so greedy output is the target's own greedy decode.

Custom MTP and draft checkpoint creation is available from the CLI. Use `pmetal train-mtp`
to export Gemma 4 assistant checkpoints or Qwen `mtp.*` predictor checkpoints, then load
Qwen predictors with `pmetal infer --mtp --mtp-model ./qwen-mtp`. Use `pmetal train-draft`
to export DFlash draft checkpoints for the dedicated `pmetal dflash` runtime.

## Embedding / Encoder Models

| Family | Architecture | Variants | `model_type` values |
|--------|-------------|----------|-------------------|
| BERT | `Bert` | BERT, RoBERTa, XLM-RoBERTa, CamemBERT, DistilBERT, ModernBERT | `bert`, `roberta`, `xlm-roberta`, `xlm_roberta`, `camembert`, `distilbert`, `modernbert` |

## LoRA / QLoRA Training Support

LoRA and QLoRA apply to every architecture the dispatcher loads (the table above). QLoRA (`--quantization nf4|fp4|int8`) packs the same model LoRA trains and attaches the adapters to it, with sequence packing, gradient checkpointing and the fused optimizer. The LM head, MoE routers and routed experts stay as loaded; LoRA adapts the routed experts of Qwen3-MoE, DeepSeek and Nemotron-H, one adapter per expert. Adapters are written as a PEFT directory (`adapter_model.safetensors` and `adapter_config.json`).

| Model kind | Command |
|------------|---------|
| Decoder language models | `pmetal train` (and `preference`, `grpo`, `rlkd`, `distill`) |
| DiffusionGemma | `pmetal train-diffusion` (`--quantization nf4` for QLoRA) |
| BERT-family encoders | `pmetal embed-train` (`--lora`) |

## GGUF Checkpoints

A `.gguf` file (or a directory whose weights are GGUF) loads for `pmetal infer`, `pmetal eval` and `pmetal serve` with its configuration and tokenizer read from the file. The architectures that load from GGUF are the dense decoders Llama, Qwen 2, Qwen 3, Mistral, Gemma 1, 2 and 3, and Phi 3 / Phi 4; load any other family from its safetensors weights. That includes the files `pmetal quantize` writes. Quantized tensors are dequantized on load (see [Quantization](/models/quantization/)).

For Gemma the file's conventions are undone on load: the norm weights, which a GGUF stores with the 1 that Gemma's norm adds already applied, the tensor names of the four-norm block (Gemma 2 and 3), the sliding window, the local and global rope bases and the two soft caps. A Gemma 4 GGUF is not loaded yet.

## Diffusion Models

| Family | Variants | Status |
|--------|----------|--------|
| Flux | 1-dev, 1-schnell | Pipeline over tokenized prompts (checked against `diffusers`); no tokenizer or CLI command yet |

`pipelines::FluxPipeline` loads a `diffusers`-layout FLUX.1 directory (transformer, CLIP-L and T5-XXL text encoders, VAE, scheduler) from `pmetal-models`, strictly.

## See Also

- [Model Merging](/models/merging/): Merge strategies
- [Quantization](/models/quantization/): GGUF quantization
