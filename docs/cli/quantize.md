# pmetal quantize

Quantize a model to GGUF (26 methods) or to MLX safetensors.

The default output is a GGUF file for efficient inference. The default method, `dynamic`, mixes precisions per tensor and is guided by an importance matrix when `--imatrix` gives one, or when `--calibration-dataset` measures one in the same run. `--kl-calibrate` picks each tensor's type by measuring the KL divergence the quantization costs. `--format mlx` writes MLX-format safetensors instead, with per-tensor bit allocation.

A LoRA adapter can be fused in before quantizing with `--lora` (GGUF only; for MLX, run [`pmetal fuse`](/cli/fuse/) first).

## Usage

```bash
pmetal quantize \
  --model <MODEL> \
  --output <OUTPUT> \
  [--method <METHOD>] \
  [OPTIONS]
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `-m`, `--model` | required | Source model: a local path or Hub ID (safetensors) |
| `-o`, `--output` | required | Output GGUF file path, or, with `--format mlx`, an output directory |
| `--method` | `dynamic` | Quantization method (see Methods) |
| `--imatrix` | none | Importance matrix file (an `imatrix.dat` or an imatrix GGUF, see [`pmetal imatrix`](/cli/imatrix/)) that weights the quantization error of the tensors |
| `--calibration-dataset` | none | Measure the importance matrix on this data first: a text file, or a JSONL or Parquet dataset (local or a Hub id). Not with `--imatrix` |
| `--calibration-ctx-size` | `512` | Tokens per calibration chunk |
| `--calibration-chunks` | `0` | Calibration chunks to run (0 runs all the data makes) |
| `--lora` | none | LoRA adapter to fuse before quantizing (GGUF only) |
| `--kl-calibrate` | `false` | Choose each tensor's quantization type by KL-divergence calibration |
| `--target-bpw` | none | Target average bits per weight, for KL calibration (e.g. `4.5`) and for MLX bit allocation |
| `--kl-threshold` | `0.01` | Quality-loss threshold for KL calibration |
| `--format` | `gguf` | `gguf` or `mlx` |
| `--bits` | `4` | Default bit-width of MLX quantization (3, 4, 5, 6 or 8) |
| `--group-size` | `64` | Group size of MLX quantization |

## Methods

These are the values `--method` accepts.

| Method | Description |
|--------|-------------|
| `dynamic` | Importance-matrix-guided mixed precision (default; pair with `--imatrix`) |
| `q8_0` | 8-bit integer, near-lossless |
| `q8_1` | 8-bit integer with dot-product sum helper |
| `q6_k` | 6-bit K-quant, high quality |
| `q5_k_m` | 5-bit K-quant medium |
| `q5_k_s` | 5-bit K-quant small |
| `q5_0` / `q5_1` | Legacy 5-bit symmetric / affine |
| `q4_k_m` | 4-bit K-quant medium (recommended 4-bit) |
| `q4_k_s` | 4-bit K-quant small |
| `q4_0` / `q4_1` | Legacy 4-bit symmetric / affine |
| `q3_k_l` / `q3_k_m` / `q3_k_s` | 3-bit K-quant large / medium / small |
| `q2_k` | 2-bit K-quant, lowest quality |
| `q1_0` | 1-bit sign |
| `tq1_0` / `tq2_0` | Ternary 1.69-bit / 2.06-bit |
| `mxfp4` / `nvfp4` | Block-floating 4-bit |
| `bf16` / `f16` / `f32` | Dense export |

The output head is written as `q6_k` for the K-quant, legacy 4- and 5-bit and 4-bit non-linear methods (a model that ties its head to the embedding has no `output.weight`, and its token embedding is then written as a head), and a token embedding beside a head of its own is stored in the type of the method, and a tensor whose rows are not a whole number of blocks falls back to a smaller block format (`q4_0` for `q2_k` and `q3_k`, `iq4_nl` for `iq4_xs`, `q5_0`, `q5_1` or `q8_0`) as the format requires. An importance matrix applies to GGUF output; `--imatrix` and `--calibration-dataset` are refused with `--format mlx`.

## Architectures

A GGUF file names and lays out its tensors per architecture, and an engine loads whatever it finds under the names it looks for. `pmetal quantize` therefore writes only the architectures whose layout it knows, and refuses any other before it reads the model. Llama 3.2, Qwen3 and Gemma 3 are checked tensor for tensor against the reference converter's files; the others use the same tables and have not been compared against real checkpoints.

| Hugging Face architecture | GGUF architecture |
|---------------------------|-------------------|
| `LlamaForCausalLM`, `MistralForCausalLM`, `MixtralForCausalLM` | `llama` |
| `Qwen2ForCausalLM`, `Qwen2MoeForCausalLM` | `qwen2`, `qwen2moe` |
| `Qwen3ForCausalLM`, `Qwen3MoeForCausalLM` | `qwen3`, `qwen3moe` |
| `GemmaForCausalLM`, `Gemma2ForCausalLM`, `Gemma3ForCausalLM`, `Gemma3ForConditionalGeneration` (the 4B, 12B and 27B models: the language model only, the vision tower and projector are left out) | `gemma`, `gemma2`, `gemma3` |
| `Gemma4ForCausalLM`, `Gemma4ForConditionalGeneration`, `Gemma4UnifiedForConditionalGeneration` (the dense text models; the towers of a multimodal checkpoint are left out) | `gemma4` |

The file holds what an engine expects, which is not always what the checkpoint holds: the query and key rows of a `llama` model are reordered for its rotary pairing, Llama 3.x's rope scaling is a `rope_freqs` tensor, Gemma 1 to 3's norms have 1 added (Gemma 4's are stored as they are), a tied output head is not written beside the embedding, and norm weights and biases are f32 whatever `--method` is. The tokenizer comes with the file: byte-level BPE for Qwen and Llama 3, SentencePiece pieces and scores (from `tokenizer.model`) for Gemma, with the special tokens, `add_bos_token`, the pre-tokenizer and the chat template.

An `f16` export saturates a weight beyond the range of f16 and says so; `bf16` holds every weight of a model published in bf16 without loss.

## Examples

```bash
# 4-bit quantization
pmetal quantize \
  --model ./output \
  --output model.gguf --method q4_k_m

# With an importance matrix
pmetal quantize \
  --model ./output \
  --output model.gguf --method q4_k_m \
  --imatrix imatrix.gguf

# Measure the importance matrix on calibration data in the same run
pmetal quantize \
  --model ./output \
  --output model.gguf --method iq4_xs \
  --calibration-dataset calibration.txt

# Dynamic per-layer quantization
pmetal quantize \
  --model ./output \
  --output model.gguf --method dynamic

# KL-calibrated quantization (per-tensor type selection)
pmetal quantize \
  --model ./output \
  --output model.gguf \
  --kl-calibrate --target-bpw 4.5

# MLX safetensors at 4 bits
pmetal quantize \
  --model Qwen/Qwen3-0.6B \
  --output ./qwen3-0.6b-mlx-4bit \
  --format mlx --bits 4
```

## See Also

- [pmetal imatrix](/cli/imatrix/): Measure an importance matrix
- [Quantization](/models/quantization/): Detailed quantization guide
- [pmetal fuse](/cli/fuse/): Fuse a LoRA adapter before quantizing
- [pmetal ollama](/cli/ollama/): Generate a Modelfile for a GGUF
