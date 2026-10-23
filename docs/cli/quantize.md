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

The output head and token embedding are written as `q6_k` for the K-quant and 4-bit non-linear methods, and a tensor whose rows are not a whole number of blocks falls back to a smaller block format (`iq4_nl`, `q5_0`, `q5_1` or `q8_0`) as the format requires. An importance matrix applies to GGUF output; `--imatrix` and `--calibration-dataset` are refused with `--format mlx`.

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
