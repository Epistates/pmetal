# pmetal quantize

Quantize models to GGUF format with 26 quantization methods.

Quantize a model to GGUF format for efficient inference. Supports importance matrices for quality-preserving quantization, either read from a file or measured on calibration data in the same run.

## Usage

```bash
pmetal quantize \
  --model <MODEL> \
  --output <OUTPUT_FILE> \
  --method <METHOD> \
  [OPTIONS]
```

## Examples

```bash
# 4-bit quantization
pmetal quantize \
  --model ./output \
  --output model.gguf --method q4_k_m

# With an importance matrix measured earlier (see `pmetal imatrix`)
pmetal quantize \
  --model ./output \
  --output model.gguf --method q4_k_m \
  --imatrix imatrix.gguf

# Measure the importance matrix on calibration data first, in the same run
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
```

## Options

| Flag | Description |
|------|-------------|
| `--model` | Source model: a Hub id or a local directory (safetensors) |
| `--output` | GGUF file to write (a directory for `--format mlx`) |
| `--method` | Quantization method, see below (default `dynamic`) |
| `--imatrix` | Importance matrix file, GGUF or legacy `.dat`, from [`pmetal imatrix`](/cli/imatrix/) |
| `--calibration-dataset` | Measure the importance matrix on this data first: a text file, or a JSONL or Parquet dataset (local or a Hub id). Not with `--imatrix` |
| `--calibration-ctx-size` | Tokens per calibration chunk (default 512) |
| `--calibration-chunks` | Calibration chunks to run (default 0, all the data makes) |
| `--lora` | Fuse a LoRA adapter before quantizing (GGUF only) |
| `--kl-calibrate`, `--target-bpw`, `--kl-threshold` | Choose each tensor's type by measuring the damage, within a bits-per-weight budget |
| `--format`, `--bits`, `--group-size` | `mlx` output and its bit width and group size |

An importance matrix applies to GGUF output; `--imatrix` and `--calibration-dataset` are refused with `--format mlx`.

## Quantization Types

These are the values `--method` accepts.

| Method | Description |
|--------|-------------|
| `dynamic` | Importance-matrix-guided mixed precision (default; pair with `--imatrix` or `--calibration-dataset`) |
| `q8_0` | 8-bit integer, near-lossless |
| `q8_1` | 8-bit integer with dot-product sum helper |
| `q6_k` | 6-bit K-quant, high quality |
| `q5_k_m` | 5-bit K-quant medium |
| `q5_k_s` | 5-bit K-quant small |
| `q5_0` / `q5_1` | Legacy 5-bit symmetric / affine |
| `q4_k_m` | 4-bit K-quant medium (recommended 4-bit) |
| `q4_k_s` | 4-bit K-quant small |
| `q4_0` / `q4_1` | Legacy 4-bit symmetric / affine |
| `iq4_nl` | 4-bit non-linear, 32-weight blocks, about 4.5 bits per weight |
| `iq4_xs` | 4-bit non-linear, 256-weight super-blocks, about 4.25 bits per weight |
| `q3_k_l` / `q3_k_m` / `q3_k_s` | 3-bit K-quant large / medium / small |
| `q2_k` | 2-bit K-quant, lowest quality |
| `q1_0` | 1-bit sign |
| `tq1_0` / `tq2_0` | Ternary 1.69-bit / 2.06-bit |
| `mxfp4` / `nvfp4` | Block-floating 4-bit |
| `bf16` / `f16` / `f32` | Dense export |

The output head and token embedding are written as `q6_k` for the K-quant and 4-bit non-linear methods, and a tensor whose rows are not a whole number of blocks falls back to a smaller block format (`iq4_nl`, `q5_0`, `q5_1` or `q8_0`) as the format requires.

## See Also

- [pmetal imatrix](/cli/imatrix/) — Measure an importance matrix
- [Quantization](/models/quantization/) — Detailed quantization guide
