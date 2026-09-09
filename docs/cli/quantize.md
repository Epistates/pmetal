# pmetal quantize

Quantize models to GGUF format with 24 quantization methods.

Quantize a model to GGUF format for efficient inference. Supports importance matrix for quality-preserving quantization.

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

# With importance matrix
pmetal quantize \
  --model ./output \
  --output model.gguf --method q4_k_m \
  --imatrix calibration.jsonl

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

## Quantization Types

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

## See Also

- [Quantization](/models/quantization/) — Detailed quantization guide
