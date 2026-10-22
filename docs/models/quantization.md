# Quantization

GGUF and MLX quantization, importance matrices and KL-calibrated per-tensor types.

`pmetal quantize` writes a GGUF file (or an MLX-format directory) from a safetensors model, optionally fusing a LoRA adapter first (GGUF only). The method is chosen with `--method`.

## GGUF Methods

24 methods:

| Group | `--method` values |
|-------|-------------------|
| Adaptive | `dynamic` (default): mixed precision, with the type of each tensor chosen per layer (guided by `--imatrix` when given) |
| K-quants | `q2_k`, `q3_k_s`, `q3_k_m`, `q3_k_l`, `q4_k_s`, `q4_k_m`, `q5_k_s`, `q5_k_m`, `q6_k` |
| Legacy | `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`, `q8_1` |
| Low-bit | `q1_0`, `tq1_0`, `tq2_0` |
| Block floating point | `mxfp4`, `nvfp4` |
| Unquantized | `bf16`, `f16`, `f32` |

```bash
pmetal quantize --model ./output --output model.gguf --method q4_k_m
```

The size suffixes follow GGUF naming: `s` small, `m` medium (the usual choice), `l` large. `pmetal quantize --help` lists every value.

## Importance Matrix

`--imatrix` takes an importance-matrix file: the legacy `imatrix.dat` format or a GGUF imatrix, as other tools write them. The matrix weights the quantization error toward the activations that matter, and steers which tensors get more bits under `dynamic`. PMetal reads these files; it does not compute one from a dataset.

```bash
pmetal quantize --model ./output --output model.gguf --method q4_k_m --imatrix imatrix.dat
```

## KL-Calibrated Quantization

`--kl-calibrate` chooses each tensor's quantization type by measuring the damage (NRMSE and cosine distance) rather than by a fixed rule. `--target-bpw` sets an average bits-per-weight budget, and `--kl-threshold` (default 0.01) the quality-loss threshold:

```bash
pmetal quantize --model ./output --output model.gguf --kl-calibrate --target-bpw 4.5
```

## MLX Format

`--format mlx` writes an MLX-layout quantized checkpoint (`model.safetensors`, `config.json` with a `quantization_config`, and the tokenizer files) into the `--output` directory, at `--bits` (3, 4, 5, 6 or 8; default 4) and `--group-size` (default 64); `--target-bpw` sets a per-tensor bit allocation budget. It takes no `--lora`: fuse the adapter first with `pmetal fuse`.

```bash
pmetal quantize --model ./output --output ./model-mlx --format mlx --bits 4
```

## Quantized Checkpoints as Input

PMetal also loads already-quantized checkpoints packed, without dequantizing them: MLX-quantized (affine, mxfp4, nvfp4, mxfp8), NVIDIA ModelOpt NVFP4 and FP8, and gpt-oss MXFP4. QLoRA (`pmetal train --quantization nf4|fp4|int8`) packs the base of the model it trains.

## FP8 Runtime Quantization

For inference-time memory reduction without GGUF conversion:

```bash
pmetal infer --model Qwen/Qwen3-4B --fp8 --chat
```

Converts to FP8 (E4M3) at load time for approximately 2x memory reduction.

## KV Cache Quantization

Separately from weights, `pmetal infer` and `pmetal serve` can compress the KV cache: `--kv-quant 8|4` (affine, with `--kv-group-size`), or `--kv-turboquant` (rotated, bit-packed keys and values; native Qwen3, Qwen3.5 and Llama 4 engines) with the `--kv-turboquant-preset q2_5|q3_5` mixed-bit presets. `--no-kv-quant` keeps the cache in fp16.

## See Also

- [pmetal quantize](/cli/quantize/): CLI reference
- [Supported Models](/models/supported/): Compatible architectures
