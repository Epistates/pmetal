# Quantization

GGUF and MLX quantization, importance matrices and KL-calibrated per-tensor types.

`pmetal quantize` writes a GGUF file (or an MLX-format directory) from a safetensors model, optionally fusing a LoRA adapter first (GGUF only). The method is chosen with `--method`.

## GGUF Methods

26 methods:

| Group | `--method` values |
|-------|-------------------|
| Adaptive | `dynamic` (default): mixed precision, with the type of each tensor chosen per layer (guided by an importance matrix when given) |
| K-quants | `q2_k`, `q3_k_s`, `q3_k_m`, `q3_k_l`, `q4_k_s`, `q4_k_m`, `q5_k_s`, `q5_k_m`, `q6_k` |
| Non-linear 4-bit | `iq4_nl` (about 4.5 bits per weight), `iq4_xs` (about 4.25) |
| Legacy | `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`, `q8_1` |
| Low-bit | `q1_0`, `tq1_0`, `tq2_0` |
| Block floating point | `mxfp4`, `nvfp4` |
| Unquantized | `bf16`, `f16`, `f32` |

```bash
pmetal quantize --model ./output --output model.gguf --method q4_k_m
```

The size suffixes follow GGUF naming: `s` small, `m` medium (the usual choice), `l` large. `pmetal quantize --help` lists every value.

A K-quant method names a mix, not one type. `q3_k_m` stores most tensors in Q3_K and keeps the ones that matter most at more bits: `attn_v` in Q5_K for the first two layers and Q4_K after, `ffn_down` in Q5_K for the first sixteenth of the layers and Q4_K after, `attn_output` in Q4_K, and the output head in Q6_K. `q4_k_m` and `q5_k_m` raise `attn_v` and `ffn_down` to Q6_K in the first and last eighth of the layers and every third layer between. The mix is the reference quantizer's for every method, layer by layer, including the special cases for 80-layer grouped-query models and eight-expert models, so a file is the same size and quality as the one that quantizer writes under the same name. A model whose head is tied to its embedding has no `output.weight`, and the embedding is then stored as a head, in Q6_K.

`iq4_nl` and `iq4_xs` store four-bit indices into a fixed non-linear table of 16 values, spaced more finely near zero where most weights are. At a similar size they usually reconstruct a tensor more closely than the linear 4-bit formats, and they benefit most from an importance matrix.

## Importance Matrix

An importance matrix records, for each weight matrix, how large the activations in each input channel are on representative text. The quantization error of a weight matters in proportion to the activations it meets, so the quantizers (legacy and K-quants and the non-linear 4-bit formats) choose their scales to minimise the error weighted by it, and under `dynamic` it also steers which tensors get more bits.

`pmetal imatrix` measures one, as a GGUF imatrix file (or the legacy `.dat` format for a `.dat` path), by running the model over calibration data in chunks of `--ctx-size` tokens:

```bash
pmetal imatrix --model ./output --dataset calibration.txt --output imatrix.gguf
pmetal quantize --model ./output --output model.gguf --method q4_k_m --imatrix imatrix.gguf
```

The calibration data is a text file (one document), or a JSONL or Parquet dataset (local or a Hub id), rendered as for training: conversations through the model's chat template. Use a few hundred chunks of text like what the model will run on. For a one-off quantization, `--calibration-dataset` does the same measurement in the same run, on the model being quantized:

```bash
pmetal quantize --model ./output --output model.gguf --method iq4_xs --calibration-dataset calibration.txt
```

A measured matrix is reusable: it depends on the model and the data, not on the quantization method, so one file serves every quantization of that model. `--imatrix` also reads the files other tools write, in either format. Mixture-of-experts layers are measured per expert, counting only the tokens routed to each. A weight matrix the measurement could not reach is reported and quantized without importance.

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

## GGUF as Input

`pmetal infer`, `pmetal eval` and `pmetal serve` load a GGUF file directly, including one written by `pmetal quantize`, for the architectures listed under [GGUF Checkpoints](/models/supported/#gguf-checkpoints). The tensors are dequantized on load, so perplexity on a GGUF is the perplexity of the quantized weights: a good check that a quantization lost little is `pmetal eval` on the safetensors model and on its GGUF.

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
- [pmetal imatrix](/cli/imatrix/): Measure an importance matrix
- [Supported Models](/models/supported/): Compatible architectures
