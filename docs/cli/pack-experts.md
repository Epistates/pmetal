# pmetal pack-experts

Pack the expert weights of a mixture-of-experts model for SSD-offloaded inference.

Reads the model's safetensors and writes one `layer_XX.bin` file per layer, with expert `E` starting at byte `E * expert_size`, and a `layout.json` that describes the binary layout to the reader. Pre-quantized models (safetensors that already hold `.scales` and `.biases` beside `.weight`) are packed as they are; full-precision models are quantized on the fly with affine per-group quantization. The routed experts then stay on disk while the rest of the model (the trunk) is resident: pass the output directory as `--experts-dir` to [`pmetal infer`](/cli/infer/), [`pmetal serve`](/cli/serve/) or [`pmetal bench-workload`](/cli/bench/).

## Usage

```bash
pmetal pack-experts --model <MODEL_DIR> [OPTIONS]
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `-m`, `--model` | required | Model directory (containing `config.json` and the safetensors) |
| `-o`, `--output` | `./packed_experts` | Output directory for the packed expert files |
| `-b`, `--bits` | `4` | Quantization bit width: `4` or `2` |

## Examples

```bash
# Pack the experts of a downloaded model to 4 bits
pmetal pack-experts \
  --model /path/to/model \
  --output ./packed_experts

# Run it with the experts on disk
pmetal infer \
  --model /path/to/model \
  --experts-dir ./packed_experts \
  --prompt "Summarize monotonic queues."
```

## See Also

- [pmetal infer](/cli/infer/): `--experts-dir`
- [pmetal serve](/cli/serve/): `--experts-dir`
- [pmetal bench](/cli/bench/): `bench-workload --experts-dir`
