# Apple Silicon Hardware Support

What PMetal detects about the machine, and what it does with it.

## Detection

At startup PMetal asks the device what it is; nothing is derived from the amount of installed RAM. `pmetal info` prints everything below (`--json` for scripts).

| Property | Source |
|----------|--------|
| GPU family (`Apple7` to `Apple10`) | Metal `supportsFamily` |
| Architecture generation (13 for M1 through 17 for M5) | The number in Metal's architecture name (`applegpu_g16s` is 16) |
| Device tier (`Base`, `Pro`, `Max`, `Ultra`) | The device's marketing name; Metal exposes no tier |
| GPU core count | The GPU's IO registry entry (chips are binned, so a 32-core and a 40-core M4 Max differ); the full chip's count when the registry will not say |
| ANE core count | 16, or 32 on an Ultra |
| Memory bandwidth | A GPU copy benchmark measured on this machine and cached on disk; otherwise Apple's published peak for the chip, tier and core count |
| NAX (neural accelerators in the GPU) | Generation 17 or later, on macOS 26.2 or later |
| Working-set limit | Metal's `recommendedMaxWorkingSetSize` |
| UltraFusion | An Ultra is two Max dies joined by UltraFusion |

## Chip families

| Chip family | GPU family | Arch gen | Published peak bandwidth (GB/s) | NAX |
|-------------|-----------|----------|---------------------------------|-----|
| M1 / Pro / Max / Ultra | Apple7 | 13 | 68.25 / 200 / 400 / 800 | No |
| M2 / Pro / Max / Ultra | Apple8 | 14 | 102.4 / 200 / 400 / 800 | No |
| M3 / Pro / Max / Ultra | Apple9 | 15 | 102.4 / 150 / 300 or 400 / 819.2 | No |
| M4 / Pro / Max | Apple9 | 16 | 120 / 273 / 409.6 or 546 | No |
| M5 / Pro / Max | Apple10 | 17 | 153.6 / 307 / 460 or 614 | Yes |

Where two figures are given for a Max, the lower one is the binned part. An M4 or M5 Ultra is reported as twice the bandwidth of its Max die, and a chip the table does not know gets a figure by tier alone (100, 200, 400 or 800 GB/s for Base, Pro, Max and Ultra). These are published peaks, not measurements; `pmetal info` says whether the figure it shows was measured or taken from this table.

## M5 and NAX

M5-generation GPUs (Apple10, generation 17) have neural accelerators in each GPU core, reached through Metal 4 tensor operations. PMetal's runtime uses MLX's NAX kernels for matrix multiplication, quantized matmul and attention on these chips, with the same gate as detection: generation 17 or later on macOS 26.2 or later. On earlier macOS releases an M5 runs the standard Metal path.

## Neural Engine

The ANE path uses Apple's private `AppleNeuralEngine.framework` APIs, loaded at runtime, so it is experimental and depends on the macOS release. Kernels are compiled to MIL programs; Apple's compiler service caches them between runs.

- **Training** (`pmetal train --ane`): training kernels compiled once, with weights packed beside activations in IOSurfaces, so a step does not recompile. RMSNorm runs in f32 on the CPU to avoid fp16 overflow on the ANE.
- **Inference** (`pmetal infer --ane`, `pmetal serve --ane`): the whole model on the ANE, several layers per program, over a KV cache held in IOSurfaces, with int8 weights by default. Context lengths are bucketed to powers of two from 512 up to `--ane-max-seq-len` (default 4096), and a smaller context runs faster. DFlash drafting is supported (`--draft-model`).

## Benchmarking your machine

- `pmetal bench` and `pmetal bench-gen` time training and the generation loop.
- `pmetal bench-workload` runs a real cached workload (inference and a short LoRA run) and records the KV cache mode it chose.
- `pmetal bench-corpus` runs a structured kernel corpus for this device tier and can write a JSON report.
- `pmetal bench-gdn` times the Qwen3.5-family Gated DeltaNet backends on real layer shapes.

Each benchmark reads the load average when it starts and ends and prints both, so a result taken on a busy machine says so.

## Open items

- Validate the local UltraFusion pipeline planner (`pmetal info` reports its plan on Ultra machines) against real Ultra hardware.
- Check in `bench-corpus` reports from more chip generations.
