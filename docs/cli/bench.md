# pmetal bench

Benchmark training and inference performance on your hardware.

Five commands share this page: `bench` (training step), `bench-gen` (generation loop), `bench-corpus` (kernel corpus), `bench-workload` (a real inference and LoRA workload) and `bench-gdn` (Gated Delta Net microbenchmarks).

Every benchmark reads the machine's load average when it starts and when it ends, prints both
beside the numbers and, in a JSON report, records them under `conditions`. When the machine was
not quiet (a load average above a quarter of its logical CPUs, and above 2) the numbers are
marked **PROVISIONAL**: a compile, a download or another run in the background takes the same
memory bandwidth and GPU time. Repeat on an idle machine before quoting a figure.

## pmetal bench

Benchmark training: the time of a LoRA training step on the model, as `pmetal train` takes it
(forward, loss, backward, optimizer; the trainer's defaults), on batches of `--batch-size` rows of
`--seq-len` tokens. Two warmup steps are run and not counted, then eight are measured; the report
gives the median, minimum and maximum step time, median tokens per second, peak memory and the
final loss. The rows are synthetic text cut to exactly `--seq-len` tokens, so the batch has the
shape asked for.

### Usage

```bash
pmetal bench [OPTIONS]
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `-m`, `--model` | `Qwen/Qwen3-0.6B` | Model to benchmark: a Hub ID or a local directory |
| `-b`, `--batch-size` | `1` | Rows per training step |
| `-s`, `--seq-len` | `512` | Tokens per row |

### Examples

```bash
pmetal bench --model Qwen/Qwen3-0.6B --batch-size 4 --seq-len 1024
```

## pmetal bench-gen

Profile the generation loop of a model: the mean time per token of each stage (reshape, forward,
logits, argmax, async evaluation, the sync point) and the effective tokens per second.

### Usage

```bash
pmetal bench-gen [--model <MODEL>]
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `-m`, `--model` | `Qwen/Qwen3-0.6B` | Model to benchmark |

### Examples

```bash
pmetal bench-gen --model Qwen/Qwen3-0.6B
```

## pmetal bench-corpus

Run the structured kernel benchmark corpus for the current Apple Silicon tier and emit a JSON artifact. The corpus covers standard-Metal hot paths on M1-M4, including fused merge and model-family coverage for Llama 4 MoE, Qwen3-MoE, Qwen3Next and NemotronH hybrid layers.

### Usage

```bash
pmetal bench-corpus [OPTIONS]
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--quick` | `false` | A shorter run with fewer iterations and smaller tier-scaled shapes |
| `--json` | `false` | Print the report to stdout as JSON |
| `-o`, `--output` | none | Write the JSON report to this path |

### Examples

```bash
pmetal bench-corpus --quick --output bench_corpus.json
```

## pmetal bench-workload

Run a real cached workload for both inference and a short LoRA training pass: the quickest way to measure end-to-end behavior on a known model and dataset instead of only synthetic kernel shapes.

The default workload is the model `Qwen/Qwen3-0.6B` with the dataset `TeichAI/gemini-3-pro-preview-high-reasoning-250x`.

- **KV cache mode.** The report records the KV cache mode it selected for inference. The default is automatic: fp16 when the model and context window fit comfortably, q8 only when the device budget is tight (Apple7-Apple9).
- **Inference context.** Without `--max-prompt-tokens`, the sampled inference context is tokenized and a p95-aligned token window is chosen, capped at 1024 so the run stays quick. `--inference-context auto` prefers the dataset's prompt field when it is substantial enough and uses a `text-prefix` continuation when the sampled prompts are too short to be a meaningful prefill benchmark; `prompt` and `text-prefix` force one source.
- **Warmup.** Untimed warmup passes run per sampled prompt before inference is measured (`--inference-warmup-passes`, default 2). Warmup-only prefetch traffic is excluded from the reported hit/miss counters.
- **Repeats.** `--inference-repeats` runs several timed passes per prompt; `--inference-session-repeats` reruns the whole warmed workload and records both aggregate and session-median throughput, which is more trustworthy on noisy machines than a single run. The report carries aggregate and median prefill and decode throughput.
- **Sparse checkpoints.** `--experts-dir` takes a packed expert directory (see [`pmetal pack-experts`](/cli/pack-experts/)) and loads through the same offload-aware path as `pmetal infer`, so a large sparse checkpoint benchmarks the resident trunk plus packed routed experts instead of dense-loading everything.
- **Training length.** Without `--max-seq-len`, the sampled training subset is tokenized and a p95-aligned sequence length is chosen, capped at 2048.

### Usage

```bash
pmetal bench-workload [OPTIONS]
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--preset` | none | `dense-qwen3`, `hybrid-qwen3next`, `hybrid-qwen35-steady` or `moe-nemotronh`; overrides the model, dataset and shape options (see below) |
| `--model` | `Qwen/Qwen3-0.6B` | Model ID or local path |
| `--dataset` | `TeichAI/gemini-3-pro-preview-high-reasoning-250x` | Dataset ID or local path |
| `--experts-dir` | none | Packed expert directory for sparse, offloaded inference workloads |
| `--prompt-samples` | `8` | Prompt samples to benchmark for inference |
| `--max-prompt-tokens` | `0` | Longest prompt per inference sample; `0` chooses from the sampled prompts, capped for a quick run |
| `--inference-context` | `auto` | Context source: `auto`, `prompt` or `text-prefix` |
| `--decode-steps` | `32` | Decode steps per prompt sample |
| `--inference-warmup-passes` | `2` | Untimed warmup passes per sampled prompt |
| `--inference-session-repeats` | `1` | Independent warmed inference sessions to run and aggregate |
| `--inference-repeats` | `1` | Timed inference passes per prompt sample |
| `--train-samples` | `8` | Raw samples in the sampled training subset |
| `--train-steps` | `4` | LoRA training steps to run |
| `--batch-size` | `1` | Training batch size |
| `--max-seq-len` | `0` | Sequence length of the short training run; `0` chooses from the sampled data, capped for a quick run |
| `--json` | `false` | Print the report as JSON |
| `-o`, `--output` | none | Write the JSON report to this path |

### Presets

A one-command regression lane instead of spelling out every option:

| Preset | Workload |
|--------|----------|
| `dense-qwen3` | The cached dense Qwen3-0.6B path |
| `hybrid-qwen3next` | The cached non-dense Qwen3.5/Qwen3Next path on `Qwen/Qwen3.5-0.8B`, with `text-prefix` inference and training skipped, as a fast inference regression lane |
| `hybrid-qwen35-steady` | The same `Qwen/Qwen3.5-0.8B` hybrid path with 2 prompt samples, 64 decode steps, 2 warmup passes per sample and 3 timed inference repeats, so decode throughput is less noisy while tuning inference |
| `moe-nemotronh` | The cached Nemotron-H sparse/hybrid inference path with a 512-token `text-prefix` window and training skipped |

### Examples

```bash
pmetal bench-workload \
  --preset dense-qwen3 \
  --train-samples 4 \
  --train-steps 2 \
  --json \
  --output bench_workload_qwen3_0_6b.json

pmetal bench-workload \
  --preset hybrid-qwen3next \
  --json \
  --output bench_workload_qwen3next_0_8b.json
```

## pmetal bench-gdn

A focused microbenchmark of Qwen3.5 / Qwen3Next Gated Delta Net on the actual model layer shapes. It loads only the hybrid trunk and benchmarks one GDN layer; it never runs routed experts, so large sparse checkpoints work without `--experts-dir`.

| `--stage` | Compares |
|-----------|----------|
| `input-proj` (default) | The four GDN input projections: `mlx_split` (four separate MLX linear projections), `mlx_combined` (one MLX matmul over concatenated weights) and `accelerate_combined` (one CPU-side Accelerate SGEMM over the same layout) |
| `out-proj` | The output projection after the recurrent update and norm: `mlx_linear`, `accelerate_combined` and `accelerate_roundtrip_linear` |
| `prefill` | The recurrent update at several chunk sizes: `mlx_sequential` (forced sequential recurrence) and `mlx_chunk_32`, `_64`, `_128`, `_256`. Use a longer `--seq-len` |

### Usage

```bash
pmetal bench-gdn [OPTIONS]
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `Qwen/Qwen3.5-0.8B` | Qwen3.5/Qwen3Next model ID or local path |
| `--stage` | `input-proj` | `input-proj`, `out-proj` or `prefill` |
| `--layer` | the first GDN layer | Transformer layer index to benchmark |
| `--batch-size` | `1` | Batch size of the synthetic input |
| `--seq-len` | `1` | Sequence length of the synthetic input (use a longer one for `prefill`) |
| `--warmup-iterations` | `10` | Warmup iterations per backend |
| `--benchmark-iterations` | `50` | Timed iterations per backend |
| `--json` | `false` | Print the report as JSON |
| `-o`, `--output` | none | Write the JSON report to this path |

### Examples

```bash
pmetal bench-gdn --json --output bench_gdn_qwen35_0_8b.json

pmetal bench-gdn --stage out-proj --json --output bench_gdn_qwen35_0_8b_out_proj.json

pmetal bench-gdn --stage prefill --seq-len 1024 --json --output bench_gdn_qwen35_0_8b_prefill.json
```

## See Also

- [Hardware Support](/hardware/apple-silicon/): Hardware capabilities
- [Kernel Tuning](/hardware/kernel-tuning/): Per-tier optimizations
- [pmetal info](/cli/info/): The device the numbers came from
