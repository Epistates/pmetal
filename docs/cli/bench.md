# pmetal bench

Benchmark training performance, generation speed, and FFI overhead.

Benchmark various aspects of PMetal's performance on your hardware.

## Subcommands

### bench

Benchmark training throughput (tokens/second, step time).

```bash
pmetal bench --model Qwen/Qwen3-0.6B --batch-size 4
```

### bench-gen

Benchmark the generation loop — tokens per second, time to first token, and decode latency.

```bash
pmetal bench-gen --model Qwen/Qwen3-0.6B --prompt "Hello" --max-tokens 100
```

### bench-ffi

Benchmark FFI overhead between Rust and Metal/MLX.

```bash
pmetal bench-ffi
```

### bench-corpus

Run the structured kernel benchmark corpus for the current Apple Silicon tier and emit a JSON artifact. This corpus covers standard-Metal hot paths on M1-M4, includes fused-merge plus initial model-family coverage for Llama 4 MoE, Qwen3-MoE, and Jamba hybrid layers, and adds MPP GEMM coverage on Apple10/M5 when NAX is available.

```bash
pmetal bench-corpus --quick --output .strategy/bench_corpus.json
```

Use `--json` to print the report to stdout, or omit `--quick` for the standard corpus run.

### bench-workload

Run a real cached workload benchmark for both inference and a short LoRA training pass. This is the quickest way to measure end-to-end M1-M4 behavior on a known model/dataset pair instead of only synthetic kernel shapes.

The current default workload is:
- model: `Qwen/Qwen3-0.6B`
- dataset: `TeichAI/gemini-3-pro-preview-high-reasoning-250x`

On Apple7-Apple9, `bench-workload` now records the KV cache mode it selected for inference. The default path is automatic: PMetal prefers fp16 KV cache when the model and context window fit comfortably, and only promotes to q8 when the device budget is tight.

The inference side is automatic by default too. If you omit `--max-prompt-tokens`, `bench-workload` tokenizes the sampled inference context, chooses a p95-aligned token window, and caps it at `1024` so the run stays quick without silently forcing everything through the old fixed `256` clamp. The default `--inference-context auto` mode prefers the dataset prompt field when it is substantial enough, but promotes to a `text-prefix` continuation context when the sampled prompts are too short to be a meaningful prefill benchmark. Pass `--max-prompt-tokens <N>` to force a specific inference token limit, or `--inference-context prompt|text-prefix` to force one context source.

`bench-workload` now also runs untimed warmup passes before measuring inference. The default is `--inference-warmup-passes 2`, applied per sampled prompt, and warmup-only prefetch traffic is excluded from the reported hit/miss counters. Lower it for faster smoke runs or raise it when comparing small end-to-end inference changes.

For noisier inference lanes, you can also use `--inference-session-repeats <N>`. That reruns the entire warmed inference workload `N` times and records both aggregate throughput and session-level median throughput in the JSON report, which is more trustworthy than comparing one warmed run.

You can also pass `--inference-repeats <N>` to run multiple timed inference passes per sampled prompt. The report now includes aggregate plus median decode/prefill throughput, which is more useful when comparing small inference changes on noisy short runs.

For large sparse checkpoints that use packed expert offload, pass `--experts-dir <packed_dir>`. `bench-workload` now uses the same offload-aware Qwen3.5 loader path as `pmetal infer`, so workloads like `Qwen3.5-122B-A10B` can benchmark the resident trunk plus packed routed experts instead of trying to dense-load the entire sparse model.

The training side is automatic too by default. If you omit `--max-seq-len`, `bench-workload` tokenizes the sampled training subset, chooses a p95-aligned sequence length, and caps it at `2048` so the run stays quick while avoiding obviously unrealistic truncation like the old fixed `512` default. Pass `--max-seq-len <N>` to force a specific value.

If you want a one-command regression lane instead of spelling out every knob, `--preset` now provides:
- `dense-qwen3`: the cached dense Qwen3-0.6B path
- `hybrid-qwen3next`: the cached non-dense Qwen3.5/Qwen3Next path using `unsloth/Qwen3.5-0.8B`, with `text-prefix` inference and training intentionally skipped so it stays a fast inference regression lane
- `hybrid-qwen35-steady`: the same cached `unsloth/Qwen3.5-0.8B` hybrid path, but with `2` prompt samples, `64` decode steps, `2` warmup passes per sample, and `3` timed inference repeats so decode throughput is less noisy when you are tuning inference
- `moe-nemotronh`: the cached Nemotron-H sparse/hybrid inference path with a `512`-token `text-prefix` window and training skipped

When `--preset` is set, it overrides the model/dataset/shape knobs below.

```bash
pmetal bench-workload \
  --preset dense-qwen3 \
  --train-samples 4 \
  --train-steps 2 \
  --json \
  --output .strategy/bench_workload_qwen3_0_6b.json
```

```bash
pmetal bench-workload \
  --preset hybrid-qwen3next \
  --json \
  --output .strategy/bench_workload_qwen3next_0_8b.json
```

```bash
pmetal bench-workload \
  --preset moe-nemotronh \
  --json \
  --output .strategy/bench_workload_nemotronh_4b.json
```

### bench-gdn

Run a focused Qwen3.5 / Qwen3Next Gated Delta Net microbenchmark against the actual model layer shapes. With the default `--stage input-proj`, this compares:
- `mlx_split`: the four separate MLX linear projections
- `mlx_combined`: one combined MLX matmul over concatenated weights
- `accelerate_combined`: one CPU-side Accelerate SGEMM over the same combined weight layout

With `--stage out-proj`, it instead compares the GDN output projection on the real layer shape using:
- `mlx_linear`
- `accelerate_combined`
- `accelerate_roundtrip_linear`

With `--stage prefill`, it benchmarks the recurrent GDN update on the real layer shape using:
- `mlx_sequential`: forced sequential recurrence
- `mlx_chunk_32`
- `mlx_chunk_64`
- `mlx_chunk_128`
- `mlx_chunk_256`

That is the right surface for testing prompt-heavy prefill ideas before changing the runtime chunk strategy.

This is the right tool when you want to evaluate flash-moe-style BLAS ideas on the real GDN projection buckets before changing runtime selection.

```bash
pmetal bench-gdn \
  --model /Users/nickpaterno/.cache/huggingface/hub/models--unsloth--Qwen3.5-0.8B/snapshots/<rev> \
  --json \
  --output .strategy/bench_gdn_qwen35_0_8b.json
```

```bash
pmetal bench-gdn \
  --stage out-proj \
  --model /Users/nickpaterno/.cache/huggingface/hub/models--unsloth--Qwen3.5-0.8B/snapshots/<rev> \
  --json \
  --output .strategy/bench_gdn_qwen35_0_8b_out_proj.json
```

```bash
pmetal bench-gdn \
  --stage prefill \
  --seq-len 1024 \
  --model /Users/nickpaterno/.cache/huggingface/hub/models--unsloth--Qwen3.5-0.8B/snapshots/<rev> \
  --json \
  --output .strategy/bench_gdn_qwen35_0_8b_prefill.json
```

For large sparse Qwen3.5 checkpoints, the command still works without `--experts-dir` because it only loads the hybrid trunk and benchmarks a GDN layer; it does not execute routed experts.

## See Also

- [Hardware Support](/hardware/apple-silicon/) — Hardware capabilities
- [Kernel Tuning](/hardware/kernel-tuning/) — Per-tier optimizations
