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

The training side is automatic too by default. If you omit `--max-seq-len`, `bench-workload` tokenizes the sampled training subset, chooses a p95-aligned sequence length, and caps it at `2048` so the run stays quick while avoiding obviously unrealistic truncation like the old fixed `512` default. Pass `--max-seq-len <N>` to force a specific value.

If you want a one-command regression lane instead of spelling out every knob, `--preset` now provides:
- `dense-qwen3`: the cached dense Qwen3-0.6B path
- `hybrid-qwen3next`: the cached non-dense Qwen3.5/Qwen3Next path with `text-prefix` inference and training intentionally skipped so it stays a fast inference regression lane

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

## See Also

- [Hardware Support](/hardware/apple-silicon/) — Hardware capabilities
- [Kernel Tuning](/hardware/kernel-tuning/) — Per-tier optimizations
