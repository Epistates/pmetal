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

Run the structured kernel benchmark corpus for the current Apple Silicon tier and emit a JSON artifact. This corpus covers standard-Metal hot paths on M1-M4 and adds MPP GEMM coverage on Apple10/M5 when NAX is available.

```bash
pmetal bench-corpus --quick --output .strategy/bench_corpus.json
```

Use `--json` to print the report to stdout, or omit `--quick` for the standard corpus run.

## See Also

- [Hardware Support](/hardware/apple-silicon/) — Hardware capabilities
- [Kernel Tuning](/hardware/kernel-tuning/) — Per-tier optimizations
