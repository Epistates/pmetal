# pmetal-mcp — MCP Server Parity Gap Audit

This document records the parity gaps found during the April 2026 audit and
the migration plan to close them. It is kept here rather than in the top-level
docs because the gaps are MCP-specific.

## Current state (post-P3 substrate landing)

The MCP server exposes ~46 tools backed by ~500 LOC of `push_opt` / `push_bool_flag`
argv construction. The `pmetal_core::jobs::*Spec` types (Phase 3 substrate) now
own the canonical field list, validation, and `to_argv()` for every subcommand.

The `train` tool has been migrated as an exemplar (see `lib.rs`). The pattern for
every remaining tool is identical:

```rust
// Inside #[tool] async fn <name>(&self, /* same params */) -> McpResult<String> {
let mut spec = <Job>Spec {
    field_a,
    field_b: field_b.unwrap_or(pmetal_core::defaults::FIELD_B_DEFAULT),
    // ... map all tool params to spec fields ...
    ..Default::default()
};
spec.normalize().map_err(|errs| McpError::invalid_params(format!(
    "validation failed: {}",
    errs.iter().map(|e| format!("{}: {}", e.field, e.message)).collect::<Vec<_>>().join("; ")
)))?;
let argv = spec.to_argv();
// Append any MCP-only overrides not yet in the spec:
push_bool_flag(&mut argv, "--flag-not-in-spec-yet", &optional_bool_param);
self.jobs.write().await.spawn("<subcommand>", argv).await
```

Tools remaining to migrate (mechanical, identical pattern):
- `distill` → `DistillSpec`
- `grpo` → `GrpoSpec`
- `rlkd` → `RlkdSpec`
- `embed_train` → `EmbedTrainSpec`
- `quantize` → `QuantizeSpec`
- `fuse_lora` → `FuseSpec`
- `merge_models` → `MergeSpec`
- `pack_experts` → `PackExpertsSpec`
- `start_serve` → `ServeSpec`

---

## Parity gap 1 — Missing inference flags in `generate`

The `generate` tool (which maps to `pmetal infer`) exposes approximately 21 flags.
The `InferSpec` in `pmetal_core::jobs::infer` models **32** fields. The following
10 are absent from the MCP `generate` tool today:

| Missing flag | `InferSpec` field | CLI flag |
|---|---|---|
| Backend selector | `backend: String` | `--backend auto\|standard\|compiled\|metal-sampler\|ane\|minimal\|dflash` |
| Draft model | `draft_model: Option<String>` | `--draft-model <path>` |
| Compiled sampling | `compiled: bool` | `--compiled` |
| Benchmark mode | `benchmark: bool` | `--benchmark` |
| Profile layers | `profile_layers: bool` | `--profile-layers` |
| KV K-bits (per-key quant) | `kv_k_bits: Option<u8>` | `--kv-k-bits <bits>` |
| KV V-bits (per-value quant) | `kv_v_bits: Option<u8>` | `--kv-v-bits <bits>` |
| KV group size | `kv_group_size: usize` | `--kv-group-size <n>` (default 64) |
| KV TurboQuant | `kv_turboquant: bool` | `--kv-turboquant` |
| Detect repetition | `detect_repetition: bool` | `--detect-repetition` |

### Closing this gap (follow-up PR)

Add the missing params to the `generate` signature and replace the manual argv
construction with `InferSpec::to_argv()`. Since `generate` currently uses
`run_pmetal_blocking` rather than a background job, the migration is:

```rust
async fn generate(&self, model: String, prompt: String,
    /* existing params... */
    backend: Option<String>,
    draft_model: Option<String>,
    compiled: Option<bool>,
    benchmark: Option<bool>,
    profile_layers: Option<bool>,
    kv_k_bits: Option<u64>,
    kv_v_bits: Option<u64>,
    kv_group_size: Option<u64>,
    kv_turboquant: Option<bool>,
    detect_repetition: Option<bool>,
) -> McpResult<String> {
    let spec = InferSpec {
        model,
        prompt,
        max_tokens: max_tokens.unwrap_or(256) as usize,
        backend: backend.unwrap_or_else(|| "auto".to_string()),
        draft_model,
        compiled: compiled.unwrap_or(false),
        benchmark: benchmark.unwrap_or(false),
        profile_layers: profile_layers.unwrap_or(false),
        kv_k_bits: kv_k_bits.map(|b| b as u8),
        kv_v_bits: kv_v_bits.map(|b| b as u8),
        kv_group_size: kv_group_size.unwrap_or(64) as usize,
        kv_turboquant: kv_turboquant.unwrap_or(false),
        detect_repetition: detect_repetition.unwrap_or(false),
        // ... remaining existing fields ...
        ..InferSpec::default()
    };
    let mut argv = vec!["infer".to_string()];
    argv.extend(spec.to_argv());
    util::run_pmetal_blocking_argv(&argv).await
}
```

---

## Parity gap 2 — Missing `tokenize` tool

The CLI has `pmetal tokenize`; the MCP server has no equivalent. `TokenizeSpec`
exists in `pmetal_core::jobs::tokenize`.

### Implementation (5 lines, follow-up PR)

```rust
/// Tokenize a JSONL text corpus into binary shards for pretraining.
/// Returns a job ID for tracking. Maps to `pmetal tokenize`.
#[tool]
async fn tokenize(
    &self,
    #[description("Input JSONL path")] input: String,
    #[description("Output shard directory")] output: String,
    #[description("Tokenizer model path or HF ID")] tokenizer: String,
    #[description("Text column in JSONL (default: text)")] text_column: Option<String>,
    #[description("Documents per shard (default: 10000)")] docs_per_shard: Option<u64>,
) -> McpResult<String> {
    let mut spec = pmetal_core::jobs::TokenizeSpec {
        input,
        output,
        tokenizer,
        text_column: text_column.unwrap_or_else(|| "text".to_string()),
        docs_per_shard: docs_per_shard.unwrap_or(10_000) as usize,
    };
    spec.normalize().map_err(into_mcp_error)?;
    let argv = spec.to_argv();
    self.jobs.write().await.spawn("tokenize", argv).await
}
```

---

## Parity gap 3 — Missing `memory` tool

The CLI has `pmetal memory` (displays device memory, model fit estimates, and
working-set usage). This is a read-only query with no `*Spec`, but the existing
`util::build_device_info_json` + `pmetal_hub::estimate_fit` already provide the
pieces.

### Implementation (follow-up PR)

```rust
/// Estimate memory requirements for a model on this device.
/// Equivalent to `pmetal memory --model <model>`.
/// Returns inference/training memory breakdown and fit level.
#[tool]
async fn memory(
    &self,
    #[description("Model ID or local path")] model: String,
    #[description("Context length (default: 4096)")] context_length: Option<u64>,
    #[description("Quantization format (fp16, q4_k_m, fp8, …)")] quantization: Option<String>,
) -> McpResult<String> {
    // Delegates to the existing model_fit logic; extract into a shared helper
    // `fn estimate_model_memory(model, context_length, quantization) -> McpResult<Value>`
    // then call it from both model_fit and memory.
    todo!("delegate to model_fit internals")
}
```

---

## Parity gap 4 — Missing `dflash` tool

The CLI has `pmetal dflash` (block-diffusion speculative decoding); the MCP
server has no equivalent. `DflashSpec` exists in `pmetal_core::jobs::dflash`.

### Implementation (follow-up PR)

```rust
/// Run block-diffusion speculative decoding (dflash).
/// Requires a target model and a draft model.
/// Returns a job ID for tracking.
#[tool]
async fn dflash(
    &self,
    #[description("Target (large) model ID or path")] target: String,
    #[description("Draft (small) model ID or path")] draft: String,
    #[description("Prompt text")] prompt: String,
    #[description("Max new tokens (default: 128)")] max_new_tokens: Option<u64>,
    #[description("Temperature (0.0 = greedy)")] temperature: Option<f64>,
    #[description("Speculative tokens per step")] speculative_tokens: Option<u64>,
    #[description("Use FP8 for draft model")] draft_fp8: Option<bool>,
    #[description("Tree budget (0 = disabled)")] tree_budget: Option<u64>,
) -> McpResult<String> {
    let mut spec = pmetal_core::jobs::DflashSpec {
        target,
        draft,
        prompt,
        max_new_tokens: max_new_tokens.unwrap_or(128) as usize,
        temperature: temperature.unwrap_or(0.0) as f32,
        speculative_tokens: speculative_tokens.map(|t| t as usize),
        draft_fp8: draft_fp8.unwrap_or(false),
        tree_budget: tree_budget.unwrap_or(0) as usize,
        ..Default::default()
    };
    spec.normalize().map_err(into_mcp_error)?;
    let argv = spec.to_argv();
    self.jobs.write().await.spawn("dflash", argv).await
}
```

---

## Architecture notes

### JobEvent JSONL consumer (landed)

`jobs.rs` now tries `pmetal_core::events::parse_event` on each stdout line before
falling back to the legacy `{"step":N,"loss":F}` flat-JSON parser. This means
MCP works with both old subprocesses (flat metrics JSON) and new ones that emit
`JobEvent` JSONL via `--log-events /dev/stdout`.

The structured path populates `JobMetrics` from `MetricPayload::Step` and
`MetricPayload::Eval`; the legacy path remains for backward compatibility until
all subcommands have been ported to emit `JobEvent` JSONL.

### Ring-buffer cap

`MAX_BUFFER_LINES = 10_000`. When full, the oldest **10%** (1000 lines) are
dropped atomically before inserting the new line. This bounds steady-state memory
at ~10 000 lines per stream regardless of job duration.

### Wire-format stability

`JobStatus` in `jobs.rs` retains `#[serde(tag = "state", rename_all = "snake_case")]`
and its current variant names (`Running`, `Stopping`, `Completed { exit_code }`,
`Failed { exit_code, error }`, `Stopped`). This is intentionally NOT migrated to
`pmetal_core::JobStatus<R>` yet because the wire format differs (core uses
`Running { progress, last_metric }` vs MCP's thin `Running`). A follow-up PR
will migrate after agreeing on the wire contract with existing MCP clients.
