# GUI DTO Migration Plan

## Current State

`commands.rs` contains 9 hand-rolled request DTOs that the Tauri IPC layer
deserialises from JSON sent by the SvelteKit frontend:

| Rust DTO | Tauri command | TS interface (api.ts) |
|---|---|---|
| `TrainingConfig` | `start_training` | `TrainingConfig` (43 fields) |
| `DistillationConfig` | `start_distillation` | `DistillationConfig` (14 fields) |
| `GrpoConfig` | `start_grpo` | `GrpoConfig` (13 fields) |
| `InferenceConfig` | `start_inference` | `InferenceConfig` (22 fields) |
| `ServeConfigDto` | `start_serve` | `ServeConfigDto` (8 fields) |
| `BenchConfigDto` | `start_bench` | `BenchConfigDto` (12 fields) |
| `EvalConfigDto` | `start_eval` | `EvalConfigDto` (7 fields) |
| `PretrainConfigDto` | `start_pretrain` | `PretrainConfigDto` (17 fields) |
| `MergeConfig` + `MergeModelEntry` | `merge_models` | `MergeConfig` (4 fields) |

The JSON shape serialised by the frontend must be byte-identical to the field
names the Rust `Deserialize` impl expects. Any field rename or removal on the
Rust side is a breaking change for the frontend.

## Target State

After the Phase 4 substrate migration completes, each `start_*` command should
accept the corresponding `pmetal::core::jobs::*Spec` directly:

```rust
// Before
#[tauri::command]
pub async fn start_training(config: TrainingConfig, ...) -> Result<String>

// After
#[tauri::command]
pub async fn start_training(spec: pmetal::core::jobs::TrainSpec, ...) -> Result<String>
```

The frontend would then construct the `*Spec` JSON shape rather than the legacy
DTO shape. `TrainSpec` uses `snake_case` field names (identical to `serde`
defaults) so most field names transfer directly, but the set of fields differs
— `TrainSpec` has `eval_dataset`, omits `dpo_loss_type` (which becomes part of
`method`), and uses `output_dir` instead of `output_dir: Option<String>`.

## Why This is Deferred

The migration is **frontend-lockstep**: the Rust side cannot accept the new
`*Spec` shape until the TypeScript side stops sending the legacy DTO shape, and
vice versa. There is no backwards-compatible incremental path because:

1. `TrainConfig::method` is a plain `String` in the current DTO; `TrainSpec`
   encodes method choice via a separate `method` field with an enum constraint.
2. `MergeConfig` uses a `Vec<MergeModelEntry>` (arbitrary-length list of
   models with weights), while `MergeSpec` is a two-model struct
   (`model_a`/`model_b`) — fundamentally incompatible shapes.
3. The SvelteKit `routes/training/+page.svelte` builds the full 43-field
   `TrainingConfig` object inline at submit time. Changing the Rust side
   without changing that page causes a silent `missing field` deserialisation
   failure at runtime.

### Call-site count (from `lib/api.ts`)

Each DTO appears exactly once as a function parameter in `api.ts` (the wrappers
are thin `invoke` shims), but the *construction sites* in route pages are
where the coupling is deepest:

| DTO | api.ts wrappers | Route construction sites |
|---|---|---|
| `TrainingConfig` | 1 (`startTraining`) | 1 (`routes/training/+page.svelte` ~43 fields) |
| `DistillationConfig` | 1 | 1 (`routes/distillation/+page.svelte`) |
| `GrpoConfig` | 1 | 1 (`routes/grpo/+page.svelte`) |
| `InferenceConfig` | 1 | 1 (`routes/inference/+page.svelte`) |
| `ServeConfigDto` | 1 | 1 (`routes/serve/+page.svelte`) |
| `BenchConfigDto` | 1 | 1 (`routes/bench/+page.svelte`) |
| `EvalConfigDto` | 1 | 1 (`routes/eval/+page.svelte`) |
| `PretrainConfigDto` | 1 | 1 (`routes/pretrain/+page.svelte`) |
| `MergeConfig` | 1 | 1 (`routes/merging/+page.svelte`) |

Even though each DTO has one construction site, each site assembles 7–43
fields from local reactive state — a non-trivial diff that must be coordinated
with a TS type change in `api.ts` and the route component simultaneously.

The full migration is scoped as a dedicated PR that rewrites all 9 route pages.
Estimated effort: 2–3 days.

## Migration Sequence

When the TS/Rust migration PR is ready, apply changes in this order to keep
each intermediate commit compilable:

### Step 1 — Rust side

For each DTO:
1. Remove the hand-rolled struct (e.g. `TrainingConfig`).
2. Change the command signature to accept `pmetal::core::jobs::TrainSpec`.
3. Remove the mapping code that converts DTO fields to `TrainingJobConfig`;
   instead call `spec.normalize()?` then `spec.to_argv()` or construct the
   backend config directly from spec fields.
4. Run `cargo build --quiet -p pmetal-gui`.

### Step 2 — api.ts

For each DTO:
1. Remove the `export interface TrainingConfig { ... }` declaration.
2. Import the equivalent type from a generated or hand-written `spec-types.ts`
   (see note below) or inline the `TrainSpec` shape.
3. Update the `startTraining(config: TrainingConfig, ...)` wrapper to
   `startTraining(spec: TrainSpec, ...)`.

Note: Until `pmetal-core-derive` emits TypeScript bindings (a later Phase),
the TS types must be maintained manually. Consider running
`cargo test -p pmetal-core -- --nocapture` to print field lists as a
generation aid.

### Step 3 — Route components

For each `routes/*/+page.svelte`:
1. Update the import to use the new TS interface name.
2. Rename the constructed object's fields to match `*Spec` names.
3. Remove fields not present in `*Spec`; add defaults for new required fields.
4. Run `pnpm --dir crates/pmetal-gui exec svelte-check`.

### Step 4 — Verification

```sh
cargo build --quiet -p pmetal-gui
pnpm --dir crates/pmetal-gui exec svelte-check
```

End-to-end smoke: open each route, submit a minimal job, confirm the Tauri
command receives a valid spec.

---

## Worked Example — `MergeConfig` → `MergeSpec`

`MergeConfig` is the smallest DTO (4 fields) but also the most structurally
different from its `MergeSpec` counterpart, so it illustrates both the simple
case and the data-shape mismatch problem.

### Current Rust DTO (`commands.rs`)

```rust
#[derive(Debug, Deserialize)]
pub struct MergeConfig {
    pub base_model: String,
    pub models: Vec<MergeModelEntry>,  // arbitrary-length list
    pub strategy: String,
    pub output: String,
}

#[derive(Debug, Deserialize)]
pub struct MergeModelEntry {
    pub model: String,
    pub weight: f64,
}
```

### Target `MergeSpec` (`pmetal-core/src/jobs/merge.rs`)

```rust
pub struct MergeSpec {
    pub model_a: String,   // first model (required)
    pub model_b: String,   // second model (required)
    pub output: String,
    pub method: String,    // "slerp" | "ties" | "dare_ties" | ...
    pub base: Option<String>,
    pub t: f32,            // SLERP interpolation
    pub weight_a: f32,
    pub weight_b: f32,
    pub density: f32,
    pub dtype: String,
}
```

### Shape mismatch

`MergeConfig` accepts an arbitrary-length `Vec<MergeModelEntry>` which the
route uses to present a dynamic "add model" UX. `MergeSpec` is a fixed
two-model struct matching the CLI `pmetal merge --model-a X --model-b Y`
interface. The migration must resolve this by one of:

a) **Restricting the frontend to exactly 2 models** (simplest — the CLI only
   supports 2 anyway).
b) **Keeping the flexible UI** but performing a pre-flight expansion: the
   command handler converts `models[0]` → `spec.model_a`, `models[1]` →
   `spec.model_b`, and rejects requests with more than 2 entries.

Option (a) is recommended: the existing `merge_models` command already calls
`pmetal merge` which only supports 2 models. The multi-model UI affordance is
aspirational.

### Migration steps for MergeConfig specifically

**Rust** (`commands.rs`):
```rust
// Remove:
pub struct MergeConfig { ... }
pub struct MergeModelEntry { ... }

// Change command signature:
pub async fn merge_models(
    state: State<'_, AppState>,
    spec: pmetal::core::jobs::MergeSpec,
) -> Result<String>

// Replace DTO-to-argv mapping with:
spec.normalize().map_err(|errs| AppError(errs[0].message.clone()))?;
let argv = spec.to_argv();
// ... spawn subprocess with argv
```

**api.ts**:
```typescript
// Remove:
export interface MergeConfig { ... }
export interface MergeModelEntry { ... }

// Add (or import from spec-types.ts):
export interface MergeSpec {
  model_a: string;
  model_b: string;
  output: string;
  method: string;
  base?: string | null;
  t?: number;
  weight_a?: number;
  weight_b?: number;
  density?: number;
  dtype?: string;
}

// Change wrapper:
export async function mergeModels(spec: MergeSpec): Promise<string> {
  return await invoke('merge_models', { spec });
}
```

**routes/merging/+page.svelte** — replace the `modelEntries` Vec with two
separate model pickers (`modelA`, `modelB`) and map the strategy picker to
`spec.method`. The `normalizeWeights()` helper becomes a simple slider between
`weight_a` and `weight_b` (constrained to `weight_a + weight_b = 1`).

### Line count estimate for MergeConfig migration
- `commands.rs`: remove ~18 lines (DTO structs), rewrite ~30 lines of merge
  handler → ~-10 net (spec.to_argv() replaces manual argv building).
- `api.ts`: remove ~12 lines, add ~14 lines (MergeSpec interface) → ~+2 net.
- `routes/merging/+page.svelte`: replace ~25 lines of modelEntries logic →
  ~-10 net (two pickers are simpler than a dynamic list).

Total estimated diff for MergeConfig alone: ~50 lines touched.

---

## Resume-from-Checkpoint (implemented)

The `TrainingConfig.resume_from: Option<String>` field is now wired through to
`TrainingJobConfig.resume = true` and `config_path = Some(resume_from)`.  
The orchestrator's `CheckpointManager` loads the latest checkpoint from the
given directory. No frontend changes are required — the field was already
present in the TS `TrainingConfig` interface and the route already sends it.

The pre-flight check (directory must exist) was added to `start_training`; an
invalid path returns `Err(AppError(...))` before creating a run record.
