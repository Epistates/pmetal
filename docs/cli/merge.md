# pmetal merge

Merge two models using 12 merge strategies.

The CLI merges exactly two models at a time; task-vector methods take a third via `--base`.
For N-way merges and slice-based frankenmerging, build a `pmetal_merge::MergeConfig` and call
`run_merge` directly — the CLI does not accept a merge config file. Supports GPU-accelerated merging, FP8-aware merging, and async double-buffered streaming for large models.

## Usage

```bash
pmetal merge \
  --model-a <MODEL_A> \
  --model-b <MODEL_B> \
  --output <OUTPUT> \
  [OPTIONS]
```

## Examples

```bash
# SLERP merge
pmetal merge \
  --model-a model-a --model-b model-b \
  --output ./merged \
  --method slerp --t 0.5

# TIES merge with sparsification
pmetal merge \
  --model-a ft-model-1 --model-b ft-model-2 \
  --base base-model --output ./merged \
  --method ties --density 0.5

# DARE-TIES with random pruning
pmetal merge \
  --model-a model-a --model-b model-b \
  --base base-model --output ./merged \
  --method dare_ties --density 0.7
```

## Strategies

| Method | Description |
|--------|-------------|
| `linear` | Simple weighted averaging |
| `slerp` | Spherical linear interpolation |
| `ties` | Task arithmetic with sparsification and sign consensus |
| `dare_ties` | Random pruning with rescaling (TIES variant) |
| `dare_linear` | Random pruning with rescaling (linear variant) |
| `task_arithmetic` | Task vector arithmetic |
| `della` | Adaptive magnitude-based pruning |
| `della_linear` | Adaptive magnitude pruning (linear variant) |
| `breadcrumbs` | Breadcrumbs merge strategy |
| `model_stock` | Geometric interpolation based on task vector similarity |
| `nearswap` | Near-swap merge strategy |
| `passthrough` | Layer passthrough composition |

Three more (`ram`, `ram_plus`, `multi_slerp`) are reachable by setting `MergeConfig.merge_method` directly.

## See Also

- [Model Merging](/models/merging/) — Detailed merge documentation
