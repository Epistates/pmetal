# Model Merging

Merge models with 15 strategies — SLERP, TIES, DARE, Task Arithmetic, and more.

PMetal supports 15 model merging strategies (12 via CLI, 3 more via `MergeConfig`). Features GPU-accelerated merging, FP8-aware operations, and async double-buffered streaming for large models.

## CLI Strategies

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

## Library-Only Strategies

Reachable by setting `MergeConfig.merge_method` directly; `pmetal merge --method` does not accept them.

| `merge_method` | Description |
|----------------|-------------|
| `ram` | RAM (Rescaled Activation Merging) |
| `ram_plus` | RAM+ variant |
| `multi_slerp` | Multi-model SLERP |

`FisherMerge`, `RegMeanMerge` and `SouperMerge` are implemented in
`pmetal_merge::methods` but have no `MergeMethodConfig` variant, so they are only usable by
constructing the `MergeMethod` trait object yourself.

## Examples

```bash
# SLERP merge of two models
pmetal merge --models model-a model-b --method slerp --t 0.5

# TIES with sparsification
pmetal merge --models base ft-1 ft-2 --method ties --density 0.5

# DARE-TIES
pmetal merge --models model-a model-b --method dare_ties --density 0.7
```

## Advanced Features

- **GPU-Accelerated Merging**: Metal-based merge operations for large models
- **FP8-Aware Merging**: Merge with FP8 quantization for memory efficiency
- **Async Merge Pipeline**: Double-buffered streaming for models that exceed memory
- **LoRA Fusing**: Merge LoRA adapters into base weights (standard and accurate modes)

## See Also

- [pmetal merge](/cli/merge/) — CLI reference
- [pmetal fuse](/cli/fuse/) — LoRA fusing
