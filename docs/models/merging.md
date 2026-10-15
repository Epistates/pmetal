# Model Merging

Merge models with 17 strategies — SLERP, TIES, DARE, DELLA, Task Arithmetic, and more.

PMetal streams merges tensor by tensor from memory-mapped checkpoints (peak memory is a few tensors, not the models), merges in f32, and writes atomic, indexed safetensors shards. FP8 checkpoints with block scales, layer-sliced frankenmerges, filter and layer-gradient parameters, and reproducible stochastic masks are supported.

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
| `model_stock` | Interpolation toward the pre-trained weights by task-vector angle |
| `consensus` | TALL-mask consensus merging |
| `multislerp` | Weighted spherical (Karcher) mean of 3+ models |
| `nearswap` | Per-parameter selection of the value nearest the base |
| `passthrough` | Layer passthrough composition |

## Library-Only Strategies

Reachable by setting `MergeConfig.merge_method` directly (or `merge_method:` in a YAML config);
`pmetal merge --method` does not accept them.

| `merge_method` | Description |
|----------------|-------------|
| `ram` | RAM (Reinforced Agent Merging) |
| `ram_plus` | RAM+ variant |
| `fisher` | Fisher-weighted averaging; needs per-model `fisher_paths` |
| `regmean` | RegMean; needs per-model `gram_paths` |

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

- **Frankenmerges**: `slices` stack layer ranges of several models; the output `config.json` gets the new layer count
- **Per-tensor parameters**: `weight`, `density`, `t`, `lambda` accept `value`/`filter` lists and layer gradients (`t: [0, 0.5, 1]`)
- **Reproducibility**: `seed` fixes the DARE and DELLA masks; each tensor and model draws its own
- **FP8 checkpoints**: E4M3 weights with `weight_scale_inv` block scales are dequantized on load
- **Vocabulary mismatches**: `vocab_mismatch: truncate | extend`
- **LoRA Fusing**: PEFT and `pmetal train` adapters, rsLoRA, DoRA, MLX-quantized bases

## See Also

- [pmetal merge](/cli/merge/) — CLI reference
- [pmetal fuse](/cli/fuse/) — LoRA fusing
