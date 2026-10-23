# Model Merging

Merge models with 18 strategies: SLERP, TIES, DARE, DELLA, Task Arithmetic, and more.

PMetal streams merges tensor by tensor from memory-mapped checkpoints (peak memory is a few tensors, not the models), merges in f32, and writes atomic, indexed safetensors shards. FP8 checkpoints with block scales, layer-sliced frankenmerges, filter and layer-gradient parameters, and reproducible stochastic masks are supported.

## CLI Strategies

`pmetal merge` merges two models (`--model-a`, `--model-b`); task-vector methods also take `--base`. The library and YAML configs merge any number.

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
| `ram` | RAM (Reinforced Agent Merging) |
| `ram_plus` | RAM+ variant |
| `passthrough` | Layer passthrough composition |

The command line, the terminal UI, the desktop app, the MCP server and the Python bindings all offer these sixteen: they read one list, the job spec's.

## Library-Only Strategies

Reachable by setting `MergeConfig.merge_method` directly (or `merge_method:` in a YAML config);
`pmetal merge --method` does not accept them.

| `merge_method` | Description |
|----------------|-------------|
| `fisher` | Fisher-weighted averaging; needs per-model `fisher_paths` |
| `regmean` | RegMean; needs per-model `gram_paths` |

## Examples

```bash
# SLERP merge of two models
pmetal merge --model-a model-a --model-b model-b --method slerp --t 0.5 --output merged

# TIES with sparsification, against the base both were fine-tuned from
pmetal merge --model-a ft-1 --model-b ft-2 --base base --method ties --density 0.5 --output merged

# DARE-TIES
pmetal merge --model-a ft-1 --model-b ft-2 --base base --method dare_ties --density 0.7 --output merged
```

## Advanced Features

- **Frankenmerges**: `slices` stack layer ranges of several models; the output `config.json` gets the new layer count
- **Per-tensor parameters**: `weight`, `density`, `t`, `lambda` accept `value`/`filter` lists and layer gradients (`t: [0, 0.5, 1]`)
- **Reproducibility**: `seed` fixes the DARE and DELLA masks; each tensor and model draws its own
- **FP8 checkpoints**: E4M3 weights with `weight_scale_inv` block scales are dequantized on load
- **Vocabulary mismatches**: `vocab_mismatch: truncate | extend`
- **LoRA Fusing**: PEFT and `pmetal train` adapters, rsLoRA, DoRA, MLX-quantized bases

## See Also

- [pmetal merge](/cli/merge/): CLI reference
- [pmetal fuse](/cli/fuse/): LoRA fusing
