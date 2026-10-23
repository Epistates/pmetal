# pmetal merge

Merge two models with one of 14 merge methods.

The command merges exactly two models at a time; task-vector methods take a base model with `--base`. For N-way merges and slice-based frankenmerging, build a `pmetal_merge::MergeConfig` and call `run_merge` directly: the command does not accept a merge config file. Merging is GPU-accelerated, FP8-aware, and streams large models. The merged model keeps the first model's tokenizer.

## Usage

```bash
pmetal merge \
  --model-a <MODEL_A> \
  --model-b <MODEL_B> \
  --output <OUTPUT> \
  [OPTIONS]
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `-a`, `--model-a` | required | First model: a path or Hub ID |
| `-b`, `--model-b` | required | Second model: a path or Hub ID |
| `-o`, `--output` | required | Output directory for the merged model |
| `--method` | `slerp` | Merge method (see Methods) |
| `--base` | none | Base model of the task-vector methods; required by `task_arithmetic`, `ties`, `dare_ties`, `dare_linear`, `della`, `della_linear`, `breadcrumbs`, `model_stock`, `consensus` and `nearswap` |
| `--t` | `0.5` | Interpolation parameter of `slerp`: `0` is model A, `1` is model B |
| `--weight-a` | `0.5` | Weight of model A in the `linear` and `ties` methods |
| `--weight-b` | `0.5` | Weight of model B in the `linear` and `ties` methods |
| `--density` | `0.5` | Fraction of parameters kept by the sparsifying methods (TIES, DARE) |
| `--dtype` | `bfloat16` | Output dtype: `float32`, `float16` or `bfloat16` |

## Methods

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
| `multislerp` | Weighted spherical mean of the models |
| `nearswap` | Per-parameter selection of the value nearest the base |
| `passthrough` | Layer passthrough composition |

More (`ram`, `ram_plus`, `fisher`, `regmean`, layer `slices`) are reachable by setting `MergeConfig.merge_method` or a YAML config directly.

## Examples

```bash
# SLERP merge
pmetal merge \
  --model-a Qwen/Qwen3-0.6B --model-b Qwen/Qwen3-0.6B-Base \
  --output ./merged \
  --method slerp --t 0.5

# TIES merge with sparsification
pmetal merge \
  --model-a ./ft-model-1 --model-b ./ft-model-2 \
  --base Qwen/Qwen3-0.6B-Base --output ./merged \
  --method ties --density 0.5

# DARE-TIES with random pruning
pmetal merge \
  --model-a ./ft-model-1 --model-b ./ft-model-2 \
  --base Qwen/Qwen3-0.6B-Base --output ./merged \
  --method dare_ties --density 0.7
```

## See Also

- [Model Merging](/models/merging/): Detailed merge documentation
- [pmetal fuse](/cli/fuse/): Fold a LoRA adapter into its base model
