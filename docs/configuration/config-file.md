# Configuration File

YAML configuration file format for PMetal training runs.

PMetal supports YAML configuration files as an alternative to CLI flags. Generate a sample config with `pmetal init`.

## Example

```yaml
model:
  model_id: "meta-llama/Llama-3.2-1B"

lora:
  r: 16
  alpha: 32.0

training:
  learning_rate: 2e-4
  batch_size: 1
  num_epochs: 3
  lr_scheduler: cosine
  warmup_steps: 100
  gradient_accumulation_steps: 4
  max_grad_norm: 1.0
  weight_decay: 0.01
  max_seq_len: 2048

dataset:
  dataset_id: "./train.jsonl"
```

## Usage

```bash
pmetal train --config training.yaml
```

CLI flags override config file values when both are specified.

## Strict loading

A key a section does not have is an error that names it and lists the keys the
section does have, so a misspelt `learning_rat` cannot leave the default in
force without telling you. A value the trainers cannot run is an error too:
every problem is listed with its dotted key (`training.learning_rate`), and so
are contradictions, such as setting both `warmup_steps` and `warmup_ratio`, or
setting `wsd_stable_ratio` under a schedule that is not `wsd`.

Options that are recognised but not available yet are refused by name rather
than run as something else: the `muon`, `adamw_schedule_free` and `adamw_8bit`
optimizers, LoRA `bias` other than `none`, and the `pissa` and `loftq` values
of `init_lora_weights`.

Files written by earlier releases still load. `epochs`, `lr_schedule`,
`cosine_num_cycles`, `name` (for `model_id`) and `path` (for `dataset_id`) are
accepted as spellings of the keys below. Keys that were accepted but never read
(`model.dtype`, `model.device`, `model.quantization`, `model.trust_remote_code`,
`model.revision`, `dataset.split`, `dataset.text_column`, `dataset.max_samples`,
`dataset.seed`) are dropped when they hold the value `pmetal init` wrote; set to
anything else they are an error, since they never did what they asked for.
`model.max_seq_len` is read as `training.max_seq_len` when the training section
leaves it out.

## Sections

### model

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_id` | string | *required* | HuggingFace ID or local path |

### lora

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `r` | integer | `16` | LoRA rank |
| `alpha` | float | `32.0` | LoRA scaling factor |
| `dropout` | float | `0.0` | Dropout on the adapter input, in [0, 1) |
| `target_modules` | list | `[]` (every linear layer) | Projections to adapt. Empty, or `all-linear`, adapts every linear layer (attention and MLP) but the output head, MoE routers and routed experts; name `lm_head` to adapt the head |
| `use_rslora` | bool | `false` | Scale by `alpha / sqrt(r)` |
| `use_dora` | bool | `false` | Weight-decomposed LoRA |
| `loraplus_lr_ratio` | float | unset | LoRA+ learning-rate ratio for the `B` matrices (the paper uses 16) |
| `init_lora_weights` | `true` or name | `true` | Initialisation of `A` and `B`: `true` (the standard one: `A` random, `B` zero), `pissa` (principal singular components), `olora` (QR factors), `loftq` (quantization-aware; needs `quantization`) |

### training

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `learning_rate` | float | `2e-4` | Peak learning rate |
| `embedding_learning_rate` | float | unset | Separate rate for embeddings |
| `batch_size` | integer | `1` | Micro-batch size |
| `gradient_accumulation_steps` | integer | `4` | Micro-batches per optimizer step |
| `num_epochs` | integer | `3` | Training epochs |
| `max_steps` | integer | unset | Stop after this many optimizer steps |
| `lr_scheduler` | string | `cosine` | `constant`, `linear`, `cosine`, `cosine_with_restarts`, `polynomial`, `wsd` |
| `warmup_steps` | integer | `0` | Warmup steps |
| `warmup_ratio` | float | unset | Warmup as a share of the run, in [0, 1); use instead of `warmup_steps` |
| `min_lr` | float | `0` | Floor the decay schedules stop at |
| `wsd_stable_ratio` | float | `0.7` | `wsd`: share of the post-warmup steps held at the peak |
| `wsd_decay_style` | string | `linear` | `wsd`: `linear`, `cosine` or `one_minus_sqrt` |
| `cosine_num_restarts` | integer | `1` | `cosine_with_restarts`: cosine cycles |
| `polynomial_power` | float | `1.0` | `polynomial`: exponent |
| `optimizer` | string | `adamw` | `adamw`, `sgd`, `lion`, `adafactor`, `muon`, `adamw_schedule_free`, `adamw_8bit` |
| `weight_decay` | float | `0.01` | Decoupled weight decay |
| `max_grad_norm` | float | `1.0` | Gradient clipping norm; 0 turns clipping off |
| `seed` | integer | `42` | Random seed |
| `logging_steps` | integer | `10` | Log every N steps |
| `eval_steps` | integer | unset | Evaluate every N steps |
| `save_steps` | integer | unset | Checkpoint every N steps |
| `output_dir` | string | `./output` | Where weights and checkpoints go |
| `use_packing` | bool | `true` | Pack short sequences |
| `max_seq_len` | integer | `8192` | Maximum sequence length; 0 takes it from the model |

### dataset

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dataset_id` | string | *required* | Dataset file path or HuggingFace ID |
| `shuffle` | bool | `true` | Shuffle each epoch |

## See Also

- [pmetal init](/cli/init/) — Generate a sample config
- [pmetal train](/cli/train/) — CLI training parameters
