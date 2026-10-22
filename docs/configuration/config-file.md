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

A command-line option that differs from its default overrides the file; an option left at its default (including one typed with the default's value) leaves the file's value in force. `--model` and `--dataset` override `model.model_id` and `dataset.dataset_id` when given.

## Strict loading

A key a section does not have is an error that names it and lists the keys the
section does have, so a misspelt `learning_rat` cannot leave the default in
force without telling you. A value the trainers cannot run is an error too:
every problem is listed with its dotted key (`training.learning_rate`), and so
are contradictions, such as setting both `warmup_steps` and `warmup_ratio`, or
setting `wsd_stable_ratio` under a schedule that is not `wsd`.

LoRA `bias` other than `none` is refused by name: training bias terms is not implemented.
`init_lora_weights: false` is refused too (a randomly initialised `B` starts the adapter away
from the base model); name an initialisation instead.

Files written by earlier releases still load. `epochs`, `lr_schedule`,
`cosine_num_cycles`, `name` (for `model_id`) and `path` (for `dataset_id`) are
accepted as spellings of the keys below. Keys that were accepted but never read
are retired: `model.dtype`, `model.device`, `model.quantization`,
`model.trust_remote_code`, `model.revision`, `model.hf_token`, `dataset.split`,
`dataset.text_column`, `dataset.max_samples` and `dataset.seed` are dropped when
they hold the value `pmetal init` wrote and are an error set to anything else,
since they never did what they asked for. `model.use_flash_attention`,
`dataset.format` and `training.gradient_checkpointing` are dropped with a note
whatever they hold (attention always uses the fused kernel where one applies, the
dataset format is detected, and gradient checkpointing is on by default;
`pmetal train --no-gradient-checkpointing` turns it off). `model.max_seq_len` is
read as `training.max_seq_len` when the training section leaves it out.

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
| `bias` | string | `none` | Only `none` is implemented |
| `init_lora_weights` | `true` or name | `true` | Initialisation of `A` and `B`: `true` or `standard` (`A` random, `B` zero), `pissa` (principal singular components), `olora` (QR factors), `loftq` (fitted to a quantizer; needs QLoRA, `pmetal train --quantization`). Config file only: there is no command-line option |

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
| `label_smoothing` | float | `0` | Label smoothing of the training loss, in [0, 1); evaluation reports the plain loss |
| `z_loss` | float | `0` | Weight of the z-loss (`weight * logsumexp^2` per token); 1e-4 is the paper's value |
| `neftune_noise_alpha` | float | unset | NEFTune noise magnitude added to the token embeddings (5 for small models, 15 for larger ones) |

### dataset

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dataset_id` | string | *required* | Dataset file path or HuggingFace ID |
| `shuffle` | bool | `true` | Shuffle each epoch |

## See Also

- [pmetal init](/cli/init/): Generate a sample config
- [pmetal train](/cli/train/): CLI training parameters
