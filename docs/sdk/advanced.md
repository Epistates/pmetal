# Advanced SDK Usage

Lower-level crate APIs for full control over training loops, models, and pipelines.

The `pmetal` crate re-exports every sub-crate, so `use pmetal::prelude::*;` is usually all you
need. Each sub-crate is also publishable on its own if you want a narrower dependency tree.

## Architecture

PMetal is a Rust workspace of 21 crates, plus the desktop app (`pmetal-gui`, its own workspace):

| Crate | Purpose |
|-------|---------|
| `pmetal` | CLI binary, TUI, inference runner, and the umbrella library |
| `pmetal-core` | Foundation: configs, traits, types, error handling, the job specs |
| `pmetal-core-derive` | Derive macro for the job specs |
| `pmetal-runtime` | Tensor runtime: arrays, autodiff, native engines, optimizers, custom Metal kernels |
| `pmetal-metal` | Metal context, ANE runtime and kernels |
| `pmetal-cache` | KV caches, prefix reuse, speculative-decoding state |
| `pmetal-models` | LLM architectures, the dispatcher and speculative decoders |
| `pmetal-lora` | LoRA/QLoRA as a property of `Linear` (`AdaptedModel`, `DynamicLoraModel`) |
| `pmetal-trainer` | Training loops (SFT, preference, GRPO, MTP, ...) |
| `pmetal-data` | Dataset loading, chat templates, tokenization, image processing |
| `pmetal-hub` | Hugging Face Hub integration + model fit estimation |
| `pmetal-distill` | Knowledge distillation (losses, offline caches, TAID) |
| `pmetal-merge` | Model merging and LoRA fusing |
| `pmetal-gguf` | GGUF reader, writer and quantization |
| `pmetal-mhc` | Manifold-Constrained Hyper-Connections |
| `pmetal-distributed` | Distributed training (mDNS discovery, authenticated ring all-reduce) |
| `pmetal-vocoder` | BigVGAN neural vocoder |
| `pmetal-serve` | OpenAI- and Anthropic-compatible inference server |
| `pmetal-supervisor` | Runs `pmetal` commands as supervised child processes (TUI, MCP server, GUI, Python) |
| `pmetal-mcp` | MCP server for Claude Desktop and other MCP clients |
| `pmetal-py` | Python bindings (maturin/PyO3) |

## Model Loading

`DynamicModel::load` detects the architecture from `config.json` and dispatches to the right
implementation. It is synchronous and takes a **local directory**; resolve HuggingFace ids first
with `pmetal_hub::resolve_model_path`.

```rust
use pmetal::prelude::*;

// Local directory
let model = DynamicModel::load("./my-model/")?;

// HuggingFace id: download (or reuse the cache), then load
let model_dir = pmetal::hub::resolve_model_path("Qwen/Qwen3-0.6B", None, None).await?;
let model = DynamicModel::load(&model_dir)?;
```

For LoRA training, load through `DynamicLoraModel::from_pretrained`, which attaches adapters to the same model the dispatcher loads:

```rust
let lora_config = pmetal::core::LoraConfig { r: 16, alpha: 32.0, ..Default::default() };
let model = DynamicLoraModel::from_pretrained(&model_dir, lora_config.clone())?;
```

## Manual Training Loop

`TrainingLoop::new` takes a single `TrainingLoopConfig` and returns an error for a configuration
that cannot work; the model and dataset are passed to `run`, which packs sequences when
`use_sequence_packing` is set. For a complete working program, see
[`finetune_manual.rs`](https://github.com/Epistates/pmetal/blob/main/crates/pmetal/examples/finetune_manual.rs).

```rust
use pmetal::prelude::*;

let tokenizer = Tokenizer::from_model_dir(&model_dir)?;
let chat_template = pmetal::data::chat_templates::detect_chat_template(
    &model_dir,
    &model_dir.to_string_lossy(),
);

let dataset = TrainingDataset::from_jsonl_tokenized(
    "train.jsonl",
    &tokenizer,
    DatasetFormat::Auto,
    2048,
    Some(&chat_template),
    None,
)?;

let loop_config = TrainingLoopConfig {
    training: pmetal::core::TrainingConfig {
        learning_rate: 2e-4,
        batch_size: 1,
        num_epochs: 3,
        max_seq_len: 2048,
        output_dir: "./output".to_string(),
        ..Default::default()
    },
    dataloader: DataLoaderConfig {
        batch_size: 1,
        max_seq_len: 2048,
        pad_token_id: tokenizer.pad_token_id().unwrap_or(0),
        ..Default::default()
    },
    use_sequence_packing: true,
    ..Default::default()
};

let checkpoints = CheckpointManager::new("./output/checkpoints")?.with_max_checkpoints(3);

let mut training_loop = TrainingLoop::new(loop_config)?;
training_loop.add_callback(Box::new(LoggingCallback::new(10)));

let mut model = model;
training_loop.run(&mut model, dataset, None, Some(&checkpoints))?;

// A PEFT adapter directory: adapter_model.safetensors and adapter_config.json
let weights = pmetal::trainer::orchestrator::save_adapter(
    &model,
    std::path::Path::new("./output"),
    &lora_config,
    Some(&model_dir.to_string_lossy()),
)?;

println!("final loss {:.4} over {} steps, adapter at {}",
    training_loop.current_loss(), training_loop.current_step(), weights.display());
```

## Callback System

`TrainingCallback` lives in `pmetal-core` and is re-exported from the prelude. Every method has a
default, so implement only the hooks you need.

```rust
use pmetal::prelude::TrainingCallback;

struct MyCallback;

impl TrainingCallback for MyCallback {
    fn on_step_start(&mut self, _step: usize) { /* ... */ }
    fn on_step_end(&mut self, _step: usize, _loss: f64) { /* ... */ }
    fn should_stop(&self) -> bool { false }
}

training_loop.add_callback(Box::new(MyCallback));
```

`on_step_end_with_metrics` gives the same hook plus timing, throughput and learning rate; its
default implementation forwards to `on_step_end`. `ProgressCallback`, `LoggingCallback` and
`MetricsJsonCallback` are provided out of the box.

## See Also

- [`crates/pmetal/examples/`](https://github.com/Epistates/pmetal/tree/main/crates/pmetal/examples/): complete working examples, including `finetune_manual.rs`, which this page mirrors
- [Python SDK](/python/full-control/): the same pipeline from Python
