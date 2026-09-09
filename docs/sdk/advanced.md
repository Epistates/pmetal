# Advanced SDK Usage

Lower-level crate APIs for full control over training loops, models, and pipelines.

The `pmetal` crate re-exports every sub-crate, so `use pmetal::prelude::*;` is usually all you
need. Each sub-crate is also publishable on its own if you want a narrower dependency tree.

## Architecture

PMetal is a Rust workspace of 20 crates:

| Crate | Purpose |
|-------|---------|
| `pmetal` | CLI binary, TUI, inference runner, and the umbrella library |
| `pmetal-core` | Foundation: configs, traits, types, error handling, job specs |
| `pmetal-core-derive` | Derive macros for the core traits |
| `pmetal-bridge` | C++ FFI bridge to MLX (replaces mlx-rs) |
| `pmetal-metal` | Custom Metal GPU kernels + ANE runtime |
| `pmetal-mlx` | MLX backend integration, KV caches, quantization |
| `pmetal-models` | LLM architectures (Llama, Qwen, DeepSeek, Gemma, etc.) |
| `pmetal-lora` | LoRA/QLoRA training implementations |
| `pmetal-trainer` | Training loops (SFT, DPO, SimPO, ORPO, KTO, GRPO) |
| `pmetal-data` | Dataset loading, chat templates, tokenization, image processing |
| `pmetal-hub` | HuggingFace Hub integration + model fit estimation |
| `pmetal-distill` | Knowledge distillation (online, offline, TAID) |
| `pmetal-merge` | Model merging (14 strategies) |
| `pmetal-gguf` | GGUF format with imatrix quantization |
| `pmetal-mhc` | Manifold-Constrained Hyper-Connections |
| `pmetal-distributed` | Distributed training (mDNS, Ring All-Reduce) |
| `pmetal-vocoder` | BigVGAN neural vocoder |
| `pmetal-serve` | OpenAI- and Anthropic-compatible inference server |
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

For LoRA training, load through `DynamicLoraModel::from_pretrained`, which wraps the base model
with adapters:

```rust
let lora_config = pmetal::core::LoraConfig { r: 16, alpha: 32.0, ..Default::default() };
let model = DynamicLoraModel::from_pretrained(&model_dir, lora_config)?;
```

## Manual Training Loop

`TrainingLoop::new` takes a single `TrainingLoopConfig`; the model and dataset are passed to
`run_packed`. Use `train_step` directly if you want to own the iteration yourself.

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

let mut training_loop = TrainingLoop::new(loop_config);
training_loop.add_callback(Box::new(ProgressCallback::new(total_steps)));

let model = training_loop.run_packed(model, dataset, None, Some(&checkpoints))?;
model.save_lora_weights("./output/lora_weights.safetensors")?;

println!("final loss {:.4} over {} steps",
    training_loop.current_loss(), training_loop.current_step());
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

- [`crates/pmetal/examples/`](https://github.com/Epistates/pmetal/tree/main/crates/pmetal/examples/) — complete working examples, including `finetune_manual.rs`, which this page mirrors
- [Python SDK](/python/full-control/) — the same pipeline from Python
