//! Fine-tune a model using manual sub-crate orchestration.
//!
//! This example mirrors the CLI's training flow, giving full control over
//! every component. For most users, the easy API (`finetune_easy.rs`) is simpler.
//!
//! ```sh
//! cargo run -p pmetal --example finetune_manual --features easy -- \
//!     --model ./path/to/model --dataset data.jsonl
//! ```

use std::env;
use std::path::PathBuf;

use pmetal::prelude::*;

type BoxResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[tokio::main]
async fn main() -> BoxResult<()> {
    tracing_subscriber::fmt::init();

    let args: Vec<String> = env::args().collect();
    let model_path = args
        .iter()
        .position(|a| a == "--model")
        .and_then(|i| args.get(i + 1))
        .expect("Usage: --model <path> --dataset <path>");
    let dataset_path = args
        .iter()
        .position(|a| a == "--dataset")
        .and_then(|i| args.get(i + 1))
        .expect("Usage: --model <path> --dataset <path>");

    let model_path = PathBuf::from(model_path);
    let output_dir = PathBuf::from("./output");

    // 1. Load tokenizer
    println!("Loading tokenizer...");
    let tokenizer = Tokenizer::from_model_dir(&model_path)?;

    // 2. Detect chat template
    let chat_template = pmetal::data::chat_templates::detect_chat_template(
        &model_path,
        &model_path.to_string_lossy(),
    );

    // 3. Load and tokenize dataset
    println!("Loading dataset...");
    let train_dataset = TrainingDataset::from_jsonl_tokenized(
        dataset_path,
        &tokenizer,
        DatasetFormat::Auto,
        2048,
        Some(&chat_template),
    )?;
    println!("Loaded {} samples", train_dataset.len());

    // 4. Create LoRA config
    let lora_config = pmetal::core::LoraConfig {
        r: 16,
        alpha: 32.0,
        ..Default::default()
    };

    // 5. Load model with LoRA adapters
    println!("Loading model with LoRA...");
    let model = DynamicLoraModel::from_pretrained(&model_path, lora_config)
        .map_err(|e| format!("Failed to load LoRA model: {e}"))?;

    // 6. Configure training
    let training_config = pmetal::core::TrainingConfig {
        learning_rate: 2e-4,
        batch_size: 4,
        num_epochs: 3,
        max_seq_len: 2048,
        output_dir: output_dir.to_string_lossy().to_string(),
        ..Default::default()
    };

    let dataloader_config = DataLoaderConfig {
        batch_size: 4,
        max_seq_len: 2048,
        shuffle: true,
        seed: 42,
        pad_token_id: tokenizer.pad_token_id().unwrap_or(0),
        drop_last: false,
    };

    let loop_config = TrainingLoopConfig {
        training: training_config,
        dataloader: dataloader_config,
        use_sequence_packing: true,
        ..Default::default()
    };

    // 7. Set up checkpoint manager
    std::fs::create_dir_all(&output_dir)?;
    let checkpoint_manager = CheckpointManager::new(output_dir.join("checkpoints"))
        .map_err(|e| format!("Failed to create checkpoint manager: {e}"))?
        .with_max_checkpoints(3);

    // 8. Run training
    println!("Starting training...");
    let mut training_loop = TrainingLoop::new(loop_config);
    let model = training_loop
        .run_packed(model, train_dataset, None, Some(&checkpoint_manager))
        .map_err(|e| format!("Training failed: {e}"))?;

    // 9. Save LoRA weights
    let weights_path = output_dir.join("lora_weights.safetensors");
    model
        .save_lora_weights(&weights_path)
        .map_err(|e| format!("Failed to save weights: {e}"))?;

    println!("\nTraining complete!");
    println!("  Final loss:   {:.4}", training_loop.current_loss());
    println!("  Total steps:  {}", training_loop.current_step());
    println!("  LoRA weights: {}", weights_path.display());

    Ok(())
}
