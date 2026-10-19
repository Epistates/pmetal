//! `pmetal init`: a sample training configuration to start from.

use pmetal_core::{DatasetConfig, LoraConfig, ModelConfig, TrainingConfig};
use pmetal_trainer::orchestrator::FullTrainingConfig;

/// Generate a sample configuration file.
pub(crate) fn generate_sample_config(output: &str) -> anyhow::Result<()> {
    let config = FullTrainingConfig {
        model: ModelConfig {
            model_id: "Qwen/Qwen3-0.6B".to_string(),
        },
        lora: LoraConfig {
            r: 16,
            alpha: 32.0,
            ..Default::default()
        },
        training: TrainingConfig {
            learning_rate: 2e-4,
            batch_size: 4,
            num_epochs: 3,
            warmup_steps: 100,
            max_seq_len: 2048,
            output_dir: "./output".to_string(),
            logging_steps: 10,
            save_steps: Some(500),
            ..Default::default()
        },
        dataset: DatasetConfig {
            dataset_id: "./data/train.jsonl".to_string(),
            shuffle: true,
        },
    };

    let yaml = serde_saphyr::to_string(&config)?;
    std::fs::write(output, yaml)?;

    println!("Sample configuration written to: {}", output);
    println!("\nYou can edit this file and run training with:");
    println!("  pmetal train --config {}", output);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_sample_config_pmetal_init_writes_loads_back_unchanged() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("sample.yaml");
        generate_sample_config(path.to_str().unwrap()).expect("write sample");
        let loaded = pmetal_core::FullTrainingConfig::from_yaml_file(&path).expect("load sample");
        assert!(loaded.notes.is_empty(), "{:?}", loaded.notes);
        assert_eq!(loaded.config.model.model_id, "Qwen/Qwen3-0.6B");
        assert_eq!(loaded.config.training.num_epochs, 3);
        assert_eq!(loaded.config.training.save_steps, Some(500));
    }
}
