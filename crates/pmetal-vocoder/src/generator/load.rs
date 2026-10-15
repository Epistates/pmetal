//! Loading and saving generator checkpoints.

use super::BigVGAN;
use crate::checkpoint;
use crate::config::BigVGANConfig;
use crate::error::{Result, VocoderError};
use pmetal_runtime::compat::Array;
use std::path::{Path, PathBuf};

/// Weight files to look for, in order: the reference release's name first.
const WEIGHT_FILES: [&str; 2] = ["bigvgan_generator.pt", "model.safetensors"];

impl BigVGAN {
    /// Load weights from a `.pt` (PyTorch) or `.safetensors` file into this
    /// model, strictly: the checkpoint must match the architecture exactly.
    pub fn load_weights(&mut self, path: &Path) -> Result<()> {
        self.load_state_dict(checkpoint::load_state_dict(path, None)?)
    }

    /// Load a model from a directory holding the reference release's
    /// `config.json` and `bigvgan_generator.pt` (or `model.safetensors`).
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let config_path = dir.join("config.json");
        let json = std::fs::read_to_string(&config_path)
            .map_err(|e| VocoderError::Config(format!("{}: {e}", config_path.display())))?;
        let config = BigVGANConfig::from_json(&json)?;
        let weights = WEIGHT_FILES
            .iter()
            .map(|f| dir.join(f))
            .find(|p| p.is_file())
            .ok_or_else(|| {
                VocoderError::WeightLoad(format!(
                    "{} holds none of {WEIGHT_FILES:?}",
                    dir.display()
                ))
            })?;
        let mut model = Self::new(config)?;
        model.load_weights(&weights)?;
        model.fuse_weight_norm();
        Ok(model)
    }

    /// Load a released model: `model_id` is a local directory (see
    /// [`from_dir`](Self::from_dir)) or a Hugging Face repository id such as
    /// `nvidia/bigvgan_v2_24khz_100band_256x`. Weight normalisation is folded
    /// into the weights, so the model is ready for inference.
    pub fn from_pretrained(model_id: &str) -> Result<Self> {
        let local = Path::new(model_id);
        if local.is_dir() {
            return Self::from_dir(local);
        }
        let client = hf_hub::HFClientSync::new().map_err(|e| VocoderError::Hub(e.to_string()))?;
        let (owner, name) = hf_hub::split_id(model_id);
        let repo = client.model(owner, name);
        let fetch = |filename: &str| -> Result<PathBuf> {
            repo.download_file()
                .filename(filename)
                .send()
                .map_err(|e| VocoderError::Hub(format!("{filename}: {e}")))
        };

        let config_path = fetch("config.json")?;
        let mut last_error = None;
        for file in WEIGHT_FILES {
            match fetch(file) {
                Ok(weights) => {
                    // Both files live in the same snapshot directory.
                    debug_assert_eq!(weights.parent(), config_path.parent());
                    return Self::from_dir(
                        config_path
                            .parent()
                            .ok_or_else(|| VocoderError::Hub("snapshot has no directory".into()))?,
                    );
                }
                Err(e) => last_error = Some(e),
            }
        }
        Err(last_error.unwrap_or_else(|| VocoderError::Hub("no weight file found".into())))
    }

    /// Save the model's tensors as safetensors in the reference's naming and
    /// layout (loadable by [`load_weights`](Self::load_weights)).
    pub fn save_safetensors(&self, path: &Path) -> Result<()> {
        let state = self.state_dict();
        let entries: Vec<(&str, &Array)> = state.iter().map(|(k, v)| (k.as_str(), v)).collect();
        let path = path
            .to_str()
            .ok_or_else(|| VocoderError::WeightLoad("non-UTF-8 path".into()))?;
        Array::save_safetensors(path, &entries).map_err(|e| VocoderError::WeightLoad(e.to_string()))
    }
}
