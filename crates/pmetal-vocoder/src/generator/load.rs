//! Loading and saving generator checkpoints.

use super::BigVGAN;
use crate::checkpoint;
use crate::config::BigVGANConfig;
use crate::error::{Result, VocoderError};
use pmetal_runtime::compat::Array;
use std::path::Path;

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
        model.prepare_for_inference();
        Ok(model)
    }

    /// Load a released model: `model_id` is a local directory (see
    /// [`from_dir`](Self::from_dir)) or a Hugging Face repository id such as
    /// `nvidia/bigvgan_v2_24khz_100band_256x`. Weight normalisation is folded
    /// into the weights, so the model is ready for inference.
    ///
    /// The files come through `pmetal-hub`, so the cache, the token, the
    /// mirror and offline mode are the ones every other command uses, and the
    /// weights are fetched from the commit the config came from.
    pub async fn from_pretrained(model_id: &str) -> Result<Self> {
        Self::from_pretrained_with(model_id, &pmetal_hub::DownloadOptions::default()).await
    }

    /// [`from_pretrained`](Self::from_pretrained) with explicit connection
    /// settings (cache directory, offline mode, revision).
    pub async fn from_pretrained_with(
        model_id: &str,
        options: &pmetal_hub::DownloadOptions,
    ) -> Result<Self> {
        let local = Path::new(model_id);
        if local.is_dir() {
            return Self::from_dir(local);
        }
        let fail = |what: &str, e: &dyn std::fmt::Display| {
            VocoderError::Hub(format!("{model_id}: {what}: {e}"))
        };
        let config_path = pmetal_hub::download_file_with(model_id, "config.json", options)
            .await
            .map_err(|e| fail("config.json", &e))?;
        let snapshot = config_path
            .parent()
            .ok_or_else(|| VocoderError::Hub("the config has no snapshot directory".into()))?;
        // A snapshot directory is named for its commit: every other file
        // comes from that one, never from whatever the branch has moved to.
        let mut pinned = options.clone();
        if let Some(commit) = snapshot.file_name().and_then(|n| n.to_str()) {
            pinned.revision = Some(commit.to_owned());
        }
        let mut last_error = None;
        for file in WEIGHT_FILES {
            match pmetal_hub::download_file_with(model_id, file, &pinned).await {
                Ok(_) => return Self::from_dir(snapshot),
                Err(e) => last_error = Some(fail(file, &e)),
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
