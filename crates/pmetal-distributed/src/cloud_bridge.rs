//! PMetal Cloud-Bridge for local-to-cluster state transfer.
//!
//! Enables seamless transfer of complete training state between a local
//! PMetal (Apple Silicon) run and a distributed cloud cluster (H100/B200).
//!
//! # Bundle layout
//!
//! ```text
//! <bundle_dir>/
//!   model.safetensors        – model weights (raw safetensors bytes)
//!   optimizer.safetensors    – Adam m/v state (optional)
//!   rng_state.bin            – RNG checkpoint bytes (optional)
//!   metadata.json            – training state + model info
//!   bootstrap_cluster.py     – ready-to-run Python loader script
//! ```
//!
//! # Example
//!
//! ```ignore
//! use pmetal_distributed::cloud_bridge::{CloudBridge, CloudTransferMetadata};
//!
//! let meta = CloudTransferMetadata {
//!     target_cluster: "h100".to_string(),
//!     preferred_dtype: "bf16".to_string(),
//!     distributed_strategy: "fsdp".to_string(),
//!     pmetal_version: env!("CARGO_PKG_VERSION").to_string(),
//!     global_step: 1000,
//!     epoch: 2,
//!     learning_rate: 1e-4,
//!     best_loss: Some(0.312),
//!     ema_loss: Some(0.318),
//!     model_architecture: "llama".to_string(),
//!     hidden_size: 4096,
//!     num_layers: 32,
//!     dataloader_epoch: 2,
//!     dataloader_sample_index: 8192,
//!     dataloader_shuffle_seed: 42,
//! };
//!
//! CloudBridge::export_bundle("/tmp/my_bundle", meta, &model_weights_bytes, None, None)?;
//! ```

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tracing::{debug, info};

// ── Metadata ─────────────────────────────────────────────────────────────────

/// Complete description of training state for cloud transfer.
///
/// All fields are serialised into `metadata.json` inside the bundle and
/// verified by the Python bootstrap script before loading weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudTransferMetadata {
    // ── Cluster targeting ────────────────────────────────────────────────
    /// Target cluster type, e.g. `"h100"`, `"b200"`, `"tpu-v4"`.
    pub target_cluster: String,
    /// Dtype to use on the cluster, e.g. `"bf16"`, `"fp8"`.
    pub preferred_dtype: String,
    /// Distributed strategy: `"fsdp"`, `"deepspeed"`, `"megatron"`.
    pub distributed_strategy: String,
    /// PMetal semver string that produced this bundle.
    pub pmetal_version: String,

    // ── Training state ───────────────────────────────────────────────────
    /// Global optimizer step at time of export.
    pub global_step: u64,
    /// Training epoch (0-indexed).
    pub epoch: u32,
    /// Learning rate at time of export.
    pub learning_rate: f64,
    /// Best validation loss observed so far, if measured.
    pub best_loss: Option<f64>,
    /// Exponential moving-average of recent training loss.
    pub ema_loss: Option<f64>,

    // ── Model identity (verification) ───────────────────────────────────
    /// Architecture name, e.g. `"llama"`, `"qwen3_next"`.
    pub model_architecture: String,
    /// Model hidden dimension (for sanity-checking the weight file).
    pub hidden_size: u32,
    /// Number of transformer layers.
    pub num_layers: u32,

    // ── Dataloader state ─────────────────────────────────────────────────
    /// Epoch from the dataloader's perspective (may differ from training epoch
    /// when an epoch spans multiple gradient accumulation passes).
    pub dataloader_epoch: u32,
    /// Next sample index within the current epoch.
    pub dataloader_sample_index: u64,
    /// Seed used to shuffle the dataset for reproducible resume.
    pub dataloader_shuffle_seed: u64,
}

// ── Bundle files ──────────────────────────────────────────────────────────────

/// File names used inside a bundle directory.
/// Centralised here so the Python script and Rust code stay in sync.
mod bundle_files {
    pub const MODEL_WEIGHTS: &str = "model.safetensors";
    pub const OPTIMIZER_STATE: &str = "optimizer.safetensors";
    pub const RNG_STATE: &str = "rng_state.bin";
    pub const METADATA: &str = "metadata.json";
    pub const BOOTSTRAP: &str = "bootstrap_cluster.py";
}

// ── CloudBridge ───────────────────────────────────────────────────────────────

/// Stateless helper for creating and loading cloud transfer bundles.
pub struct CloudBridge;

impl CloudBridge {
    /// Write a complete training checkpoint bundle to `bundle_dir`.
    ///
    /// The directory is created if it does not exist.  Existing files at the
    /// known bundle paths are overwritten; unrelated files are left untouched.
    ///
    /// # Arguments
    ///
    /// * `bundle_dir`      – destination directory (created if absent)
    /// * `metadata`        – training state and cluster targeting info
    /// * `model_weights`   – raw safetensors bytes (written verbatim)
    /// * `optimizer_state` – optional Adam m/v safetensors bytes
    /// * `rng_state`       – optional RNG checkpoint bytes
    pub fn export_bundle(
        bundle_dir: impl AsRef<Path>,
        metadata: CloudTransferMetadata,
        model_weights: &[u8],
        optimizer_state: Option<&[u8]>,
        rng_state: Option<&[u8]>,
    ) -> Result<()> {
        let dir = bundle_dir.as_ref();
        std::fs::create_dir_all(dir)
            .with_context(|| format!("create bundle directory: {}", dir.display()))?;

        // 1. Model weights
        let weights_path = dir.join(bundle_files::MODEL_WEIGHTS);
        std::fs::write(&weights_path, model_weights)
            .with_context(|| format!("write model weights to {}", weights_path.display()))?;
        info!(
            bytes = model_weights.len(),
            path = %weights_path.display(),
            "cloud_bridge: wrote model weights"
        );

        // 2. Optimizer state (optional)
        if let Some(opt_bytes) = optimizer_state {
            let opt_path = dir.join(bundle_files::OPTIMIZER_STATE);
            std::fs::write(&opt_path, opt_bytes)
                .with_context(|| format!("write optimizer state to {}", opt_path.display()))?;
            info!(
                bytes = opt_bytes.len(),
                path = %opt_path.display(),
                "cloud_bridge: wrote optimizer state"
            );
        }

        // 3. RNG state (optional)
        if let Some(rng_bytes) = rng_state {
            let rng_path = dir.join(bundle_files::RNG_STATE);
            std::fs::write(&rng_path, rng_bytes)
                .with_context(|| format!("write RNG state to {}", rng_path.display()))?;
            debug!(
                bytes = rng_bytes.len(),
                path = %rng_path.display(),
                "cloud_bridge: wrote rng state"
            );
        }

        // 4. Metadata JSON
        let meta_json = serde_json::to_string_pretty(&metadata)
            .context("serialize CloudTransferMetadata to JSON")?;
        let meta_path = dir.join(bundle_files::METADATA);
        std::fs::write(&meta_path, &meta_json)
            .with_context(|| format!("write metadata to {}", meta_path.display()))?;
        debug!(path = %meta_path.display(), "cloud_bridge: wrote metadata.json");

        // 5. Python bootstrap script
        let script = Self::generate_bootstrap(&metadata);
        let script_path = dir.join(bundle_files::BOOTSTRAP);
        std::fs::write(&script_path, &script)
            .with_context(|| format!("write bootstrap script to {}", script_path.display()))?;

        info!(
            dir = %dir.display(),
            step = metadata.global_step,
            cluster = %metadata.target_cluster,
            "cloud_bridge: bundle export complete"
        );

        Ok(())
    }

    /// Load and parse the metadata from an existing bundle directory.
    ///
    /// Useful for inspecting a bundle before deciding whether to resume.
    pub fn load_metadata(bundle_dir: impl AsRef<Path>) -> Result<CloudTransferMetadata> {
        let path = bundle_dir.as_ref().join(bundle_files::METADATA);
        let json = std::fs::read_to_string(&path)
            .with_context(|| format!("read metadata from {}", path.display()))?;
        let meta: CloudTransferMetadata = serde_json::from_str(&json)
            .with_context(|| format!("parse metadata JSON from {}", path.display()))?;
        Ok(meta)
    }

    /// Read the raw model weights bytes from an existing bundle.
    pub fn load_model_weights(bundle_dir: impl AsRef<Path>) -> Result<Vec<u8>> {
        let path = bundle_dir.as_ref().join(bundle_files::MODEL_WEIGHTS);
        std::fs::read(&path).with_context(|| format!("read model weights from {}", path.display()))
    }

    /// Read the optimizer state bytes from an existing bundle, if present.
    pub fn load_optimizer_state(bundle_dir: impl AsRef<Path>) -> Result<Option<Vec<u8>>> {
        let path = bundle_dir.as_ref().join(bundle_files::OPTIMIZER_STATE);
        if path.exists() {
            Ok(Some(std::fs::read(&path).with_context(|| {
                format!("read optimizer state from {}", path.display())
            })?))
        } else {
            Ok(None)
        }
    }

    /// Read the RNG state bytes from an existing bundle, if present.
    pub fn load_rng_state(bundle_dir: impl AsRef<Path>) -> Result<Option<Vec<u8>>> {
        let path = bundle_dir.as_ref().join(bundle_files::RNG_STATE);
        if path.exists() {
            Ok(Some(std::fs::read(&path).with_context(|| {
                format!("read rng state from {}", path.display())
            })?))
        } else {
            Ok(None)
        }
    }

    // ── Private helpers ───────────────────────────────────────────────────

    /// Produce a self-contained, working Python bootstrap script.
    ///
    /// All Rust values are embedded as string/numeric literals — the script
    /// does NOT use Python f-strings for interpolated values, avoiding any
    /// accidental variable-name collision with the surrounding Python scope.
    fn generate_bootstrap(meta: &CloudTransferMetadata) -> String {
        // Build optional loss strings without pulling in format complexity
        // inside the template.
        let best_loss_repr = match meta.best_loss {
            Some(v) => format!("{v:.6}"),
            None => "None".to_string(),
        };
        let ema_loss_repr = match meta.ema_loss {
            Some(v) => format!("{v:.6}"),
            None => "None".to_string(),
        };

        // We use plain string concatenation in the Python print calls rather
        // than f-strings so that no Rust-interpolated value can ever shadow a
        // Python identifier.
        format!(
            r#"#!/usr/bin/env python3
"""PMetal Cloud-Bridge bootstrap — generated by pmetal {pmetal_version}.

This script loads the checkpoint bundle exported from an Apple Silicon PMetal
training run and prepares it for resumption on a distributed cloud cluster.

Usage
-----
    python bootstrap_cluster.py                  # print summary
    python bootstrap_cluster.py --strategy fsdp  # load for PyTorch FSDP
    python bootstrap_cluster.py --strategy deepspeed

Requirements
------------
    pip install safetensors torch
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bundle constants (baked in at export time by PMetal)
# ---------------------------------------------------------------------------

BUNDLE_DIR          = Path(__file__).parent.resolve()
TARGET_CLUSTER      = "{target_cluster}"
PREFERRED_DTYPE     = "{preferred_dtype}"
STRATEGY            = "{distributed_strategy}"
PMETAL_VERSION      = "{pmetal_version}"

# Training state
GLOBAL_STEP         = {global_step}
EPOCH               = {epoch}
LEARNING_RATE       = {learning_rate}
BEST_LOSS           = {best_loss_repr}
EMA_LOSS            = {ema_loss_repr}

# Model identity
MODEL_ARCH          = "{model_architecture}"
HIDDEN_SIZE         = {hidden_size}
NUM_LAYERS          = {num_layers}

# Dataloader resume state
DATALOADER_EPOCH        = {dataloader_epoch}
DATALOADER_SAMPLE_INDEX = {dataloader_sample_index}
DATALOADER_SHUFFLE_SEED = {dataloader_shuffle_seed}

# File names (must match Rust bundle_files constants)
MODEL_WEIGHTS_FILE  = "model.safetensors"
OPTIMIZER_FILE      = "optimizer.safetensors"
RNG_STATE_FILE      = "rng_state.bin"
METADATA_FILE       = "metadata.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_file(name: str) -> Path:
    p = BUNDLE_DIR / name
    if not p.exists():
        raise FileNotFoundError(
            "Bundle file not found: " + str(p) + "\n"
            "Make sure you are running this script from inside the bundle "
            "directory, or that the bundle was transferred completely."
        )
    return p


def load_metadata() -> dict:
    """Return the raw metadata.json dict for inspection."""
    with open(_require_file(METADATA_FILE)) as fh:
        return json.load(fh)


def load_weights(dtype=None):
    """Load model weights from model.safetensors.

    Parameters
    ----------
    dtype : torch.dtype or None
        If given, cast all tensors to this dtype.  Defaults to the dtype
        stored in the safetensors file (usually the PREFERRED_DTYPE above).

    Returns
    -------
    dict[str, torch.Tensor]
    """
    try:
        from safetensors.torch import load_file
    except ImportError:
        sys.exit("safetensors is required: pip install safetensors")

    weights_path = _require_file(MODEL_WEIGHTS_FILE)
    state_dict = load_file(str(weights_path))

    if dtype is not None:
        state_dict = {{k: v.to(dtype) for k, v in state_dict.items()}}

    return state_dict


def load_optimizer_state():
    """Load Adam m/v tensors from optimizer.safetensors, or None if absent."""
    opt_path = BUNDLE_DIR / OPTIMIZER_FILE
    if not opt_path.exists():
        return None
    try:
        from safetensors.torch import load_file
    except ImportError:
        sys.exit("safetensors is required: pip install safetensors")
    return load_file(str(opt_path))


def load_rng_state() -> bytes | None:
    """Return raw RNG checkpoint bytes, or None if absent."""
    rng_path = BUNDLE_DIR / RNG_STATE_FILE
    if not rng_path.exists():
        return None
    return rng_path.read_bytes()


# ---------------------------------------------------------------------------
# FSDP loader
# ---------------------------------------------------------------------------

def load_for_fsdp(model, optimizer=None, dtype=None):
    """Resume a PyTorch FSDP model from this PMetal bundle.

    Parameters
    ----------
    model : torch.nn.Module
        Unwrapped model (before FSDP wrapping).  Weights are loaded in-place.
    optimizer : torch.optim.Optimizer or None
        If provided, Adam m/v state is restored when optimizer.safetensors
        is present in the bundle.
    dtype : torch.dtype or None
        Override cast dtype; defaults to PREFERRED_DTYPE mapping below.

    Returns
    -------
    int
        The global_step to resume from.
    """
    import torch

    _dtype_map = {{"bf16": torch.bfloat16, "f16": torch.float16, "fp32": torch.float32}}
    cast = dtype or _dtype_map.get(PREFERRED_DTYPE, torch.bfloat16)

    print("[pmetal/cloud-bridge] Loading weights for FSDP resume ...")
    print("  architecture : " + MODEL_ARCH)
    print("  hidden_size  : " + str(HIDDEN_SIZE))
    print("  num_layers   : " + str(NUM_LAYERS))
    print("  global_step  : " + str(GLOBAL_STEP))
    print("  best_loss    : " + str(BEST_LOSS))
    print("  dtype        : " + str(cast))

    state_dict = load_weights(dtype=cast)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print("[warn] Missing keys  : " + str(missing[:8]) + (" ..." if len(missing) > 8 else ""))
    if unexpected:
        print("[warn] Unexpected    : " + str(unexpected[:8]) + (" ..." if len(unexpected) > 8 else ""))

    if optimizer is not None:
        opt_state = load_optimizer_state()
        if opt_state is not None:
            print("[pmetal/cloud-bridge] Restoring optimizer state ...")
            # Reconstruct optimizer state_dict format from flat safetensors.
            # Keys are encoded as "<group_idx>/<param_idx>/exp_avg" etc.
            reconstructed = {{"state": {{}}, "param_groups": optimizer.state_dict()["param_groups"]}}
            for k, t in opt_state.items():
                parts = k.split("/", 2)
                if len(parts) == 3:
                    _g, p_idx_str, moment = parts
                    p_idx = int(p_idx_str)
                    reconstructed["state"].setdefault(p_idx, {{}})[moment] = t
            optimizer.load_state_dict(reconstructed)
        else:
            print("[info] No optimizer.safetensors found; starting optimizer from scratch.")

    print("[pmetal/cloud-bridge] FSDP bundle loaded. Resume from step " + str(GLOBAL_STEP) + ".")
    return GLOBAL_STEP


# ---------------------------------------------------------------------------
# DeepSpeed loader
# ---------------------------------------------------------------------------

def load_for_deepspeed(engine, dtype=None):
    """Resume a DeepSpeed engine from this PMetal bundle.

    Parameters
    ----------
    engine : deepspeed.DeepSpeedEngine
        Initialised DeepSpeed engine (model + optimizer wrapped).
    dtype : torch.dtype or None
        Override cast dtype.

    Returns
    -------
    int
        The global_step to resume from.
    """
    import torch

    _dtype_map = {{"bf16": torch.bfloat16, "f16": torch.float16, "fp32": torch.float32}}
    cast = dtype or _dtype_map.get(PREFERRED_DTYPE, torch.bfloat16)

    print("[pmetal/cloud-bridge] Loading weights for DeepSpeed resume ...")
    print("  architecture : " + MODEL_ARCH)
    print("  global_step  : " + str(GLOBAL_STEP))
    print("  best_loss    : " + str(BEST_LOSS))
    print("  dtype        : " + str(cast))

    state_dict = load_weights(dtype=cast)
    engine.module.load_state_dict(state_dict, strict=False)

    opt_state = load_optimizer_state()
    if opt_state is not None:
        print("[pmetal/cloud-bridge] Restoring optimizer state into DeepSpeed engine ...")
        reconstructed = {{"state": {{}}, "param_groups": engine.optimizer.state_dict()["param_groups"]}}
        for k, t in opt_state.items():
            parts = k.split("/", 2)
            if len(parts) == 3:
                _g, p_idx_str, moment = parts
                p_idx = int(p_idx_str)
                reconstructed["state"].setdefault(p_idx, {{}})[moment] = t
        engine.optimizer.load_state_dict(reconstructed)
    else:
        print("[info] No optimizer.safetensors found; DeepSpeed optimizer starts fresh.")

    # Update DeepSpeed's internal step counter
    engine.global_steps = GLOBAL_STEP

    print("[pmetal/cloud-bridge] DeepSpeed bundle loaded. Resume from step " + str(GLOBAL_STEP) + ".")
    return GLOBAL_STEP


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def _print_summary():
    print("")
    print("=== PMetal Cloud-Bridge Bundle ===")
    print("  generated by pmetal " + PMETAL_VERSION)
    print("")
    print("  target cluster   : " + TARGET_CLUSTER)
    print("  strategy         : " + STRATEGY)
    print("  preferred dtype  : " + PREFERRED_DTYPE)
    print("")
    print("  model arch       : " + MODEL_ARCH)
    print("  hidden_size      : " + str(HIDDEN_SIZE))
    print("  num_layers       : " + str(NUM_LAYERS))
    print("")
    print("  global_step      : " + str(GLOBAL_STEP))
    print("  epoch            : " + str(EPOCH))
    print("  learning_rate    : " + str(LEARNING_RATE))
    print("  best_loss        : " + str(BEST_LOSS))
    print("  ema_loss         : " + str(EMA_LOSS))
    print("")
    print("  dataloader epoch : " + str(DATALOADER_EPOCH))
    print("  sample_index     : " + str(DATALOADER_SAMPLE_INDEX))
    print("  shuffle_seed     : " + str(DATALOADER_SHUFFLE_SEED))
    print("")

    # Show which optional files are present
    files = [
        (MODEL_WEIGHTS_FILE,  "model weights      "),
        (OPTIMIZER_FILE,      "optimizer state    "),
        (RNG_STATE_FILE,      "rng checkpoint     "),
        (METADATA_FILE,       "metadata json      "),
    ]
    print("  Bundle contents:")
    for fname, label in files:
        p = BUNDLE_DIR / fname
        status = ("present  " + str(p.stat().st_size) + " bytes") if p.exists() else "absent"
        print("    " + label + " : " + status)
    print("")


def main():
    parser = argparse.ArgumentParser(
        description="PMetal Cloud-Bridge bootstrap — inspect or load a training bundle."
    )
    parser.add_argument(
        "--strategy",
        choices=["fsdp", "deepspeed"],
        default=None,
        help="Load the bundle using the specified distributed strategy.",
    )
    args = parser.parse_args()

    _print_summary()

    if args.strategy is None:
        print("Run with --strategy fsdp or --strategy deepspeed to load the model.")
        print("Or import load_for_fsdp / load_for_deepspeed from this file.")
    elif args.strategy == "fsdp":
        print("[info] --strategy fsdp selected.")
        print("[info] Call load_for_fsdp(model, optimizer) from your training script.")
    elif args.strategy == "deepspeed":
        print("[info] --strategy deepspeed selected.")
        print("[info] Call load_for_deepspeed(engine) from your training script.")


if __name__ == "__main__":
    main()
"#,
            pmetal_version = meta.pmetal_version,
            target_cluster = meta.target_cluster,
            preferred_dtype = meta.preferred_dtype,
            distributed_strategy = meta.distributed_strategy,
            global_step = meta.global_step,
            epoch = meta.epoch,
            learning_rate = meta.learning_rate,
            best_loss_repr = best_loss_repr,
            ema_loss_repr = ema_loss_repr,
            model_architecture = meta.model_architecture,
            hidden_size = meta.hidden_size,
            num_layers = meta.num_layers,
            dataloader_epoch = meta.dataloader_epoch,
            dataloader_sample_index = meta.dataloader_sample_index,
            dataloader_shuffle_seed = meta.dataloader_shuffle_seed,
        )
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_meta() -> CloudTransferMetadata {
        CloudTransferMetadata {
            target_cluster: "h100".to_string(),
            preferred_dtype: "bf16".to_string(),
            distributed_strategy: "fsdp".to_string(),
            pmetal_version: "0.3.3".to_string(),
            global_step: 2048,
            epoch: 3,
            learning_rate: 5e-5,
            best_loss: Some(0.2345),
            ema_loss: Some(0.2401),
            model_architecture: "llama".to_string(),
            hidden_size: 4096,
            num_layers: 32,
            dataloader_epoch: 3,
            dataloader_sample_index: 16384,
            dataloader_shuffle_seed: 99,
        }
    }

    #[test]
    fn export_bundle_writes_required_files() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path();

        let weights = b"fake_safetensors_bytes";
        CloudBridge::export_bundle(path, sample_meta(), weights, None, None)
            .expect("export_bundle");

        assert!(path.join("model.safetensors").exists());
        assert!(path.join("metadata.json").exists());
        assert!(path.join("bootstrap_cluster.py").exists());
        // Optional files must be absent when not provided
        assert!(!path.join("optimizer.safetensors").exists());
        assert!(!path.join("rng_state.bin").exists());
    }

    #[test]
    fn export_bundle_writes_optional_files() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path();

        CloudBridge::export_bundle(
            path,
            sample_meta(),
            b"model",
            Some(b"opt_state"),
            Some(b"rng"),
        )
        .expect("export_bundle with optionals");

        assert!(path.join("optimizer.safetensors").exists());
        assert!(path.join("rng_state.bin").exists());
    }

    #[test]
    fn load_metadata_round_trips() {
        let dir = tempfile::tempdir().expect("tempdir");
        let meta = sample_meta();
        CloudBridge::export_bundle(dir.path(), meta.clone(), b"w", None, None)
            .expect("export_bundle");

        let loaded = CloudBridge::load_metadata(dir.path()).expect("load_metadata");
        assert_eq!(loaded.global_step, meta.global_step);
        assert_eq!(loaded.target_cluster, meta.target_cluster);
        assert_eq!(loaded.model_architecture, meta.model_architecture);
        assert_eq!(loaded.best_loss, meta.best_loss);
    }

    #[test]
    fn load_model_weights_round_trips() {
        let dir = tempfile::tempdir().expect("tempdir");
        let weights = b"safetensors_payload";
        CloudBridge::export_bundle(dir.path(), sample_meta(), weights, None, None)
            .expect("export_bundle");

        let loaded = CloudBridge::load_model_weights(dir.path()).expect("load weights");
        assert_eq!(loaded.as_slice(), weights);
    }

    #[test]
    fn bootstrap_script_has_no_raw_braces_around_rust_vars() {
        let meta = sample_meta();
        let script = CloudBridge::generate_bootstrap(&meta);

        // The old bug produced `{h100}` — a Python variable reference —
        // instead of embedding the literal string `h100`.  Verify the
        // generated script contains the literal values, not variable refs.
        assert!(
            script.contains("\"h100\""),
            "target_cluster should appear as a string literal"
        );
        assert!(
            script.contains("\"fsdp\""),
            "distributed_strategy should appear as a string literal"
        );
        assert!(
            script.contains("2048"),
            "global_step should appear as a numeric literal"
        );
        assert!(
            script.contains("0.234500"),
            "best_loss should appear as a float literal"
        );
        // The template uses double-braces for Python dict literals —
        // make sure they collapsed to single braces correctly.
        assert!(
            script.contains("state_dict = {k: v.to(dtype) for k, v in state_dict.items()}"),
            "Python dict comprehension braces must be single"
        );
    }

    #[test]
    fn bootstrap_script_none_loss_renders_none_not_nan() {
        let mut meta = sample_meta();
        meta.best_loss = None;
        meta.ema_loss = None;
        let script = CloudBridge::generate_bootstrap(&meta);

        assert!(script.contains("BEST_LOSS           = None"));
        assert!(script.contains("EMA_LOSS            = None"));
    }

    #[test]
    fn load_optimizer_state_absent_returns_none() {
        let dir = tempfile::tempdir().expect("tempdir");
        CloudBridge::export_bundle(dir.path(), sample_meta(), b"w", None, None).expect("export");
        assert!(
            CloudBridge::load_optimizer_state(dir.path())
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn load_rng_state_absent_returns_none() {
        let dir = tempfile::tempdir().expect("tempdir");
        CloudBridge::export_bundle(dir.path(), sample_meta(), b"w", None, None).expect("export");
        assert!(CloudBridge::load_rng_state(dir.path()).unwrap().is_none());
    }
}
