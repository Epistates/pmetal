//! Expert weight layout for SSD-offloaded MoE inference.
//!
//! Defines the binary layout of packed expert weight files. Each layer's experts
//! are stored contiguously in a single file with fixed-size records, enabling
//! fast parallel `pread()` access.
//!
//! # File Format
//!
//! ```text
//! packed_experts/
//!   layout.json       — ExpertPackLayout metadata
//!   layer_00.bin      — 4-bit packed experts for layer 0
//!   layer_01.bin
//!   ...
//!   layer_59.bin
//! ```
//!
//! Within each layer file, expert `E` starts at byte offset `E * expert_size`.
//! Each expert contains gate/up/down projections with weights + scales + biases.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Quantization bit width for packed experts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PackedBits {
    /// 4-bit affine quantization (8 values per uint32).
    Four,
    /// 2-bit affine quantization (16 values per uint32).
    Two,
}

impl PackedBits {
    /// Bit width of each packed value.
    pub fn bit_width(self) -> u32 {
        match self {
            Self::Four => 4,
            Self::Two => 2,
        }
    }

    /// Values packed per uint32.
    pub fn pack_factor(self) -> usize {
        match self {
            Self::Four => 8,
            Self::Two => 16,
        }
    }
}

/// A single component (weight/scales/biases) within an expert.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertComponent {
    /// Byte offset from start of expert record.
    pub offset: usize,
    /// Size in bytes.
    pub size: usize,
    /// Shape of the component (e.g., `[out_dim, packed_cols]` for weights).
    pub shape: Vec<usize>,
}

/// Layout of a single expert's projections within the packed file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertRecord {
    /// Gate projection packed weights.
    pub gate_weight: ExpertComponent,
    /// Gate projection scales (bf16).
    pub gate_scales: ExpertComponent,
    /// Gate projection biases (bf16).
    pub gate_biases: ExpertComponent,
    /// Up projection packed weights.
    pub up_weight: ExpertComponent,
    /// Up projection scales (bf16).
    pub up_scales: ExpertComponent,
    /// Up projection biases (bf16).
    pub up_biases: ExpertComponent,
    /// Down projection packed weights.
    pub down_weight: ExpertComponent,
    /// Down projection scales (bf16).
    pub down_scales: ExpertComponent,
    /// Down projection biases (bf16).
    pub down_biases: ExpertComponent,
}

impl ExpertRecord {
    /// Total size of this expert record in bytes.
    pub fn total_size(&self) -> usize {
        self.down_biases.offset + self.down_biases.size
    }

    /// Compute layout for given dimensions and quantization.
    pub fn compute(
        hidden_dim: usize,
        intermediate_dim: usize,
        group_size: usize,
        bits: PackedBits,
    ) -> Self {
        let pf = bits.pack_factor();
        let num_groups_hidden = hidden_dim / group_size;
        let num_groups_intermediate = intermediate_dim / group_size;
        let packed_cols_hidden = hidden_dim / pf;
        let packed_cols_intermediate = intermediate_dim / pf;

        // Gate: [intermediate, hidden/pf] weights + [intermediate, hidden/gs] scales/biases
        let gate_w_size = intermediate_dim * packed_cols_hidden * 4; // uint32
        let gate_s_size = intermediate_dim * num_groups_hidden * 2; // bf16
        let gate_b_size = gate_s_size;

        // Up: same as gate
        let up_w_size = gate_w_size;
        let up_s_size = gate_s_size;
        let up_b_size = gate_b_size;

        // Down: [hidden, intermediate/pf] weights + [hidden, intermediate/gs] scales/biases
        let down_w_size = hidden_dim * packed_cols_intermediate * 4;
        let down_s_size = hidden_dim * num_groups_intermediate * 2;
        let down_b_size = down_s_size;

        let mut offset = 0;
        let mut next = |size: usize, shape: Vec<usize>| -> ExpertComponent {
            let comp = ExpertComponent {
                offset,
                size,
                shape,
            };
            offset += size;
            comp
        };

        ExpertRecord {
            gate_weight: next(gate_w_size, vec![intermediate_dim, packed_cols_hidden]),
            gate_scales: next(gate_s_size, vec![intermediate_dim, num_groups_hidden]),
            gate_biases: next(gate_b_size, vec![intermediate_dim, num_groups_hidden]),
            up_weight: next(up_w_size, vec![intermediate_dim, packed_cols_hidden]),
            up_scales: next(up_s_size, vec![intermediate_dim, num_groups_hidden]),
            up_biases: next(up_b_size, vec![intermediate_dim, num_groups_hidden]),
            down_weight: next(down_w_size, vec![hidden_dim, packed_cols_intermediate]),
            down_scales: next(down_s_size, vec![hidden_dim, num_groups_intermediate]),
            down_biases: next(down_b_size, vec![hidden_dim, num_groups_intermediate]),
        }
    }
}

/// Full layout metadata for a packed expert directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertPackLayout {
    /// Model name or identifier.
    pub model_name: String,
    /// Number of transformer layers with MoE.
    pub num_layers: usize,
    /// Number of routed experts per layer.
    pub num_experts: usize,
    /// Hidden dimension.
    pub hidden_dim: usize,
    /// MoE intermediate dimension.
    pub intermediate_dim: usize,
    /// Quantization group size.
    pub group_size: usize,
    /// Quantization bit width.
    pub bits: PackedBits,
    /// Byte size of each expert record within a layer file.
    pub expert_size: usize,
    /// Which layer indices have MoE (indices into transformer layers).
    pub moe_layer_indices: Vec<usize>,
    /// Expert record layout (same for all experts).
    pub record: ExpertRecord,
}

impl ExpertPackLayout {
    /// Compute layout from model config parameters.
    pub fn new(
        model_name: String,
        num_layers: usize,
        num_experts: usize,
        hidden_dim: usize,
        intermediate_dim: usize,
        group_size: usize,
        bits: PackedBits,
        moe_layer_indices: Vec<usize>,
    ) -> Self {
        let record = ExpertRecord::compute(hidden_dim, intermediate_dim, group_size, bits);
        let expert_size = record.total_size();

        Self {
            model_name,
            num_layers,
            num_experts,
            hidden_dim,
            intermediate_dim,
            group_size,
            bits,
            expert_size,
            moe_layer_indices,
            record,
        }
    }

    /// Get the byte offset of expert `expert_idx` within a layer file.
    pub fn expert_offset(&self, expert_idx: usize) -> usize {
        expert_idx * self.expert_size
    }

    /// Get the path to a layer's packed expert file.
    pub fn layer_file_path(&self, base_dir: &Path, layer_idx: usize) -> PathBuf {
        base_dir.join(format!("layer_{:02}.bin", layer_idx))
    }

    /// Save layout to JSON file.
    pub fn save(&self, base_dir: &Path) -> std::io::Result<()> {
        let path = base_dir.join("layout.json");
        let json = serde_json::to_string_pretty(self).map_err(std::io::Error::other)?;
        std::fs::write(path, json)
    }

    /// Load layout from JSON file.
    pub fn load(base_dir: &Path) -> std::io::Result<Self> {
        let path = base_dir.join("layout.json");
        let json = std::fs::read_to_string(path)?;
        serde_json::from_str(&json).map_err(std::io::Error::other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expert_record_qwen3_5_4bit() {
        // Qwen3.5-397B: hidden=4096, intermediate=1024, group_size=64, 4-bit
        // But actually moe_intermediate_size=256 for routed experts
        let record = ExpertRecord::compute(4096, 1024, 64, PackedBits::Four);

        // gate_weight: [1024, 4096/8] * 4 bytes = [1024, 512] * 4 = 2,097,152
        assert_eq!(record.gate_weight.size, 1024 * 512 * 4);
        // gate_scales: [1024, 4096/64] * 2 bytes = [1024, 64] * 2 = 131,072
        assert_eq!(record.gate_scales.size, 1024 * 64 * 2);

        // Verify flash-moe reference value for this config
        let total = record.total_size();
        assert_eq!(
            total, 7_077_888,
            "Expert size should match flash-moe's EXPERT_SIZE"
        );
    }

    #[test]
    fn test_expert_record_2bit() {
        let record = ExpertRecord::compute(4096, 1024, 64, PackedBits::Two);

        // gate_weight: [1024, 4096/16] * 4 bytes = [1024, 256] * 4 = 1,048,576
        assert_eq!(record.gate_weight.size, 1024 * 256 * 4);

        // 2-bit should be ~44% smaller than 4-bit
        let four_bit = ExpertRecord::compute(4096, 1024, 64, PackedBits::Four);
        let ratio = record.total_size() as f64 / four_bit.total_size() as f64;
        assert!(
            ratio < 0.6,
            "2-bit should be <60% of 4-bit size, got {:.1}%",
            ratio * 100.0
        );
    }

    #[test]
    fn test_layout_serialization() {
        let layout = ExpertPackLayout::new(
            "test-model".to_string(),
            28,
            512,
            4096,
            1024,
            64,
            PackedBits::Four,
            (4..28).collect(),
        );

        let json = serde_json::to_string(&layout).unwrap();
        let recovered: ExpertPackLayout = serde_json::from_str(&json).unwrap();
        assert_eq!(recovered.num_experts, 512);
        assert_eq!(recovered.expert_size, layout.expert_size);
    }
}
