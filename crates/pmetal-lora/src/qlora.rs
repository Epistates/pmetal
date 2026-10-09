//! QLoRA: low-rank adapters over a frozen base whose projections are packed.
//!
//! QLoRA (Dettmers et al., 2023, arXiv 2305.14314) is LoRA with the base
//! weights stored in four bits and dequantized to the compute dtype inside
//! the forward pass, the adapters staying in full precision. Here that is two
//! properties of the same [`Linear`](pmetal_bridge::compat::Linear): its weight
//! is packed ([`quantize_base`]) and it carries an adapter
//! ([`crate::AdaptedModel`]). Nothing per-architecture is involved, so every
//! architecture the dispatcher builds can be fine-tuned this way, through the
//! same forward pass it is served with.
//!
//! # Schemes
//!
//! Measured on Qwen3-0.6B (bf16, 24 conversations of up to 512 tokens, mean
//! cross-entropy 2.604 unquantized), every projection but the LM head packed:
//!
//! | scheme | storage | bits/weight | loss |
//! |---|---|---|---|
//! | `nf4` | NF4 codes, bf16 absmax per 64 | 4.25 | 2.648 |
//! | `nf4` + double quant | absmax packed to 8 bits | 4.13 | 2.647 |
//! | `fp4` | NVFP4: E2M1, E4M3 scale per 16, FP32 per tensor | 4.5 | 2.660 |
//! | `int8` | 8-bit affine per 64 | 8.5 | 2.601 |
//!
//! NF4 is the paper's data type: sixteen levels at the quantiles of a normal
//! distribution, which is how pretrained weights are distributed (§3). It
//! beat MLX's own 4-bit affine format here at the same 4.5 bits (2.684), as the
//! paper found against Int4. MLX has no NF4 kernel, so an NF4 layer unpacks to
//! the compute dtype for its matmul, which is what the paper's CUDA kernels do
//! too; `fp4` and `int8` run MLX's fused quantized matmul. Without NVFP4's
//! per-tensor scale the E4M3 block scales of a typical weight are subnormal and
//! the loss is 2.703.

use pmetal_bridge::QuantizedMode;
use pmetal_bridge::compat::{Linear, ModuleParameters, VisitLinears};
use pmetal_bridge::native_weight::QuantParams;

use super::LoraError;

/// NF4's sixteen values, from Dettmers et al. (2023), appendix E.
pub const NF4_CODEBOOK: [f32; 16] = [
    -1.0,
    -0.696_192_8,
    -0.525_073_05,
    -0.394_917_5,
    -0.284_441_38,
    -0.184_773_43,
    -0.091_050_036,
    0.0,
    0.079_580_3,
    0.160_930_2,
    0.246_112_3,
    0.337_915_24,
    0.440_709_83,
    0.562_617,
    0.722_956_84,
    1.0,
];

/// How a QLoRA base weight is stored. See the [module docs](self) for the
/// measurements behind each.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QLoraScheme {
    /// 4-bit NormalFloat over groups of `group_size`, the QLoRA paper's
    /// data type.
    #[default]
    Nf4,
    /// NVFP4: E2M1 values, an E4M3 scale per 16, an FP32 scale per tensor.
    Fp4,
    /// 8-bit affine over groups of `group_size`.
    Int8,
}

impl std::fmt::Display for QLoraScheme {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Nf4 => "nf4",
            Self::Fp4 => "fp4",
            Self::Int8 => "int8",
        })
    }
}

/// How to pack a model's base for QLoRA.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QLoraConfig {
    pub scheme: QLoraScheme,
    /// Weights per scale. `nf4` and `int8` take 32, 64 or 128; NVFP4's block
    /// is 16 by definition, so `fp4` ignores it.
    pub group_size: i32,
    /// Pack `nf4`'s absmax values to 8 bits as well, the paper's double
    /// quantization.
    pub double_quant: bool,
}

impl Default for QLoraConfig {
    /// NF4 over groups of 64, the paper's block size.
    fn default() -> Self {
        Self {
            scheme: QLoraScheme::Nf4,
            group_size: 64,
            double_quant: false,
        }
    }
}

impl QLoraConfig {
    /// An error naming what is wrong with the config, or `Ok`.
    pub fn validate(&self) -> Result<(), LoraError> {
        if self.scheme != QLoraScheme::Fp4 && ![32, 64, 128].contains(&self.group_size) {
            return Err(LoraError::InvalidState(format!(
                "{} takes a group size of 32, 64 or 128, not {}",
                self.scheme, self.group_size
            )));
        }
        if self.double_quant && self.scheme != QLoraScheme::Nf4 {
            return Err(LoraError::InvalidState(format!(
                "double quantization packs nf4's absmax values; {} has none",
                self.scheme
            )));
        }
        Ok(())
    }

    /// The group the layer's input width has to divide into.
    fn group(&self) -> i32 {
        match self.scheme {
            QLoraScheme::Fp4 => 16,
            _ => self.group_size,
        }
    }

    fn pack(&self, linear: &mut Linear) -> Result<(), pmetal_bridge::compat::Exception> {
        match self.scheme {
            QLoraScheme::Nf4 => {
                linear.quantize_codebook(&NF4_CODEBOOK, self.group_size, self.double_quant)
            }
            QLoraScheme::Fp4 => linear.quantize(QuantParams {
                group_size: 16,
                bits: 4,
                mode: QuantizedMode::Nvfp4,
            }),
            QLoraScheme::Int8 => linear.quantize(QuantParams {
                group_size: self.group_size,
                bits: 8,
                mode: QuantizedMode::Affine,
            }),
        }
    }
}

/// What [`quantize_base`] did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PackedBase {
    /// Projections now packed.
    pub packed: usize,
    /// Projections left as they were, by path, with the reason.
    pub kept: Vec<(String, &'static str)>,
    /// Bytes the packed projections occupied before.
    pub dense_bytes: usize,
    /// Bytes they occupy now, scales included.
    pub packed_bytes: usize,
}

/// Why the projection at `path` keeps its weight as loaded, if it does.
///
/// The LM head stays in the compute dtype, as the paper's (and the reference
/// implementation's) QLoRA leaves it. A MoE router stays, because its argmax
/// picks the experts and is the one place four bits of noise change which
/// computation runs, and routed experts stay as every per-architecture QLoRA
/// model this replaced left them; see [`crate::adapted::moe_role`].
fn kept_because(path: &str) -> Option<&'static str> {
    if path.rsplit('.').next() == Some("lm_head") {
        return Some("the LM head stays in the compute dtype");
    }
    crate::adapted::moe_role(path)
}

/// Pack every projection of `model` that QLoRA packs, as `config` says.
///
/// Projections already packed (a checkpoint saved quantized) are left as
/// they are, as are those [`kept_because`] names and any whose input width
/// doesn't divide into the scheme's groups.
pub fn quantize_base(
    model: &mut (impl VisitLinears + ?Sized),
    config: &QLoraConfig,
) -> Result<PackedBase, LoraError> {
    config.validate()?;
    let group = config.group();
    let mut report = PackedBase::default();
    let mut failure = None;
    model.visit_linears_mut("", &mut |path, linear| {
        if failure.is_some() {
            return;
        }
        if linear.quant.is_some() {
            report.kept.push((path.to_string(), "already packed"));
            return;
        }
        // An FP8 checkpoint's scale can sit outside the layer (Nemotron-H's
        // `weight_scale`), where packing would never see it.
        if linear.weight.value.dtype() == pmetal_bridge::compat::Dtype::Uint8 {
            report
                .kept
                .push((path.to_string(), "FP8 weights stay as loaded"));
            return;
        }
        if let Some(reason) = kept_because(path) {
            report.kept.push((path.to_string(), reason));
            return;
        }
        if linear.shape().1 % group != 0 {
            report.kept.push((
                path.to_string(),
                "its input width doesn't divide into the scheme's groups",
            ));
            return;
        }
        let dense = linear.weight.value.nbytes();
        // Evaluated here, one layer at a time: a lazy packed weight holds
        // its dense source, and the whole model's packing would otherwise
        // run, intermediates and all, inside the first training step.
        let packed = config.pack(linear).and_then(|()| {
            pmetal_bridge::compat::eval(linear.parameters().into_values().filter_map(|value| {
                match value {
                    pmetal_bridge::compat::NestedValue::Value(a) => Some(a),
                    pmetal_bridge::compat::NestedValue::Map(_) => None,
                }
            }))?;
            pmetal_bridge::check_last_error().map_err(|e| {
                pmetal_bridge::compat::Exception::custom(format!("packing {path}: {e}"))
            })
        });
        match packed {
            Ok(()) => {
                report.packed += 1;
                report.dense_bytes += dense;
                report.packed_bytes += linear
                    .parameters()
                    .iter()
                    .filter(|(name, _)| !name.starts_with("lora_") && name.as_ref() != "bias")
                    .map(|(_, value)| match value {
                        pmetal_bridge::compat::NestedValue::Value(a) => a.nbytes(),
                        pmetal_bridge::compat::NestedValue::Map(_) => 0,
                    })
                    .sum::<usize>();
            }
            Err(e) => failure = Some(LoraError::Mlx(e)),
        }
    });
    match failure {
        Some(e) => Err(e),
        None => Ok(report),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_bridge::compat::{Array, Dtype, random};

    fn layer(in_features: i32) -> Linear {
        Linear::new(in_features, 256, false).expect("layer")
    }

    /// One projection per scheme: packed, with the bytes the scheme implies.
    #[test]
    fn each_scheme_packs_to_its_width() {
        // Wide enough that the per-layer codebook and tensor scale vanish
        // into the rounding.
        let n = 512 * 256;
        for (scheme, bits_per_weight) in [
            (QLoraScheme::Nf4, 4.25),
            (QLoraScheme::Fp4, 4.5),
            (QLoraScheme::Int8, 9.0),
        ] {
            let mut model = vec![layer(512)];
            let config = QLoraConfig {
                scheme,
                ..Default::default()
            };
            let report = quantize_base(&mut model, &config).expect("pack");
            pmetal_bridge::check_last_error().expect("no bridge error");
            assert_eq!(report.packed, 1, "{scheme}");
            assert_eq!(report.dense_bytes, n * 4, "{scheme}: f32 dense");
            // The 4-byte NVFP4 tensor scale rounds into the comparison.
            let bits = report.packed_bytes as f64 * 8.0 / n as f64;
            assert!(
                (bits - bits_per_weight).abs() < 0.01,
                "{scheme}: {bits} bits per weight, expected {bits_per_weight}"
            );
            assert!(model[0].quant.is_some(), "{scheme}");
        }
    }

    #[test]
    fn double_quant_packs_the_absmax_too() {
        let mut model = vec![layer(512)];
        let config = QLoraConfig {
            double_quant: true,
            ..Default::default()
        };
        let report = quantize_base(&mut model, &config).expect("pack");
        let bits = report.packed_bytes as f64 * 8.0 / (512.0 * 256.0);
        // 4 + (8 + 2·16/64) / 64.
        assert!((bits - 4.133).abs() < 0.01, "{bits} bits per weight");
    }

    /// The paths QLoRA leaves alone, and a width no group divides.
    #[test]
    fn heads_routers_experts_and_odd_widths_stay_as_loaded() {
        assert!(kept_because("lm_head").is_some());
        assert!(kept_because("model.layers.0.mlp.gate").is_some());
        assert!(kept_because("model.layers.0.mlp.router").is_some());
        assert!(kept_because("model.layers.0.mlp.experts.3.w1").is_some());
        assert!(
            kept_because("model.layers.0.mlp.weight").is_some(),
            "DeepSeek's router"
        );
        assert!(
            kept_because("model.layers.1.block_sparse_moe.router.layer").is_some(),
            "Granite 4's router"
        );
        assert!(
            kept_because("model.layers.2.router.proj").is_some(),
            "Gemma 4's router"
        );
        assert!(
            kept_because("decoder.layers.0.router.proj").is_some(),
            "DiffusionGemma's router"
        );
        assert!(kept_because("model.layers.0.mlp.gate_proj").is_none());
        assert!(kept_because("model.layers.0.self_attn.q_proj").is_none());

        let mut model = vec![layer(48)];
        let report = quantize_base(&mut model, &QLoraConfig::default()).expect("pack");
        assert_eq!(report.packed, 0);
        assert_eq!(report.kept.len(), 1);
        assert!(model[0].quant.is_none());
    }

    #[test]
    fn configs_that_cannot_pack_are_refused_by_name() {
        let bad_group = QLoraConfig {
            group_size: 48,
            ..Default::default()
        };
        let err = quantize_base(&mut vec![layer(128)], &bad_group).unwrap_err();
        assert!(err.to_string().contains("48"), "{err}");

        let dq_on_int8 = QLoraConfig {
            scheme: QLoraScheme::Int8,
            double_quant: true,
            ..Default::default()
        };
        assert!(dq_on_int8.validate().is_err());
    }

    /// Each scheme reconstructs a normally distributed weight to within its
    /// width: four bits to about a tenth of the weight's norm, eight to under
    /// a hundredth.
    #[test]
    fn each_scheme_reconstructs_a_normal_weight() {
        let weight = random::normal(&[256, 512], Dtype::Float32).multiply(&Array::from_f32(0.02));
        let norm = weight.square().sum(None);
        norm.eval();
        let norm = norm.item_f32().sqrt();
        for (scheme, bound) in [
            (QLoraScheme::Nf4, 0.12),
            (QLoraScheme::Fp4, 0.12),
            (QLoraScheme::Int8, 0.01),
        ] {
            let mut model = vec![layer(512)];
            model[0].weight.value = weight.clone();
            let config = QLoraConfig {
                scheme,
                ..Default::default()
            };
            quantize_base(&mut model, &config).expect("pack");
            let diff = model[0]
                .dense_weight()
                .as_dtype(Dtype::Float32.as_i32())
                .subtract(&weight)
                .square()
                .sum(None);
            diff.eval();
            pmetal_bridge::check_last_error().expect("no bridge error");
            let relative = diff.item_f32().sqrt() / norm;
            assert!(relative < bound, "{scheme}: relative error {relative}");
        }
    }
}
