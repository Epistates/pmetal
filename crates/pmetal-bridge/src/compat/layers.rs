use super::{
    Array, Exception, LoraAdapter, ModuleParamMut, ModuleParamRef, ModuleParameters, NestedValue,
    Param, Parameter, ops, random,
};
use crate::QuantizedMode;
use crate::native_weight::QuantParams;
use std::collections::HashMap;
use std::rc::Rc;

fn fp8_weight_for_compute(weight: &Array) -> Array {
    if weight.dtype() == super::Dtype::Uint8 {
        ops::from_fp8(weight, super::Dtype::Bfloat16).expect("failed to dequantize FP8 weight")
    } else {
        weight.clone()
    }
}

fn linear_forward_array(x: &Array, weight: &Array, bias: Option<&Array>) -> Array {
    let weight = fp8_weight_for_compute(weight);
    let x = if weight.dtype() == super::Dtype::Bfloat16 && x.dtype() != super::Dtype::Bfloat16 {
        x.as_dtype(super::Dtype::Bfloat16.as_i32())
    } else {
        x.clone()
    };

    let output = x.matmul(&weight.t());
    if let Some(bias) = bias {
        output.add(bias)
    } else {
        output
    }
}

// ── Linear ────────────────────────────────────────────────────────────────

/// How a [`Linear`]'s weight is packed for MLX's quantized matmul: the packed
/// `uint32` weight sits in `weight` as usual, `[out, in·bits/32]`, and its
/// scales (and, for affine quantization, biases) beside it. This is
/// `mlx.nn.QuantizedLinear`'s layout, and the parameter tree names them the
/// same way (`scales`, `biases`).
#[derive(Debug, Clone)]
pub struct LinearQuant {
    pub scales: Array,
    pub biases: Option<Array>,
    pub params: QuantParams,
    /// The weight's dtype before packing, which the layer keeps computing
    /// in. The floating-point modes' scales are `uint8` (E8M0, E4M3), so
    /// they can't stand in for it the way affine scales do.
    pub dtype: super::Dtype,
    /// NVFP4's per-tensor FP32 scale, which the product is multiplied by.
    /// See [`Linear::quantize`].
    pub tensor_scale: Option<Array>,
    /// Set for a codebook format such as NF4, where `weight` holds 4-bit
    /// indices into a table rather than affine levels. See
    /// [`Linear::quantize_codebook`].
    pub codebook: Option<Codebook>,
}

/// A 4-bit codebook weight format: each weight is `values[code] · absmax`
/// over its group, the scheme NF4 (Dettmers et al., 2023, §3) uses.
///
/// The codes are packed in MLX's 4-bit layout, so MLX's own dequantize, run
/// with unit scales and zero biases, unpacks them; the table lookup and the
/// absmax scale follow on the GPU. MLX has no codebook mode of its own.
///
/// [`LinearQuant::scales`] holds the per-group absmax, `[out, in / group]`,
/// unless it is itself packed (`absmax_packing`): QLoRA's double
/// quantization, here 8-bit affine over runs of 64 absmax values.
#[derive(Debug, Clone)]
pub struct Codebook {
    /// The 16 code values, in ascending order, in `[-1, 1]`. A value may
    /// repeat (E2M1 has a negative zero).
    pub values: Array,
    /// When the absmax is packed: its scales and biases, and the shape it
    /// unpacks to.
    pub absmax_packing: Option<(Array, Array)>,
}

/// Group size of the 8-bit affine packing double quantization applies to
/// the absmax values.
const ABSMAX_GROUP: i32 = 64;

/// Group size the packed codes are unpacked over, independent of the absmax
/// group: the smallest MLX's affine dequantize takes.
const CODE_GROUP: i32 = 32;

/// Pack `dense` (`[out, in]`) as indices into `codebook`, by absmax over
/// groups of `group_size`. Returns the packed codes `[out, in / 8]` and the
/// absmax `[out, in / group_size]` in f32.
fn codebook_pack(
    dense: &Array,
    codebook: &[f32],
    group_size: i32,
) -> Result<(Array, Array), Exception> {
    let (out, inp) = (dense.dim(0), dense.dim(1));
    if codebook.len() != 16 || codebook.windows(2).any(|w| w[0] > w[1]) {
        return Err(Exception::custom(
            "Linear::quantize_codebook: the codebook must hold 16 values in ascending order",
        ));
    }
    if group_size <= 0 || inp % group_size != 0 || inp % CODE_GROUP != 0 {
        return Err(Exception::custom(format!(
            "Linear::quantize_codebook: {inp} input features don't divide into groups of \
             {group_size} and words of {CODE_GROUP}"
        )));
    }
    let f32_ = super::Dtype::Float32.as_i32();
    let u32_ = super::Dtype::Uint32.as_i32();
    let groups = inp / group_size;
    let w = dense.as_dtype(f32_).reshape(&[out, groups, group_size]);
    let absmax = w
        .abs()
        .max_axis(-1, true)
        .maximum(&Array::from_f32(f32::MIN_POSITIVE));
    let normalized = w.divide(&absmax);
    // The nearest code is the number of midpoints below the value.
    let mut codes = ops::zeros(&[out, groups, group_size], super::Dtype::Uint32);
    for pair in codebook.windows(2) {
        let midpoint = Array::from_f32((pair[0] + pair[1]) / 2.0);
        codes = codes.add(&normalized.greater(&midpoint).as_dtype(u32_));
    }
    // Eight codes to a word, lowest bits first: MLX's 4-bit layout.
    let places: Vec<u32> = (0..8).map(|j| 1u32 << (4 * j)).collect();
    let packed = codes
        .reshape(&[out, inp / 8, 8])
        .multiply(&Array::from_u32_slice(&places, &[8]))
        .sum_axis(-1, false);
    Ok((packed, absmax.reshape(&[out, groups])))
}

/// Unpack a codebook weight to dense `[out, in]` in `dtype`.
fn codebook_dense(
    packed: &Array,
    absmax: &Array,
    codebook: &Codebook,
    group_size: i32,
    dtype: super::Dtype,
) -> Array {
    let (out, inp) = (packed.dim(0), packed.dim(1) * 8);
    let groups = inp / group_size;
    // Everything at the layer's own width: a dense f32 copy per layer is
    // twice the transient memory, and the table's values lose nothing that
    // four bits had kept.
    let absmax = match &codebook.absmax_packing {
        Some((scales, biases)) => absmax
            .dequantize(scales, biases, ABSMAX_GROUP, 8)
            .reshape(&[out, groups]),
        None => absmax.clone(),
    }
    .as_dtype(dtype.as_i32());
    // Unit scales and zero biases dequantize each code to its own value,
    // exactly: the codes are 0..15, which bf16 holds exactly.
    let code_groups = inp / CODE_GROUP;
    let bf16 = super::Dtype::Bfloat16;
    let codes = packed.dequantize(
        &ops::ones(&[out, code_groups], bf16),
        &ops::zeros(&[out, code_groups], bf16),
        CODE_GROUP,
        4,
    );
    codebook
        .values
        .as_dtype(dtype.as_i32())
        .take_axis(&codes.as_dtype(super::Dtype::Uint32.as_i32()), 0)
        .reshape(&[out, groups, group_size])
        .multiply(&absmax.expand_dims(-1))
        .reshape(&[out, inp])
}

/// Affine linear layer: `y = x @ W^T + b`, optionally low-rank adapted.
///
/// `adapter` is the seam that makes every architecture in the workspace
/// fine-tunable without knowing it. See [`LoraAdapter`] for why it lives here
/// rather than in a parallel `LoraLinear` type.
#[derive(Debug, Clone)]
pub struct Linear {
    pub weight: Param<Array>,
    pub bias: Param<Option<Array>>,
    /// Low-rank adapter, when one has been attached. `None` is a plain dense
    /// layer and costs one predictable branch per forward.
    pub adapter: Option<Box<LoraAdapter>>,
    /// Set when `weight` is packed for MLX's quantized matmul
    /// ([`quantize`](Self::quantize)). An adapter works the same on a packed
    /// weight, which is QLoRA: the base stays packed and frozen, the adapter
    /// trains.
    pub quant: Option<LinearQuant>,
}

impl Linear {
    pub const DEFAULT_BIAS: bool = true;

    /// Attach a low-rank adapter, replacing any existing one.
    ///
    /// The layer's output is unchanged until the adapter is trained: `B` is
    /// zero-initialised.
    pub fn attach_lora(
        &mut self,
        rank: i32,
        alpha: f32,
        use_rslora: bool,
    ) -> Result<&mut LoraAdapter, super::Exception> {
        let (out_features, in_features) = self.shape();
        let adapter = LoraAdapter::new(in_features, out_features, rank, alpha, use_rslora)?;
        Ok(self.adapter.insert(Box::new(adapter)))
    }

    /// Drop the adapter, leaving the frozen base weight as it stands.
    ///
    /// Returns the adapter so a caller can keep it. This does *not* fold it in;
    /// use [`merge_lora`](Self::merge_lora) for that.
    pub fn detach_lora(&mut self) -> Option<Box<LoraAdapter>> {
        self.adapter.take()
    }

    /// Fold the adapter into the base weight, keeping it attached.
    ///
    /// Reversible with [`unmerge_lora`](Self::unmerge_lora): a merged layer runs
    /// one matmul instead of three, which is worth it for evaluation mid-run,
    /// and training can resume afterwards. Use [`fuse_lora`](Self::fuse_lora)
    /// when the fold should be permanent.
    ///
    /// A packed weight is unpacked first: the merged weight isn't the packed
    /// one plus a delta, so the layer is dense from then on.
    pub fn merge_lora(&mut self) {
        if self.adapter.as_ref().is_some_and(|a| !a.merged) {
            self.dequantize();
        }
        if let Some(adapter) = self.adapter.as_mut() {
            adapter.merge_into(&mut self.weight.value);
        }
    }

    /// Undo [`merge_lora`](Self::merge_lora).
    pub fn unmerge_lora(&mut self) {
        if let Some(adapter) = self.adapter.as_mut() {
            adapter.unmerge_from(&mut self.weight.value);
        }
    }

    /// Fold the adapter in and drop it, leaving an ordinary dense layer.
    ///
    /// This is what `pmetal fuse` produces: a model that serves without
    /// pmetal-lora in the picture, and cannot be un-fused.
    pub fn fuse_lora(&mut self) {
        self.merge_lora();
        self.adapter = None;
    }

    /// Whether a low-rank adapter is attached.
    pub fn is_adapted(&self) -> bool {
        self.adapter.is_some()
    }

    /// Put any attached adapter into training mode, enabling its dropout.
    pub fn set_adapter_training(&mut self, training: bool) {
        if let Some(adapter) = self.adapter.as_mut() {
            adapter.training = training;
        }
    }

    pub fn new(in_dims: i32, out_dims: i32, with_bias: bool) -> Result<Self, super::Exception> {
        let scale = f32::sqrt(1.0 / in_dims as f32);
        let weight =
            random::uniform_range(-scale, scale, &[out_dims, in_dims], super::Dtype::Float32);
        let bias = if with_bias {
            Some(random::uniform_range(
                -scale,
                scale,
                &[out_dims],
                super::Dtype::Float32,
            ))
        } else {
            None
        };
        Ok(Self {
            weight: Param::new(weight),
            bias: Param::new(bias),
            adapter: None,
            quant: None,
        })
    }

    /// Infallible constructor variant for internal use.
    pub fn create(in_dims: i32, out_dims: i32, with_bias: bool) -> Self {
        Self::new(in_dims, out_dims, with_bias).unwrap()
    }

    pub fn forward(&self, x: &Array) -> Array {
        let bias = self.bias.value.as_ref();
        let quant = match &self.quant {
            Some(quant) if quant.codebook.is_none() => quant,
            // A codebook has no fused kernel: unpack, then multiply dense.
            Some(_) => {
                let weight = self.dense_weight();
                return match self.adapter.as_deref() {
                    None => linear_forward_array(x, &weight, bias),
                    Some(adapter) => adapter.apply(x, &weight, bias),
                };
            }
            None => {
                return match self.adapter.as_deref() {
                    None => linear_forward_array(x, &self.weight.value, bias),
                    Some(adapter) => adapter.apply(x, &self.weight.value, bias),
                };
            }
        };
        // The kernel takes the activations in the dtype the weight had.
        let x = if x.dtype() == quant.dtype {
            x.clone()
        } else {
            x.as_dtype(quant.dtype.as_i32())
        };
        let y = x.quantized_matmul_mode(
            &self.weight.value,
            &quant.scales,
            quant.biases.as_ref(),
            true,
            quant.params.group_size,
            quant.params.bits,
            quant.params.mode,
        );
        let y = match &quant.tensor_scale {
            Some(scale) => y.multiply(&scale.as_dtype(y.dtype().as_i32())),
            None => y,
        };
        match self.adapter.as_deref() {
            None => match bias {
                Some(b) => y.add(b),
                None => y,
            },
            Some(adapter) => adapter.apply_to_product(&x, y, &|| self.dense_weight(), bias),
        }
    }

    /// `(out_features, in_features)`, whether the weight is packed or not.
    pub fn shape(&self) -> (i32, i32) {
        let s = self.weight.value.shape();
        match &self.quant {
            Some(quant) => (s[0], s[1] * 32 / quant.params.bits),
            None => (s[0], s[1]),
        }
    }

    /// Pack the weight for MLX's quantized matmul, as
    /// `mlx.nn.QuantizedLinear.from_linear` does. Smaller and, where reading
    /// the weight is the cost (decode), faster; the layer's parameters stop
    /// training, though an adapter on it still does.
    ///
    /// Refused while an adapter is merged in, which a packed weight couldn't
    /// unmerge.
    pub fn quantize(&mut self, params: QuantParams) -> Result<(), Exception> {
        if self.quant.is_some() {
            return Ok(());
        }
        if self.adapter.as_ref().is_some_and(|a| a.merged) {
            return Err(Exception::custom(
                "Linear::quantize: unmerge the adapter before packing the weight",
            ));
        }
        let dense = fp8_weight_for_compute(&self.weight.value);
        let dtype = dense.dtype();
        // NVFP4 is two-level: an FP32 scale per tensor, so that each block's
        // E4M3 scale lands in E4M3's normal range, then E2M1 values over
        // blocks of 16. One level alone puts a typical weight's block scales
        // (absmax / 6, around 1e-3) among E4M3's subnormals, which hold two
        // bits or fewer: Qwen3-0.6B lost 0.10 nats instead of 0.06. The scale
        // is MLX's: the tensor's amax over 448 · 6, the largest product of an
        // E4M3 scale and an E2M1 value.
        let tensor_scale = (params.mode == QuantizedMode::Nvfp4).then(|| {
            dense
                .as_dtype(super::Dtype::Float32.as_i32())
                .abs()
                .max(None)
                .maximum(&Array::from_f32(f32::MIN_POSITIVE))
                .divide(&Array::from_f32(448.0 * 6.0))
        });
        let dense = match &tensor_scale {
            Some(scale) => dense
                .as_dtype(super::Dtype::Float32.as_i32())
                .divide(scale)
                .as_dtype(dtype.as_i32()),
            None => dense,
        };
        let (weight, scales, biases) = match params.mode {
            QuantizedMode::Affine => {
                let (w, s, b) = dense.quantize_weights(params.group_size, params.bits);
                (w, s, Some(b))
            }
            mode => {
                let (w, s) = dense.quantize_weights_mode(params.group_size, params.bits, mode);
                (w, s, None)
            }
        };
        crate::check_last_error()
            .map_err(|e| Exception::custom(format!("Linear::quantize: {e}")))?;
        self.weight.value = weight;
        self.quant = Some(LinearQuant {
            scales,
            biases,
            params,
            dtype,
            tensor_scale,
            codebook: None,
        });
        Ok(())
    }

    /// Pack the weight as 4-bit indices into `codebook` (16 ascending values
    /// in `[-1, 1]`), scaled by each group's absmax: NF4 when `codebook` is
    /// NF4's table. `double_quant` packs the absmax values themselves to 8
    /// bits, QLoRA's double quantization.
    ///
    /// Same contract as [`quantize`](Self::quantize): the layer stops
    /// training, an adapter on it still trains, and merging unpacks it.
    pub fn quantize_codebook(
        &mut self,
        codebook: &[f32],
        group_size: i32,
        double_quant: bool,
    ) -> Result<(), Exception> {
        if self.quant.is_some() {
            return Ok(());
        }
        if self.adapter.as_ref().is_some_and(|a| a.merged) {
            return Err(Exception::custom(
                "Linear::quantize_codebook: unmerge the adapter before packing the weight",
            ));
        }
        let dense = fp8_weight_for_compute(&self.weight.value);
        let dtype = dense.dtype();
        let (packed, absmax) = codebook_pack(&dense, codebook, group_size)?;
        let count = absmax.dim(0) * absmax.dim(1);
        let (scales, absmax_packing) = if double_quant && count % ABSMAX_GROUP == 0 {
            // bf16 first, so the packing's scales and biases are bf16 too.
            let (q, s, b) = absmax
                .reshape(&[count / ABSMAX_GROUP, ABSMAX_GROUP])
                .as_dtype(super::Dtype::Bfloat16.as_i32())
                .quantize_weights(ABSMAX_GROUP, 8);
            (q, Some((s, b)))
        } else {
            (absmax.as_dtype(super::Dtype::Bfloat16.as_i32()), None)
        };
        crate::check_last_error()
            .map_err(|e| Exception::custom(format!("Linear::quantize_codebook: {e}")))?;
        self.weight.value = packed;
        self.quant = Some(LinearQuant {
            scales,
            biases: None,
            params: QuantParams {
                group_size,
                bits: 4,
                mode: QuantizedMode::Affine,
            },
            dtype,
            tensor_scale: None,
            codebook: Some(Codebook {
                values: Array::from_f32_slice(codebook, &[16]),
                absmax_packing,
            }),
        });
        Ok(())
    }

    /// The weight as a dense `[out, in]` array, unpacked if it's packed.
    pub fn dense_weight(&self) -> Array {
        let Some(quant) = &self.quant else {
            return fp8_weight_for_compute(&self.weight.value);
        };
        if let Some(codebook) = &quant.codebook {
            return codebook_dense(
                &self.weight.value,
                &quant.scales,
                codebook,
                quant.params.group_size,
                quant.dtype,
            );
        }
        let dense = self.weight.value.dequantize_mode(
            &quant.scales,
            quant.biases.as_ref(),
            quant.params.group_size,
            quant.params.bits,
            quant.params.mode,
        );
        let dense = match &quant.tensor_scale {
            Some(scale) => dense
                .as_dtype(super::Dtype::Float32.as_i32())
                .multiply(scale),
            None => dense,
        };
        if dense.dtype() == quant.dtype {
            dense
        } else {
            dense.as_dtype(quant.dtype.as_i32())
        }
    }

    /// The weight as stored, for a fast path that multiplies by it itself
    /// (concatenated with its neighbours', flattened for decode, handed to a
    /// fused kernel) instead of calling [`forward`](Self::forward).
    ///
    /// `Some` only when that computes what `forward` does and stays in
    /// autograd's view: no adapter that isn't merged into the weight (a fast
    /// path would skip it), the weight not packed (it would be read as its
    /// packed words), and the weight not being differentiated (a fast path
    /// caches what it builds from it). An FP8 weight comes back as stored, for
    /// the caller to dequantize. Every fast path asks this, rather than
    /// checking the layer's fields itself, so a new way for a layer to compute
    /// something other than `x·Wᵀ + b` has one place to be declared.
    pub fn plain_weight(&self) -> Option<&Array> {
        let adapted = self.adapter.as_ref().is_some_and(|a| !a.merged);
        let plain = !adapted && self.quant.is_none() && !self.weight.value.is_tracer();
        plain.then_some(&self.weight.value)
    }

    /// The dense `[out, in]` weight `W` for which [`forward`](Self::forward)
    /// computes `x·Wᵀ + b`: unpacked, with an attached adapter folded in.
    ///
    /// For a caller that multiplies by the whole matrix in its own way, cut
    /// cross-entropy over an LM head for one. Folding is exact, gradients
    /// included: `x·(W + s·B·A)ᵀ` is the same function of `A` and `B` as the
    /// adapter's own two small matmuls. `None` while the adapter applies
    /// dropout, which acts on the input and no single weight reproduces.
    pub fn effective_weight(&self) -> Option<Array> {
        let dense = self.dense_weight();
        match self.adapter.as_deref() {
            Some(adapter) if !adapter.merged => {
                if adapter.training && adapter.dropout > 0.0 {
                    None
                } else {
                    Some(adapter.merged_weight(&dense))
                }
            }
            _ => Some(dense),
        }
    }

    /// Undo [`quantize`](Self::quantize), leaving the unpacked weight (with
    /// the rounding packing introduced).
    pub fn dequantize(&mut self) {
        if self.quant.is_some() {
            self.weight.value = self.dense_weight();
            self.quant = None;
        }
    }

    #[inline]
    pub fn unwrap(self) -> Self {
        self
    }

    #[inline]
    pub fn expect(self, _msg: &str) -> Self {
        self
    }
}

// Hand-written rather than `impl_module_params!` because the adapter must
// flatten *beside* the weight, not under it. The generated impl would nest the
// adapter's own map and produce `q_proj.adapter.lora_a`; every adapter file ever
// written by pmetal says `q_proj.lora_a`.
impl ModuleParameters for Linear {
    fn num_parameters(&self) -> usize {
        Parameter::count_params(&self.weight)
            + Parameter::count_params(&self.bias)
            + self.quant.as_ref().map_or(0, |q| {
                1 + usize::from(q.biases.is_some())
                    + usize::from(q.tensor_scale.is_some())
                    + q.codebook
                        .as_ref()
                        .map_or(0, |c| 1 + 2 * usize::from(c.absmax_packing.is_some()))
            })
            + self
                .adapter
                .as_ref()
                .map_or(0, |a| 2 + usize::from(a.magnitude.is_some()))
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        let mut out = HashMap::new();
        Parameter::collect_params(&self.weight, "weight", &mut out);
        Parameter::collect_params(&self.bias, "bias", &mut out);
        if let Some(quant) = &self.quant {
            out.insert(Rc::from("scales"), NestedValue::Value(&quant.scales));
            if let Some(biases) = &quant.biases {
                out.insert(Rc::from("biases"), NestedValue::Value(biases));
            }
            if let Some(scale) = &quant.tensor_scale {
                out.insert(Rc::from("tensor_scale"), NestedValue::Value(scale));
            }
            if let Some(codebook) = &quant.codebook {
                out.insert(Rc::from("codebook"), NestedValue::Value(&codebook.values));
                if let Some((scales, biases)) = &codebook.absmax_packing {
                    out.insert(Rc::from("absmax_scales"), NestedValue::Value(scales));
                    out.insert(Rc::from("absmax_biases"), NestedValue::Value(biases));
                }
            }
        }
        if let Some(adapter) = self.adapter.as_deref() {
            out.insert(Rc::from("lora_a"), NestedValue::Value(&adapter.a));
            out.insert(Rc::from("lora_b"), NestedValue::Value(&adapter.b));
            if let Some(magnitude) = adapter.magnitude.as_ref() {
                out.insert(Rc::from("lora_magnitude"), NestedValue::Value(magnitude));
            }
        }
        out
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        let mut out = HashMap::new();
        Parameter::collect_params_mut(&mut self.weight, "weight", &mut out);
        Parameter::collect_params_mut(&mut self.bias, "bias", &mut out);
        if let Some(quant) = &mut self.quant {
            out.insert(Rc::from("scales"), NestedValue::Value(&mut quant.scales));
            if let Some(biases) = &mut quant.biases {
                out.insert(Rc::from("biases"), NestedValue::Value(biases));
            }
            if let Some(scale) = &mut quant.tensor_scale {
                out.insert(Rc::from("tensor_scale"), NestedValue::Value(scale));
            }
            if let Some(codebook) = &mut quant.codebook {
                out.insert(
                    Rc::from("codebook"),
                    NestedValue::Value(&mut codebook.values),
                );
                if let Some((scales, biases)) = &mut codebook.absmax_packing {
                    out.insert(Rc::from("absmax_scales"), NestedValue::Value(scales));
                    out.insert(Rc::from("absmax_biases"), NestedValue::Value(biases));
                }
            }
        }
        if let Some(adapter) = self.adapter.as_deref_mut() {
            out.insert(Rc::from("lora_a"), NestedValue::Value(&mut adapter.a));
            out.insert(Rc::from("lora_b"), NestedValue::Value(&mut adapter.b));
            if let Some(magnitude) = adapter.magnitude.as_mut() {
                out.insert(Rc::from("lora_magnitude"), NestedValue::Value(magnitude));
            }
        }
        out
    }

    /// An adapted layer trains its adapter and nothing else: that is what makes
    /// it LoRA rather than a full fine-tune. An unadapted one is ordinary and
    /// trains everything, unless its weight is packed, which can't train
    /// (`mlx.nn.QuantizedLinear` freezes itself the same way).
    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        let Some(adapter) = self.adapter.as_deref() else {
            if self.quant.is_some() {
                return HashMap::new();
            }
            return self.parameters();
        };
        let mut out = HashMap::new();
        out.insert(Rc::from("lora_a"), NestedValue::Value(&adapter.a));
        out.insert(Rc::from("lora_b"), NestedValue::Value(&adapter.b));
        if let Some(magnitude) = adapter.magnitude.as_ref() {
            out.insert(Rc::from("lora_magnitude"), NestedValue::Value(magnitude));
        }
        out
    }
}

/// Builder for [`Linear`].
pub struct LinearBuilder {
    in_dims: i32,
    out_dims: i32,
    bias: bool,
}

impl LinearBuilder {
    pub fn new(in_dims: i32, out_dims: i32) -> Self {
        Self {
            in_dims,
            out_dims,
            bias: Linear::DEFAULT_BIAS,
        }
    }
    pub fn bias(mut self, b: bool) -> Self {
        self.bias = b;
        self
    }
    pub fn build(self) -> Result<Linear, Exception> {
        Linear::new(self.in_dims, self.out_dims, self.bias)
    }
}

// ── RmsNorm ───────────────────────────────────────────────────────────────

/// RMS layer normalization.
#[derive(Debug, Clone)]
pub struct RmsNorm {
    pub weight: Param<Array>,
    pub eps: f32,
}

impl RmsNorm {
    pub const DEFAULT_EPS: f32 = 1e-5;

    pub fn new(dims: i32) -> Result<Self, Exception> {
        Ok(Self::with_eps(dims, Self::DEFAULT_EPS))
    }

    pub fn with_eps(dims: i32, eps: f32) -> Self {
        let weight = ops::ones(&[dims], super::Dtype::Float32);
        Self {
            weight: Param::new(weight),
            eps,
        }
    }

    pub fn forward(&self, x: &Array) -> Array {
        let weight = fp8_weight_for_compute(&self.weight.value);
        x.rms_norm(Some(&weight), self.eps)
    }
}

crate::impl_module_params!(RmsNorm; weight);

/// Builder for [`RmsNorm`].
pub struct RmsNormBuilder {
    dims: i32,
    eps: f32,
}

impl RmsNormBuilder {
    pub fn new(dims: i32) -> Self {
        Self {
            dims,
            eps: RmsNorm::DEFAULT_EPS,
        }
    }
    pub fn eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }
    pub fn build(self) -> Result<RmsNorm, Exception> {
        Ok(RmsNorm::with_eps(self.dims, self.eps))
    }
}

// ── LayerNorm ─────────────────────────────────────────────────────────────

/// Layer normalization.
#[derive(Debug, Clone)]
pub struct LayerNorm {
    pub dimensions: i32,
    pub eps: f32,
    pub weight: Param<Option<Array>>,
    pub bias: Param<Option<Array>>,
}

impl LayerNorm {
    pub const DEFAULT_EPS: f32 = 1e-5;
    pub const DEFAULT_AFFINE: bool = true;

    pub fn with_affine(dims: i32, eps: f32, affine: bool) -> Self {
        let (w, b) = if affine {
            (
                Some(ops::ones(&[dims], super::Dtype::Float32)),
                Some(ops::zeros(&[dims], super::Dtype::Float32)),
            )
        } else {
            (None, None)
        };
        Self {
            dimensions: dims,
            eps,
            weight: Param::new(w),
            bias: Param::new(b),
        }
    }

    pub fn forward(&self, x: &Array) -> Array {
        let w = self.weight.value.as_ref().map(fp8_weight_for_compute);
        let b = self.bias.value.as_ref().map(fp8_weight_for_compute);
        x.layer_norm(w.as_ref(), b.as_ref(), self.eps)
    }
}

crate::impl_module_params!(LayerNorm; weight, bias);

/// Builder for [`LayerNorm`].
pub struct LayerNormBuilder {
    dims: i32,
    eps: f32,
    affine: bool,
}

impl LayerNormBuilder {
    pub fn new(dims: i32) -> Self {
        Self {
            dims,
            eps: LayerNorm::DEFAULT_EPS,
            affine: LayerNorm::DEFAULT_AFFINE,
        }
    }
    pub fn eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }
    pub fn affine(mut self, a: bool) -> Self {
        self.affine = a;
        self
    }
    pub fn build(self) -> Result<LayerNorm, Exception> {
        Ok(LayerNorm::with_affine(self.dims, self.eps, self.affine))
    }
}

// ── GroupNorm ─────────────────────────────────────────────────────────────

/// Group normalization.
#[derive(Debug, Clone)]
pub struct GroupNorm {
    pub group_count: i32,
    pub dimensions: i32,
    pub eps: Array,
    pub pytorch_compatible: bool,
    pub weight: Param<Option<Array>>,
    pub bias: Param<Option<Array>>,
}

impl GroupNorm {
    pub const DEFAULT_EPS: f32 = 1e-5;
    pub const DEFAULT_AFFINE: bool = true;
    pub const DEFAULT_PYTORCH_COMPATIBLE: bool = false;

    pub fn new(
        group_count: i32,
        dims: i32,
        eps: f32,
        affine: bool,
        pytorch_compatible: bool,
    ) -> Self {
        let (w, b) = if affine {
            (
                Some(ops::ones(&[dims], super::Dtype::Float32)),
                Some(ops::zeros(&[dims], super::Dtype::Float32)),
            )
        } else {
            (None, None)
        };
        Self {
            group_count,
            dimensions: dims,
            eps: Array::from_f32(eps),
            pytorch_compatible,
            weight: Param::new(w),
            bias: Param::new(b),
        }
    }

    pub fn forward(&self, x: &Array) -> Array {
        let eps_f = self.eps.clone().item_f32();
        let batch = x.dim(0);
        let dims = x.dim(-1);
        let group_size = dims / self.group_count;

        if self.pytorch_compatible {
            // PyTorch layout: [B, H, W, C] → reshape to [B, H*W, groups, group_size]
            let x2 = x.reshape(&[batch, -1, self.group_count, group_size]);
            let x2 = x2
                .transpose_axes(&[0, 2, 1, 3])
                .reshape(&[batch, self.group_count, -1]);
            let x2 = x2.layer_norm(None, None, eps_f);
            let ndim = x.ndim();
            let new_shape: Vec<i32> = std::iter::once(batch)
                .chain(x.shape()[1..(ndim as usize - 1)].iter().copied())
                .chain(std::iter::once(dims))
                .collect();
            let x2 = x2.reshape(&[batch, self.group_count, -1, group_size]);
            let x2 = x2.transpose_axes(&[0, 2, 1, 3]).reshape(&new_shape);
            self.apply_affine(x2)
        } else {
            let x2 = x.reshape(&[batch, -1, self.group_count]);
            // instance norm per group
            let mean = x2.mean_axis(1, true);
            let var = x2.subtract(&mean).square().mean_axis(1, true);
            let eps_arr = Array::from_f32(eps_f);
            let x2 = x2.subtract(&mean).multiply(&var.add(&eps_arr).rsqrt());
            let ndim = x.ndim();
            let new_shape: Vec<i32> = std::iter::once(batch)
                .chain(x.shape()[1..(ndim as usize - 1)].iter().copied())
                .chain(std::iter::once(dims))
                .collect();
            let x2 = x2.reshape(&new_shape);
            self.apply_affine(x2)
        }
    }

    fn apply_affine(&self, x: Array) -> Array {
        match (&self.weight.value, &self.bias.value) {
            (Some(w), Some(b)) => {
                let w = fp8_weight_for_compute(w);
                let b = fp8_weight_for_compute(b);
                x.multiply(&w).add(&b)
            }
            (Some(w), None) => {
                let w = fp8_weight_for_compute(w);
                x.multiply(&w)
            }
            (None, Some(b)) => {
                let b = fp8_weight_for_compute(b);
                x.add(&b)
            }
            (None, None) => x,
        }
    }
}

crate::impl_module_params!(GroupNorm; weight, bias);

/// Builder for [`GroupNorm`].
pub struct GroupNormBuilder {
    group_count: i32,
    dims: i32,
    eps: f32,
    affine: bool,
    pytorch_compatible: bool,
}

impl GroupNormBuilder {
    pub fn new(group_count: i32, dims: i32) -> Self {
        Self {
            group_count,
            dims,
            eps: GroupNorm::DEFAULT_EPS,
            affine: GroupNorm::DEFAULT_AFFINE,
            pytorch_compatible: GroupNorm::DEFAULT_PYTORCH_COMPATIBLE,
        }
    }
    pub fn eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }
    pub fn affine(mut self, a: bool) -> Self {
        self.affine = a;
        self
    }
    pub fn pytorch_compatible(mut self, p: bool) -> Self {
        self.pytorch_compatible = p;
        self
    }
    pub fn build(self) -> Result<GroupNorm, Exception> {
        Ok(GroupNorm::new(
            self.group_count,
            self.dims,
            self.eps,
            self.affine,
            self.pytorch_compatible,
        ))
    }
}

// ── Embedding ─────────────────────────────────────────────────────────────

/// Simple embedding lookup table.
///
/// Carries the NEFTune noise scale as a property, the same way [`Linear`]
/// carries its adapter: it is a thing that happens to this layer's output, and
/// putting it here keeps every architecture from having to know about it.
#[derive(Debug, Clone)]
pub struct Embedding {
    pub weight: Param<Array>,
    /// NEFTune noise scale, when training with it. See [`Embedding::forward`].
    pub neftune_alpha: Option<f32>,
}

impl Embedding {
    pub fn new(num_embeddings: i32, dims: i32) -> Result<Self, Exception> {
        let scale = f32::sqrt(1.0 / dims as f32);
        let weight = random::uniform_range(
            -scale,
            scale,
            &[num_embeddings, dims],
            super::Dtype::Float32,
        );
        Ok(Self {
            weight: Param::new(weight),
            neftune_alpha: None,
        })
    }

    /// Look up `x`, adding NEFTune noise when [`Embedding::neftune_alpha`] is set.
    ///
    /// NEFTune (Jain et al., 2023) adds `U(-mag, mag)` to the embedding output
    /// during training, with `mag = alpha / sqrt(seq_len * dims)`. This is the
    /// same point TRL injects at, since its hook fires on the embedding
    /// module's output: for Gemma that means the noise goes in *before* the
    /// `sqrt(hidden_size)` scale and gets multiplied up with everything else.
    ///
    /// Only the token-embedding layer should carry an alpha. A tied LM head
    /// goes through [`Embedding::as_linear`], which never adds noise.
    pub fn forward(&self, x: &Array) -> Array {
        let weight = fp8_weight_for_compute(&self.weight.value);
        // MLX >= 0.32 raises "[gather] Cannot calculate VJP with respect to
        // indices" when token indices sit inside the grad trace; the lookup
        // must never be differentiated w.r.t. its indices.
        let embedded = weight.take_axis(&x.stop_gradient(), 0);
        match self.neftune_alpha {
            Some(alpha) if alpha > 0.0 => add_neftune_noise(&embedded, alpha),
            _ => embedded,
        }
    }

    pub fn as_linear(&self, x: &Array) -> Array {
        linear_forward_array(x, &self.weight.value, None)
    }
}

/// Add NEFTune uniform noise to an embedding output.
///
/// The noise is drawn in `embedded`'s own dtype. Drawing it in f32 against a
/// bf16 checkpoint promotes the sum, and MLX carries that promotion through
/// every op downstream, so the whole forward silently runs in f32 for the rest
/// of the model.
fn add_neftune_noise(embedded: &Array, alpha: f32) -> Array {
    let shape = embedded.shape();
    let dims = shape[shape.len() - 1] as f32;
    let seq_len = shape[shape.len() - 2] as f32;
    let magnitude = alpha / (seq_len * dims).sqrt();
    let noise = random::uniform_range(-magnitude, magnitude, shape, embedded.dtype());
    embedded.add(&noise)
}

crate::impl_module_params!(Embedding; weight);

// ── Conv1d ────────────────────────────────────────────────────────────────

/// 1D convolution layer.
#[derive(Debug, Clone)]
pub struct Conv1d {
    pub weight: Param<Array>,
    pub bias: Param<Option<Array>>,
    pub stride: i32,
    pub padding: i32,
    pub dilation: i32,
    pub groups: i32,
}

impl Conv1d {
    pub const DEFAULT_BIAS: bool = true;
    pub const DEFAULT_STRIDE: i32 = 1;
    pub const DEFAULT_PADDING: i32 = 0;
    pub const DEFAULT_DILATION: i32 = 1;
    pub const DEFAULT_GROUPS: i32 = 1;

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: i32,
        out_channels: i32,
        kernel_size: i32,
        stride: i32,
        padding: i32,
        dilation: i32,
        groups: i32,
        with_bias: bool,
    ) -> Self {
        let scale = f32::sqrt(1.0 / (in_channels * kernel_size) as f32);
        // weight shape: [out_channels, kernel_size, in_channels/groups]
        let weight = random::uniform_range(
            -scale,
            scale,
            &[out_channels, kernel_size, in_channels / groups],
            super::Dtype::Float32,
        );
        let bias = if with_bias {
            Some(ops::zeros(&[out_channels], super::Dtype::Float32))
        } else {
            None
        };
        Self {
            weight: Param::new(weight),
            bias: Param::new(bias),
            stride,
            padding,
            dilation,
            groups,
        }
    }

    pub fn forward(&self, x: &Array) -> Array {
        let y = ops::conv1d(
            x,
            &self.weight.value,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        );
        match &self.bias.value {
            Some(b) => y.add(b),
            None => y,
        }
    }
}

crate::impl_module_params!(Conv1d; weight, bias);

/// Builder for [`Conv1d`].
pub struct Conv1dBuilder {
    in_ch: i32,
    out_ch: i32,
    kernel: i32,
    bias: bool,
    stride: i32,
    padding: i32,
    dilation: i32,
    groups: i32,
}

impl Conv1dBuilder {
    pub fn new(in_ch: i32, out_ch: i32, kernel: i32) -> Self {
        Self {
            in_ch,
            out_ch,
            kernel,
            bias: Conv1d::DEFAULT_BIAS,
            stride: Conv1d::DEFAULT_STRIDE,
            padding: Conv1d::DEFAULT_PADDING,
            dilation: Conv1d::DEFAULT_DILATION,
            groups: Conv1d::DEFAULT_GROUPS,
        }
    }
    pub fn bias(mut self, b: bool) -> Self {
        self.bias = b;
        self
    }
    pub fn stride(mut self, s: i32) -> Self {
        self.stride = s;
        self
    }
    pub fn padding(mut self, p: i32) -> Self {
        self.padding = p;
        self
    }
    pub fn dilation(mut self, d: i32) -> Self {
        self.dilation = d;
        self
    }
    pub fn groups(mut self, g: i32) -> Self {
        self.groups = g;
        self
    }
    pub fn build(self) -> Result<Conv1d, Exception> {
        Ok(Conv1d::new(
            self.in_ch,
            self.out_ch,
            self.kernel,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.bias,
        ))
    }
}

// ── Vec<T> where T: ModuleParameters ─────────────────────────────────────

impl<T: ModuleParameters> ModuleParameters for Vec<T> {
    fn num_parameters(&self) -> usize {
        self.iter().map(|m| m.num_parameters()).sum()
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        let mut out = HashMap::new();
        for (i, m) in self.iter().enumerate() {
            let sub = m.parameters();
            for (k, v) in sub {
                let full: Rc<str> = format!("{i}.{k}").into();
                out.insert(full, unsafe { super::clone_nested_ref_lifetime(v) });
            }
        }
        out
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        let mut out = HashMap::new();
        for (i, m) in self.iter_mut().enumerate() {
            let sub = m.parameters_mut();
            for (k, v) in sub {
                let full: Rc<str> = format!("{i}.{k}").into();
                out.insert(full, unsafe { super::clone_nested_mut_lifetime(v) });
            }
        }
        out
    }

    /// Each element's own trainable set. The default would be every
    /// parameter, a frozen or packed base weight included, and a checkpointed
    /// layer differentiates whatever this reports.
    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        let mut out = HashMap::new();
        for (i, m) in self.iter().enumerate() {
            for (k, v) in m.trainable_parameters() {
                let full: Rc<str> = format!("{i}.{k}").into();
                out.insert(full, unsafe { super::clone_nested_ref_lifetime(v) });
            }
        }
        out
    }
}

// ── Option<T> where T: ModuleParameters ──────────────────────────────────

impl<T: ModuleParameters> ModuleParameters for Option<T> {
    fn num_parameters(&self) -> usize {
        self.as_ref().map_or(0, |m| m.num_parameters())
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        self.as_ref().map_or(HashMap::new(), |m| m.parameters())
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        self.as_mut().map_or(HashMap::new(), |m| m.parameters_mut())
    }

    /// The module's own trainable set; see `Vec<T>`'s.
    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        self.as_ref()
            .map_or(HashMap::new(), |m| m.trainable_parameters())
    }
}

// ── Module<&Array> impls for layer types ──────────────────────────────────
//
// These allow `Module::forward(&mut self.layer, x)?` to work, matching
// the mlx-rs call pattern used throughout the architecture files.

impl super::Module<&Array> for Linear {
    type Output = Array;
    type Error = super::Exception;
    fn forward(&mut self, x: &Array) -> Result<Array, super::Exception> {
        Ok(Linear::forward(self, x))
    }
    fn training_mode(&mut self, _mode: bool) {}
}

impl super::Module<&Array> for RmsNorm {
    type Output = Array;
    type Error = super::Exception;
    fn forward(&mut self, x: &Array) -> Result<Array, super::Exception> {
        Ok(RmsNorm::forward(self, x))
    }
    fn training_mode(&mut self, _mode: bool) {}
}

impl super::Module<&Array> for LayerNorm {
    type Output = Array;
    type Error = super::Exception;
    fn forward(&mut self, x: &Array) -> Result<Array, super::Exception> {
        Ok(LayerNorm::forward(self, x))
    }
    fn training_mode(&mut self, _mode: bool) {}
}

impl super::Module<&Array> for GroupNorm {
    type Output = Array;
    type Error = super::Exception;
    fn forward(&mut self, x: &Array) -> Result<Array, super::Exception> {
        Ok(GroupNorm::forward(self, x))
    }
    fn training_mode(&mut self, _mode: bool) {}
}

impl super::Module<&Array> for Embedding {
    type Output = Array;
    type Error = super::Exception;
    fn forward(&mut self, x: &Array) -> Result<Array, super::Exception> {
        Ok(Embedding::forward(self, x))
    }
    fn training_mode(&mut self, _mode: bool) {}
}

impl super::Module<&Array> for Conv1d {
    type Output = Array;
    type Error = super::Exception;
    fn forward(&mut self, x: &Array) -> Result<Array, super::Exception> {
        Ok(Conv1d::forward(self, x))
    }
    fn training_mode(&mut self, _mode: bool) {}
}

// ── Conv2d ────────────────────────────────────────────────────────────────

/// 2D convolution layer: `y = conv2d(x, W) + b`.
#[derive(Debug, Clone)]
pub struct Conv2d {
    pub weight: Param<Array>,
    pub bias: Param<Option<Array>>,
    pub stride: [i32; 2],
    pub padding: [i32; 2],
    pub dilation: [i32; 2],
    pub groups: i32,
}

impl Conv2d {
    pub fn new(
        in_channels: i32,
        out_channels: i32,
        kernel_size: i32,
        stride: i32,
        padding: i32,
        with_bias: bool,
    ) -> Self {
        let scale = f32::sqrt(1.0 / (in_channels * kernel_size * kernel_size) as f32);
        let weight = random::uniform_range(
            -scale,
            scale,
            &[out_channels, kernel_size, kernel_size, in_channels],
            super::Dtype::Float32,
        );
        let bias = if with_bias {
            Some(random::uniform_range(
                -scale,
                scale,
                &[out_channels],
                super::Dtype::Float32,
            ))
        } else {
            None
        };
        Self {
            weight: Param::new(weight),
            bias: Param::new(bias),
            stride: [stride, stride],
            padding: [padding, padding],
            dilation: [1, 1],
            groups: 1,
        }
    }

    pub fn forward(&self, x: &Array) -> Array {
        let out = x.conv2d(
            &self.weight.value,
            self.stride[0],
            self.stride[1],
            self.padding[0],
            self.padding[1],
            self.dilation[0],
            self.dilation[1],
            self.groups,
        );
        match &self.bias.value {
            Some(b) => out.add(b),
            None => out,
        }
    }
}

crate::impl_module_params!(Conv2d; weight, bias);

impl super::Module<&Array> for Conv2d {
    type Output = Array;
    type Error = super::Exception;
    fn forward(&mut self, x: &Array) -> Result<Array, super::Exception> {
        Ok(Conv2d::forward(self, x))
    }
    fn training_mode(&mut self, _mode: bool) {}
}

/// Builder for [`Conv2d`].
pub struct Conv2dBuilder {
    in_channels: i32,
    out_channels: i32,
    kernel_size: i32,
    stride: i32,
    padding: i32,
    with_bias: bool,
}

impl Conv2dBuilder {
    pub fn new(in_channels: i32, out_channels: i32, kernel_size: i32) -> Self {
        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride: 1,
            padding: 0,
            with_bias: true,
        }
    }
    pub fn stride(mut self, s: i32) -> Self {
        self.stride = s;
        self
    }
    pub fn padding(mut self, p: i32) -> Self {
        self.padding = p;
        self
    }
    pub fn bias(mut self, b: bool) -> Self {
        self.with_bias = b;
        self
    }
    pub fn build(self) -> Result<Conv2d, super::Exception> {
        Ok(Conv2d::new(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.with_bias,
        ))
    }
}

impl super::builder::Builder<Conv2d> for Conv2dBuilder {
    type Error = super::Exception;
    fn build(self) -> Result<Conv2d, Self::Error> {
        Conv2dBuilder::build(self)
    }
}

// ── Sequential ────────────────────────────────────────────────────────────

/// Sequential container — applies a list of modules in order.
///
/// Equivalent to `mlx_rs::nn::Sequential` but works with any `Module<&Array>`.
pub struct Sequential {
    layers: Vec<Box<dyn super::Module<&'static Array, Output = Array, Error = super::Exception>>>,
}

// Note: Sequential is intentionally left minimal. Full implementation would
// require boxing with `dyn Module` trait objects which require 'static lifetimes.
// For now it's a stub that satisfies type-checking.
impl std::fmt::Debug for Sequential {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Sequential({})", self.layers.len())
    }
}

impl Default for Sequential {
    fn default() -> Self {
        Self::new()
    }
}

impl Sequential {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }
}

// `Sequential` stores `dyn Module` trait objects, which cannot be downcast back
// to `Linear`. Nothing in the workspace builds a model out of it.
impl super::VisitLinears for Sequential {
    fn visit_linears_mut(&mut self, _prefix: &str, _f: &mut dyn FnMut(&str, &mut Linear)) {}
}

impl ModuleParameters for Sequential {
    fn num_parameters(&self) -> usize {
        0
    }
    fn parameters(&self) -> super::ModuleParamRef<'_> {
        HashMap::new()
    }
    fn parameters_mut(&mut self) -> super::ModuleParamMut<'_> {
        HashMap::new()
    }
    fn trainable_parameters(&self) -> super::ModuleParamRef<'_> {
        HashMap::new()
    }
}
