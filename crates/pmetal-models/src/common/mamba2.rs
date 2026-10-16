//! The Mamba-2 mixer, shared by every architecture that has one.
//!
//! Nemotron-H and Granite 4.0-H run the same block: `in_proj` produces
//! `[gate | x B C | dt]`, `x B C` goes through a causal depthwise convolution
//! and SiLU, the selective scan ([`pmetal_runtime::nn::ssd`]) mixes the
//! sequence, and the scan's output is multiplied by `silu(gate)`, normalized
//! and projected out. What differs is only configuration: how wide a
//! normalization group is, and the floor of the step size.
//!
//! Field and parameter names are the released checkpoints' (`in_proj`,
//! `conv1d`, `A_log`, `D`, `dt_bias`, `norm.weight`, `out_proj`), so a state
//! dict loads and saves without a rename table.
//!
//! Everything after the convolution runs in float32, whatever the model's
//! dtype, and the SSM state lives in the cache in float32: the recurrence
//! multiplies a state by decays hundreds of times and a bf16 state is a bf16
//! rounding per token.

use std::collections::HashMap;

use pmetal_cache::kv_cache::MambaCacheEntry;
use pmetal_runtime::compat::{Array, Dtype, Exception, Module, Param, nn, ops};
use pmetal_runtime::impl_module_params;
use pmetal_runtime::nn::ssd::{
    SsdParams, gated_group_rms_norm, ssd_scan, ssd_scan_segmented, ssd_step,
};

/// Everything that shapes one Mamba-2 mixer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mamba2Config {
    pub hidden_size: i32,
    pub num_heads: i32,
    pub head_dim: i32,
    pub n_groups: i32,
    pub state_size: i32,
    pub conv_kernel: i32,
    /// Tokens per chunk of the scan. Any value gives the same output up to
    /// rounding; it sets how much of the work is dense matmul.
    pub chunk_size: i32,
    /// Channels per normalization group of the gated norm: the width over
    /// `n_groups` for Nemotron-H, the whole width for Granite.
    pub norm_group_size: i32,
    pub norm_eps: f32,
    /// Floor of the step size after the softplus (`0.0` for none).
    pub dt_min: f32,
    /// Ceiling of the step size (`f32::INFINITY` for none).
    pub dt_max: f32,
    pub proj_bias: bool,
    pub conv_bias: bool,
}

impl Mamba2Config {
    /// `num_heads * head_dim`, the width of the scan.
    pub fn intermediate_size(&self) -> i32 {
        self.num_heads * self.head_dim
    }

    /// The channels the convolution runs over: `x`, `B` and `C`.
    pub fn conv_dim(&self) -> i32 {
        self.intermediate_size() + 2 * self.n_groups * self.state_size
    }

    /// Refuse a shape the scan cannot be built on.
    pub fn validate(&self) -> Result<(), Exception> {
        let ok = self.num_heads > 0
            && self.head_dim > 0
            && self.n_groups > 0
            && self.num_heads % self.n_groups == 0
            && self.state_size > 0
            && self.conv_kernel > 0
            && self.chunk_size > 0
            && self.norm_group_size > 0
            && self.intermediate_size() % self.norm_group_size == 0;
        if ok {
            Ok(())
        } else {
            Err(Exception::custom(format!(
                "invalid Mamba-2 shape: {self:?}"
            )))
        }
    }
}

/// The gated RMS norm's scale, `norm.weight`.
#[derive(Debug)]
pub struct Mamba2Norm {
    pub weight: Param<Array>,
}
impl_module_params!(Mamba2Norm; weight);

/// A Mamba-2 mixer.
#[allow(non_snake_case)]
#[derive(Debug)]
pub struct Mamba2Mixer {
    pub in_proj: nn::Linear,
    /// Depthwise causal conv, MLX layout `[conv_dim, kernel, 1]`. A released
    /// `[conv_dim, 1, kernel]` weight is reshaped into it at load
    /// ([`conv_weights_to_runtime_layout`]).
    pub conv1d: nn::Conv1d,
    pub dt_bias: Param<Array>,
    pub A_log: Param<Array>,
    pub D: Param<Array>,
    pub norm: Mamba2Norm,
    pub out_proj: nn::Linear,
    pub config: Mamba2Config,
}
impl_module_params!(Mamba2Mixer; in_proj, conv1d, dt_bias, A_log, D, norm, out_proj);

impl Mamba2Mixer {
    pub fn new(config: Mamba2Config) -> Result<Self, Exception> {
        config.validate()?;
        let intermediate = config.intermediate_size();
        let conv_dim = config.conv_dim();
        let in_proj = nn::LinearBuilder::new(
            config.hidden_size,
            intermediate + conv_dim + config.num_heads,
        )
        .bias(config.proj_bias)
        .build()?;
        let conv1d = nn::Conv1dBuilder::new(conv_dim, conv_dim, config.conv_kernel)
            .groups(conv_dim)
            .bias(config.conv_bias)
            .padding(0)
            .build()?;
        let out_proj = nn::LinearBuilder::new(intermediate, config.hidden_size)
            .bias(config.proj_bias)
            .build()?;

        // The reference's own initialisation: `A = arange(1, H + 1)`, unit
        // `D` and `dt_bias`. A loaded checkpoint overwrites all three.
        let a: Vec<f32> = (1..=config.num_heads).map(|h| (h as f32).ln()).collect();
        Ok(Self {
            in_proj,
            conv1d,
            dt_bias: Param::new(Array::ones_f32(&[config.num_heads])),
            A_log: Param::new(Array::from_slice(&a, &[config.num_heads])),
            D: Param::new(Array::ones_f32(&[config.num_heads])),
            norm: Mamba2Norm {
                weight: Param::new(Array::ones_f32(&[intermediate])),
            },
            out_proj,
            config,
        })
    }

    /// Mix `x` `[B, L, hidden]`, advancing `cache` when one is given.
    ///
    /// With no cache the sequence starts from zero conv and SSM state. With
    /// one, the conv reads the entry's last `kernel - 1` inputs and the scan
    /// starts from its SSM state; both are written back. One token against
    /// existing state takes the recurrent step, anything else the chunked
    /// scan.
    pub fn forward(
        &mut self,
        x: &Array,
        cache: Option<&mut MambaCacheEntry>,
    ) -> Result<Array, Exception> {
        self.mix(x, cache, None)
    }

    /// [`forward`](Self::forward) over a row that packs several sequences:
    /// `segments` `[B, L]` numbers the sequence of each token
    /// ([`segments_from_positions`]). Neither the convolution nor the scan
    /// reads across a boundary, so each sequence is mixed as if it were alone;
    /// the attention mask of a packed batch cannot reach this layer. There is
    /// no cache: a packed row is a training input.
    pub fn forward_packed(&mut self, x: &Array, segments: &Array) -> Result<Array, Exception> {
        self.mix(x, None, Some(segments))
    }

    fn mix(
        &mut self,
        x: &Array,
        mut cache: Option<&mut MambaCacheEntry>,
        segments: Option<&Array>,
    ) -> Result<Array, Exception> {
        let cfg = self.config;
        let (batch, seq_len) = (x.dim(0), x.dim(1));
        let input_dtype = x.dtype();
        let intermediate = cfg.intermediate_size();

        let projected = Module::forward(&mut self.in_proj, x)?;
        let parts = ops::split_sections(
            &projected,
            &[intermediate, intermediate + cfg.conv_dim()],
            -1,
        );
        let (gate, xbc, dt) = (&parts[0], &parts[1], &parts[2]);

        // Causal depthwise conv: zero history for a fresh sequence, the
        // cached history otherwise.
        let conv = match segments {
            Some(segments) => self.packed_conv(xbc, segments),
            None => {
                let padded = match cache.as_deref_mut() {
                    Some(entry) => entry.update_conv_state(xbc, cfg.conv_kernel)?,
                    None => ops::pad(
                        xbc,
                        &[(0, 0), (cfg.conv_kernel - 1, 0), (0, 0)],
                        None,
                        Some(0.0),
                    ),
                };
                let conv = Module::forward(&mut self.conv1d, &padded)?;
                ops::slice_axis(&conv, 1, conv.dim(1) - seq_len, conv.dim(1))
            }
        };
        let conv = nn::silu(&conv);

        let bc = cfg.n_groups * cfg.state_size;
        let conv_parts = ops::split_sections(&conv, &[intermediate, intermediate + bc], -1);
        let xs = conv_parts[0].reshape(&[batch, seq_len, cfg.num_heads, cfg.head_dim]);
        let b = conv_parts[1].reshape(&[batch, seq_len, cfg.n_groups, cfg.state_size]);
        let c = conv_parts[2].reshape(&[batch, seq_len, cfg.n_groups, cfg.state_size]);

        let params = SsdParams {
            a_log: self.A_log.as_ref(),
            d: self.D.as_ref(),
            dt_bias: self.dt_bias.as_ref(),
            dt_min: cfg.dt_min,
            dt_max: cfg.dt_max,
        };
        let previous = cache
            .as_ref()
            .and_then(|entry| entry.get_ssm_state().cloned());
        let (y, state) = match (segments, seq_len, previous.as_ref()) {
            (Some(segments), _, state) => {
                ssd_scan_segmented(&xs, dt, &b, &c, params, state, cfg.chunk_size, segments)?
            }
            (None, 1, Some(state)) => ssd_step(&xs, dt, &b, &c, params, state)?,
            (None, _, state) => ssd_scan(&xs, dt, &b, &c, params, state, cfg.chunk_size)?,
        };
        if let Some(entry) = cache {
            entry.set_ssm_state(state);
        }

        let y = y.reshape(&[batch, seq_len, intermediate]);
        let normed = gated_group_rms_norm(
            &y,
            Some(gate),
            self.norm.weight.as_ref(),
            cfg.norm_eps,
            cfg.norm_group_size,
        )?;
        Module::forward(&mut self.out_proj, &normed.as_dtype(input_dtype.as_i32()))
    }

    /// The causal depthwise convolution of a packed row, `[B, L, C]` to the
    /// same: tap `k` of an output reads the input `kernel - 1 - k` tokens
    /// back only when that token is in the same sequence, so no sequence sees
    /// the inputs of the one before it (the zeros before the row stand in for
    /// the inputs that are not there). Accumulated in float32.
    fn packed_conv(&self, xbc: &Array, segments: &Array) -> Array {
        let cfg = self.config;
        let (batch, seq_len, channels) = (xbc.dim(0), xbc.dim(1), xbc.dim(2));
        let taps = cfg.conv_kernel;
        let history = taps - 1;
        let f32 = Dtype::Float32;

        let padded = ops::pad(
            &xbc.as_dtype(f32.as_i32()),
            &[(0, 0), (history, 0), (0, 0)],
            None,
            Some(0.0),
        );
        // No sequence has a negative number, so the padding matches nothing.
        let before = ops::full(&[batch, history], -1.0, Dtype::Int32);
        let numbered = ops::concatenate_axis(&[&before, segments], 1);
        // [C, K, 1] -> [K, C]
        let weights = self
            .conv1d
            .weight
            .as_ref()
            .as_dtype(f32.as_i32())
            .reshape(&[channels, taps])
            .transpose_axes(&[1, 0]);

        let mut out: Option<Array> = None;
        for tap in 0..taps {
            let inputs = ops::slice_axis(&padded, 1, tap, tap + seq_len);
            let same = segments
                .equal(&ops::slice_axis(&numbered, 1, tap, tap + seq_len))
                .as_dtype(f32.as_i32())
                .reshape(&[batch, seq_len, 1]);
            let weight = ops::slice_axis(&weights, 0, tap, tap + 1); // [1, C]
            let term = inputs.multiply(&same).multiply(&weight);
            out = Some(match out {
                Some(sum) => sum.add(&term),
                None => term,
            });
        }
        let mut out = out.expect("a convolution has at least one tap");
        if let Some(bias) = self.conv1d.bias.as_ref() {
            out = out.add(&bias.as_dtype(f32.as_i32()));
        }
        out.as_dtype(xbc.dtype().as_i32())
    }
}

/// The sequence number of every token of a packed row, from its position ids:
/// the numbers count the tokens at position 0, each of which starts a
/// sequence. `positions` is `[L]` (shared by every row) or `[B, L]`; the
/// result is `[B, L]`, `int32`.
pub fn segments_from_positions(positions: &Array, batch: i32) -> Array {
    let starts = positions
        .equal(&Array::from_i32(0))
        .as_dtype(Dtype::Int32.as_i32());
    let numbers = starts.cumsum(-1);
    if numbers.ndim() == 1 {
        let len = numbers.dim(0);
        numbers.reshape(&[1, len]).broadcast_to(&[batch, len])
    } else {
        numbers
    }
}

/// A released depthwise conv weight is `[C, 1, K]`; the runtime's conv wants
/// `[C, K, 1]`. The middle axis is a singleton, so this is a reshape, not a
/// transpose. An already converted `[C, K, 1]` weight is left alone.
pub fn conv_weights_to_runtime_layout(weights: &mut HashMap<String, Array>) {
    for (key, value) in weights.iter_mut() {
        if key.ends_with(".conv1d.weight") && value.ndim() == 3 && value.dim(1) == 1 {
            let (channels, kernel) = (value.dim(0), value.dim(2));
            *value = value.reshape(&[channels, kernel, 1]);
        }
    }
}
