//! The rotary embedding a typed architecture config describes.
//!
//! Each architecture deserializes its own config struct, but the RoPE keys in
//! it (`rope_theta`, `rope_scaling`, `rope_parameters`,
//! `partial_rotary_factor`, `max_position_embeddings`) mean the same thing
//! everywhere. This hands them to [`pmetal_bridge::rope`], the one parser and
//! the one set of frequency formulas every engine uses, and names the model
//! type in its errors (an unknown `rope_type` among them).

use pmetal_bridge::compat::Exception;
use pmetal_bridge::rope::{RopeConfig, RotaryEmbedding};

/// The rotary embedding for `config`'s keys. `config.rope_theta` doubles as
/// the default base when the rope dict carries none.
pub fn rotary_embedding(
    model_type: &str,
    head_dim: i32,
    config: RopeConfig<'_>,
    default_partial_rotary_factor: f64,
    traditional: bool,
) -> Result<RotaryEmbedding, Exception> {
    let theta = config.rope_theta.unwrap_or(10_000.0);
    RotaryEmbedding::from_config(
        head_dim,
        config,
        theta,
        default_partial_rotary_factor,
        traditional,
    )
    .map_err(|e| Exception::custom(format!("{model_type} config: {e}")))
}
