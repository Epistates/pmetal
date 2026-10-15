//! Model-side operations over the runtime's arrays.
//!
//! - [`fused_attention`]: scaled dot-product attention with masks, sinks and
//!   logit softcapping
//! - [`gated_delta`]: the gated delta rule (Qwen 3.5's linear attention), with
//!   a chunkwise parallel form for training
//! - [`cut_cross_entropy`]: the LM-head loss without materializing the logits
//! - [`mod@rope`]: rotary embeddings at contiguous or explicit positions
//! - [`fused_moe`]: the combine step of a mixture-of-experts block
//! - [`training_context`]: the training-mode flag the model forwards read
//! - [`utils`]: whether autograd is tracing a set of arrays

pub mod cut_cross_entropy;
pub mod fused_attention;
pub mod fused_moe;
pub mod gated_delta;
pub mod rope;
pub mod training_context;
pub mod utils;

pub use cut_cross_entropy::*;
pub use fused_attention::*;
pub use fused_moe::*;
pub use gated_delta::*;
pub use rope::*;
pub use training_context::*;
pub use utils::*;
