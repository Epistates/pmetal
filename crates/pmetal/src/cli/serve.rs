//! Clap argument struct for `pmetal serve`.

use clap::Args;
use pmetal_core::jobs::ServeSpec;

/// Start an OpenAI- and Anthropic-compatible inference server.
#[derive(Args, Debug)]
pub struct ServeArgs {
    /// Model ID or path
    #[arg(short, long = "model")]
    pub model: String,

    /// Port to listen on
    #[arg(short, long = "port", default_value_t = 8080)]
    pub port: u16,

    /// Host to bind to
    #[arg(long = "host", default_value = "127.0.0.1")]
    pub host: String,

    /// API key clients must send (`Authorization: Bearer <key>` or
    /// `x-api-key: <key>`); repeat for several. The PMETAL_API_KEY environment
    /// variable (comma-separated) sets the same, without the key appearing in
    /// the process list. With none, the server is open: bind it to loopback.
    #[arg(long = "api-key")]
    pub api_key: Vec<String>,

    /// Browser origin whose pages may call the server (CORS), or `*` for
    /// any; repeat for several. Without one, no CORS headers are sent.
    #[arg(long = "cors-origin")]
    pub cors_origin: Vec<String>,

    /// Seconds one request may run before the server ends it (a 504, or an
    /// error frame in a stream); 0 for no limit.
    #[arg(long = "request-timeout", default_value_t = 1800)]
    pub request_timeout: u64,

    /// Context length the server holds per sequence (prompt plus output).
    /// Default: the model's context window (max_position_embeddings, or the
    /// YaRN-stretched length), capped by what the device's memory holds for
    /// the weights plus that much KV cache, and never under 4096.
    #[arg(long = "max-seq-len")]
    pub max_seq_len: Option<usize>,

    /// Path to packed expert weights directory for SSD-offloaded MoE inference.
    #[arg(long = "experts-dir")]
    pub experts_dir: Option<String>,

    /// Quantize weights to FP8 E4M3 at load time (~2x memory savings).
    #[arg(long = "fp8")]
    pub fp8: bool,

    /// KV cache quantization bits: 8 = q8_0, 4 = q4_0, 0 = fp16.
    #[arg(long = "kv-quant")]
    pub kv_quant: Option<u8>,

    /// Disable KV cache quantization (use fp16 KV cache).
    #[arg(long = "no-kv-quant")]
    pub no_kv_quant: bool,

    /// Quantization group size for KV cache.
    #[arg(long = "kv-group-size", default_value_t = 64)]
    pub kv_group_size: usize,

    /// Enable TurboQuant KV cache compression.
    #[arg(long = "kv-turboquant")]
    pub kv_turboquant: bool,

    /// TurboQuant quality preset: q2_5 or q3_5.
    #[arg(long = "kv-turboquant-preset", value_parser = ["q2_5", "q3_5"])]
    pub kv_turboquant_preset: Option<String>,

    /// Generate on the ANE (Apple Neural Engine) (experimental). A model the
    /// ANE cannot run is an error, not a quiet GPU run.
    #[arg(long = "ane")]
    pub ane: bool,

    /// Largest context (prompt plus output, in tokens) the ANE compiles a
    /// model for. Each request gets the smallest power of two from 512 that
    /// fits it, up to this; a smaller context runs faster.
    #[arg(long = "ane-max-seq-len", default_value_t = 4096)]
    pub ane_max_seq_len: usize,

    /// Draft model (HF id or local path) that speeds decoding up without
    /// changing the output: a Gemma 4 assistant, or a DFlash draft model. The
    /// target model checks every token, so a reply is what the model alone
    /// gives (greedy) or is drawn from the same distribution (sampled). With
    /// --ane, a DFlash draft model drafts on the GPU for the model on the ANE.
    #[arg(long = "draft-model")]
    pub draft_model: Option<String>,

    /// Draft with a Qwen3.5/3.6/3.8 model's own multi-token-prediction head,
    /// the mtp.* weights bundled in the model. Not with --draft-model.
    #[arg(long = "mtp")]
    pub mtp: bool,

    /// Qwen MTP checkpoint directory; defaults to the mtp.* weights bundled
    /// in the model.
    #[arg(long = "mtp-model")]
    pub mtp_model: Option<String>,

    /// Tokens a draft round proposes (default: 3 for MTP, the assistant's own
    /// setting, the DFlash block size).
    #[arg(long = "draft-tokens")]
    pub draft_tokens: Option<usize>,

    /// Enable continuous batching.
    #[arg(long = "continuous-batch")]
    pub continuous_batch: bool,

    /// Maximum concurrent slots when --continuous-batch is set.
    #[arg(long = "cb-max-slots", default_value_t = 8)]
    pub cb_max_slots: usize,

    /// Maximum pending-queue depth when --continuous-batch is set.
    #[arg(long = "cb-max-queue-depth", default_value_t = 256)]
    pub cb_max_queue_depth: usize,

    /// Token block size the continuous-batching admission counts KV memory in.
    #[arg(long = "cb-block-size", default_value_t = 32)]
    pub cb_block_size: usize,

    /// Maximum active token blocks for continuous batching (0 = what the
    /// device can hold beside the model).
    #[arg(long = "cb-max-blocks", default_value_t = 0)]
    pub cb_max_blocks: usize,

    /// Decode each request on its own cache, one forward per request, so a
    /// reply is bit for bit what the request gets alone. By default a model
    /// with a batched forward decodes all requests together, several times
    /// faster under load, and a request that shares its batch can differ from
    /// its solo reply in the last bit of a logit (a greedy near-tie may fall
    /// the other way).
    #[arg(long = "cb-exact")]
    pub cb_exact: bool,
}

impl ServeArgs {
    /// The spec these flags describe, and the API keys, which stay out of it:
    /// a spec becomes a command line (the MCP server's, the TUI's), and a key
    /// on a command line shows in every process list.
    pub fn into_spec(self) -> (ServeSpec, Vec<String>) {
        let spec = ServeSpec {
            model: self.model,
            host: self.host,
            port: self.port,
            cors_origin: self.cors_origin,
            request_timeout: self.request_timeout,
            max_seq_len: self.max_seq_len,
            experts_dir: self.experts_dir,
            fp8: self.fp8,
            kv_quant: self.kv_quant,
            no_kv_quant: self.no_kv_quant,
            kv_group_size: self.kv_group_size,
            kv_turboquant: self.kv_turboquant,
            kv_turboquant_preset: self.kv_turboquant_preset,
            ane: self.ane,
            ane_max_seq_len: self.ane_max_seq_len,
            draft_model: self.draft_model,
            mtp: self.mtp,
            mtp_model: self.mtp_model,
            draft_tokens: self.draft_tokens,
            continuous_batch: self.continuous_batch,
            cb_max_slots: self.cb_max_slots,
            cb_max_queue_depth: self.cb_max_queue_depth,
            cb_block_size: self.cb_block_size,
            cb_max_blocks: self.cb_max_blocks,
            cb_exact: self.cb_exact,
        };
        (spec, self.api_key)
    }
}
