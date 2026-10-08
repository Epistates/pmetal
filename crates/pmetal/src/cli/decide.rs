//! Clap argument struct for `pmetal decide`.

use clap::Args;

/// Thin clap argument struct for `pmetal decide`.
#[derive(Args, Debug)]
pub struct DecideArgs {
    /// Decision model ID or path (a Clef release, e.g. `Cloudflare/clef-flash`)
    #[arg(short, long = "model")]
    pub model: String,

    /// `/v1/systemone` request body: a JSON file, or `-` for stdin. Its
    /// `images` may be file paths, base64 or data URIs; its `videos` lists of
    /// frames, or `{"frames": [...], "fps": N}`
    #[arg(short, long = "request")]
    pub request: String,

    /// Longest prompt in tokens; the state is truncated to fit
    #[arg(long = "max-length", default_value_t = pmetal_models::decision::DEFAULT_MAX_LENGTH)]
    pub max_length: usize,

    /// Print the response on one line instead of indented
    #[arg(long = "compact")]
    pub compact: bool,
}
