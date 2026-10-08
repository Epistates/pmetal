//! Clap argument structs for `pmetal cluster …` subcommands.
//!
//! All cluster subcommands are gated behind the `distributed` feature.
//! See `commands/cluster.rs` for the implementations.

use clap::{Args, Subcommand};

#[derive(Args, Debug)]
pub struct ClusterArgs {
    #[command(subcommand)]
    pub command: ClusterSubcommand,
}

#[derive(Subcommand, Debug)]
pub enum ClusterSubcommand {
    /// Join the cluster: probe local NICs, advertise via mDNS, wait for
    /// peers, and form a ring. Holds the connection open so other nodes
    /// can reach this node for training/inference.
    Up(UpArgs),

    /// Print local interfaces, discovered peers, and per-edge fabric
    /// classification.
    Status(StatusArgs),

    /// Run an all-reduce micro-benchmark across the ring and report
    /// throughput per fabric.
    Bench(BenchArgs),

    /// Drive the pipeline-parallel harness end-to-end across discovered
    /// peers using a stub forward function. Verifies that activation
    /// transport + result loopback work correctly across machines.
    /// (The model integration is wired separately in the inference engine.)
    PipelineBench(PipelineBenchArgs),
}

#[derive(Args, Debug)]
pub struct UpArgs {
    /// Port for gradient/activation exchange.
    #[arg(long = "gradient-port", default_value = "52416")]
    pub gradient_port: u16,

    /// Port for libp2p / mDNS discovery.
    #[arg(long = "discovery-port", default_value = "52415")]
    pub discovery_port: u16,

    /// Minimum peer count before forming a ring (excluding self).
    #[arg(long = "min-peers", default_value = "1")]
    pub min_peers: usize,

    /// Discovery timeout in seconds.
    #[arg(long = "timeout", default_value = "60")]
    pub timeout_secs: u64,

    /// Print status snapshot every N seconds while running.
    #[arg(long = "status-interval", default_value = "10")]
    pub status_interval_secs: u64,
}

#[derive(Args, Debug)]
pub struct StatusArgs {
    /// Discovery port (must match the port the cluster uses).
    #[arg(long = "discovery-port", default_value = "52415")]
    pub discovery_port: u16,

    /// Gradient port (used to populate addresses in the status snapshot).
    #[arg(long = "gradient-port", default_value = "52416")]
    pub gradient_port: u16,

    /// How long to wait for peer announcements before printing.
    /// 30s gives a freshly-booted neighbour enough time to come up on
    /// Wi-Fi/Ethernet; bump higher when bringing up a cold cluster.
    #[arg(long = "wait", default_value = "30")]
    pub wait_secs: u64,

    /// Output as JSON instead of a human-readable table.
    #[arg(long)]
    pub json: bool,

    /// Skip discovery — just print local interface fabric classification.
    #[arg(long = "no-discovery")]
    pub no_discovery: bool,
}

#[derive(Args, Debug)]
pub struct PipelineBenchArgs {
    /// Discovery port.
    #[arg(long = "discovery-port", default_value = "52415")]
    pub discovery_port: u16,

    /// Activation TCP port (next/prev ring).
    #[arg(long = "activation-port", default_value = "52417")]
    pub activation_port: u16,

    /// Result-loopback port (last-rank → first-rank tokens).
    #[arg(long = "result-port", default_value = "52418")]
    pub result_port: u16,

    /// Discovery timeout in seconds.
    #[arg(long = "timeout", default_value = "60")]
    pub timeout_secs: u64,

    /// Minimum peers to wait for (excluding self).
    #[arg(long = "min-peers", default_value = "1")]
    pub min_peers: usize,

    /// Number of layers to spread across stages (only affects layer-range
    /// printout for the stub forward function).
    #[arg(long = "layers", default_value = "32")]
    pub num_layers: usize,

    /// Tokens to generate.
    #[arg(long = "tokens", default_value = "16")]
    pub tokens: usize,

    /// Hidden-state dimension for the stub activations.
    #[arg(long = "hidden-dim", default_value = "4096")]
    pub hidden_dim: usize,

    /// Vocabulary size for the stub logits (sets payload size last-rank → first).
    #[arg(long = "vocab", default_value = "32000")]
    pub vocab: usize,

    /// Output as JSON.
    #[arg(long)]
    pub json: bool,
}

#[derive(Args, Debug)]
pub struct BenchArgs {
    /// Discovery port.
    #[arg(long = "discovery-port", default_value = "52415")]
    pub discovery_port: u16,

    /// Gradient port.
    #[arg(long = "gradient-port", default_value = "52416")]
    pub gradient_port: u16,

    /// Discovery timeout.
    #[arg(long = "timeout", default_value = "60")]
    pub timeout_secs: u64,

    /// Per-iteration payload size in MiB.
    #[arg(long = "mb", default_value = "64")]
    pub payload_mb: usize,

    /// Number of iterations.
    #[arg(long = "iters", default_value = "10")]
    pub iters: usize,

    /// Minimum peers to wait for.
    #[arg(long = "min-peers", default_value = "1")]
    pub min_peers: usize,

    /// Output as JSON.
    #[arg(long)]
    pub json: bool,
}
