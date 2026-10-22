# pmetal-distributed

Distributed training and inference for home clusters of Apple Silicon Macs.

## Overview

This crate runs the same model across 2-8 Macs joined by Thunderbolt or
Ethernet: ring collectives (all-reduce, reduce-scatter, all-gather, broadcast)
over authenticated, encrypted links; zero-configuration discovery; gradient
compression with error feedback; ZeRO-style optimizer sharding; DiLoCo
low-communication training; pipeline, tensor, expert and context parallelism;
and a local UltraFusion planner that wires same-process stage links over
in-memory channels instead of TCP.

`pmetal-trainer` uses it to synchronise data-parallel gradients (with
compression) and the `pmetal` CLI for the `cluster` subcommands, including the
pipeline benchmark. It depends on
`pmetal-runtime` only for the `mlx-collectives` feature family.

## Security

Anything on the LAN can open a TCP connection to a rank, so a link is trusted
only after a [Noise](https://noiseprotocol.org/) handshake keyed by a 256-bit
**cluster secret**. The secret never crosses the wire; every frame after the
handshake is AEAD-protected, so a machine on the path can neither read
gradients nor inject, alter, replay, reorder or truncate them. Peers also
exchange the ring's membership, so machines configured with different node
lists refuse each other with that reason.

Create the secret once and copy it to every machine, as you would an SSH key:

```text
pmetal cluster keygen                # writes ~/.pmetal/cluster_secret (mode 600)
scp ~/.pmetal/cluster_secret other-mac:.pmetal/cluster_secret
```

A machine looks for its secret in `PMETAL_CLUSTER_SECRET` (64 hex characters),
then in `~/.pmetal/cluster_secret`, and refuses to start without one. Plaintext
links exist for tests and throwaway local runs only, behind an explicit
`allow_plaintext`. Auto-discovery admits a peer to the cluster only once it
proves it holds the secret, so an unauthorised machine is never counted as a
member.

## Architecture

```text
                    ┌──────────────────────────┐
                    │  DistributedContext       │
                    │  (Backend-agnostic API)   │
                    └────────────┬─────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        ▼                        ▼                        ▼
┌───────────────────┐   ┌──────────────────┐   ┌───────────────────────┐
│ AutoDiscovery     │   │ RingBackend      │   │ MlxDistributedBackend │
│ (mDNS + secret)   │   │ (manual peers)   │   │ (MLX ring / JACCL)    │
└─────────┬─────────┘   └────────┬─────────┘   └───────────────────────┘
          └───────────┬──────────┘
                      ▼
            ┌───────────────────┐        ┌───────────────┐
            │ RingCollective    │───────▶│  Compression  │
            │ (one ring impl.)  │        │ (error fdbk.) │
            └─────────┬─────────┘        └───────────────┘
                      ▼
            ┌───────────────────┐
            │ transport         │  Noise handshake, AEAD frames,
            │ (TCP, encrypted)  │  poison-on-failure
            └───────────────────┘
```

## Installation

```toml
[dependencies]
pmetal-distributed = "0.7"
```

## Usage

### Auto-Discovery (Zero-Config)

`all_reduce_f32` reduces a slice of `f32` in place, and takes the reduction op.

```rust,no_run
use std::time::Duration;

use pmetal_distributed::{AutoDiscoveryBackend, DistributedContext, ReduceOp};

async fn sync(gradients: &mut [f32]) -> anyhow::Result<()> {
    // Finds the cluster secret in PMETAL_CLUSTER_SECRET or ~/.pmetal/cluster_secret.
    let backend = AutoDiscoveryBackend::new().await?;
    backend.wait_for_peers(1, Duration::from_secs(30)).await?;
    backend.establish_ring().await?;

    let ctx = DistributedContext::new(Box::new(backend));
    ctx.all_reduce_f32(gradients, ReduceOp::Mean).await
}
```

### Manual Configuration

```rust,no_run
use pmetal_distributed::{ClusterSecret, DistributedConfig, DistributedContext, RingBackend};

async fn manual() -> Result<DistributedContext, Box<dyn std::error::Error>> {
    // Every rank passes the same node list, in the same order.
    let config = DistributedConfig::new(
        vec!["192.168.1.10:52416".parse()?, "192.168.1.11:52416".parse()?],
        0, // This node's rank
    )
    .with_secret(ClusterSecret::resolve()?);

    let backend = RingBackend::new(config).await?;
    Ok(DistributedContext::new(Box::new(backend)))
}
```

### A Ring in One Process

`RingBackend::with_listener` takes a socket the caller has already bound, so a
program that picks free ports (a test, a supervisor) keeps each port from the
moment it chooses it. Two ranks over loopback:

```rust
use std::net::TcpListener;

use pmetal_distributed::{
    ClusterSecret, DistributedConfig, DistributedContext, ReduceOp, RingBackend,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let secret = ClusterSecret::generate();
    let listeners = [TcpListener::bind("127.0.0.1:0")?, TcpListener::bind("127.0.0.1:0")?];
    let nodes = listeners
        .iter()
        .map(|listener| listener.local_addr())
        .collect::<Result<Vec<_>, _>>()?;

    let mut ranks = Vec::new();
    for (rank, listener) in listeners.into_iter().enumerate() {
        let config = DistributedConfig::new(nodes.clone(), rank).with_secret(secret.clone());
        ranks.push(tokio::spawn(async move {
            let backend = RingBackend::with_listener(config, listener).await?;
            let ctx = DistributedContext::new(Box::new(backend));
            let mut gradients = vec![rank as f32 + 1.0; 4];
            ctx.all_reduce_f32(&mut gradients, ReduceOp::Mean).await?;
            anyhow::Ok(gradients)
        }));
    }
    for rank in ranks {
        assert_eq!(rank.await??, [1.5; 4]); // the mean of 1.0 and 2.0
    }
    Ok(())
}
```

### Gradient Compression

Compressed gradients cannot be added on the wire, so ranks exchange them and
each sums what they decode to; every rank ends with the same bits.

```rust,no_run
use pmetal_distributed::{
    CompressionStrategy, DistributedContext, GradientCompressor, ReduceOp,
};

async fn compressed(ctx: &DistributedContext, gradients: &mut [f32]) -> anyhow::Result<()> {
    // Keep one compressor per buffer: its error feedback carries what a step
    // could not send into the next.
    let mut compressor = GradientCompressor::with_seed(
        CompressionStrategy::TopK { ratio: 0.1 },
        true, // error feedback
        ctx.rank() as u64, // random masks differ between ranks
    );
    ctx.all_reduce_compressed(&mut compressor, gradients, ReduceOp::Mean)
        .await
}
```

## Compression Strategies

| Strategy | Description | Ratio |
|----------|-------------|-------|
| **TopK** | Keep top-k% gradients by magnitude | Configurable |
| **Random** | Probabilistic sparsification (unbiased without error feedback) | Configurable |
| **FP16** | Half-precision quantization (saturating) | 2x |
| **BF16** | Brain float quantization | 2x |
| **INT8** | 8-bit quantization, one scale per tensor | 4x |
| **OneBit** | Sign bit with one scale (needs error feedback) | 32x |

Error feedback keeps `transmitted + owed = fed in` for every method.

## Feature flags

None is on by default: the TCP ring, discovery, compression, pipeline inference
and the planners are always built. `full` turns on every flag below.

| Feature | What it does |
|---------|--------------|
| `zero` | ZeRO stage 1/2: optimizer state and update sharded across ranks (`ZeroOptimizer`) |
| `diloco` | DiLoCo: one synchronisation per H inner steps (`DiLoCo`) |
| `pipeline-training` | 1F1B and zero-bubble schedules, and a simulator that checks them |
| `mlx-collectives` | MLX's collectives (ring, JACCL over Thunderbolt RDMA) as a backend (`MlxDistributedBackend`); pulls in `pmetal-runtime` |
| `tensor-parallel` | Weight sharding and column/row-parallel linear layers |
| `context-parallel` | Ring attention (pass-KV, pass-Q; causal; zig-zag layout) |
| `expert-parallel` | Mixture-of-experts dispatch (all-to-all) and capacity control |

The MLX-based features run as separate processes joined by MLX's own
environment (`MLX_RANK`, `MLX_HOSTFILE`, ...); the TCP ring does not need it.

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gradient_port` | 52416 | TCP port for gradient exchange |
| `discovery_port` | 52415 | mDNS discovery port |
| `min_peers` | 1 | Minimum peers before ring formation |
| `peer_timeout` | 60s | Discovery timeout |
| `connection_timeout_ms` | 30000 | Ring connection timeout |
| `max_retries` | 50 | Connection attempts, across all of a peer's addresses |
| `secret` | resolved | The cluster secret (environment, then file) |
| `allow_plaintext` | `false` | Unauthenticated links, for tests only |

## Modules

| Module | Description |
|--------|-------------|
| `auto` | Auto-discovery backend with mDNS |
| `ring` | Ring backend for manual configuration |
| `collective` | The ring algorithms: all-reduce, reduce-scatter, all-gather, broadcast, barrier |
| `transport` | Authenticated, encrypted TCP links |
| `secret` | The cluster secret |
| `discovery` | libp2p peer discovery service |
| `compression` | Gradient compression with error feedback |
| `topology` | Cluster graph with node and connection profiles |
| `layer_assignment` | Layer partitioning within memory, and ring ordering by link speed |
| `identity` | Persistent Ed25519 keypair management |
| `namespace` | Discovery protocol namespacing (not a security boundary) |
| `metrics` | Observability counters and gauges |
| `fabric` | What kind of link joins two peers (Thunderbolt bridge, Ethernet, Wi-Fi) and its score |
| `apportion` | Splitting indivisible items (layers, experts) over nodes in proportion to weights |
| `pipeline`, `pipeline_harness`, `activation_transport`, `activation_codec` | Pipeline-parallel inference: stage runtime, the activations passed between stages, their codec |
| `ultrafusion` | Planning a dual-die Mac as a local multi-stage pipeline over in-memory channels |
| `expert_shard` | The one expert-to-rank mapping that expert and tensor parallelism share |
| `cluster_runtime` | The short-lived discovery session behind the `pmetal cluster` subcommands |
| `config`, `error` | `DistributedConfig`, `DistributedError` |

## Requirements

- Rust 1.95 or later
- macOS on Apple Silicon for the `mlx-collectives` features (they drive `pmetal-runtime`); the rest is plain Rust
- Every machine in a cluster needs the same cluster secret

## License

MIT OR Apache-2.0
