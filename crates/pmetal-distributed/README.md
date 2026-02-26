# pmetal-distributed

Distributed training backend for home clusters on Apple Silicon.

## Overview

This crate provides peer-to-peer distributed training infrastructure designed for small Apple Silicon clusters (2-8 nodes). It features zero-configuration mDNS discovery, ring all-reduce gradient synchronization, and pluggable compression strategies for bandwidth-efficient training.

## Architecture

```
                    ┌──────────────────────────┐
                    │  DistributedContext       │
                    │  (Backend-agnostic API)   │
                    └────────────┬─────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                                     │
              ▼                                     ▼
┌──────────────────────────┐         ┌──────────────────────────┐
│  AutoDiscoveryBackend    │         │  RingBackend             │
│  (Zero-config mDNS)      │         │  (Manual peer list)      │
└────────────┬─────────────┘         └──────────────────────────┘
             │
     ┌───────┼───────┬───────────┐
     ▼       ▼       ▼           ▼
 Identity  Discovery  Topology  Election
 (Ed25519) (libp2p)  (petgraph) (Seniority)
     │       │        │           │
     └───────┼────────┴───────────┘
             ▼
     ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
     │  Transport    │  │  Compression  │  │  Health       │
     │  (TCP Ring)   │  │  (TopK/Quant) │  │  (Heartbeat)  │
     └───────────────┘  └───────────────┘  └───────────────┘
```

## Features

- **Zero-Configuration Discovery**: Automatic peer detection via mDNS/Bonjour on local networks
- **Ring All-Reduce**: Bandwidth-optimal gradient synchronization with scatter-reduce and all-gather phases
- **Persistent Identity**: Ed25519 keypairs stored at `~/.pmetal/node_keypair`
- **Topology Awareness**: Graph-based cluster representation with node capability and connection profiling
- **Master Election**: Seniority-based distributed leader election with PeerId tiebreaking
- **Health Monitoring**: Heartbeat-based peer tracking with exponential moving average latency
- **Gradient Compression**: TopK, random sparsification, FP16/BF16/INT8 quantization, and PowerSGD with error feedback
- **Network Isolation**: SHA3-256 PSK namespacing to prevent cross-cluster communication
- **Metrics**: Counters, gauges, and histograms for all-reduce duration, bytes processed, and failures

## Usage

### Auto-Discovery (Zero-Config)

```rust
use pmetal_distributed::{AutoDiscoveryBackend, DistributedContext};
use std::time::Duration;

let backend = AutoDiscoveryBackend::new().await?;
backend.wait_for_peers(1, Duration::from_secs(30)).await?;
backend.establish_ring().await?;

let ctx = DistributedContext::new(Box::new(backend));
ctx.all_reduce(&mut gradient_buffer).await?;
```

### Manual Configuration

```rust
use pmetal_distributed::{DistributedConfig, RingBackend, DistributedContext};

let config = DistributedConfig::new(
    vec!["192.168.1.10:52416".parse()?, "192.168.1.11:52416".parse()?],
    0, // This node's rank
);

let backend = RingBackend::new(config).await?;
let ctx = DistributedContext::new(Box::new(backend));
```

### Gradient Compression

```rust
use pmetal_distributed::{GradientCompressor, CompressionStrategy};

let mut compressor = GradientCompressor::new(
    CompressionStrategy::TopK { ratio: 0.1 },
    true, // enable error feedback
);

let compressed = compressor.compress(&gradients);
```

## Compression Strategies

| Strategy | Description | Ratio |
|----------|-------------|-------|
| **TopK** | Keep top-k% gradients by magnitude | Configurable |
| **Random** | Probabilistic sparsification | Configurable |
| **FP16** | Half-precision quantization | 2x |
| **BF16** | Brain float quantization | 2x |
| **INT8** | 8-bit quantization | 4x |
| **PowerSGD** | Low-rank approximation | Rank-dependent |

## Collective Operations

| Strategy | Latency | Bandwidth | Best For |
|----------|---------|-----------|----------|
| **Ring** | O(n) | O(1)/node | Large gradients, balanced clusters |
| **Tree** | O(log n) | O(log n)/node | Small messages, low latency |
| **Centralized** | O(n) | O(n)/root | Very small clusters (2-3 nodes) |

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gradient_port` | 52416 | TCP port for gradient exchange |
| `discovery_port` | 52415 | mDNS discovery port |
| `min_peers` | 1 | Minimum peers before ring formation |
| `peer_timeout` | 30s | Discovery timeout |
| `connection_timeout_ms` | 5000 | TCP connection timeout |
| `max_retries` | 3 | Connection retry limit |

## Modules

| Module | Description |
|--------|-------------|
| `auto` | Auto-discovery backend with mDNS |
| `ring` | Ring-based all-reduce for manual configuration |
| `discovery` | libp2p peer discovery service |
| `transport` | TCP transport with connection pooling |
| `collective` | Pluggable collective operation strategies |
| `compression` | Gradient compression with error feedback |
| `topology` | Cluster graph with node and connection profiles |
| `identity` | Persistent Ed25519 keypair management |
| `election` | Distributed master election |
| `health` | Heartbeat-based peer monitoring |
| `namespace` | PSK network isolation |
| `metrics` | Observability counters and gauges |

## License

MIT OR Apache-2.0
