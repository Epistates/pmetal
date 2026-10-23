# pmetal cluster

Multi-machine operations: discover peers, form a ring (Thunderbolt links preferred), and run distributed training or inference across it.

Machines find each other over mDNS and libp2p. Every link between ranks is authenticated and encrypted with keys derived from one shared secret, so create it once with `pmetal cluster keygen` and copy it to every machine, the way an SSH key is. A machine reads its secret from the `PMETAL_CLUSTER_SECRET` environment variable (64 hex characters), else from `~/.pmetal/cluster_secret` (readable by its owner only).

`pmetal train` joins a cluster with `--distributed-auto` (discover peers over mDNS) or `--distributed-peers` (explicit `ip:port` addresses), and `--compression-strategy` for the gradient exchange. The command is part of the default build (the `distributed` feature).

## Usage

```bash
pmetal cluster keygen [OPTIONS]
pmetal cluster up [OPTIONS]
pmetal cluster status [OPTIONS]
pmetal cluster bench [OPTIONS]
pmetal cluster pipeline-bench [OPTIONS]
```

## pmetal cluster keygen

Create the cluster secret.

| Option | Default | Description |
|--------|---------|-------------|
| `--output` | `~/.pmetal/cluster_secret` | Where to write the secret |
| `--force` | `false` | Replace an existing secret. Machines still holding the old one can no longer join the cluster |

## pmetal cluster up

Join the cluster: probe the local network interfaces, advertise over mDNS, wait for peers and form a ring. The command keeps the connection open so other nodes can reach this one for training or inference.

| Option | Default | Description |
|--------|---------|-------------|
| `--gradient-port` | `52416` | Port for gradient and activation exchange |
| `--discovery-port` | `52415` | Port for libp2p and mDNS discovery |
| `--min-peers` | `1` | Minimum peer count before forming a ring (excluding this machine) |
| `--timeout` | `60` | Discovery timeout in seconds |
| `--status-interval` | `10` | Print a status snapshot every N seconds while running |

## pmetal cluster status

Print the local network interfaces, the discovered peers and the fabric classification of each link.

| Option | Default | Description |
|--------|---------|-------------|
| `--discovery-port` | `52415` | Discovery port (must match the cluster's) |
| `--gradient-port` | `52416` | Gradient port, used to fill in addresses in the snapshot |
| `--wait` | `30` | Seconds to wait for peer announcements before printing; raise it when bringing up a cold cluster |
| `--json` | `false` | Print JSON instead of a table |
| `--no-discovery` | `false` | Skip discovery and print only the local interfaces' fabric classification |

## pmetal cluster bench

Run an all-reduce micro-benchmark across the ring and report the throughput per fabric.

| Option | Default | Description |
|--------|---------|-------------|
| `--discovery-port` | `52415` | Discovery port |
| `--gradient-port` | `52416` | Gradient port |
| `--timeout` | `60` | Discovery timeout in seconds |
| `--min-peers` | `1` | Minimum peers to wait for |
| `--mb` | `64` | Per-iteration payload in MiB |
| `--iters` | `10` | Number of iterations |
| `--json` | `false` | Print JSON |

## pmetal cluster pipeline-bench

Drive the pipeline-parallel harness end to end across the discovered peers, with a stub forward function. It verifies that activation transport and result loopback work across machines; it does not run a model.

| Option | Default | Description |
|--------|---------|-------------|
| `--discovery-port` | `52415` | Discovery port |
| `--activation-port` | `52417` | Activation TCP port (next and previous in the ring) |
| `--result-port` | `52418` | Result-loopback port (last rank to first rank) |
| `--timeout` | `60` | Discovery timeout in seconds |
| `--min-peers` | `1` | Minimum peers to wait for (excluding this machine) |
| `--layers` | `32` | Layers to spread across the stages (only affects the printed layer ranges) |
| `--tokens` | `16` | Tokens to generate |
| `--hidden-dim` | `4096` | Hidden-state dimension of the stub activations |
| `--vocab` | `32000` | Vocabulary size of the stub logits (sets the payload size from the last rank to the first) |
| `--json` | `false` | Print JSON |

## Examples

```bash
# On one machine: create the secret, then copy ~/.pmetal/cluster_secret to the others
pmetal cluster keygen

# On every machine: look at the interfaces and who is reachable
pmetal cluster status

# On every machine: join and measure the links
pmetal cluster up --min-peers 1
pmetal cluster bench --mb 64 --iters 10

# Train across the ring
pmetal train --model Qwen/Qwen3-0.6B --dataset train.jsonl --distributed-auto
```

## See Also

- [pmetal train](/cli/train/): `--distributed-auto`, `--distributed-peers` and `--compression-strategy`
- [Feature Flags](/configuration/feature-flags/): The `distributed` feature
