//! Implementations of `pmetal cluster …` subcommands.
//!
//! Each subcommand spins up its own [`AutoDiscoveryBackend`], does its job,
//! and exits. Multiple machines invoking the same subcommand at the same
//! time will discover each other via mDNS — there is no daemon.

use anyhow::{Context, Result};
use pmetal_distributed::DistributedBackend;
use pmetal_distributed::activation_codec::ActivationCodec;
use pmetal_distributed::activation_transport::DtypeTag;
use pmetal_distributed::auto::{AutoDiscoveryBackend, AutoDiscoveryConfig};
use pmetal_distributed::cluster_runtime::{
    BenchSample, ClusterStatus, join_cluster, median_gbps, run_allreduce_bench, snapshot_status,
};
use pmetal_distributed::fabric::probe_local_fabric;
use pmetal_distributed::pipeline::PipelineGenerationLoop;
use pmetal_distributed::pipeline_harness::{PipelineHarness, PipelineHarnessConfig};
use pmetal_distributed::topology::NodeProfile;
use std::sync::Arc;
use std::time::Duration;

use crate::cli::cluster::{BenchArgs, ClusterSubcommand, PipelineBenchArgs, StatusArgs, UpArgs};

pub async fn run(cmd: ClusterSubcommand) -> Result<()> {
    match cmd {
        ClusterSubcommand::Up(args) => run_up(args).await,
        ClusterSubcommand::Status(args) => run_status(args).await,
        ClusterSubcommand::Bench(args) => run_bench(args).await,
        ClusterSubcommand::PipelineBench(args) => run_pipeline_bench(args).await,
    }
}

async fn run_up(args: UpArgs) -> Result<()> {
    let cfg = AutoDiscoveryConfig {
        gradient_port: args.gradient_port,
        discovery_port: args.discovery_port,
        min_peers: args.min_peers,
        peer_timeout: Duration::from_secs(args.timeout_secs),
        profile: NodeProfile::default(),
    };

    println!(
        "pmetal cluster up — joining cluster on ports {} (discovery) / {} (gradient)",
        args.discovery_port, args.gradient_port
    );

    // Stand the backend up first so mDNS announcements start immediately;
    // peers can then discover *us* even while we're still waiting for them.
    let backend = std::sync::Arc::new(
        AutoDiscoveryBackend::with_config(cfg)
            .await
            .context("failed to start auto-discovery backend")?,
    );

    // Discovery retry loop. Each iteration waits up to `timeout_secs` for
    // `min_peers` peers; on timeout we don't bail — we re-announce and try
    // again. Only Ctrl+C exits without a ring. This handles the common case
    // of the user starting `cluster up` on machine A several minutes before
    // machine B comes online.
    let per_attempt = Duration::from_secs(args.timeout_secs.max(5));
    let mut attempt = 1usize;
    loop {
        tokio::select! {
            res = backend.wait_for_peers(args.min_peers, per_attempt) => {
                match res {
                    Ok(found) => {
                        println!(
                            "Found {} peer(s) (≥ {} required). Establishing ring...",
                            found, args.min_peers
                        );
                        if let Err(e) = backend.establish_ring().await {
                            eprintln!(
                                "Ring establishment failed (attempt {}): {e}. Retrying discovery...",
                                attempt
                            );
                            attempt += 1;
                            continue;
                        }
                        break;
                    }
                    Err(e) => {
                        println!(
                            "Discovery attempt {} timed out after {}s ({e}). Retrying — peers may still be coming online...",
                            attempt, per_attempt.as_secs()
                        );
                        attempt += 1;
                    }
                }
            }
            _ = tokio::signal::ctrl_c() => {
                println!("\nReceived Ctrl+C before ring formed, leaving.");
                return Ok(());
            }
        }
    }

    println!("Ring established. Holding connection open; press Ctrl+C to leave.\n");
    print_status_snapshot(&backend, /*json=*/ false);

    let interval = Duration::from_secs(args.status_interval_secs.max(1));
    let mut ticker = tokio::time::interval(interval);
    ticker.tick().await; // first tick is immediate

    loop {
        tokio::select! {
            _ = ticker.tick() => {
                print_status_snapshot(&backend, /*json=*/ false);
            }
            _ = tokio::signal::ctrl_c() => {
                println!("\nReceived Ctrl+C, leaving cluster.");
                return Ok(());
            }
        }
    }
}

async fn run_status(args: StatusArgs) -> Result<()> {
    if args.no_discovery {
        let fabric = probe_local_fabric();
        if args.json {
            let local: Vec<_> = fabric
                .interfaces()
                .iter()
                .map(|i| {
                    serde_json::json!({
                        "name": i.name,
                        "kind": i.kind.tag(),
                        "addrs": i.addrs.iter().map(|a| a.to_string()).collect::<Vec<_>>(),
                    })
                })
                .collect();
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "local_interfaces": local,
                }))?
            );
        } else {
            println!("Local fabric (no discovery):");
            for i in fabric.interfaces() {
                let addrs = i
                    .addrs
                    .iter()
                    .map(|a| a.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                println!("  {:<10} {:<12}  {}", i.name, i.kind.tag(), addrs);
            }
        }
        return Ok(());
    }

    let cfg = AutoDiscoveryConfig {
        gradient_port: args.gradient_port,
        discovery_port: args.discovery_port,
        min_peers: 0,
        peer_timeout: Duration::from_secs(args.wait_secs),
        profile: NodeProfile::default(),
    };

    let backend = Arc::new(AutoDiscoveryBackend::with_config(cfg).await?);
    // Give peers a few seconds to announce themselves.
    let _ = backend
        .wait_for_peers(0, Duration::from_secs(args.wait_secs))
        .await;
    // Drain the discovery queue so PeerConnected events get folded in.
    tokio::time::sleep(Duration::from_millis(200)).await;

    print_status_snapshot(&backend, args.json);
    Ok(())
}

async fn run_bench(args: BenchArgs) -> Result<()> {
    let cfg = AutoDiscoveryConfig {
        gradient_port: args.gradient_port,
        discovery_port: args.discovery_port,
        min_peers: args.min_peers,
        peer_timeout: Duration::from_secs(args.timeout_secs),
        profile: NodeProfile::default(),
    };

    println!(
        "pmetal cluster bench — discovering peers (timeout {}s, min={})",
        args.timeout_secs, args.min_peers
    );

    let backend = join_cluster(cfg, args.min_peers, Duration::from_secs(args.timeout_secs))
        .await
        .context("failed to join cluster for bench")?;

    let world = backend.world_size();
    println!(
        "Ring formed: rank {}/{}. Running {} all-reduce iterations of {} MiB...",
        backend.rank(),
        world,
        args.iters,
        args.payload_mb
    );

    let samples: Vec<BenchSample> =
        run_allreduce_bench(backend.as_ref(), args.payload_mb, args.iters)
            .await
            .context("all-reduce bench failed")?;

    let median = median_gbps(&samples);
    let min = samples
        .iter()
        .map(|s| s.gbps())
        .fold(f64::INFINITY, f64::min);
    let max = samples
        .iter()
        .map(|s| s.gbps())
        .fold(f64::NEG_INFINITY, f64::max);

    let median_link = if let Some(s) = samples
        .iter()
        .find(|s| (s.gbps() - median).abs() < 1e-9)
        .or(samples.first())
    {
        s.link_gbps()
    } else {
        0.0
    };

    if args.json {
        let json = serde_json::json!({
            "rank": backend.rank(),
            "world_size": world,
            "payload_mb": args.payload_mb,
            "iters": args.iters,
            "median_allreduce_gbps": median,
            "median_link_gbps": median_link,
            "min_gbps": min,
            "max_gbps": max,
            "samples": samples,
        });
        println!("{}", serde_json::to_string_pretty(&json)?);
    } else {
        println!("\nResults:");
        for (i, s) in samples.iter().enumerate() {
            println!(
                "  iter {:>3}: {:>8.2} ms   {:>8.2} Gbps all-reduce  ({:>8.2} Gbps/link)",
                i + 1,
                s.elapsed.as_secs_f64() * 1000.0,
                s.gbps(),
                s.link_gbps(),
            );
        }
        println!(
            "\nmedian {:.2} Gbps all-reduce  ({:.2} Gbps/link)\nmin {:.2} Gbps  max {:.2} Gbps",
            median, median_link, min, max
        );
    }
    Ok(())
}

async fn run_pipeline_bench(args: PipelineBenchArgs) -> Result<()> {
    let cfg = AutoDiscoveryConfig {
        gradient_port: args.activation_port + 100,
        discovery_port: args.discovery_port,
        min_peers: args.min_peers,
        peer_timeout: Duration::from_secs(args.timeout_secs),
        profile: NodeProfile::default(),
    };

    println!(
        "pmetal cluster pipeline-bench — discovering peers (timeout {}s, min={})",
        args.timeout_secs, args.min_peers
    );

    let backend = join_cluster(cfg, args.min_peers, Duration::from_secs(args.timeout_secs))
        .await
        .context("failed to join cluster for pipeline bench")?;

    let world = backend.world_size();
    if world < 2 {
        anyhow::bail!("pipeline-bench requires at least 2 ranks (got {})", world);
    }

    let harness_cfg = PipelineHarnessConfig {
        num_layers: args.num_layers,
        activation_port: args.activation_port,
        result_port: args.result_port,
        wire_dtype: DtypeTag::Float32,
        codec: ActivationCodec::None,
        connection_timeout_ms: 30_000,
        max_retries: 60,
    };

    let harness = PipelineHarness::start(backend.clone(), harness_cfg)
        .await
        .context("pipeline harness start")?;
    let local_rank = harness.plan.local_rank;
    let world_size = harness.plan.world_size();
    let layer_range = harness.plan.layer_range();

    println!(
        "Pipeline ready: rank {}/{}, layer_range={:?}",
        local_rank, world_size, layer_range
    );

    if local_rank == 0 {
        let mut gen_loop = PipelineGenerationLoop::new(harness.stage, args.tokens, vec![u32::MAX]);

        // Stub hidden state: zero-filled f32 [1, 1, hidden_dim].
        let hidden: Vec<u8> = vec![0u8; args.hidden_dim * 4];
        let shape = [1u32, 1, args.hidden_dim as u32];

        let start = std::time::Instant::now();
        let tokens = gen_loop
            .generate_first_shard(&hidden, &shape, args.vocab as u32)
            .await
            .context("first-shard generate")?;
        let elapsed = start.elapsed();

        let tps = tokens.len() as f64 / elapsed.as_secs_f64();
        if args.json {
            let json = serde_json::json!({
                "rank": 0,
                "world_size": world_size,
                "tokens": tokens.len(),
                "elapsed_ms": elapsed.as_secs_f64() * 1000.0,
                "tokens_per_sec": tps,
                "hidden_dim": args.hidden_dim,
                "vocab": args.vocab,
            });
            println!("{}", serde_json::to_string_pretty(&json)?);
        } else {
            println!(
                "Generated {} tokens in {:.2} ms ({:.2} tok/s, stub forward)",
                tokens.len(),
                elapsed.as_secs_f64() * 1000.0,
                tps,
            );
        }
    } else {
        // Non-zero ranks: receive activation, build a stub logit vector, send back.
        let is_last = harness.plan.local_stage().is_last;
        let mut gen_loop = PipelineGenerationLoop::new(harness.stage, 0, vec![]);
        let vocab = args.vocab;

        let mut iters = 0usize;
        while iters < args.tokens {
            let msg = match gen_loop.stage.recv_from_prev().await {
                Ok(m) => m,
                Err(_) => break,
            };
            let nonce = msg.nonce;
            let logits = vec![0u8; vocab * 4];
            if is_last {
                let token: u32 = 0;
                gen_loop
                    .stage
                    .send_result(&token.to_le_bytes(), &[1], nonce)
                    .await
                    .context("send_result")?;
            } else {
                let last_layer = gen_loop.stage.config().layer_range.end.saturating_sub(1) as u32;
                gen_loop
                    .stage
                    .send_to_next(&logits, &[1, 1, vocab as u32], nonce, last_layer)
                    .await
                    .context("send_to_next")?;
            }
            iters += 1;
        }
        if !args.json {
            println!("rank {} served {} iterations", local_rank, iters);
        }
    }

    Ok(())
}

fn print_status_snapshot(backend: &Arc<AutoDiscoveryBackend>, json: bool) {
    let snap: ClusterStatus = snapshot_status(backend);
    if json {
        match serde_json::to_string_pretty(&snap) {
            Ok(s) => println!("{}", s),
            Err(e) => eprintln!("error serialising status: {}", e),
        }
    } else {
        println!("{}", snap.render_table());
    }
}
