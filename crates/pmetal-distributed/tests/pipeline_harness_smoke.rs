//! 2-process pipeline-parallel harness smoke test (explicit-config path).
//!
//! Proves [`PipelineHarness::start_explicit`] correctly wires up:
//!   * activation TCP ring (rank 0 → rank 1, with rank 1 → rank 0 wrap-around
//!     deliberately unused — pipeline mode treats the ring as a chain)
//!   * result loopback (rank 1 → rank 0) so generated tokens flow back to
//!     the API node
//!   * `PipelineGenerationLoop` driving end-to-end token generation through
//!     a stub forward function on rank 1.
//!
//! Two child processes are spawned. The parent picks free ports and feeds
//! them through env vars, identical to the existing `distributed_smoke`
//! test in pmetal-trainer.

use pmetal_distributed::activation_codec::ActivationCodec;
use pmetal_distributed::activation_transport::DtypeTag;
use pmetal_distributed::pipeline::PipelineGenerationLoop;
use pmetal_distributed::pipeline_harness::{PipelineHarness, PipelineHarnessConfig};
use pmetal_distributed::topology::NodeProfile;
use std::net::{SocketAddr, TcpListener};

fn grab_free_port() -> u16 {
    let l = TcpListener::bind("127.0.0.1:0").expect("bind port 0");
    l.local_addr().expect("local_addr").port()
}

const ENV_RANK: &str = "PMETAL_PHA_RANK";
const ENV_ACT0: &str = "PMETAL_PHA_ACT0";
const ENV_ACT1: &str = "PMETAL_PHA_ACT1";
const ENV_RESULT: &str = "PMETAL_PHA_RESULT";

const HIDDEN_DIM: usize = 8;
const VOCAB: usize = 32;
const NUM_LAYERS: usize = 4;
const TOKENS_TO_GENERATE: usize = 3;
const EXPECTED_TOKEN: u32 = 17;
const STOP_TOKEN: u32 = 9999;

async fn run_worker() {
    use std::str::FromStr;

    let rank: usize = usize::from_str(&std::env::var(ENV_RANK).expect("rank")).expect("parse rank");
    let port_0: u16 = u16::from_str(&std::env::var(ENV_ACT0).expect("act0")).expect("parse act0");
    let port_1: u16 = u16::from_str(&std::env::var(ENV_ACT1).expect("act1")).expect("parse act1");
    let result_port: u16 =
        u16::from_str(&std::env::var(ENV_RESULT).expect("res")).expect("parse res");

    let addr_0: SocketAddr = format!("127.0.0.1:{port_0}").parse().unwrap();
    let addr_1: SocketAddr = format!("127.0.0.1:{port_1}").parse().unwrap();

    // Per-rank fabric-ranked address list; only one entry each since we're
    // on loopback. The harness binds 0.0.0.0:port internally.
    let peer_addrs = vec![vec![addr_0], vec![addr_1]];
    let profiles = vec![
        NodeProfile {
            available_ram: 1 << 30,
            ..NodeProfile::default()
        },
        NodeProfile {
            available_ram: 1 << 30,
            ..NodeProfile::default()
        },
    ];

    let cfg = PipelineHarnessConfig {
        num_layers: NUM_LAYERS,
        activation_port: if rank == 0 { port_0 } else { port_1 },
        result_port,
        wire_dtype: DtypeTag::Float32,
        codec: ActivationCodec::None,
        connection_timeout_ms: 30_000,
        max_retries: 60,
    };

    let harness = PipelineHarness::start_explicit(peer_addrs, profiles, rank, cfg)
        .await
        .expect("pipeline harness start_explicit");

    eprintln!(
        "rank {} ready: layer_range={:?}, world_size={}",
        rank,
        harness.plan.layer_range(),
        harness.plan.world_size()
    );

    if rank == 0 {
        let mut gen_loop =
            PipelineGenerationLoop::new(harness.stage, TOKENS_TO_GENERATE, vec![STOP_TOKEN]);

        let hidden: Vec<u8> = vec![0u8; HIDDEN_DIM * 4];
        let shape = [1u32, 1, HIDDEN_DIM as u32];

        let tokens = gen_loop
            .generate_first_shard(&hidden, &shape, VOCAB as u32)
            .await
            .expect("generate first shard");

        assert_eq!(tokens.len(), TOKENS_TO_GENERATE, "expected N tokens");
        for (i, &tok) in tokens.iter().enumerate() {
            assert_eq!(
                tok, EXPECTED_TOKEN,
                "token {i} mismatch: got {tok}, expected {EXPECTED_TOKEN}"
            );
        }
        eprintln!("rank 0 OK — got {:?}", tokens);
    } else {
        // Last rank: receive activation, build a one-hot logit, send back.
        let mut gen_loop = PipelineGenerationLoop::new(harness.stage, 0, vec![]);

        let forward_fn =
            |_data: &[u8],
             _shape: &[u32]|
             -> pmetal_distributed::error::DistributedResult<(Vec<u8>, Vec<u32>)> {
                let mut logits = vec![0.0_f32; VOCAB];
                logits[EXPECTED_TOKEN as usize] = 1.0;
                let mut bytes = Vec::with_capacity(VOCAB * 4);
                for f in logits {
                    bytes.extend_from_slice(&f.to_le_bytes());
                }
                Ok((bytes, vec![1, 1, VOCAB as u32]))
            };

        let mut iters = 0usize;
        let res: pmetal_distributed::error::DistributedResult<()> = async {
            for _ in 0..TOKENS_TO_GENERATE {
                let msg = gen_loop.stage.recv_from_prev().await?;
                let nonce = msg.nonce;
                let (out_data, _out_shape) = forward_fn(&msg.data, &msg.shape)?;
                let token = greedy_argmax_last_position(&out_data, VOCAB);
                gen_loop
                    .stage
                    .send_result(&token.to_le_bytes(), &[1], nonce)
                    .await?;
                iters += 1;
            }
            Ok(())
        }
        .await;

        if let Err(e) = res {
            panic!("rank 1 shard loop failed: {e}");
        }
        eprintln!("rank 1 OK — served {} tokens", iters);
    }
}

fn greedy_argmax_last_position(data: &[u8], vocab: usize) -> u32 {
    let total = data.len() / 4;
    let last_pos_start = (total - vocab) * 4;
    let mut best_idx: u32 = 0;
    let mut best_val: f32 = f32::NEG_INFINITY;
    for (i, c) in data[last_pos_start..].as_chunks::<4>().0.iter().enumerate() {
        let v = f32::from_le_bytes(*c);
        if v > best_val {
            best_val = v;
            best_idx = i as u32;
        }
    }
    best_idx
}

#[test]
fn pipeline_harness_two_process_smoke() {
    if std::env::var(ENV_RANK).is_ok() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("rt");
        rt.block_on(run_worker());
        return;
    }

    let bin = std::env::current_exe().expect("current_exe");
    let port_0 = grab_free_port();
    let port_1 = grab_free_port();
    let result_port = grab_free_port();

    eprintln!("parent: spawning ranks (act0={port_0}, act1={port_1}, result={result_port})");

    let mut child_0 = std::process::Command::new(&bin)
        .env(ENV_RANK, "0")
        .env(ENV_ACT0, port_0.to_string())
        .env(ENV_ACT1, port_1.to_string())
        .env(ENV_RESULT, result_port.to_string())
        .arg("pipeline_harness_two_process_smoke")
        .arg("--exact")
        .arg("--nocapture")
        .spawn()
        .expect("spawn rank 0");

    let mut child_1 = std::process::Command::new(&bin)
        .env(ENV_RANK, "1")
        .env(ENV_ACT0, port_0.to_string())
        .env(ENV_ACT1, port_1.to_string())
        .env(ENV_RESULT, result_port.to_string())
        .arg("pipeline_harness_two_process_smoke")
        .arg("--exact")
        .arg("--nocapture")
        .spawn()
        .expect("spawn rank 1");

    let s0 = child_0.wait().expect("wait rank 0");
    let s1 = child_1.wait().expect("wait rank 1");
    assert!(s0.success(), "rank 0 failed: {:?}", s0);
    assert!(s1.success(), "rank 1 failed: {:?}", s1);
}
