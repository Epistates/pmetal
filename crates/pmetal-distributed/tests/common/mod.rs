//! Run a test body as several MLX-distributed ranks over loopback.
//!
//! MLX joins a group once per process, from the environment, so ranks have to
//! be processes. The parent test spawns this same test binary once per rank,
//! each with `MLX_RANK` and a `MLX_HOSTFILE` naming loopback ports (the ring
//! backend: every rank listens on its own address and dials its right
//! neighbour's), and the child that finds `MLX_RANK` set runs the body.

#![allow(dead_code)]

use std::net::TcpListener;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// True in a spawned rank.
pub fn is_rank() -> bool {
    std::env::var("MLX_RANK").is_ok()
}

/// `n` distinct free loopback ports: the probes stay open until all are chosen.
fn free_ports(n: usize) -> Vec<u16> {
    let probes: Vec<TcpListener> = (0..n)
        .map(|_| TcpListener::bind("127.0.0.1:0").expect("bind"))
        .collect();
    probes
        .iter()
        .map(|probe| probe.local_addr().expect("addr").port())
        .collect()
}

/// Spawn `ranks` copies of this test binary running the test `name`, and
/// require every one to succeed within `timeout`.
pub fn launch(ranks: usize, name: &str, timeout: Duration) {
    let dir = tempfile::tempdir().expect("tempdir");
    let hosts: Vec<Vec<String>> = free_ports(ranks)
        .into_iter()
        .map(|port| vec![format!("127.0.0.1:{port}")])
        .collect();
    let hostfile = dir.path().join("hosts.json");
    std::fs::write(&hostfile, serde_json::to_vec(&hosts).expect("json")).expect("hostfile");

    let exe = std::env::current_exe().expect("current_exe");
    let mut children: Vec<_> = (0..ranks)
        .map(|rank| {
            Command::new(&exe)
                .env("MLX_RANK", rank.to_string())
                .env("MLX_HOSTFILE", &hostfile)
                .args([name, "--exact", "--nocapture", "--test-threads=1"])
                .stdout(Stdio::inherit())
                .stderr(Stdio::inherit())
                .spawn()
                .expect("spawn rank")
        })
        .collect();

    let deadline = Instant::now() + timeout;
    let mut failed = None;
    let mut done = vec![false; ranks];
    while done.iter().any(|d| !d) && failed.is_none() {
        for (rank, child) in children.iter_mut().enumerate() {
            if done[rank] {
                continue;
            }
            if let Some(status) = child.try_wait().expect("try_wait") {
                done[rank] = true;
                if !status.success() {
                    failed = Some(format!("rank {rank} exited with {status}"));
                }
            }
        }
        if Instant::now() > deadline {
            failed = Some(format!("timed out after {timeout:?}"));
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    if let Some(reason) = failed {
        for child in &mut children {
            let _ = child.kill();
            let _ = child.wait();
        }
        panic!("{name}: {reason}");
    }
}
