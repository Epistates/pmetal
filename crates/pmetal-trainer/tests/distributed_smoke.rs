//! 2-process distributed gradient sync smoke test.
//!
//! Proves that `RingBackend::all_reduce` produces the correct element-wise
//! mean across two processes communicating over localhost TCP.
//!
//! Run with:
//!   cargo test -p pmetal-trainer --features distributed --test distributed_smoke
//!
//! The entire test is feature-gated; without `--features distributed` the test
//! file compiles to an empty crate and nothing runs.

#[cfg(feature = "distributed")]
mod inner {
    use pmetal_distributed::{DistributedBackend, DistributedConfig, ReduceOp, RingBackend};
    use std::net::{SocketAddr, TcpListener};

    // ── Port helpers ─────────────────────────────────────────────────────────

    /// Bind a `TcpListener` on `127.0.0.1:0` and return the OS-assigned port.
    ///
    /// The listener is dropped immediately after reading the port, releasing
    /// the socket so `RingBackend` can re-bind it.  There is a short TOCTOU
    /// window, but on a loopback interface in a single-machine test this is
    /// negligible.
    fn grab_free_port() -> u16 {
        let l = TcpListener::bind("127.0.0.1:0").expect("bind port 0");
        l.local_addr().expect("local_addr").port()
    }

    // ── Env-var keys ─────────────────────────────────────────────────────────

    const ENV_RANK: &str = "PMETAL_TEST_RANK";
    const ENV_PORT_0: &str = "PMETAL_TEST_PORT_0";
    const ENV_PORT_1: &str = "PMETAL_TEST_PORT_1";

    // ── Worker entry point ───────────────────────────────────────────────────

    /// Called by each child process.  Reads its rank + peer ports from env
    /// vars, runs `all_reduce(Mean)` on a small gradient buffer, and panics
    /// if the result is wrong.
    async fn run_worker() {
        let rank: usize = std::env::var(ENV_RANK)
            .expect("PMETAL_TEST_RANK not set")
            .parse()
            .expect("PMETAL_TEST_RANK must be usize");

        let port_0: u16 = std::env::var(ENV_PORT_0)
            .expect("PMETAL_TEST_PORT_0 not set")
            .parse()
            .expect("PMETAL_TEST_PORT_0 must be u16");

        let port_1: u16 = std::env::var(ENV_PORT_1)
            .expect("PMETAL_TEST_PORT_1 not set")
            .parse()
            .expect("PMETAL_TEST_PORT_1 must be u16");

        let addr_0: SocketAddr = format!("127.0.0.1:{port_0}").parse().unwrap();
        let addr_1: SocketAddr = format!("127.0.0.1:{port_1}").parse().unwrap();

        let config = DistributedConfig::new(vec![addr_0, addr_1], rank);

        let backend = RingBackend::new(config)
            .await
            .unwrap_or_else(|e| panic!("RingBackend::new failed (rank {rank}): {e}"));

        // Each rank contributes a distinct gradient vector.
        //   rank 0: [1.0, 2.0, 3.0]
        //   rank 1: [4.0, 5.0, 6.0]
        // Mean across 2 nodes: [2.5, 3.5, 4.5]
        let gradients: &[f32] = match rank {
            0 => &[1.0_f32, 2.0, 3.0],
            1 => &[4.0_f32, 5.0, 6.0],
            _ => panic!("unexpected rank {rank}"),
        };

        // Reinterpret as bytes for all_reduce (Vec<f32> guarantees 4-byte
        // alignment — the same pattern used in distributed_bridge.rs).
        let mut buf: Vec<f32> = gradients.to_vec();
        let byte_len = buf.len() * 4;
        let byte_ptr = buf.as_mut_ptr().cast::<u8>();
        #[allow(unsafe_code)]
        let byte_buf = unsafe { std::slice::from_raw_parts_mut(byte_ptr, byte_len) };

        backend
            .all_reduce(byte_buf, ReduceOp::Mean)
            .await
            .unwrap_or_else(|e| panic!("all_reduce failed (rank {rank}): {e}"));

        // Verify element-wise mean [2.5, 3.5, 4.5].
        let expected = [2.5_f32, 3.5, 4.5];
        for (i, (&got, &exp)) in buf.iter().zip(expected.iter()).enumerate() {
            let abs_err = (got - exp).abs();
            assert!(
                abs_err < 1e-5,
                "rank {rank}: element {i} mismatch — expected {exp}, got {got} (err {abs_err})"
            );
        }

        eprintln!("rank {rank}: all_reduce OK — result = {buf:?}");
    }

    // ── Parent process (test entry point) ────────────────────────────────────

    /// Spawns two child processes and waits for both to succeed.
    ///
    /// Port allocation: two free ports are grabbed in the parent before any
    /// child is spawned so both children can be given the *same* port pair.
    #[test]
    fn ring_allreduce_two_process_smoke() {
        // Grab two free ports before spawning children.
        let port_0 = grab_free_port();
        let port_1 = grab_free_port();

        // Detect the current test binary so we can re-invoke ourselves.
        // `cargo test` sets the test binary as the current executable.
        let bin = std::env::current_exe().expect("current_exe");

        // Spawn rank 0.
        let mut child_0 = std::process::Command::new(&bin)
            .env(ENV_RANK, "0")
            .env(ENV_PORT_0, port_0.to_string())
            .env(ENV_PORT_1, port_1.to_string())
            // Tell the test harness to run only this function in the child.
            .arg("distributed_smoke_worker")
            // Suppress test harness output so only our eprintln! shows.
            .arg("--nocapture")
            .spawn()
            .expect("failed to spawn child rank 0");

        // Spawn rank 1.
        let mut child_1 = std::process::Command::new(&bin)
            .env(ENV_RANK, "1")
            .env(ENV_PORT_0, port_0.to_string())
            .env(ENV_PORT_1, port_1.to_string())
            .arg("distributed_smoke_worker")
            .arg("--nocapture")
            .spawn()
            .expect("failed to spawn child rank 1");

        let status_0 = child_0.wait().expect("wait child 0");
        let status_1 = child_1.wait().expect("wait child 1");

        assert!(
            status_0.success(),
            "rank 0 child exited with failure: {status_0}"
        );
        assert!(
            status_1.success(),
            "rank 1 child exited with failure: {status_1}"
        );
    }

    // ── Worker shim ─────────────────────────────────────────────────────────

    /// This function is the test that *child* processes run.
    ///
    /// When the parent spawns `--test distributed_smoke_worker`, the cargo
    /// test harness invokes this function.  It drives an async runtime and
    /// calls `run_worker()`.
    #[test]
    fn distributed_smoke_worker() {
        // Only run as a worker when the env var is set; this prevents the
        // function from doing real work when the parent harness enumerates
        // tests (e.g. `cargo test -- --list`).
        if std::env::var(ENV_RANK).is_err() {
            return;
        }

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("tokio runtime");

        rt.block_on(run_worker());
    }
}
