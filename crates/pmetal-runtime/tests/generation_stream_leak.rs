//! What a thread that runs MLX ops costs the process once it is gone.
//!
//! MLX makes streams for every thread that runs an op, and a worker thread for
//! each of its CPU streams, and keeps them until the thread gives them back. A
//! server that ran each request on a thread of its own left about 100 KiB of
//! resident memory behind per request, for the life of the process.
//! `ThreadStreams` gives them back.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier, Mutex};

use pmetal_runtime::compat::Dtype;
use pmetal_runtime::inline_array::{GenerationStream, InlineArray, ThreadStreams};

/// The tests measure the whole process's memory, so they take turns.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn resident_bytes() -> u64 {
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .expect("ps runs");
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse::<u64>()
        .expect("ps prints a size in KiB")
        * 1024
}

/// A thread that runs a small op, on its generation stream or without one,
/// and gives its streams back or not.
fn run_on_a_new_thread(generation_stream: bool, release: bool) {
    std::thread::spawn(move || {
        let _release = release.then(ThreadStreams::release_on_drop);
        let _stream = generation_stream.then(GenerationStream::enter);
        let a = InlineArray::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let out = a.matmul(&a).sum(None);
        out.eval();
        pmetal_runtime::check_unobserved_error().expect("the op ran");
        assert_eq!(out.item_f32(), 54.0);
    })
    .join()
    .expect("thread ran");
}

/// Resident KiB each of `threads` threads leaves behind.
fn kib_left_per_thread(generation_stream: bool, release: bool, threads: usize) -> f64 {
    // Warm up the allocator and the kernels.
    for _ in 0..50 {
        run_on_a_new_thread(generation_stream, release);
    }
    let before = resident_bytes();
    for _ in 0..threads {
        run_on_a_new_thread(generation_stream, release);
    }
    resident_bytes().saturating_sub(before) as f64 / 1024.0 / threads as f64
}

/// Threads that come and go give back what their streams took.
#[test]
fn a_thread_that_releases_its_streams_leaves_nothing_behind() {
    let _turn = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let leaked = kib_left_per_thread(true, false, 300);
    let generating = kib_left_per_thread(true, true, 600);
    let op_only = kib_left_per_thread(false, true, 600);
    println!(
        "left per thread: {leaked:.1} KiB without releasing, {generating:.1} KiB generating \
         and releasing, {op_only:.1} KiB running an op and releasing"
    );
    assert!(
        leaked > 30.0,
        "the threads of this test no longer leak ({leaked:.1} KiB each): the measurement is broken"
    );
    assert!(
        generating < 10.0,
        "a generating thread leaves {generating:.1} KiB behind"
    );
    assert!(
        op_only < 10.0,
        "a thread that ran an op leaves {op_only:.1} KiB behind"
    );
}

/// Releasing empties MLX's shared compile cache, which other threads are
/// using: threads that run compiled fused ops all the while must come to no
/// harm as short-lived threads release theirs.
#[test]
fn releasing_streams_beside_threads_that_run_compiled_ops_is_safe() {
    let _turn = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let stop = Arc::new(AtomicBool::new(false));
    let ready = Arc::new(Barrier::new(5));
    let workers: Vec<_> = (0..4)
        .map(|_| {
            let (stop, ready) = (Arc::clone(&stop), Arc::clone(&ready));
            std::thread::spawn(move || {
                ready.wait();
                let mut rounds = 0;
                while !stop.load(Ordering::Relaxed) || rounds < 20 {
                    let x = InlineArray::from_f32_slice(&[0.5, -1.0, 2.0, 0.25], &[4])
                        .as_dtype(Dtype::Bfloat16.as_i32());
                    let mut y = x.fused_silu().as_dtype(Dtype::Float32.as_i32());
                    let got = y.to_f32_vec(4).expect("silu readback");
                    assert!((got[2] - 2.0 / (1.0 + (-2.0f32).exp()) * 1.0).abs() < 2e-2);
                    rounds += 1;
                }
                pmetal_runtime::check_unobserved_error().expect("bridge error");
            })
        })
        .collect();
    ready.wait();
    for _ in 0..200 {
        run_on_a_new_thread(true, true);
    }
    stop.store(true, Ordering::Relaxed);
    for w in workers {
        w.join().expect("worker thread");
    }
}
