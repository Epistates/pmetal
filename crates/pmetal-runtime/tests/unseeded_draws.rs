//! A test that draws without seeding draws the same numbers on every run.
//!
//! Under `strict-errors` (which this crate's tests, and every dependent's,
//! turn on) the first unkeyed draw on a thread nothing has seeded seeds it from
//! the thread's name. Without that a test's inputs come from the clock and it
//! passes or fails by chance.

use pmetal_runtime::compat::{Array, Dtype, random};

/// A draw a test makes.
type Draw = fn() -> Array;

/// What `draw` produces on a fresh thread called `name`.
fn drawn_on(name: &str, draw: Draw) -> Vec<f32> {
    std::thread::Builder::new()
        .name(name.to_string())
        .spawn(move || {
            let mut values = draw();
            let len = values.shape().iter().product::<i32>() as usize;
            values.to_f32_vec(len).unwrap()
        })
        .unwrap()
        .join()
        .unwrap()
}

#[test]
fn every_unkeyed_draw_is_the_same_on_every_run_of_a_test() {
    let draws: [(&str, Draw); 6] = [
        ("normal", || random::normal(&[8], Dtype::Float32)),
        ("uniform", || random::uniform(&[8], Dtype::Float32)),
        ("uniform_range", || {
            random::uniform_range(0.0, 1.0, &[8], Dtype::Float32)
        }),
        ("randint", || {
            random::randint(0, 1000, &[8], Dtype::Int32, None).as_dtype(Dtype::Float32.as_i32())
        }),
        ("bernoulli", || {
            random::bernoulli(&Array::from_f32(0.5), &[8]).as_dtype(Dtype::Float32.as_i32())
        }),
        ("dropout_mask", || {
            random::dropout_mask(0.5, &[8]).as_dtype(Dtype::Float32.as_i32())
        }),
    ];
    for (op, draw) in draws {
        // Two runs of "the same test" (a thread of the same name) agree.
        let first = drawn_on("tests::some_test", draw);
        let again = drawn_on("tests::some_test", draw);
        assert_eq!(first, again, "{op} differs between two runs of one test");
    }
}

/// What a test that never seeds draws, printed for the process that started it
/// to read. Does nothing unless that process asked for it.
#[test]
fn an_unseeded_test_prints_its_draws() {
    if std::env::var_os(WORKER).is_none() {
        return;
    }
    let mut normal = random::normal(&[16], Dtype::Float32);
    let mut uniform = random::uniform(&[16], Dtype::Float32);
    let mut dropout = random::dropout_mask(0.5, &[16]).as_dtype(Dtype::Float32.as_i32());
    let bits = |values: Vec<f32>| {
        values
            .iter()
            .map(|v| format!("{:08x}", v.to_bits()))
            .collect::<String>()
    };
    println!(
        "DRAWS {} {} {}",
        bits(normal.to_f32_vec(16).unwrap()),
        bits(uniform.to_f32_vec(16).unwrap()),
        bits(dropout.to_f32_vec(16).unwrap())
    );
}

const WORKER: &str = "PMETAL_TEST_PRINT_DRAWS";

/// The same test draws the same numbers in a process started a second later:
/// MLX seeds a thread's key from the clock, in whole seconds, so two processes
/// (two runs of a test suite) a second apart used to start from different
/// keys, while two threads in one process, which is all the tests above can
/// see, usually did not.
#[test]
fn a_test_draws_the_same_numbers_in_every_process() {
    let run = || {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .env(WORKER, "1")
            .args([
                "--exact",
                "an_unseeded_test_prints_its_draws",
                "--nocapture",
            ])
            .output()
            .unwrap();
        assert!(output.status.success());
        // The test harness prints "test <name> ... " ahead of the output.
        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let at = stdout.find("DRAWS ").expect("the worker printed its draws");
        stdout[at..].lines().next().unwrap().to_string()
    };
    let first = run();
    std::thread::sleep(std::time::Duration::from_millis(1100));
    assert_eq!(first, run());
}

#[test]
fn different_tests_draw_different_numbers() {
    let draw = || random::normal(&[8], Dtype::Float32);
    assert_ne!(drawn_on("tests::one", draw), drawn_on("tests::two", draw));
}

#[test]
fn categorical_draws_agree_too() {
    let draw = || {
        let logits = Array::from_slice(&[0.0f32; 64], &[8, 8]);
        random::categorical(&logits, -1, None).as_dtype(Dtype::Float32.as_i32())
    };
    assert_eq!(
        drawn_on("tests::sampling", draw),
        drawn_on("tests::sampling", draw)
    );
}

#[test]
fn an_explicit_seed_wins_over_the_name() {
    let seeded = |name: &str| {
        let name = name.to_string();
        std::thread::Builder::new()
            .name(name)
            .spawn(|| {
                random::seed(5);
                let mut values = random::normal(&[8], Dtype::Float32);
                values.to_f32_vec(8).unwrap()
            })
            .unwrap()
            .join()
            .unwrap()
    };
    assert_eq!(seeded("tests::one"), seeded("tests::two"));
}

#[test]
fn a_keyed_draw_does_not_touch_the_threads_key() {
    // A draw with a key is a function of the key and leaves the thread's own
    // stream where it was.
    let draw = || {
        let key = random::key(3);
        let _ = random::normal_keyed(&[8], Dtype::Float32, Some(&key));
        random::normal(&[8], Dtype::Float32)
    };
    let plain = || random::normal(&[8], Dtype::Float32);
    assert_eq!(
        drawn_on("tests::keyed", draw),
        drawn_on("tests::keyed", plain)
    );
}
