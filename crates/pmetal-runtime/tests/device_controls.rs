//! Memory limits, the device description and the generation-stream guard.

use pmetal_runtime::check_unobserved_error;
use pmetal_runtime::inline_array::{
    self as bridge, GenerationStream, InlineArray, device_info, get_max_recommended_size,
    get_memory_limit, set_cache_limit, set_memory_limit,
};

/// The limits are process-wide, so the tests that set them (and the one that
/// reads the cache) run one at a time.
static LIMITS: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn the_memory_limit_round_trips() {
    let _serial = LIMITS.lock().unwrap_or_else(|e| e.into_inner());
    let original = get_memory_limit();
    assert!(original > 0, "MLX reports no memory limit");

    // Half of it: legal, and it must read back.
    let half = original / 2;
    let previous = set_memory_limit(half);
    assert_eq!(previous, original, "set returns the previous limit");
    assert_eq!(get_memory_limit(), half);

    assert_eq!(set_memory_limit(original), half);
    assert_eq!(get_memory_limit(), original);
    check_unobserved_error().expect("bridge error");
}

#[test]
fn the_cache_limit_returns_the_previous_one() {
    let _serial = LIMITS.lock().unwrap_or_else(|e| e.into_inner());
    let original = set_cache_limit(0);
    // With the cache off, a freed buffer is not kept.
    let x = InlineArray::from_f32_slice(&vec![1.0; 1 << 16], &[1 << 16]);
    x.add(&x).eval();
    drop(x);
    assert_eq!(bridge::get_cache_memory(), 0, "cache limit 0 keeps nothing");

    assert_eq!(set_cache_limit(original), 0);
    check_unobserved_error().expect("bridge error");
}

/// The description comes from the device, and agrees with the queries that
/// read the same numbers a different way.
#[test]
fn device_info_describes_the_gpu() {
    let info = device_info().expect("a Metal device");
    assert!(!info.device_name.is_empty());
    assert!(
        info.architecture.starts_with("applegpu"),
        "{}",
        info.architecture
    );
    assert!(
        info.memory_size >= 8 << 30,
        "{} bytes of RAM",
        info.memory_size
    );
    assert!(info.max_recommended_working_set_size <= info.memory_size);
    assert_eq!(
        info.max_recommended_working_set_size,
        get_max_recommended_size() as u64
    );
    assert!(info.max_buffer_length > 0);
    assert!(info.resource_limit > 0);
    check_unobserved_error().expect("bridge error");
}

fn product_on_the_stream(x: f32) -> f32 {
    let _stream = GenerationStream::enter();
    let a = InlineArray::from_f32_slice(&[x], &[1]);
    let out = a.multiply(&InlineArray::from_f32_slice(&[3.0], &[1]));
    out.async_eval_ref();
    out.item_f32()
}

/// The guard replaces the four-call pairing, on every thread, and nests.
#[test]
fn the_generation_stream_guard_works_on_any_thread_and_nests() {
    let _serial = LIMITS.lock().unwrap_or_else(|e| e.into_inner());
    for x in [1.0f32, 2.0, 5.0] {
        let got = std::thread::spawn(move || {
            let outer = GenerationStream::enter();
            let inner = product_on_the_stream(x);
            // The outer guard is still open here; dropping the inner one must
            // not have restored the default stream under it.
            let again = product_on_the_stream(x + 1.0);
            drop(outer);
            (inner, again)
        })
        .join()
        .expect("thread ran");
        assert_eq!(got, (3.0 * x, 3.0 * (x + 1.0)));
    }
    check_unobserved_error().expect("bridge error");
}
