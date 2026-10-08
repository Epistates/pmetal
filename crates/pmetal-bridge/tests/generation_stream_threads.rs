//! Every thread that generates gets a generation stream it can use.
//!
//! MLX streams belong to the thread that created them. The bridge kept one
//! generation stream for the whole process, made by the first thread to ask,
//! and every other thread that made it its default stream failed with "There
//! is no Stream(gpu, N) in current thread": a server's second request on
//! another pool thread, or a second model thread in the same process.

use pmetal_bridge::inline_array::{self as bridge, InlineArray};

fn generate_on_this_thread(x: f32) -> f32 {
    bridge::new_generation_stream();
    bridge::set_generation_stream();
    let a = InlineArray::from_f32_slice(&[x], &[1]);
    let out = a.multiply(&InlineArray::from_f32_slice(&[3.0], &[1]));
    out.eval();
    bridge::synchronize();
    bridge::reset_default_stream();
    pmetal_bridge::check_last_error().expect("the generation stream works on this thread");
    out.item_f32()
}

#[test]
fn each_thread_generates_on_its_own_stream() {
    for x in [1.0f32, 2.0, 5.0] {
        let got = std::thread::spawn(move || generate_on_this_thread(x))
            .join()
            .expect("thread ran");
        assert_eq!(got, 3.0 * x);
    }
}
