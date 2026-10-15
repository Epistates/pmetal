//! The Metal layer and the runtime describe the same GPU.
//!
//! Both read the device through Metal, one for pmetal's own kernels and one
//! for the tensor runtime. A figure that differed between them (the working
//! set a limit is checked against, the generation a kernel is gated on) would
//! make one stack take a path the other refuses, so the two are held to each
//! other here.

use pmetal_metal::MetalContext;
use pmetal_runtime::check_unobserved_error;
use pmetal_runtime::inline_array::device_info;

#[test]
fn the_metal_layer_and_the_runtime_agree_on_the_device() {
    let ctx = MetalContext::new().expect("a Metal device");
    let props = ctx.properties();
    let info = device_info().expect("the runtime sees a Metal device");

    assert_eq!(props.name, info.device_name);
    assert_eq!(
        props.recommended_working_set_size, info.max_recommended_working_set_size,
        "the working set a wired limit is checked against"
    );
    assert_eq!(props.max_buffer_length, info.max_buffer_length);

    // The generation is the number in the architecture name both read.
    let generation: u32 = info
        .architecture
        .strip_prefix("applegpu_g")
        .and_then(|rest| {
            let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
            digits.parse().ok()
        })
        .expect("an applegpu architecture name");
    assert_eq!(props.architecture_gen, generation, "{}", info.architecture);

    check_unobserved_error().expect("runtime error");
}
