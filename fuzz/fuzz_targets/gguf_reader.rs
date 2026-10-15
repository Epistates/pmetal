#![no_main]

use libfuzzer_sys::fuzz_target;
use pmetal_gguf::{GgufContent, dequant};
use std::io::Cursor;

fuzz_target!(|data: &[u8]| {
    // Parse the header, metadata and tensor-info section. Every count and
    // length in a hostile header is checked against the bytes the input has
    // left, and every tensor's extent against its size, so neither parsing
    // nor reading any tensor may allocate more than the input is long.
    let mut cursor = Cursor::new(data);
    let Ok(content) = GgufContent::read(&mut cursor) else {
        return;
    };

    for info in content.tensors_by_offset() {
        let mut reader = Cursor::new(data);
        let bytes = content
            .read_tensor_data(&mut reader, &info.name)
            .expect("a tensor accepted at open lies inside the file");
        assert_eq!(bytes.len(), info.byte_size());

        // Decode what is cheap to decode; the shape is the file's own.
        if info.n_elements() <= 1 << 16 && dequant::is_supported(info.dtype) {
            let shape: Vec<i32> = info
                .dimensions
                .iter()
                .map(|&d| i32::try_from(d).unwrap_or(i32::MAX))
                .collect();
            let _ = dequant::dequantize(&bytes, info.dtype, &shape);
        }
    }
});
