#![no_main]

use libfuzzer_sys::fuzz_target;
use pmetal_gguf::{GgmlType, dequant};

fuzz_target!(|input: &[u8]| {
    // Byte 0 picks the tensor type; the rest is raw block data. Whole blocks
    // exercise every decoder on arbitrary bit patterns (any scale, any grid
    // index, any sign pattern); a ragged tail must be an error, not a panic.
    let Some((&type_id, data)) = input.split_first() else {
        return;
    };
    let Ok(dtype) = GgmlType::try_from(u32::from(type_id) % 48) else {
        return;
    };
    let block_bytes = dtype.type_size();
    let block_len = dtype.block_size();

    let n_blocks = data.len() / block_bytes;
    let whole = &data[..n_blocks * block_bytes];
    let shape = [1, (n_blocks * block_len) as i32];
    match dequant::dequantize(whole, dtype, &shape) {
        Ok(values) => assert_eq!(values.len(), n_blocks * block_len),
        Err(dequant::DequantError::UnsupportedType(_)) => assert!(!dequant::is_supported(dtype)),
        Err(e) => panic!("whole blocks must decode: {e}"),
    }

    // The ragged input is rejected cleanly.
    if data.len() != n_blocks * block_bytes {
        assert!(dequant::dequantize(data, dtype, &shape).is_err());
    }
});
