#![no_main]

use libfuzzer_sys::fuzz_target;
use pmetal_gguf::{GgmlType, dequant, quantize};

fuzz_target!(|input: &[u8]| {
    // Byte 0 picks the type, byte 1 whether to pass an importance vector;
    // the rest is f32 data of any value, including NaN and infinities.
    let [type_id, with_importance, rest @ ..] = input else {
        return;
    };
    let Ok(dtype) = GgmlType::try_from(u32::from(*type_id) % 48) else {
        return;
    };
    if !quantize::can_quantize(dtype) {
        return;
    }
    let values: Vec<f32> = rest
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    let block = dtype.block_size();
    let n = values.len() / block * block;
    if n == 0 {
        return;
    }
    let data = &values[..n];

    // Importance: finite and non-negative or it is rejected.
    let importance: Vec<f32> = data.iter().map(|v| v.abs().min(1e6)).collect();
    let importance = (*with_importance & 1 == 1)
        .then_some(importance.as_slice())
        .filter(|w| w.iter().all(|v| v.is_finite()));

    let bytes = quantize::quantize_tensor(data, dtype, n, importance)
        .expect("block-aligned input always quantizes");
    assert_eq!(bytes.len(), dtype.tensor_size(n));
    if dequant::is_supported(dtype) {
        let back = dequant::dequantize(&bytes, dtype, &[1, n as i32]).expect("own output decodes");
        assert_eq!(back.len(), n);
        // Finite input of modest magnitude must not decode to NaN.
        if data.iter().all(|v| v.is_finite() && v.abs() < 1e4) {
            assert!(
                back.iter().all(|v| !v.is_nan()),
                "{dtype:?} produced NaN from finite input"
            );
        }
    }
});
