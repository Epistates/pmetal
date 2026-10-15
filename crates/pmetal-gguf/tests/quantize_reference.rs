//! Bit-exact quantization checks against the reference quantizers.
//!
//! `fixtures/quantize_reference.blocks` holds, for each format whose reference
//! quantizer is simple enough to be reproduced exactly, an input tensor and the
//! bytes the reference produces for it. The input includes the cases that
//! stress the rounding rules: an all-zero block, a constant block, a block
//! dominated by one outlier, and values that are exact halves.
//!
//! Fixture layout (little endian): `u32 entries`, then per entry
//! `u32 ggml_type, u32 n_values, u32 n_bytes`, the `f32` input and the
//! expected bytes.

use pmetal_gguf::{GgmlType, quantize};

const FIXTURE: &[u8] = include_bytes!("fixtures/quantize_reference.blocks");

fn u32_at(pos: &mut usize) -> u32 {
    let v = u32::from_le_bytes(FIXTURE[*pos..*pos + 4].try_into().unwrap());
    *pos += 4;
    v
}

fn check(dtype: GgmlType) {
    let mut pos = 0;
    let entries = u32_at(&mut pos);
    for _ in 0..entries {
        let ty = u32_at(&mut pos);
        let n_values = u32_at(&mut pos) as usize;
        let n_bytes = u32_at(&mut pos) as usize;
        let input: Vec<f32> = FIXTURE[pos..pos + 4 * n_values]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        pos += 4 * n_values;
        let expected = &FIXTURE[pos..pos + n_bytes];
        pos += n_bytes;
        if GgmlType::try_from(ty).unwrap() != dtype {
            continue;
        }

        // As one row and as rows of 256: blocks are independent of how the
        // tensor is cut into rows.
        for n_per_row in [n_values, 256] {
            let got = quantize::quantize_tensor(&input, dtype, n_per_row, None).unwrap();
            assert_eq!(got.len(), expected.len());
            let block = dtype.type_size();
            if let Some(i) = (0..got.len()).find(|&i| got[i] != expected[i]) {
                panic!(
                    "{dtype:?}: first differing byte {i} (block {}, offset {} of {block}): got {:#04x}, reference {:#04x}; {} bytes differ",
                    i / block,
                    i % block,
                    got[i],
                    expected[i],
                    got.iter().zip(expected).filter(|(a, b)| a != b).count()
                );
            }
        }
        return;
    }
    panic!("no fixture for {dtype:?}");
}

macro_rules! reference_tests {
    ($($name:ident => $ty:expr),* $(,)?) => {
        $(#[test] fn $name() { check($ty); })*
    };
}

reference_tests! {
    q4_0 => GgmlType::Q4_0,
    q4_1 => GgmlType::Q4_1,
    q5_0 => GgmlType::Q5_0,
    q5_1 => GgmlType::Q5_1,
    q8_0 => GgmlType::Q8_0,
    tq1_0 => GgmlType::Tq1_0,
    tq2_0 => GgmlType::Tq2_0,
    mxfp4 => GgmlType::Mxfp4,
}
