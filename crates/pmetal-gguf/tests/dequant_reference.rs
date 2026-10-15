//! Bit-exact dequantization checks against the reference ggml block definitions.
//!
//! `fixtures/dequant_reference.blocks` holds, for every decodable tensor type, a
//! handful of pseudo-random raw blocks together with the f32 values the
//! reference decoder produces for them. Scale fields in the blocks are powers
//! of two, so every product the reference forms is exact in f32 and the
//! expected values do not depend on multiplication order: any mismatch is a
//! bit-layout or table error, never rounding.
//!
//! Fixture layout (little endian): `u32 entries`, then per entry
//! `u32 ggml_type, u32 n_blocks, u32 raw_len, u32 n_values`, the raw bytes,
//! and `n_values` f32 values.

use pmetal_gguf::{GgmlType, dequant};

const FIXTURE: &[u8] = include_bytes!("fixtures/dequant_reference.blocks");

struct Entry {
    dtype: GgmlType,
    raw: &'static [u8],
    expected: Vec<f32>,
}

fn u32_at(bytes: &[u8], pos: &mut usize) -> u32 {
    let v = u32::from_le_bytes(bytes[*pos..*pos + 4].try_into().unwrap());
    *pos += 4;
    v
}

fn entries() -> Vec<Entry> {
    let mut pos = 0;
    let count = u32_at(FIXTURE, &mut pos);
    (0..count)
        .map(|_| {
            let ty = u32_at(FIXTURE, &mut pos);
            let _n_blocks = u32_at(FIXTURE, &mut pos);
            let raw_len = u32_at(FIXTURE, &mut pos) as usize;
            let n_values = u32_at(FIXTURE, &mut pos) as usize;
            let raw = &FIXTURE[pos..pos + raw_len];
            pos += raw_len;
            let expected = FIXTURE[pos..pos + n_values * 4]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            pos += n_values * 4;
            Entry {
                dtype: GgmlType::try_from(ty).expect("fixture type id"),
                raw,
                expected,
            }
        })
        .collect()
}

fn check(dtype: GgmlType) {
    let entry = entries()
        .into_iter()
        .find(|e| e.dtype == dtype)
        .unwrap_or_else(|| panic!("no fixture for {dtype:?}"));
    assert_eq!(
        entry.raw.len(),
        dtype.tensor_size(entry.expected.len()),
        "{dtype:?}: fixture raw size disagrees with the type's block size"
    );
    let shape = [entry.expected.len() as i32];
    let got = dequant::dequantize(entry.raw, dtype, &shape).expect("dequantize");
    assert_eq!(got.len(), entry.expected.len());
    let mismatches: Vec<usize> = (0..got.len())
        .filter(|&i| got[i].to_bits() != entry.expected[i].to_bits())
        .collect();
    assert!(
        mismatches.is_empty(),
        "{dtype:?}: {} of {} values differ from the reference; first at {}: got {} expected {}",
        mismatches.len(),
        got.len(),
        mismatches[0],
        got[mismatches[0]],
        entry.expected[mismatches[0]]
    );
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
    q8_1 => GgmlType::Q8_1,
    q2_k => GgmlType::Q2K,
    q3_k => GgmlType::Q3K,
    q4_k => GgmlType::Q4K,
    q5_k => GgmlType::Q5K,
    q6_k => GgmlType::Q6K,
    q8_k => GgmlType::Q8K,
    iq2_xxs => GgmlType::Iq2Xxs,
    iq2_xs => GgmlType::Iq2Xs,
    iq2_s => GgmlType::Iq2S,
    iq3_xxs => GgmlType::Iq3Xxs,
    iq3_s => GgmlType::Iq3S,
    iq1_s => GgmlType::Iq1S,
    iq1_m => GgmlType::Iq1M,
    iq4_nl => GgmlType::Iq4Nl,
    iq4_xs => GgmlType::Iq4Xs,
    tq1_0 => GgmlType::Tq1_0,
    tq2_0 => GgmlType::Tq2_0,
    mxfp4 => GgmlType::Mxfp4,
    nvfp4 => GgmlType::Nvfp4,
    q1_0 => GgmlType::Q1_0,
}
