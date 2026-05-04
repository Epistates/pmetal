pub(crate) const KVALUES_MXFP4: [i8; 16] =
    [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

#[inline]
pub(crate) fn e8m0_to_fp32_half(x: u8) -> f32 {
    let bits = if x < 2 {
        0x0020_0000u32 << x
    } else {
        (u32::from(x) - 1) << 23
    };
    f32::from_bits(bits)
}

#[inline]
pub(crate) fn ue4m3_to_fp32(x: u8) -> f32 {
    if x == 0 || x == 0x7f {
        return 0.0;
    }

    let exp = ((x >> 3) & 0x0f) as i32;
    let man = (x & 0x07) as i32;
    let raw = if exp == 0 {
        man as f32 * 2.0f32.powi(-9)
    } else {
        (1.0 + man as f32 / 8.0) * 2.0f32.powi(exp - 7)
    };
    raw * 0.5
}

#[inline]
pub(crate) fn fp32_to_ue4m3(mut x: f32) -> u8 {
    if x <= 0.0 || !x.is_finite() {
        return 0;
    }
    if x > 448.0 {
        x = 448.0;
    }

    let bits = x.to_bits();
    let fp32_exp = ((bits >> 23) & 0xff) as i32 - 127;
    let fp32_man = ((bits >> 20) & 0x07) as i32;
    let mut ue4m3_exp = fp32_exp + 7;

    if ue4m3_exp <= 0 {
        let man = (x * 512.0 + 0.5) as i32;
        if man > 7 {
            return 7;
        }
        if man < 1 {
            return 0;
        }
        return man as u8;
    }
    if ue4m3_exp >= 15 {
        return 0x7e;
    }

    let round_bit = ((bits >> 19) & 1) as i32;
    let mut ue4m3_man = fp32_man + round_bit;
    if ue4m3_man > 7 {
        ue4m3_man = 0;
        ue4m3_exp += 1;
        if ue4m3_exp >= 15 {
            return 0x7e;
        }
    }

    ((ue4m3_exp << 3) | ue4m3_man) as u8
}

#[inline]
pub(crate) fn best_index_mxfp4(value: f32, scale: f32) -> u8 {
    let mut best_index = 0usize;
    let mut best_error = (KVALUES_MXFP4[0] as f32 * scale - value).abs();

    for (index, quant) in KVALUES_MXFP4.iter().enumerate().skip(1) {
        let error = (*quant as f32 * scale - value).abs();
        if error < best_error {
            best_index = index;
            best_error = error;
        }
    }

    best_index as u8
}
