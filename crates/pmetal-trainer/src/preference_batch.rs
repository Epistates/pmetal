use mlx_rs::Array;
use mlx_rs::error::Exception;

pub(crate) fn pad_u32_sequences(
    sequences: &[Vec<u32>],
    pad_value: u32,
) -> Result<Array, Exception> {
    let batch = sequences.len();
    let max_len = sequences.iter().map(Vec::len).max().unwrap_or(1).max(1);
    let mut values = Vec::with_capacity(batch * max_len);

    for sequence in sequences {
        values.extend(sequence.iter().map(|&token| token as i32));
        values.extend(std::iter::repeat_n(
            pad_value as i32,
            max_len - sequence.len(),
        ));
    }

    Ok(Array::from_slice(&values, &[batch as i32, max_len as i32]))
}

pub(crate) fn pad_i64_sequences(
    sequences: &[Vec<i64>],
    pad_value: i64,
) -> Result<Array, Exception> {
    let batch = sequences.len();
    let max_len = sequences.iter().map(Vec::len).max().unwrap_or(1).max(1);
    let mut values = Vec::with_capacity(batch * max_len);

    for sequence in sequences {
        values.extend(sequence.iter().copied());
        values.extend(std::iter::repeat_n(pad_value, max_len - sequence.len()));
    }

    Ok(Array::from_slice(&values, &[batch as i32, max_len as i32]))
}

pub(crate) fn pad_f32_sequences(
    sequences: &[Vec<f32>],
    pad_value: f32,
) -> Result<Array, Exception> {
    let batch = sequences.len();
    let max_len = sequences.iter().map(Vec::len).max().unwrap_or(1).max(1);
    let mut values = Vec::with_capacity(batch * max_len);

    for sequence in sequences {
        values.extend(sequence.iter().copied());
        values.extend(std::iter::repeat_n(pad_value, max_len - sequence.len()));
    }

    Ok(Array::from_slice(&values, &[batch as i32, max_len as i32]))
}
