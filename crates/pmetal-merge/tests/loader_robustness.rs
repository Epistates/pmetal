//! The loader must accept any valid safetensors file, not only ones whose
//! tensor data happens to be aligned.

use pmetal_merge::{SafetensorsLoader, TensorLoader};

#[test]
fn a_tensor_at_an_unaligned_offset_loads() {
    let dir = tempfile::tempdir().unwrap();
    let data: Vec<u8> = [1.5_f32, -2.0, 3.25]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let mut header = format!(
        r#"{{"w":{{"dtype":"F32","shape":[3],"data_offsets":[0,{}]}}}}"#,
        data.len()
    )
    .into_bytes();
    // Pad the header so the data starts one byte past a 4-byte boundary.
    while (8 + header.len()) % 4 != 1 {
        header.push(b' ');
    }
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(&header);
    file.extend_from_slice(&data);
    std::fs::write(dir.path().join("model.safetensors"), &file).unwrap();

    let loader = SafetensorsLoader::new(dir.path()).unwrap();
    let mut array = loader.load_tensor("w").unwrap();
    assert_eq!(array.to_f32_vec(3).unwrap(), vec![1.5, -2.0, 3.25]);
    pmetal_runtime::check_unobserved_error().unwrap();
}
