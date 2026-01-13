use pmetal_gguf::imatrix::IMatrix;
use std::path::Path;

#[test]
fn test_imatrix_loading() {
    let path = Path::new("dummy_imatrix.dat");
    if !path.exists() {
        // Skip if generated file is missing (e.g. in CI without python)
        return;
    }

    let imatrix = IMatrix::load(path).expect("Failed to load IMatrix");

    assert!(imatrix.data.contains_key("blk.0.attn_q.weight"));
    assert!(imatrix.data.contains_key("output.weight"));

    let q_data = imatrix.data.get("blk.0.attn_q.weight").unwrap();
    assert_eq!(q_data.len(), 3);
    assert_eq!(q_data[0], 1.0);
}
