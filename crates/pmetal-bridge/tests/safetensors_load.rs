//! Loading safetensors says why it failed, and never narrows a value quietly.

use std::collections::HashMap;
use std::path::PathBuf;

use pmetal_bridge::compat::Dtype;
use pmetal_bridge::inline_array::{InlineArray, try_load_safetensors_shard};
use pmetal_bridge::{check_unobserved_error, expect_bridge_errors};
use safetensors::tensor::{Dtype as SafeDtype, TensorView};

fn temp_file(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("pmetal-load-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    dir.join(name)
}

fn write(path: &PathBuf, tensors: Vec<(&str, SafeDtype, Vec<usize>, Vec<u8>)>) {
    let views: HashMap<String, TensorView<'_>> = tensors
        .iter()
        .map(|(name, dtype, shape, bytes)| {
            (
                name.to_string(),
                TensorView::new(*dtype, shape.clone(), bytes).expect("view"),
            )
        })
        .collect();
    safetensors::serialize_to_file(&views, None, path).expect("write safetensors");
}

/// The loader returned `None` for every failure, so a missing file, a corrupt
/// header and an unsupported dtype all surfaced as "failed to load".
#[test]
fn a_missing_file_is_an_error_that_names_the_file() {
    let path = temp_file("absent.safetensors");
    let err = try_load_safetensors_shard(path.to_str().unwrap()).expect_err("no such file");
    let text = err.to_string();
    assert!(text.contains("absent.safetensors"), "{text}");
    assert!(text.contains("cannot open"), "{text}");
}

#[test]
fn a_corrupt_header_is_an_error() {
    let path = temp_file("garbage.safetensors");
    std::fs::write(&path, b"this is not a safetensors file at all").unwrap();
    let err = try_load_safetensors_shard(path.to_str().unwrap()).expect_err("garbage");
    assert!(err.to_string().contains("header"), "{err}");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn a_good_file_loads_with_its_values() {
    let path = temp_file("good.safetensors");
    let values = [1.5f32, -2.0, 3.25];
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    write(&path, vec![("w", SafeDtype::F32, vec![3], bytes)]);

    let mut loaded = try_load_safetensors_shard(path.to_str().unwrap()).expect("loads");
    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded[0].0, "w");
    assert_eq!(loaded[0].1.to_f32_vec(3).unwrap(), values);
    check_unobserved_error().expect("bridge error");
    let _ = std::fs::remove_file(&path);
}

/// An FP8 payload loads as the uint8 bytes MLX stores it as (whichever of the
/// two loaders reads it).
#[test]
fn an_fp8_payload_loads_as_uint8_bytes() {
    let path = temp_file("fp8.safetensors");
    let payload: Vec<u8> = vec![0x38, 0x40, 0xb8, 0x00];
    write(
        &path,
        vec![("w", SafeDtype::F8_E4M3, vec![4], payload.clone())],
    );

    let loaded = expect_bridge_errors(|| try_load_safetensors_shard(path.to_str().unwrap()))
        .expect("an FP8 file loads");
    pmetal_bridge::clear_last_error();
    assert_eq!(loaded[0].1.dtype(), Dtype::Uint8);
    assert_eq!(loaded[0].1.size(), 4);
    let mut bytes = loaded[0].1.clone().as_dtype(Dtype::Float32.as_i32());
    assert_eq!(
        bytes.to_f32_vec(4).unwrap(),
        payload.iter().map(|&b| f32::from(b)).collect::<Vec<_>>()
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn a_single_key_load_reports_a_missing_key() {
    let path = temp_file("one.safetensors");
    let bytes: Vec<u8> = 1.0f32.to_le_bytes().to_vec();
    write(&path, vec![("w", SafeDtype::F32, vec![1], bytes)]);

    let err =
        expect_bridge_errors(|| InlineArray::try_load_safetensors(path.to_str().unwrap(), "nope"))
            .expect_err("no tensor named nope");
    pmetal_bridge::clear_last_error();
    assert!(err.to_string().contains("nope"), "{err}");

    let w = InlineArray::try_load_safetensors(path.to_str().unwrap(), "w").expect("w loads");
    assert_eq!(w.item_f32(), 1.0);
    let _ = std::fs::remove_file(&path);
}
