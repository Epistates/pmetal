//! `InlineArray::save_safetensors` writes the file it was asked to write, all
//! or nothing, and says when it could not.

use std::path::Path;

use pmetal_runtime::compat::Dtype;
use pmetal_runtime::inline_array::load_safetensors_shard;
use pmetal_runtime::{InlineArray, RuntimeError, check_unobserved_error};

fn save(path: &Path, entries: &[(&str, &InlineArray)]) -> Result<(), RuntimeError> {
    InlineArray::save_safetensors(path.to_str().expect("utf-8 path"), entries)
}

fn load(path: &Path) -> Vec<(String, InlineArray)> {
    let mut tensors =
        load_safetensors_shard(path.to_str().expect("utf-8 path")).expect("the file loads");
    tensors.sort_by(|a, b| a.0.cmp(&b.0));
    tensors
}

/// MLX's path overload appends ".safetensors" to a name without it, so a save
/// to `best_snapshot.safetensors.tmp` (the checkpoint writer's temp name)
/// wrote `best_snapshot.safetensors.tmp.safetensors` and the rename that
/// followed failed on a file that was never there.
#[test]
fn the_file_is_written_at_exactly_the_given_path() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let dir = tmp.path();
    let path = dir.join("snapshot.safetensors.tmp");
    let w = InlineArray::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    save(&path, &[("w", &w)]).expect("save");

    assert!(path.exists(), "nothing at {path:?}");
    assert!(!dir.join("snapshot.safetensors.tmp.safetensors").exists());
    assert!(
        !dir.join("snapshot.safetensors.tmp.partial").exists(),
        "the partial file is gone after a save"
    );

    let final_path = dir.join("snapshot.safetensors");
    std::fs::rename(&path, &final_path).expect("rename");
    let mut loaded = load(&final_path);
    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded[0].0, "w");
    assert_eq!(loaded[0].1.shape(), &[2, 2]);
    assert_eq!(loaded[0].1.to_f32_vec(4).unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    check_unobserved_error().expect("bridge error");
}

#[test]
fn dtypes_round_trip_bit_for_bit() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let dir = tmp.path();
    let path = dir.join("m.safetensors");
    let f32s = InlineArray::from_f32_slice(&[0.5, -1.5, 3.25], &[3]);
    let bf16s =
        InlineArray::from_f32_slice(&[0.5, -1.5, 3.25], &[3]).as_dtype(Dtype::Bfloat16.as_i32());
    let u32s = InlineArray::from_u32_slice(&[7, 8, 4_000_000_000], &[3]);
    save(&path, &[("a", &f32s), ("b", &bf16s), ("c", &u32s)]).expect("save");

    let loaded = load(&path);
    let names: Vec<_> = loaded.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(names, ["a", "b", "c"]);
    assert_eq!(loaded[0].1.dtype(), Dtype::Float32);
    assert_eq!(loaded[1].1.dtype(), Dtype::Bfloat16);
    assert_eq!(loaded[2].1.dtype(), Dtype::Uint32);
    let mut b = loaded[1].1.clone();
    assert_eq!(b.to_f32_vec(3).unwrap(), vec![0.5, -1.5, 3.25]);
    assert_eq!(loaded[2].1.as_slice::<u32>(), &[7, 8, 4_000_000_000]);
    check_unobserved_error().expect("bridge error");
}

/// A save that cannot finish must not touch the file it was going to replace.
#[test]
fn a_failed_save_leaves_the_previous_file_alone() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let dir = tmp.path();
    let path = dir.join("m.safetensors");
    let v1 = InlineArray::from_f32_slice(&[1.0, 2.0], &[2]);
    save(&path, &[("w", &v1)]).expect("first save");
    let before = std::fs::read(&path).expect("read v1");

    let v2 = InlineArray::from_f32_slice(&[9.0, 9.0], &[2]);
    save(&path, &[("w", &v2), ("w", &v2)]).expect_err("a duplicate name");

    assert_eq!(std::fs::read(&path).expect("read after"), before);
    assert!(!dir.join("m.safetensors.partial").exists());
    pmetal_runtime::clear_last_error();
}

/// A save that fails after the bytes are written (here the rename, onto a
/// directory) takes its partial file with it.
#[test]
fn a_failed_rename_removes_the_partial_file() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let dir = tmp.path();
    let path = dir.join("m.safetensors");
    std::fs::create_dir(&path).expect("a directory where the file should go");
    let a = InlineArray::from_f32_slice(&[1.0], &[1]);
    let err = save(&path, &[("w", &a)]).expect_err("cannot replace a directory");
    assert!(err.to_string().contains("unable to move"), "{err}");
    assert!(!dir.join("m.safetensors.partial").exists());
    assert!(path.is_dir());
    pmetal_runtime::clear_last_error();
}

#[test]
fn a_duplicate_name_is_an_error_not_a_dropped_tensor() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let dir = tmp.path();
    let path = dir.join("m.safetensors");
    let a = InlineArray::from_f32_slice(&[1.0], &[1]);
    let b = InlineArray::from_f32_slice(&[2.0], &[1]);
    let err = save(&path, &[("w", &a), ("w", &b)]).expect_err("two tensors named w");
    assert!(err.to_string().contains("duplicate"), "{err}");
    assert!(!path.exists());
    pmetal_runtime::clear_last_error();
}

/// The save used to return `()`, so a missing directory or a full disk went
/// unnoticed by every caller that wrote a checkpoint.
#[test]
fn an_unwritable_path_is_an_error() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let dir = tmp.path();
    let path = dir.join("no-such-dir").join("m.safetensors");
    let a = InlineArray::from_f32_slice(&[1.0], &[1]);
    let err = save(&path, &[("w", &a)]).expect_err("the directory does not exist");
    assert!(err.to_string().contains("save_safetensors"), "{err}");
    pmetal_runtime::clear_last_error();
}

#[test]
fn a_nul_in_a_name_is_an_error_not_a_panic() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let dir = tmp.path();
    let a = InlineArray::from_f32_slice(&[1.0], &[1]);
    assert!(save(&dir.join("m.safetensors"), &[("w\0x", &a)]).is_err());
}
