use super::testing::{CheckpointLayout, FixtureTensor, checkpoint, stored_zip};
use super::*;

fn write(dir: &Path, name: &str, bytes: &[u8]) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, bytes).unwrap();
    path
}

fn sample() -> Vec<FixtureTensor> {
    vec![
        FixtureTensor::f32(
            "model.embed.weight",
            &[2, 3],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ),
        FixtureTensor::bf16_bits("model.norm.weight", &[4], &[0x3f80, 0x4000, 0x4040, 0x4080]),
        FixtureTensor::f32("model.scalar", &[], &[7.5]),
    ]
}

#[test]
fn tensors_are_read_with_names_dtypes_shapes_and_bytes() {
    for zip64 in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let layout = CheckpointLayout {
            zip64,
            ..Default::default()
        };
        let path = write(
            dir.path(),
            "pytorch_model.bin",
            &checkpoint(&sample(), &layout),
        );

        let archive = TorchArchive::open(&path).unwrap();
        let mut tensors = archive.tensors(None).unwrap();
        tensors.sort_by(|a, b| a.name.cmp(&b.name));

        let names: Vec<_> = tensors.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(
            names,
            ["model.embed.weight", "model.norm.weight", "model.scalar"]
        );
        assert_eq!(tensors[0].dtype, TorchDtype::F32);
        assert_eq!(tensors[0].shape, [2, 3]);
        assert_eq!(tensors[0].numel(), 6);
        let values: Vec<f32> = tensors[0]
            .data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        assert_eq!(values, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "zip64: {zip64}");
        assert_eq!(tensors[1].dtype, TorchDtype::Bf16);
        assert_eq!(
            tensors[1].data,
            [0x80, 0x3f, 0x00, 0x40, 0x40, 0x40, 0x80, 0x40]
        );
        assert!(tensors[2].shape.is_empty());
        assert_eq!(tensors[2].numel(), 1);
    }
}

#[test]
fn a_wrapped_state_dict_is_found_and_groups_are_listed() {
    let dir = tempfile::tempdir().unwrap();
    let layout = CheckpointLayout {
        wrapper: Some("state_dict".into()),
        ..Default::default()
    };
    let path = write(dir.path(), "m.pt", &checkpoint(&sample(), &layout));
    let archive = TorchArchive::open(&path).unwrap();

    assert_eq!(archive.groups(), ["state_dict"]);
    assert_eq!(archive.tensors(None).unwrap().len(), 3);
    assert_eq!(archive.tensors(Some("state_dict")).unwrap().len(), 3);
    let err = archive.tensors(Some("generator")).unwrap_err().to_string();
    assert!(err.contains("generator"), "{err}");
}

#[test]
fn a_tensor_longer_than_its_storage_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let mut liar = FixtureTensor::f32("w", &[4], &[1.0, 2.0, 3.0, 4.0]);
    liar.data.truncate(8); // the storage holds two of the four elements
    let path = write(
        dir.path(),
        "m.bin",
        &checkpoint(&[liar], &CheckpointLayout::default()),
    );
    let err = TorchArchive::open(&path).unwrap_err().to_string();
    assert!(err.contains("past the end"), "{err}");
}

#[test]
fn files_that_are_not_checkpoints_are_errors() {
    let dir = tempfile::tempdir().unwrap();
    for junk in [&b""[..], b"hello", &[0u8; 100][..]] {
        let path = write(dir.path(), "junk.bin", junk);
        assert!(TorchArchive::open(&path).is_err());
    }
    // A valid archive without a pickle.
    let zip = stored_zip(&[("a/readme.txt", b"hi")], false);
    let path = write(dir.path(), "nopickle.bin", &zip);
    let err = TorchArchive::open(&path).unwrap_err().to_string();
    assert!(err.contains("data.pkl"), "{err}");
    assert!(TorchArchive::open(dir.path().join("missing.bin")).is_err());
}

#[test]
fn a_big_endian_checkpoint_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let good = checkpoint(&sample(), &CheckpointLayout::default());
    let swapped = good
        .windows(6)
        .position(|w| w == b"little")
        .map(|at| {
            let mut bytes = good.clone();
            bytes[at..at + 6].copy_from_slice(b"bigbig");
            bytes
        })
        .unwrap();
    let err = TorchArchive::open(write(dir.path(), "be.bin", &swapped))
        .unwrap_err()
        .to_string();
    assert!(err.contains("big-endian"), "{err}");
}

#[test]
fn a_pickle_that_calls_anything_else_is_refused_and_runs_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let marker = dir.path().join("pwned");
    // `os.system("touch <marker>")`: the classic.
    let mut pickle = vec![0x80, 2];
    pickle.extend_from_slice(b"cos\nsystem\n");
    let command = format!("touch {}", marker.display());
    pickle.push(b'X');
    pickle.extend_from_slice(&(command.len() as u32).to_le_bytes());
    pickle.extend_from_slice(command.as_bytes());
    pickle.extend_from_slice(&[0x85, b'R', b'.']);
    let zip = stored_zip(
        &[
            ("archive/data.pkl", &pickle),
            ("archive/byteorder", b"little"),
        ],
        false,
    );
    let path = write(dir.path(), "evil.bin", &zip);

    let err = TorchArchive::open(&path).unwrap_err().to_string();

    assert!(err.contains("os.system"), "{err}");
    assert!(!marker.exists(), "the pickle's command must not run");
}

#[test]
fn safetensors_written_from_a_checkpoint_read_back_exactly() {
    let dir = tempfile::tempdir().unwrap();
    let path = write(
        dir.path(),
        "m.bin",
        &checkpoint(&sample(), &CheckpointLayout::default()),
    );
    let archive = TorchArchive::open(&path).unwrap();
    let dst = dir.path().join("out.safetensors");
    write_safetensors(&archive.tensors(None).unwrap(), &dst).unwrap();

    // The reference reader agrees on names, types, shapes and bytes.
    let bytes = std::fs::read(&dst).unwrap();
    let reference = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    assert_eq!(reference.len(), 3);
    let embed = reference.tensor("model.embed.weight").unwrap();
    assert_eq!(embed.dtype(), safetensors::Dtype::F32);
    assert_eq!(embed.shape(), [2, 3]);
    assert_eq!(embed.data().len(), 24);
    let norm = reference.tensor("model.norm.weight").unwrap();
    assert_eq!(norm.dtype(), safetensors::Dtype::BF16);
    assert_eq!(
        norm.data(),
        [0x80, 0x3f, 0x00, 0x40, 0x40, 0x40, 0x80, 0x40]
    );
    assert_eq!(
        reference.tensor("model.scalar").unwrap().shape(),
        Vec::<usize>::new()
    );
    // And so does this crate's own header check.
    assert_eq!(
        crate::tensors::summarize_safetensors(&dst).unwrap().tensors,
        3
    );
    assert!(!dir.path().join("out.safetensors.tmp").exists());
}

#[test]
fn a_single_file_model_directory_converts_and_verifies() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "config.json", br#"{"model_type":"llama"}"#);
    write(
        dir.path(),
        "pytorch_model.bin",
        &checkpoint(&sample(), &CheckpointLayout::default()),
    );

    let outputs = convert_pytorch_dir(dir.path()).unwrap();

    assert_eq!(outputs, [dir.path().join("model.safetensors")]);
    let report = crate::verify_model_dir(dir.path());
    assert!(report.is_complete(), "{:?}", report.problems);
    assert_eq!(report.format, Some(crate::WeightFormat::Safetensors));
    // Converting again changes nothing.
    let before = std::fs::metadata(&outputs[0]).unwrap().modified().unwrap();
    convert_pytorch_dir(dir.path()).unwrap();
    assert_eq!(
        std::fs::metadata(&outputs[0]).unwrap().modified().unwrap(),
        before
    );
}

#[test]
fn a_sharded_model_directory_converts_with_an_index() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "config.json", b"{}");
    let shard1 = vec![FixtureTensor::f32("a.weight", &[2], &[1.0, 2.0])];
    let shard2 = vec![FixtureTensor::f32("b.weight", &[3], &[3.0, 4.0, 5.0])];
    write(
        dir.path(),
        "pytorch_model-00001-of-00002.bin",
        &checkpoint(&shard1, &CheckpointLayout::default()),
    );
    write(
        dir.path(),
        "pytorch_model-00002-of-00002.bin",
        &checkpoint(&shard2, &CheckpointLayout::default()),
    );
    write(
        dir.path(),
        "pytorch_model.bin.index.json",
        br#"{"metadata":{"total_size":20},
            "weight_map":{"a.weight":"pytorch_model-00001-of-00002.bin",
                          "b.weight":"pytorch_model-00002-of-00002.bin"}}"#,
    );

    let outputs = convert_pytorch_dir(dir.path()).unwrap();

    assert_eq!(outputs.len(), 2);
    let index: serde_json::Value = serde_json::from_slice(
        &std::fs::read(dir.path().join("model.safetensors.index.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        index["weight_map"]["a.weight"],
        "model-00001-of-00002.safetensors"
    );
    assert_eq!(
        index["weight_map"]["b.weight"],
        "model-00002-of-00002.safetensors"
    );
    assert_eq!(index["metadata"]["total_size"], 20);
    let report = crate::verify_model_dir(dir.path());
    assert!(report.is_complete(), "{:?}", report.problems);
    assert_eq!(report.safetensors.unwrap().tensors, 2);
}

/// A checkpoint written by PyTorch itself, named by `PMETAL_TEST_TORCH_CHECKPOINT`
/// (a `.pt`/`.pth`/`.bin`): every tensor is read, written as safetensors, and
/// read back by the reference reader with the same bytes.
#[test]
#[ignore = "needs a real checkpoint: set PMETAL_TEST_TORCH_CHECKPOINT"]
fn a_checkpoint_written_by_pytorch_reads_and_converts() {
    let path = std::env::var("PMETAL_TEST_TORCH_CHECKPOINT")
        .expect("PMETAL_TEST_TORCH_CHECKPOINT names a checkpoint file");
    let archive = TorchArchive::open(&path).unwrap();
    let groups = archive.groups();
    eprintln!("groups: {groups:?}");
    // A trainer's checkpoint holds several groups and no default.
    let tensors = archive
        .tensors(None)
        .or_else(|_| archive.tensors(Some(groups[0])))
        .unwrap();
    assert!(!tensors.is_empty());
    for t in &tensors {
        assert_eq!(t.data.len(), t.numel() * t.dtype.size(), "{}", t.name);
    }

    let dir = tempfile::tempdir().unwrap();
    let dst = dir.path().join("converted.safetensors");
    write_safetensors(&tensors, &dst).unwrap();
    let bytes = std::fs::read(&dst).unwrap();
    let reference = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    assert_eq!(reference.len(), tensors.len());
    for t in &tensors {
        let back = reference.tensor(&t.name).unwrap();
        assert_eq!(back.data(), t.data, "{}", t.name);
        assert_eq!(back.shape(), t.shape, "{}", t.name);
    }
    eprintln!("{} tensors, {} bytes", tensors.len(), bytes.len());
}

#[test]
fn converting_a_directory_without_pickles_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    assert!(convert_pytorch_dir(dir.path()).is_err());
}

#[test]
fn two_tensors_with_one_name_cannot_be_written() {
    let dir = tempfile::tempdir().unwrap();
    let t = TorchTensor {
        name: "w".into(),
        dtype: TorchDtype::F32,
        shape: vec![1],
        data: &[0, 0, 0, 0],
    };
    let err = write_safetensors(&[t.clone(), t], &dir.path().join("x.safetensors")).unwrap_err();
    assert!(err.to_string().contains("share a name"), "{err}");
}
