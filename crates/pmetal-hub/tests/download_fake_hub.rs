//! Model and dataset downloads against a fake Hub on loopback.
//!
//! One hub serves the whole test binary (the endpoint and cache directory are
//! process-wide environment), so every test works on repositories of its own.

#![allow(unsafe_code)] // `std::env::set_var` (edition 2024), set once before any client exists.

mod support;

use std::path::Path;
use std::sync::OnceLock;

use pmetal_core::SecretString;
use pmetal_hub::{download_dataset_parquet, download_model, find_cached_model};
use support::fixtures::{gguf_file, safetensors_file};
use support::{Access, FakeHub, Kind, Repo, commit_id};

fn hub() -> &'static FakeHub {
    static HUB: OnceLock<FakeHub> = OnceLock::new();
    HUB.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                tx.send(FakeHub::start().await).unwrap();
                std::future::pending::<()>().await;
            });
        });
        let hub = rx.recv().unwrap();
        let cache = tempfile::tempdir().unwrap().keep();
        let home = tempfile::tempdir().unwrap().keep();
        // SAFETY: this runs once, before the first Hub client is built, and no
        // test in this binary reads the environment concurrently.
        unsafe {
            std::env::set_var("HF_ENDPOINT", &hub.base);
            std::env::set_var("HF_HUB_CACHE", &cache);
            std::env::set_var("HF_HOME", &home);
            std::env::remove_var("HF_TOKEN");
            std::env::remove_var("HF_TOKEN_PATH");
            std::env::remove_var("HF_HUB_OFFLINE");
            std::env::remove_var("HF_HUB_DISABLE_IMPLICIT_TOKEN");
        }
        hub
    })
}

const CONFIG: &[u8] = br#"{"model_type":"llama"}"#;

fn big(seed: u8, len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| seed.wrapping_add((i % 251) as u8))
        .collect()
}

fn read(dir: &Path, rel: &str) -> Vec<u8> {
    std::fs::read(dir.join(rel)).unwrap_or_else(|e| panic!("{}: {e}", dir.join(rel).display()))
}

#[tokio::test]
async fn a_download_is_one_consistent_commit_even_if_the_branch_moves() {
    let hub = hub();
    let weights_v1 = safetensors_file("one", 2000);
    let weights_v2 = safetensors_file("two", 2000);
    let mut repo = Repo::default();
    repo.commit(
        "main",
        "pin-1",
        &[
            ("config.json", CONFIG),
            ("model.safetensors", &weights_v1),
            ("tokenizer.json", b"{}"),
        ],
    );
    hub.add_model("pin/model", repo);
    // A push lands as soon as the first file has been fetched.
    let v2 = weights_v2.clone();
    hub.after_download("config.json", move |repos| {
        let repo = repos
            .get_mut(&(Kind::Model, "pin/model".to_owned()))
            .unwrap();
        repo.commit(
            "main",
            "pin-2",
            &[
                ("config.json", CONFIG),
                ("model.safetensors", &v2),
                ("tokenizer.json", b"{}"),
            ],
        );
    });

    let dir = download_model("pin/model", None, None).await.unwrap();

    // Every file is in the one snapshot directory that was returned, and all
    // of them are from the same commit: the one `main` pointed at when the
    // download started.
    assert_eq!(
        dir.file_name().unwrap().to_str().unwrap(),
        commit_id("pin-1")
    );
    assert_eq!(read(&dir, "config.json"), CONFIG);
    assert_eq!(read(&dir, "tokenizer.json"), b"{}");
    assert_eq!(read(&dir, "model.safetensors"), weights_v1);
}

#[tokio::test]
async fn every_part_of_a_split_gguf_is_downloaded() {
    let hub = hub();
    let part1 = gguf_file(Some((0, 2, 2)), &[("a", 1.0)]);
    let part2 = gguf_file(Some((1, 2, 2)), &[("b", 2.0)]);
    hub.add_model(
        "gguf/split",
        Repo::with_files(&[
            ("README.md", b"# split"),
            ("model-Q4_K_M-00001-of-00002.gguf", &part1),
            ("model-Q4_K_M-00002-of-00002.gguf", &part2),
            ("model-Q8_0.gguf", &gguf_file(None, &[("a", 3.0)])),
        ]),
    );

    let dir = download_model("gguf/split", None, None).await.unwrap();

    assert_eq!(read(&dir, "model-Q4_K_M-00001-of-00002.gguf"), part1);
    assert_eq!(read(&dir, "model-Q4_K_M-00002-of-00002.gguf"), part2);
    assert!(
        !dir.join("model-Q8_0.gguf").exists(),
        "only one quantization is pulled"
    );
}

#[tokio::test]
async fn shards_the_index_does_not_name_are_not_downloaded() {
    let hub = hub();
    let index = br#"{"weight_map":{"a":"model-00001-of-00002.safetensors",
                                    "b":"model-00002-of-00002.safetensors"}}"#;
    hub.add_model(
        "shards/indexed",
        Repo::with_files(&[
            ("config.json", CONFIG),
            ("model.safetensors.index.json", index),
            (
                "model-00001-of-00002.safetensors",
                &safetensors_file("a", 1000),
            ),
            (
                "model-00002-of-00002.safetensors",
                &safetensors_file("b", 1000),
            ),
            // A second copy of the weights in another layout.
            ("consolidated.safetensors", &safetensors_file("c", 3000)),
        ]),
    );

    let dir = download_model("shards/indexed", None, None).await.unwrap();

    assert!(dir.join("model-00002-of-00002.safetensors").exists());
    assert!(!dir.join("consolidated.safetensors").exists());
    assert!(
        !hub.downloaded()
            .iter()
            .any(|p| p.contains("shards/indexed") && p.contains("consolidated")),
        "the unreferenced file must not even be fetched"
    );
}

#[tokio::test]
async fn a_gated_repo_says_how_to_get_access() {
    let hub = hub();
    let mut repo = Repo::with_files(&[("config.json", CONFIG)]);
    repo.access = Access::Gated(vec!["hf_accepted_the_license_0000".into()]);
    hub.add_model("gated/llama", repo);

    let token = SecretString::new("hf_valid_but_not_accepted_1111");
    let err = download_model("gated/llama", None, Some(&token))
        .await
        .unwrap_err()
        .to_string();

    assert!(err.contains("gated"), "{err}");
    assert!(err.contains("https://huggingface.co/gated/llama"), "{err}");
    assert!(!err.contains("hf_valid_but_not_accepted_1111"), "{err}");
}

#[tokio::test]
async fn a_truncated_transfer_is_an_error_and_leaves_nothing_cached() {
    let hub = hub();
    hub.add_model(
        "flaky/model",
        Repo::with_files(&[
            ("config.json", CONFIG),
            ("model.safetensors", &safetensors_file("w", 4096)),
        ]),
    );
    hub.truncate_downloads("flaky/model", "model.safetensors");

    let result = download_model("flaky/model", None, None).await;

    assert!(result.is_err(), "a half-sent weights file is not a model");
    assert!(find_cached_model("flaky/model").is_none());
}

#[tokio::test]
async fn every_shard_of_a_multi_shard_dataset_split_is_found() {
    let hub = hub();
    hub.add_dataset(
        "shard/data",
        Repo::with_files(&[
            ("README.md", b"# data"),
            ("data/train-00000-of-00003.parquet", &big(10, 1500)),
            ("data/train-00001-of-00003.parquet", &big(11, 1500)),
            ("data/train-00002-of-00003.parquet", &big(12, 1500)),
            ("data/test-00000-of-00001.parquet", &big(13, 1500)),
        ]),
    );

    let paths = download_dataset_parquet("shard/data", "train", None, None)
        .await
        .unwrap();

    let mut names: Vec<_> = paths
        .iter()
        .map(|p| p.file_name().unwrap().to_str().unwrap().to_owned())
        .collect();
    names.sort();
    assert_eq!(
        names,
        [
            "train-00000-of-00003.parquet",
            "train-00001-of-00003.parquet",
            "train-00002-of-00003.parquet",
        ]
    );
}

#[tokio::test]
async fn finding_a_split_does_not_probe_the_hub_with_guesses() {
    let hub = hub();
    hub.add_dataset(
        "probe/data",
        Repo::with_files(&[("data/train-00000-of-00001.parquet", &big(14, 1200))]),
    );

    download_dataset_parquet("probe/data", "train", None, None)
        .await
        .unwrap();

    let requests = hub
        .log()
        .into_iter()
        .filter(|r| r.path.contains("probe/data"))
        .count();
    assert!(
        requests <= 10,
        "listing the repository once is enough; {requests} requests were made"
    );
}
