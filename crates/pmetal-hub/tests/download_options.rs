//! Model downloads with explicit options against a fake Hub of the test's own:
//! revisions, the cache, offline mode, credentials, failures, progress.

mod support;

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use pmetal_core::SecretString;
use pmetal_hub::torch::testing::{CheckpointLayout, FixtureTensor, checkpoint};
use pmetal_hub::verify_model_dir;
use pmetal_hub::{
    DownloadOptions, HubOptions, TransferProgress, Verification, WeightFormat,
    download_dataset_with, download_model_with, find_cached_model_in, plan_model_download,
    resolve_model_path_with,
};
use support::fixtures::{gguf_file, isolate_env, safetensors_file};
use support::{Access, FakeHub, Kind, Repo, commit_id};

const CONFIG: &[u8] = br#"{"model_type":"llama"}"#;

async fn hub_with_model(id: &str) -> (FakeHub, tempfile::TempDir) {
    isolate_env();
    let hub = FakeHub::start().await;
    hub.add_model(
        id,
        Repo::with_files(&[
            ("config.json", CONFIG),
            ("tokenizer.json", b"{}"),
            ("model.safetensors", &safetensors_file("w", 3000)),
        ]),
    );
    (hub, tempfile::tempdir().unwrap())
}

fn options(hub: &FakeHub, cache: &Path) -> DownloadOptions {
    DownloadOptions {
        hub: HubOptions {
            endpoint: Some(hub.base.clone()),
            cache_dir: Some(cache.to_path_buf()),
            offline: Some(false),
            max_retries: Some(1),
            ..Default::default()
        },
        ..Default::default()
    }
}

#[tokio::test]
async fn a_named_branch_is_recorded_so_offline_mode_finds_it() {
    let (hub, cache) = hub_with_model("o/m").await;
    let mut opts = options(&hub, cache.path());
    opts.revision = Some("main".into());

    let online = download_model_with("o/m", &opts).await.unwrap();
    assert_eq!(
        online.file_name().unwrap().to_str().unwrap(),
        commit_id("c1")
    );

    hub.clear_log();
    opts.hub.offline = Some(true);
    let offline = download_model_with("o/m", &opts).await.unwrap();
    assert_eq!(offline, online);
    assert!(hub.log().is_empty(), "offline mode sent {:?}", hub.log());
}

#[tokio::test]
async fn offline_mode_without_a_cached_copy_fails_clearly() {
    let (hub, cache) = hub_with_model("o/m").await;
    let mut opts = options(&hub, cache.path());
    opts.hub.offline = Some(true);

    let err = download_model_with("o/m", &opts)
        .await
        .unwrap_err()
        .to_string();

    assert!(err.contains("offline"), "{err}");
    assert!(err.contains("o/m"), "{err}");
    assert!(hub.log().is_empty());
}

#[tokio::test]
async fn with_no_revision_a_cached_model_costs_no_requests() {
    let (hub, cache) = hub_with_model("o/m").await;
    let opts = options(&hub, cache.path());
    let first = download_model_with("o/m", &opts).await.unwrap();

    hub.clear_log();
    let second = download_model_with("o/m", &opts).await.unwrap();

    assert_eq!(first, second);
    assert!(hub.log().is_empty(), "{:?}", hub.log());
}

#[tokio::test]
async fn naming_the_branch_picks_up_a_newer_commit() {
    let (hub, cache) = hub_with_model("o/m").await;
    let mut opts = options(&hub, cache.path());
    let first = download_model_with("o/m", &opts).await.unwrap();

    hub.edit(Kind::Model, "o/m", |repo| {
        repo.commit(
            "main",
            "c2",
            &[
                ("config.json", CONFIG),
                ("model.safetensors", &safetensors_file("w2", 3000)),
            ],
        );
    });
    opts.revision = Some("main".into());
    let second = download_model_with("o/m", &opts).await.unwrap();

    assert_ne!(first, second);
    assert_eq!(
        second.file_name().unwrap().to_str().unwrap(),
        commit_id("c2")
    );
    // The old commit is still there, and `main` now names the new one.
    assert!(first.join("model.safetensors").exists());
    assert_eq!(
        find_cached_model_in(cache.path(), "o/m", None).as_deref(),
        Some(second.as_path())
    );
}

#[tokio::test]
async fn an_unreachable_hub_falls_back_to_the_branch_last_cached() {
    let (hub, cache) = hub_with_model("o/m").await;
    let mut opts = options(&hub, cache.path());
    opts.revision = Some("main".into());
    let online = download_model_with("o/m", &opts).await.unwrap();

    hub.fail("/api/models/o/m", 503, 1000);
    let fallback = download_model_with("o/m", &opts).await.unwrap();

    assert_eq!(fallback, online);
}

#[tokio::test]
async fn an_unreachable_hub_with_nothing_cached_is_an_error() {
    let (hub, cache) = hub_with_model("o/m").await;
    hub.fail("/api/models/o/m", 503, 1000);
    let err = download_model_with("o/m", &options(&hub, cache.path()))
        .await
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("503") || err.to_lowercase().contains("unavailable"),
        "{err}"
    );
}

#[tokio::test]
async fn a_hub_that_stops_answering_times_out_instead_of_hanging() {
    let (hub, cache) = hub_with_model("o/m").await;
    hub.fail("/api/models/o/m", 0, 1000); // status 0: never answer
    let mut opts = options(&hub, cache.path());
    opts.hub.stall_timeout = Some(Duration::from_millis(300));
    opts.hub.max_retries = Some(0);

    let started = std::time::Instant::now();
    let result =
        tokio::time::timeout(Duration::from_secs(20), download_model_with("o/m", &opts)).await;

    assert!(result.expect("the call must not hang").is_err());
    assert!(started.elapsed() < Duration::from_secs(10));
}

#[tokio::test]
async fn transient_failures_are_retried_and_then_succeed() {
    let (hub, cache) = hub_with_model("o/m").await;
    hub.fail("/api/models/o/m", 503, 1);
    let dir = download_model_with("o/m", &options(&hub, cache.path()))
        .await
        .unwrap();
    assert!(dir.join("model.safetensors").exists());
}

#[tokio::test]
async fn a_rate_limit_is_reported_with_the_wait() {
    let (hub, cache) = hub_with_model("o/m").await;
    hub.fail("/api/models/o/m", 429, 1000);
    let mut opts = options(&hub, cache.path());
    opts.hub.max_retries = Some(0);
    let err = download_model_with("o/m", &opts)
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("rate limiting"), "{err}");
    assert!(err.contains("Retry in 1s"), "{err}");
    // Bounded: one request, no retry loop.
    assert_eq!(hub.requests_to("/api/models/o/m").len(), 1);
}

#[tokio::test]
async fn a_private_repository_needs_the_right_token_and_never_echoes_it() {
    let (hub, cache) = hub_with_model("o/private").await;
    hub.edit(Kind::Model, "o/private", |repo| {
        repo.access = Access::Private("hf_right_token_0123456789abcdef".into());
    });
    let mut opts = options(&hub, cache.path());

    let err = download_model_with("o/private", &opts)
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("HF_TOKEN"), "{err}");

    opts.hub.token = Some(SecretString::new("hf_wrong_token_0123456789abcdef"));
    let err = download_model_with("o/private", &opts)
        .await
        .unwrap_err()
        .to_string();
    assert!(!err.contains("hf_wrong_token"), "{err}");
    assert!(
        !format!("{opts:?}").contains("hf_wrong_token"),
        "token leaked into Debug"
    );

    hub.clear_log();
    opts.hub.token = Some(SecretString::new("  hf_right_token_0123456789abcdef\n"));
    let dir = download_model_with("o/private", &opts).await.unwrap();
    assert!(dir.join("config.json").exists());
    // Every request that reached the repository carried the (trimmed) token.
    for request in hub.requests_to("o/private") {
        if request.authorization.is_some() {
            assert_eq!(
                request.authorization.as_deref(),
                Some("Bearer hf_right_token_0123456789abcdef")
            );
        }
    }
    assert!(
        hub.requests_to("o/private")
            .iter()
            .any(|r| r.authorization.is_some())
    );
}

#[tokio::test]
async fn a_quantization_can_be_pulled_out_of_a_multi_quant_repo() {
    isolate_env();
    let hub = FakeHub::start().await;
    let cache = tempfile::tempdir().unwrap();
    hub.add_model(
        "o/m-GGUF",
        Repo::with_files(&[
            ("README.md", b"# quants"),
            ("m-Q4_K_M.gguf", &gguf_file(None, &[("a", 1.0)])),
            ("m-Q8_0.gguf", &gguf_file(None, &[("a", 2.0), ("b", 3.0)])),
            ("mmproj-F16.gguf", &gguf_file(None, &[("v", 1.0)])),
        ]),
    );
    let mut opts = options(&hub, cache.path());

    // By default: the best general-purpose quantization.
    let default = download_model_with("o/m-GGUF", &opts).await.unwrap();
    assert!(default.join("m-Q4_K_M.gguf").exists());
    assert!(!default.join("m-Q8_0.gguf").exists());
    // A GGUF-only repository gets the config it lacks, from the file itself.
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(default.join("config.json")).unwrap()).unwrap();
    assert_eq!(config["model_type"], "llama");

    // Asking for another one fetches that one, even though a copy is cached.
    opts.gguf_quant = Some("q8_0".into());
    let q8 = download_model_with("o/m-GGUF", &opts).await.unwrap();
    assert_eq!(q8, default, "same commit, same snapshot directory");
    assert!(q8.join("m-Q8_0.gguf").exists());
    assert!(!q8.join("mmproj-F16.gguf").exists());

    opts.gguf_quant = Some("Q2_K".into());
    let err = download_model_with("o/m-GGUF", &opts)
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("Q4_K_M") && err.contains("Q8_0"), "{err}");
}

#[tokio::test]
async fn a_colon_suffix_selects_the_quantization_when_resolving() {
    isolate_env();
    let hub = FakeHub::start().await;
    let cache = tempfile::tempdir().unwrap();
    hub.add_model(
        "o/m-GGUF",
        Repo::with_files(&[
            ("m-Q4_K_M.gguf", &gguf_file(None, &[("a", 1.0)])),
            ("m-Q8_0.gguf", &gguf_file(None, &[("a", 2.0)])),
        ]),
    );
    let dir = resolve_model_path_with("o/m-GGUF:Q8_0", &options(&hub, cache.path()))
        .await
        .unwrap();
    assert!(dir.join("m-Q8_0.gguf").exists());
    assert!(!dir.join("m-Q4_K_M.gguf").exists());
}

#[tokio::test]
async fn a_repository_with_only_pickles_comes_back_as_safetensors() {
    isolate_env();
    let hub = FakeHub::start().await;
    let cache = tempfile::tempdir().unwrap();
    let weights = checkpoint(
        &[
            FixtureTensor::f32("embed.weight", &[2, 2], &[1.0, 2.0, 3.0, 4.0]),
            FixtureTensor::f32("head.weight", &[2], &[5.0, 6.0]),
        ],
        &CheckpointLayout::default(),
    );
    hub.add_model(
        "o/old",
        Repo::with_files(&[("config.json", CONFIG), ("pytorch_model.bin", &weights)]),
    );
    let opts = options(&hub, cache.path());

    let dir = download_model_with("o/old", &opts).await.unwrap();

    // The pickles came down, and what loads is the safetensors beside them.
    assert!(dir.join("pytorch_model.bin").exists());
    let report = verify_model_dir(&dir);
    assert!(report.is_complete(), "{:?}", report.problems);
    assert_eq!(report.format, Some(WeightFormat::Safetensors));
    assert_eq!(report.safetensors.unwrap().tensors, 2);

    // A later call is served from the cache, converted, with no requests.
    hub.clear_log();
    let again = download_model_with("o/old", &opts).await.unwrap();
    assert_eq!(again, dir);
    assert!(hub.log().is_empty());
}

#[tokio::test]
async fn a_pickle_snapshot_from_another_tool_is_converted_on_first_use() {
    isolate_env();
    let hub = FakeHub::start().await;
    let cache = tempfile::tempdir().unwrap();
    let weights = checkpoint(
        &[FixtureTensor::f32("w", &[2], &[1.0, 2.0])],
        &CheckpointLayout::default(),
    );
    hub.add_model(
        "o/old",
        Repo::with_files(&[("config.json", CONFIG), ("pytorch_model.bin", &weights)]),
    );
    let mut opts = options(&hub, cache.path());
    // Offline, the cache answers: first a snapshot that only has the pickles.
    let dir = download_model_with("o/old", &opts).await.unwrap();
    std::fs::remove_file(dir.join("model.safetensors")).unwrap();
    assert_eq!(verify_model_dir(&dir).format, Some(WeightFormat::PyTorch));

    opts.hub.offline = Some(true);
    let again = download_model_with("o/old", &opts).await.unwrap();

    assert_eq!(
        verify_model_dir(&again).format,
        Some(WeightFormat::Safetensors)
    );
}

#[tokio::test]
async fn a_malicious_pickle_is_an_error_and_runs_nothing() {
    isolate_env();
    let hub = FakeHub::start().await;
    let cache = tempfile::tempdir().unwrap();
    let marker = cache.path().join("pwned");
    let mut pickle = vec![0x80, 2];
    pickle.extend_from_slice(b"cos\nsystem\n");
    let command = format!("touch {}", marker.display());
    pickle.push(b'X');
    pickle.extend_from_slice(&(command.len() as u32).to_le_bytes());
    pickle.extend_from_slice(command.as_bytes());
    pickle.extend_from_slice(&[0x85, b'R', b'.']);
    let evil = pmetal_hub::torch::testing::stored_zip(
        &[
            ("archive/data.pkl", &pickle),
            ("archive/byteorder", b"little"),
        ],
        false,
    );
    hub.add_model(
        "o/evil",
        Repo::with_files(&[("config.json", CONFIG), ("pytorch_model.bin", &evil)]),
    );

    let err = download_model_with("o/evil", &options(&hub, cache.path()))
        .await
        .unwrap_err()
        .to_string();

    assert!(err.contains("os.system"), "{err}");
    assert!(!marker.exists());
}

#[tokio::test]
async fn progress_reports_every_byte_and_ends_finished() {
    let (hub, cache) = hub_with_model("o/m").await;
    let seen: Arc<Mutex<Vec<TransferProgress>>> = Arc::default();
    let sink = seen.clone();
    let mut opts = options(&hub, cache.path());
    opts.progress = Some(Arc::new(move |p| sink.lock().unwrap().push(p.clone())));

    download_model_with("o/m", &opts).await.unwrap();

    let seen = seen.lock().unwrap();
    let last = seen.last().expect("progress was reported");
    assert!(last.finished);
    assert_eq!(last.files_done, 3);
    assert_eq!(last.bytes_done, last.total_bytes);
    assert!(last.total_bytes > 6000, "{last:?}");
    assert!(seen.windows(2).all(|w| w[0].bytes_done <= w[1].bytes_done));
}

#[tokio::test]
async fn a_dry_run_lists_the_files_and_their_size_without_fetching_weights() {
    let (hub, cache) = hub_with_model("o/m").await;
    let plan = plan_model_download("o/m", &options(&hub, cache.path()))
        .await
        .unwrap();

    assert_eq!(plan.commit, commit_id("c1"));
    let paths: Vec<_> = plan.model.files.iter().map(|f| f.path.as_str()).collect();
    assert_eq!(
        paths,
        ["config.json", "model.safetensors", "tokenizer.json"]
    );
    assert!(plan.model.total_bytes() > 6000);
    assert!(hub.downloaded().is_empty());
    assert!(find_cached_model_in(cache.path(), "o/m", None).is_none());
}

#[tokio::test]
async fn verification_can_hash_large_files_against_the_hubs_digest() {
    let (hub, cache) = hub_with_model("o/m").await;
    let mut opts = options(&hub, cache.path());
    opts.verify = Verification::Sha256;
    let dir = download_model_with("o/m", &opts).await.unwrap();
    assert!(dir.join("model.safetensors").exists());
}

#[tokio::test]
async fn sha256_verification_catches_a_same_length_corruption() {
    let (hub, cache) = hub_with_model("o/m").await;
    hub.corrupt_downloads("o/m", "model.safetensors");

    // The size check cannot see a flipped byte; that is what Sha256 is for.
    let size_only = download_model_with("o/m", &options(&hub, cache.path())).await;
    assert!(size_only.is_ok(), "{size_only:?}");

    let cache = tempfile::tempdir().unwrap();
    let mut opts = options(&hub, cache.path());
    opts.verify = Verification::Sha256;
    let err = download_model_with("o/m", &opts).await.unwrap_err();
    let message = err.to_string();
    assert!(
        message.contains("model.safetensors") && message.contains("SHA-256"),
        "{message}"
    );
    // The bad file is not left behind to be taken for the real one.
    assert!(find_cached_model_in(cache.path(), "o/m", None).is_none());
}

#[tokio::test]
async fn digests_survive_a_paginated_listing() {
    let (hub, cache) = hub_with_model("o/m").await;
    hub.paginate_listings(1);
    hub.corrupt_downloads("o/m", "model.safetensors");
    let mut opts = options(&hub, cache.path());
    opts.verify = Verification::Sha256;

    // model.safetensors is the second of three entries: its digest comes from
    // the page after the first.
    let err = download_model_with("o/m", &opts).await.unwrap_err();
    assert!(err.to_string().contains("SHA-256"), "{err}");
    assert!(hub.requests_to("/tree/").len() >= 3);
}

#[tokio::test]
async fn an_intact_download_passes_sha256_verification_across_pages() {
    let (hub, cache) = hub_with_model("o/m").await;
    hub.paginate_listings(2);
    let mut opts = options(&hub, cache.path());
    opts.verify = Verification::Sha256;
    let dir = download_model_with("o/m", &opts).await.unwrap();
    assert!(dir.join("model.safetensors").exists());
    assert!(dir.join("tokenizer.json").exists());
}

#[tokio::test]
async fn a_listing_cannot_send_the_client_to_another_host() {
    let (hub, cache) = hub_with_model("o/m").await;
    hub.paginate_listings(1);
    hub.link_listings_to("http://listing-elsewhere.invalid");
    let mut opts = options(&hub, cache.path());
    opts.hub.token = Some(SecretString::new("hf_secret"));

    let err = download_model_with("o/m", &opts).await.unwrap_err();
    assert!(err.to_string().contains("another host"), "{err}");
    assert!(!err.to_string().contains("hf_secret"));
}

#[tokio::test]
async fn a_dataset_download_is_pinned_and_limited_to_its_split() {
    isolate_env();
    let hub = FakeHub::start().await;
    let cache = tempfile::tempdir().unwrap();
    hub.add_dataset(
        "o/d",
        Repo::with_files(&[
            ("README.md", b"# d"),
            ("data/train-00000-of-00001.parquet", &[1u8; 2000]),
            ("data/test-00000-of-00001.parquet", &[2u8; 2000]),
        ]),
    );
    let opts = options(&hub, cache.path());

    let dir = download_dataset_with("o/d", Some("train"), &opts)
        .await
        .unwrap();

    assert_eq!(dir.file_name().unwrap().to_str().unwrap(), commit_id("c1"));
    assert!(dir.join("data/train-00000-of-00001.parquet").exists());
    assert!(!dir.join("data/test-00000-of-00001.parquet").exists());
    assert!(dir.join("README.md").exists());

    // Offline, the cached split is found; another split is not.
    let mut offline = opts.clone();
    offline.hub.offline = Some(true);
    assert!(
        download_dataset_with("o/d", Some("train"), &offline)
            .await
            .is_ok()
    );
    assert!(
        download_dataset_with("o/d", Some("test"), &offline)
            .await
            .is_err()
    );
}

#[tokio::test]
async fn a_transfer_that_returns_an_error_page_leaves_nothing_poisoned_behind() {
    let (hub, cache) = hub_with_model("o/m").await;
    let opts = options(&hub, cache.path());
    // The metadata request succeeds; the transfer after it answers 403 with a
    // JSON error body, as an expired signed link does.
    hub.fail_get("tokenizer.json", 403, 1);

    let first = download_model_with("o/m", &opts).await;
    assert!(first.is_err(), "an error page is not a tokenizer");

    // With the fault gone, the same call gets the real file, not the page.
    let dir = download_model_with("o/m", &opts).await.unwrap();
    assert_eq!(std::fs::read(dir.join("tokenizer.json")).unwrap(), b"{}");
}

#[tokio::test]
async fn repositories_that_are_not_language_models_still_download() {
    isolate_env();
    let hub = FakeHub::start().await;
    let cache = tempfile::tempdir().unwrap();
    // A checkpoint with no config.json at all.
    hub.add_model(
        "o/plain",
        Repo::with_files(&[
            ("model_index.json", b"{}"),
            ("flux.safetensors", &safetensors_file("a", 600)),
            ("ae.safetensors", &safetensors_file("b", 600)),
        ]),
    );
    // Weights kept in component folders.
    hub.add_model(
        "o/parts",
        Repo::with_files(&[
            ("model_index.json", b"{}"),
            (
                "unet/diffusion_pytorch_model.safetensors",
                &safetensors_file("u", 600),
            ),
            (
                "vae/diffusion_pytorch_model.safetensors",
                &safetensors_file("v", 600),
            ),
        ]),
    );
    let opts = options(&hub, cache.path());

    let plain = download_model_with("o/plain", &opts).await.unwrap();
    assert!(plain.join("flux.safetensors").exists() && plain.join("ae.safetensors").exists());
    let parts = download_model_with("o/parts", &opts).await.unwrap();
    assert!(
        parts
            .join("unet/diffusion_pytorch_model.safetensors")
            .exists()
    );
    assert!(
        parts
            .join("vae/diffusion_pytorch_model.safetensors")
            .exists()
    );
}

#[tokio::test]
async fn file_names_with_glob_characters_download_as_themselves() {
    isolate_env();
    let hub = FakeHub::start().await;
    let cache = tempfile::tempdir().unwrap();
    hub.add_model(
        "o/odd",
        Repo::with_files(&[
            ("config.json", CONFIG),
            ("model.safetensors", &safetensors_file("w", 100)),
            ("notes [draft] {v1}*.txt", b"x"),
        ]),
    );
    let dir = download_model_with("o/odd", &options(&hub, cache.path()))
        .await
        .unwrap();
    assert!(dir.join("notes [draft] {v1}*.txt").exists());
}
