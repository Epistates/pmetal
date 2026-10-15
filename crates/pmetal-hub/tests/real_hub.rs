//! Checks against the real Hugging Face Hub. Opt-in: every test is ignored
//! unless run with `--ignored`, needs network access, and downloads into the
//! directory named by `PMETAL_TEST_HUB_CACHE` (a few megabytes).

mod support;

use std::path::PathBuf;

use pmetal_hub::search::{
    SearchOptions, fetch_model_card, fetch_model_meta_with, search_models_with,
};
use pmetal_hub::{
    DeviceSpec, DownloadOptions, HubOptions, Verification, download_model_with,
    plan_model_download, verify_model_dir,
};
use support::fixtures::isolate_env;

/// A one-megaparameter model with safetensors weights, kept for tests.
const TINY: &str = "trl-internal-testing/tiny-random-LlamaForCausalLM";

fn cache() -> PathBuf {
    PathBuf::from(
        std::env::var_os("PMETAL_TEST_HUB_CACHE")
            .expect("PMETAL_TEST_HUB_CACHE names a directory to download into"),
    )
}

fn options() -> DownloadOptions {
    isolate_env();
    DownloadOptions {
        hub: HubOptions {
            cache_dir: Some(cache()),
            offline: Some(false),
            ..Default::default()
        },
        verify: Verification::Sha256,
        ..Default::default()
    }
}

#[tokio::test]
#[ignore = "network"]
async fn a_real_model_downloads_verifies_and_is_found_offline() {
    let mut opts = options();
    let plan = plan_model_download(TINY, &opts).await.unwrap();
    assert!(plan.model.files.iter().any(|f| f.path == "config.json"));
    assert!(plan.model.weight_files().count() >= 1);

    let dir = download_model_with(TINY, &opts).await.unwrap();
    let report = verify_model_dir(&dir);
    assert!(report.is_loadable_language_model(), "{report:?}");
    // The directory is exactly the commit that was planned.
    assert_eq!(dir.file_name().unwrap().to_str().unwrap(), plan.commit);

    // Offline, by name: the Hub recorded `main`.
    opts.revision = Some("main".into());
    opts.hub.offline = Some(true);
    assert_eq!(download_model_with(TINY, &opts).await.unwrap(), dir);
}

/// One quantization out of a multi-quant GGUF repository (about 300 MB).
#[tokio::test]
#[ignore = "network, downloads about 300 MB"]
async fn one_gguf_quantization_comes_out_of_a_multi_quant_repo() {
    let hub = options().hub;
    let variants = pmetal_hub::search::list_gguf_variants("bartowski/Qwen_Qwen3-0.6B-GGUF", &hub)
        .await
        .unwrap();
    assert!(variants.len() > 5, "{variants:?}");
    assert!(
        variants
            .windows(2)
            .all(|w| w[0].total_bytes() <= w[1].total_bytes())
    );
    let q2 = variants
        .iter()
        .find(|v| v.quant == "Q2_K")
        .expect("a Q2_K quantization");

    let dir =
        pmetal_hub::resolve_model_path_with("bartowski/Qwen_Qwen3-0.6B-GGUF:Q2_K", &options())
            .await
            .unwrap();

    let report = verify_model_dir(&dir);
    assert!(report.is_complete(), "{:?}", report.problems);
    assert_eq!(report.weight_files.len(), 1);
    assert_eq!(report.weight_bytes, q2.total_bytes());
    // A GGUF-only repository has no config.json; one was written from the file.
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap()).unwrap();
    assert_eq!(config["model_type"], "qwen3");
}

#[tokio::test]
#[ignore = "network"]
async fn the_real_search_api_gives_sizes_and_the_model_info_matches_the_files() {
    isolate_env();
    let hub = HubOptions {
        cache_dir: Some(cache()),
        offline: Some(false),
        ..Default::default()
    };
    let mut search = SearchOptions::new("tiny-random-LlamaForCausalLM");
    search.limit = 20;
    search.author = Some("trl-internal-testing".into());
    search.hub = hub.clone();
    let results = search_models_with(&search).await.unwrap();
    let ours = results
        .iter()
        .find(|r| r.model_id == TINY)
        .expect("the tiny model is listed");

    // The size the Hub reports for the tensors is what the files hold.
    let bytes = ours
        .safetensors_bytes
        .expect("the Hub reports tensor sizes");
    let meta = fetch_model_meta_with(TINY, &hub).await.unwrap();
    assert_eq!(meta.safetensors_bytes, Some(bytes));
    assert_eq!(meta.model_type.as_deref(), Some("llama"));

    let dir = download_model_with(TINY, &options()).await.unwrap();
    let local = verify_model_dir(&dir).safetensors.unwrap();
    assert_eq!(
        local.data_bytes(),
        bytes,
        "header sizes agree with the Hub's count"
    );

    // And a fit can be made from it.
    let device = DeviceSpec::from_device(16_000_000_000, 120.0, true);
    let fit = ours.fit(&device, 4096).unwrap();
    assert!(fit.weights_measured);
    assert!((fit.weights_gb * 1e9 - bytes as f64).abs() < 1.0);

    let card = fetch_model_card(TINY, &hub).await;
    assert!(card.is_ok(), "{card:?}");
}
