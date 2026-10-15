//! Search, model metadata, GGUF variant listing and model cards against a fake
//! Hub on loopback.

mod support;

use pmetal_core::SecretString;
use pmetal_hub::{
    DeviceSpec, HubOptions, SearchOptions, fetch_model_card, fetch_model_meta_with,
    list_gguf_variants, search_models_with,
};
use serde_json::json;
use support::fixtures::{gguf_file, isolate_env};
use support::{FakeHub, Repo};

fn hub_options(hub: &FakeHub) -> HubOptions {
    HubOptions {
        endpoint: Some(hub.base.clone()),
        cache_dir: Some(tempfile::tempdir().unwrap().keep()),
        offline: Some(false),
        max_retries: Some(1),
        ..Default::default()
    }
}

fn repo_with_info(files: &[(&str, &[u8])], info: serde_json::Value) -> Repo {
    let mut repo = Repo::with_files(files);
    repo.info_extra = info;
    repo
}

async fn hub_with_catalog() -> FakeHub {
    isolate_env();
    let hub = FakeHub::start().await;
    hub.add_model(
        "Qwen/Qwen3-0.6B",
        repo_with_info(
            &[],
            json!({
                "downloads": 900, "likes": 12, "pipeline_tag": "text-generation",
                "tags": ["safetensors", "text-generation", "license:apache-2.0"],
                "safetensors": {"parameters": {"BF16": 751632384u64}, "total": 751632384u64},
            }),
        ),
    );
    hub.add_model(
        "Qwen/Qwen3-30B-A3B-GGUF",
        repo_with_info(
            &[],
            json!({
                "downloads": 50, "pipeline_tag": "text-generation",
                "tags": ["gguf", "text-generation", "moe"],
                "gguf": {"total": 30532122624u64, "architecture": "qwen3moe", "context_length": 262144},
            }),
        ),
    );
    hub.add_model(
        "other/embedder",
        repo_with_info(
            &[],
            json!({"pipeline_tag": "feature-extraction", "tags": ["safetensors"]}),
        ),
    );
    hub
}

#[tokio::test]
async fn search_honors_the_endpoint_filters_and_asks_for_the_fields_it_needs() {
    let hub = hub_with_catalog().await;
    let mut options = SearchOptions::new("qwen");
    options.hub = hub_options(&hub);
    options.hub.token = Some(SecretString::new("hf_search_token_0123456789"));

    let results = search_models_with(&options).await.unwrap();

    let ids: Vec<_> = results.iter().map(|r| r.model_id.as_str()).collect();
    assert_eq!(ids, ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-30B-A3B-GGUF"]);
    assert_eq!(results[0].safetensors_bytes, Some(1_503_264_768));

    let request = &hub.requests_to("/api/models")[0];
    assert_eq!(
        request.authorization.as_deref(),
        Some("Bearer hf_search_token_0123456789")
    );
    assert!(
        request.query.contains("expand%5B%5D=safetensors")
            || request.query.contains("expand[]=safetensors")
    );
    assert!(request.query.contains("pipeline_tag=text-generation"));

    // Tag filters narrow the results on the Hub.
    options.tags = vec!["gguf".into()];
    let ggufs = search_models_with(&options).await.unwrap();
    assert_eq!(ggufs.len(), 1);
    assert_eq!(ggufs[0].gguf_architecture.as_deref(), Some("qwen3moe"));
}

#[tokio::test]
async fn search_results_are_sized_for_the_device() {
    let hub = hub_with_catalog().await;
    let mut options = SearchOptions::new("qwen");
    options.hub = hub_options(&hub);
    let results = search_models_with(&options).await.unwrap();

    let small = DeviceSpec::from_device(8_000_000_000, 100.0, true);
    let dense = &results[0];
    let fit = dense.fit(&small, 4096).unwrap();
    assert!(fit.weights_measured);
    assert_eq!(fit.fit_level.label(), "Fits");

    // The GGUF repo has parameters but no weight size; its name says A3B.
    let moe = &results[1];
    let spec = moe.model_spec(4096);
    assert!(spec.is_moe);
    assert_eq!(spec.active_params_b, Some(3.0));
    assert_eq!(spec.max_context_length, Some(262_144));
    assert_eq!(
        moe.fit(&small, 4096).unwrap().fit_level.label(),
        "Too Large"
    );
}

#[tokio::test]
async fn search_rate_limits_are_explained_and_retries_are_bounded() {
    let hub = hub_with_catalog().await;
    hub.fail("/api/models", 429, 1000);
    let mut options = SearchOptions::new("qwen");
    options.hub = hub_options(&hub);
    options.hub.max_retries = Some(1);

    let err = search_models_with(&options).await.unwrap_err().to_string();

    assert!(err.contains("rate limiting"), "{err}");
    assert_eq!(
        hub.requests_to("/api/models").len(),
        2,
        "one try and one retry"
    );
}

#[tokio::test]
async fn search_recovers_from_a_transient_failure() {
    let hub = hub_with_catalog().await;
    hub.fail("/api/models", 503, 1);
    let mut options = SearchOptions::new("qwen");
    options.hub = hub_options(&hub);
    assert_eq!(search_models_with(&options).await.unwrap().len(), 2);
}

#[tokio::test]
async fn search_refuses_to_read_an_oversized_response() {
    isolate_env();
    let hub = FakeHub::start().await;
    hub.add_model(
        "huge/model",
        repo_with_info(
            &[],
            json!({"pipeline_tag": "text-generation", "junk": "x".repeat(17 * 1024 * 1024)}),
        ),
    );
    let mut options = SearchOptions::new("huge");
    options.hub = hub_options(&hub);
    let err = search_models_with(&options).await.unwrap_err().to_string();
    assert!(err.contains("larger than"), "{err}");
}

#[tokio::test]
async fn search_in_offline_mode_does_not_touch_the_network() {
    let hub = hub_with_catalog().await;
    let mut options = SearchOptions::new("qwen");
    options.hub = hub_options(&hub);
    options.hub.offline = Some(true);
    let err = search_models_with(&options).await.unwrap_err().to_string();
    assert!(err.contains("offline"), "{err}");
    assert!(hub.log().is_empty());
}

#[tokio::test]
async fn model_meta_reads_the_config_at_the_commit_the_info_describes() {
    isolate_env();
    let hub = FakeHub::start().await;
    let config = json!({
        "model_type": "qwen3", "hidden_size": 1024, "num_hidden_layers": 28,
        "intermediate_size": 3072, "num_attention_heads": 16, "num_key_value_heads": 8,
        "head_dim": 128, "vocab_size": 151936, "max_position_embeddings": 40960,
        "tie_word_embeddings": true,
    })
    .to_string();
    hub.add_model(
        "Qwen/Qwen3-0.6B",
        repo_with_info(
            &[("config.json", config.as_bytes())],
            json!({
                "tags": ["license:apache-2.0"], "library_name": "transformers", "gated": false,
                "safetensors": {"parameters": {"BF16": 751632384u64}, "total": 751632384u64},
            }),
        ),
    );

    let meta = fetch_model_meta_with("Qwen/Qwen3-0.6B", &hub_options(&hub))
        .await
        .unwrap();

    assert_eq!(meta.model_type.as_deref(), Some("qwen3"));
    assert_eq!(meta.num_hidden_layers, Some(28));
    assert_eq!(meta.safetensors_bytes, Some(1_503_264_768));
    assert_eq!(meta.license.as_deref(), Some("apache-2.0"));
    // Counted from the architecture (tied embeddings once): 0.6B.
    assert!(
        (meta.estimated_params_b - 0.6).abs() < 0.05,
        "{}",
        meta.estimated_params_b
    );
    // The exact KV layout: 28 layers * 8 kv heads * 128 * 2 (k,v) * 2 B * 4096.
    let spec = meta.model_spec(4096);
    let fit =
        pmetal_hub::estimate_fit(&spec, &DeviceSpec::from_device(16_000_000_000, 120.0, true));
    assert!((fit.kv_cache_gb - 28.0 * 8.0 * 128.0 * 2.0 * 2.0 * 4096.0 / 1e9).abs() < 1e-9);
    assert!(fit.weights_measured);
}

#[tokio::test]
async fn model_meta_for_a_gguf_only_repo_has_no_config_and_no_error() {
    isolate_env();
    let hub = FakeHub::start().await;
    hub.add_model(
        "o/m-GGUF",
        repo_with_info(
            &[("m-Q4_K_M.gguf", &gguf_file(None, &[("a", 1.0)]))],
            json!({"gguf": {"total": 8_000_000_000u64, "architecture": "llama", "context_length": 8192}}),
        ),
    );
    let meta = fetch_model_meta_with("o/m-GGUF", &hub_options(&hub))
        .await
        .unwrap();
    assert!(meta.model_type.is_none());
    assert!((meta.estimated_params_b - 8.0).abs() < 1e-9);
    assert_eq!(meta.quantization, "Q4_K_M");
}

#[tokio::test]
async fn gguf_variants_list_each_quantization_with_its_size() {
    isolate_env();
    let hub = FakeHub::start().await;
    let q4 = gguf_file(None, &[("a", 1.0)]);
    let q8 = gguf_file(None, &[("a", 1.0), ("b", 2.0), ("c", 3.0)]);
    hub.add_model(
        "o/m-GGUF",
        Repo::with_files(&[
            ("m-Q8_0.gguf", &q8),
            ("m-Q4_K_M.gguf", &q4),
            ("README.md", b"x"),
        ]),
    );
    let variants = list_gguf_variants("o/m-GGUF", &hub_options(&hub))
        .await
        .unwrap();
    let summary: Vec<_> = variants
        .iter()
        .map(|v| (v.quant.as_str(), v.total_bytes()))
        .collect();
    assert_eq!(
        summary,
        [("Q4_K_M", q4.len() as u64), ("Q8_0", q8.len() as u64)]
    );
}

#[tokio::test]
async fn a_model_card_is_fetched_and_parsed() {
    isolate_env();
    let hub = FakeHub::start().await;
    let readme = b"---\nlicense: mit\nbase_model: org/base\n---\n# Card\n";
    hub.add_model("o/m", Repo::with_files(&[("README.md", readme)]));
    let card = fetch_model_card("o/m", &hub_options(&hub)).await.unwrap();
    assert_eq!(card.license.as_deref(), Some("mit"));
    assert_eq!(card.base_models, ["org/base"]);
    assert_eq!(card.body, "# Card\n");
}
