//! Real-weight parity of a released sentence-transformers model.
//!
//! `all_minilm_l6_v2_reference.safetensors` holds `sentence-transformers/
//! all-MiniLM-L6-v2` run in float32 over three padded sentences: the encoder's
//! states and the model's embeddings (mean pooling over the attention mask,
//! then L2 normalisation, as its `modules.json` pipeline says). This reads that
//! pipeline from the directory, loads the checkpoint through `DynamicModel`,
//! and reproduces the embeddings.
//!
//! Needs the model (~90 MB): set `PMETAL_MINILM_DIR` to an
//! `all-MiniLM-L6-v2` snapshot. Skipped, loudly, without it.

mod common;

use common::{fixture_path, load_shard, ref_tensor};
use pmetal_cache::test_utils::{ParityReport, Tolerance, print_report_table};
use pmetal_models::pooling::{PoolingMode, SentenceTransformerConfig, normalize_embeddings, pool};
use pmetal_models::{DynamicModel, ModelArchitecture};

fn check(tolerance: Tolerance, min_cosine: f32) {
    let Ok(dir) = std::env::var("PMETAL_MINILM_DIR") else {
        eprintln!(
            "SKIPPED: set PMETAL_MINILM_DIR to a sentence-transformers/all-MiniLM-L6-v2 snapshot"
        );
        return;
    };
    let dir = std::path::Path::new(&dir);

    let pipeline = SentenceTransformerConfig::from_dir(dir)
        .expect("modules.json reads")
        .expect("the directory is a sentence-transformers model");
    assert_eq!(pipeline.pooling, PoolingMode::Mean);
    assert!(pipeline.normalize);
    assert_eq!(pipeline.max_seq_length, Some(256));
    assert_eq!(pipeline.transformer_path, "");

    assert_eq!(
        ModelArchitecture::detect(dir).expect("detects"),
        ModelArchitecture::Bert
    );
    let mut model = DynamicModel::load(dir).expect("the released encoder loads strictly");
    let reference = load_shard(&fixture_path("all_minilm_l6_v2_reference.safetensors"));
    let ids = ref_tensor(&reference, "input_ids");
    let mask = ref_tensor(&reference, "attention_mask");

    let hidden = model.forward_hidden(ids, Some(mask)).expect("forward");
    pmetal_runtime::check_unobserved_error().expect("no runtime op failed");
    let pooled = pool(&hidden, mask, pipeline.pooling).expect("pool");
    let embeddings = normalize_embeddings(&pooled).expect("normalize");

    let reports = [
        ParityReport::compute(
            "last_hidden_state",
            &hidden,
            ref_tensor(&reference, "last_hidden_state"),
            tolerance,
        ),
        ParityReport::compute(
            "embeddings",
            &embeddings,
            ref_tensor(&reference, "embeddings"),
            tolerance,
        ),
    ];
    print_report_table(&reports);
    for report in &reports {
        assert!(report.passed(), "{}: {report:?}", report.name);
        assert!(report.cosine_similarity > min_cosine, "{}", report.name);
    }
}

#[test]
fn the_released_minilm_reproduces_its_sentence_embeddings() {
    check(Tolerance::new(1e-4, 1e-5), 0.99999);
}
