//! Numerical parity of the BERT-family encoder against `transformers`.
//!
//! Three tiny seeded encoders: `BertModel` (segment ids), `RobertaModel`
//! (positions counted from the padding index, one token type) and
//! `DistilBertModel` (a differently named tree, no token types). Each is run
//! on a batch of three rows: full, right-padded, and left-padded, so that the
//! padding mask and RoBERTa's pad-relative positions both matter. Every
//! position is compared, padding included: padded rows still attend to their
//! real keys, so their states are as well defined as any.

mod common;

use std::collections::HashMap;

use common::{fixture_path, load_shard, ref_tensor};
use pmetal_cache::test_utils::{ParityReport, Tolerance, max_abs_diff, print_report_table};
use pmetal_models::architectures::bert::{BertConfig, BertModel};
use pmetal_models::architectures::encoder::EncoderModel;
use pmetal_models::loader::{LoadError, bert_checkpoint_tensors, load_bert_weights};
use pmetal_runtime::compat::{Array, Dtype};

struct Family {
    name: &'static str,
    config: fn() -> BertConfig,
    prefix: &'static str,
}

fn bert() -> BertConfig {
    BertConfig {
        vocab_size: 120,
        hidden_size: 32,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        intermediate_size: 64,
        max_position_embeddings: 24,
        type_vocab_size: 2,
        layer_norm_eps: 1e-12,
        model_type: Some("bert".into()),
        ..Default::default()
    }
}

fn roberta() -> BertConfig {
    BertConfig {
        max_position_embeddings: 26,
        type_vocab_size: 1,
        layer_norm_eps: 1e-5,
        pad_token_id: Some(1),
        model_type: Some("roberta".into()),
        ..bert()
    }
}

fn distilbert() -> BertConfig {
    BertConfig {
        type_vocab_size: 2, // ignored: DistilBERT has no token types
        model_type: Some("distilbert".into()),
        ..bert()
    }
}

const FAMILIES: [Family; 3] = [
    Family {
        name: "bert",
        config: bert,
        prefix: "bert.",
    },
    Family {
        name: "roberta",
        config: roberta,
        prefix: "roberta.",
    },
    Family {
        name: "distilbert",
        config: distilbert,
        prefix: "distilbert.",
    },
];

fn weights(family: &Family) -> HashMap<String, Array> {
    load_shard(&fixture_path(&format!(
        "bert_{}_weights.safetensors",
        family.name
    )))
}

fn reference(family: &Family) -> HashMap<String, Array> {
    load_shard(&fixture_path(&format!(
        "bert_{}_reference.safetensors",
        family.name
    )))
}

fn model(family: &Family) -> BertModel {
    let mut model = BertModel::new((family.config)()).expect("model builds");
    load_bert_weights(&mut model, weights(family)).expect("weights load");
    model
}

fn run(model: &mut BertModel, reference: &HashMap<String, Array>, masked: bool) -> Array {
    model
        .forward_with_token_types(
            ref_tensor(reference, "input_ids"),
            masked.then(|| ref_tensor(reference, "attention_mask")),
            reference.get("token_type_ids"),
        )
        .expect("forward")
}

fn compare(name: &str, got: &Array, want: &Array) -> ParityReport {
    let report =
        ParityReport::compute_with_per_position(name, got, want, Tolerance::new(3e-5, 3e-6));
    print_report_table(std::slice::from_ref(&report));
    report
}

#[test]
fn each_family_matches_transformers() {
    for family in &FAMILIES {
        let reference = reference(family);
        let mut model = model(family);

        let embeddings = model
            .embeddings
            .forward(
                ref_tensor(&reference, "input_ids"),
                reference.get("token_type_ids"),
            )
            .expect("embeddings");
        let masked = run(&mut model, &reference, true);
        let unmasked = run(&mut model, &reference, false);
        pmetal_runtime::check_unobserved_error().expect("no runtime op failed");

        for (name, got, want) in [
            ("embeddings", &embeddings, "embeddings"),
            ("last_hidden_state", &masked, "last_hidden_state"),
            ("unmasked", &unmasked, "last_hidden_state_unmasked"),
        ] {
            let report = compare(name, got, ref_tensor(&reference, want));
            assert!(report.passed(), "{} {name}: {report:?}", family.name);
        }
    }
}

/// The comparison can only fail if the padding mask, the segments and the
/// position rule matter in the fixtures.
#[test]
fn the_fixtures_are_sensitive_to_mask_positions_and_segments() {
    for family in &FAMILIES {
        let reference = reference(family);
        let gap = max_abs_diff(
            ref_tensor(&reference, "last_hidden_state"),
            ref_tensor(&reference, "last_hidden_state_unmasked"),
        );
        assert!(
            gap > 1e-2,
            "{}: the mask moves nothing ({gap:.2e})",
            family.name
        );
    }

    // RoBERTa's positions are counted over the non-padding tokens. Counted
    // BERT's way (0, 1, 2, ... whatever the padding) the left-padded row's
    // states are different.
    let family = &FAMILIES[1];
    let reference = reference(family);
    let mut as_bert = BertModel::new(BertConfig {
        model_type: Some("bert".into()),
        ..(family.config)()
    })
    .expect("builds");
    load_bert_weights(&mut as_bert, weights(family)).expect("loads");
    let wrong = run(&mut as_bert, &reference, true);
    let gap = max_abs_diff(&wrong, ref_tensor(&reference, "last_hidden_state"));
    assert!(
        gap > 1e-2,
        "RoBERTa positions are not exercised ({gap:.2e})"
    );
}

/// A checkpoint saved from a task model puts the encoder under the family
/// prefix and carries the pooler and heads, which the encoder does not use.
#[test]
fn a_task_model_checkpoint_loads_its_encoder() {
    for family in &FAMILIES {
        let mut task: HashMap<String, Array> = weights(family)
            .into_iter()
            .map(|(k, v)| (format!("{}{k}", family.prefix), v))
            .collect();
        for head in [
            "cls.predictions.bias",
            "cls.predictions.transform.dense.weight",
            "lm_head.decoder.weight",
            "classifier.weight",
            "vocab_projector.bias",
            "pre_classifier.weight",
        ] {
            task.insert(head.into(), Array::from_slice(&[0.5f32], &[1]));
        }
        let mut model = BertModel::new((family.config)()).expect("builds");
        load_bert_weights(&mut model, task).unwrap_or_else(|e| panic!("{}: {e}", family.name));
    }
}

/// What a fine-tuned encoder saves is what `transformers` reads: the same names
/// (bar the pooler this model never had) and the same values, and the model
/// loads it back.
#[test]
fn the_exported_checkpoint_has_the_released_names_and_loads_back() {
    for family in &FAMILIES {
        let released = weights(family);
        let model = model(family);
        let exported: HashMap<String, Array> = bert_checkpoint_tensors(&model)
            .expect("export")
            .into_iter()
            .collect();
        let mut released_names: Vec<&String> = released
            .keys()
            .filter(|k| !k.starts_with("pooler."))
            .collect();
        let mut exported_names: Vec<&String> = exported.keys().collect();
        released_names.sort();
        exported_names.sort();
        assert_eq!(released_names, exported_names, "{}", family.name);
        for name in released_names {
            assert_eq!(
                max_abs_diff(&released[name], &exported[name]),
                0.0,
                "{name}"
            );
        }

        let mut again = BertModel::new((family.config)()).expect("builds");
        load_bert_weights(&mut again, exported).expect("the export loads");
    }
}

#[test]
fn a_bf16_encoder_stays_in_bf16() {
    for family in &FAMILIES {
        let reference = reference(family);
        let mut model = BertModel::new((family.config)()).expect("builds");
        let half = weights(family)
            .into_iter()
            .map(|(k, v)| (k, v.as_dtype(Dtype::Bfloat16.as_i32())))
            .collect();
        load_bert_weights(&mut model, half).expect("loads");
        let out = run(&mut model, &reference, true);
        assert_eq!(out.dtype(), Dtype::Bfloat16, "{}", family.name);
        pmetal_runtime::check_unobserved_error().expect("no runtime op failed");
        let gap = max_abs_diff(&out, ref_tensor(&reference, "last_hidden_state"));
        assert!(gap < 0.3, "{}: bf16 is {gap:.2e} from float32", family.name);
    }
}

#[test]
fn loading_is_strict() {
    let family = &FAMILIES[0];
    let mut extra = weights(family);
    extra.insert(
        "encoder.layer.0.attention.self.mystery.weight".into(),
        Array::from_slice(&[1.0f32], &[1]),
    );
    let mut m = BertModel::new((family.config)()).unwrap();
    let err = load_bert_weights(&mut m, extra).expect_err("a stray tensor");
    assert!(err.to_string().contains("mystery"), "{err}");

    let mut short = weights(family);
    short.remove("encoder.layer.1.output.dense.bias");
    let mut m = BertModel::new((family.config)()).unwrap();
    let err = load_bert_weights(&mut m, short).expect_err("a missing tensor");
    assert!(
        err.to_string().contains("layers.1.output.dense.bias"),
        "{err}"
    );

    let mut wrong = weights(family);
    wrong.insert(
        "embeddings.word_embeddings.weight".into(),
        Array::from_slice(&vec![0.0f32; 32 * 10], &[10, 32]),
    );
    let mut m = BertModel::new((family.config)()).unwrap();
    let err = load_bert_weights(&mut m, wrong).expect_err("a wrong shape");
    assert!(matches!(err, LoadError::ShapeMismatch { .. }), "{err}");

    // A BERT checkpoint into a DistilBERT model (no token-type table, other
    // names) is refused rather than half-loaded.
    let mut m = BertModel::new(distilbert()).unwrap();
    assert!(load_bert_weights(&mut m, weights(family)).is_err());
}

#[test]
fn sequences_beyond_the_position_table_are_an_error() {
    let mut bert_model = model(&FAMILIES[0]);
    let ids = Array::from_slice(&[5i32; 25], &[1, 25]);
    let err = bert_model.forward(&ids, None).expect_err("25 > 24");
    assert!(err.to_string().contains("25"), "{err}");

    // RoBERTa's table is two rows longer than its longest sequence.
    let mut roberta_model = model(&FAMILIES[1]);
    assert_eq!(roberta_model.config.max_sequence_length(), 24);
    let ids = Array::from_slice(&[5i32; 25], &[1, 25]);
    assert!(roberta_model.forward(&ids, None).is_err());
    let ids = Array::from_slice(&[5i32; 24], &[1, 24]);
    roberta_model.forward(&ids, None).expect("24 fits");
    pmetal_runtime::check_unobserved_error().expect("no runtime op failed");
}

/// A row embeds the same alone and padded into a batch once the padding is
/// hidden; without the mask it attends to the padding and changes. This is what
/// `/v1/embeddings` relies on, through `DynamicModel::forward_hidden`.
#[test]
fn a_rows_states_do_not_depend_on_the_padding_around_it() {
    use pmetal_models::DynamicModel;

    for family in &FAMILIES {
        let pad = (family.config)().padding_idx() as i32;
        let mut encoder = DynamicModel::Bert(EncoderModel::Bert(model(family)));
        assert!(encoder.takes_padding_mask());

        let row = [7i32, 12, 33, 8, 90];
        let alone = encoder
            .forward_hidden(&Array::from_slice(&row, &[1, 5]), None)
            .expect("alone");
        let mut padded = row.to_vec();
        padded.extend([pad; 4]);
        let mut mask = vec![1i32; 5];
        mask.extend([0; 4]);
        let ids = Array::from_slice(&padded, &[1, 9]);
        let mask = Array::from_slice(&mask, &[1, 9]);
        let masked = encoder.forward_hidden(&ids, Some(&mask)).expect("masked");
        let unmasked = encoder.forward_hidden(&ids, None).expect("unmasked");
        pmetal_runtime::check_unobserved_error().expect("no runtime op failed");

        let real = |hidden: &Array| pmetal_runtime::compat::ops::slice_axis(hidden, 1, 0, 5);
        assert!(
            max_abs_diff(&real(&masked), &alone) < 1e-5,
            "{}: padding leaks through the mask",
            family.name
        );
        assert!(
            max_abs_diff(&real(&unmasked), &alone) > 1e-3,
            "{}: the padding does not matter, so the test proves nothing",
            family.name
        );
    }
}

#[test]
fn a_mask_that_is_not_batch_by_sequence_is_an_error() {
    let mut m = model(&FAMILIES[0]);
    let ids = Array::from_slice(&[5i32; 8], &[2, 4]);
    let causal_style = Array::from_slice(&[0.0f32; 16], &[1, 1, 4, 4]);
    assert!(m.forward(&ids, Some(&causal_style)).is_err());
}

#[test]
fn released_configs_parse() {
    // bert-base-uncased, roberta-base and distilbert-base-uncased.
    let bert: BertConfig = serde_json::from_str(
        r#"{"architectures": ["BertForMaskedLM"], "hidden_act": "gelu", "hidden_size": 768,
            "intermediate_size": 3072, "layer_norm_eps": 1e-12, "max_position_embeddings": 512,
            "model_type": "bert", "num_attention_heads": 12, "num_hidden_layers": 12,
            "pad_token_id": 0, "position_embedding_type": "absolute", "type_vocab_size": 2,
            "vocab_size": 30522}"#,
    )
    .unwrap();
    assert_eq!(bert.token_types(), 2);
    assert_eq!(bert.max_sequence_length(), 512);

    let roberta: BertConfig = serde_json::from_str(
        r#"{"architectures": ["RobertaForMaskedLM"], "hidden_act": "gelu", "hidden_size": 768,
            "intermediate_size": 3072, "layer_norm_eps": 1e-05, "max_position_embeddings": 514,
            "model_type": "roberta", "num_attention_heads": 12, "num_hidden_layers": 12,
            "pad_token_id": 1, "type_vocab_size": 1, "vocab_size": 50265}"#,
    )
    .unwrap();
    assert_eq!(roberta.padding_idx(), 1);
    assert_eq!(roberta.max_sequence_length(), 512);
    assert_eq!(roberta.norm_eps(), 1e-5);

    let distil: BertConfig = serde_json::from_str(
        r#"{"activation": "gelu", "architectures": ["DistilBertForMaskedLM"], "dim": 768,
            "hidden_dim": 3072, "max_position_embeddings": 512, "model_type": "distilbert",
            "n_heads": 12, "n_layers": 6, "pad_token_id": 0, "sinusoidal_pos_embds": false,
            "vocab_size": 30522}"#,
    )
    .unwrap();
    assert_eq!((distil.hidden_size, distil.num_hidden_layers), (768, 6));
    assert_eq!(distil.token_types(), 0);
    assert_eq!(distil.norm_eps(), 1e-12);
    assert_eq!(distil.hidden_act, "gelu");
}

#[test]
fn unimplemented_variants_are_refused_by_name() {
    for (patch, needle) in [
        (
            r#""position_embedding_type": "relative_key""#,
            "relative_key",
        ),
        (r#""sinusoidal_pos_embds": true"#, "sinusoidal"),
        (r#""model_type": "albert""#, "albert"),
    ] {
        let raw = format!(
            r#"{{"vocab_size": 10, "hidden_size": 8, "num_hidden_layers": 1,
                 "num_attention_heads": 2, "intermediate_size": 16, {patch}}}"#
        );
        let err = BertModel::from_config_str(&raw).expect_err(patch);
        assert!(err.to_string().contains(needle), "{err}");
    }
}
