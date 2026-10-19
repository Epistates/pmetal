//! The argv each job spec spells, parsed by the real argument parser.

use super::{Cli, Commands};
use clap::Parser;
use pmetal_core::JobFields;
#[cfg(feature = "serve")]
use pmetal_core::jobs::ServeSpec;
use pmetal_core::jobs::{
    BenchSpec, DecideSpec, DflashSpec, DistillSpec, EmbedTrainSpec, EvalSpec, FuseSpec, GrpoSpec,
    InferSpec, MergeSpec, PackExpertsSpec, PretrainSpec, QuantizeSpec, RlkdSpec, TokenizeSpec,
    TrainSpec,
};

/// Parse an argv slice built from a spec's `to_argv()` output.  Returns
/// `Err(String)` on failure so the test body can provide a descriptive message.
fn try_parse(subcommand: &str, spec_argv: Vec<String>) -> Result<Commands, String> {
    let mut argv = vec!["pmetal".to_string(), subcommand.to_string()];
    argv.extend(spec_argv);
    Cli::try_parse_from(&argv)
        .map(|cli| cli.command)
        .map_err(|e| e.to_string())
}

#[test]
fn train_spec_round_trip() {
    let spec = TrainSpec {
        model: "Qwen/Qwen3-0.6B".into(),
        dataset: "data/train.jsonl".into(),
        ..Default::default()
    };
    let result = try_parse("train", spec.to_argv());
    assert!(
        result.is_ok(),
        "TrainSpec argv failed to parse: {}",
        result.unwrap_err()
    );
}

#[cfg(feature = "trainer")]
#[test]
fn train_mtp_draft_and_diffusion_specs_round_trip() {
    use pmetal_core::jobs::{TrainDiffusionSpec, TrainDraftSpec, TrainMtpSpec};
    for family in ["auto", "qwen3-next", "gemma4"] {
        let spec = TrainMtpSpec {
            model: "m".into(),
            family: family.into(),
            shards: vec!["a.bin".into(), "b.bin".into()],
            lr_schedule: "wsd".into(),
            init_mtp: Some("mtp".into()),
            ..Default::default()
        };
        let parsed = try_parse("train-mtp", spec.to_argv())
            .unwrap_or_else(|e| panic!("family {family}: {e}"));
        let Commands::TrainMtp(args) = parsed else {
            panic!("not train-mtp")
        };
        assert_eq!(args.shards, ["a.bin", "b.bin"]);
        assert_eq!(args.lr_schedule, pmetal_core::LrSchedulerType::Wsd);
    }
    let spec = TrainDraftSpec {
        target: "t".into(),
        draft_config: Some("c.json".into()),
        shards: vec!["a.bin".into()],
        ..Default::default()
    };
    let Commands::TrainDraft(args) = try_parse("train-draft", spec.to_argv()).unwrap() else {
        panic!("not train-draft")
    };
    assert_eq!(args.draft_config.as_deref(), Some("c.json"));
    for quantization in ["none", "nf4", "fp4", "int8"] {
        let spec = TrainDiffusionSpec {
            model: "m".into(),
            dataset: "d".into(),
            quantization: quantization.into(),
            double_quant: quantization == "nf4",
            ..Default::default()
        };
        let parsed = try_parse("train-diffusion", spec.to_argv())
            .unwrap_or_else(|e| panic!("quantization {quantization}: {e}"));
        let Commands::TrainDiffusion(args) = parsed else {
            panic!("not train-diffusion")
        };
        assert!(args.lora_targets.is_empty());
    }
}

#[cfg(feature = "trainer")]
#[test]
fn train_diffusion_corrupted_only_can_be_turned_off() {
    // `--corrupted-only` defaults to true and used to be a set-true flag,
    // so the documented "when false, average over the whole canvas" mode
    // could not be selected.
    let parse = |extra: &[&str]| {
        let mut argv: Vec<String> = ["--model", "m", "--dataset", "d"]
            .map(String::from)
            .to_vec();
        argv.extend(extra.iter().map(|s| s.to_string()));
        match try_parse("train-diffusion", argv) {
            Ok(Commands::TrainDiffusion(args)) => args.corrupted_only,
            Ok(_) => panic!("not train-diffusion"),
            Err(e) => panic!("{e}"),
        }
    };
    assert!(parse(&[]));
    assert!(parse(&["--corrupted-only", "true"]));
    assert!(!parse(&["--corrupted-only", "false"]));
}

#[cfg(feature = "trainer")]
#[test]
fn a_schedule_the_trainer_supports_reaches_it_and_a_typo_is_refused() {
    // `train-mtp` and `train-draft` used to read any name but constant
    // and linear as cosine, so `wsd` trained on the wrong curve.
    for (subcommand, model_flag) in [("train-mtp", "--model"), ("train-draft", "--target")] {
        let parse = |schedule: &str| {
            try_parse(
                subcommand,
                [model_flag, "t", "--shards", "s", "--lr-schedule", schedule]
                    .map(String::from)
                    .to_vec(),
            )
        };
        let parsed = parse("wsd").unwrap_or_else(|e| panic!("{subcommand}: {e}"));
        let got = match parsed {
            Commands::TrainMtp(a) => a.lr_schedule,
            Commands::TrainDraft(a) => a.lr_schedule,
            _ => panic!("{subcommand} parsed as another command"),
        };
        assert_eq!(got, pmetal_core::LrSchedulerType::Wsd);
        let err = parse("cosin").unwrap_err();
        assert!(err.contains("cosin"), "{subcommand}: {err}");
    }
}

#[test]
fn distill_spec_round_trip() {
    let spec = DistillSpec {
        teacher: "Qwen/Qwen3-7B".into(),
        student: "Qwen/Qwen3-0.6B".into(),
        dataset: "data.jsonl".into(),
        ..Default::default()
    };
    let result = try_parse("distill", spec.to_argv());
    assert!(
        result.is_ok(),
        "DistillSpec argv failed to parse: {}",
        result.unwrap_err()
    );
}

#[test]
fn grpo_spec_round_trip() {
    let spec = GrpoSpec {
        model: "model".into(),
        dataset: "data.jsonl".into(),
        ..Default::default()
    };
    let result = try_parse("grpo", spec.to_argv());
    assert!(
        result.is_ok(),
        "GrpoSpec argv failed to parse: {}",
        result.unwrap_err()
    );
}

/// Every field survives spec → argv → clap → spec, defaults included,
/// so the spec's `default_*` attributes and the CLI's defaults agree.
#[test]
fn preference_spec_round_trip() {
    use pmetal_core::jobs::PreferenceSpec;
    let defaults = PreferenceSpec {
        model: "model".into(),
        dataset: "data.jsonl".into(),
        ..Default::default()
    };
    let custom = PreferenceSpec {
        loss: "kto".into(),
        beta: Some(0.3),
        max_steps: Some(12),
        undesirable_weight: 1.5,
        log_metrics: Some("m.jsonl".into()),
        ..defaults.clone()
    };
    for spec in [defaults, custom] {
        let parsed = match try_parse("preference", spec.to_argv()) {
            Ok(Commands::Preference(args)) => PreferenceSpec::from(args),
            Ok(_) => panic!("parsed as another command"),
            Err(e) => panic!("PreferenceSpec argv failed to parse: {e}"),
        };
        assert_eq!(
            serde_json::to_value(&parsed).unwrap(),
            serde_json::to_value(&spec).unwrap()
        );
    }
    assert!(matches!(
        try_parse(
            "dpo",
            vec!["-m".into(), "x".into(), "-d".into(), "y".into()]
        ),
        Ok(Commands::Preference(_))
    ));
}

#[test]
fn rlkd_spec_round_trip() {
    let spec = RlkdSpec {
        model: "model".into(),
        teacher_model: "teacher".into(),
        dataset: "data.jsonl".into(),
        ..Default::default()
    };
    let result = try_parse("rlkd", spec.to_argv());
    assert!(
        result.is_ok(),
        "RlkdSpec argv failed to parse: {}",
        result.unwrap_err()
    );
}

#[test]
fn embed_train_spec_round_trip() {
    let spec = EmbedTrainSpec {
        model: "model".into(),
        dataset: "data.jsonl".into(),
        ..Default::default()
    };
    let result = try_parse("embed-train", spec.to_argv());
    assert!(
        result.is_ok(),
        "EmbedTrainSpec argv failed to parse: {}",
        result.unwrap_err()
    );
}

#[test]
fn pretrain_spec_round_trip() {
    let spec = PretrainSpec {
        arch: "gpt-oss".into(),
        ..Default::default()
    };
    let result = try_parse("pretrain", spec.to_argv());
    assert!(
        result.is_ok(),
        "PretrainSpec argv failed to parse: {}",
        result.unwrap_err()
    );
}

#[test]
fn infer_spec_round_trip() {
    let spec = InferSpec {
        model: "model".into(),
        prompt: "Hello".into(),
        ..Default::default()
    };
    let result = try_parse("infer", spec.to_argv());
    assert!(
        result.is_ok(),
        "InferSpec argv failed to parse: {}",
        result.unwrap_err()
    );
}

#[test]
fn infer_video_flags_parse() {
    let parse = |extra: &[&str]| {
        let mut argv = vec![
            "--model".to_string(),
            "m".into(),
            "--prompt".into(),
            "p".into(),
        ];
        argv.extend(extra.iter().map(|s| s.to_string()));
        try_parse("infer", argv)
    };
    match parse(&["--video", "a", "--video", "b", "--video-fps", "2.5"]) {
        Ok(Commands::Infer(args)) => {
            assert_eq!(args.video, ["a", "b"]);
            assert_eq!(args.video_fps, Some(2.5));
        }
        other => panic!("{other:?}"),
    }
    assert!(parse(&["--video-fps", "2"]).is_err(), "fps without a video");
    for fps in ["0", "-1", "inf", "NaN", "two"] {
        assert!(
            parse(&["--video", "a", "--video-fps", fps]).is_err(),
            "{fps}"
        );
    }
}

/// `--backend` is the one way to pick a generation path: the spec, the
/// help text and the parser offer the same variants, and the per-path
/// flags it replaced are gone.
#[test]
fn infer_backend_is_the_only_path_selector() {
    use clap::CommandFactory;
    use pmetal_data::inference_config::InferenceBackend;

    let names: Vec<&str> = InferenceBackend::ALL.iter().map(|b| b.as_str()).collect();
    let spec_options = InferSpec::field_descriptors()
        .iter()
        .find(|d| d.name == "backend")
        .map(|d| match d.kind {
            pmetal_core::FieldKind::Enum { options } => options.to_vec(),
            other => panic!("backend is not an enum field: {other:?}"),
        })
        .expect("InferSpec has a backend field");
    assert_eq!(spec_options, names);

    let cli = Cli::command();
    let backend = cli
        .find_subcommand("infer")
        .and_then(|infer| infer.get_arguments().find(|a| a.get_id() == "backend"))
        .expect("infer has --backend");
    let help = backend.get_help().expect("--backend has help").to_string();
    let listed = help
        .split_once(':')
        .map(|(_, list)| list.trim().trim_end_matches('.'))
        .expect("--backend help lists the variants");
    assert_eq!(listed.split(" | ").collect::<Vec<_>>(), names);

    for name in &names {
        let spec = InferSpec {
            model: "model".into(),
            prompt: "Hello".into(),
            backend: (*name).to_string(),
            ..Default::default()
        };
        assert!(try_parse("infer", spec.to_argv()).is_ok(), "{name}");
    }
    for legacy in ["--compiled", "--metal-sampler", "--minimal"] {
        let argv = vec![
            "--model".to_string(),
            "m".into(),
            "--prompt".into(),
            "p".into(),
            legacy.into(),
        ];
        assert!(try_parse("infer", argv).is_err(), "{legacy} still parses");
    }
}

// Commands::Serve is only compiled when `--features serve` is active.
#[cfg(feature = "serve")]
#[test]
fn serve_spec_round_trip() {
    let spec = ServeSpec {
        model: "model".into(),
        ..Default::default()
    };
    let result = try_parse("serve", spec.to_argv());
    assert!(
        result.is_ok(),
        "ServeSpec argv failed to parse: {}",
        result.unwrap_err()
    );
}

#[test]
fn bench_spec_round_trip() {
    let spec = BenchSpec::default();
    let result = try_parse("bench", spec.to_argv());
    assert!(
        result.is_ok(),
        "BenchSpec argv failed to parse: {}",
        result.unwrap_err()
    );
}

#[test]
fn eval_spec_round_trip() {
    let spec = EvalSpec {
        model: "model".into(),
        dataset: "data.jsonl".into(),
        ..Default::default()
    };
    let result = try_parse("eval", spec.to_argv());
    assert!(
        result.is_ok(),
        "EvalSpec argv failed to parse: {}",
        result.unwrap_err()
    );
}

#[test]
fn quantize_spec_round_trip() {
    let spec = QuantizeSpec {
        model: "model".into(),
        output: "out.gguf".into(),
        ..Default::default()
    };
    let result = try_parse("quantize", spec.to_argv());
    assert!(
        result.is_ok(),
        "QuantizeSpec argv failed to parse: {}",
        result.unwrap_err()
    );
}

#[test]
fn quantize_methods_parse_canonical_names() {
    let methods = [
        "dynamic", "q1_0", "q2_k", "q3_k_s", "q3_k_m", "q3_k_l", "q4_0", "q4_1", "q4_k_s",
        "q4_k_m", "q5_0", "q5_1", "q5_k_s", "q5_k_m", "q6_k", "q8_0", "q8_1", "tq1_0", "tq2_0",
        "mxfp4", "nvfp4", "bf16", "f16", "f32",
    ];

    for method in methods {
        let argv = [
            "pmetal", "quantize", "--model", "model", "--output", "out.gguf", "--method", method,
        ];
        let result = Cli::try_parse_from(argv);
        if let Err(err) = result {
            panic!("quantize method {method} failed to parse: {err}");
        }
    }
}

// Commands::Merge requires `feature = "merge"` (a default feature but
// not implied by trainer alone when building with --no-default-features).
#[cfg(feature = "merge")]
#[test]
fn merge_spec_round_trip() {
    let spec = MergeSpec {
        model_a: "a".into(),
        model_b: "b".into(),
        output: "out".into(),
        ..Default::default()
    };
    let result = try_parse("merge", spec.to_argv());
    assert!(
        result.is_ok(),
        "MergeSpec argv failed to parse: {}",
        result.unwrap_err()
    );
}

#[test]
fn fuse_spec_round_trip() {
    let spec = FuseSpec {
        model: "model".into(),
        lora: "adapter".into(),
        output: "out".into(),
        ..Default::default()
    };
    let result = try_parse("fuse", spec.to_argv());
    assert!(
        result.is_ok(),
        "FuseSpec argv failed to parse: {}",
        result.unwrap_err()
    );
}

#[test]
fn dflash_spec_round_trip() {
    let spec = DflashSpec {
        target: "target-model".into(),
        draft: "draft-model".into(),
        prompt: "Hello".into(),
        ..Default::default()
    };
    let result = try_parse("dflash", spec.to_argv());
    assert!(
        result.is_ok(),
        "DflashSpec argv failed to parse: {}",
        result.unwrap_err()
    );
}

#[test]
fn pack_experts_spec_round_trip() {
    let spec = PackExpertsSpec {
        model: "model".into(),
        ..Default::default()
    };
    let result = try_parse("pack-experts", spec.to_argv());
    assert!(
        result.is_ok(),
        "PackExpertsSpec argv failed to parse: {}",
        result.unwrap_err()
    );
}

#[test]
fn decide_spec_round_trip() {
    let spec = DecideSpec {
        model: "Cloudflare/clef-flash".into(),
        request: "request.json".into(),
        compact: true,
        ..Default::default()
    };
    let result = try_parse("decide", spec.to_argv());
    assert!(
        result.is_ok(),
        "DecideSpec argv failed to parse: {}",
        result.unwrap_err()
    );
}

#[test]
fn tokenize_spec_round_trip() {
    let spec = TokenizeSpec {
        input: "corpus.jsonl".into(),
        output: "shards/".into(),
        tokenizer: "Qwen/Qwen3-0.6B".into(),
        ..Default::default()
    };
    let result = try_parse("tokenize", spec.to_argv());
    assert!(
        result.is_ok(),
        "TokenizeSpec argv failed to parse: {}",
        result.unwrap_err()
    );
}

// Smoke-test: default flags only contain the minimum required fields.
// Any flag that is present in to_argv() but not recognized by clap would
// have already failed the round-trip test above.
#[test]
fn train_spec_no_unexpected_flags() {
    let spec = TrainSpec {
        model: "m".into(),
        dataset: "d".into(),
        no_sequence_packing: true,
        cut_cross_entropy: true,
        no_adaptive_lr: true,
        ane: true,
        resume: true,
        ..Default::default()
    };
    let result = try_parse("train", spec.to_argv());
    assert!(
        result.is_ok(),
        "TrainSpec flag-heavy argv failed to parse: {}",
        result.unwrap_err()
    );
}
