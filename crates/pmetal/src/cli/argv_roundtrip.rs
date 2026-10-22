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

/// A spec as JSON, for comparing two specs field by field.
fn json<S: serde::Serialize>(spec: &S) -> serde_json::Value {
    serde_json::to_value(spec).expect("a spec serializes")
}

/// A spec with every field off its default, so a field the parser drops or
/// misreads shows as a difference.
#[cfg(feature = "trainer")]
fn busy_train_spec() -> TrainSpec {
    TrainSpec {
        model: "org/model".into(),
        dataset: "data.jsonl".into(),
        eval_dataset: Some("eval.jsonl".into()),
        output_dir: "out".into(),
        learning_rate: 1e-4,
        embedding_lr: Some(5e-5),
        label_smoothing: 0.1,
        z_loss: 1e-4,
        neftune_alpha: Some(5.0),
        batch_size: 2,
        epochs: 3,
        max_steps: Some(40),
        max_seq_len: 512,
        gradient_accumulation_steps: 8,
        no_gradient_checkpointing: true,
        max_grad_norm: 0.5,
        warmup_steps: 7,
        warmup_ratio: Some(0.05),
        weight_decay: 0.02,
        optimizer: "lion".into(),
        lr_schedule: "wsd".into(),
        min_lr: Some(1e-6),
        wsd_stable_ratio: Some(0.6),
        wsd_decay_style: Some("one_minus_sqrt".into()),
        cosine_num_restarts: Some(3),
        polynomial_power: Some(2.0),
        logging_steps: 5,
        eval_steps: Some(20),
        save_steps: Some(25),
        seed: 7,
        loss_scale: 4.0,
        lora_r: 8,
        lora_alpha: 24.0,
        dora: true,
        rslora: true,
        lora_dropout: 0.05,
        lora_targets: vec!["q_proj".into(), "v_proj".into()],
        quantization: Some("nf4".into()),
        quant_block_size: 128,
        double_quant: true,
        text_column: Some("body".into()),
        text_columns: Some("a,b".into()),
        column_separator: Some("\n---\n".into()),
        prompt_column: Some("q".into()),
        response_column: Some("r".into()),
        no_sequence_packing: true,
        cut_cross_entropy: true,
        no_adaptive_lr: true,
        ane: true,
        pack_max_seq_len: Some(1024),
        distributed_peers: Some("10.0.0.2:7000,10.0.0.3:7000".into()),
        distributed_auto: true,
        compression_strategy: "topk".into(),
        config_path: Some("train.yaml".into()),
        resume: true,
        log_metrics: Some("metrics.jsonl".into()),
    }
}

/// Every field of a train spec survives spec -> argv -> parser -> spec, and
/// so does a spec that is all defaults, so the parser's defaults are the
/// spec's.
#[cfg(feature = "trainer")]
#[test]
fn train_spec_round_trips_every_field() {
    let plain = TrainSpec {
        model: "m".into(),
        dataset: "d".into(),
        ..Default::default()
    };
    for spec in [plain, busy_train_spec()] {
        let parsed = match try_parse("train", spec.to_argv()) {
            Ok(Commands::Train(args)) => TrainSpec::from(args),
            Ok(_) => panic!("parsed as another command"),
            Err(e) => panic!("TrainSpec argv failed to parse: {e}"),
        };
        assert_eq!(json(&parsed), json(&spec));
    }
}

/// The options that shape a run's length, schedule and cadence reach the
/// trainer's config from the command line.
#[cfg(feature = "trainer")]
#[test]
fn train_flags_reach_the_job_config() {
    use pmetal_trainer::orchestrator::TrainingJobConfig;

    let spec = busy_train_spec();
    let Ok(Commands::Train(args)) = try_parse("train", spec.to_argv()) else {
        panic!("not a train command")
    };
    let job = TrainingJobConfig::from_spec(&TrainSpec::from(args)).unwrap();
    let t = &job.training;
    assert_eq!(t.max_steps, Some(40));
    assert_eq!(t.warmup_steps, 7);
    assert_eq!(t.warmup_ratio, Some(0.05));
    assert_eq!(t.min_lr, Some(1e-6));
    assert_eq!(t.lr_scheduler, pmetal_core::LrSchedulerType::Wsd);
    assert_eq!(t.wsd_stable_ratio, Some(0.6));
    assert_eq!(
        t.wsd_decay_style,
        Some(pmetal_core::WsdDecayStyle::OneMinusSqrt)
    );
    assert_eq!(t.cosine_num_restarts, Some(3));
    assert_eq!(t.polynomial_power, Some(2.0));
    assert_eq!(
        (t.logging_steps, t.eval_steps, t.save_steps),
        (5, Some(20), Some(25))
    );
    assert_eq!(job.lora.target_modules, ["q_proj", "v_proj"]);
    assert_eq!(job.log_metrics.as_deref(), Some("metrics.jsonl"));
    assert_eq!(
        job.columns
            .as_ref()
            .and_then(|c| c.column_separator.as_deref()),
        Some("\n---\n")
    );
    let qlora = job.qlora.expect("nf4 asks for QLoRA");
    assert_eq!((qlora.block_size, qlora.double_quant), (128, true));
}

/// A run configured by a file needs no model or dataset on the command line,
/// and one with neither is refused with the flag named.
#[cfg(feature = "trainer")]
#[test]
fn train_needs_a_model_and_dataset_unless_a_config_file_gives_them() {
    let parse =
        |extra: &[&str]| match try_parse("train", extra.iter().map(|s| s.to_string()).collect()) {
            Ok(Commands::Train(args)) => TrainSpec::from(args),
            other => panic!("{other:?}"),
        };
    assert!(parse(&["--config", "train.yaml"]).normalize().is_ok());
    let errors = parse(&[]).normalize().unwrap_err();
    let message = crate::commands::invalid_spec::<TrainSpec>(&errors).to_string();
    assert!(message.contains("--model is required"), "{message}");
    assert!(message.contains("--dataset is required"), "{message}");
    // The option a value cannot be is named too.
    let errors = parse(&["--config", "c.yaml", "--learning-rate", "5"])
        .normalize()
        .unwrap_err();
    let message = crate::commands::invalid_spec::<TrainSpec>(&errors).to_string();
    assert!(message.contains("--learning-rate"), "{message}");
}

#[cfg(feature = "trainer")]
#[test]
fn train_refuses_what_it_cannot_run() {
    for (flag, value) in [
        ("--lr-schedule", "cosin"),
        ("--optimizer", "adam"),
        ("--wsd-decay-style", "sharp"),
        ("--compression-strategy", "zstd"),
        ("--quantization", "nf8"),
    ] {
        let argv = ["--model", "m", "--dataset", "d", flag, value]
            .map(String::from)
            .to_vec();
        let error = try_parse("train", argv).expect_err(flag);
        assert!(error.contains(value), "{flag}: {error}");
    }
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

/// `--format` is gguf or mlx: any other word used to quantize to GGUF without
/// saying so.
#[test]
fn quantize_refuses_a_format_it_does_not_have() {
    let argv = |format: &str| {
        ["--model", "m", "--output", "o", "--format", format]
            .map(String::from)
            .to_vec()
    };
    assert!(try_parse("quantize", argv("mlx")).is_ok());
    for typo in ["MLX", "safetensors", "gguff"] {
        let error = try_parse("quantize", argv(typo)).unwrap_err();
        assert!(
            error.contains(typo) && error.contains("gguf"),
            "{typo}: {error}"
        );
    }
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

/// Assert a spec survives spec -> argv -> parser -> spec, field by field.
/// The specs below are written out in full, so a field added to one fails to
/// compile here until the round trip covers it.
#[cfg(feature = "trainer")]
fn assert_round_trip<S, A>(spec: S, subcommand: &str, args: impl FnOnce(Commands) -> Option<A>)
where
    S: JobFields + serde::Serialize + From<A>,
{
    let parsed = try_parse(subcommand, spec.to_argv())
        .unwrap_or_else(|e| panic!("`{subcommand}` argv failed to parse: {e}"));
    let args = args(parsed).unwrap_or_else(|| panic!("parsed as another command"));
    assert_eq!(json(&S::from(args)), json(&spec), "{subcommand}");
}

#[cfg(feature = "trainer")]
#[test]
fn distill_spec_round_trips_every_field() {
    let busy = DistillSpec {
        teacher: "org/teacher".into(),
        student: "org/student".into(),
        dataset: "d.jsonl".into(),
        eval_dataset: Some("e.jsonl".into()),
        output_dir: "out".into(),
        method: "offline".into(),
        offline: false,
        offline_cache: Some("cache".into()),
        offline_generate: true,
        offline_compression: "int8".into(),
        offline_top_k: 64,
        loss_type: "skew_kl".into(),
        skew_alpha: Some(0.2),
        universal_logit_top_k: Some(32),
        temperature: 3.0,
        alpha: 0.7,
        rationale: true,
        rationale_weight: 2.0,
        on_policy: 0.25,
        on_policy_max_tokens: 128,
        lora_r: 8,
        lora_alpha: 16.0,
        learning_rate: 1e-4,
        optimizer: "lion".into(),
        batch_size: 2,
        epochs: 2,
        max_seq_len: 2048,
        seed: 3,
        checkpoint_every: 10,
        text_column: Some("t".into()),
        text_columns: Some("a,b".into()),
        column_separator: Some("|".into()),
        prompt_column: Some("p".into()),
        response_column: Some("r".into()),
        log_metrics: Some("m.jsonl".into()),
        resume: true,
    };
    let plain = DistillSpec {
        teacher: "t".into(),
        student: "s".into(),
        dataset: "d".into(),
        ..Default::default()
    };
    for spec in [plain, busy] {
        assert_round_trip(spec, "distill", |c| match c {
            Commands::Distill(a) => Some(a),
            _ => None,
        });
    }
}

#[cfg(feature = "trainer")]
#[test]
fn grpo_and_rlkd_specs_round_trip_every_field() {
    let grpo = GrpoSpec {
        model: "m".into(),
        dataset: "d".into(),
        output_dir: "out".into(),
        num_generations: 4,
        beta: 0.04,
        learning_rate: 1e-6,
        optimizer: "lion".into(),
        epochs: 2,
        lora_r: 8,
        lora_alpha: 16.0,
        max_seq_len: 1024,
        max_completion_length: 256,
        seed: 9,
        checkpoint_every: 5,
        dapo: true,
        loss_type: "gspo".into(),
        num_iterations: 2,
        reasoning_rewards: true,
        vlm: true,
        max_image_size: 224,
        reward_model: Some("rm".into()),
        reward_model_max_length: 1024,
        reward_model_weight: 0.5,
        reward_model_template: Some("chatml".into()),
        async_rewards: true,
        prompt_column: Some("q".into()),
        response_column: Some("a".into()),
        grpo_kv_bits: Some(4),
        log_metrics: Some("m.jsonl".into()),
        resume: true,
    };
    for spec in [
        GrpoSpec {
            model: "m".into(),
            dataset: "d".into(),
            ..Default::default()
        },
        grpo,
    ] {
        assert_round_trip(spec, "grpo", |c| match c {
            Commands::Grpo(a) => Some(a),
            _ => None,
        });
    }
    let rlkd = RlkdSpec {
        model: "m".into(),
        teacher_model: "t".into(),
        dataset: "d".into(),
        output_dir: "out".into(),
        distill_alpha: 0.4,
        final_alpha: 0.1,
        anneal_alpha: true,
        distill_temperature: 3.0,
        num_generations: 4,
        beta: 0.01,
        learning_rate: 1e-6,
        epochs: 2,
        lora_r: 8,
        lora_alpha: 16.0,
        max_seq_len: 1024,
        max_completion_length: 256,
        seed: 9,
        reasoning_rewards: true,
        prompt_column: Some("q".into()),
        response_column: Some("a".into()),
        log_metrics: Some("m.jsonl".into()),
    };
    for spec in [
        RlkdSpec {
            model: "m".into(),
            teacher_model: "t".into(),
            dataset: "d".into(),
            ..Default::default()
        },
        rlkd,
    ] {
        assert_round_trip(spec, "rlkd", |c| match c {
            Commands::Rlkd(a) => Some(a),
            _ => None,
        });
    }
}

#[cfg(feature = "trainer")]
#[test]
fn train_diffusion_spec_round_trips_every_field() {
    use pmetal_core::jobs::TrainDiffusionSpec;
    let busy = TrainDiffusionSpec {
        model: "m".into(),
        dataset: "d".into(),
        output: "out".into(),
        lora_r: 8,
        lora_alpha: 16.0,
        lora_targets: vec!["q_proj".into(), "v_proj".into()],
        corrupted_only: false,
        quantization: "nf4".into(),
        quant_block_size: 32,
        double_quant: true,
        steps: 50,
        learning_rate: 1e-4,
        weight_decay: 0.01,
        max_grad_norm: 0.5,
        max_context_len: 256,
        checkpoint_every: 10,
        log_every: 5,
        seed: 3,
    };
    let plain = TrainDiffusionSpec {
        model: "m".into(),
        dataset: "d".into(),
        ..Default::default()
    };
    for spec in [plain, busy] {
        assert_round_trip(spec, "train-diffusion", |c| match c {
            Commands::TrainDiffusion(a) => Some(a),
            _ => None,
        });
    }
}

#[test]
fn infer_spec_round_trips_every_field() {
    let busy = InferSpec {
        model: "m".into(),
        lora: Some("adapter".into()),
        prompt: "hello there".into(),
        image: vec!["a.png".into(), "b.png".into()],
        video: vec!["frames".into()],
        video_fps: Some(2.5),
        json: true,
        max_tokens: Some(64),
        temperature: Some(0.7),
        top_k: Some(40),
        top_p: Some(0.9),
        min_p: Some(0.05),
        repetition_penalty: Some(1.1),
        frequency_penalty: Some(0.2),
        presence_penalty: Some(0.3),
        seed: Some(5),
        chat: true,
        system: Some("be brief".into()),
        no_thinking: true,
        reasoning_effort: Some("low".into()),
        no_preserve_thinking: true,
        chat_template_kwargs: Some("{\"x\": 1}".into()),
        hide_thinking: true,
        mode: "thinking-general".into(),
        backend: "standard".into(),
        draft_model: Some("draft".into()),
        mtp: true,
        mtp_model: Some("mtp".into()),
        mtp_draft_tokens: 4,
        tools: Some("tools.json".into()),
        fp8: true,
        experts_dir: Some("experts".into()),
        ane: true,
        ane_max_seq_len: 2048,
        benchmark: true,
        benchmark_iters: 3,
        benchmark_prompt_tokens: Some(128),
        profile_layers: true,
        profile_output: Some("p.json".into()),
        kv_quant: Some(8),
        kv_k_bits: Some(4),
        kv_v_bits: Some(2),
        kv_group_size: 32,
        kv_turboquant: true,
        kv_turboquant_preset: Some("q3_5".into()),
        kv_quant_preset: Some("q2_5".into()),
        no_kv_quant: true,
        kv_qjl: true,
        detect_repetition: true,
    };
    let plain = InferSpec {
        model: "m".into(),
        prompt: "hi".into(),
        ..Default::default()
    };
    for spec in [plain, busy] {
        let parsed = match try_parse("infer", spec.to_argv()) {
            Ok(Commands::Infer(a)) => InferSpec::from(a),
            other => panic!("{other:?}"),
        };
        assert_eq!(json(&parsed), json(&spec));
    }
}

/// Spec -> argv -> parser -> spec for a command whose `From<Args>` is plain.
fn round_trips<S, A>(subcommand: &str, specs: [S; 2], args: impl Fn(Commands) -> Option<A>)
where
    S: JobFields + serde::Serialize + From<A>,
{
    for spec in specs {
        let parsed = try_parse(subcommand, spec.to_argv())
            .unwrap_or_else(|e| panic!("`{subcommand}` argv failed to parse: {e}"));
        let args =
            args(parsed).unwrap_or_else(|| panic!("`{subcommand}` parsed as another command"));
        assert_eq!(json(&S::from(args)), json(&spec), "{subcommand}");
    }
}

#[test]
fn model_commands_round_trip_every_field() {
    use pmetal_core::jobs::VocodeSpec;
    round_trips(
        "merge",
        [
            MergeSpec {
                model_a: "a".into(),
                model_b: "b".into(),
                output: "out".into(),
                ..Default::default()
            },
            MergeSpec {
                model_a: "a".into(),
                model_b: "b".into(),
                output: "out".into(),
                method: "ties".into(),
                base: Some("base".into()),
                t: 0.3,
                weight_a: 0.6,
                weight_b: 0.4,
                density: 0.7,
                dtype: "float16".into(),
            },
        ],
        |c| match c {
            Commands::Merge(a) => Some(a),
            _ => None,
        },
    );
    round_trips(
        "quantize",
        [
            QuantizeSpec {
                model: "m".into(),
                output: "o.gguf".into(),
                ..Default::default()
            },
            QuantizeSpec {
                model: "m".into(),
                output: "o".into(),
                imatrix: Some("im.dat".into()),
                method: "q5_k_m".into(),
                lora: Some("adapter".into()),
                kl_calibrate: true,
                target_bpw: Some(4.5),
                kl_threshold: 0.02,
                format: "mlx".into(),
                bits: 8,
                group_size: 32,
            },
        ],
        |c| match c {
            Commands::Quantize(a) => Some(a),
            _ => None,
        },
    );
    round_trips(
        "fuse",
        [
            FuseSpec {
                model: "m".into(),
                lora: "l".into(),
                output: "o".into(),
                ..Default::default()
            },
            FuseSpec {
                model: "m".into(),
                lora: "l".into(),
                output: "o".into(),
                alpha: Some(8.0),
                rank: Some(4),
                low_memory: true,
            },
        ],
        |c| match c {
            Commands::Fuse(a) => Some(a),
            _ => None,
        },
    );
    round_trips(
        "vocode",
        [
            VocodeSpec {
                model: "m".into(),
                input: "i.wav".into(),
                output: "o.wav".into(),
                ..Default::default()
            },
            VocodeSpec {
                model: "m".into(),
                input: "i.wav".into(),
                output: "o.wav".into(),
                chunk_frames: 64,
            },
        ],
        |c| match c {
            Commands::Vocode(a) => Some(a),
            _ => None,
        },
    );
}

#[test]
fn data_and_tool_commands_round_trip_every_field() {
    round_trips(
        "tokenize",
        [
            TokenizeSpec {
                input: "i".into(),
                output: "o".into(),
                tokenizer: "t".into(),
                ..Default::default()
            },
            TokenizeSpec {
                input: "i".into(),
                output: "o".into(),
                tokenizer: "t".into(),
                text_column: "body".into(),
                docs_per_shard: 50,
            },
        ],
        |c| match c {
            Commands::Tokenize(a) => Some(a),
            _ => None,
        },
    );
    round_trips(
        "pack-experts",
        [
            PackExpertsSpec {
                model: "m".into(),
                ..Default::default()
            },
            PackExpertsSpec {
                model: "m".into(),
                output: "packed".into(),
                bits: Some(4),
            },
        ],
        |c| match c {
            Commands::PackExperts(a) => Some(a),
            _ => None,
        },
    );
    round_trips(
        "decide",
        [
            DecideSpec {
                model: "m".into(),
                request: "r.json".into(),
                ..Default::default()
            },
            DecideSpec {
                model: "m".into(),
                request: "r.json".into(),
                max_length: 2048,
                compact: true,
            },
        ],
        |c| match c {
            Commands::Decide(a) => Some(a),
            _ => None,
        },
    );
    round_trips(
        "dflash",
        [
            DflashSpec {
                target: "t".into(),
                draft: "d".into(),
                prompt: "p".into(),
                ..Default::default()
            },
            DflashSpec {
                target: "t".into(),
                draft: "d".into(),
                prompt: "p".into(),
                max_new_tokens: 64,
                temperature: 0.7,
                speculative_tokens: Some(8),
                draft_fp8: true,
                json: true,
                no_chat: true,
                tree_budget: 4,
                compare_greedy: true,
            },
        ],
        |c| match c {
            Commands::Dflash(a) => Some(a),
            _ => None,
        },
    );
    round_trips(
        "bench",
        [
            BenchSpec::default(),
            BenchSpec {
                model: "m".into(),
                batch_size: 4,
                seq_len: 1024,
            },
        ],
        |c| match c {
            Commands::Bench(a) => Some(a),
            _ => None,
        },
    );
}

#[cfg(feature = "trainer")]
#[test]
fn eval_spec_round_trips_every_field() {
    round_trips(
        "eval",
        [
            EvalSpec {
                model: "m".into(),
                dataset: "d".into(),
                ..Default::default()
            },
            EvalSpec {
                model: "m".into(),
                dataset: "d".into(),
                lora: Some("adapter".into()),
                max_seq_len: 512,
                num_samples: 20,
                json: true,
            },
        ],
        |c| match c {
            Commands::Eval(a) => Some(a),
            _ => None,
        },
    );
}

/// The server's API keys are not a spec field: they stay on the command line
/// the operator typed (or in the environment), never in an argv built from a
/// spec.
#[cfg(feature = "serve")]
#[test]
fn serve_spec_round_trips_every_field_but_the_keys() {
    let busy = ServeSpec {
        model: "m".into(),
        host: "0.0.0.0".into(),
        port: 9000,
        cors_origin: vec!["https://a.example".into(), "*".into()],
        request_timeout: 60,
        max_seq_len: Some(8192),
        experts_dir: Some("experts".into()),
        fp8: true,
        kv_quant: Some(4),
        no_kv_quant: true,
        kv_group_size: 32,
        kv_turboquant: true,
        kv_turboquant_preset: Some("q2_5".into()),
        ane: true,
        ane_max_seq_len: 2048,
        draft_model: Some("draft".into()),
        mtp: true,
        mtp_model: Some("head".into()),
        draft_tokens: Some(5),
        continuous_batch: true,
        cb_max_slots: 4,
        cb_max_queue_depth: 16,
        cb_block_size: 16,
        cb_max_blocks: 100,
        cb_exact: true,
    };
    let plain = ServeSpec {
        model: "m".into(),
        ..Default::default()
    };
    for spec in [plain, busy] {
        let Ok(Commands::Serve(args)) = try_parse("serve", spec.to_argv()) else {
            panic!("serve argv failed to parse")
        };
        let (parsed, keys) = args.into_spec();
        assert_eq!(json(&parsed), json(&spec));
        assert!(keys.is_empty());
    }
    assert!(!ServeSpec::argv_flags().contains(&"--api-key"));
    let Ok(Commands::Serve(args)) = try_parse(
        "serve",
        ["--model", "m", "--api-key", "k1", "--api-key", "k2"]
            .map(String::from)
            .to_vec(),
    ) else {
        panic!("--api-key is a serve flag")
    };
    assert_eq!(args.into_spec().1, ["k1", "k2"]);
}

#[cfg(feature = "trainer")]
#[test]
fn train_mtp_and_draft_specs_round_trip_every_field_resume_included() {
    use pmetal_core::jobs::{TrainDraftSpec, TrainMtpSpec};
    round_trips(
        "train-mtp",
        [
            TrainMtpSpec {
                model: "m".into(),
                shards: vec!["a.bin".into()],
                ..Default::default()
            },
            TrainMtpSpec {
                model: "m".into(),
                family: "gemma4".into(),
                init_mtp: None,
                mtp_layers: 2,
                assistant_layers: 3,
                num_assistant_tokens: 4,
                shards: vec!["a.bin".into(), "b.bin".into()],
                eos_token_id: 2,
                output: "out".into(),
                seq_len: 256,
                batch_size: 2,
                steps: 50,
                learning_rate: 1e-4,
                min_lr: 1e-6,
                warmup_steps: 5,
                lr_schedule: "wsd".into(),
                weight_decay: 0.02,
                max_grad_norm: 0.5,
                checkpoint_every: 10,
                log_every: 2,
                seed: 7,
                resume: Some("out/checkpoints/step_10".into()),
            },
        ],
        |c| match c {
            Commands::TrainMtp(a) => Some(a),
            _ => None,
        },
    );
    round_trips(
        "train-draft",
        [
            TrainDraftSpec {
                target: "t".into(),
                draft_config: Some("c.json".into()),
                shards: vec!["a.bin".into()],
                ..Default::default()
            },
            TrainDraftSpec {
                target: "t".into(),
                draft: None,
                draft_config: None,
                shards: vec!["a.bin".into(), "b.bin".into()],
                eos_token_id: 2,
                output: "out".into(),
                seq_len: 256,
                batch_size: 2,
                steps: 50,
                learning_rate: 1e-4,
                min_lr: 1e-6,
                warmup_steps: 5,
                lr_schedule: "linear".into(),
                weight_decay: 0.02,
                max_grad_norm: 0.5,
                checkpoint_every: 10,
                log_every: 2,
                seed: 7,
                resume: Some("out".into()),
            },
        ],
        |c| match c {
            Commands::TrainDraft(a) => Some(a),
            _ => None,
        },
    );
}

/// A resumed run continues from its own checkpoint: naming the weights to
/// start from as well is a contradiction, refused by the parser and by a spec
/// that did not come through it.
#[cfg(feature = "trainer")]
#[test]
fn resume_and_a_starting_point_are_one_or_the_other() {
    use pmetal_core::jobs::{TrainDraftSpec, TrainMtpSpec};
    let argv = |extra: &[&str]| {
        let mut v: Vec<String> = ["--model", "m", "--shards", "a.bin"]
            .map(String::from)
            .to_vec();
        v.extend(extra.iter().map(|s| s.to_string()));
        v
    };
    assert!(try_parse("train-mtp", argv(&["--resume", "out"])).is_ok());
    assert!(try_parse("train-mtp", argv(&["--resume", "out", "--init-mtp", "x"])).is_err());
    let draft = |extra: &[&str]| {
        let mut v: Vec<String> = ["--target", "t", "--shards", "a.bin"]
            .map(String::from)
            .to_vec();
        v.extend(extra.iter().map(|s| s.to_string()));
        v
    };
    assert!(try_parse("train-draft", draft(&["--resume", "out"])).is_ok());
    assert!(try_parse("train-draft", draft(&["--resume", "out", "--draft", "d"])).is_err());

    let mut mtp = TrainMtpSpec {
        model: "m".into(),
        shards: vec!["a".into()],
        resume: Some("out".into()),
        init_mtp: Some("x".into()),
        ..Default::default()
    };
    assert!(mtp.normalize().is_err());
    let mut draft_spec = TrainDraftSpec {
        target: "t".into(),
        shards: vec!["a".into()],
        resume: Some("out".into()),
        draft: Some("d".into()),
        ..Default::default()
    };
    assert!(draft_spec.normalize().is_err());
}
