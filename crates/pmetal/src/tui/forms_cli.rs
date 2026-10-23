//! A form's command line against the real parser, and its fields against its
//! spec's flags.
//!
//! Each test sets a form's fields to values that are not the defaults, takes
//! the command the form would launch, has `Cli::try_parse_from` read it, turns
//! what the parser read back into the spec, and compares that to the spec the
//! form built. A field the form drops, a flag the parser does not know, or a
//! value one side reads differently from the other shows up as a difference.

use pmetal_core::JobFields;
use pmetal_core::jobs::{
    BenchSpec, DflashSpec, DistillSpec, EmbedTrainSpec, EvalSpec, GrpoSpec, MergeSpec,
    PreferenceSpec, PretrainSpec, QuantizeSpec, RlkdSpec, ServeSpec, Spec, TokenizeSpec, TrainSpec,
};
use serde_json::{Value, json};

use crate::cli::Commands;
use crate::tui::app::Forms;
use crate::tui::cli_check::assert_parses;
use crate::tui::tabs::{FormTab, Pick};

fn model(id: &str) -> Pick<'_> {
    Pick::Model {
        id,
        architecture: None,
    }
}

fn set(form: &mut impl FormTab, label: &str, value: &str) {
    form.panel_mut().form.set_value(label, value);
}

/// Every form with what it needs to start.
fn filled() -> Forms {
    let mut f = Forms::new();
    f.training.on_pick("Model", model("org/Base"));
    f.training.on_pick("Dataset", Pick::Dataset("data.jsonl"));
    f.distillation.on_pick("Teacher", model("big/Teacher"));
    f.distillation.on_pick("Student", model("small/Student"));
    f.distillation
        .on_pick("Dataset", Pick::Dataset("data.jsonl"));
    f.grpo.on_pick("Model", model("org/Base"));
    f.grpo.on_pick("Dataset", Pick::Dataset("prompts.jsonl"));
    f.rlkd.on_pick("Policy Model", model("org/Policy"));
    f.rlkd.on_pick("Teacher Model", model("big/Teacher"));
    f.rlkd.on_pick("Dataset", Pick::Dataset("data.jsonl"));
    f.preference.on_pick("Model", model("org/Base"));
    f.preference
        .on_pick("Dataset", Pick::Dataset("pairs.jsonl"));
    f.embed_train.on_pick("Model", model("org/Embedder"));
    f.embed_train
        .on_pick("Dataset", Pick::Dataset("pairs.jsonl"));
    set(&mut f.pretrain, "Architecture", "llama");
    set(&mut f.pretrain, "Shards (csv)", "a.bin,b.bin");
    set(&mut f.tokenize, "Input JSONL", "in.jsonl");
    set(&mut f.tokenize, "Output Dir", "shards");
    f.tokenize.on_pick("Tokenizer Model", model("org/Base"));
    f.dflash.on_pick("Target Model", model("org/Target"));
    f.dflash.on_pick("Draft Model", model("org/Draft"));
    set(&mut f.dflash, "Prompt", "hello");
    f.serve.on_pick("Model", model("org/Base"));
    f.quantize.on_pick("Model", model("org/Base"));
    f.merge.on_pick("Model A", model("org/A"));
    f.merge.on_pick("Model B", model("org/B"));
    f.eval.on_pick("Model", model("org/Base"));
    f.eval.on_pick("Dataset", Pick::Dataset("eval.jsonl"));
    set(&mut f.ollama, "Base Model", "/m.gguf");
    f
}

#[test]
fn every_form_launches_a_command_the_cli_parses() {
    let forms = filled();
    for (tab, form) in forms.each() {
        let plan = form
            .plan()
            .unwrap_or_else(|issues| panic!("{tab}: {issues:?}"));
        // A command this build was compiled without (the server needs
        // `--features serve`) cannot be parsed; the interface says so instead.
        if crate::tui::jobs::missing_command(&plan).is_some() {
            assert_eq!(
                plan.subcommand, "serve",
                "{tab} launches a command that is not in this build"
            );
            continue;
        }
        assert_parses(&plan);
    }
}

/// The flags a spec can spell that its form has no field for.
fn omitted<S: JobFields>() -> Vec<&'static str> {
    let in_form: Vec<&str> = S::field_descriptors()
        .iter()
        .filter_map(|d| d.argv)
        .collect();
    S::argv_flags()
        .iter()
        .copied()
        .filter(|flag| !in_form.contains(flag))
        .collect()
}

/// The flags that are off the forms on purpose: `(subcommand, flag, why)`. A
/// flag the spec has that is in neither the form nor this table is a control
/// the interface lacks: add the field, or the reason.
const LEFT_OFF: &[(&str, &str, &str)] = &[
    (
        "train",
        "--config",
        "a config file replaces the form's values, so it would make the form lie about the run",
    ),
    (
        "train",
        "--log-metrics",
        "the interface sets it to the run's output directory, where the Jobs tab reads it",
    ),
];

#[test]
fn every_spec_flag_is_on_the_form_or_left_off_for_a_stated_reason() {
    let mut surprises = Vec::new();
    let mut stale = Vec::new();
    macro_rules! check {
        ($spec:ty) => {
            let omitted = omitted::<$spec>();
            for flag in &omitted {
                if !LEFT_OFF
                    .iter()
                    .any(|(sub, f, _)| *sub == <$spec>::subcommand() && f == flag)
                {
                    surprises.push(format!("{} {flag}", <$spec>::subcommand()));
                }
            }
            for (sub, flag, _) in LEFT_OFF {
                if *sub == <$spec>::subcommand() && !omitted.contains(flag) {
                    stale.push(format!("{sub} {flag}"));
                }
            }
        };
    }
    check!(TrainSpec);
    check!(DistillSpec);
    check!(GrpoSpec);
    check!(RlkdSpec);
    check!(PreferenceSpec);
    check!(PretrainSpec);
    check!(EmbedTrainSpec);
    check!(TokenizeSpec);
    check!(DflashSpec);
    check!(ServeSpec);
    check!(QuantizeSpec);
    check!(MergeSpec);
    check!(EvalSpec);
    check!(BenchSpec);
    assert!(
        surprises.is_empty(),
        "flags with no field on their form and no stated reason: {surprises:?}"
    );
    assert!(
        stale.is_empty(),
        "reasons for flags that now have a field: {stale:?}"
    );
}

#[test]
fn a_spec_field_with_no_descriptor_would_be_found_by_the_flag_check() {
    // `--log-metrics` is a flag of RlkdSpec too; it is on the form (it has a
    // descriptor), unlike TrainSpec's, so the table need not name it.
    assert!(
        omitted::<RlkdSpec>().is_empty(),
        "{:?}",
        omitted::<RlkdSpec>()
    );
    assert_eq!(omitted::<TrainSpec>(), ["--config", "--log-metrics"]);
}

/// The spec the form built, as JSON, with the fields the interface sets itself
/// (where a run logs) taken from the parsed one.
fn form_spec<S: Spec>(form: &impl FormTab, parsed: &Value) -> Value {
    let mut spec =
        serde_json::to_value(form.panel().form.to_spec::<S>().expect("the form is valid")).unwrap();
    if let Some(log) = parsed.get("log_metrics") {
        spec["log_metrics"] = log.clone();
    }
    spec
}

fn json_of<S: serde::Serialize>(spec: S) -> Value {
    serde_json::to_value(spec).unwrap()
}

#[test]
fn what_the_cli_reads_from_each_form_is_the_spec_the_form_built() {
    let mut f = filled();

    // Training: numbers, a toggle, an option, an optional, a list.
    set(&mut f.training, "Epochs", "3");
    set(&mut f.training, "Learning Rate", "1e-5");
    set(&mut f.training, "Batch Size", "2");
    set(&mut f.training, "Quantization", "nf4");
    set(&mut f.training, "Cut Cross-Entropy", "Enabled");
    set(&mut f.training, "Resume", "Enabled");
    set(&mut f.training, "Eval Dataset", "eval.jsonl");
    let plan = f.training.plan().unwrap();
    let Commands::Train(args) = assert_parses(&plan) else {
        panic!("not a train command")
    };
    let parsed = json_of(TrainSpec::from(args));
    assert_eq!(parsed["epochs"], 3);
    assert_eq!(parsed["resume"], true);
    assert_eq!(form_spec::<TrainSpec>(&f.training, &parsed), parsed);

    // Distillation.
    set(&mut f.distillation, "Temperature", "3.5");
    set(&mut f.distillation, "Resume", "Enabled");
    let plan = f.distillation.plan().unwrap();
    let Commands::Distill(args) = assert_parses(&plan) else {
        panic!("not a distill command")
    };
    let parsed = json_of(DistillSpec::from(args));
    assert_eq!(parsed["resume"], true);
    assert_eq!(form_spec::<DistillSpec>(&f.distillation, &parsed), parsed);

    // GRPO: the fields the old hand-written mapping never read.
    set(&mut f.grpo, "Loss Type", "gspo");
    set(&mut f.grpo, "Updates per Batch", "2");
    set(&mut f.grpo, "Checkpoint Every", "10");
    set(&mut f.grpo, "Optimizer", "lion");
    set(&mut f.grpo, "Resume", "Enabled");
    let plan = f.grpo.plan().unwrap();
    let Commands::Grpo(args) = assert_parses(&plan) else {
        panic!("not a grpo command")
    };
    let parsed = json_of(GrpoSpec::from(args));
    assert_eq!(parsed["loss_type"], "gspo");
    assert_eq!(parsed["optimizer"], "lion");
    assert_eq!(form_spec::<GrpoSpec>(&f.grpo, &parsed), parsed);

    // RLKD and preference optimization.
    set(&mut f.rlkd, "Learning Rate", "2e-6");
    let plan = f.rlkd.plan().unwrap();
    let Commands::Rlkd(args) = assert_parses(&plan) else {
        panic!("not an rlkd command")
    };
    let parsed = json_of(RlkdSpec::from(args));
    assert_eq!(form_spec::<RlkdSpec>(&f.rlkd, &parsed), parsed);
    set(&mut f.preference, "Learning Rate", "5e-6");
    let plan = f.preference.plan().unwrap();
    let Commands::Preference(args) = assert_parses(&plan) else {
        panic!("not a preference command")
    };
    let parsed = json_of(PreferenceSpec::from(args));
    assert_eq!(form_spec::<PreferenceSpec>(&f.preference, &parsed), parsed);

    // Quantize, merge, eval, dflash, tokenize.
    set(&mut f.quantize, "Method", "dynamic");
    set(&mut f.quantize, "KL Calibration", "Enabled");
    set(&mut f.quantize, "Target BPW", "4.5");
    let plan = f.quantize.plan().unwrap();
    let Commands::Quantize(args) = assert_parses(&plan) else {
        panic!("not a quantize command")
    };
    let parsed = json_of(QuantizeSpec::from(args));
    assert_eq!(parsed["kl_calibrate"], true);
    assert_eq!(form_spec::<QuantizeSpec>(&f.quantize, &parsed), parsed);

    set(&mut f.merge, "Method", "slerp");
    set(&mut f.merge, "SLERP t", "0.25");
    let plan = f.merge.plan().unwrap();
    let Commands::Merge(args) = assert_parses(&plan) else {
        panic!("not a merge command")
    };
    let parsed = json_of(MergeSpec::from(args));
    assert_eq!(form_spec::<MergeSpec>(&f.merge, &parsed), parsed);

    set(&mut f.eval, "Num Samples", "50");
    set(&mut f.eval, "JSON Output", "Enabled");
    let plan = f.eval.plan().unwrap();
    let Commands::Eval(args) = assert_parses(&plan) else {
        panic!("not an eval command")
    };
    let parsed = json_of(EvalSpec::from(args));
    assert_eq!(parsed["json"], true);
    assert_eq!(form_spec::<EvalSpec>(&f.eval, &parsed), parsed);

    set(&mut f.dflash, "Max New Tokens", "64");
    set(&mut f.dflash, "Draft FP8", "Enabled");
    let plan = f.dflash.plan().unwrap();
    let Commands::Dflash(args) = assert_parses(&plan) else {
        panic!("not a dflash command")
    };
    let parsed = json_of(DflashSpec::from(args));
    assert_eq!(form_spec::<DflashSpec>(&f.dflash, &parsed), parsed);

    set(&mut f.tokenize, "Docs Per Shard", "500");
    let plan = f.tokenize.plan().unwrap();
    let Commands::Tokenize(args) = assert_parses(&plan) else {
        panic!("not a tokenize command")
    };
    let parsed = json_of(TokenizeSpec::from(args));
    assert_eq!(form_spec::<TokenizeSpec>(&f.tokenize, &parsed), parsed);
}

#[test]
fn a_benchmark_form_reads_back_in_both_modes() {
    let mut f = filled();
    set(&mut f.bench, "Mode", "basic");
    set(&mut f.bench, "Batch Size", "2");
    set(&mut f.bench, "Seq Len", "256");
    let plan = f.bench.plan().unwrap();
    let Commands::Bench(args) = assert_parses(&plan) else {
        panic!("not a bench command")
    };
    let parsed = json_of(BenchSpec::from(args));
    assert_eq!(
        parsed,
        json!({"model": "Qwen/Qwen3-0.6B", "batch_size": 2, "seq_len": 256})
    );
    assert_eq!(form_spec::<BenchSpec>(&f.bench, &parsed), parsed);

    set(&mut f.bench, "Mode", "workload");
    set(&mut f.bench, "Prompt Samples", "4");
    set(&mut f.bench, "Decode Steps", "16");
    let Commands::BenchWorkload {
        prompt_samples,
        decode_steps,
        preset,
        ..
    } = assert_parses(&f.bench.plan().unwrap())
    else {
        panic!("not a bench-workload command")
    };
    assert_eq!((prompt_samples, decode_steps), (4, 16));
    assert!(preset.is_none());
}

#[cfg(feature = "serve")]
#[test]
fn a_serve_form_reads_back() {
    let mut f = filled();
    set(&mut f.serve, "Port", "9090");
    set(&mut f.serve, "FP8 Weights", "Enabled");
    set(&mut f.serve, "CORS Origins", "http://localhost:3000");
    let plan = f.serve.plan().unwrap();
    let Commands::Serve(args) = assert_parses(&plan) else {
        panic!("not a serve command")
    };
    let (spec, keys) = args.into_spec();
    assert!(keys.is_empty());
    assert_eq!(spec.port, 9090);
    assert!(spec.fp8);
    let parsed = json_of(spec);
    assert_eq!(form_spec::<ServeSpec>(&f.serve, &parsed), parsed);
}
