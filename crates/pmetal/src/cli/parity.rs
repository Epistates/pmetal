//! Every command's flags, read off the real argument parser, against the
//! job spec the other surfaces build the same command from.
//!
//! The CLI, the TUI, the GUI, the MCP server and the Python bindings all start
//! a job from its `pmetal_core::jobs` spec. These tests hold the parser to that
//! contract without naming a single flag: they walk clap's description of each
//! subcommand and the spec's, and compare them.
//!
//! * the flag sets match, up to a short list of flags that exist only on the
//!   command line, each with its reason;
//! * a flag that takes a value has the spec's default, so the two cannot be
//!   edited apart;
//! * a flag with a fixed set of values offers the spec's set;
//! * the argv a populated spec spells parses, and parses back into a spec that
//!   spells the same argv.

use clap::{Arg, ArgAction, Command, CommandFactory};
use pmetal_core::{DefaultValue, FieldKind, JobFields};
use std::collections::BTreeSet;

use crate::Cli;

/// Flags the command line has and the spec does not, with why. A flag listed
/// here is one the spec cannot drive; it is for a surface that is not a job
/// (a path to print to, a mode that never runs a job).
const CLI_ONLY: &[(&str, &str, &str)] = &[(
    "serve",
    "--api-key",
    "a secret: a spec becomes a command line, and a key on one shows in every process list; \
     the server also reads PMETAL_API_KEY",
)];

/// Flags the spec spells and the parser does not take. There should be none.
const SPEC_ONLY: &[(&str, &str, &str)] = &[];

fn flags_of(command: &Command) -> BTreeSet<String> {
    command
        .get_arguments()
        .filter(|a| !a.is_global_set())
        .filter(|a| !matches!(a.get_id().as_str(), "help" | "version"))
        .filter_map(|a| a.get_long().map(|long| format!("--{long}")))
        .collect()
}

fn argument<'a>(command: &'a Command, flag: &str) -> Option<&'a Arg> {
    command
        .get_arguments()
        .find(|a| a.get_long().map(|long| format!("--{long}")).as_deref() == Some(flag))
}

fn takes_value(arg: &Arg) -> bool {
    matches!(arg.get_action(), ArgAction::Set | ArgAction::Append)
}

/// Whether two renderings of a default are the same number or the same text.
fn same_default(cli: &str, spec: &str) -> bool {
    if cli == spec {
        return true;
    }
    match (cli.parse::<f64>(), spec.parse::<f64>()) {
        (Ok(a), Ok(b)) => (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0),
        _ => false,
    }
}

/// Compare one subcommand with its spec; the failures, one line each.
fn compare<S: JobFields>(cli: &Command) -> Vec<String> {
    let name = S::subcommand();
    let Some(command) = cli.find_subcommand(name) else {
        return vec![format!("`pmetal {name}` does not exist")];
    };
    let mut problems = Vec::new();

    let cli_flags = flags_of(command);
    let spec_flags: BTreeSet<String> = S::argv_flags().iter().map(|f| f.to_string()).collect();
    let allowed = |table: &[(&str, &str, &str)], flag: &str| {
        table
            .iter()
            .any(|(cmd, listed, _)| *cmd == name && *listed == flag)
    };
    for flag in cli_flags.difference(&spec_flags) {
        if !allowed(CLI_ONLY, flag) {
            problems.push(format!(
                "`pmetal {name} {flag}` has no field on the spec: a form, a tool or an \
                 argv built from the spec cannot reach it"
            ));
        }
    }
    for flag in spec_flags.difference(&cli_flags) {
        if !allowed(SPEC_ONLY, flag) {
            problems.push(format!(
                "the spec spells `{flag}` but `pmetal {name}` does not take it"
            ));
        }
    }

    for descriptor in S::field_descriptors() {
        let Some(flag) = descriptor.argv else {
            continue;
        };
        let Some(arg) = argument(command, flag) else {
            continue;
        };
        if takes_value(arg) {
            let cli_default = arg
                .get_default_values()
                .first()
                .map(|v| v.to_string_lossy().into_owned());
            match (&descriptor.default, cli_default) {
                (DefaultValue::None, None) => {}
                (DefaultValue::None, Some(cli)) => problems.push(format!(
                    "`{name} {flag}` defaults to `{cli}` on the command line and to nothing \
                     on the spec"
                )),
                (spec, None) => problems.push(format!(
                    "`{name} {flag}` defaults to `{}` on the spec and to nothing on the \
                     command line",
                    spec.display()
                )),
                (spec, Some(cli)) => {
                    if !same_default(&cli, &spec.display()) {
                        problems.push(format!(
                            "`{name} {flag}` defaults to `{cli}` on the command line and to \
                             `{}` on the spec",
                            spec.display()
                        ));
                    }
                }
            }
            if let FieldKind::Enum { options } = descriptor.kind {
                let values: Vec<String> = arg
                    .get_possible_values()
                    .iter()
                    .map(|v| v.get_name().to_string())
                    .collect();
                if !values.is_empty() {
                    let mut cli_set = values.clone();
                    let mut spec_set: Vec<String> = options.iter().map(|o| o.to_string()).collect();
                    cli_set.sort();
                    spec_set.sort();
                    if cli_set != spec_set {
                        problems.push(format!(
                            "`{name} {flag}` accepts {cli_set:?} on the command line and \
                             {spec_set:?} on the spec"
                        ));
                    }
                }
            }
        } else if matches!(descriptor.default, DefaultValue::Bool(true)) {
            problems.push(format!(
                "`{name} {flag}` is a switch, which can only turn a behaviour on, but the \
                 spec defaults the field to true"
            ));
        }
    }
    problems
}

macro_rules! parity {
    ($test:ident, $spec:ty $(, $cfg:meta)?) => {
        $(#[cfg($cfg)])?
        #[test]
        fn $test() {
            let problems = compare::<$spec>(&Cli::command());
            assert!(problems.is_empty(), "\n{}", problems.join("\n"));
        }
    };
}

use pmetal_core::jobs::*;

parity!(train, TrainSpec, feature = "trainer");
parity!(train_mtp, TrainMtpSpec, feature = "trainer");
parity!(train_draft, TrainDraftSpec, feature = "trainer");
parity!(train_diffusion, TrainDiffusionSpec, feature = "trainer");
parity!(pretrain, PretrainSpec, feature = "trainer");
parity!(distill, DistillSpec, feature = "trainer");
parity!(grpo, GrpoSpec, feature = "trainer");
parity!(rlkd, RlkdSpec, feature = "trainer");
parity!(preference, PreferenceSpec, feature = "trainer");
parity!(embed_train, EmbedTrainSpec, feature = "trainer");
parity!(bench, BenchSpec, feature = "trainer");
parity!(infer, InferSpec);
parity!(decide, DecideSpec);
parity!(dflash, DflashSpec);
parity!(serve, ServeSpec, feature = "serve");
parity!(pack_experts, PackExpertsSpec);
parity!(quantize, QuantizeSpec);
parity!(tokenize, TokenizeSpec);
parity!(fuse, FuseSpec, feature = "lora");
parity!(eval, EvalSpec, all(feature = "lora", feature = "trainer"));
parity!(merge, MergeSpec, feature = "merge");
parity!(vocode, VocodeSpec, feature = "vocoder");
