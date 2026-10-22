//! Per-subcommand clap argument structs.
//!
//! Each module contains a thin `<Subcommand>Args` struct that carries only clap
//! metadata.  See `README.md` in this directory for the full migration pattern.

use clap::{Parser, Subcommand};
#[cfg(feature = "trainer")]
use std::path::PathBuf;

#[cfg(feature = "trainer")]
use self::bench::{GdnBenchmarkStage, WorkloadBenchmarkPreset, WorkloadInferenceContext};
use self::dataset::DatasetAction;
use self::ollama::OllamaAction;

pub mod bench;
#[cfg(feature = "distributed")]
pub mod cluster;
pub mod dataset;
pub mod decide;
pub mod dflash;
pub mod distill;
pub mod embed_train;
pub mod eval;
pub mod fuse;
pub mod grpo;
pub mod hub;
pub mod infer;
pub mod merge;
pub mod ollama;
pub mod pack_experts;
pub mod preference;
pub mod pretrain;
pub mod quantize;
pub mod rlkd;
pub mod serve;
pub mod tokenize;
pub mod train;
#[cfg(feature = "trainer")]
pub mod train_diffusion;
#[cfg(feature = "trainer")]
pub mod train_draft;
#[cfg(feature = "trainer")]
pub mod train_mtp;
#[cfg(feature = "vocoder")]
pub mod vocode;

#[derive(Parser)]
#[command(name = "pmetal")]
#[command(author, version, about = "LLM fine-tuning optimized for Apple Silicon", long_about = None)]
pub struct Cli {
    /// Write every JobEvent of a training run as a JSONL line to this file.
    ///
    /// Works with train, distill, grpo, rlkd and preference; any other command
    /// refuses it, rather than leave an empty file. The TUI, the desktop app
    /// and the Python bindings follow a run through it. A write that fails is
    /// reported when the run ends.
    #[cfg(feature = "trainer")]
    #[arg(long, global = true, value_name = "PATH")]
    pub log_events: Option<PathBuf>,

    #[command(subcommand)]
    pub command: Commands,
}

impl Cli {
    /// Parse `args`, and say which flags of the chosen subcommand were given
    /// (spelled `--like-this`), as opposed to left at their defaults.
    ///
    /// A setting has three sources: a flag given, a config file, the default.
    /// A default must not override the file, and a flag given its default's
    /// value must; only the parser knows which is which (`clap`'s
    /// `value_source`), so it says so here for the commands that merge a file.
    pub fn parse_with_given_flags(
        args: impl IntoIterator<Item = std::ffi::OsString>,
    ) -> Result<(Self, std::collections::BTreeSet<String>), clap::Error> {
        use clap::{CommandFactory, FromArgMatches, parser::ValueSource};
        let command = Self::command();
        let matches = command.clone().try_get_matches_from(args)?;
        let cli = Self::from_arg_matches(&matches)?;
        let given = matches
            .subcommand()
            .and_then(|(name, sub)| Some((command.find_subcommand(name)?, sub)))
            .map(|(definition, sub)| {
                sub.ids()
                    .filter(|id| {
                        sub.value_source(id.as_str())
                            .is_some_and(|source| source != ValueSource::DefaultValue)
                    })
                    .filter_map(|id| {
                        let argument = definition.get_arguments().find(|a| a.get_id() == id)?;
                        Some(format!("--{}", argument.get_long()?))
                    })
                    .collect()
            })
            .unwrap_or_default();
        Ok((cli, given))
    }

    /// [`parse_with_given_flags`](Self::parse_with_given_flags) on the process's
    /// own arguments; a usage error or `--help` ends the process as clap does.
    pub fn parse_from_env() -> (Self, std::collections::BTreeSet<String>) {
        use clap::CommandFactory;
        Self::parse_with_given_flags(std::env::args_os())
            .unwrap_or_else(|e| e.format(&mut Self::command()).exit())
    }
}

#[derive(Subcommand, Debug)]
#[allow(clippy::large_enum_variant)]
pub enum Commands {
    /// Fine-tune a model using LoRA/QLoRA
    #[cfg(feature = "trainer")]
    Train(crate::cli::train::TrainArgs),

    /// Train a Gemma 4 assistant or Qwen3Next/Qwen3.6 MTP predictor
    #[command(name = "train-mtp")]
    #[cfg(feature = "trainer")]
    TrainMtp(crate::cli::train_mtp::TrainMtpArgs),

    /// Train a DFlash block-diffusion draft checkpoint
    #[command(name = "train-draft")]
    #[cfg(feature = "trainer")]
    TrainDraft(crate::cli::train_draft::TrainDraftArgs),

    /// LoRA fine-tune a DiffusionGemma block-diffusion model
    #[command(name = "train-diffusion")]
    #[cfg(feature = "trainer")]
    TrainDiffusion(crate::cli::train_diffusion::TrainDiffusionArgs),

    /// Pretrain a model from scratch (full-parameter, no LoRA)
    #[cfg(feature = "trainer")]
    Pretrain(crate::cli::pretrain::PretrainArgs),

    /// Run inference with a model
    Infer(crate::cli::infer::InferArgs),

    /// Answer typed questions about a state with a decision model (Clef)
    Decide(crate::cli::decide::DecideArgs),

    /// Download a model from HuggingFace
    Download(hub::DownloadArgs),

    /// Publish a model directory to HuggingFace (creates the repo if needed)
    Upload(hub::UploadArgs),

    /// Check that a model directory is complete and undamaged
    Verify(hub::VerifyArgs),

    /// Print the shell completion script (bash, zsh, fish, elvish, powershell)
    Completions(hub::CompletionsArgs),

    /// Search HuggingFace Hub for models and show device fit
    Search(hub::SearchArgs),

    /// Block-diffusion speculative decoding (DFlash).
    ///
    /// Runs a Qwen3 target alongside a DFlash draft to emit multiple tokens
    /// per forward pass with bit-identical output to greedy baseline.
    Dflash(crate::cli::dflash::DflashArgs),

    /// Show memory usage and available capacity
    Memory {
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Benchmark generation loop timing (detailed profiling)
    #[cfg(feature = "trainer")]
    BenchGen {
        /// Model to benchmark
        #[arg(short, long, default_value = "Qwen/Qwen3-0.6B")]
        model: String,
    },

    /// Benchmark training performance
    #[cfg(feature = "trainer")]
    Bench(crate::cli::bench::BenchArgs),

    /// Run a structured kernel benchmark corpus for this device tier
    #[cfg(feature = "trainer")]
    BenchCorpus {
        /// Use a shorter run with fewer iterations and smaller tier-scaled shapes
        #[arg(long)]
        quick: bool,

        /// Output results as JSON
        #[arg(long)]
        json: bool,

        /// Optional output path for the JSON report
        #[arg(short, long)]
        output: Option<String>,
    },

    /// Benchmark a real cached workload for inference and short LoRA training
    #[cfg(feature = "trainer")]
    BenchWorkload {
        /// Named preset that overrides the model/dataset/shape knobs below
        #[arg(long, value_enum)]
        preset: Option<WorkloadBenchmarkPreset>,

        /// Model ID or local path
        #[arg(long, default_value = "Qwen/Qwen3-0.6B")]
        model: String,

        /// Dataset ID or local path
        #[arg(
            long,
            default_value = "TeichAI/gemini-3-pro-preview-high-reasoning-250x"
        )]
        dataset: String,

        /// Packed expert directory for sparse/offloaded inference workloads
        #[arg(long)]
        experts_dir: Option<String>,

        /// Number of prompt samples to benchmark for inference
        #[arg(long, default_value = "8")]
        prompt_samples: usize,

        /// Maximum prompt tokens per inference sample (0 = auto-select from sampled prompts, capped for a quick run)
        #[arg(long, default_value = "0")]
        max_prompt_tokens: usize,

        /// Inference context source for workload benchmarking
        #[arg(long, value_enum, default_value = "auto")]
        inference_context: WorkloadInferenceContext,

        /// Number of decode steps per prompt sample
        #[arg(long, default_value = "32")]
        decode_steps: usize,

        /// Number of untimed warmup passes to run per sampled inference prompt before measurement
        #[arg(long, default_value = "2")]
        inference_warmup_passes: usize,

        /// Number of independent warmed inference sessions to run and aggregate
        #[arg(long, default_value = "1")]
        inference_session_repeats: usize,

        /// Number of timed inference passes per prompt sample
        #[arg(long, default_value = "1")]
        inference_repeats: usize,

        /// Number of raw samples to include in the sampled training subset
        #[arg(long, default_value = "8")]
        train_samples: usize,

        /// Number of LoRA training steps to run
        #[arg(long, default_value = "4")]
        train_steps: usize,

        /// Training batch size
        #[arg(long, default_value = "1")]
        batch_size: usize,

        /// Maximum sequence length for the short training run (0 = auto-select from sampled data, capped for a quick run)
        #[arg(long, default_value = "0")]
        max_seq_len: usize,

        /// Output results as JSON
        #[arg(long)]
        json: bool,

        /// Optional output path for the JSON report
        #[arg(short, long)]
        output: Option<String>,
    },

    /// Benchmark Qwen3.5 GDN backends on the actual model layer shapes
    #[cfg(feature = "trainer")]
    BenchGdn {
        /// Qwen3.5/Qwen3Next model ID or local path
        #[arg(long, default_value = "Qwen/Qwen3.5-0.8B")]
        model: String,

        /// Projection stage to benchmark
        #[arg(long, value_enum, default_value = "input-proj")]
        stage: GdnBenchmarkStage,

        /// Optional transformer layer index to benchmark (defaults to the first GDN layer)
        #[arg(long)]
        layer: Option<usize>,

        /// Batch size for the synthetic benchmark input
        #[arg(long, default_value = "1")]
        batch_size: usize,

        /// Sequence length for the synthetic input (use a longer value for prefill benchmarking)
        #[arg(long, default_value = "1")]
        seq_len: usize,

        /// Warmup iterations per backend
        #[arg(long, default_value = "10")]
        warmup_iterations: usize,

        /// Timed iterations per backend
        #[arg(long, default_value = "50")]
        benchmark_iterations: usize,

        /// Output results as JSON
        #[arg(long)]
        json: bool,

        /// Optional output path for the JSON report
        #[arg(short, long)]
        output: Option<String>,
    },

    /// Generate a sample configuration file
    #[cfg(feature = "trainer")]
    Init {
        /// Output path for the config file
        #[arg(short, long, default_value = "config.yaml")]
        output: String,
    },

    /// Pack expert weights for SSD-offloaded MoE inference
    PackExperts(crate::cli::pack_experts::PackExpertsArgs),

    /// Export trained model for Ollama
    Ollama {
        #[command(subcommand)]
        action: OllamaAction,
    },

    /// Fuse LoRA adapter weights into a base model and save as a complete model
    #[cfg(feature = "lora")]
    Fuse(crate::cli::fuse::FuseArgs),

    /// Quantize a model to GGUF format (supports Dynamic 2.0 and KL-calibrated)
    Quantize(crate::cli::quantize::QuantizeArgs),

    /// Knowledge Distillation from teacher to student
    #[cfg(feature = "trainer")]
    Distill(crate::cli::distill::DistillArgs),

    /// Group Relative Policy Optimization (GRPO) for reasoning models
    #[cfg(feature = "trainer")]
    Grpo(crate::cli::grpo::GrpoArgs),

    /// RLKD: Reinforcement Learning with Knowledge Distillation.
    ///
    /// Combines GRPO policy gradient optimization with knowledge distillation
    /// from a teacher model in a single training loop.
    ///
    /// Loss formula: L = (1 - alpha) * L_grpo + alpha * L_distill
    #[cfg(feature = "trainer")]
    Rlkd(crate::cli::rlkd::RlkdArgs),

    /// Preference optimization with LoRA: DPO, IPO, hinge, SimPO and ORPO on
    /// prompt/chosen/rejected pairs, or KTO on completions labelled good or bad.
    #[cfg(feature = "trainer")]
    #[command(visible_alias = "dpo")]
    Preference(crate::cli::preference::PreferenceArgs),

    /// Start MCP server for Claude Desktop integration
    #[cfg(feature = "mcp")]
    Mcp,

    /// Start an OpenAI-compatible inference server
    #[cfg(feature = "serve")]
    Serve(crate::cli::serve::ServeArgs),

    /// Dataset utilities for preparing and analyzing training data
    Dataset {
        #[command(subcommand)]
        action: DatasetAction,
    },

    /// Tokenize a text corpus into binary shards for pretraining
    Tokenize(crate::cli::tokenize::TokenizeArgs),

    /// Real-time training dashboard (loss curves, ANE utilization, timing)
    Dashboard {
        /// Path to training metrics JSONL file to visualize
        #[arg(short, long)]
        metrics_file: Option<String>,
    },

    /// Full TUI control center (device, models, datasets, training, inference)
    #[cfg(feature = "dashboard")]
    Tui {
        /// Path to training metrics JSONL file to visualize in the dashboard tab
        #[arg(short, long)]
        metrics_file: Option<String>,
    },

    /// Show device information (GPU architecture, ANE cores, bandwidth, NAX support)
    Info {
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Merge two models with SLERP, TIES, DARE, linear, task arithmetic and other methods
    #[cfg(feature = "merge")]
    Merge(crate::cli::merge::MergeArgs),

    /// Synthesize a waveform from a mel spectrogram with a neural vocoder (BigVGAN)
    #[cfg(feature = "vocoder")]
    Vocode(crate::cli::vocode::VocodeArgs),

    /// Evaluate a model's perplexity on a dataset
    #[cfg(all(feature = "lora", feature = "trainer"))]
    Eval(crate::cli::eval::EvalArgs),

    /// Train a sentence embedding model (BERT / encoder-only) with contrastive losses.
    ///
    /// Supports InfoNCE (default), triplet, CoSENT, and cosine-similarity losses.
    /// Input data must be JSONL with pair or triplet format.
    ///
    /// Pair JSONL:
    ///   {"text_a": "...", "text_b": "...", "label": 0.9}
    ///   {"sentence1": "...", "sentence2": "..."}
    ///   {"query": "...", "positive": "..."}
    ///
    /// Triplet JSONL:
    ///   {"anchor": "...", "positive": "...", "negative": "..."}
    #[command(name = "embed-train")]
    #[cfg(feature = "trainer")]
    EmbedTrain(crate::cli::embed_train::EmbedTrainArgs),

    /// Multi-machine cluster operations: discover peers, form a Thunderbolt-
    /// preferred ring, and run distributed training or inference.
    #[cfg(feature = "distributed")]
    Cluster(crate::cli::cluster::ClusterArgs),
}

impl Commands {
    /// Whether this command writes the `--log-events` file. A spec says so
    /// itself (`#[spec(events)]`, `JobFields::REPORTS_EVENTS`): the same fact
    /// every surface that starts a job asks the file by, so the one list is the
    /// specs'. `main` refuses the flag for a command that writes no event log.
    #[cfg(feature = "trainer")]
    pub fn writes_event_log(&self) -> bool {
        use pmetal_core::JobFields;
        use pmetal_core::jobs::{DistillSpec, GrpoSpec, PreferenceSpec, RlkdSpec, TrainSpec};
        match self {
            Commands::Train(_) => TrainSpec::REPORTS_EVENTS,
            Commands::Distill(_) => DistillSpec::REPORTS_EVENTS,
            Commands::Grpo(_) => GrpoSpec::REPORTS_EVENTS,
            Commands::Rlkd(_) => RlkdSpec::REPORTS_EVENTS,
            Commands::Preference(_) => PreferenceSpec::REPORTS_EVENTS,
            _ => false,
        }
    }

    /// The commands that write the `--log-events` file, by name, for the
    /// message that refuses the flag to the others.
    #[cfg(feature = "trainer")]
    pub fn event_log_commands() -> Vec<&'static str> {
        use pmetal_core::JobFields;
        use pmetal_core::jobs::{DistillSpec, GrpoSpec, PreferenceSpec, RlkdSpec, TrainSpec};
        [
            (TrainSpec::subcommand(), TrainSpec::REPORTS_EVENTS),
            (DistillSpec::subcommand(), DistillSpec::REPORTS_EVENTS),
            (GrpoSpec::subcommand(), GrpoSpec::REPORTS_EVENTS),
            (RlkdSpec::subcommand(), RlkdSpec::REPORTS_EVENTS),
            (PreferenceSpec::subcommand(), PreferenceSpec::REPORTS_EVENTS),
        ]
        .into_iter()
        .filter_map(|(name, writes)| writes.then_some(name))
        .collect()
    }

    /// Whether this command runs anything on MLX. The ones that don't skip the
    /// metallib search and the device check in `main`, so a missing or broken
    /// Metal library (or no network for the download fallback) can't stop
    /// them. A new command counts as using MLX until it is listed here.
    pub fn uses_mlx(&self) -> bool {
        match self {
            #[cfg(feature = "trainer")]
            Commands::Init { .. } => false,
            Commands::Download(_)
            | Commands::Upload(_)
            | Commands::Verify(_)
            | Commands::Completions(_)
            | Commands::Search(_)
            | Commands::Dataset { .. }
            | Commands::Tokenize(_) => false,
            _ => true,
        }
    }
}

#[cfg(test)]
mod doc_examples;

#[cfg(all(test, feature = "trainer"))]
mod given_flags_tests {
    use super::Cli;

    fn given(args: &[&str]) -> Vec<String> {
        let mut full = vec!["pmetal"];
        full.extend_from_slice(args);
        let (_, flags) =
            Cli::parse_with_given_flags(full.into_iter().map(Into::into)).expect("parses");
        flags.into_iter().collect()
    }

    /// What the parser says was given is what the config file yields to:
    /// a flag given its default's value counts, a default does not.
    #[test]
    fn a_flag_given_is_told_from_a_default() {
        assert_eq!(
            given(&["train", "-m", "m", "-d", "d"]),
            ["--dataset", "--model"]
        );
        assert_eq!(
            given(&["train", "-m", "m", "-d", "d", "--epochs", "1", "--dora"]),
            ["--dataset", "--dora", "--epochs", "--model"]
        );
        // Whatever the subcommand has none of is nothing, and so is no subcommand.
        assert!(given(&["info"]).is_empty());
    }

    /// Every flag of `train` that `FILE_KEYS` names is a real flag the parser
    /// reports when given, so a key can never be waiting on a flag that does
    /// not exist.
    #[test]
    fn every_file_key_flag_is_a_flag_of_train() {
        use clap::CommandFactory;
        let command = Cli::command();
        let train = command.find_subcommand("train").unwrap();
        for (flag, _, _) in pmetal_core::jobs::FILE_KEYS {
            assert!(
                train
                    .get_arguments()
                    .any(|a| a.get_long().map(|l| format!("--{l}")).as_deref() == Some(*flag)),
                "{flag}"
            );
        }
    }
}

#[cfg(all(test, feature = "trainer"))]
mod event_log_tests {
    use super::Cli;
    use clap::Parser;

    fn writes_event_log(argv: &[&str]) -> bool {
        let mut full = vec!["pmetal"];
        full.extend_from_slice(argv);
        full.extend(["--log-events", "events.jsonl"]);
        Cli::try_parse_from(full)
            .expect("the argv parses")
            .command
            .writes_event_log()
    }

    #[test]
    fn only_the_commands_that_write_an_event_log_accept_the_flag() {
        for argv in [
            &["train", "-m", "m", "-d", "d"][..],
            &["distill", "-t", "t", "-s", "s", "-d", "d"],
            &["grpo", "-m", "m", "-d", "d"],
            &["rlkd", "-m", "m", "--teacher-model", "t", "-d", "d"],
            &["preference", "-m", "m", "-d", "d"],
        ] {
            assert!(writes_event_log(argv), "{argv:?}");
        }
        for argv in [
            &["pretrain", "-a", "llama", "-s", "a.bin"][..],
            &["embed-train", "-m", "m", "-d", "d"],
            &["train-mtp", "-m", "m", "-s", "a.bin"],
            &["train-draft", "--target", "t", "-s", "a.bin"],
            &["train-diffusion", "--model", "m", "--dataset", "d"],
            &["info"],
        ] {
            assert!(!writes_event_log(argv), "{argv:?}");
        }
    }

    #[test]
    fn pretrain_takes_several_shards_as_a_glob_expands() {
        let cli = Cli::try_parse_from([
            "pmetal",
            "pretrain",
            "--arch",
            "llama",
            "--shards",
            "a.bin",
            "b.bin",
            "c.bin,d.bin",
            "--eval-shards",
            "e.bin",
            "f.bin",
        ])
        .expect("several shards parse");
        let super::Commands::Pretrain(args) = cli.command else {
            panic!("not pretrain");
        };
        assert_eq!(args.shards, ["a.bin", "b.bin", "c.bin", "d.bin"]);
        assert_eq!(args.eval_shards, ["e.bin", "f.bin"]);
    }
}

#[cfg(all(test, feature = "trainer", feature = "core"))]
mod argv_roundtrip;
#[cfg(all(test, feature = "mcp", feature = "trainer"))]
mod mcp_tools;
#[cfg(test)]
mod parity;
#[cfg(all(test, feature = "mcp", feature = "trainer", feature = "dashboard"))]
mod surfaces;
