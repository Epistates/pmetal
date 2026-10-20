//! PMetal MCP server: the PMetal toolkit over the Model Context Protocol.
//!
//! The server speaks MCP on stdio. Its tools come from three places, and none is
//! written twice:
//!
//! - **Job specs** (`pmetal-core`): training, inference, quantization and the rest
//!   are generated from the spec of the `pmetal` subcommand they run, so a tool's
//!   parameters, ranges, options and defaults are the CLI's ([`spec_tools`]).
//! - **CLI tables**: the dataset and Ollama commands, which have no spec
//!   ([`cli_tools`]).
//! - **Native tools**: the Hub, the model cache, job tracking and control, and a
//!   running `pmetal serve` ([`native`]).
//!
//! Anything that writes or reads a path goes through [`policy::PathPolicy`]; jobs
//! run under [`jobs::JobManager`], which bounds their memory and cleans them up.

pub mod catalog;
pub mod cli_tools;
pub mod jobs;
pub mod native;
pub mod policy;
pub mod results;
pub mod schema_check;
pub mod serve_client;
pub mod spec_tools;
pub mod staging;
pub mod tooldef;
pub mod util;

use std::collections::HashMap;
use std::sync::Arc;

use pmetal_core::jobs::{
    DflashSpec, DistillSpec, EmbedTrainSpec, EvalSpec, FuseSpec, GrpoSpec, InferSpec, MergeSpec,
    PackExpertsSpec, PreferenceSpec, PretrainSpec, QuantizeSpec, RlkdSpec, ServeSpec, TokenizeSpec,
    TrainDiffusionSpec, TrainDraftSpec, TrainMtpSpec, TrainSpec, VocodeSpec,
};
use serde_json::Value;
use turbomcp::prelude::*;
use turbomcp_types::{
    PromptsCapabilities, ResourcesCapabilities, ServerCapabilities, ToolsCapabilities,
};

use crate::jobs::{JobManager, JobRequest};
use crate::native::Core;
use crate::policy::PathPolicy;
use crate::results::{JobStarted, structured, text};
use crate::spec_tools::{Mode, SpecEntry, SpecTool, SpecToolConfig};
use crate::tooldef::Hints;

/// What a client is told about the server on `initialize`.
const INSTRUCTIONS: &str = "\
PMetal trains, runs and manages language models on this Mac (Apple Silicon).

Start with device_info and model_fit to see what fits. Long work (train, pretrain, distill, grpo, \
rlkd, preference, embed_train, train_diffusion, train_draft, train_mtp, quantize, fuse_lora, \
merge_models, pack_experts, tokenize, dflash, start_serve) runs as a background job: the tool \
returns a job_id at once. Follow it with job_status (state, step, loss), job_logs (output; pass \
`since` for only what is new) and stop_job. Steer a running training job with job_set_lr, \
job_reduce_lr, job_reset_lr, job_save_checkpoint and job_graceful_stop. When it finishes, \
inspect_output shows what it wrote.\n\n\
generate, eval_perplexity, bench_train, bench_gen, bench_corpus, decide and chat run to \
completion and return the result. The dataset_* tools prepare training data.\n\n\
Every tool's parameters are those of the matching `pmetal` command, with the same defaults: \
leave a parameter out to get the CLI's default. Paths must lie under the server's working \
directory, the temporary directory, the model cache, or a directory listed in PMETAL_MCP_ROOTS.";

/// Everything the server shares across requests.
struct Inner {
    core: Core,
    spec_tools: Vec<Box<dyn SpecTool>>,
    cli_tools: &'static [cli_tools::CliTool],
    native_tools: Vec<Tool>,
    /// Tool name → where it is defined, for dispatch.
    index: HashMap<String, Kind>,
}

#[derive(Clone, Copy)]
enum Kind {
    Spec(usize),
    Cli(usize),
    Native,
}

/// MCP server exposing all PMetal functionality.
#[derive(Clone)]
pub struct PmetalMcpServer {
    inner: Arc<Inner>,
}

macro_rules! spec_tool {
    ($spec:ty, $name:literal, $desc:literal, $mode:expr, $hints:expr) => {
        spec_tool!($spec, $name, $desc, $mode, $hints, docs: &[], paths: &[], check: None)
    };
    ($spec:ty, $name:literal, $desc:literal, $mode:expr, $hints:expr,
     docs: $docs:expr, paths: $paths:expr, check: $check:expr) => {
        Box::new(SpecEntry::<$spec>::new(SpecToolConfig {
            name: $name,
            description: $desc,
            mode: $mode,
            hints: $hints,
            extra_docs: $docs,
            extra_path_fields: $paths,
            check: $check,
        })) as Box<dyn SpecTool>
    };
}

/// A server may bind only to this machine unless `PMETAL_MCP_ALLOW_REMOTE_BIND`
/// is set: the inference server it starts has no authentication.
fn serve_host_check(fields: &serde_json::Map<String, Value>) -> Result<(), String> {
    let host = fields
        .get("host")
        .and_then(Value::as_str)
        .unwrap_or("127.0.0.1");
    let loopback = host == "localhost"
        || host
            .trim_matches(['[', ']'])
            .parse::<std::net::IpAddr>()
            .is_ok_and(|ip| ip.is_loopback());
    let allowed = std::env::var("PMETAL_MCP_ALLOW_REMOTE_BIND").is_ok_and(|v| v == "1");
    if loopback || allowed {
        Ok(())
    } else {
        Err(format!(
            "host: {host} would expose an unauthenticated inference server to the network. \
             Use 127.0.0.1, or set PMETAL_MCP_ALLOW_REMOTE_BIND=1 in the server's environment"
        ))
    }
}

/// The tool generated from each job spec.
pub fn spec_tool_list() -> Vec<Box<dyn SpecTool>> {
    const JOB: Mode = Mode::Job;
    const NOW: Mode = Mode::Blocking;
    vec![
        // Training.
        // `pmetal train` writes step metrics to `--log-metrics`, which its spec
        // does not model, so the server adds the flag.
        Box::new(SpecEntry::<TrainSpec>::new(
            SpecToolConfig::new(
                "train",
                "Start LoRA/QLoRA fine-tuning of a model (pmetal train). Returns a job ID: follow it with job_status and job_logs, steer it with job_set_lr and friends.",
                JOB,
                Hints::CREATE,
            )
            .docs(&[
                ("config_path", "YAML config file to load defaults from"),
                ("resume", "Resume from the latest checkpoint in the output directory"),
            ])
            .paths(&["config_path"]),
        )) as Box<dyn SpecTool>,
        spec_tool!(
            PretrainSpec, "pretrain",
            "Start full-parameter pretraining from tokenized binary shards (pmetal pretrain). Returns a job ID.",
            JOB, Hints::CREATE,
            docs: &[], paths: &["shards", "eval_shards"], check: None
        ),
        spec_tool!(
            DistillSpec, "distill",
            "Start knowledge distillation from a teacher to a student model (pmetal distill). Returns a job ID.",
            JOB, Hints::CREATE
        ),
        spec_tool!(
            GrpoSpec, "grpo",
            "Start Group Relative Policy Optimization for reasoning models (pmetal grpo). Returns a job ID.",
            JOB, Hints::CREATE
        ),
        spec_tool!(
            RlkdSpec, "rlkd",
            "Start Reinforcement Learning from Knowledge Distillation, GRPO combined with distillation from a teacher (pmetal rlkd). Returns a job ID.",
            JOB, Hints::CREATE
        ),
        spec_tool!(
            PreferenceSpec, "preference",
            "Start preference optimization with LoRA: DPO (Robust DPO with label smoothing), IPO, hinge, SimPO or ORPO on prompt/chosen/rejected pairs, or KTO on labelled completions (pmetal preference). Returns a job ID.",
            JOB, Hints::CREATE
        ),
        spec_tool!(
            EmbedTrainSpec, "embed_train",
            "Train a sentence-embedding model (BERT or other encoder) with contrastive losses (pmetal embed-train). Returns a job ID.",
            JOB, Hints::CREATE
        ),
        spec_tool!(
            TrainDiffusionSpec, "train_diffusion",
            "Start block-diffusion LoRA fine-tuning of DiffusionGemma (pmetal train-diffusion). Returns a job ID.",
            JOB, Hints::CREATE
        ),
        spec_tool!(
            TrainDraftSpec, "train_draft",
            "Start training a DFlash block-diffusion draft model for a target model (pmetal train-draft). Returns a job ID.",
            JOB, Hints::CREATE,
            docs: &[], paths: &["shards"], check: None
        ),
        spec_tool!(
            TrainMtpSpec, "train_mtp",
            "Start training a multi-token-prediction head or assistant for a model (pmetal train-mtp). Returns a job ID.",
            JOB, Hints::CREATE,
            docs: &[], paths: &["shards"], check: None
        ),
        spec_tool!(
            DflashSpec, "dflash",
            "Run block-diffusion speculative decoding with a target and a draft model (pmetal dflash). Returns a job ID.",
            JOB, Hints::CREATE
        ),
        // Model operations.
        spec_tool!(
            QuantizeSpec, "quantize",
            "Quantize a model to GGUF, from Q2_K to Q8_0 and dynamic quantization with importance matrices (pmetal quantize). Returns a job ID.",
            JOB, Hints::OVERWRITE
        ),
        spec_tool!(
            FuseSpec, "fuse_lora",
            "Fuse a LoRA adapter into its base model, writing a standalone model (pmetal fuse). Returns a job ID.",
            JOB, Hints::OVERWRITE
        ),
        spec_tool!(
            MergeSpec, "merge_models",
            "Merge two models with SLERP, TIES, DARE, linear or task arithmetic (pmetal merge). Returns a job ID.",
            JOB, Hints::OVERWRITE
        ),
        spec_tool!(
            PackExpertsSpec, "pack_experts",
            "Pack a mixture-of-experts model's expert weights for SSD-offloaded inference, to run models larger than memory (pmetal pack-experts). Returns a job ID.",
            JOB, Hints::OVERWRITE
        ),
        spec_tool!(
            TokenizeSpec, "tokenize",
            "Tokenize a JSONL text corpus into binary shards for pretraining (pmetal tokenize). Returns a job ID.",
            JOB, Hints::OVERWRITE
        ),
        spec_tool!(
            ServeSpec, "start_serve",
            "Start an OpenAI- and Anthropic-compatible inference server as a background job (pmetal serve). Returns a job ID; talk to it with chat. Binds to 127.0.0.1 by default.",
            JOB, Hints::CREATE,
            docs: &[], paths: &[], check: Some(serve_host_check)
        ),
        // Run to completion.
        spec_tool!(
            InferSpec, "generate",
            "Generate text with a model and return it (pmetal infer). Blocks until generation completes; best for short outputs. For longer sessions use start_serve and chat.",
            NOW, Hints::CREATE.idempotent(false)
        ),
        spec_tool!(
            VocodeSpec, "vocode",
            "Synthesize a waveform from a mel spectrogram or a WAV file with a BigVGAN neural vocoder, writing a 16-bit WAV (pmetal vocode). Blocks until done.",
            NOW, Hints::OVERWRITE
        ),
        spec_tool!(
            EvalSpec, "eval_perplexity",
            "Evaluate a model's perplexity on a dataset, optionally with a LoRA adapter (pmetal eval). Blocks until done.",
            NOW, Hints::READ
        ),
        spec_tool!(
            pmetal_core::jobs::BenchSpec, "bench_train",
            "Benchmark training throughput: tokens per second, memory, MFU (pmetal bench). Blocks until done.",
            NOW, Hints::READ.idempotent(false)
        ),
    ]
}

impl PmetalMcpServer {
    fn new() -> Self {
        Self::with(JobManager::new(), PathPolicy::from_env())
    }

    /// A server whose jobs run under `jobs` and whose paths are checked by
    /// `policy`.
    pub fn with(jobs: JobManager, policy: PathPolicy) -> Self {
        let spec_tools = spec_tool_list();
        let cli = cli_tools::CLI_TOOLS;
        let native_tools = native::definitions();
        let mut index = HashMap::new();
        for (i, tool) in spec_tools.iter().enumerate() {
            index.insert(tool.name().to_string(), Kind::Spec(i));
        }
        for (i, tool) in cli.iter().enumerate() {
            index.insert(tool.name.to_string(), Kind::Cli(i));
        }
        for tool in &native_tools {
            index.insert(tool.name.clone(), Kind::Native);
        }
        Self {
            inner: Arc::new(Inner {
                core: Core::new(jobs, policy),
                spec_tools,
                cli_tools: cli,
                native_tools,
                index,
            }),
        }
    }

    /// Stop the jobs still running.
    pub async fn shutdown(&self) {
        self.inner.core.jobs.shutdown().await;
    }

    /// The policy for one request: the server's own, plus the directories the
    /// client lists as its roots (MCP `roots/list`), when it has any. A client
    /// that does not support roots, or does not answer in a couple of seconds,
    /// adds none.
    async fn policy_for(&self, ctx: &RequestContext) -> PathPolicy {
        let base = &self.inner.core.policy;
        let roots = tokio::time::timeout(std::time::Duration::from_secs(2), ctx.list_roots()).await;
        match roots {
            Ok(Ok(roots)) => base.with_extra_roots(
                roots
                    .iter()
                    .filter_map(|root| policy::root_uri_to_path(&root.uri)),
            ),
            _ => base.clone(),
        }
    }

    async fn dispatch(
        &self,
        kind: Kind,
        name: &str,
        args: Value,
        ctx: &RequestContext,
    ) -> McpResult<ToolResult> {
        let inner = &*self.inner;
        let takes_paths = match kind {
            Kind::Spec(_) | Kind::Cli(_) => true,
            Kind::Native => native::takes_paths(name),
        };
        let policy = if takes_paths {
            self.policy_for(ctx).await
        } else {
            inner.core.policy.clone()
        };
        match kind {
            Kind::Spec(i) => {
                let tool = &inner.spec_tools[i];
                let prepared = tool.prepare(args, &policy)?;
                match tool.mode() {
                    Mode::Job => {
                        let output = prepared.output.as_ref().map(|p| p.display().to_string());
                        let id = inner
                            .core
                            .jobs
                            .spawn(JobRequest {
                                subcommand: prepared.subcommand.to_string(),
                                argv: prepared.argv,
                                output: prepared.output,
                                metrics_file: prepared.metrics_file,
                                ..JobRequest::default()
                            })
                            .await?;
                        structured(&JobStarted::new(&id, prepared.subcommand, output))
                    }
                    Mode::Blocking => {
                        let mut argv = vec![prepared.subcommand.to_string()];
                        argv.extend(prepared.argv);
                        inner.core.run(&argv, ctx).await.map(text)
                    }
                }
            }
            Kind::Cli(i) => {
                let tool = &inner.cli_tools[i];
                let argv = tool.argv(args, &policy)?;
                inner.core.run(&argv, ctx).await.map(text)
            }
            Kind::Native => {
                // Hold the call to the schema the tool advertised.
                if let Some(tool) = inner.native_tools.iter().find(|t| t.name == name) {
                    let schema = serde_json::to_value(&tool.input_schema)
                        .map_err(|e| McpError::internal(e.to_string()))?;
                    schema_check::check_arguments(name, &schema, &args)
                        .map_err(McpError::invalid_params)?;
                }
                native::call(&inner.core, &policy, name, args, ctx)
                    .await
                    .unwrap_or_else(|| Err(McpError::tool_not_found(name)))
            }
        }
    }
}

/// Start the MCP server on stdio. Returns when the client disconnects or the
/// process is asked to stop, after stopping the jobs it started.
pub async fn run_stdio() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let server = PmetalMcpServer::new();
    let result = tokio::select! {
        result = server.run_stdio() => result.map_err(|e| e.to_string()),
        () = termination() => Ok(()),
    };
    // Whichever way it ended, no job outlives the server.
    server.shutdown().await;
    result.map_err(Into::into)
}

/// Resolves when the process is sent SIGTERM or SIGINT.
async fn termination() {
    use tokio::signal::unix::{SignalKind, signal};
    let (Ok(mut term), Ok(mut int)) = (
        signal(SignalKind::terminate()),
        signal(SignalKind::interrupt()),
    ) else {
        // Without handlers the default action ends the process, as before.
        std::future::pending::<()>().await;
        return;
    };
    tokio::select! {
        _ = term.recv() => {}
        _ = int.recv() => {}
    }
}

impl McpHandler for PmetalMcpServer {
    fn server_info(&self) -> ServerInfo {
        ServerInfo::new("pmetal", env!("CARGO_PKG_VERSION"))
            .with_title("PMetal")
            .with_description(
                "PMetal ML toolkit for Apple Silicon: training, inference, quantization and model management",
            )
    }

    fn instructions(&self) -> Option<String> {
        Some(INSTRUCTIONS.to_string())
    }

    fn server_capabilities(&self) -> ServerCapabilities {
        // The tool list never changes while the server runs.
        ServerCapabilities {
            tools: Some(ToolsCapabilities {
                list_changed: Some(false),
            }),
            resources: Some(ResourcesCapabilities {
                subscribe: None,
                list_changed: Some(false),
            }),
            prompts: Some(PromptsCapabilities {
                list_changed: Some(false),
            }),
            logging: Some(Default::default()),
            ..ServerCapabilities::default()
        }
    }

    fn list_tools(&self) -> Vec<Tool> {
        let inner = &*self.inner;
        let mut tools: Vec<Tool> = inner.spec_tools.iter().map(|t| t.definition()).collect();
        tools.extend(inner.cli_tools.iter().map(cli_tools::CliTool::definition));
        tools.extend(inner.native_tools.iter().cloned());
        tools.sort_by(|a, b| a.name.cmp(&b.name));
        tools
    }

    fn list_resources(&self) -> Vec<Resource> {
        catalog::resources()
    }

    fn list_resource_templates(&self) -> Vec<ResourceTemplate> {
        catalog::resource_templates()
    }

    fn list_prompts(&self) -> Vec<Prompt> {
        catalog::prompts()
    }

    async fn call_tool<'a>(
        &'a self,
        name: &'a str,
        args: Value,
        ctx: &'a RequestContext,
    ) -> McpResult<ToolResult> {
        // An unknown tool is a protocol error; a tool that was found and failed,
        // for whatever reason, is a result the model can read and act on.
        let Some(&kind) = self.inner.index.get(name) else {
            return Err(McpError::tool_not_found(name));
        };
        Ok(match self.dispatch(kind, name, args, ctx).await {
            Ok(result) => result,
            Err(error) => {
                tracing::debug!("tool {name} failed: {error}");
                results::failure(&error)
            }
        })
    }

    async fn read_resource<'a>(
        &'a self,
        uri: &'a str,
        _ctx: &'a RequestContext,
    ) -> McpResult<ResourceResult> {
        catalog::read(&self.inner.core, uri)
    }

    async fn get_prompt<'a>(
        &'a self,
        name: &'a str,
        args: Option<Value>,
        _ctx: &'a RequestContext,
    ) -> McpResult<PromptResult> {
        catalog::get_prompt(name, args)
    }
}
