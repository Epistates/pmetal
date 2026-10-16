//! The server's resources and prompts.
//!
//! Resources let a client read state without a tool call: the device, and every
//! job's summary and output. Prompts are workflows a user can start from the
//! client's menu; each one tells the model which tools to use, in what order,
//! for a job the server can do.

use serde_json::Value;
use turbomcp::prelude::*;

use crate::native::Core;
use crate::util;

pub const DEVICE_URI: &str = "pmetal://device";
pub const JOBS_URI: &str = "pmetal://jobs";
const JOB_TEMPLATE: &str = "pmetal://jobs/{job_id}";
const JOB_LOGS_TEMPLATE: &str = "pmetal://jobs/{job_id}/logs";

/// Lines of a job's output a resource read returns.
const LOG_LINES: usize = 200;

pub fn resources() -> Vec<Resource> {
    vec![
        Resource::new(DEVICE_URI, "device")
            .with_description(
                "This Mac's GPU, Neural Engine, memory and bandwidth, as device_info reports them.",
            )
            .with_mime_type("application/json"),
        Resource::new(JOBS_URI, "jobs")
            .with_description("Every background job with its status and metrics, newest first.")
            .with_mime_type("application/json"),
    ]
}

pub fn resource_templates() -> Vec<ResourceTemplate> {
    vec![
        ResourceTemplate::new(JOB_TEMPLATE, "job")
            .with_description("One background job: status, exit code, timing and metrics."),
        ResourceTemplate::new(JOB_LOGS_TEMPLATE, "job-logs").with_description(
            "The most recent output of a background job, stdout and stderr in order.",
        ),
    ]
}

/// Read the resource at `uri`.
pub fn read(core: &Core, uri: &str) -> McpResult<ResourceResult> {
    if uri == DEVICE_URI {
        return json_resource(uri, &util::build_device_info_json()?);
    }
    if uri == JOBS_URI {
        return json_resource(
            uri,
            &serde_json::json!({ "jobs": core.jobs.list_summaries() }),
        );
    }
    let rest = uri
        .strip_prefix("pmetal://jobs/")
        .ok_or_else(|| McpError::resource_not_found(uri))?;
    match rest.split_once('/') {
        None => {
            let summary = core
                .jobs
                .get_summary(rest)
                .map_err(|_| McpError::resource_not_found(uri))?;
            json_resource(uri, &summary)
        }
        Some((id, "logs")) => {
            let page = core
                .jobs
                .get_logs(id, LOG_LINES, None)
                .map_err(|_| McpError::resource_not_found(uri))?;
            Ok(ResourceResult::text(uri, page.lines.join("\n")))
        }
        Some(_) => Err(McpError::resource_not_found(uri)),
    }
}

fn json_resource<T: serde::Serialize>(uri: &str, value: &T) -> McpResult<ResourceResult> {
    ResourceResult::json(uri, value).map_err(|e| McpError::internal(e.to_string()))
}

pub fn prompts() -> Vec<Prompt> {
    vec![
        Prompt::new(
            "finetune",
            "Fine-tune a model on a dataset with LoRA: check it fits, prepare the data, train, watch the run, evaluate.",
        )
        .with_required_arg("model", "Model ID or local path to fine-tune")
        .with_required_arg("dataset", "Path to the training data (JSONL) or a Hugging Face dataset ID")
        .with_optional_arg("goal", "What the tuned model should do better"),
        Prompt::new(
            "evaluate_adapter",
            "Compare a LoRA adapter with its base model: perplexity on a held-out set, then a few generations.",
        )
        .with_required_arg("model", "Base model ID or local path")
        .with_required_arg("adapter", "Path to the trained adapter")
        .with_required_arg("dataset", "Held-out evaluation dataset (JSONL)"),
        Prompt::new(
            "export_for_ollama",
            "Fuse an adapter into its base model, quantize it to GGUF, and write the Ollama Modelfile.",
        )
        .with_required_arg("model", "Base model ID or local path")
        .with_required_arg("adapter", "Path to the trained adapter")
        .with_required_arg("output", "Directory to put the result in"),
    ]
}

pub fn get_prompt(name: &str, args: Option<Value>) -> McpResult<PromptResult> {
    let arg = |key: &str| -> McpResult<String> {
        args.as_ref()
            .and_then(|a| a.get(key))
            .and_then(Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| McpError::invalid_params(format!("the prompt needs `{key}`")))
    };
    let optional = |key: &str| {
        args.as_ref()
            .and_then(|a| a.get(key))
            .and_then(Value::as_str)
            .map(str::to_owned)
    };
    let text = match name {
        "finetune" => {
            let (model, dataset) = (arg("model")?, arg("dataset")?);
            let goal = optional("goal")
                .map(|g| format!(" The goal: {g}."))
                .unwrap_or_default();
            format!(
                "Fine-tune {model} on {dataset} with LoRA on this Mac.{goal}\n\n\
                 1. Call model_fit for {model} to see that training fits in memory, and note the recommended batch size.\n\
                 2. Call dataset_validate on the data, and dataset_analyze for its length statistics. Fix what they report with the dataset tools.\n\
                 3. Call train with the model and dataset. It returns a job_id.\n\
                 4. Poll job_status for the loss and step, and job_logs for output. If the loss spikes or stalls, adjust with job_reduce_lr or job_set_lr; use job_save_checkpoint before risky changes.\n\
                 5. When the job completes, call inspect_output on its output directory, then eval_perplexity with the adapter on a held-out set.\n\
                 Report the final loss, the perplexity before and after, and where the adapter is."
            )
        }
        "evaluate_adapter" => {
            let (model, adapter, dataset) = (arg("model")?, arg("adapter")?, arg("dataset")?);
            format!(
                "Evaluate the adapter at {adapter} against its base model {model} on {dataset}.\n\n\
                 1. Call eval_perplexity on {model} and {dataset} for the base figure.\n\
                 2. Call eval_perplexity again with lora set to {adapter}.\n\
                 3. Call generate with and without the adapter on two or three prompts from the dataset and compare the answers.\n\
                 Report both perplexities and say whether the adapter helps."
            )
        }
        "export_for_ollama" => {
            let (model, adapter, output) = (arg("model")?, arg("adapter")?, arg("output")?);
            format!(
                "Export {model} with the adapter {adapter} for Ollama into {output}.\n\n\
                 1. Call quantize with model {model}, lora {adapter} and an output GGUF path in {output}.\n\
                 2. Call ollama_create with the GGUF as base to write the Modelfile.\n\
                 3. Tell the user the `ollama create` command to run."
            )
        }
        other => return Err(McpError::prompt_not_found(other)),
    };
    Ok(PromptResult::user(text))
}
