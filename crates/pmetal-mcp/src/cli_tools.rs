//! Tools for `pmetal` subcommands that have no job spec.
//!
//! The dataset and Ollama commands take flags the CLI defines and nothing else
//! describes, so each tool is a table row: its parameters, how each becomes an
//! argument, and what it may touch. The tool's schema and the command line are
//! both generated from the row. Defaults are the CLI's: a parameter the caller
//! leaves out is not passed.
//!
//! Every row is checked against the real CLI parser by a test in the `pmetal`
//! crate (`mcp_tools_parse`), because the flags here have to match flags that
//! live elsewhere.

use serde_json::{Map, Value, json};
use turbomcp::prelude::*;

use crate::policy::PathPolicy;
use crate::schema_check;
use crate::tooldef::Hints;

/// The type of a parameter.
#[derive(Debug, Clone, Copy)]
pub enum Ty {
    Text,
    /// A filesystem path, which must lie under an allowed root.
    Path,
    /// A Hub ID or a path.
    Reference,
    /// Several paths.
    Paths,
    Int(i64, i64),
    Num(f64, f64),
    Bool,
    Enum(&'static [&'static str]),
}

/// How a parameter reaches the command line.
#[derive(Debug, Clone, Copy)]
pub enum Flag {
    /// `--flag value`.
    Option(&'static str),
    /// `--flag`, given when the parameter is true.
    Switch(&'static str),
    /// A bare argument, in the order the parameters are declared.
    Positional,
}

#[derive(Debug)]
pub struct Param {
    pub name: &'static str,
    pub doc: &'static str,
    pub ty: Ty,
    pub flag: Flag,
    pub required: bool,
}

/// A tool that runs `pmetal <command…>` and returns what it prints.
#[derive(Debug)]
pub struct CliTool {
    pub name: &'static str,
    pub description: &'static str,
    pub command: &'static [&'static str],
    pub params: &'static [Param],
    pub hints: Hints,
}

const fn p(name: &'static str, doc: &'static str, ty: Ty, flag: Flag, required: bool) -> Param {
    Param {
        name,
        doc,
        ty,
        flag,
        required,
    }
}

const SEED: Param = p(
    "seed",
    "Random seed (the CLI's default is 42)",
    Ty::Int(0, i64::MAX),
    Flag::Option("--seed"),
    false,
);

const TEMPLATES: &[&str] = &[
    "chatml", "llama3", "llama2", "mistral", "qwen", "phi", "gemma", "raw", "auto",
];

const OLLAMA_TEMPLATES: &[&str] = &["llama3", "qwen3", "gemma", "mistral", "phi3", "deep-seek"];

pub const CLI_TOOLS: &[CliTool] = &[
    CliTool {
        name: "dataset_analyze",
        description: "Analyze a dataset: sample count, format detection, character and token length statistics.",
        command: &["dataset", "analyze"],
        params: &[
            p(
                "path",
                "Path to the dataset file or directory (JSONL, JSON, Parquet, Arrow, CSV)",
                Ty::Path,
                Flag::Option("--path"),
                true,
            ),
            p(
                "model",
                "Model ID or path whose tokenizer counts tokens and whose chat template renders conversations",
                Ty::Reference,
                Flag::Option("--model"),
                false,
            ),
            p(
                "detailed",
                "Show per-sample statistics",
                Ty::Bool,
                Flag::Switch("--detailed"),
                false,
            ),
        ],
        hints: Hints::READ.open_world(),
    },
    CliTool {
        name: "dataset_preview",
        description: "Preview the first samples of a Hugging Face dataset.",
        command: &["dataset", "preview"],
        params: &[
            p(
                "dataset_id",
                "Hugging Face dataset ID (e.g. tatsu-lab/alpaca)",
                Ty::Reference,
                Flag::Positional,
                true,
            ),
            p(
                "split",
                "Dataset split (the CLI's default is train)",
                Ty::Text,
                Flag::Option("--split"),
                false,
            ),
            p(
                "num",
                "Number of samples to preview (the CLI's default is 5)",
                Ty::Int(1, 1000),
                Flag::Option("--num"),
                false,
            ),
        ],
        hints: Hints::READ.open_world(),
    },
    CliTool {
        name: "dataset_validate",
        description: "Validate a JSONL dataset for training: format, encoding and sequence lengths. Reports the problems it finds.",
        command: &["dataset", "validate"],
        params: &[
            p(
                "path",
                "Path to the dataset file (JSONL)",
                Ty::Path,
                Flag::Option("--path"),
                true,
            ),
            p(
                "model",
                "Model ID or path whose tokenizer measures lengths",
                Ty::Reference,
                Flag::Option("--model"),
                false,
            ),
            p(
                "max_seq_len",
                "Longest sequence to allow (the CLI's default is 2048)",
                Ty::Int(1, 4_194_304),
                Flag::Option("--max-seq-len"),
                false,
            ),
        ],
        hints: Hints::READ.open_world(),
    },
    CliTool {
        name: "dataset_download",
        description: "Download a dataset from the Hugging Face Hub and convert it to JSONL.",
        command: &["dataset", "download"],
        params: &[
            p(
                "dataset_id",
                "Hugging Face dataset ID",
                Ty::Reference,
                Flag::Positional,
                true,
            ),
            p(
                "split",
                "Dataset split (the CLI's default is train)",
                Ty::Text,
                Flag::Option("--split"),
                false,
            ),
            p(
                "output",
                "Where to write the JSONL file",
                Ty::Path,
                Flag::Option("--output"),
                false,
            ),
            p(
                "revision",
                "Branch, tag or commit",
                Ty::Text,
                Flag::Option("--revision"),
                false,
            ),
        ],
        hints: Hints::OVERWRITE.open_world(),
    },
    CliTool {
        name: "dataset_convert",
        description: "Convert a dataset file (Parquet, Arrow, JSON, JSONL, CSV) to JSONL, optionally renaming columns.",
        command: &["dataset", "convert"],
        params: &[
            p(
                "input",
                "Input file",
                Ty::Path,
                Flag::Option("--input"),
                true,
            ),
            p(
                "output",
                "Output JSONL file",
                Ty::Path,
                Flag::Option("--output"),
                true,
            ),
            p(
                "columns",
                "Rename columns as target=source pairs, e.g. 'text=content,prompt=instruction' takes text from the column content",
                Ty::Text,
                Flag::Option("--columns"),
                false,
            ),
            p(
                "shuffle",
                "Shuffle the output",
                Ty::Bool,
                Flag::Switch("--shuffle"),
                false,
            ),
            SEED,
        ],
        hints: Hints::OVERWRITE,
    },
    CliTool {
        name: "dataset_filter",
        description: "Filter a JSONL dataset by token length, duplicates, a pattern, or completeness.",
        command: &["dataset", "filter"],
        params: &[
            p(
                "input",
                "Input dataset (a file or directory: JSONL, JSON, Parquet, Arrow, CSV)",
                Ty::Path,
                Flag::Option("--input"),
                true,
            ),
            p(
                "output",
                "Output dataset (JSONL)",
                Ty::Path,
                Flag::Option("--output"),
                true,
            ),
            p(
                "model",
                "Model ID or path whose tokenizer counts tokens (needed for min_tokens and max_tokens)",
                Ty::Reference,
                Flag::Option("--model"),
                false,
            ),
            p(
                "min_tokens",
                "Fewest tokens a sample may have",
                Ty::Int(0, i64::MAX),
                Flag::Option("--min-tokens"),
                false,
            ),
            p(
                "max_tokens",
                "Most tokens a sample may have",
                Ty::Int(0, i64::MAX),
                Flag::Option("--max-tokens"),
                false,
            ),
            p(
                "dedup",
                "Remove exact duplicates",
                Ty::Bool,
                Flag::Switch("--dedup"),
                false,
            ),
            p(
                "pattern",
                "Regular expression; keeps the samples that match",
                Ty::Text,
                Flag::Option("--pattern"),
                false,
            ),
            p(
                "invert",
                "Keep the samples that do not match pattern",
                Ty::Bool,
                Flag::Switch("--invert"),
                false,
            ),
            p(
                "complete_only",
                "Drop conversations with missing turns",
                Ty::Bool,
                Flag::Switch("--complete-only"),
                false,
            ),
        ],
        hints: Hints::OVERWRITE.open_world(),
    },
    CliTool {
        name: "dataset_split",
        description: "Split a JSONL dataset into train, validation and test files.",
        command: &["dataset", "split"],
        params: &[
            p(
                "input",
                "Input dataset (a file or directory: JSONL, JSON, Parquet, Arrow, CSV)",
                Ty::Path,
                Flag::Option("--input"),
                true,
            ),
            p(
                "output_dir",
                "Directory for the split files",
                Ty::Path,
                Flag::Option("--output-dir"),
                true,
            ),
            p(
                "val_ratio",
                "Validation share, 0 to 1 (the CLI's default is 0.1)",
                Ty::Num(0.0, 1.0),
                Flag::Option("--val-ratio"),
                false,
            ),
            p(
                "test_ratio",
                "Test share, 0 to 1 (the CLI's default is 0)",
                Ty::Num(0.0, 1.0),
                Flag::Option("--test-ratio"),
                false,
            ),
            SEED,
            p(
                "stratify",
                "Field to stratify by, e.g. 'category'",
                Ty::Text,
                Flag::Option("--stratify"),
                false,
            ),
        ],
        hints: Hints::OVERWRITE,
    },
    CliTool {
        name: "dataset_merge",
        description: "Merge JSONL datasets into one, optionally shuffled, interleaved or weighted.",
        command: &["dataset", "merge"],
        params: &[
            p(
                "inputs",
                "Input datasets (files or directories)",
                Ty::Paths,
                Flag::Option("--inputs"),
                true,
            ),
            p(
                "output",
                "Output dataset (JSONL)",
                Ty::Path,
                Flag::Option("--output"),
                true,
            ),
            p(
                "shuffle",
                "Shuffle after merging",
                Ty::Bool,
                Flag::Switch("--shuffle"),
                false,
            ),
            SEED,
            p(
                "interleave",
                "Alternate samples from each input",
                Ty::Bool,
                Flag::Switch("--interleave"),
                false,
            ),
            p(
                "weights",
                "A weight per input, comma-separated, e.g. '1.0,2.0'",
                Ty::Text,
                Flag::Option("--weights"),
                false,
            ),
        ],
        hints: Hints::OVERWRITE,
    },
    CliTool {
        name: "dataset_sample",
        description: "Take a random sample of rows from a JSONL dataset.",
        command: &["dataset", "sample"],
        params: &[
            p(
                "input",
                "Input dataset (a file or directory: JSONL, JSON, Parquet, Arrow, CSV)",
                Ty::Path,
                Flag::Option("--input"),
                true,
            ),
            p(
                "output",
                "Output dataset (JSONL)",
                Ty::Path,
                Flag::Option("--output"),
                true,
            ),
            p(
                "num",
                "Number of samples",
                Ty::Int(1, i64::MAX),
                Flag::Option("--num"),
                true,
            ),
            SEED,
        ],
        hints: Hints::OVERWRITE,
    },
    CliTool {
        name: "dataset_template",
        description: "Apply a chat template to a conversation dataset, producing the text a model trains on.",
        command: &["dataset", "template"],
        params: &[
            p(
                "input",
                "Input dataset (a file or directory)",
                Ty::Path,
                Flag::Option("--input"),
                true,
            ),
            p(
                "output",
                "Output dataset (JSONL)",
                Ty::Path,
                Flag::Option("--output"),
                true,
            ),
            p(
                "template",
                "A built-in chat template in place of the model's own (without it and without a model, chatml)",
                Ty::Enum(TEMPLATES),
                Flag::Option("--template"),
                false,
            ),
            p(
                "system",
                "System message that replaces the default",
                Ty::Text,
                Flag::Option("--system"),
                false,
            ),
            p(
                "model",
                "Model ID or path whose chat template formats the conversations",
                Ty::Reference,
                Flag::Option("--model"),
                false,
            ),
            p(
                "add_generation_prompt",
                "Write each conversation only up to the generation prompt of its last assistant turn",
                Ty::Bool,
                Flag::Switch("--add-generation-prompt"),
                false,
            ),
            p(
                "mask_prompt",
                "Write each sample as a prompt and a completion (the last assistant turn), so a trainer trains on the completion only",
                Ty::Bool,
                Flag::Switch("--mask-prompt"),
                false,
            ),
        ],
        hints: Hints::OVERWRITE.open_world(),
    },
    CliTool {
        name: "dataset_prepare",
        description: "Download, template, filter, split and validate a dataset in one step. Produces a directory ready to train on.",
        command: &["dataset", "prepare"],
        params: &[
            p(
                "dataset",
                "Hugging Face dataset ID or local path",
                Ty::Reference,
                Flag::Positional,
                true,
            ),
            p(
                "output_dir",
                "Output directory",
                Ty::Path,
                Flag::Option("--output-dir"),
                true,
            ),
            p(
                "model",
                "Model ID or path whose tokenizer measures lengths",
                Ty::Reference,
                Flag::Option("--model"),
                true,
            ),
            p(
                "template",
                "A built-in chat template in place of the model's own",
                Ty::Enum(TEMPLATES),
                Flag::Option("--template"),
                false,
            ),
            p(
                "max_seq_len",
                "Drop samples longer than this (the CLI's default is 2048)",
                Ty::Int(1, 4_194_304),
                Flag::Option("--max-seq-len"),
                false,
            ),
            p(
                "val_ratio",
                "Validation share (the CLI's default is 0.05)",
                Ty::Num(0.0, 1.0),
                Flag::Option("--val-ratio"),
                false,
            ),
            SEED,
            p(
                "no_dedup",
                "Keep duplicates",
                Ty::Bool,
                Flag::Switch("--no-dedup"),
                false,
            ),
            p(
                "columns",
                "Column mapping, e.g. 'instruction=problem,output=solution'",
                Ty::Text,
                Flag::Option("--columns"),
                false,
            ),
        ],
        hints: Hints::OVERWRITE.open_world(),
    },
    CliTool {
        name: "bench_gen",
        description: "Benchmark the generation loop of a model: where the time goes in prefill and decode.",
        command: &["bench-gen"],
        params: &[p(
            "model",
            "Model ID or path (the CLI's default is Qwen/Qwen3-0.6B)",
            Ty::Reference,
            Flag::Option("--model"),
            false,
        )],
        hints: Hints::READ.open_world().idempotent(false),
    },
    CliTool {
        name: "bench_corpus",
        description: "Run the kernel benchmark corpus for this device tier and report the results.",
        command: &["bench-corpus"],
        params: &[
            p(
                "quick",
                "A shorter run with fewer iterations",
                Ty::Bool,
                Flag::Switch("--quick"),
                false,
            ),
            p(
                "json",
                "Report as JSON",
                Ty::Bool,
                Flag::Switch("--json"),
                false,
            ),
            p(
                "output",
                "Also write the JSON report here",
                Ty::Path,
                Flag::Option("--output"),
                false,
            ),
        ],
        hints: Hints::OVERWRITE.idempotent(false),
    },
    CliTool {
        name: "ollama_create",
        description: "Write the Modelfile that registers a pmetal model or adapter with Ollama.",
        command: &["ollama", "create"],
        params: &[
            p(
                "name",
                "Name for the Ollama model",
                Ty::Text,
                Flag::Option("--name"),
                true,
            ),
            p(
                "base",
                "Base model: a GGUF path or an Ollama model name",
                Ty::Text,
                Flag::Option("--base"),
                true,
            ),
            p(
                "lora",
                "LoRA adapter path",
                Ty::Path,
                Flag::Option("--lora"),
                false,
            ),
            p(
                "system",
                "System prompt",
                Ty::Text,
                Flag::Option("--system"),
                false,
            ),
            p(
                "temperature",
                "Sampling temperature",
                Ty::Num(0.0, 2.0),
                Flag::Option("--temperature"),
                false,
            ),
            p(
                "num_ctx",
                "Context window size",
                Ty::Int(1, 4_194_304),
                Flag::Option("--num-ctx"),
                false,
            ),
            p(
                "output",
                "Where to write the Modelfile (the CLI's default is ./Modelfile)",
                Ty::Path,
                Flag::Option("--output"),
                false,
            ),
            p(
                "template",
                "Model template; detected from the architecture when left out",
                Ty::Enum(OLLAMA_TEMPLATES),
                Flag::Option("--template"),
                false,
            ),
        ],
        hints: Hints::OVERWRITE,
    },
    CliTool {
        name: "ollama_modelfile",
        description: "Generate an Ollama Modelfile for a base model and optional adapter.",
        command: &["ollama", "modelfile"],
        params: &[
            p(
                "base",
                "Base model: a GGUF path or an Ollama model name",
                Ty::Text,
                Flag::Option("--base"),
                true,
            ),
            p(
                "lora",
                "LoRA adapter path",
                Ty::Path,
                Flag::Option("--lora"),
                false,
            ),
            p(
                "output",
                "Where to write the Modelfile (the CLI's default is ./Modelfile)",
                Ty::Path,
                Flag::Option("--output"),
                false,
            ),
            p(
                "system",
                "System prompt",
                Ty::Text,
                Flag::Option("--system"),
                false,
            ),
            p(
                "temperature",
                "Sampling temperature",
                Ty::Num(0.0, 2.0),
                Flag::Option("--temperature"),
                false,
            ),
            p(
                "num_ctx",
                "Context window size",
                Ty::Int(1, 4_194_304),
                Flag::Option("--num-ctx"),
                false,
            ),
            p(
                "top_k",
                "Top-k sampling",
                Ty::Int(0, 100_000),
                Flag::Option("--top-k"),
                false,
            ),
            p(
                "top_p",
                "Top-p (nucleus) sampling",
                Ty::Num(0.0, 1.0),
                Flag::Option("--top-p"),
                false,
            ),
            p(
                "template",
                "Model template; detected from the architecture when left out",
                Ty::Enum(OLLAMA_TEMPLATES),
                Flag::Option("--template"),
                false,
            ),
            p(
                "license",
                "License text",
                Ty::Text,
                Flag::Option("--license"),
                false,
            ),
        ],
        hints: Hints::OVERWRITE,
    },
];

impl CliTool {
    /// The tool definition: the schema generated from the parameters.
    pub fn definition(&self) -> Tool {
        Tool::new(self.name, self.description)
            .with_schema(turbomcp::ToolInputSchema::from_value(self.input_schema()))
            .with_annotations(self.hints.annotations())
    }

    /// The input schema: a closed object with one property per parameter.
    fn input_schema(&self) -> Value {
        let properties: Map<String, Value> = self
            .params
            .iter()
            .map(|p| (p.name.to_string(), p.schema()))
            .collect();
        let mut schema = json!({
            "type": "object",
            "additionalProperties": false,
            "properties": properties,
        });
        let required: Vec<&str> = self
            .params
            .iter()
            .filter(|p| p.required)
            .map(|p| p.name)
            .collect();
        if !required.is_empty() {
            schema["required"] = json!(required);
        }
        schema
    }

    /// The command line for `args`: everything after the binary name.
    pub fn argv(&self, args: Value, policy: &PathPolicy) -> McpResult<Vec<String>> {
        schema_check::check_arguments(self.name, &self.input_schema(), &args)
            .map_err(McpError::invalid_params)?;
        let args = args.as_object().cloned().unwrap_or_default();
        let mut argv: Vec<String> = self.command.iter().map(|s| (*s).to_string()).collect();
        let mut options: Vec<String> = Vec::new();
        for param in self.params {
            let Some(value) = args.get(param.name).filter(|v| !v.is_null()) else {
                continue;
            };
            let rendered = param
                .render(value, policy)
                .map_err(McpError::invalid_params)?;
            match (param.flag, rendered) {
                (Flag::Positional, Rendered::One(v)) => argv.push(v),
                // `--flag=value`, so a value that starts with '-' stays a value.
                (Flag::Option(flag), Rendered::One(v)) => options.push(format!("{flag}={v}")),
                (Flag::Option(flag), Rendered::Many(vs)) => {
                    options.push(flag.to_string());
                    options.extend(vs);
                }
                (Flag::Switch(flag), Rendered::Flag(true)) => options.push(flag.to_string()),
                (Flag::Switch(_), Rendered::Flag(false)) => {}
                (flag, rendered) => {
                    return Err(McpError::internal(format!(
                        "{}: {flag:?} cannot take {rendered:?}",
                        param.name
                    )));
                }
            }
        }
        // A value that starts with `-` would be read as a flag.
        for value in argv.iter().skip(self.command.len()) {
            if value.starts_with('-') {
                return Err(McpError::invalid_params(format!(
                    "`{value}` starts with '-' and would be read as an option"
                )));
            }
        }
        argv.extend(options);
        Ok(argv)
    }
}

#[derive(Debug)]
enum Rendered {
    One(String),
    Many(Vec<String>),
    Flag(bool),
}

impl Param {
    fn schema(&self) -> Value {
        let mut schema = match self.ty {
            Ty::Text | Ty::Path | Ty::Reference => json!({"type": "string"}),
            Ty::Paths => json!({"type": "array", "items": {"type": "string"}, "minItems": 1}),
            Ty::Int(min, max) => {
                let mut s = json!({"type": "integer", "minimum": min});
                if max != i64::MAX {
                    s["maximum"] = json!(max);
                }
                s
            }
            Ty::Num(min, max) => json!({"type": "number", "minimum": min, "maximum": max}),
            Ty::Bool => json!({"type": "boolean"}),
            Ty::Enum(options) => json!({"type": "string", "enum": options}),
        };
        schema["description"] = json!(self.doc);
        schema
    }

    /// The command-line form of `value`, whose type and range the schema check
    /// has already vouched for: paths are checked against the policy, text must
    /// not carry a NUL.
    fn render(&self, value: &Value, policy: &PathPolicy) -> Result<Rendered, String> {
        let name = self.name;
        let text = || {
            value
                .as_str()
                .ok_or_else(|| format!("`{name}` must be a string, got {value}"))
        };
        match self.ty {
            Ty::Text | Ty::Enum(_) => {
                let s = text()?;
                no_nul(name, s)?;
                Ok(Rendered::One(s.to_string()))
            }
            Ty::Path => Ok(Rendered::One(
                policy.check_path(name, text()?)?.display().to_string(),
            )),
            Ty::Reference => Ok(Rendered::One(policy.check_reference(name, text()?)?)),
            Ty::Paths => value
                .as_array()
                .into_iter()
                .flatten()
                .map(|v| {
                    let s = v
                        .as_str()
                        .ok_or_else(|| format!("`{name}` must be an array of strings"))?;
                    Ok(policy.check_path(name, s)?.display().to_string())
                })
                .collect::<Result<Vec<_>, String>>()
                .map(Rendered::Many),
            Ty::Int(..) | Ty::Num(..) => Ok(Rendered::One(
                value
                    .as_i64()
                    .map(|n| n.to_string())
                    .or_else(|| value.as_f64().map(|n| n.to_string()))
                    .ok_or_else(|| format!("`{name}` must be a number, got {value}"))?,
            )),
            Ty::Bool => {
                Ok(Rendered::Flag(value.as_bool().ok_or_else(|| {
                    format!("`{name}` must be a boolean, got {value}")
                })?))
            }
        }
    }
}

fn no_nul(name: &str, s: &str) -> Result<(), String> {
    if s.contains('\0') {
        Err(format!("`{name}` contains a NUL byte"))
    } else {
        Ok(())
    }
}

impl Param {
    /// A value of this parameter that its type accepts.
    fn example(&self, root: &std::path::Path, enum_index: usize) -> Value {
        match self.ty {
            Ty::Text => json!("example"),
            Ty::Path => json!(root.join(self.name).display().to_string()),
            Ty::Reference => json!("owner/example"),
            Ty::Paths => json!([
                root.join("one.jsonl").display().to_string(),
                root.join("two.jsonl").display().to_string()
            ]),
            Ty::Int(min, max) => json!(5.clamp(min, max)),
            Ty::Num(min, max) => json!(0.5_f64.clamp(min, max)),
            Ty::Bool => json!(true),
            Ty::Enum(options) => json!(options[enum_index.min(options.len() - 1)]),
        }
    }
}

impl CliTool {
    /// Command lines that exercise every parameter and every option of every
    /// enumeration, as a client would cause them: the required parameters
    /// alone, all parameters, and one call per option.
    ///
    /// The CLI's own parser is the judge of whether a flag is real, and it lives
    /// in another crate, which checks these.
    #[doc(hidden)]
    pub fn example_invocations(&self, root: &std::path::Path) -> Vec<Vec<String>> {
        let policy = PathPolicy::new(root.to_path_buf(), vec![root.to_path_buf()]);
        let build = |keep: &dyn Fn(&Param) -> bool, enum_index: usize| -> Vec<String> {
            let args: Map<String, Value> = self
                .params
                .iter()
                .filter(|p| keep(p))
                .map(|p| (p.name.to_string(), p.example(root, enum_index)))
                .collect();
            self.argv(Value::Object(args), &policy)
                .unwrap_or_else(|e| panic!("{}: {e}", self.name))
        };
        let mut out = vec![build(&|p| p.required, 0), build(&|_| true, 0)];
        let longest = self
            .params
            .iter()
            .filter_map(|p| match p.ty {
                Ty::Enum(options) => Some(options.len()),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        for i in 1..longest {
            out.push(build(&|_| true, i));
        }
        out
    }
}
