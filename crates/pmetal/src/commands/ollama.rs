use pmetal_models::ollama::{ModelfileBuilder, templates as ollama_templates};

use crate::OllamaAction;
use crate::OllamaTemplate;

/// Run Ollama subcommands.
pub(crate) async fn run_ollama_command(action: OllamaAction) -> anyhow::Result<()> {
    match action {
        OllamaAction::Modelfile {
            base,
            lora,
            output,
            system,
            temperature,
            num_ctx,
            top_k,
            top_p,
            template,
            license,
        } => {
            generate_modelfile(
                &base,
                lora.as_deref(),
                &output,
                system.as_deref(),
                temperature,
                num_ctx,
                top_k,
                top_p,
                template,
                license.as_deref(),
            )?;
        }

        OllamaAction::Create {
            name,
            base,
            lora,
            system,
            temperature,
            num_ctx,
            template,
        } => {
            create_ollama_model(
                &name,
                &base,
                lora.as_deref(),
                system.as_deref(),
                temperature,
                num_ctx,
                template,
            )?;
        }

        OllamaAction::Templates => {
            print_ollama_templates();
        }
    }

    Ok(())
}

/// Generate a Modelfile for Ollama.
fn generate_modelfile(
    base: &str,
    lora: Option<&str>,
    output: &str,
    system: Option<&str>,
    temperature: Option<f32>,
    num_ctx: Option<i32>,
    top_k: Option<i32>,
    top_p: Option<f32>,
    template: Option<OllamaTemplate>,
    license: Option<&str>,
) -> anyhow::Result<()> {
    // Validate output path to prevent path traversal
    let output_path = validate_file_path(output, true)?;

    println!("========================================");
    println!("  PMetal Ollama Export");
    println!("========================================");
    println!("Base Model:  {}", base);
    if let Some(lora_path) = lora {
        println!("LoRA:        {}", lora_path);
    }
    println!("Output:      {}", output_path.display());
    println!("========================================\n");

    // Build Modelfile
    let mut builder = ModelfileBuilder::new().from(base);

    // Add LoRA adapter if specified
    if let Some(lora_path) = lora {
        builder = builder.adapter(lora_path);
    }

    // Add system prompt
    if let Some(sys) = system {
        builder = builder.system(sys);
    }

    // Add parameters
    if let Some(temp) = temperature {
        builder = builder.temperature(temp);
    }
    if let Some(ctx) = num_ctx {
        builder = builder.num_ctx(ctx);
    }
    if let Some(k) = top_k {
        builder = builder.top_k(k);
    }
    if let Some(p) = top_p {
        builder = builder.top_p(p);
    }

    // Add template
    if let Some(tmpl) = template {
        let template_str = get_ollama_template(tmpl);
        builder = builder.template(template_str);
    } else {
        // Try to auto-detect template from base model name
        if let Some(detected_template) = detect_template_from_model(base) {
            builder = builder.template(detected_template);
            println!("Auto-detected template from model name");
        }
    }

    // Add license
    if let Some(lic) = license {
        builder = builder.license(lic);
    }

    // Build and write
    builder.write_to_file(&output_path)?;
    println!("Modelfile written to: {}", output_path.display());

    println!("\nTo create the model in Ollama, run:");
    println!("  ollama create <model-name> -f {}", output_path.display());

    Ok(())
}

/// Validate model name for Ollama (prevent command injection).
fn validate_ollama_model_name(name: &str) -> anyhow::Result<()> {
    // Allow alphanumeric, hyphen, underscore, period, forward slash (for namespaces)
    // Reject anything that could be interpreted as shell metacharacters
    if name.is_empty() {
        anyhow::bail!("Model name cannot be empty");
    }
    if name.len() > 255 {
        anyhow::bail!("Model name too long (max 255 characters)");
    }
    if name.starts_with('.') || name.starts_with('-') {
        anyhow::bail!("Model name cannot start with '.' or '-'");
    }
    if !name
        .chars()
        .all(|c| c.is_alphanumeric() || matches!(c, '-' | '_' | '.' | '/' | ':'))
    {
        anyhow::bail!(
            "Invalid model name '{}'. Name must contain only alphanumeric characters, \
             hyphens, underscores, periods, colons, and forward slashes.",
            name
        );
    }
    Ok(())
}

/// Validate file path (prevent path traversal).
pub(crate) fn validate_file_path(
    path: &str,
    allow_creation: bool,
) -> anyhow::Result<std::path::PathBuf> {
    let path = std::path::Path::new(path);

    // Prevent path traversal
    if path
        .components()
        .any(|c| matches!(c, std::path::Component::ParentDir))
    {
        anyhow::bail!("Invalid path: path traversal detected (.. not allowed)");
    }

    // Get canonical path
    let canonical = if path.exists() {
        path.canonicalize()?
    } else if allow_creation {
        // If file doesn't exist yet, canonicalize parent
        if let Some(parent) = path.parent() {
            if parent.as_os_str().is_empty() {
                let file_name = path
                    .file_name()
                    .ok_or_else(|| anyhow::anyhow!("Invalid path: no file name"))?;
                std::env::current_dir()?.join(file_name)
            } else {
                let file_name = path
                    .file_name()
                    .ok_or_else(|| anyhow::anyhow!("Invalid path: no file name"))?;
                parent.canonicalize()?.join(file_name)
            }
        } else {
            std::env::current_dir()?.join(path)
        }
    } else {
        anyhow::bail!("Path does not exist: {}", path.display());
    };

    Ok(canonical)
}

/// Create and register a model with Ollama.
fn create_ollama_model(
    name: &str,
    base: &str,
    lora: Option<&str>,
    system: Option<&str>,
    temperature: Option<f32>,
    num_ctx: Option<i32>,
    template: Option<OllamaTemplate>,
) -> anyhow::Result<()> {
    // Validate model name to prevent command injection
    validate_ollama_model_name(name)?;

    // Create secure temporary file (auto-cleaned on drop)
    let modelfile = tempfile::Builder::new()
        .prefix("pmetal-modelfile-")
        .suffix(".txt")
        .tempfile()?;
    let modelfile_path = modelfile.path().to_path_buf();
    let modelfile_str = modelfile_path.to_string_lossy().to_string();

    generate_modelfile(
        base,
        lora,
        &modelfile_str,
        system,
        temperature,
        num_ctx,
        None,
        None,
        template,
        None,
    )?;

    println!("\nCreating Ollama model '{}'...", name);

    // Run ollama create
    let status = std::process::Command::new("ollama")
        .args(["create", name, "-f", &modelfile_str])
        .status();

    match status {
        Ok(exit_status) if exit_status.success() => {
            println!("\nModel '{}' created successfully!", name);
            println!("\nTo use the model, run:");
            println!("  ollama run {}", name);
            // modelfile is auto-cleaned on drop
        }
        Ok(exit_status) => {
            // Persist the temp file so user can inspect it
            let persisted = modelfile.into_temp_path();
            let kept_path = persisted.keep()?;
            anyhow::bail!(
                "ollama create failed with exit code: {:?}. \
                 Modelfile saved at: {}",
                exit_status.code(),
                kept_path.display()
            );
        }
        Err(e) => {
            if e.kind() == std::io::ErrorKind::NotFound {
                let persisted = modelfile.into_temp_path();
                let kept_path = persisted.keep()?;
                println!("\nOllama not found. Please install Ollama first:");
                println!("  https://ollama.ai/download");
                println!("\nModelfile has been saved to: {}", kept_path.display());
                println!("Once Ollama is installed, run:");
                println!("  ollama create {} -f {}", name, kept_path.display());
            } else {
                anyhow::bail!("Failed to run ollama: {}", e);
            }
        }
    }

    Ok(())
}

/// Print available Ollama templates.
fn print_ollama_templates() {
    println!("Available Ollama Templates:");
    println!("========================================\n");

    println!("llama3 - Llama 3 Chat Format");
    println!("  Uses: <|start_header_id|>...<|end_header_id|> format");
    println!("  Best for: Llama 3, Llama 3.1, Llama 3.2, Llama 4\n");

    println!("qwen3 - Qwen3/ChatML Format");
    println!("  Uses: <|im_start|>...<|im_end|> format");
    println!("  Best for: Qwen 2, Qwen 2.5, Qwen 3\n");

    println!("gemma - Gemma Instruct Format");
    println!("  Uses: <start_of_turn>...<end_of_turn> format");
    println!("  Best for: Gemma 2, Gemma 3\n");

    println!("mistral - Mistral Instruct Format");
    println!("  Uses: [INST]...[/INST] format");
    println!("  Best for: Mistral, Mixtral\n");

    println!("phi3 - Phi-3 Instruct Format");
    println!("  Uses: <|system|>...<|end|> format");
    println!("  Best for: Phi 3, Phi 4\n");

    println!("deepseek - DeepSeek Chat Format");
    println!("  Uses: <|begin_of_sentence|>User:...Assistant: format");
    println!("  Best for: DeepSeek, DeepSeek-V2, DeepSeek-V3\n");

    println!("========================================");
    println!("Usage: pmetal ollama modelfile --base <model> --template <template>");
}

/// Get the Ollama template string for a template type.
fn get_ollama_template(template: OllamaTemplate) -> &'static str {
    match template {
        OllamaTemplate::Llama3 => ollama_templates::LLAMA3_CHAT,
        OllamaTemplate::Qwen3 => ollama_templates::QWEN3_CHAT,
        OllamaTemplate::Gemma => ollama_templates::GEMMA_INSTRUCT,
        OllamaTemplate::Mistral => ollama_templates::MISTRAL_INSTRUCT,
        OllamaTemplate::Phi3 => ollama_templates::PHI3_INSTRUCT,
        OllamaTemplate::DeepSeek => ollama_templates::DEEPSEEK_CHAT,
    }
}

/// Try to detect the appropriate template from the model name.
fn detect_template_from_model(model: &str) -> Option<&'static str> {
    let lower = model.to_lowercase();

    if lower.contains("llama") || lower.contains("meta-llama") {
        Some(ollama_templates::LLAMA3_CHAT)
    } else if lower.contains("qwen") {
        Some(ollama_templates::QWEN3_CHAT)
    } else if lower.contains("gemma") {
        Some(ollama_templates::GEMMA_INSTRUCT)
    } else if lower.contains("mistral") || lower.contains("mixtral") {
        Some(ollama_templates::MISTRAL_INSTRUCT)
    } else if lower.contains("phi") {
        Some(ollama_templates::PHI3_INSTRUCT)
    } else if lower.contains("deepseek") {
        Some(ollama_templates::DEEPSEEK_CHAT)
    } else {
        None
    }
}
