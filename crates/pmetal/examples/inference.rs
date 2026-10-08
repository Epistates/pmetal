//! Run inference using the SDK directly.
//!
//! ```sh
//! cargo run -p pmetal --example inference -- \
//!     --model Qwen/Qwen3-0.6B --prompt "What is 2+2?"
//! ```

use std::env;

use pmetal::data::Tokenizer;
use pmetal::data::chat_templates::{Message, detect_chat_template};
use pmetal::hub::resolve_model_path;
use pmetal::models::{DynamicModel, GenerationConfig, generate_cached_async};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let args: Vec<String> = env::args().collect();
    let model_id = args
        .iter()
        .position(|a| a == "--model")
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
        .unwrap_or("Qwen/Qwen3-0.6B");
    let prompt = args
        .iter()
        .position(|a| a == "--prompt")
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
        .unwrap_or("What is the capital of France?");

    println!("Running inference with {model_id}...\n");

    // Resolve model (downloads from HF if needed)
    let model_path = resolve_model_path(model_id, None, None).await?;

    // Load tokenizer and detect chat template
    let tokenizer = Tokenizer::from_model_dir(&model_path)?;
    let template = detect_chat_template(&model_path, model_id);
    let formatted = template.apply(&[Message::user(prompt)]).text;
    let input_ids = tokenizer.encode_with_special_tokens(&formatted)?;

    // Load model
    let mut model = DynamicModel::load(&model_path)?;

    // Configure generation — load model's recommended defaults
    let defaults = pmetal::data::inference_config::load_sampling_defaults(
        &model_path,
        pmetal::data::inference_config::SamplingMode::Auto,
        false,
    );
    let max_tokens = 256;
    let mut gen_config = GenerationConfig::sampling(max_tokens, defaults.temperature)
        .with_top_k(defaults.top_k)
        .with_top_p(defaults.top_p)
        .with_min_p(defaults.min_p)
        .with_repetition_penalty(defaults.repetition_penalty);

    // Collect ALL stop tokens (multi-EOS models, chat template EOS, well-known tokens)
    let stop_tokens = pmetal::data::inference_config::collect_all_stop_tokens(
        &model_path,
        &tokenizer,
        Some(template.template_type),
    );
    gen_config = gen_config.with_stop_tokens(stop_tokens);

    // Create caches (supports hybrid models with mamba cache)
    let mut cache = model.create_cache(input_ids.len() + max_tokens + 64);
    let mut mamba_cache = model.create_mamba_cache();

    // Generate
    let start = std::time::Instant::now();
    let output = generate_cached_async(
        |input, cache| {
            model.forward_with_hybrid_cache(input, None, Some(cache), mamba_cache.as_mut())
        },
        &input_ids,
        gen_config,
        &mut cache,
    )?;
    let elapsed = start.elapsed();

    // Decode
    let generated_tokens = &output.token_ids[input_ids.len()..];
    let text = tokenizer.decode(generated_tokens)?;

    println!("{text}\n");
    println!("---");
    println!(
        "Generated {} tokens ({:.1} tok/s)",
        output.num_generated,
        output.num_generated as f64 / elapsed.as_secs_f64()
    );

    Ok(())
}
