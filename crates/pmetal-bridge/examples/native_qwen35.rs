//! Standalone Qwen3.5 inference — SINGLE libmlx.a, no mlx-rs.
//! Proves the performance achievable without dual-MLX-instance interference.
//!
//! Usage: cargo run -p pmetal-bridge --release --example native_qwen35 -- \
//!          --model /path/to/Qwen3.5-0.8B --max-tokens 200

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut model_path = String::new();
    let mut max_tokens: usize = 200;
    let mut temperature: f32 = 0.0;
    let mut tq_bits: Option<u8> = None;
    let mut dump_decode_graph = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                model_path = args[i].clone();
            }
            "--max-tokens" => {
                i += 1;
                max_tokens = args[i].parse().unwrap();
            }
            "--temperature" => {
                i += 1;
                temperature = args[i].parse().unwrap();
            }
            "--turboquant" => {
                i += 1;
                tq_bits = Some(args[i].parse().unwrap());
            }
            "--dump-decode-graph" => {
                dump_decode_graph = true;
            }
            _ => {}
        }
        i += 1;
    }

    if model_path.is_empty() {
        eprintln!(
            "Usage: native_qwen35 --model <path> [--max-tokens N] [--temperature T] [--turboquant BITS] [--dump-decode-graph]"
        );
        std::process::exit(1);
    }

    let model_dir = std::path::Path::new(&model_path);
    let config = pmetal_bridge::qwen3_native::load_config(model_dir)
        .unwrap_or_else(|e| panic!("Config: {e}"));

    eprintln!(
        "Model: {} layers, hidden={}, GDN nk={} nv={}",
        config.num_hidden_layers,
        config.hidden_size,
        config.gdn_nk(),
        config.gdn_nv()
    );

    let t0 = std::time::Instant::now();
    let weights = pmetal_bridge::qwen3_native::load_model(model_dir, &config)
        .unwrap_or_else(|e| panic!("Weights: {e}"));
    eprintln!(
        "Loaded in {:.1}s, active={:.0}MB",
        t0.elapsed().as_secs_f64(),
        pmetal_bridge::inline_array::get_active_memory() as f64 / 1e6
    );

    // Dummy prefill with a few tokens (no tokenizer in this example)
    let token_ids: Vec<i32> = vec![151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13];
    let mut cache = if let Some(bits) = tq_bits {
        let tq_config = pmetal_bridge::turboquant::TurboQuantConfig::uniform(bits, bits);
        eprintln!("TurboQuant: {bits}-bit KV cache compression");
        pmetal_bridge::qwen3_native::NativeCache::new_with_turboquant(&weights, Some(tq_config))
    } else {
        pmetal_bridge::qwen3_native::NativeCache::new_empty(&weights)
    };

    let input = pmetal_bridge::InlineArray::from_i32_slice(&token_ids)
        .reshape(&[1, token_ids.len() as i32]);
    eprintln!("Running prefill...");
    let logits = pmetal_bridge::qwen3_native::forward_step(&weights, &input, &mut cache);
    eprintln!("Prefill graph built. Evaluating...");

    let seq_len = token_ids.len() as i32;
    let vocab = weights.embed_w.dim(0);
    let last_logits = logits
        .reshape(&[seq_len, vocab])
        .slice(&[seq_len - 1, 0], &[seq_len, vocab]);
    let first_tok = last_logits.argmax(-1);
    first_tok.eval();
    let first_tok_id = first_tok.item_u32();
    eprintln!("First token: {first_tok_id}");

    if dump_decode_graph {
        pmetal_bridge::decode::begin_generation_session("NATIVE-GRAPH", weights.model_dtype);
        cache.eval_and_detach_states();
        pmetal_bridge::inline_array::clear_cache();

        let decode_input =
            pmetal_bridge::InlineArray::from_i32(first_tok_id as i32).reshape(&[1, 1]);
        let decode_logits =
            pmetal_bridge::qwen3_native::forward_step(&weights, &decode_input, &mut cache);

        let node_count = pmetal_bridge::inline_array::graph_node_count(&decode_logits);
        let desc_count = pmetal_bridge::inline_array::graph_desc_count(&decode_logits);
        eprintln!("Decode graph nodes={node_count} descs={desc_count}");
        pmetal_bridge::inline_array::graph_dump(&decode_logits);
        std::process::exit(0);
    }

    let t0 = std::time::Instant::now();
    eprintln!("Using canonical bridge decode path");
    let (tokens, metrics) = pmetal_bridge::qwen3_native::generate_canonical(
        &weights,
        &mut cache,
        &config,
        first_tok_id,
        max_tokens,
        temperature,
        tq_bits.map(|bits| pmetal_bridge::turboquant::TurboQuantConfig::uniform(bits, bits)),
        |_tok| true,
    );
    let elapsed = t0.elapsed().as_secs_f64();
    eprintln!(
        "Generated {} tokens in {:.2}s ({:.1} tok/s)",
        tokens.len(),
        elapsed,
        tokens.len() as f64 / elapsed
    );
    if let Some(m) = metrics {
        eprintln!(
            "Decode (bridge): {:.0} tok/s (avg={:.2}ms p50={:.2}ms over {} steps)",
            m.tok_per_sec, m.avg_step_ms, m.p50_step_ms, m.measured_steps,
        );
    }
}
