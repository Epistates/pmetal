use pmetal_data::Tokenizer;

use crate::ChatTemplate;
use crate::DatasetAction;

/// Run dataset subcommands.
pub(crate) async fn run_dataset_command(action: DatasetAction) -> anyhow::Result<()> {
    use std::collections::HashMap;
    use std::io::{BufRead, BufReader, Write};

    match action {
        DatasetAction::Analyze {
            path,
            model,
            detailed,
        } => {
            println!("========================================");
            println!("  PMetal Dataset Analysis");
            println!("========================================");
            println!("Path: {}", path);
            println!("========================================\n");

            // Load tokenizer if model specified
            let tokenizer = if let Some(model_id) = &model {
                println!("Loading tokenizer from {}...", model_id);
                let model_path =
                    if model_id.contains('/') && !std::path::Path::new(model_id).exists() {
                        pmetal_hub::download_model(model_id, None, None).await?
                    } else {
                        std::path::PathBuf::from(model_id)
                    };
                Some(Tokenizer::from_model_dir(&model_path)?)
            } else {
                None
            };

            // Read JSONL file
            let file = std::fs::File::open(&path)?;
            let reader = BufReader::new(file);

            let mut total_samples = 0usize;
            let mut char_lengths = Vec::new();
            let mut token_lengths = Vec::new();
            let mut formats_detected: HashMap<String, usize> = HashMap::new();
            let mut empty_samples = 0usize;

            for line in reader.lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }

                total_samples += 1;

                // Parse JSON to detect format
                let json: serde_json::Value = serde_json::from_str(&line)?;

                // Detect format
                let format = if json.get("text").is_some() {
                    "simple"
                } else if json.get("conversations").is_some() {
                    "sharegpt"
                } else if json.get("instruction").is_some() {
                    "alpaca"
                } else if json.get("messages").is_some() {
                    "messages"
                } else {
                    "unknown"
                };
                *formats_detected.entry(format.to_string()).or_insert(0) += 1;

                // Extract text content
                let text = if let Some(t) = json.get("text").and_then(|v| v.as_str()) {
                    t.to_string()
                } else if let Some(convs) = json.get("conversations").and_then(|v| v.as_array()) {
                    convs
                        .iter()
                        .filter_map(|c| c.get("value").and_then(|v| v.as_str()))
                        .collect::<Vec<_>>()
                        .join(" ")
                } else if let Some(inst) = json.get("instruction").and_then(|v| v.as_str()) {
                    let input = json.get("input").and_then(|v| v.as_str()).unwrap_or("");
                    let output = json.get("output").and_then(|v| v.as_str()).unwrap_or("");
                    format!("{} {} {}", inst, input, output)
                } else {
                    String::new()
                };

                if text.is_empty() {
                    empty_samples += 1;
                    continue;
                }

                char_lengths.push(text.len());

                // Tokenize if tokenizer available
                if let Some(ref tok) = tokenizer {
                    let tokens = tok.encode(&text)?;
                    token_lengths.push(tokens.len());
                }
            }

            // Compute statistics
            println!("=== Dataset Statistics ===\n");
            println!("Total samples:    {}", total_samples);
            println!("Empty samples:    {}", empty_samples);
            println!("Valid samples:    {}", total_samples - empty_samples);
            println!();

            println!("Detected formats:");
            for (format, count) in &formats_detected {
                println!(
                    "  {}: {} ({:.1}%)",
                    format,
                    count,
                    100.0 * *count as f64 / total_samples as f64
                );
            }
            println!();

            if !char_lengths.is_empty() {
                char_lengths.sort();
                let min = char_lengths[0];
                let max = *char_lengths.last().unwrap();
                let mean = char_lengths.iter().sum::<usize>() as f64 / char_lengths.len() as f64;
                let median = char_lengths[char_lengths.len() / 2];
                let p90 = char_lengths[(char_lengths.len() as f64 * 0.9) as usize];
                let p95 = char_lengths[(char_lengths.len() as f64 * 0.95) as usize];

                println!("Character lengths:");
                println!("  Min:    {}", min);
                println!("  Max:    {}", max);
                println!("  Mean:   {:.0}", mean);
                println!("  Median: {}", median);
                println!("  P90:    {}", p90);
                println!("  P95:    {}", p95);
                println!();
            }

            if !token_lengths.is_empty() {
                token_lengths.sort();
                let min = token_lengths[0];
                let max = *token_lengths.last().unwrap();
                let mean = token_lengths.iter().sum::<usize>() as f64 / token_lengths.len() as f64;
                let median = token_lengths[token_lengths.len() / 2];
                let p90 = token_lengths[(token_lengths.len() as f64 * 0.9) as usize];
                let p95 = token_lengths[(token_lengths.len() as f64 * 0.95) as usize];
                let p99 = token_lengths[(token_lengths.len() as f64 * 0.99) as usize];

                println!("Token lengths:");
                println!("  Min:    {}", min);
                println!("  Max:    {}", max);
                println!("  Mean:   {:.0}", mean);
                println!("  Median: {}", median);
                println!("  P90:    {}", p90);
                println!("  P95:    {} (recommended max_seq_len)", p95);
                println!("  P99:    {}", p99);
                println!();

                // Histogram
                let buckets = [512, 1024, 2048, 4096, 8192, 16384, 32768];
                println!("Token length distribution:");
                for (i, &bucket) in buckets.iter().enumerate() {
                    let lower = if i == 0 { 0 } else { buckets[i - 1] };
                    let count = token_lengths
                        .iter()
                        .filter(|&&l| l > lower && l <= bucket)
                        .count();
                    let pct = 100.0 * count as f64 / token_lengths.len() as f64;
                    let bar = "█".repeat((pct / 2.0) as usize);
                    println!(
                        "  {:>5}-{:<5}: {:>5} ({:5.1}%) {}",
                        lower, bucket, count, pct, bar
                    );
                }
                let over_max = token_lengths
                    .iter()
                    .filter(|&&l| l > *buckets.last().unwrap())
                    .count();
                if over_max > 0 {
                    let pct = 100.0 * over_max as f64 / token_lengths.len() as f64;
                    println!(
                        "  >{:5}:     {:>5} ({:5.1}%)",
                        buckets.last().unwrap(),
                        over_max,
                        pct
                    );
                }
            }

            if detailed && !token_lengths.is_empty() {
                println!("\n=== Sample Details ===");
                for (i, len) in token_lengths.iter().take(10).enumerate() {
                    println!("  Sample {}: {} tokens", i + 1, len);
                }
                if token_lengths.len() > 10 {
                    println!("  ... and {} more samples", token_lengths.len() - 10);
                }
            }
        }

        DatasetAction::Download {
            dataset_id,
            split,
            output,
            revision,
        } => {
            println!("========================================");
            println!("  PMetal Dataset Download");
            println!("========================================");
            println!("Dataset:  {}", dataset_id);
            println!("Split:    {}", split);
            println!("========================================\n");

            // Download parquet files
            println!("Downloading dataset from HuggingFace Hub...");
            let parquet_paths = pmetal_hub::download_dataset_parquet(
                &dataset_id,
                &split,
                revision.as_deref(),
                None,
            )
            .await?;

            println!("Downloaded {} parquet file(s)", parquet_paths.len());

            // Determine output path
            let output_path = output.unwrap_or_else(|| {
                let safe_name = dataset_id.replace('/', "_");
                format!("{}.jsonl", safe_name)
            });

            let validated_download_output =
                crate::validate_output_path(&output_path, "dataset download output")?;
            println!(
                "Converting to JSONL: {}",
                validated_download_output.display()
            );

            // Convert parquet to JSONL using arrow-parquet
            let mut output_file = std::fs::File::create(&validated_download_output)?;
            let mut total_rows = 0usize;

            for parquet_path in &parquet_paths {
                use arrow_array::RecordBatchReader;
                use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

                let file = std::fs::File::open(parquet_path)?;
                let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
                let reader = builder.build()?;

                // Get column names from schema
                let schema = reader.schema();
                let columns: Vec<String> =
                    schema.fields().iter().map(|f| f.name().clone()).collect();
                println!("  Columns: {:?}", columns);

                // Use arrow-json to properly serialize record batches
                use arrow_json::writer::LineDelimitedWriter;

                for batch_result in reader {
                    let batch = batch_result?;

                    // Use arrow-json for proper nested type serialization
                    let mut json_buf = Vec::new();
                    {
                        let mut json_writer = LineDelimitedWriter::new(&mut json_buf);
                        json_writer.write(&batch)?;
                        json_writer.finish()?;
                    }

                    // Parse and re-write each line to handle conversions
                    for line in std::str::from_utf8(&json_buf)?.lines() {
                        if line.trim().is_empty() {
                            continue;
                        }

                        // Parse and potentially transform the JSON
                        let mut obj: serde_json::Value = serde_json::from_str(line)?;

                        // If it has "conversations" field as a string, parse it
                        if let Some(serde_json::Value::String(conv_str)) = obj.get("conversations")
                        {
                            if let Ok(convs) = serde_json::from_str::<serde_json::Value>(conv_str) {
                                obj["conversations"] = convs;
                            }
                        }

                        // If it has "messages" field, convert to ShareGPT format
                        if let Some(serde_json::Value::Array(msgs)) = obj.get("messages").cloned() {
                            let conversations: Vec<_> = msgs
                                .iter()
                                .filter_map(|m| {
                                    let role = m.get("role")?.as_str()?;
                                    let content = m.get("content")?.as_str()?;
                                    let from = match role {
                                        "user" => "human",
                                        "assistant" => "gpt",
                                        "system" => "system",
                                        _ => role,
                                    };
                                    Some(serde_json::json!({
                                        "from": from,
                                        "value": content
                                    }))
                                })
                                .collect();

                            // Replace messages with conversations in ShareGPT format
                            if let Some(obj_map) = obj.as_object_mut() {
                                obj_map.remove("messages");
                                obj_map.insert(
                                    "conversations".to_string(),
                                    serde_json::Value::Array(conversations),
                                );
                            }
                        }

                        writeln!(output_file, "{}", serde_json::to_string(&obj)?)?;
                        total_rows += 1;
                    }
                }
            }

            println!("\n========================================");
            println!("  Download Complete!");
            println!("========================================");
            println!("Total samples: {}", total_rows);
            println!("Output:        {}", output_path);
            println!("========================================");
        }

        DatasetAction::Convert {
            input,
            output,
            format,
            columns,
            shuffle,
            seed,
        } => {
            println!("========================================");
            println!("  PMetal Dataset Conversion");
            println!("========================================");
            println!("Input:    {}", input);
            println!("Output:   {}", output);
            if let Some(ref f) = format {
                println!("Format:   {:?}", f);
            }
            println!("Shuffle:  {}", shuffle);
            println!("========================================\n");

            // Parse column mappings
            let col_map: HashMap<String, String> = columns
                .map(|c| {
                    c.split(',')
                        .filter_map(|pair| {
                            let parts: Vec<&str> = pair.split('=').collect();
                            if parts.len() == 2 {
                                Some((parts[0].to_string(), parts[1].to_string()))
                            } else {
                                None
                            }
                        })
                        .collect()
                })
                .unwrap_or_default();

            // Read input
            let mut samples: Vec<serde_json::Value> = Vec::new();

            let input_path = std::path::Path::new(&input);
            let extension = input_path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("");

            match extension.to_lowercase().as_str() {
                "parquet" => {
                    use arrow_array::RecordBatchReader;
                    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

                    let file = std::fs::File::open(&input)?;
                    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
                    let reader = builder.build()?;

                    let schema = reader.schema();
                    let columns: Vec<String> =
                        schema.fields().iter().map(|f| f.name().clone()).collect();

                    for batch_result in reader {
                        let batch = batch_result?;
                        let num_rows = batch.num_rows();

                        for row_idx in 0..num_rows {
                            let mut obj = serde_json::Map::new();

                            for (col_idx, col_name) in columns.iter().enumerate() {
                                let target_name = col_map.get(col_name).unwrap_or(col_name);
                                let col = batch.column(col_idx);

                                use arrow_array::{Array, cast::AsArray};
                                let value = if let Some(arr) = col.as_string_opt::<i32>() {
                                    if arr.is_null(row_idx) {
                                        serde_json::Value::Null
                                    } else {
                                        serde_json::Value::String(arr.value(row_idx).to_string())
                                    }
                                } else if let Some(arr) = col.as_string_opt::<i64>() {
                                    if arr.is_null(row_idx) {
                                        serde_json::Value::Null
                                    } else {
                                        serde_json::Value::String(arr.value(row_idx).to_string())
                                    }
                                } else {
                                    serde_json::Value::Null
                                };

                                obj.insert(target_name.clone(), value);
                            }
                            samples.push(serde_json::Value::Object(obj));
                        }
                    }
                }
                "jsonl" | "json" => {
                    let file = std::fs::File::open(&input)?;
                    let reader = BufReader::new(file);

                    for line in reader.lines() {
                        let line = line?;
                        if line.trim().is_empty() {
                            continue;
                        }
                        let obj: serde_json::Value = serde_json::from_str(&line)?;
                        samples.push(obj);
                    }
                }
                _ => {
                    return Err(anyhow::anyhow!("Unsupported input format: {}", extension));
                }
            }

            println!("Loaded {} samples", samples.len());

            // Shuffle if requested
            if shuffle {
                use rand::SeedableRng;
                use rand::seq::SliceRandom;
                let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
                samples.shuffle(&mut rng);
                println!("Shuffled with seed {}", seed);
            }

            // Validate and write output
            let validated_convert_output =
                crate::validate_output_path(&output, "dataset convert output")?;
            let mut output_file = std::fs::File::create(&validated_convert_output)?;
            for sample in &samples {
                writeln!(output_file, "{}", serde_json::to_string(sample)?)?;
            }

            println!("\n========================================");
            println!("  Conversion Complete!");
            println!("========================================");
            println!("Output:  {}", output);
            println!("Samples: {}", samples.len());
            println!("========================================");
        }

        DatasetAction::Validate {
            path,
            model,
            max_seq_len,
        } => {
            println!("========================================");
            println!("  PMetal Dataset Validation");
            println!("========================================");
            println!("Path:        {}", path);
            println!("Max Seq Len: {}", max_seq_len);
            println!("========================================\n");

            // Load tokenizer
            let tokenizer = if let Some(model_id) = &model {
                println!("Loading tokenizer from {}...", model_id);
                let model_path =
                    if model_id.contains('/') && !std::path::Path::new(model_id).exists() {
                        pmetal_hub::download_model(model_id, None, None).await?
                    } else {
                        std::path::PathBuf::from(model_id)
                    };
                Some(Tokenizer::from_model_dir(&model_path)?)
            } else {
                None
            };

            // Read JSONL file
            let file = std::fs::File::open(&path)?;
            let reader = BufReader::new(file);

            let mut total_samples = 0usize;
            let mut valid_samples = 0usize;
            let mut too_long = 0usize;
            let mut empty = 0usize;
            let mut parse_errors = 0usize;
            let mut issues: Vec<String> = Vec::new();

            for (line_num, line) in reader.lines().enumerate() {
                let line = match line {
                    Ok(l) => l,
                    Err(e) => {
                        issues.push(format!("Line {}: Read error: {}", line_num + 1, e));
                        parse_errors += 1;
                        continue;
                    }
                };

                if line.trim().is_empty() {
                    continue;
                }

                total_samples += 1;

                // Parse JSON
                let json: serde_json::Value = match serde_json::from_str(&line) {
                    Ok(j) => j,
                    Err(e) => {
                        issues.push(format!("Line {}: JSON parse error: {}", line_num + 1, e));
                        parse_errors += 1;
                        continue;
                    }
                };

                // Extract text
                let text = if let Some(t) = json.get("text").and_then(|v| v.as_str()) {
                    t.to_string()
                } else if let Some(convs) = json.get("conversations").and_then(|v| v.as_array()) {
                    convs
                        .iter()
                        .filter_map(|c| c.get("value").and_then(|v| v.as_str()))
                        .collect::<Vec<_>>()
                        .join(" ")
                } else if let Some(inst) = json.get("instruction").and_then(|v| v.as_str()) {
                    let input = json.get("input").and_then(|v| v.as_str()).unwrap_or("");
                    let output = json.get("output").and_then(|v| v.as_str()).unwrap_or("");
                    format!("{} {} {}", inst, input, output)
                } else {
                    issues.push(format!("Line {}: No recognizable text field", line_num + 1));
                    empty += 1;
                    continue;
                };

                if text.trim().is_empty() {
                    issues.push(format!("Line {}: Empty text content", line_num + 1));
                    empty += 1;
                    continue;
                }

                // Tokenize and check length
                if let Some(ref tok) = tokenizer {
                    let tokens = tok.encode(&text)?;
                    if tokens.len() > max_seq_len {
                        too_long += 1;
                        if issues.len() < 10 {
                            issues.push(format!(
                                "Line {}: Token length {} exceeds max_seq_len {}",
                                line_num + 1,
                                tokens.len(),
                                max_seq_len
                            ));
                        }
                        continue;
                    }
                }

                valid_samples += 1;
            }

            // Report results
            println!("=== Validation Results ===\n");
            println!("Total samples:   {}", total_samples);
            println!(
                "Valid samples:   {} ({:.1}%)",
                valid_samples,
                100.0 * valid_samples as f64 / total_samples as f64
            );
            println!("Parse errors:    {}", parse_errors);
            println!("Empty samples:   {}", empty);
            println!("Too long:        {} (>{} tokens)", too_long, max_seq_len);
            println!();

            if !issues.is_empty() {
                println!("Issues (first 10):");
                for issue in issues.iter().take(10) {
                    println!("  - {}", issue);
                }
                if issues.len() > 10 {
                    println!("  ... and {} more issues", issues.len() - 10);
                }
            }

            // Overall status
            if parse_errors == 0 && empty == 0 {
                println!("\n✓ Dataset is valid for training");
                if too_long > 0 {
                    println!(
                        "  Note: {} samples exceed max_seq_len and will be truncated",
                        too_long
                    );
                }
            } else {
                println!("\n✗ Dataset has issues that need attention");
            }
        }

        DatasetAction::Preview {
            dataset_id,
            split,
            num,
        } => {
            println!("========================================");
            println!("  PMetal Dataset Preview");
            println!("========================================");
            println!("Dataset: {}", dataset_id);
            println!("Split:   {}", split);
            println!("Samples: {}", num);
            println!("========================================\n");

            // Download parquet files
            println!("Fetching dataset...");
            let parquet_paths =
                pmetal_hub::download_dataset_parquet(&dataset_id, &split, None, None).await?;

            let mut shown = 0usize;
            'outer: for parquet_path in &parquet_paths {
                use arrow_array::RecordBatchReader;
                use arrow_json::writer::LineDelimitedWriter;
                use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

                let file = std::fs::File::open(parquet_path)?;
                let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
                let reader = builder.build()?;

                let schema = reader.schema();
                if shown == 0 {
                    let columns: Vec<&str> =
                        schema.fields().iter().map(|f| f.name().as_str()).collect();
                    println!("Columns: {:?}\n", columns);
                }

                for batch_result in reader {
                    let batch = batch_result?;

                    // Serialize batch to JSON
                    let mut json_buf = Vec::new();
                    {
                        let mut json_writer = LineDelimitedWriter::new(&mut json_buf);
                        json_writer.write(&batch)?;
                        json_writer.finish()?;
                    }

                    for line in std::str::from_utf8(&json_buf)?.lines() {
                        if line.trim().is_empty() {
                            continue;
                        }
                        if shown >= num {
                            break 'outer;
                        }

                        let obj: serde_json::Value = serde_json::from_str(line)?;
                        println!("--- Sample {} ---", shown + 1);
                        println!("{}", serde_json::to_string_pretty(&obj)?);
                        println!();
                        shown += 1;
                    }
                }
            }

            println!("Showed {} sample(s)", shown);
        }

        DatasetAction::Filter {
            input,
            output,
            model,
            min_tokens,
            max_tokens,
            dedup,
            pattern,
            invert,
            complete_only,
        } => {
            println!("========================================");
            println!("  PMetal Dataset Filter");
            println!("========================================");
            println!("Input:  {}", input);
            println!("Output: {}", output);
            if let Some(min) = min_tokens {
                println!("Min tokens: {}", min);
            }
            if let Some(max) = max_tokens {
                println!("Max tokens: {}", max);
            }
            if dedup {
                println!("Deduplication: enabled");
            }
            if let Some(ref p) = pattern {
                println!("Pattern: {} (invert: {})", p, invert);
            }
            if complete_only {
                println!("Complete only: enabled");
            }
            println!("========================================\n");

            // Load tokenizer if needed for token filtering
            let tokenizer = if min_tokens.is_some() || max_tokens.is_some() {
                if let Some(model_id) = &model {
                    println!("Loading tokenizer from {}...", model_id);
                    let model_path =
                        if model_id.contains('/') && !std::path::Path::new(model_id).exists() {
                            pmetal_hub::download_model(model_id, None, None).await?
                        } else {
                            std::path::PathBuf::from(model_id)
                        };
                    Some(Tokenizer::from_model_dir(&model_path)?)
                } else {
                    return Err(anyhow::anyhow!(
                        "--model required for token-based filtering"
                    ));
                }
            } else {
                None
            };

            // Compile regex if provided
            let regex = if let Some(ref p) = pattern {
                Some(regex::Regex::new(p)?)
            } else {
                None
            };

            // For deduplication
            let mut seen_hashes: std::collections::HashSet<u64> = std::collections::HashSet::new();

            let validated_filter_output =
                crate::validate_output_path(&output, "dataset filter output")?;
            let file = std::fs::File::open(&input)?;
            let reader = BufReader::new(file);
            let mut output_file = std::fs::File::create(&validated_filter_output)?;

            let mut total = 0usize;
            let mut kept = 0usize;
            let mut filtered_tokens = 0usize;
            let mut filtered_pattern = 0usize;
            let mut filtered_dedup = 0usize;
            let mut filtered_incomplete = 0usize;

            for line in reader.lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                total += 1;

                let json: serde_json::Value = serde_json::from_str(&line)?;

                // Extract text for filtering
                let text = extract_text_from_sample(&json);

                // Check completeness for ShareGPT
                if complete_only {
                    if let Some(convs) = json.get("conversations").and_then(|v| v.as_array()) {
                        let has_human = convs
                            .iter()
                            .any(|c| c.get("from").and_then(|v| v.as_str()) == Some("human"));
                        let has_gpt = convs
                            .iter()
                            .any(|c| c.get("from").and_then(|v| v.as_str()) == Some("gpt"));
                        if !has_human || !has_gpt {
                            filtered_incomplete += 1;
                            continue;
                        }
                    }
                }

                // Token length filtering
                if let Some(ref tok) = tokenizer {
                    let tokens = tok.encode(&text)?;
                    let len = tokens.len();
                    if let Some(min) = min_tokens {
                        if len < min {
                            filtered_tokens += 1;
                            continue;
                        }
                    }
                    if let Some(max) = max_tokens {
                        if len > max {
                            filtered_tokens += 1;
                            continue;
                        }
                    }
                }

                // Pattern filtering
                if let Some(ref re) = regex {
                    let matches = re.is_match(&text);
                    let keep = if invert { !matches } else { matches };
                    if !keep {
                        filtered_pattern += 1;
                        continue;
                    }
                }

                // Deduplication
                if dedup {
                    use std::hash::{Hash, Hasher};
                    let mut hasher = std::collections::hash_map::DefaultHasher::new();
                    text.hash(&mut hasher);
                    let hash = hasher.finish();
                    if !seen_hashes.insert(hash) {
                        filtered_dedup += 1;
                        continue;
                    }
                }

                // Keep this sample
                writeln!(output_file, "{}", line)?;
                kept += 1;
            }

            println!("\n========================================");
            println!("  Filter Complete!");
            println!("========================================");
            println!("Total samples:        {}", total);
            println!(
                "Kept samples:         {} ({:.1}%)",
                kept,
                100.0 * kept as f64 / total as f64
            );
            if filtered_tokens > 0 {
                println!("Filtered (tokens):    {}", filtered_tokens);
            }
            if filtered_pattern > 0 {
                println!("Filtered (pattern):   {}", filtered_pattern);
            }
            if filtered_dedup > 0 {
                println!("Filtered (duplicate): {}", filtered_dedup);
            }
            if filtered_incomplete > 0 {
                println!("Filtered (incomplete):{}", filtered_incomplete);
            }
            println!("Output: {}", output);
            println!("========================================");
        }

        DatasetAction::Split {
            input,
            output_dir,
            val_ratio,
            test_ratio,
            seed,
            stratify,
        } => {
            println!("========================================");
            println!("  PMetal Dataset Split");
            println!("========================================");
            println!("Input:     {}", input);
            println!("Output:    {}", output_dir);
            println!("Val ratio: {:.2}", val_ratio);
            println!("Test ratio:{:.2}", test_ratio);
            println!("Seed:      {}", seed);
            if let Some(ref s) = stratify {
                println!("Stratify:  {}", s);
            }
            println!("========================================\n");

            // Validate ratios
            if val_ratio + test_ratio >= 1.0 {
                return Err(anyhow::anyhow!("val_ratio + test_ratio must be < 1.0"));
            }

            // Read all samples
            let file = std::fs::File::open(&input)?;
            let reader = BufReader::new(file);
            let mut samples: Vec<String> = Vec::new();

            for line in reader.lines() {
                let line = line?;
                if !line.trim().is_empty() {
                    samples.push(line);
                }
            }

            println!("Loaded {} samples", samples.len());

            // Shuffle
            use rand::SeedableRng;
            use rand::seq::SliceRandom;
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            samples.shuffle(&mut rng);

            // Calculate split indices
            let total = samples.len();
            let test_count = (total as f64 * test_ratio).round() as usize;
            let val_count = (total as f64 * val_ratio).round() as usize;
            let train_count = total - test_count - val_count;

            // Validate and create output directory
            let validated_split_dir =
                crate::validate_output_path(&output_dir, "dataset split output")?;
            std::fs::create_dir_all(&validated_split_dir)?;

            // Write splits
            let train_path = validated_split_dir.join("train.jsonl");
            let val_path = validated_split_dir.join("val.jsonl");
            let test_path = validated_split_dir.join("test.jsonl");

            let mut train_file = std::fs::File::create(&train_path)?;
            for sample in samples.iter().take(train_count) {
                writeln!(train_file, "{}", sample)?;
            }
            println!("Train: {} samples -> {}", train_count, train_path.display());

            let mut val_file = std::fs::File::create(&val_path)?;
            for sample in samples.iter().skip(train_count).take(val_count) {
                writeln!(val_file, "{}", sample)?;
            }
            println!("Val:   {} samples -> {}", val_count, val_path.display());

            if test_count > 0 {
                let mut test_file = std::fs::File::create(&test_path)?;
                for sample in samples.iter().skip(train_count + val_count) {
                    writeln!(test_file, "{}", sample)?;
                }
                println!("Test:  {} samples -> {}", test_count, test_path.display());
            }

            println!("\n========================================");
            println!("  Split Complete!");
            println!("========================================");
        }

        DatasetAction::Merge {
            inputs,
            output,
            shuffle,
            seed,
            interleave,
            weights,
        } => {
            println!("========================================");
            println!("  PMetal Dataset Merge");
            println!("========================================");
            for (i, input) in inputs.iter().enumerate() {
                println!("Input {}: {}", i + 1, input);
            }
            println!("Output:     {}", output);
            println!("Shuffle:    {}", shuffle);
            println!("Interleave: {}", interleave);
            println!("========================================\n");

            // Parse weights if provided
            let weights_vec: Vec<f64> = if let Some(ref w) = weights {
                w.split(',')
                    .map(|s| s.trim().parse::<f64>().unwrap_or(1.0))
                    .collect()
            } else {
                vec![1.0; inputs.len()]
            };

            // Read all datasets
            let mut all_samples: Vec<Vec<String>> = Vec::new();
            for input in &inputs {
                let file = std::fs::File::open(input)?;
                let reader = BufReader::new(file);
                let samples: Vec<String> = reader
                    .lines()
                    .map_while(Result::ok)
                    .filter(|l| !l.trim().is_empty())
                    .collect();
                println!("Loaded {} samples from {}", samples.len(), input);
                all_samples.push(samples);
            }

            let mut merged: Vec<String> = Vec::new();

            if interleave {
                // Interleave samples from each dataset
                let max_len = all_samples.iter().map(|s| s.len()).max().unwrap_or(0);
                for i in 0..max_len {
                    for (dataset_idx, samples) in all_samples.iter().enumerate() {
                        let weight = weights_vec.get(dataset_idx).copied().unwrap_or(1.0);
                        if i < samples.len() && rand::random::<f64>() < weight {
                            merged.push(samples[i].clone());
                        }
                    }
                }
            } else {
                // Simple concatenation with optional weighting (sampling)
                for (dataset_idx, samples) in all_samples.iter().enumerate() {
                    let weight = weights_vec.get(dataset_idx).copied().unwrap_or(1.0);
                    if weight >= 1.0 {
                        // Include all samples, possibly multiple times
                        let repeat = weight.floor() as usize;
                        for _ in 0..repeat.max(1) {
                            merged.extend(samples.iter().cloned());
                        }
                    } else {
                        // Sample a fraction
                        use rand::SeedableRng;
                        let mut rng = rand::rngs::StdRng::seed_from_u64(seed + dataset_idx as u64);
                        for sample in samples {
                            if rand::RngExt::random::<f64>(&mut rng) < weight {
                                merged.push(sample.clone());
                            }
                        }
                    }
                }
            }

            // Shuffle if requested
            if shuffle {
                use rand::SeedableRng;
                use rand::seq::SliceRandom;
                let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
                merged.shuffle(&mut rng);
                println!("Shuffled with seed {}", seed);
            }

            // Write output
            let mut output_file = std::fs::File::create(&output)?;
            for sample in &merged {
                writeln!(output_file, "{}", sample)?;
            }

            println!("\n========================================");
            println!("  Merge Complete!");
            println!("========================================");
            println!("Total samples: {}", merged.len());
            println!("Output: {}", output);
            println!("========================================");
        }

        DatasetAction::Sample {
            input,
            output,
            num,
            seed,
        } => {
            println!("========================================");
            println!("  PMetal Dataset Sample");
            println!("========================================");
            println!("Input:   {}", input);
            println!("Output:  {}", output);
            println!("Samples: {}", num);
            println!("Seed:    {}", seed);
            println!("========================================\n");

            // Read all samples
            let file = std::fs::File::open(&input)?;
            let reader = BufReader::new(file);
            let mut samples: Vec<String> = reader
                .lines()
                .map_while(Result::ok)
                .filter(|l| !l.trim().is_empty())
                .collect();

            println!("Loaded {} samples", samples.len());

            // Shuffle and take first N
            use rand::SeedableRng;
            use rand::seq::SliceRandom;
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            samples.shuffle(&mut rng);

            let take_count = num.min(samples.len());
            let mut output_file = std::fs::File::create(&output)?;
            for sample in samples.iter().take(take_count) {
                writeln!(output_file, "{}", sample)?;
            }

            println!("\n========================================");
            println!("  Sample Complete!");
            println!("========================================");
            println!("Sampled {} of {} samples", take_count, samples.len());
            println!("Output: {}", output);
            println!("========================================");
        }

        DatasetAction::Template {
            input,
            output,
            template,
            system,
            model,
            add_generation_prompt,
            mask_prompt: _,
        } => {
            println!("========================================");
            println!("  PMetal Chat Template");
            println!("========================================");
            println!("Input:    {}", input);
            println!("Output:   {}", output);
            println!("Template: {:?}", template);
            if let Some(ref s) = system {
                println!("System:   {}", s);
            }
            if add_generation_prompt {
                println!("Add generation prompt: yes");
            }
            println!("========================================\n");

            // Resolve EOS token from the model's tokenizer when a model is
            // provided and we are in training mode (not generation prompt).
            let template_eos_token: Option<String> = if !add_generation_prompt {
                if let Some(ref model_id) = model {
                    let model_dir =
                        if model_id.contains('/') && !std::path::Path::new(model_id).exists() {
                            pmetal_hub::download_model(model_id, None, None).await?
                        } else {
                            std::path::PathBuf::from(model_id)
                        };
                    match Tokenizer::from_model_dir(&model_dir) {
                        Ok(tok) => {
                            let eos = tok.eos_token_str();
                            if let Some(ref s) = eos {
                                println!("EOS token: {:?}", s);
                            }
                            eos
                        }
                        Err(e) => {
                            eprintln!(
                                "Warning: could not load tokenizer for EOS resolution: {}",
                                e
                            );
                            None
                        }
                    }
                } else {
                    None
                }
            } else {
                None
            };

            let file = std::fs::File::open(&input)?;
            let reader = BufReader::new(file);
            let mut output_file = std::fs::File::create(&output)?;

            let mut total = 0usize;
            let mut templated = 0usize;

            for line in reader.lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                total += 1;

                let json: serde_json::Value = serde_json::from_str(&line)?;

                // Extract conversations
                let conversations =
                    if let Some(convs) = json.get("conversations").and_then(|v| v.as_array()) {
                        convs
                            .iter()
                            .filter_map(|c| {
                                let from = c.get("from")?.as_str()?;
                                let value = c.get("value")?.as_str()?;
                                Some((from.to_string(), value.to_string()))
                            })
                            .collect::<Vec<_>>()
                    } else if let Some(inst) = json.get("instruction").and_then(|v| v.as_str()) {
                        // Convert Alpaca to conversations
                        let input_text = json.get("input").and_then(|v| v.as_str()).unwrap_or("");
                        let output_text = json.get("output").and_then(|v| v.as_str()).unwrap_or("");
                        let user_msg = if input_text.is_empty() {
                            inst.to_string()
                        } else {
                            format!("{}\n\n{}", inst, input_text)
                        };
                        vec![
                            ("human".to_string(), user_msg),
                            ("gpt".to_string(), output_text.to_string()),
                        ]
                    } else if let Some(text) = json.get("text").and_then(|v| v.as_str()) {
                        // Already raw text, just wrap it
                        writeln!(output_file, "{}", serde_json::json!({"text": text}))?;
                        templated += 1;
                        continue;
                    } else {
                        // Skip samples without recognizable format
                        continue;
                    };

                // Apply chat template
                let formatted = format_conversations(
                    &template,
                    &conversations,
                    system.as_deref(),
                    add_generation_prompt,
                    template_eos_token.as_deref(),
                );

                // Write output
                let out_json = serde_json::json!({ "text": formatted });
                writeln!(output_file, "{}", out_json)?;
                templated += 1;
            }

            println!("\n========================================");
            println!("  Template Complete!");
            println!("========================================");
            println!("Total samples:     {}", total);
            println!("Templated samples: {}", templated);
            println!("Output: {}", output);
            println!("========================================");
        }

        DatasetAction::Prepare {
            dataset,
            output_dir,
            model,
            template,
            max_seq_len,
            val_ratio,
            seed,
            no_dedup,
            columns,
        } => {
            println!("========================================");
            println!("  PMetal Dataset Prepare");
            println!("========================================");
            println!("Dataset:     {}", dataset);
            println!("Output:      {}", output_dir);
            println!("Model:       {}", model);
            println!("Template:    {:?}", template);
            println!("Max seq len: {}", max_seq_len);
            println!("Val ratio:   {:.2}", val_ratio);
            println!("Seed:        {}", seed);
            if let Some(ref col_str) = columns {
                println!("Columns:     {}", col_str);
            }
            println!("========================================\n");

            // Create output directory
            std::fs::create_dir_all(&output_dir)?;

            // Step 1: Download or load dataset
            println!("[1/5] Loading dataset...");
            let raw_path = format!("{}/raw.jsonl", output_dir);

            if dataset.contains('/') && !std::path::Path::new(&dataset).exists() {
                // HuggingFace dataset
                let parquet_paths =
                    pmetal_hub::download_dataset_parquet(&dataset, "train", None, None).await?;

                // Convert to JSONL
                let mut output_file = std::fs::File::create(&raw_path)?;
                let mut total_rows = 0usize;

                for parquet_path in &parquet_paths {
                    use arrow_json::writer::LineDelimitedWriter;
                    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

                    let file = std::fs::File::open(parquet_path)?;
                    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
                    let reader = builder.build()?;

                    for batch_result in reader {
                        let batch = batch_result?;

                        let mut json_buf = Vec::new();
                        {
                            let mut json_writer = LineDelimitedWriter::new(&mut json_buf);
                            json_writer.write(&batch)?;
                            json_writer.finish()?;
                        }

                        for line in std::str::from_utf8(&json_buf)?.lines() {
                            if line.trim().is_empty() {
                                continue;
                            }

                            let mut obj: serde_json::Value = serde_json::from_str(line)?;

                            // Convert messages to ShareGPT format if needed
                            if let Some(serde_json::Value::Array(msgs)) =
                                obj.get("messages").cloned()
                            {
                                let conversations: Vec<_> = msgs
                                    .iter()
                                    .filter_map(|m| {
                                        let role = m.get("role")?.as_str()?;
                                        let content = m.get("content")?.as_str()?;
                                        let from = match role {
                                            "user" => "human",
                                            "assistant" => "gpt",
                                            "system" => "system",
                                            _ => role,
                                        };
                                        Some(serde_json::json!({
                                            "from": from,
                                            "value": content
                                        }))
                                    })
                                    .collect();

                                if let Some(obj_map) = obj.as_object_mut() {
                                    obj_map.remove("messages");
                                    obj_map.insert(
                                        "conversations".to_string(),
                                        serde_json::Value::Array(conversations),
                                    );
                                }
                            }

                            writeln!(output_file, "{}", serde_json::to_string(&obj)?)?;
                            total_rows += 1;
                        }
                    }
                }
                println!("  Downloaded {} samples", total_rows);
            } else {
                // Local file - copy
                std::fs::copy(&dataset, &raw_path)?;
                println!("  Copied local dataset");
            }

            // Step 2: Load tokenizer
            println!("[2/5] Loading tokenizer...");
            let model_path = if model.contains('/') && !std::path::Path::new(&model).exists() {
                pmetal_hub::download_model(&model, None, None).await?
            } else {
                std::path::PathBuf::from(&model)
            };
            let tokenizer = Tokenizer::from_model_dir(&model_path)?;
            println!("  Loaded tokenizer from {}", model_path.display());

            // Resolve the model's true EOS token string from the tokenizer
            // vocabulary.  This is authoritative: e.g. Qwen3 resolves to
            // `<|endoftext|>` (ID 151643) rather than `<|im_end|>` (turn
            // delimiter).  Training sequences must end with this token so
            // the model learns to terminate generation.
            let eos_token_str: Option<String> = tokenizer.eos_token_str();
            if let Some(ref eos) = eos_token_str {
                println!("  EOS token:  {:?}", eos);
            } else {
                println!(
                    "  EOS token:  (not found — training sequences will not have EOS appended)"
                );
            }

            // Step 3: Apply template and filter
            println!("[3/5] Applying template and filtering...");
            let templated_path = format!("{}/templated.jsonl", output_dir);
            let raw_file = std::fs::File::open(&raw_path)?;
            let raw_reader = BufReader::new(raw_file);
            let mut templated_file = std::fs::File::create(&templated_path)?;

            // Parse column mapping once: "target=source,..." => { source -> target }
            // The user writes "target=source" meaning "rename source column to target".
            let col_map: Option<std::collections::HashMap<String, String>> =
                columns.as_deref().map(|s| {
                    s.split(',')
                        .filter_map(|pair| {
                            let mut parts = pair.splitn(2, '=');
                            let target = parts.next()?.trim().to_string();
                            let source = parts.next()?.trim().to_string();
                            Some((source, target))
                        })
                        .collect()
                });

            let mut seen_hashes: std::collections::HashSet<u64> = std::collections::HashSet::new();
            let mut total = 0usize;
            let mut kept = 0usize;
            let mut filtered_long = 0usize;
            let mut filtered_dedup = 0usize;

            for line in raw_reader.lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                total += 1;

                let raw_json: serde_json::Value = serde_json::from_str(&line)?;

                // Apply column mapping if provided: rename source keys to target keys.
                let json = if let Some(ref map) = col_map {
                    let mut obj = serde_json::Map::new();
                    if let Some(src_obj) = raw_json.as_object() {
                        for (k, v) in src_obj {
                            let new_key = map.get(k.as_str()).cloned().unwrap_or_else(|| k.clone());
                            obj.insert(new_key, v.clone());
                        }
                    }
                    serde_json::Value::Object(obj)
                } else {
                    raw_json
                };

                // Extract conversations
                let conversations =
                    if let Some(convs) = json.get("conversations").and_then(|v| v.as_array()) {
                        convs
                            .iter()
                            .filter_map(|c| {
                                let from = c.get("from")?.as_str()?;
                                let value = c.get("value")?.as_str()?;
                                Some((from.to_string(), value.to_string()))
                            })
                            .collect::<Vec<_>>()
                    } else if let Some(problem) = json.get("problem").and_then(|v| v.as_str()) {
                        // Reasoning format: problem / thinking / solution
                        let thinking = json.get("thinking").and_then(|v| v.as_str()).unwrap_or("");
                        let solution = json.get("solution").and_then(|v| v.as_str()).unwrap_or("");
                        let assistant_msg = if thinking.is_empty() {
                            solution.to_string()
                        } else {
                            format!("<think>\n{}\n</think>\n\n{}", thinking, solution)
                        };
                        vec![
                            ("human".to_string(), problem.to_string()),
                            ("gpt".to_string(), assistant_msg),
                        ]
                    } else if let Some(inst) = json.get("instruction").and_then(|v| v.as_str()) {
                        let input_text = json.get("input").and_then(|v| v.as_str()).unwrap_or("");
                        let output_text = json.get("output").and_then(|v| v.as_str()).unwrap_or("");
                        let user_msg = if input_text.is_empty() {
                            inst.to_string()
                        } else {
                            format!("{}\n\n{}", inst, input_text)
                        };
                        vec![
                            ("human".to_string(), user_msg),
                            ("gpt".to_string(), output_text.to_string()),
                        ]
                    } else if let Some(text) = json.get("text").and_then(|v| v.as_str()) {
                        // Raw text path: append EOS if missing so the model
                        // learns to stop generating.
                        let text_with_eos = match eos_token_str.as_deref() {
                            Some(eos) if !text.ends_with(eos) => {
                                format!("{}{}", text, eos)
                            }
                            _ => text.to_string(),
                        };

                        // Check length
                        let tokens = tokenizer.encode(&text_with_eos)?;
                        if tokens.len() > max_seq_len {
                            filtered_long += 1;
                            continue;
                        }

                        // Check dedup
                        if !no_dedup {
                            use std::hash::{Hash, Hasher};
                            let mut hasher = std::collections::hash_map::DefaultHasher::new();
                            text_with_eos.hash(&mut hasher);
                            let hash = hasher.finish();
                            if !seen_hashes.insert(hash) {
                                filtered_dedup += 1;
                                continue;
                            }
                        }

                        writeln!(
                            templated_file,
                            "{}",
                            serde_json::json!({"text": text_with_eos})
                        )?;
                        kept += 1;
                        continue;
                    } else {
                        continue;
                    };

                // Apply template (training mode: add_generation_prompt=false,
                // EOS token appended by format_conversations).
                let formatted = format_conversations(
                    &template,
                    &conversations,
                    None,
                    false,
                    eos_token_str.as_deref(),
                );

                // Check token length
                let tokens = tokenizer.encode(&formatted)?;
                if tokens.len() > max_seq_len {
                    filtered_long += 1;
                    continue;
                }

                // Check dedup
                if !no_dedup {
                    use std::hash::{Hash, Hasher};
                    let mut hasher = std::collections::hash_map::DefaultHasher::new();
                    formatted.hash(&mut hasher);
                    let hash = hasher.finish();
                    if !seen_hashes.insert(hash) {
                        filtered_dedup += 1;
                        continue;
                    }
                }

                writeln!(templated_file, "{}", serde_json::json!({"text": formatted}))?;
                kept += 1;
            }

            println!(
                "  Total: {}, Kept: {}, Filtered (long): {}, Filtered (dup): {}",
                total, kept, filtered_long, filtered_dedup
            );

            // Step 4: Split
            println!("[4/5] Splitting dataset...");
            let templated_file = std::fs::File::open(&templated_path)?;
            let templated_reader = BufReader::new(templated_file);
            let mut samples: Vec<String> = templated_reader
                .lines()
                .map_while(Result::ok)
                .filter(|l| !l.trim().is_empty())
                .collect();

            use rand::SeedableRng;
            use rand::seq::SliceRandom;
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            samples.shuffle(&mut rng);

            let val_count = (samples.len() as f64 * val_ratio).round() as usize;
            let train_count = samples.len() - val_count;

            let train_path = format!("{}/train.jsonl", output_dir);
            let val_path = format!("{}/val.jsonl", output_dir);

            let mut train_file = std::fs::File::create(&train_path)?;
            for sample in samples.iter().take(train_count) {
                writeln!(train_file, "{}", sample)?;
            }

            let mut val_file = std::fs::File::create(&val_path)?;
            for sample in samples.iter().skip(train_count) {
                writeln!(val_file, "{}", sample)?;
            }

            println!("  Train: {} samples", train_count);
            println!("  Val:   {} samples", val_count);

            // Step 5: Statistics
            println!("[5/5] Computing statistics...");
            let train_file = std::fs::File::open(&train_path)?;
            let train_reader = BufReader::new(train_file);
            let mut token_lengths: Vec<usize> = Vec::new();

            for line in train_reader.lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                let json: serde_json::Value = serde_json::from_str(&line)?;
                if let Some(text) = json.get("text").and_then(|v| v.as_str()) {
                    let tokens = tokenizer.encode(text)?;
                    token_lengths.push(tokens.len());
                }
            }

            token_lengths.sort();

            if token_lengths.is_empty() {
                println!("\n  No samples to compute statistics for.");
                println!(
                    "  Check your dataset format — supported: conversations, problem/solution, instruction/output, text"
                );
                println!("  Use --columns to remap custom column names.");
                return Ok(());
            }

            let p50 = token_lengths[token_lengths.len() / 2];
            let p95 = token_lengths[(token_lengths.len() as f64 * 0.95) as usize];
            let max_len = *token_lengths.last().unwrap_or(&0);

            println!("\n========================================");
            println!("  Prepare Complete!");
            println!("========================================");
            println!("Train samples: {}", train_count);
            println!("Val samples:   {}", val_count);
            println!("Token P50:     {}", p50);
            println!("Token P95:     {}", p95);
            println!("Token Max:     {}", max_len);
            println!("\nOutput files:");
            println!("  {}", train_path);
            println!("  {}", val_path);
            println!("========================================");
        }

        DatasetAction::Formats => {
            println!("========================================");
            println!("  PMetal Supported Formats & Templates");
            println!("========================================\n");

            println!("INPUT FORMATS:");
            println!("--------------");
            println!("1. ShareGPT (recommended):");
            println!(
                r#"   {{"conversations": [{{"from": "human", "value": "..."}}, {{"from": "gpt", "value": "..."}}]}}"#
            );
            println!();
            println!("2. Alpaca:");
            println!(r#"   {{"instruction": "...", "input": "...", "output": "..."}}"#);
            println!();
            println!("3. OpenAI Messages:");
            println!(
                r#"   {{"messages": [{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}]}}"#
            );
            println!();
            println!("4. Simple text:");
            println!(r#"   {{"text": "The full formatted text for training"}}"#);
            println!();

            println!("CHAT TEMPLATES:");
            println!("---------------");
            println!("1. chatml (default):");
            println!("   <|im_start|>system");
            println!("   {{system_message}}<|im_end|>");
            println!("   <|im_start|>user");
            println!("   {{user_message}}<|im_end|>");
            println!("   <|im_start|>assistant");
            println!("   {{assistant_message}}<|im_end|>");
            println!();
            println!("2. llama3:");
            println!("   <|start_header_id|>system<|end_header_id|>");
            println!("   {{system_message}}<|eot_id|>");
            println!("   <|start_header_id|>user<|end_header_id|>");
            println!("   {{user_message}}<|eot_id|>");
            println!("   <|start_header_id|>assistant<|end_header_id|>");
            println!("   {{assistant_message}}<|eot_id|>");
            println!();
            println!("3. llama2:");
            println!("   [INST] <<SYS>>{{system_message}}<</SYS>>");
            println!("   {{user_message}} [/INST] {{assistant_message}} </s>");
            println!();
            println!("4. mistral:");
            println!("   <s>[INST] {{user_message}} [/INST] {{assistant_message}}</s>");
            println!();
            println!("5. phi:");
            println!("   <|system|>{{system_message}}<|end|>");
            println!("   <|user|>{{user_message}}<|end|>");
            println!("   <|assistant|>{{assistant_message}}<|end|>");
            println!();
            println!("6. gemma:");
            println!("   <start_of_turn>user");
            println!("   {{user_message}}<end_of_turn>");
            println!("   <start_of_turn>model");
            println!("   {{assistant_message}}<end_of_turn>");
            println!();
            println!("7. qwen:");
            println!("   Same as ChatML");
            println!();
            println!("8. raw:");
            println!("   No template, concatenates messages with newlines");
            println!();
            println!("9. auto:");
            println!("   Uses the tokenizer's built-in chat_template");
            println!();

            println!("EXAMPLE WORKFLOW:");
            println!("-----------------");
            println!("# Download and preview");
            println!("pmetal dataset preview tatsu-lab/alpaca --num 3");
            println!();
            println!("# Full preparation pipeline");
            println!("pmetal dataset prepare tatsu-lab/alpaca \\");
            println!("  --output-dir ./alpaca_prepared \\");
            println!("  --model unsloth/Qwen3-0.6B \\");
            println!("  --template chatml \\");
            println!("  --max-seq-len 2048 \\");
            println!("  --val-ratio 0.05");
            println!();
            println!("# Or step by step:");
            println!("pmetal dataset download tatsu-lab/alpaca -o raw.jsonl");
            println!(
                "pmetal dataset filter -i raw.jsonl -o filtered.jsonl --model ... --max-tokens 2048 --dedup"
            );
            println!(
                "pmetal dataset template -i filtered.jsonl -o templated.jsonl --template chatml"
            );
            println!("pmetal dataset split -i templated.jsonl -o ./splits --val-ratio 0.1");
            println!();
            println!("# Analyze your data");
            println!("pmetal dataset analyze -p train.jsonl --model unsloth/Qwen3-0.6B");
        }
    }

    Ok(())
}

/// Extract text content from a sample for filtering/analysis.
fn extract_text_from_sample(json: &serde_json::Value) -> String {
    if let Some(t) = json.get("text").and_then(|v| v.as_str()) {
        t.to_string()
    } else if let Some(convs) = json.get("conversations").and_then(|v| v.as_array()) {
        convs
            .iter()
            .filter_map(|c| c.get("value").and_then(|v| v.as_str()))
            .collect::<Vec<_>>()
            .join(" ")
    } else if let Some(inst) = json.get("instruction").and_then(|v| v.as_str()) {
        let input = json.get("input").and_then(|v| v.as_str()).unwrap_or("");
        let output = json.get("output").and_then(|v| v.as_str()).unwrap_or("");
        format!("{} {} {}", inst, input, output)
    } else if let Some(msgs) = json.get("messages").and_then(|v| v.as_array()) {
        msgs.iter()
            .filter_map(|m| m.get("content").and_then(|v| v.as_str()))
            .collect::<Vec<_>>()
            .join(" ")
    } else {
        String::new()
    }
}

/// Format conversations with a chat template.
///
/// When `eos_token` is `Some` and `add_generation_prompt` is `false` (i.e.
/// training mode), the EOS token is appended after the final turn so the model
/// learns to terminate generation.  Callers in inference / generation-prompt
/// mode should pass `None` or `add_generation_prompt = true`.
pub(crate) fn format_conversations(
    template: &ChatTemplate,
    conversations: &[(String, String)],
    system_msg: Option<&str>,
    add_generation_prompt: bool,
    eos_token: Option<&str>,
) -> String {
    let mut output = String::new();

    match template {
        ChatTemplate::Chatml | ChatTemplate::Qwen => {
            if let Some(sys) = system_msg {
                output.push_str(&format!("<|im_start|>system\n{}<|im_end|>\n", sys));
            }
            for (role, content) in conversations {
                let role_name = match role.as_str() {
                    "human" | "user" => "user",
                    "gpt" | "assistant" => "assistant",
                    "system" => "system",
                    _ => role.as_str(),
                };
                output.push_str(&format!(
                    "<|im_start|>{}\n{}<|im_end|>\n",
                    role_name, content
                ));
            }
            if add_generation_prompt {
                output.push_str("<|im_start|>assistant\n");
            }
        }

        ChatTemplate::Llama3 => {
            output.push_str("<|begin_of_text|>");
            if let Some(sys) = system_msg {
                output.push_str(&format!(
                    "<|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>",
                    sys
                ));
            }
            for (role, content) in conversations {
                let role_name = match role.as_str() {
                    "human" | "user" => "user",
                    "gpt" | "assistant" => "assistant",
                    "system" => "system",
                    _ => role.as_str(),
                };
                output.push_str(&format!(
                    "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
                    role_name, content
                ));
            }
            if add_generation_prompt {
                output.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
            }
        }

        ChatTemplate::Llama2 => {
            output.push_str("<s>");
            let mut first_user = true;
            for (role, content) in conversations {
                match role.as_str() {
                    "human" | "user" => {
                        if first_user {
                            if let Some(sys) = system_msg {
                                output.push_str(&format!(
                                    "[INST] <<SYS>>\n{}\n<</SYS>>\n\n{} [/INST] ",
                                    sys, content
                                ));
                            } else {
                                output.push_str(&format!("[INST] {} [/INST] ", content));
                            }
                            first_user = false;
                        } else {
                            output.push_str(&format!("<s>[INST] {} [/INST] ", content));
                        }
                    }
                    "gpt" | "assistant" => {
                        output.push_str(&format!("{} </s>", content));
                    }
                    _ => {}
                }
            }
            if add_generation_prompt {
                output.push_str("[INST] ");
            }
        }

        ChatTemplate::Mistral => {
            output.push_str("<s>");
            for (role, content) in conversations {
                match role.as_str() {
                    "human" | "user" => {
                        output.push_str(&format!("[INST] {} [/INST]", content));
                    }
                    "gpt" | "assistant" => {
                        output.push_str(&format!("{}</s>", content));
                    }
                    _ => {}
                }
            }
            if add_generation_prompt {
                output.push_str("[INST] ");
            }
        }

        ChatTemplate::Phi => {
            if let Some(sys) = system_msg {
                output.push_str(&format!("<|system|>\n{}<|end|>\n", sys));
            }
            for (role, content) in conversations {
                let role_name = match role.as_str() {
                    "human" | "user" => "user",
                    "gpt" | "assistant" => "assistant",
                    "system" => "system",
                    _ => role.as_str(),
                };
                output.push_str(&format!("<|{}|>\n{}<|end|>\n", role_name, content));
            }
            if add_generation_prompt {
                output.push_str("<|assistant|>\n");
            }
        }

        ChatTemplate::Gemma => {
            for (role, content) in conversations {
                let role_name = match role.as_str() {
                    "human" | "user" => "user",
                    "gpt" | "assistant" => "model",
                    _ => role.as_str(),
                };
                output.push_str(&format!(
                    "<start_of_turn>{}\n{}<end_of_turn>\n",
                    role_name, content
                ));
            }
            if add_generation_prompt {
                output.push_str("<start_of_turn>model\n");
            }
        }

        ChatTemplate::Raw => {
            for (_, content) in conversations {
                output.push_str(content);
                output.push('\n');
            }
        }

        ChatTemplate::Auto => {
            // For auto, we'd need to load the tokenizer's chat_template
            // For now, fall back to ChatML
            if let Some(sys) = system_msg {
                output.push_str(&format!("<|im_start|>system\n{}<|im_end|>\n", sys));
            }
            for (role, content) in conversations {
                let role_name = match role.as_str() {
                    "human" | "user" => "user",
                    "gpt" | "assistant" => "assistant",
                    _ => role.as_str(),
                };
                output.push_str(&format!(
                    "<|im_start|>{}\n{}<|im_end|>\n",
                    role_name, content
                ));
            }
            if add_generation_prompt {
                output.push_str("<|im_start|>assistant\n");
            }
        }
    }

    // Append EOS token at the end of training sequences so the model learns
    // to stop generating.  Skip in generation-prompt mode (inference) and for
    // Raw template (pre-formatted text may already contain EOS).
    if !add_generation_prompt && !matches!(template, ChatTemplate::Raw) {
        if let Some(eos) = eos_token {
            if !output.ends_with(eos) {
                output.push_str(eos);
            }
        }
    }

    output
}
