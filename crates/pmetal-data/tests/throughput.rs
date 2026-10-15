//! Throughput of reading, tokenizing and packing a dataset. Ignored by default;
//! run with `cargo test --release -p pmetal-data --test throughput -- --ignored
//! --nocapture`. Set `PMETAL_BENCH_TOKENIZER` to a model directory to include a
//! real tokenizer in the loading figures.

use std::io::Write;
use std::path::Path;
use std::time::Instant;

use pmetal_data::rows::RowReader;
use pmetal_data::{
    DatasetFormat, DatasetLoadOptions, PackerConfig, SequencePacker, Tokenizer, TrainingDataset,
};
use serde_json::json;

const WORDS: [&str; 16] = [
    "the",
    "model",
    "learns",
    "from",
    "data",
    "and",
    "every",
    "token",
    "counts",
    "when",
    "we",
    "train",
    "on",
    "many",
    "short",
    "conversations",
];

/// A chat row of `turns` user/assistant pairs of a few dozen words.
fn chat_row(i: usize) -> serde_json::Value {
    let sentence = |seed: usize, words: usize| -> String {
        (0..words)
            .map(|w| WORDS[(seed * 7 + w * 3 + w / 5) % WORDS.len()])
            .collect::<Vec<_>>()
            .join(" ")
    };
    let turns = 1 + i % 3;
    let mut messages = Vec::new();
    for t in 0..turns {
        messages.push(json!({"role": "user", "content": sentence(i + t, 20 + (i + t) % 40)}));
        messages
            .push(json!({"role": "assistant", "content": sentence(i * 3 + t, 30 + (i * t) % 90)}));
    }
    json!({"messages": messages})
}

fn write_jsonl(path: &Path, rows: usize) {
    let mut out = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
    for i in 0..rows {
        writeln!(out, "{}", chat_row(i)).unwrap();
    }
}

fn write_parquet(jsonl: &Path, path: &Path) {
    use arrow::json::ReaderBuilder;
    use arrow::json::reader::infer_json_schema_from_seekable;
    let mut file = std::io::BufReader::new(std::fs::File::open(jsonl).unwrap());
    let (schema, _) = infer_json_schema_from_seekable(&mut file, Some(1000)).unwrap();
    let schema = std::sync::Arc::new(schema);
    let reader = ReaderBuilder::new(schema.clone())
        .with_batch_size(8192)
        .build(std::io::BufReader::new(std::fs::File::open(jsonl).unwrap()))
        .unwrap();
    let mut writer =
        parquet::arrow::ArrowWriter::try_new(std::fs::File::create(path).unwrap(), schema, None)
            .unwrap();
    for batch in reader {
        writer.write(&batch.unwrap()).unwrap();
    }
    writer.close().unwrap();
}

/// Resizing a photo-sized image with the Pillow-exact resampler.
#[test]
#[ignore = "benchmark"]
fn image_resize_throughput() {
    use image::{DynamicImage, RgbImage};
    use pmetal_data::image_processing::{FixedSizeImageProcessor, FixedSizeImageProcessorConfig};
    use pmetal_data::pillow_resample::{ResampleFilter, resize_rgb8, resize_rgb8_torchvision};

    let source = RgbImage::from_fn(4000, 3000, |x, y| {
        image::Rgb([
            (x * 7 % 256) as u8,
            (y * 11 % 256) as u8,
            ((x + y) * 3 % 256) as u8,
        ])
    });
    let time = |name: &str, f: &dyn Fn()| {
        f(); // warm up
        let start = Instant::now();
        for _ in 0..5 {
            f();
        }
        println!(
            "{name}: {:.1} ms per 4000x3000 image",
            start.elapsed().as_secs_f64() * 1e3 / 5.0
        );
    };
    time("pillow bicubic -> 560x560", &|| {
        std::hint::black_box(resize_rgb8(&source, 560, 560, ResampleFilter::Bicubic));
    });
    time("pillow bilinear -> 560x560", &|| {
        std::hint::black_box(resize_rgb8(&source, 560, 560, ResampleFilter::Bilinear));
    });
    time("torchvision bicubic -> 1008x756", &|| {
        std::hint::black_box(resize_rgb8_torchvision(
            &source,
            1008,
            756,
            ResampleFilter::Bicubic,
        ));
    });
    let processor = FixedSizeImageProcessor::new(FixedSizeImageProcessorConfig::default());
    let image = DynamicImage::ImageRgb8(source.clone());
    time("fixed-size processor -> 560x560", &|| {
        std::hint::black_box(processor.process_image(&image).unwrap());
    });
}

#[test]
#[ignore = "benchmark"]
fn reading_and_loading_throughput() {
    let rows = 200_000;
    let dir = tempfile::tempdir().unwrap();
    let jsonl = dir.path().join("chat.jsonl");
    let parquet = dir.path().join("chat.parquet");
    write_jsonl(&jsonl, rows);
    write_parquet(&jsonl, &parquet);
    let mb = |p: &Path| std::fs::metadata(p).unwrap().len() as f64 / 1e6;
    println!(
        "{rows} chat rows: jsonl {:.0} MB, parquet {:.0} MB",
        mb(&jsonl),
        mb(&parquet)
    );

    for (name, path) in [("jsonl", &jsonl), ("parquet", &parquet)] {
        let start = Instant::now();
        let n = RowReader::open(path)
            .unwrap()
            .filter(|row| row.is_ok())
            .count();
        assert_eq!(n, rows);
        let secs = start.elapsed().as_secs_f64();
        println!(
            "read {name}: {n} rows in {secs:.2}s ({:.0} rows/s)",
            n as f64 / secs
        );
    }

    let Some(dir_env) = std::env::var_os("PMETAL_BENCH_TOKENIZER") else {
        println!("PMETAL_BENCH_TOKENIZER not set: skipping tokenized loading");
        return;
    };
    let tokenizer = Tokenizer::from_model_dir(dir_env).expect("a tokenizer");
    let template = pmetal_data::chat_templates::ChatTemplate::chatml();
    for (name, path) in [("jsonl", &jsonl), ("parquet", &parquet)] {
        let start = Instant::now();
        let ds = TrainingDataset::load(
            path,
            &tokenizer,
            &DatasetLoadOptions::new(2048)
                .format(DatasetFormat::Auto)
                .template(Some(&template)),
        )
        .unwrap();
        let secs = start.elapsed().as_secs_f64();
        let tokens: usize = ds.samples().iter().map(|s| s.input_ids.len()).sum();
        println!(
            "load {name}: {} samples, {tokens} tokens in {secs:.2}s ({:.0} samples/s, {:.2}M tok/s)",
            ds.len(),
            ds.len() as f64 / secs,
            tokens as f64 / secs / 1e6,
        );

        let start = Instant::now();
        let (batches, stats) = SequencePacker::new(PackerConfig::with_max_length(4096))
            .pack_with_stats(ds.samples())
            .unwrap();
        let secs = start.elapsed().as_secs_f64();
        println!(
            "pack: {} rows in {secs:.2}s ({:.0} samples/s), {:.2}% efficient",
            batches.len(),
            ds.len() as f64 / secs,
            stats.efficiency * 100.0
        );
    }
}
