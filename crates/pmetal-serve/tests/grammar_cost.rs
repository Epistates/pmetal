//! Where a grammar's per-token cost goes, on a real vocabulary: building the
//! compiler, compiling a schema, and one step (mask, upload, logits). Opt-in:
//! needs only a Qwen3 checkpoint's tokenizer.
//!
//! ```text
//! PMETAL_QWEN3_DIR=<snapshot> cargo test --release -p pmetal-serve \
//!     --test grammar_cost -- --nocapture
//! ```

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use pmetal_runtime::InlineArray;
use pmetal_serve::structured::{ConstraintSpec, GrammarFactory};
use pmetal_serve::vocab::ByteVocab;

#[test]
fn grammar_cost_breakdown() {
    let Some(dir) = std::env::var_os("PMETAL_QWEN3_DIR").map(PathBuf::from) else {
        eprintln!("PMETAL_QWEN3_DIR not set; skipping");
        return;
    };
    let tokenizer = Arc::new(pmetal_data::Tokenizer::from_model_dir(&dir).unwrap());
    let started = Instant::now();
    let bytes = ByteVocab::from_tokenizer(&tokenizer).expect("byte vocabulary");
    println!(
        "byte vocabulary of {} tokens: {:?}",
        bytes.len(),
        started.elapsed()
    );
    let eos = tokenizer.eos_token_id().unwrap();
    let vocab = bytes.len().max(151_936);
    let started = Instant::now();
    let factory = GrammarFactory::new(Arc::clone(&tokenizer), &bytes, vocab, &[eos]).unwrap();
    println!("grammar factory: {:?}", started.elapsed());

    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"}, "age": {"type": "integer"},
            "tags": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["name", "age"]
    });
    let started = Instant::now();
    let constraint = factory
        .compile(&ConstraintSpec::JsonSchema(schema), &tokenizer, &[], false)
        .unwrap();
    println!("compile a schema: {:?}", started.elapsed());

    // Walk a JSON object token by token: the allowed token a hash picks.
    let mut processor = constraint.processor();
    let logits = InlineArray::from_f32_slice(&vec![0.0f32; vocab], &[1, vocab as i32]);
    logits.eval();
    let mut step_us = Vec::new();
    let mut allowed_counts = Vec::new();
    for step in 0..120 {
        let started = Instant::now();
        let masked = processor.process(&logits);
        masked.eval();
        step_us.push(started.elapsed().as_micros() as f64);
        let host = masked.as_slice::<f32>();
        let allowed: Vec<u32> = (0..vocab as u32)
            .filter(|&i| host[i as usize] > f32::NEG_INFINITY)
            .collect();
        allowed_counts.push(allowed.len());
        if allowed.is_empty() {
            break;
        }
        let token = allowed[(step * 7919 + 13) % allowed.len()];
        processor.accept(token);
        if token == eos {
            break;
        }
    }
    step_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    allowed_counts.sort_unstable();
    println!(
        "one grammar step (mask, upload, add, eval) over {} steps: p50 {:.0} us, p95 {:.0} us, \
         max {:.0} us; tokens allowed per step: median {}",
        step_us.len(),
        step_us[step_us.len() / 2],
        step_us[step_us.len() * 95 / 100],
        step_us.last().unwrap(),
        allowed_counts[allowed_counts.len() / 2],
    );
    // The same eval with no grammar, for the fixed cost of an op.
    let started = Instant::now();
    for _ in 0..100 {
        let l = logits.add(&InlineArray::from_f32(0.0));
        l.eval();
    }
    println!(
        "an eval with no grammar: {:.0} us",
        started.elapsed().as_micros() as f64 / 100.0
    );
}
