//! Importance-matrix collection: run a model over calibration text and measure
//! the activations each of its weight matrices meets.
//!
//! The quantizers weigh their error per input channel by how large the
//! activations in that channel are (see [`pmetal_gguf::imatrix`]). This module
//! produces those weights. It runs the model's own forward over chunks of
//! calibration tokens with a [`Collector`] installed; the layers that multiply
//! by a weight report the activation they were given
//! ([`pmetal_runtime::activation_stats`]), and the sums, kept on the GPU in
//! `f32`, come back once at the end. Memory is the model's plus its widths,
//! whatever the amount of calibration data.
//!
//! The output logits are never computed (the head's *input* is what is
//! measured, and the graph is evaluated only as far as the sums need), which
//! spares the largest matmul of every chunk.
//!
//! Matrices are named as the quantizer names the tensors it writes (GGUF
//! names), so an [`IMatrix`] from here is looked up by the quantizer without
//! translation. A stack of routed experts is one matrix of
//! `experts * in_features` values with a token count per expert; only the
//! tokens an expert was routed to count towards it.
//!
//! What cannot be measured is reported, not hidden: a projection the forward
//! reaches by some path that does not report (an architecture whose experts
//! are not the shared routed layer, say) is listed in
//! [`ImatrixRun::unmeasured`] and quantizes without importance.

use std::collections::{HashMap, HashSet};

use pmetal_gguf::imatrix::IMatrix;
use pmetal_runtime::activation_stats::Collector;
use pmetal_runtime::compat::{Array, Exception, ModuleParametersExt};

use crate::DynamicModel;
use crate::weight_format::WeightLoader;

/// What a calibration run measured.
#[derive(Debug, Clone)]
pub struct ImatrixRun {
    /// The measured matrices, by GGUF tensor name.
    pub imatrix: IMatrix,
    /// Chunks run through the model.
    pub chunks: usize,
    /// Tokens in those chunks.
    pub tokens: usize,
    /// Weight matrices of the model the forward never reported on, by GGUF
    /// name (lookup tables excluded): they quantize without importance.
    pub unmeasured: Vec<String>,
    /// Observations the collector could not use. Anything but zero means the
    /// statistics of some matrix are incomplete.
    pub rejected: u64,
}

/// Run `model` over `chunks` of token ids, `batch_size` equal-length chunks to
/// a forward, and measure the activations at its weight matrices.
///
/// `progress` is called with `(chunks done, chunks in all)` after each
/// forward.
///
/// # Errors
///
/// The forward fails, a runtime op throws, or a weight cannot be registered.
pub fn collect_imatrix(
    model: &mut DynamicModel,
    chunks: &[Vec<u32>],
    batch_size: usize,
    mut progress: impl FnMut(usize, usize),
) -> Result<ImatrixRun, Exception> {
    // Every 2-D matrix and every stack of experts, under the name the GGUF
    // writer gives it.
    let params = model.flatten_params();
    let mut gguf_names: HashMap<String, String> = HashMap::new();
    let mut targets: Vec<(String, &Array)> = Vec::new();
    let mut sorted: Vec<_> = params.iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(b.0));
    for (name, array) in sorted {
        if !name.ends_with(".weight") || !matches!(array.ndim(), 2 | 3) {
            continue;
        }
        let gguf = WeightLoader::hf_to_gguf_name(name);
        // Two parameters under one GGUF name would be one matrix written
        // twice; the first wins and the other is left unmeasured.
        if gguf_names.insert(gguf.clone(), name.to_string()).is_some() {
            continue;
        }
        targets.push((gguf, array));
    }
    let mut collector = Collector::install(targets.iter().map(|(n, a)| (n.clone(), *a)))
        .map_err(Exception::from)?;
    drop(targets);

    let total = chunks.len();
    let (mut done, mut tokens) = (0usize, 0usize);
    let batch_size = batch_size.max(1);
    let mut next = 0;
    while next < total {
        // The chunks that make one forward: up to `batch_size` of one length.
        let len = chunks[next].len();
        let end = chunks[next..]
            .iter()
            .take(batch_size)
            .take_while(|c| c.len() == len)
            .count()
            + next;
        let batch = &chunks[next..end];
        if len > 0 {
            let flat: Vec<i32> = batch.iter().flatten().map(|&t| t as i32).collect();
            let ids = Array::from_slice(&flat, &[batch.len() as i32, len as i32]);
            // The logits are dropped unevaluated: the sums need the graph only
            // up to the head's input.
            let _logits = model.forward(&ids, None)?;
            collector.flush();
            pmetal_runtime::check_last_error().map_err(|e| Exception::custom(e.to_string()))?;
            tokens += batch.len() * len;
        }
        done += batch.len();
        next = end;
        progress(done, total);
    }

    let rejected = collector.rejected();
    let stats = collector.finish();

    let mut imatrix = IMatrix::new();
    imatrix.last_chunk = i32::try_from(done).ok();
    imatrix.chunk_size = chunks
        .iter()
        .map(Vec::len)
        .max()
        .and_then(|n| u32::try_from(n).ok());
    let mut measured: HashSet<&str> = HashSet::new();
    for s in &stats {
        let mean = s.mean_sq();
        if mean.iter().any(|v| !v.is_finite()) {
            return Err(Exception::custom(format!(
                "the activations of {} are not finite; the model overflowed on the calibration data",
                s.name
            )));
        }
        imatrix.insert_measured(
            s.name.clone(),
            mean,
            s.counts.iter().map(|&c| c as f32).collect(),
        );
        measured.insert(s.name.as_str());
    }
    let mut unmeasured: Vec<String> = gguf_names
        .keys()
        .filter(|g| !measured.contains(g.as_str()) && !is_lookup_table(g))
        .cloned()
        .collect();
    unmeasured.sort();

    Ok(ImatrixRun {
        imatrix,
        chunks: done,
        tokens,
        unmeasured,
        rejected,
    })
}

/// A matrix the forward reads rows of rather than multiplies by (the token
/// embedding of a model with its own output head): it has no input
/// activations to measure.
fn is_lookup_table(gguf_name: &str) -> bool {
    matches!(
        gguf_name,
        "token_embd.weight" | "position_embd.weight" | "token_types.weight"
    ) || gguf_name.contains("embed_tokens")
}
