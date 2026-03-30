//! Contrastive and similarity-based loss functions for embedding training.
//!
//! Supports:
//! - InfoNCE / Multiple Negatives Ranking Loss — best for large batch sizes
//! - Triplet margin loss — anchor/positive/negative
//! - CoSENT (Cosine Sentence Embedding Training) — circle loss formulation
//! - Cosine similarity MSE loss — direct pairwise regression
//!
//! All losses operate on L2-normalised embeddings `[batch, dim]`.
//! Normalisation should be applied by the caller (use `pool::normalize_embeddings`).

use pmetal_bridge::compat::{Array, Dtype, Exception, indexing::IndexOp, module::Module, ops};

// ---------------------------------------------------------------------------
// Public loss functions
// ---------------------------------------------------------------------------

/// InfoNCE loss with in-batch negatives.
///
/// Given anchor embeddings `A` and positive embeddings `P` (both `[batch, dim]`):
/// - Similarity matrix `S = A @ P.T / temperature`
/// - Labels are the diagonal (each anchor matches its corresponding positive)
/// - `loss = cross_entropy(S, diag_labels)`
///
/// Every other positive in the batch acts as a hard negative.
pub fn info_nce_loss(
    anchors: &Array,
    positives: &Array,
    temperature: f32,
) -> Result<Array, Exception> {
    let batch_size = anchors.dim(0);

    // Similarity matrix [batch, batch]
    let sim = anchors.matmul(&positives.transpose_axes(&[1, 0]));
    let sim_scaled = sim.divide(&Array::from_f32(temperature));

    // Diagonal labels: [0, 1, 2, ..., batch-1]
    let labels: Vec<i32> = (0..batch_size).collect();
    let labels_arr = Array::from_slice(&labels, &[batch_size]);

    let ce = pmetal_bridge::compat::losses::CrossEntropy::new()?;
    let loss = ce.apply(&sim_scaled, &labels_arr)?;
    Ok(loss.mean(None))
}

/// Triplet margin loss using cosine distance.
///
/// `L = mean(max(0, margin + d(anchor, positive) - d(anchor, negative)))`
///
/// where `d(a, b) = 1 - cos_sim(a, b)` (cosine distance in `[0, 2]`).
pub fn triplet_loss(
    anchors: &Array,
    positives: &Array,
    negatives: &Array,
    margin: f32,
) -> Result<Array, Exception> {
    let pos_sim = pairwise_cosine_similarity(anchors, positives)?;
    let neg_sim = pairwise_cosine_similarity(anchors, negatives)?;

    // loss = max(0, margin - pos_sim + neg_sim)
    let diff = Array::from_f32(margin).subtract(&pos_sim).add(&neg_sim);
    let zero = Array::from_f32(0.0);
    let loss = ops::maximum(&diff, &zero);
    Ok(loss.mean(None))
}

/// CoSENT (Cosine Sentence Embedding Training) loss.
///
/// Circle-loss style formulation for pairwise binary labels.
/// For pairs labelled 1 (similar) we want high similarity, for 0 we want low.
///
/// Implementation follows the original CoSENT paper:
/// `loss = log(1 + Σ_{i≠j, y_i=1, y_j=0} exp(cos(i_neg) - cos(i_pos)) / T)`
///
/// The per-sample approximation used here:
/// `loss = log(1 + exp(lse(neg_logits) - lse(pos_logits)))`
/// averaged over the batch.
pub fn cosent_loss(
    embeddings_a: &Array,
    embeddings_b: &Array,
    labels: &Array,
    temperature: f32,
) -> Result<Array, Exception> {
    // Full similarity matrix [batch, batch]
    let sim = cosine_similarity_matrix(embeddings_a, embeddings_b)?;
    let sim_scaled = sim.divide(&Array::from_f32(temperature));

    // Cast labels to float for masking arithmetic
    let labels_f = labels.as_dtype(Dtype::Float32.as_i32());
    // labels_f is [batch]; broadcast to [batch, batch] via outer product
    let pos_mask = labels_f
        .reshape(&[-1, 1])
        .multiply(&labels_f.reshape(&[1, -1]));
    let neg_mask = Array::from_f32(1.0).subtract(&pos_mask);

    // Guard: CoSENT requires at least one positive pair; if all labels are 0 the
    // pos_logits matrix is entirely -1e9 and logsumexp(-1e9) - logsumexp(actual)
    // overflows to +inf.  Return zero loss instead.
    let mut pos_count = pos_mask.sum(None);
    pos_count.eval();
    if pos_count.item::<f32>() < 0.5 {
        return Ok(Array::from_f32(0.0));
    }

    // Mask out diagonal (self-similarity is always pos=1, which is trivial)
    let batch = embeddings_a.dim(0);
    let diag_mask = diagonal_zeros(batch)?;
    let pos_mask = pos_mask.multiply(&diag_mask);
    let neg_mask = neg_mask.multiply(&diag_mask);

    // Replace zeros with very negative values before logsumexp
    let neg_large = Array::from_f32(-1e9_f32);

    let pos_logits = sim_scaled.add(
        &Array::from_f32(1.0)
            .subtract(&pos_mask)
            .multiply(&neg_large),
    );
    let neg_logits = sim_scaled.add(
        &Array::from_f32(1.0)
            .subtract(&neg_mask)
            .multiply(&neg_large),
    );

    // LogSumExp over last axis for each row
    let pos_lse = pos_logits.logsumexp_axis(-1, false); // [batch]
    let neg_lse = neg_logits.logsumexp_axis(-1, false); // [batch]

    // Softplus: log(1 + exp(neg_lse - pos_lse))
    let diff = neg_lse.subtract(&pos_lse);
    // softplus(x) = log(1 + exp(x)) — use log1p for numerical stability
    let loss = ops::log1p(&diff.exp());
    Ok(loss.mean(None))
}

/// Multiple Negatives Ranking Loss (MNRL).
///
/// Equivalent to InfoNCE with cosine similarity. The most widely used loss
/// for training sentence transformers from (anchor, positive) pair data.
pub fn multiple_negatives_ranking_loss(
    anchors: &Array,
    positives: &Array,
    temperature: f32,
) -> Result<Array, Exception> {
    info_nce_loss(anchors, positives, temperature)
}

/// Cosine similarity MSE loss for similarity score regression.
///
/// `L = mean((cos_sim(a, b) - label)²)`
///
/// Labels should be in `[-1, 1]` or `[0, 1]` (common: STS benchmark uses `[-1, 1]`).
pub fn cosine_similarity_loss(
    embeddings_a: &Array,
    embeddings_b: &Array,
    labels: &Array,
) -> Result<Array, Exception> {
    let sim = pairwise_cosine_similarity(embeddings_a, embeddings_b)?;
    let labels_f = labels.as_dtype(Dtype::Float32.as_i32());
    let diff = sim.subtract(&labels_f);
    Ok(diff.square().mean(None))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Per-row cosine similarity between corresponding rows of two matrices.
///
/// Returns `[batch]` of similarity values in `[-1, 1]`.
pub(crate) fn pairwise_cosine_similarity(a: &Array, b: &Array) -> Result<Array, Exception> {
    let dot = a.multiply(b).sum_axes(&[-1], false); // [batch]
    let norm_a = a.square().sum_axes(&[-1], false).sqrt(); // [batch]
    let norm_b = b.square().sum_axes(&[-1], false).sqrt(); // [batch]
    let norms = norm_a.multiply(&norm_b);
    let norms = ops::maximum(&norms, &Array::from_f32(1e-8));
    Ok(dot.divide(&norms))
}

/// Full pairwise cosine similarity matrix between two sets of embeddings.
///
/// Returns `[batch_a, batch_b]` matrix.
pub(crate) fn cosine_similarity_matrix(a: &Array, b: &Array) -> Result<Array, Exception> {
    let norm_a = a.square().sum_axes(&[-1], true).sqrt();
    let norm_b = b.square().sum_axes(&[-1], true).sqrt();
    let a_normed = a.divide(&ops::maximum(&norm_a, &Array::from_f32(1e-8)));
    let b_normed = b.divide(&ops::maximum(&norm_b, &Array::from_f32(1e-8)));
    Ok(a_normed.matmul(&b_normed.transpose_axes(&[1, 0])))
}

/// Return a `[n, n]` float32 matrix that is 0.0 on the diagonal and 1.0 elsewhere.
///
/// Used to exclude trivial self-similarity pairs from CoSENT.
fn diagonal_zeros(n: i32) -> Result<Array, Exception> {
    let mut data = vec![1.0f32; (n * n) as usize];
    for i in 0..n as usize {
        data[i * n as usize + i] = 0.0;
    }
    Ok(Array::from_slice(&data, &[n, n]))
}

#[cfg(test)]
mod tests {
    use super::*;
    // IndexOp already imported via top-level use

    fn unit_embeddings(batch: i32, dim: i32) -> Array {
        // Each row is [1, 0, 0, ..., 0] (already unit-normed)
        let mut data = vec![0.0f32; (batch * dim) as usize];
        for b in 0..batch as usize {
            data[b * dim as usize] = 1.0;
        }
        Array::from_slice(&data, &[batch, dim])
    }

    #[test]
    fn test_info_nce_loss_perfect() {
        // When anchors == positives the diagonal similarities are maximal
        // and cross-entropy should be close to zero (bounded by temperature/log(N)).
        let emb = unit_embeddings(4, 16);
        let loss = info_nce_loss(&emb, &emb, 0.05).unwrap();
        let val: f32 = loss.item();
        // With temperature=0.05 and batch=4, minimum CE ≈ log(4) * 0.05 / 1.0
        // The exact value depends on the other rows — just check it's finite and positive.
        assert!(val.is_finite(), "loss should be finite");
        assert!(val >= 0.0, "loss should be non-negative");
    }

    #[test]
    fn test_pairwise_cosine_similarity() {
        // Identical vectors → similarity = 1.0
        let a = Array::from_slice(&[1.0f32, 0.0, 0.0, 1.0, 0.0, 0.0], &[2, 3]);
        let sim = pairwise_cosine_similarity(&a, &a).unwrap();
        sim.eval().unwrap();
        let flat = sim.flatten(0, -1);
        let v0: f32 = flat.index(0).item();
        let v1: f32 = flat.index(1).item();
        assert!((v0 - 1.0).abs() < 1e-5);
        assert!((v1 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_triplet_loss_zero_margin() {
        // When pos == neg, loss should be max(0, 0) = 0 (with margin=0)
        let anchor = Array::from_slice(&[1.0f32, 0.0, 0.0, 1.0, 0.0, 0.0], &[2, 3]);
        let pos = anchor.clone();
        let neg = anchor.clone();
        let loss = triplet_loss(&anchor, &pos, &neg, 0.0).unwrap();
        let val: f32 = loss.item();
        assert!(val.abs() < 1e-5, "loss should be ~0, got {}", val);
    }

    #[test]
    fn test_cosine_similarity_loss() {
        let a = Array::from_slice(&[1.0f32, 0.0], &[1, 2]);
        let b = Array::from_slice(&[1.0f32, 0.0], &[1, 2]);
        // cos_sim = 1.0, label = 1.0 → MSE = 0
        let labels = Array::from_slice(&[1.0f32], &[1]);
        let loss = cosine_similarity_loss(&a, &b, &labels).unwrap();
        let val: f32 = loss.item();
        assert!(val.abs() < 1e-5, "loss should be ~0, got {}", val);
    }
}
