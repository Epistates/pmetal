//! Contrastive and similarity-based loss functions for embedding training.
//!
//! Supports:
//! - InfoNCE / Multiple Negatives Ranking Loss — best for large batch sizes
//! - Triplet margin loss — anchor/positive/negative
//! - CoSENT — ranks pairs by their similarity labels
//! - Cosine similarity MSE loss — direct pairwise regression
//!
//! All losses operate on L2-normalised embeddings `[batch, dim]`.
//! Normalisation should be applied by the caller (use `pool::normalize_embeddings`).

use pmetal_bridge::compat::{Array, Dtype, Exception, ops};

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

/// CoSENT loss (Su Jianlin, *CoSENT: A more efficient sentence vector scheme
/// than Sentence-BERT*, 2022), as sentence-transformers' `CoSENTLoss` computes it.
///
/// Each row is a pair `(a_i, b_i)` with a similarity label `y_i` (binary or
/// graded). With `s_i = cos(a_i, b_i) / temperature`, every pair labelled
/// less similar than another should score lower:
///
/// ```text
/// loss = log(1 + Σ_{(i, j): y_i < y_j} exp(s_i − s_j))
/// ```
///
/// over all ordered pairs of rows in the batch. `temperature` is 1/λ; the
/// reference scale λ = 20 is `temperature = 0.05`. A batch whose labels are
/// all equal has no such pair, and its loss is log 1 = 0.
pub fn cosent_loss(
    embeddings_a: &Array,
    embeddings_b: &Array,
    labels: &Array,
    temperature: f32,
) -> Result<Array, Exception> {
    let f32_dtype = Dtype::Float32.as_i32();
    let scores = pairwise_cosine_similarity(
        &embeddings_a.as_dtype(f32_dtype),
        &embeddings_b.as_dtype(f32_dtype),
    )?
    .divide(&Array::from_f32(temperature)); // [batch]
    let labels = labels.as_dtype(f32_dtype);

    // diff[i, j] = s_i − s_j, kept where y_i < y_j and sent to −∞ elsewhere.
    let diff = scores.reshape(&[-1, 1]).subtract(&scores.reshape(&[1, -1]));
    let lower = labels
        .reshape(&[-1, 1])
        .less(&labels.reshape(&[1, -1]))
        .as_dtype(f32_dtype);
    let masked = diff.add(
        &Array::from_f32(1.0)
            .subtract(&lower)
            .multiply(&Array::from_f32(-1e12)),
    );

    // log(1 + Σ exp(·)): a logsumexp over the pairs with a 0 for the 1.
    let terms = ops::concatenate_axis(
        &[&Array::from_f32(0.0).reshape(&[1]), &masked.reshape(&[-1])],
        0,
    );
    Ok(terms.logsumexp_axis(0, false))
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
    fn cosent_loss_does_not_overflow() {
        // Scores 1/0.01 apart: s = [100, -100, 100] with labels [1, 1, 0], so
        // the negative row outranks the second positive by 200, where exp
        // overflows f32. The loss is log(1 + e^0 + e^200) ≈ 200, finite.
        let a = Array::from_slice(&[1.0f32, 0.0, 1.0, 0.0, 1.0, 0.0], &[3, 2]);
        let b = Array::from_slice(&[1.0f32, 0.0, -1.0, 0.0, 1.0, 0.0], &[3, 2]);
        let labels = Array::from_slice(&[1.0f32, 1.0, 0.0], &[3]);
        let loss = cosent_loss(&a, &b, &labels, 0.01).unwrap();
        loss.eval();
        pmetal_bridge::check_last_error().unwrap();
        let val = loss.item_f32();
        assert!((val - 200.0).abs() < 1e-3, "cosent loss: {val}");
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

    /// Pairs whose cosines are `cos`: a_i = [1, 0], b_i = [cos, sin].
    fn pairs_with_cosines(cos: &[f32]) -> (Array, Array) {
        let n = cos.len() as i32;
        let a: Vec<f32> = cos.iter().flat_map(|_| [1.0f32, 0.0]).collect();
        let b: Vec<f32> = cos
            .iter()
            .flat_map(|&c| [c, (1.0 - c * c).sqrt()])
            .collect();
        (
            Array::from_slice(&a, &[n, 2]),
            Array::from_slice(&b, &[n, 2]),
        )
    }

    fn cosent(cos: &[f32], labels: &[f32]) -> f32 {
        let (a, b) = pairs_with_cosines(cos);
        let labels = Array::from_slice(labels, &[labels.len() as i32]);
        let loss = cosent_loss(&a, &b, &labels, 0.05).unwrap();
        loss.eval();
        pmetal_bridge::check_last_error().unwrap();
        loss.item_f32()
    }

    #[test]
    fn cosent_matches_a_hand_computed_graded_batch() {
        // s = 20·cos = [18, 4, 10]. Pairs labelled lower than another:
        // (1, 0): 4 − 18, (1, 2): 4 − 10, (2, 0): 10 − 18.
        let want = (1.0 + (-14.0f64).exp() + (-6.0f64).exp() + (-8.0f64).exp()).ln();
        let got = cosent(&[0.9, 0.2, 0.5], &[1.0, 0.0, 0.5]);
        assert!((got as f64 - want).abs() < 1e-6, "got {got}, want {want}");
    }

    #[test]
    fn cosent_on_a_mixed_binary_batch_is_the_ranking_loss() {
        // A negative pair scoring above a positive one costs about
        // 20·(0.9 − 0.1) = 16; ranked the right way round it costs e^-16.
        let wrong = cosent(&[0.1, 0.9], &[1.0, 0.0]);
        let want = (1.0 + 16.0f64.exp()).ln();
        assert!(
            (wrong as f64 - want).abs() < 1e-4,
            "got {wrong}, want {want}"
        );
        let right = cosent(&[0.9, 0.1], &[1.0, 0.0]);
        assert!((right as f64 - (-16.0f64).exp().ln_1p()).abs() < 1e-6);
        // Two positives and two negatives, all ranked correctly by 0.5:
        // four pairs, each e^-10.
        let four = cosent(&[0.8, 0.7, 0.3, 0.2], &[1.0, 1.0, 0.0, 0.0]);
        let want =
            (1.0 + (-10.0f64).exp() + (-12.0f64).exp() + (-8.0f64).exp() + (-10.0f64).exp()).ln();
        assert!((four as f64 - want).abs() < 1e-6, "got {four}, want {want}");
        // Equal labels: no pair to rank, loss 0.
        assert_eq!(cosent(&[0.3, 0.9], &[1.0, 1.0]), 0.0);
    }
}
