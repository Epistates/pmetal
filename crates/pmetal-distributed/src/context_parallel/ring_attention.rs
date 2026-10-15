//! Ring attention for context parallelism.
//!
//! The sequence is cut between ranks ([`SequenceLayout`]); each rank holds a
//! local run of queries, keys and values. Blocks travel around the ring and
//! every rank computes partial attention of the blocks it meets against its
//! own, merging the partial results with an online softmax, so the final
//! output is mathematically full attention over the whole sequence. This is
//! an exact algorithm, not an approximation.
//!
//! # Algorithm (Pass-KV)
//!
//! ```text
//! rank r holds Q_r, K_r, V_r.  For step s in 0..N:
//!   1. the block in hand is (K_j, V_j) from rank j = (r - s) mod N
//!   2. start sending it to rank r+1 and receiving (K_j', V_j') from rank r-1
//!   3. while that is in flight, on the GPU: scores = Q_r K_j^T * scale,
//!      masked by token position when causal, giving the block's output O_j
//!      and log-sum-exp L_j (per query)
//!   4. merge:  L = log(exp(L_acc) + exp(L_j))
//!              O = O_acc exp(L_acc - L) + O_j exp(L_j - L)
//! ```
//!
//! Position, not rank, decides what a query may see: every block carries the
//! global positions of its tokens, so causal masking, uneven shares and the
//! zig-zag layout need no special cases. A block no query can see is neither
//! computed nor, past the last rank that wants it, sent.
//!
//! # Pass-Q
//!
//! When the queries are fewer than the keys (a short prompt over a long
//! cache) it is cheaper to move the queries. Each rank's query block then
//! travels the ring carrying its partial output and log-sum-exp, every rank
//! merging in its own keys, and arrives home after `N` hops.
//!
//! # Deadlock
//!
//! MLX's point-to-point `send` blocks until the neighbour reads, so a ring in
//! which every rank sends first deadlocks as soon as a block outgrows the
//! socket buffers. The exchange ([`crate::mlx_dist::exchange`]) orders sends
//! and receives by rank parity, and starts the transfer asynchronously so the
//! GPU computes the block in hand meanwhile.
//!
//! # Reference
//!
//! - Ring Attention with Blockwise Transformers (Liu et al., 2023)
//! - Striped Attention (Brandon et al., 2023)
//! - Context Parallelism for Scalable Million-Token Inference (Yang et al., 2024)
//!
//! Forward only: `send`/`recv` have no gradient, so this serves prefill and
//! inference, not training.

use super::sequence_split::SequenceLayout;
use crate::mlx_dist::exchange;
use crate::mlx_dist::group::DistributedGroup;
use pmetal_runtime::compat::{Array, Dtype, Exception, ops as mlx_ops};

/// Which blocks travel the ring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CPMode {
    /// Key/value blocks travel; each rank keeps its queries. The default for
    /// prefill, where queries and keys are the same length.
    PassKV,
    /// Query blocks (with their running output) travel; each rank keeps its
    /// keys. Cheaper when queries are much fewer than keys.
    PassQ,
}

/// How a ring-attention call is laid out.
#[derive(Debug, Clone, Copy)]
pub struct RingAttentionConfig {
    /// Attention scale, typically `1 / sqrt(head_dim)`.
    pub scale: f32,
    /// Mask keys at later positions than the query.
    pub causal: bool,
    /// Which blocks travel.
    pub mode: CPMode,
    /// How the sequence is cut between ranks.
    pub layout: SequenceLayout,
    /// Length of the whole sequence.
    pub total_len: usize,
}

/// A finite stand-in for minus infinity: it keeps fully masked rows free of
/// NaN (`-inf - -inf`), and `exp` of it is exactly zero.
const NEG: f32 = -1.0e30;

fn runtime_error(e: pmetal_runtime::RuntimeError) -> Exception {
    Exception::custom(e.to_string())
}

/// Ring attention forward pass.
///
/// # Arguments
///
/// * `query` — this rank's queries `[B, H, S_local, D]`
/// * `key`, `value` — this rank's keys and values `[B, Hkv, S_local, D]`
///   (`H` a multiple of `Hkv`: grouped-query attention)
/// * `config` — scale, causality, mode, layout and total sequence length
/// * `group` — the ring
///
/// `S_local` must be [`SequenceLayout::local_len`] for this rank, and tokens
/// must sit in the local tensor in [`SequenceLayout::local_positions`] order
/// ([`split_sequence_with_layout`](super::sequence_split::split_sequence_with_layout)
/// does that).
///
/// # Returns
///
/// This rank's attention output `[B, H, S_local, D]`, in the dtype of `query`.
pub fn ring_attention_forward(
    query: &Array,
    key: &Array,
    value: &Array,
    config: &RingAttentionConfig,
    group: &DistributedGroup,
) -> Result<Array, Exception> {
    let ring = Ring::new(query, key, value, config, group)?;
    let out = match config.mode {
        CPMode::PassKV => ring.pass_kv(query, key, value)?,
        CPMode::PassQ => ring.pass_q(query, key, value)?,
    };
    Ok(out.as_dtype(query.dtype().as_i32()))
}

/// The ring being run: sizes, positions and who needs which block.
struct Ring<'a> {
    group: &'a DistributedGroup,
    rank: usize,
    size: usize,
    config: RingAttentionConfig,
    /// Global positions of each rank's tokens.
    positions: Vec<Vec<u32>>,
    heads: i32,
    kv_heads: i32,
    head_dim: i32,
    batch: i32,
}

impl<'a> Ring<'a> {
    fn new(
        query: &Array,
        key: &Array,
        value: &Array,
        config: &RingAttentionConfig,
        group: &'a DistributedGroup,
    ) -> Result<Self, Exception> {
        let (rank, size) = (group.rank() as usize, group.size() as usize);
        if query.ndim() != 4 || key.ndim() != 4 || key.shape() != value.shape() {
            return Err(Exception::custom(format!(
                "ring attention wants q [B,H,S,D] and k, v [B,Hkv,S,D], got {:?}, {:?}, {:?}",
                query.shape(),
                key.shape(),
                value.shape()
            )));
        }
        let (b, h, s, d) = (
            query.shape()[0],
            query.shape()[1],
            query.shape()[2],
            query.shape()[3],
        );
        let hkv = key.shape()[1];
        if hkv == 0 || h % hkv != 0 || key.shape()[0] != b || key.shape()[3] != d {
            return Err(Exception::custom(format!(
                "ring attention: {h} query heads do not group over {hkv} KV heads \
                 (or batch/head_dim differ): q {:?}, k {:?}",
                query.shape(),
                key.shape()
            )));
        }
        let positions: Vec<Vec<u32>> = (0..size)
            .map(|r| config.layout.local_positions(config.total_len, r, size))
            .collect();
        if let Some(r) = positions.iter().position(Vec::is_empty) {
            return Err(Exception::custom(format!(
                "ring attention: {} tokens leave rank {r} of {size} with none",
                config.total_len
            )));
        }
        if key.shape()[2] != s || s as usize != positions[rank].len() {
            return Err(Exception::custom(format!(
                "ring attention: rank {rank} holds {s} query and {} key tokens but the {:?} \
                 layout of {} tokens gives it {}",
                key.shape()[2],
                config.layout,
                config.total_len,
                positions[rank].len()
            )));
        }
        Ok(Self {
            group,
            rank,
            size,
            config: *config,
            positions,
            heads: h,
            kv_heads: hkv,
            head_dim: d,
            batch: b,
        })
    }

    /// Can any query of rank `queries` see a key held by rank `keys`?
    fn visible(&self, queries: usize, keys: usize) -> bool {
        if !self.config.causal {
            return true;
        }
        let first_key = *self.positions[keys].iter().min().expect("non-empty");
        let last_query = *self.positions[queries].iter().max().expect("non-empty");
        first_key <= last_query
    }

    /// How many hops block `origin` must travel for every rank that wants it
    /// to have seen it.
    fn hops_needed(&self, origin: usize) -> usize {
        (0..self.size)
            .filter(|&r| self.visible(r, origin))
            .map(|r| (r + self.size - origin) % self.size)
            .max()
            .unwrap_or(0)
    }

    fn position_array(&self, rank: usize) -> Array {
        let rows: Vec<i32> = self.positions[rank].iter().map(|&p| p as i32).collect();
        Array::from_i32_slice(&rows)
    }

    /// `[B, H, S, D]` queries as `[B, Hkv, H/Hkv, S, D]` in f32.
    fn grouped_queries(&self, q: &Array) -> Array {
        let s = q.shape()[2];
        q.as_dtype(Dtype::Float32.as_i32()).reshape(&[
            self.batch,
            self.kv_heads,
            self.heads / self.kv_heads,
            s,
            self.head_dim,
        ])
    }

    /// Output and log-sum-exp of `q5` against one block of keys and values
    /// held by rank `keys`, for queries of rank `queries`.
    fn block_attention(
        &self,
        q5: &Array,
        queries: usize,
        k: &Array,
        v: &Array,
        keys: usize,
    ) -> (Array, Array) {
        let f32 = Dtype::Float32.as_i32();
        let k5 = k.as_dtype(f32).expand_dims(2); // [B, Hkv, 1, Sk, D]
        let v5 = v.as_dtype(f32).expand_dims(2);
        let scale = Array::from_f32_slice(&[self.config.scale], &[1]);
        let mut scores = mlx_ops::matmul(q5, &k5.swap_axes(3, 4)).multiply(&scale);
        if self.config.causal {
            let q_pos = self.position_array(queries);
            let k_pos = self.position_array(keys);
            let (sq, sk) = (q_pos.shape()[0], k_pos.shape()[0]);
            let visible =
                mlx_ops::greater_equal(&q_pos.reshape(&[sq, 1]), &k_pos.reshape(&[1, sk]));
            scores = mlx_ops::which(&visible, &scores, &Array::from_f32_slice(&[NEG], &[1]));
        }
        let lse = scores.logsumexp(-1, true);
        let out = mlx_ops::matmul(&mlx_ops::exp(&scores.subtract(&lse)), &v5);
        (out, lse)
    }

    fn pass_kv(&self, query: &Array, key: &Array, value: &Array) -> Result<Array, Exception> {
        let q5 = self.grouped_queries(query);
        let s = query.shape()[2];
        let acc_shape = [self.batch, self.kv_heads, self.heads / self.kv_heads, s];
        let mut out = mlx_ops::zeros(
            &[
                acc_shape[0],
                acc_shape[1],
                acc_shape[2],
                acc_shape[3],
                self.head_dim,
            ],
            Dtype::Float32,
        );
        let mut lse = mlx_ops::full(
            &[acc_shape[0], acc_shape[1], acc_shape[2], acc_shape[3], 1],
            NEG,
            Dtype::Float32,
        );

        // The block in hand, as one [2, B, Hkv, S, D] array so it moves in a
        // single message.
        let mut held = mlx_ops::stack_axis(&[key.clone(), value.clone()], 0);
        let dtype = key.dtype();
        let last_step = self.size - 1;

        for step in 0..self.size {
            let origin = (self.rank + self.size - step) % self.size;
            let next_origin = (self.rank + 2 * self.size - step - 1) % self.size;
            let forward = step < last_step && step < self.hops_needed(origin);
            let receive = step < last_step && step < self.hops_needed(next_origin);

            let incoming_shape = [
                2,
                self.batch,
                self.kv_heads,
                self.positions[next_origin].len() as i32,
                self.head_dim,
            ];
            let exchange = exchange::start(
                self.group,
                forward.then_some(&held),
                receive.then_some((&incoming_shape[..], dtype)),
            )?;

            if self.visible(self.rank, origin) {
                let k = held
                    .slice(
                        &[0, 0, 0, 0, 0],
                        &[1, self.batch, self.kv_heads, held.shape()[3], self.head_dim],
                    )
                    .squeeze(0);
                let v = held
                    .slice(
                        &[1, 0, 0, 0, 0],
                        &[2, self.batch, self.kv_heads, held.shape()[3], self.head_dim],
                    )
                    .squeeze(0);
                let (block_out, block_lse) = self.block_attention(&q5, self.rank, &k, &v, origin);
                (out, lse) = merge_partial_attention(&out, &lse, &block_out, &block_lse);
                // The GPU works on this while the block moves.
                out.try_async_eval().map_err(runtime_error)?;
                lse.try_async_eval().map_err(runtime_error)?;
            }

            if let Some(next) = exchange.finish()? {
                held = next;
            }
        }
        out.try_eval().map_err(runtime_error)?;
        Ok(out.reshape(&[self.batch, self.heads, s, self.head_dim]))
    }

    fn pass_q(&self, query: &Array, key: &Array, value: &Array) -> Result<Array, Exception> {
        let f32 = Dtype::Float32.as_i32();
        let group_size = self.heads / self.kv_heads;
        let s_of = |r: usize| self.positions[r].len() as i32;
        // Everything one query block carries, flattened into one f32 message:
        // the queries, then its running output, then its running log-sum-exp.
        let q_len = |r: usize| (self.batch * self.heads * s_of(r) * self.head_dim) as usize;
        let lse_len = |r: usize| (self.batch * self.heads * s_of(r)) as usize;
        let packet_len = |r: usize| q_len(r) * 2 + lse_len(r);

        let pack = |q: &Array, out: &Array, lse: &Array| {
            mlx_ops::concatenate_axis(
                &[&q.reshape(&[-1]), &out.reshape(&[-1]), &lse.reshape(&[-1])],
                0,
            )
        };

        let own = query.as_dtype(f32);
        let mut held = pack(
            &own,
            &mlx_ops::zeros(
                &[self.batch, self.heads, s_of(self.rank), self.head_dim],
                Dtype::Float32,
            ),
            &mlx_ops::full(
                &[self.batch, self.heads, s_of(self.rank)],
                NEG,
                Dtype::Float32,
            ),
        );

        for step in 0..self.size {
            let origin = (self.rank + self.size - step) % self.size;
            let next_origin = (self.rank + 2 * self.size - step - 1) % self.size;

            // Merge this rank's keys into the block in hand. The updated block
            // is what moves on, so unlike Pass-KV the transfer cannot overlap
            // this step's compute.
            if self.visible(origin, self.rank) {
                let (s, h, d, b) = (s_of(origin), self.heads, self.head_dim, self.batch);
                let n = q_len(origin) as i32;
                let q = held.slice(&[0], &[n]).reshape(&[b, h, s, d]);
                let acc = held
                    .slice(&[n], &[2 * n])
                    .reshape(&[b, self.kv_heads, group_size, s, d]);
                let acc_lse = held
                    .slice(&[2 * n], &[2 * n + lse_len(origin) as i32])
                    .reshape(&[b, self.kv_heads, group_size, s, 1]);
                let q5 = self.grouped_queries(&q);
                let (block_out, block_lse) =
                    self.block_attention(&q5, origin, key, value, self.rank);
                let (out, lse) = merge_partial_attention(&acc, &acc_lse, &block_out, &block_lse);
                held = pack(&q, &out, &lse);
                held.try_eval().map_err(runtime_error)?;
            }

            // Every query block makes the whole trip: it has to end up home.
            if self.size > 1 {
                let incoming_shape = [packet_len(next_origin) as i32];
                held = exchange::exchange(
                    self.group,
                    Some(&held),
                    Some((&incoming_shape[..], Dtype::Float32)),
                )?
                .expect("a receive was started");
            }
        }

        // Our own block has come home with every rank's keys merged in.
        let (s, b, d) = (s_of(self.rank), self.batch, self.head_dim);
        let n = q_len(self.rank) as i32;
        Ok(held.slice(&[n], &[2 * n]).reshape(&[b, self.heads, s, d]))
    }
}

/// Combine two partial attention results over disjoint sets of keys into the
/// result over their union.
///
/// Each is a normalised output `O` (softmax-weighted values over its own keys)
/// with the log-sum-exp `L` of its scores. The union's log-sum-exp is
/// `log(exp(L1) + exp(L2))` and its output weights each part by its share of
/// that total: `O = O1 exp(L1 - L) + O2 exp(L2 - L)`.
pub fn merge_partial_attention(
    output1: &Array,
    lse1: &Array,
    output2: &Array,
    lse2: &Array,
) -> (Array, Array) {
    // new_lse = max + log(exp(lse1 - max) + exp(lse2 - max)), stable for any inputs.
    let max_lse = mlx_ops::maximum(lse1, lse2);
    let sum_exp =
        mlx_ops::exp(&lse1.subtract(&max_lse)).add(&mlx_ops::exp(&lse2.subtract(&max_lse)));
    let new_lse = max_lse.add(&mlx_ops::log(&sum_exp));

    let weight1 = mlx_ops::exp(&lse1.subtract(&new_lse));
    let weight2 = mlx_ops::exp(&lse2.subtract(&new_lse));
    let new_output = output1.multiply(&weight1).add(&output2.multiply(&weight2));
    (new_output, new_lse)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eval(a: &Array, n: usize) -> Vec<f32> {
        let mut a = a.clone();
        a.try_eval().unwrap();
        a.to_f32_vec(n).unwrap()
    }

    #[test]
    fn merging_equal_scores_averages_the_outputs() {
        let out1 = Array::from_f32_slice(&[1.0f32, 2.0], &[1, 1, 1, 2]);
        let lse1 = Array::from_f32_slice(&[0.5f32], &[1, 1, 1, 1]);
        let out2 = Array::from_f32_slice(&[3.0f32, 4.0], &[1, 1, 1, 2]);
        let lse2 = Array::from_f32_slice(&[0.5f32], &[1, 1, 1, 1]);

        let (combined, new_lse) = merge_partial_attention(&out1, &lse1, &out2, &lse2);

        let data = eval(&combined, 2);
        assert!((data[0] - 2.0).abs() < 1e-6);
        assert!((data[1] - 3.0).abs() < 1e-6);
        // log(e^0.5 + e^0.5) = 0.5 + ln 2
        assert!((eval(&new_lse, 1)[0] - (0.5 + std::f32::consts::LN_2)).abs() < 1e-6);
    }

    /// A part with no visible keys (log-sum-exp at the finite floor) must
    /// contribute exactly nothing, in either position.
    #[test]
    fn a_fully_masked_part_contributes_nothing() {
        let real = Array::from_f32_slice(&[5.0f32, -7.0], &[1, 1, 1, 2]);
        let real_lse = Array::from_f32_slice(&[1.25f32], &[1, 1, 1, 1]);
        let junk = Array::from_f32_slice(&[99.0f32, 99.0], &[1, 1, 1, 2]);
        let floor = Array::from_f32_slice(&[NEG], &[1, 1, 1, 1]);

        for (a, la, b, lb) in [
            (&junk, &floor, &real, &real_lse),
            (&real, &real_lse, &junk, &floor),
        ] {
            let (out, lse) = merge_partial_attention(a, la, b, lb);
            assert_eq!(eval(&out, 2), vec![5.0, -7.0]);
            assert_eq!(eval(&lse, 1), vec![1.25]);
        }
    }

    /// Merging is associative: splitting the keys into three parts and
    /// merging in any order gives one answer.
    #[test]
    fn merge_is_order_independent() {
        let part = |o: f32, l: f32| {
            (
                Array::from_f32_slice(&[o], &[1, 1, 1, 1]),
                Array::from_f32_slice(&[l], &[1, 1, 1, 1]),
            )
        };
        let (a, b, c) = (part(1.0, 0.3), part(-2.0, 1.1), part(4.0, -0.7));
        let left = {
            let ab = merge_partial_attention(&a.0, &a.1, &b.0, &b.1);
            merge_partial_attention(&ab.0, &ab.1, &c.0, &c.1)
        };
        let right = {
            let bc = merge_partial_attention(&b.0, &b.1, &c.0, &c.1);
            merge_partial_attention(&a.0, &a.1, &bc.0, &bc.1)
        };
        assert!((eval(&left.0, 1)[0] - eval(&right.0, 1)[0]).abs() < 1e-6);
        assert!((eval(&left.1, 1)[0] - eval(&right.1, 1)[0]).abs() < 1e-6);
    }
}
