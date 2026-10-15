//! Ring collectives over a pair of transport halves.
//!
//! One implementation of the ring algorithms serves every backend that owns a
//! ring (the manual-peer [`RingBackend`](crate::ring::RingBackend) and the
//! auto-discovery backend), so a fix lands in both.
//!
//! # Semantics
//!
//! All collectives work on `f32` (gradients are summed in single precision,
//! as every rank holds them) or on raw bytes. For a buffer of `len` elements
//! on `N` ranks the buffer is cut into `N` chunks ([`chunk_range`]: the first
//! `len % N` chunks hold one extra element) and:
//!
//! - [`reduce_scatter`](RingCollective::reduce_scatter) leaves rank `r` with
//!   the reduced chunk `r` and nothing meaningful elsewhere;
//! - [`all_gather_in_place`](RingCollective::all_gather_in_place) takes each
//!   rank's chunk `r` to every rank;
//! - [`all_reduce`](RingCollective::all_reduce) is the two in sequence, moving
//!   `2 (N - 1) / N` of the buffer per rank, which is bandwidth-optimal.
//!
//! # Determinism
//!
//! The order in which contributions to chunk `c` are added is fixed by the
//! ring: `((x[c+1] + x[c+2]) + ...) + x[c]` over ranks, whatever the timing,
//! so repeated runs give the same bits. All-gather copies the finished chunk,
//! so every rank ends with the same bits too. Chunks are summed in different
//! rank orders from one another; that is inherent to a ring and deterministic.
//!
//! # Failure
//!
//! Every collective starts with a small header naming the operation, its
//! sequence number and its size. Ranks that disagree (different lengths, one
//! rank in a barrier while another reduces) fail with an error saying so,
//! instead of exchanging bytes that were never meant to meet. After any
//! failure, or if a collective's future is dropped before it finishes, the
//! ring is poisoned: its streams are at unknown positions, and every later
//! call fails with [`DistributedError::ChannelPoisoned`].

use crate::ReduceOp;
use crate::error::DistributedError;
use crate::transport::{TransportReceiver, TransportSender};
use anyhow::Result;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::Mutex;
use zerocopy::{FromBytes, IntoBytes};

#[cfg(not(target_endian = "little"))]
compile_error!("the ring moves f32 in native byte order and assumes a little-endian host");

/// Elements per wire message inside a ring step (8 MiB of `f32`). A step is
/// sent as a run of these so the receiver can reduce one while the next is in
/// flight, and so no message approaches any size limit.
pub const SEGMENT_ELEMS: usize = 2 << 20;

/// The `idx`-th of `parts` contiguous chunks of `len` elements; the first
/// `len % parts` chunks are one element longer.
pub fn chunk_range(len: usize, parts: usize, idx: usize) -> Range<usize> {
    let base = len / parts;
    let extra = len % parts;
    let start = idx * base + idx.min(extra);
    start..start + base + usize::from(idx < extra)
}

/// What a collective is, as announced to the next rank at its start.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OpHeader {
    kind: OpKind,
    seq: u64,
    /// Elements (or bytes) this rank contributes.
    len: u64,
    /// The reduction, or the broadcast root.
    arg: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum OpKind {
    AllReduce = 1,
    ReduceScatter = 2,
    AllGather = 3,
    AllGatherBytes = 4,
    Broadcast = 5,
    Barrier = 6,
}

impl OpKind {
    fn name(self) -> &'static str {
        match self {
            Self::AllReduce => "all_reduce",
            Self::ReduceScatter => "reduce_scatter",
            Self::AllGather => "all_gather",
            Self::AllGatherBytes => "all_gather_bytes",
            Self::Broadcast => "broadcast",
            Self::Barrier => "barrier",
        }
    }

    fn from_u8(v: u8) -> Option<Self> {
        Some(match v {
            1 => Self::AllReduce,
            2 => Self::ReduceScatter,
            3 => Self::AllGather,
            4 => Self::AllGatherBytes,
            5 => Self::Broadcast,
            6 => Self::Barrier,
            _ => return None,
        })
    }
}

impl OpHeader {
    const LEN: usize = 1 + 8 + 8 + 4;

    fn encode(&self) -> [u8; Self::LEN] {
        let mut out = [0u8; Self::LEN];
        out[0] = self.kind as u8;
        out[1..9].copy_from_slice(&self.seq.to_le_bytes());
        out[9..17].copy_from_slice(&self.len.to_le_bytes());
        out[17..].copy_from_slice(&self.arg.to_le_bytes());
        out
    }

    fn decode(b: &[u8; Self::LEN]) -> Result<Self, DistributedError> {
        let kind = OpKind::from_u8(b[0]).ok_or_else(|| {
            DistributedError::Protocol(format!("unknown collective kind {} from peer", b[0]))
        })?;
        Ok(Self {
            kind,
            seq: u64::from_le_bytes(b[1..9].try_into().expect("8 bytes")),
            len: u64::from_le_bytes(b[9..17].try_into().expect("8 bytes")),
            arg: u32::from_le_bytes(b[17..].try_into().expect("4 bytes")),
        })
    }
}

struct RingLink {
    next: TransportSender,
    prev: TransportReceiver,
    /// Collectives started so far on this ring; every rank counts alike.
    seq: u64,
}

/// A ring of ranks and the collectives that run on it.
pub struct RingCollective {
    rank: usize,
    world_size: usize,
    link: Mutex<RingLink>,
    poisoned: AtomicBool,
}

/// Poisons the ring unless the collective it guards ran to completion, which
/// covers errors and also a future dropped mid-flight.
struct OpGuard<'a> {
    poisoned: &'a AtomicBool,
    finished: bool,
}

impl Drop for OpGuard<'_> {
    fn drop(&mut self) {
        if !self.finished {
            self.poisoned.store(true, Ordering::SeqCst);
        }
    }
}

/// Split one slice into an immutable view of `send` and a mutable view of
/// `recv`. The ranges must be disjoint.
fn disjoint(data: &mut [f32], send: Range<usize>, recv: Range<usize>) -> (&[f32], &mut [f32]) {
    debug_assert!(send.end <= recv.start || recv.end <= send.start);
    if send.start <= recv.start {
        let (head, tail) = data.split_at_mut(recv.start);
        (&head[send], &mut tail[..recv.end - recv.start])
    } else {
        let (head, tail) = data.split_at_mut(send.start);
        (&tail[..send.end - send.start], &mut head[recv])
    }
}

impl RingCollective {
    /// Wrap the two halves of a formed ring: `next` sends to rank
    /// `(rank + 1) % world_size`, `prev` receives from rank
    /// `(rank - 1) % world_size`.
    pub fn new(
        rank: usize,
        world_size: usize,
        next: TransportSender,
        prev: TransportReceiver,
    ) -> Self {
        assert!(
            world_size >= 1 && rank < world_size,
            "rank {rank} outside a world of {world_size}"
        );
        Self {
            rank,
            world_size,
            link: Mutex::new(RingLink { next, prev, seq: 0 }),
            poisoned: AtomicBool::new(false),
        }
    }

    /// This rank.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Number of ranks.
    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// Whether an earlier failure has made this ring unusable.
    pub fn is_poisoned(&self) -> bool {
        self.poisoned.load(Ordering::SeqCst)
    }

    /// Lock the link, start a collective of `kind`, and exchange its header
    /// with the neighbours. The returned guard poisons the ring unless
    /// [`Self::finish`] is called.
    async fn begin(
        &self,
        kind: OpKind,
        len: usize,
        arg: u32,
    ) -> Result<(tokio::sync::MutexGuard<'_, RingLink>, OpGuard<'_>)> {
        if self.is_poisoned() {
            return Err(DistributedError::ChannelPoisoned.into());
        }
        let mut link = self.link.lock().await;
        if self.is_poisoned() {
            return Err(DistributedError::ChannelPoisoned.into());
        }
        let guard = OpGuard {
            poisoned: &self.poisoned,
            finished: false,
        };
        let mine = OpHeader {
            kind,
            seq: link.seq,
            len: len as u64,
            arg,
        };
        link.seq += 1;
        if self.world_size > 1 {
            let outgoing = mine.encode();
            let mut theirs = [0u8; OpHeader::LEN];
            let RingLink { next, prev, .. } = &mut *link;
            tokio::try_join!(next.send(&outgoing), prev.recv(&mut theirs))?;
            let theirs = OpHeader::decode(&theirs)?;
            check_header(&mine, &theirs, self.world_size)?;
        }
        Ok((link, guard))
    }

    fn finish(guard: OpGuard<'_>) {
        let mut guard = guard;
        guard.finished = true;
    }

    /// Reduce `data` across the ring so that this rank holds the reduced
    /// chunk `rank` ([`chunk_range`]); the rest of `data` is left holding
    /// partial sums and must not be read. Returns the owned range.
    pub async fn reduce_scatter(&self, data: &mut [f32], op: ReduceOp) -> Result<Range<usize>> {
        let (mut link, guard) = self
            .begin(OpKind::ReduceScatter, data.len(), op_code(op))
            .await?;
        let owned = self.reduce_scatter_inner(&mut link, data, op).await?;
        Self::finish(guard);
        Ok(owned)
    }

    async fn reduce_scatter_inner(
        &self,
        link: &mut RingLink,
        data: &mut [f32],
        op: ReduceOp,
    ) -> Result<Range<usize>> {
        let (n, rank, len) = (self.world_size, self.rank, data.len());
        let owned = chunk_range(len, n, rank);
        if n > 1 {
            let largest = chunk_range(len, n, 0).len().min(SEGMENT_ELEMS);
            let mut scratch = vec![0f32; largest];
            for step in 0..n - 1 {
                // The chunk this rank forwards was reduced by its
                // predecessor one step ago; the one it receives is the
                // predecessor's, which it adds its own contribution to.
                let send = chunk_range(len, n, (rank + 2 * n - step - 1) % n);
                let recv = chunk_range(len, n, (rank + 2 * n - step - 2) % n);
                let (send, recv) = disjoint(data, send, recv);
                let RingLink { next, prev, .. } = &mut *link;
                tokio::try_join!(send_f32(next, send), recv_add_f32(prev, recv, &mut scratch))?;
            }
        }
        if op == ReduceOp::Mean {
            let divisor = n as f32;
            for x in &mut data[owned.clone()] {
                *x /= divisor;
            }
        }
        Ok(owned)
    }

    /// Complete `data` on every rank, given that each rank's chunk `rank`
    /// ([`chunk_range`]) is final.
    pub async fn all_gather_in_place(&self, data: &mut [f32]) -> Result<()> {
        let (mut link, guard) = self.begin(OpKind::AllGather, data.len(), 0).await?;
        self.all_gather_inner(&mut link, data).await?;
        Self::finish(guard);
        Ok(())
    }

    async fn all_gather_inner(&self, link: &mut RingLink, data: &mut [f32]) -> Result<()> {
        let (n, rank, len) = (self.world_size, self.rank, data.len());
        for step in 0..n.saturating_sub(1) {
            let send = chunk_range(len, n, (rank + n - step) % n);
            let recv = chunk_range(len, n, (rank + 2 * n - step - 1) % n);
            let (send, recv) = disjoint(data, send, recv);
            let RingLink { next, prev, .. } = &mut *link;
            tokio::try_join!(send_f32(next, send), recv_f32(prev, recv))?;
        }
        Ok(())
    }

    /// Sum (or average) `data` across the ring; every rank ends with the
    /// same result.
    pub async fn all_reduce(&self, data: &mut [f32], op: ReduceOp) -> Result<()> {
        let (mut link, guard) = self
            .begin(OpKind::AllReduce, data.len(), op_code(op))
            .await?;
        self.reduce_scatter_inner(&mut link, data, op).await?;
        self.all_gather_inner(&mut link, data).await?;
        Self::finish(guard);
        Ok(())
    }

    /// Collect every rank's `local` bytes, which may differ in length:
    /// element `r` of the result is rank `r`'s contribution.
    pub async fn all_gather_bytes(&self, local: &[u8]) -> Result<Vec<Vec<u8>>> {
        let (mut link, guard) = self.begin(OpKind::AllGatherBytes, local.len(), 0).await?;
        let (n, rank) = (self.world_size, self.rank);

        // Round 1: everyone learns every length.
        let mut lengths = vec![0u64; n];
        lengths[rank] = local.len() as u64;
        for step in 0..n - 1 {
            let send = (rank + n - step) % n;
            let recv = (rank + 2 * n - step - 1) % n;
            let outgoing = lengths[send].to_le_bytes();
            let mut incoming = [0u8; 8];
            let RingLink { next, prev, .. } = &mut *link;
            tokio::try_join!(next.send(&outgoing), prev.recv(&mut incoming))?;
            lengths[recv] = u64::from_le_bytes(incoming);
        }

        // Round 2: the blocks themselves.
        let mut blocks: Vec<Vec<u8>> = Vec::with_capacity(n);
        for (r, &len) in lengths.iter().enumerate() {
            blocks.push(if r == rank {
                local.to_vec()
            } else {
                vec![
                    0u8;
                    usize::try_from(len).map_err(|_| {
                        DistributedError::Protocol(format!("rank {r} announced {len} bytes"))
                    })?
                ]
            });
        }
        for step in 0..n - 1 {
            let send = (rank + n - step) % n;
            let recv = (rank + 2 * n - step - 1) % n;
            let outgoing = std::mem::take(&mut blocks[send]);
            let RingLink { next, prev, .. } = &mut *link;
            let result = tokio::try_join!(next.send(&outgoing), prev.recv(&mut blocks[recv]));
            blocks[send] = outgoing;
            result?;
        }
        Self::finish(guard);
        Ok(blocks)
    }

    /// Copy `data` from rank `root` to every rank. All ranks pass a buffer
    /// of the same length.
    pub async fn broadcast(&self, data: &mut [u8], root: usize) -> Result<()> {
        if root >= self.world_size {
            return Err(DistributedError::Config(format!(
                "broadcast root {root} outside a world of {}",
                self.world_size
            ))
            .into());
        }
        let (mut link, guard) = self
            .begin(OpKind::Broadcast, data.len(), root as u32)
            .await?;
        let n = self.world_size;
        if n > 1 {
            let is_root = self.rank == root;
            // The rank just before the root is the end of the chain.
            let is_last = (self.rank + 1) % n == root;
            let RingLink { next, prev, .. } = &mut *link;
            for segment in data.chunks_mut(SEGMENT_ELEMS * 4) {
                if !is_root {
                    prev.recv(segment).await?;
                }
                if !is_last {
                    next.send(segment).await?;
                }
            }
        }
        Self::finish(guard);
        Ok(())
    }

    /// Return once every rank has called it.
    pub async fn barrier(&self) -> Result<()> {
        // The header exchange in `begin` is round zero: receiving it proves
        // the predecessor has entered. A rank sends round `k` only after it
        // has received round `k - 1`, so round `k` arriving proves the `k + 1`
        // ranks behind it have entered; `n - 2` further rounds cover the
        // other `n - 1` ranks.
        let (mut link, guard) = self.begin(OpKind::Barrier, 0, 0).await?;
        let n = self.world_size;
        for round in 0..n.saturating_sub(2) {
            let outgoing = (round as u64).to_le_bytes();
            let mut incoming = [0u8; 8];
            let RingLink { next, prev, .. } = &mut *link;
            tokio::try_join!(next.send(&outgoing), prev.recv(&mut incoming))?;
            if u64::from_le_bytes(incoming) != round as u64 {
                return Err(DistributedError::Protocol(format!(
                    "barrier round mismatch: expected {round}, got {}",
                    u64::from_le_bytes(incoming)
                ))
                .into());
            }
        }
        Self::finish(guard);
        Ok(())
    }
}

fn op_code(op: ReduceOp) -> u32 {
    match op {
        ReduceOp::Sum => 0,
        ReduceOp::Mean => 1,
    }
}

fn check_header(
    mine: &OpHeader,
    theirs: &OpHeader,
    world_size: usize,
) -> Result<(), DistributedError> {
    let mismatch = |what: String| {
        Err(DistributedError::Protocol(format!(
            "ranks disagree about the collective in progress: {what}"
        )))
    };
    if theirs.kind != mine.kind || theirs.seq != mine.seq {
        return mismatch(format!(
            "this rank is in {} #{} but its predecessor is in {} #{}",
            mine.kind.name(),
            mine.seq,
            theirs.kind.name(),
            theirs.seq
        ));
    }
    if theirs.arg != mine.arg {
        return mismatch(format!(
            "{} argument differs ({} here, {} on the predecessor)",
            mine.kind.name(),
            mine.arg,
            theirs.arg
        ));
    }
    let size_must_match = !matches!(mine.kind, OpKind::AllGatherBytes);
    if size_must_match && theirs.len != mine.len {
        return mismatch(format!(
            "{} of {} elements here but {} on the predecessor ({world_size} ranks)",
            mine.kind.name(),
            mine.len,
            theirs.len
        ));
    }
    Ok(())
}

async fn send_f32(next: &mut TransportSender, data: &[f32]) -> Result<()> {
    for segment in data.chunks(SEGMENT_ELEMS) {
        next.send(segment.as_bytes()).await?;
    }
    Ok(())
}

async fn recv_f32(prev: &mut TransportReceiver, into: &mut [f32]) -> Result<()> {
    for segment in into.chunks_mut(SEGMENT_ELEMS) {
        prev.recv(segment.as_mut_bytes()).await?;
    }
    Ok(())
}

/// Receive a chunk a segment at a time and add each segment into `into` as it
/// lands, so the addition overlaps the next segment's transfer.
async fn recv_add_f32(
    prev: &mut TransportReceiver,
    into: &mut [f32],
    scratch: &mut [f32],
) -> Result<()> {
    for segment in into.chunks_mut(SEGMENT_ELEMS) {
        let incoming = &mut scratch[..segment.len()];
        prev.recv(incoming.as_mut_bytes()).await?;
        for (x, y) in segment.iter_mut().zip(incoming.iter()) {
            *x += *y;
        }
    }
    Ok(())
}

/// View a byte buffer as `f32`s when it is aligned and sized for it.
pub(crate) fn bytes_as_f32_mut(buffer: &mut [u8]) -> Option<&mut [f32]> {
    <[f32]>::mut_from_bytes(buffer).ok()
}

#[cfg(test)]
#[path = "collective_tests.rs"]
mod tests;
