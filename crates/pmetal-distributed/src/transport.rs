//! Authenticated, encrypted point-to-point transport between ranks.
//!
//! # Security model
//!
//! Anything on the LAN can open a TCP connection to a rank, so a connection
//! is trusted only after a [Noise] handshake (`NNpsk0`, X25519, AES-256-GCM,
//! SHA-256) keyed by the [`ClusterSecret`]. Only a machine that holds the
//! secret can complete it, and the secret never crosses the wire. Every byte
//! after the handshake is AEAD-protected: a machine on the path can neither
//! read gradients or activations nor inject, alter, replay, reorder or
//! truncate them (each frame is bound to a counter that must advance by
//! exactly one). Fresh ephemeral keys per connection mean a secret that leaks
//! later does not expose traffic recorded earlier.
//!
//! The handshake also carries a `Hello` naming the channel, the world size,
//! the sender's rank and a digest of the ring membership. Two machines that
//! disagree about who is in the ring fail the handshake with that reason
//! instead of exchanging gradients that were never meant to meet.
//!
//! Plaintext framing exists for in-process tests only and must be asked for
//! with [`LinkSecurity::Plaintext`].
//!
//! # Wire format
//!
//! After the handshake, a stream is a sequence of frames, each a 2-byte
//! little-endian body length followed by the body. A body is at most 65535
//! bytes, the Noise message limit: the ciphertext of one chunk of payload plus
//! a 16-byte tag, encrypted under the frame's counter.
//!
//! A message is a run of frames. The first chunk's plaintext starts with the
//! message's total length as a `u64`; chunks follow until that many bytes have
//! arrived. There is no size limit in the framing (a 40 GB gradient is one
//! message) and no truncation: the length is `u64` and authenticated.
//!
//! # Failure
//!
//! A stream that hit an error or a timeout has an unknown position inside a
//! frame, and trying to carry on would misread gradient bytes as headers. The
//! first failure therefore *poisons* the half, and every later call fails at
//! once with [`DistributedError::ChannelPoisoned`]. Reads and writes time out
//! when they make no progress for [`IO_IDLE_TIMEOUT`], not when a whole
//! transfer takes long, so a slow link is never mistaken for a dead peer.
//!
//! [Noise]: https://noiseprotocol.org/noise.html

use crate::config::DistributedConfig;
use crate::error::DistributedError;
use crate::secret::ClusterSecret;
use anyhow::Result;
use rayon::prelude::*;
use sha3::{Digest, Sha3_256};
use snow::{Builder, StatelessTransportState};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::tcp::{OwnedReadHalf, OwnedWriteHalf};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;
use tokio::time::timeout;
use tracing::{debug, info, warn};

/// How long a read or a write may go without moving a single byte before the
/// peer is declared dead.
pub const IO_IDLE_TIMEOUT: Duration = Duration::from_secs(60);

/// The largest message [`TransportReceiver::recv_vec`] will allocate for.
/// Receivers that know the size (`recv`) are not subject to it.
pub const MAX_VEC_MESSAGE_BYTES: u64 = 4 << 30;

/// Maximum backoff delay between connection retries.
const MAX_BACKOFF_MS: u64 = 5000;

/// Initial backoff delay between connection retries.
const INITIAL_BACKOFF_MS: u64 = 100;

/// Longest a peer may take over the handshake before it is dropped.
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(5);

/// Handshake wire version. Bump on any change to the handshake or framing.
const WIRE_VERSION: u16 = 1;

const NOISE_PATTERN: &str = "Noise_NNpsk0_25519_AESGCM_SHA256";
const NOISE_PROLOGUE: &[u8] = b"pmetal-distributed/wire/1";

const TAG_LEN: usize = 16;
/// Largest frame body: the Noise message limit.
const MAX_FRAME_BODY: usize = 65_535;
/// Receive-side read buffer.
const READ_BUFFER_BYTES: usize = 1 << 19;
/// Frames sealed or opened together (about 1 MiB).
const FRAMES_PER_BATCH: usize = 16;
/// Fewest frames in a batch for which sealing or opening them on several
/// cores pays for the hand-off.
const PARALLEL_MIN_FRAMES: usize = 4;

/// Which logical stream a connection carries. A mismatch fails the handshake
/// so, for example, an activation link can never be wired to a gradient ring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum Channel {
    /// The gradient all-reduce ring.
    Ring = 1,
    /// The pipeline's activation ring.
    Activations = 2,
    /// Sampled tokens flowing from the last stage back to the first.
    Results = 3,
}

/// How a link is protected.
#[derive(Debug, Clone)]
pub enum LinkSecurity {
    /// Authenticated and encrypted with keys derived from the secret.
    Authenticated(ClusterSecret),
    /// No authentication and no encryption. For in-process tests: anything
    /// on the network can read and forge this traffic.
    Plaintext,
}

impl LinkSecurity {
    /// The security a configuration asks for: its own secret, else the
    /// machine's ([`ClusterSecret::resolve`]), else an error, unless the
    /// configuration explicitly allows plaintext.
    pub fn from_config(config: &DistributedConfig) -> Result<Self, DistributedError> {
        if let Some(secret) = &config.secret {
            return Ok(Self::Authenticated(secret.clone()));
        }
        if config.allow_plaintext {
            warn!(
                "distributed link is UNAUTHENTICATED and UNENCRYPTED \
                 (`allow_plaintext`): anyone on the network can read or forge it"
            );
            return Ok(Self::Plaintext);
        }
        ClusterSecret::resolve().map(Self::Authenticated)
    }

    fn is_encrypted(&self) -> bool {
        matches!(self, Self::Authenticated(_))
    }
}

/// Who is in the ring, as one rank sees it. Digests must be equal on every
/// rank or the handshake between neighbours fails.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RingMembership {
    /// This rank.
    pub rank: usize,
    /// Number of ranks.
    pub world_size: usize,
    /// Digest of the ordered member list.
    pub digest: [u8; 32],
}

impl RingMembership {
    /// Membership of the ring a configuration describes: the ordered primary
    /// address of each rank.
    pub fn from_config(config: &DistributedConfig) -> Self {
        let mut hasher = Sha3_256::new();
        hasher.update(b"pmetal-distributed/membership/addrs/v1");
        for addr in &config.nodes {
            let text = addr.to_string();
            hasher.update((text.len() as u64).to_le_bytes());
            hasher.update(text.as_bytes());
        }
        Self {
            rank: config.rank,
            world_size: config.nodes.len(),
            digest: hasher.finalize().into(),
        }
    }

    /// Membership given as an ordered list of stable identifiers (peer ids).
    pub fn from_ids<I: AsRef<[u8]>>(rank: usize, ids: &[I]) -> Self {
        let mut hasher = Sha3_256::new();
        hasher.update(b"pmetal-distributed/membership/ids/v1");
        for id in ids {
            let id = id.as_ref();
            hasher.update((id.len() as u64).to_le_bytes());
            hasher.update(id);
        }
        Self {
            rank,
            world_size: ids.len(),
            digest: hasher.finalize().into(),
        }
    }
}

/// What each end of a connection says about itself, inside the handshake.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Hello {
    version: u16,
    channel: u16,
    world_size: u32,
    rank: u32,
    membership: [u8; 32],
}

impl Hello {
    const LEN: usize = 2 + 2 + 4 + 4 + 32;

    fn new(channel: Channel, membership: &RingMembership) -> Self {
        Self {
            version: WIRE_VERSION,
            channel: channel as u16,
            world_size: membership.world_size as u32,
            rank: membership.rank as u32,
            membership: membership.digest,
        }
    }

    fn encode(&self) -> [u8; Self::LEN] {
        let mut out = [0u8; Self::LEN];
        out[0..2].copy_from_slice(&self.version.to_le_bytes());
        out[2..4].copy_from_slice(&self.channel.to_le_bytes());
        out[4..8].copy_from_slice(&self.world_size.to_le_bytes());
        out[8..12].copy_from_slice(&self.rank.to_le_bytes());
        out[12..].copy_from_slice(&self.membership);
        out
    }

    fn decode(bytes: &[u8]) -> Result<Self, DistributedError> {
        if bytes.len() != Self::LEN {
            return Err(DistributedError::Protocol(format!(
                "handshake hello is {} bytes, expected {}",
                bytes.len(),
                Self::LEN
            )));
        }
        let u16_at = |at: usize| u16::from_le_bytes([bytes[at], bytes[at + 1]]);
        let u32_at = |at: usize| {
            u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
        };
        let mut membership = [0u8; 32];
        membership.copy_from_slice(&bytes[12..]);
        Ok(Self {
            version: u16_at(0),
            channel: u16_at(2),
            world_size: u32_at(4),
            rank: u32_at(8),
            membership,
        })
    }

    /// Check a peer's hello against ours. `expected_rank` is the rank this
    /// connection is supposed to reach.
    fn check_peer(&self, peer: &Hello, expected_rank: usize) -> Result<(), DistributedError> {
        let mismatch = |what: String| Err(DistributedError::ClusterMismatch(what));
        if peer.version != self.version {
            return Err(DistributedError::VersionMismatch {
                local: self.version.to_string(),
                remote: peer.version.to_string(),
            });
        }
        if peer.channel != self.channel {
            return mismatch(format!(
                "peer opened channel {} but this end expected channel {}",
                peer.channel, self.channel
            ));
        }
        if peer.world_size != self.world_size {
            return mismatch(format!(
                "peer believes the world has {} ranks, this end {}",
                peer.world_size, self.world_size
            ));
        }
        if peer.membership != self.membership {
            return mismatch(
                "peer's ring membership differs from this end's (the two sides were \
                 configured with different node lists)"
                    .to_string(),
            );
        }
        if peer.rank as usize != expected_rank {
            return mismatch(format!(
                "peer is rank {} but this connection should reach rank {expected_rank}",
                peer.rank
            ));
        }
        Ok(())
    }
}

// ─── Framing ────────────────────────────────────────────────────────────────

/// Frame protection for one connection, shared by its two halves.
#[derive(Clone)]
enum Protection {
    Plain,
    Noise(Arc<StatelessTransportState>),
}

impl Protection {
    fn tag_len(&self) -> usize {
        match self {
            Self::Plain => 0,
            Self::Noise(_) => TAG_LEN,
        }
    }

    fn max_chunk(&self) -> usize {
        MAX_FRAME_BODY - self.tag_len()
    }

    /// Encrypt `plain` as frame `nonce` into `out`; returns the body length.
    fn seal(&self, nonce: u64, plain: &[u8], out: &mut [u8]) -> Result<usize, DistributedError> {
        match self {
            Self::Plain => {
                out[..plain.len()].copy_from_slice(plain);
                Ok(plain.len())
            }
            Self::Noise(state) => state
                .write_message(nonce, plain, out)
                .map_err(|e| DistributedError::Protocol(format!("encrypt failed: {e}"))),
        }
    }

    /// Decrypt frame `nonce`'s body into `out`; returns the plaintext length.
    fn open(&self, nonce: u64, body: &[u8], out: &mut [u8]) -> Result<usize, DistributedError> {
        match self {
            Self::Plain => {
                let dst = out.get_mut(..body.len()).ok_or_else(|| {
                    DistributedError::Protocol("frame overruns the message".into())
                })?;
                dst.copy_from_slice(body);
                Ok(body.len())
            }
            Self::Noise(state) => {
                if out.len() < body.len().saturating_sub(TAG_LEN) {
                    return Err(DistributedError::Protocol(
                        "frame overruns the message".into(),
                    ));
                }
                state.read_message(nonce, body, out).map_err(|_| {
                    DistributedError::Auth(
                        "a frame failed authentication: it was altered, replayed, reordered, \
                         or the peer is using a different key"
                            .into(),
                    )
                })
            }
        }
    }
}

impl Protection {
    /// Seal `plains` as consecutive frames, nonces counting up from
    /// `first_nonce`, into `out`, which holds exactly their frames: a 2-byte
    /// length, then the sealed body, for each. Large batches are sealed on
    /// several cores: every frame has its own nonce, so the frames are
    /// independent.
    fn seal_many(
        &self,
        first_nonce: u64,
        plains: &[&[u8]],
        out: &mut [u8],
    ) -> Result<(), DistributedError> {
        let tag = self.tag_len();
        let mut regions: Vec<&mut [u8]> = Vec::with_capacity(plains.len());
        let mut rest = out;
        for plain in plains {
            let (region, tail) = rest.split_at_mut(2 + plain.len() + tag);
            regions.push(region);
            rest = tail;
        }
        debug_assert!(rest.is_empty());
        let seal_one = |(i, (region, plain)): (usize, (&mut [u8], &&[u8]))| {
            let n = self.seal(first_nonce + i as u64, plain, &mut region[2..])?;
            region[..2].copy_from_slice(&(n as u16).to_le_bytes());
            Ok(())
        };
        if worth_parallelising(plains.len()) {
            regions
                .into_par_iter()
                .zip(plains.par_iter())
                .enumerate()
                .try_for_each(seal_one)
        } else {
            regions
                .into_iter()
                .zip(plains)
                .enumerate()
                .try_for_each(seal_one)
        }
    }

    /// Open `bodies` as consecutive frames, nonces counting up from
    /// `first_nonce`, writing their plaintexts back to back into `out`, which
    /// holds exactly their total. The counterpart of [`Self::seal_many`].
    fn open_many(
        &self,
        first_nonce: u64,
        bodies: &[&[u8]],
        out: &mut [u8],
    ) -> Result<(), DistributedError> {
        let tag = self.tag_len();
        let mut regions: Vec<&mut [u8]> = Vec::with_capacity(bodies.len());
        let mut rest = out;
        for body in bodies {
            let (region, tail) = rest.split_at_mut(body.len() - tag);
            regions.push(region);
            rest = tail;
        }
        debug_assert!(rest.is_empty());
        let open_one = |(i, (region, body)): (usize, (&mut [u8], &&[u8]))| {
            self.open(first_nonce + i as u64, body, region).map(|_| ())
        };
        if worth_parallelising(bodies.len()) {
            regions
                .into_par_iter()
                .zip(bodies.par_iter())
                .enumerate()
                .try_for_each(open_one)
        } else {
            regions
                .into_iter()
                .zip(bodies)
                .enumerate()
                .try_for_each(open_one)
        }
    }
}

/// Whether a batch of frames is big enough to be worth fanning out to threads.
fn worth_parallelising(frames: usize) -> bool {
    frames >= PARALLEL_MIN_FRAMES
}

/// Sender half of the transport.
pub struct TransportSender {
    inner: TransportSenderInner,
}

/// Receiver half of the transport.
pub struct TransportReceiver {
    inner: TransportReceiverInner,
}

enum TransportSenderInner {
    Tcp(TcpWriter),
    Memory(mpsc::Sender<Vec<u8>>),
}

enum TransportReceiverInner {
    Tcp(TcpReader),
    Memory(mpsc::Receiver<Vec<u8>>),
}

struct TcpWriter {
    stream: OwnedWriteHalf,
    protection: Protection,
    nonce: u64,
    batch: Vec<u8>,
    /// Reused buffer for the first frame of a message (length header + chunk).
    first: Vec<u8>,
    poisoned: bool,
}

struct TcpReader {
    stream: BufReader<OwnedReadHalf>,
    protection: Protection,
    nonce: u64,
    /// Frame headers and the first frame's body.
    body: Vec<u8>,
    scratch: Vec<u8>,
    /// The bodies of the frames of one batch, back to back.
    batch_bodies: Vec<u8>,
    /// The length of each body in `batch_bodies`.
    batch_lens: Vec<usize>,
    poisoned: bool,
}

/// The outcome of a timed `read_exact`, with a peer that vanished or went
/// quiet named as such.
fn read_outcome(
    outcome: Result<std::io::Result<usize>, tokio::time::error::Elapsed>,
) -> Result<(), DistributedError> {
    match outcome {
        Ok(Ok(_)) => Ok(()),
        Ok(Err(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
            Err(DistributedError::Protocol(
                "peer closed the connection (it exited, crashed, or was cut off)".into(),
            ))
        }
        Ok(Err(e)) => Err(e.into()),
        Err(_) => Err(DistributedError::Protocol(format!(
            "no data from peer for {IO_IDLE_TIMEOUT:?}; peer presumed dead"
        ))),
    }
}

impl TcpWriter {
    fn new(stream: OwnedWriteHalf, protection: Protection) -> Self {
        Self {
            stream,
            protection,
            nonce: 0,
            batch: Vec::with_capacity(FRAMES_PER_BATCH * (MAX_FRAME_BODY + 2)),
            first: Vec::with_capacity(MAX_FRAME_BODY),
            poisoned: false,
        }
    }

    async fn send(&mut self, data: &[u8]) -> Result<(), DistributedError> {
        if self.poisoned {
            return Err(DistributedError::ChannelPoisoned);
        }
        let result = self.send_inner(data).await;
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }

    async fn send_inner(&mut self, data: &[u8]) -> Result<(), DistributedError> {
        let max_chunk = self.protection.max_chunk();
        // The first frame carries the message length ahead of the first chunk.
        let head = data.len().min(max_chunk - 8);
        let mut first = std::mem::take(&mut self.first);
        first.clear();
        first.extend_from_slice(&(data.len() as u64).to_le_bytes());
        first.extend_from_slice(&data[..head]);

        let outcome = self
            .send_frames(&first, data[head..].chunks(max_chunk))
            .await;
        self.first = first;
        outcome
    }

    /// Seal and write `first` and then `rest`, as frames, a batch at a time.
    async fn send_frames<'a>(
        &mut self,
        first: &[u8],
        mut rest: impl Iterator<Item = &'a [u8]>,
    ) -> Result<(), DistributedError> {
        let mut frames: Vec<&[u8]> = Vec::with_capacity(FRAMES_PER_BATCH);
        frames.push(first);
        loop {
            while frames.len() < FRAMES_PER_BATCH {
                match rest.next() {
                    Some(chunk) => frames.push(chunk),
                    None => break,
                }
            }
            self.seal_batch(&frames)?;
            self.flush().await?;
            if frames.len() < FRAMES_PER_BATCH {
                return Ok(());
            }
            frames.clear();
        }
    }

    /// Append `plains`, sealed as consecutive frames, to the batch buffer.
    fn seal_batch(&mut self, plains: &[&[u8]]) -> Result<(), DistributedError> {
        let tag = self.protection.tag_len();
        let at = self.batch.len();
        let total: usize = plains.iter().map(|p| 2 + p.len() + tag).sum();
        self.batch.resize(at + total, 0);
        self.protection
            .seal_many(self.nonce, plains, &mut self.batch[at..])?;
        self.nonce += plains.len() as u64;
        Ok(())
    }

    async fn flush(&mut self) -> Result<(), DistributedError> {
        let write = self.stream.write_all(&self.batch);
        let outcome = timeout(IO_IDLE_TIMEOUT, write).await;
        self.batch.clear();
        match outcome {
            Ok(result) => Ok(result?),
            Err(_) => Err(DistributedError::Protocol(format!(
                "no progress writing to peer for {IO_IDLE_TIMEOUT:?}; peer presumed dead"
            ))),
        }
    }
}

impl TcpReader {
    fn new(stream: OwnedReadHalf, protection: Protection) -> Self {
        Self {
            stream: BufReader::with_capacity(READ_BUFFER_BYTES, stream),
            protection,
            nonce: 0,
            body: vec![0u8; MAX_FRAME_BODY],
            scratch: vec![0u8; MAX_FRAME_BODY],
            batch_bodies: vec![0u8; FRAMES_PER_BATCH * MAX_FRAME_BODY],
            batch_lens: Vec::with_capacity(FRAMES_PER_BATCH),
            poisoned: false,
        }
    }

    /// Read exactly `len` bytes into the start of `self.body` (frame headers
    /// and bodies alike).
    async fn read_exact(&mut self, len: usize) -> Result<(), DistributedError> {
        let read = self.stream.read_exact(&mut self.body[..len]);
        read_outcome(timeout(IO_IDLE_TIMEOUT, read).await)
    }

    /// Read one frame's body into `self.body`; returns its length.
    async fn read_frame(&mut self) -> Result<usize, DistributedError> {
        self.read_exact(2).await?;
        let len = u16::from_le_bytes([self.body[0], self.body[1]]) as usize;
        if len < self.protection.tag_len() {
            return Err(DistributedError::Protocol(format!(
                "frame of {len} bytes is shorter than its authentication tag"
            )));
        }
        self.read_exact(len).await?;
        Ok(len)
    }

    /// Read the first frame of a message: returns the declared total length
    /// and how many payload bytes of it arrived in the first frame (left at
    /// the start of `self.scratch`).
    async fn read_header(&mut self) -> Result<(u64, usize), DistributedError> {
        let len = self.read_frame().await?;
        let n = {
            let (body, scratch) = (&self.body[..len], &mut self.scratch[..]);
            self.protection.open(self.nonce, body, scratch)?
        };
        self.nonce += 1;
        if n < 8 {
            return Err(DistributedError::Protocol(
                "first frame of a message is shorter than its length header".into(),
            ));
        }
        let total = u64::from_le_bytes(self.scratch[..8].try_into().expect("8 bytes"));
        self.scratch.copy_within(8..n, 0);
        Ok((total, n - 8))
    }

    /// Read the continuation frames of a message into `dst[filled..]`, a batch
    /// at a time: the frames of a batch are read off the socket in order,
    /// then opened together (on several cores when the batch is large).
    ///
    /// A peer's declared lengths are checked before anything is written: a
    /// frame that would run past the end of `dst` is an error.
    async fn fill(&mut self, dst: &mut [u8], mut filled: usize) -> Result<(), DistributedError> {
        let tag = self.protection.tag_len();
        while filled < dst.len() {
            self.batch_lens.clear();
            let (mut stored, mut planned) = (0usize, 0usize);
            while self.batch_lens.len() < FRAMES_PER_BATCH && filled + planned < dst.len() {
                self.read_exact(2).await?;
                let len = u16::from_le_bytes([self.body[0], self.body[1]]) as usize;
                if len <= tag {
                    return Err(DistributedError::Protocol(
                        "empty continuation frame".into(),
                    ));
                }
                if filled + planned + (len - tag) > dst.len() {
                    return Err(DistributedError::Protocol(
                        "frame overruns the message".into(),
                    ));
                }
                self.read_body(stored, len).await?;
                self.batch_lens.push(len);
                stored += len;
                planned += len - tag;
            }

            let mut bodies: Vec<&[u8]> = Vec::with_capacity(self.batch_lens.len());
            let mut at = 0;
            for &len in &self.batch_lens {
                bodies.push(&self.batch_bodies[at..at + len]);
                at += len;
            }
            self.protection
                .open_many(self.nonce, &bodies, &mut dst[filled..filled + planned])?;
            self.nonce += bodies.len() as u64;
            filled += planned;
        }
        Ok(())
    }

    /// Read a frame body of `len` bytes into `batch_bodies` at `at`.
    async fn read_body(&mut self, at: usize, len: usize) -> Result<(), DistributedError> {
        let read = self.stream.read_exact(&mut self.batch_bodies[at..at + len]);
        read_outcome(timeout(IO_IDLE_TIMEOUT, read).await)
    }

    /// Receive a message of exactly `buffer.len()` bytes into `buffer`.
    async fn recv(&mut self, buffer: &mut [u8]) -> Result<(), DistributedError> {
        if self.poisoned {
            return Err(DistributedError::ChannelPoisoned);
        }
        let result = self.recv_inner(buffer).await;
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }

    async fn recv_inner(&mut self, buffer: &mut [u8]) -> Result<(), DistributedError> {
        let (total, head) = self.read_header().await?;
        if total != buffer.len() as u64 {
            return Err(DistributedError::Protocol(format!(
                "Expected {} bytes, got {total}",
                buffer.len()
            )));
        }
        if head > buffer.len() {
            return Err(DistributedError::Protocol(
                "first frame overruns the message".into(),
            ));
        }
        buffer[..head].copy_from_slice(&self.scratch[..head]);
        self.fill(buffer, head).await
    }

    async fn recv_vec(&mut self) -> Result<Vec<u8>, DistributedError> {
        if self.poisoned {
            return Err(DistributedError::ChannelPoisoned);
        }
        let result = self.recv_vec_inner().await;
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }

    async fn recv_vec_inner(&mut self) -> Result<Vec<u8>, DistributedError> {
        let (total, head) = self.read_header().await?;
        if total > MAX_VEC_MESSAGE_BYTES {
            return Err(DistributedError::Protocol(format!(
                "Message size {total} bytes exceeds maximum {MAX_VEC_MESSAGE_BYTES} bytes"
            )));
        }
        let mut data = vec![0u8; total as usize];
        if head > data.len() {
            return Err(DistributedError::Protocol(
                "first frame overruns the message".into(),
            ));
        }
        data[..head].copy_from_slice(&self.scratch[..head]);
        self.fill(&mut data, head).await?;
        Ok(data)
    }
}

impl TransportSender {
    /// Send one message.
    pub async fn send(&mut self, data: &[u8]) -> Result<()> {
        match &mut self.inner {
            TransportSenderInner::Tcp(writer) => writer.send(data).await?,
            TransportSenderInner::Memory(sender) => {
                sender.send(data.to_vec()).await.map_err(|e| {
                    DistributedError::Protocol(format!("in-memory send failed: {e}"))
                })?;
            }
        }
        Ok(())
    }
}

impl TransportReceiver {
    /// Receive one message into `buffer`, which must be exactly its size.
    ///
    /// The sender's declared length is checked against `buffer` before any
    /// payload is written, so a peer cannot overrun it.
    pub async fn recv(&mut self, buffer: &mut [u8]) -> Result<()> {
        match &mut self.inner {
            TransportReceiverInner::Tcp(reader) => reader.recv(buffer).await?,
            TransportReceiverInner::Memory(receiver) => {
                let data = recv_memory(receiver).await?;
                if data.len() != buffer.len() {
                    return Err(DistributedError::Protocol(format!(
                        "Expected {} bytes, got {}",
                        buffer.len(),
                        data.len()
                    ))
                    .into());
                }
                buffer.copy_from_slice(&data);
            }
        }
        Ok(())
    }

    /// Receive one message of unknown size (activations with variable
    /// shapes), at most [`MAX_VEC_MESSAGE_BYTES`].
    pub async fn recv_vec(&mut self) -> Result<Vec<u8>> {
        match &mut self.inner {
            TransportReceiverInner::Tcp(reader) => Ok(reader.recv_vec().await?),
            TransportReceiverInner::Memory(receiver) => {
                let data = recv_memory(receiver).await?;
                if data.len() as u64 > MAX_VEC_MESSAGE_BYTES {
                    return Err(DistributedError::Protocol(format!(
                        "Message size {} bytes exceeds maximum {MAX_VEC_MESSAGE_BYTES} bytes",
                        data.len()
                    ))
                    .into());
                }
                Ok(data)
            }
        }
    }
}

async fn recv_memory(receiver: &mut mpsc::Receiver<Vec<u8>>) -> Result<Vec<u8>, DistributedError> {
    timeout(IO_IDLE_TIMEOUT, receiver.recv())
        .await
        .map_err(|_| {
            DistributedError::Protocol(format!(
                "Timed out waiting for in-memory payload ({IO_IDLE_TIMEOUT:?})"
            ))
        })?
        .ok_or_else(|| DistributedError::Protocol("in-memory channel closed".into()))
}

/// Create a unidirectional in-memory transport channel.
///
/// This is used for local same-process pipeline stages, such as UltraFusion
/// multi-die execution planning, where TCP framing would add unnecessary cost.
pub fn in_memory_channel(capacity: usize) -> (TransportSender, TransportReceiver) {
    let (tx, rx) = mpsc::channel(capacity.max(1));
    (
        TransportSender {
            inner: TransportSenderInner::Memory(tx),
        },
        TransportReceiver {
            inner: TransportReceiverInner::Memory(rx),
        },
    )
}

// ─── Handshake ──────────────────────────────────────────────────────────────

/// A connection that has passed the handshake, as its two halves.
pub struct Link {
    /// Send to the peer.
    pub sender: TransportSender,
    /// Receive from the peer.
    pub receiver: TransportReceiver,
}

/// What a handshake needs to know about the connection being made.
#[derive(Clone)]
pub struct LinkParams {
    /// Protection for the link.
    pub security: LinkSecurity,
    /// The logical channel this link carries.
    pub channel: Channel,
    /// This end's view of the ring.
    pub membership: RingMembership,
    /// The rank the other end must be.
    pub peer_rank: usize,
}

fn noise_builder(psk: &[u8; 32]) -> Result<Builder<'_>, DistributedError> {
    let params = NOISE_PATTERN
        .parse()
        .map_err(|e| DistributedError::Protocol(format!("bad noise pattern: {e}")))?;
    Builder::new(params)
        .psk(0, psk)
        .and_then(|b| b.prologue(NOISE_PROLOGUE))
        .map_err(|e| DistributedError::Protocol(format!("noise setup failed: {e}")))
}

async fn write_handshake_message(
    stream: &mut TcpStream,
    message: &[u8],
) -> Result<(), DistributedError> {
    let mut framed = Vec::with_capacity(2 + message.len());
    framed.extend_from_slice(&(message.len() as u16).to_le_bytes());
    framed.extend_from_slice(message);
    timeout(HANDSHAKE_TIMEOUT, stream.write_all(&framed))
        .await
        .map_err(|_| DistributedError::Protocol("handshake write timed out".into()))??;
    Ok(())
}

async fn read_handshake_message(stream: &mut TcpStream) -> Result<Vec<u8>, DistributedError> {
    let timed_out = || DistributedError::Protocol("handshake read timed out".into());
    let mut len = [0u8; 2];
    timeout(HANDSHAKE_TIMEOUT, stream.read_exact(&mut len))
        .await
        .map_err(|_| timed_out())??;
    let mut message = vec![0u8; u16::from_le_bytes(len) as usize];
    timeout(HANDSHAKE_TIMEOUT, stream.read_exact(&mut message))
        .await
        .map_err(|_| timed_out())??;
    Ok(message)
}

fn split(stream: TcpStream, protection: Protection) -> Link {
    let _ = stream.set_nodelay(true);
    let (read, write) = stream.into_split();
    Link {
        sender: TransportSender {
            inner: TransportSenderInner::Tcp(TcpWriter::new(write, protection.clone())),
        },
        receiver: TransportReceiver {
            inner: TransportReceiverInner::Tcp(TcpReader::new(read, protection)),
        },
    }
}

/// Run the initiator side of the handshake on a connected stream.
pub async fn client_handshake(
    mut stream: TcpStream,
    params: &LinkParams,
) -> Result<Link, DistributedError> {
    let hello = Hello::new(params.channel, &params.membership);
    match &params.security {
        LinkSecurity::Plaintext => {
            write_handshake_message(&mut stream, &hello.encode()).await?;
            let theirs = Hello::decode(&read_handshake_message(&mut stream).await?)?;
            hello.check_peer(&theirs, params.peer_rank)?;
            Ok(split(stream, Protection::Plain))
        }
        LinkSecurity::Authenticated(secret) => {
            let psk = secret.ring_psk();
            let mut state = noise_builder(&psk)?
                .build_initiator()
                .map_err(|e| DistributedError::Protocol(format!("noise setup failed: {e}")))?;
            let mut out = vec![0u8; MAX_FRAME_BODY];
            let n = state
                .write_message(&hello.encode(), &mut out)
                .map_err(|e| DistributedError::Protocol(format!("handshake failed: {e}")))?;
            write_handshake_message(&mut stream, &out[..n]).await?;

            // A responder that cannot authenticate us drops the connection
            // without a word, which is all an eavesdropper must learn.
            let reply = read_handshake_message(&mut stream).await.map_err(|_| {
                DistributedError::Auth(
                    "peer closed the connection during the handshake: it is not a pmetal \
                     peer, or its cluster secret differs from this machine's"
                        .into(),
                )
            })?;
            let mut payload = vec![0u8; MAX_FRAME_BODY];
            let n = state.read_message(&reply, &mut payload).map_err(|_| {
                DistributedError::Auth(
                    "peer's handshake reply did not authenticate (cluster secrets differ)".into(),
                )
            })?;
            let theirs = Hello::decode(&payload[..n])?;
            let transport = state
                .into_stateless_transport_mode()
                .map_err(|e| DistributedError::Protocol(format!("handshake failed: {e}")))?;
            hello.check_peer(&theirs, params.peer_rank)?;
            Ok(split(stream, Protection::Noise(Arc::new(transport))))
        }
    }
}

/// Run the responder side of the handshake on an accepted stream.
pub async fn server_handshake(
    mut stream: TcpStream,
    params: &LinkParams,
) -> Result<Link, DistributedError> {
    let hello = Hello::new(params.channel, &params.membership);
    match &params.security {
        LinkSecurity::Plaintext => {
            let theirs = Hello::decode(&read_handshake_message(&mut stream).await?)?;
            write_handshake_message(&mut stream, &hello.encode()).await?;
            hello.check_peer(&theirs, params.peer_rank)?;
            Ok(split(stream, Protection::Plain))
        }
        LinkSecurity::Authenticated(secret) => {
            let psk = secret.ring_psk();
            let mut state = noise_builder(&psk)?
                .build_responder()
                .map_err(|e| DistributedError::Protocol(format!("noise setup failed: {e}")))?;
            let first = read_handshake_message(&mut stream).await?;
            let mut payload = vec![0u8; MAX_FRAME_BODY];
            let n = state.read_message(&first, &mut payload).map_err(|_| {
                DistributedError::Auth(
                    "a connection failed authentication (wrong cluster secret, or not a \
                     pmetal peer)"
                        .into(),
                )
            })?;
            let theirs = Hello::decode(&payload[..n])?;

            // The peer holds the secret, so telling it our hello (and then
            // why we are about to refuse it) leaks nothing.
            let mut out = vec![0u8; MAX_FRAME_BODY];
            let n = state
                .write_message(&hello.encode(), &mut out)
                .map_err(|e| DistributedError::Protocol(format!("handshake failed: {e}")))?;
            write_handshake_message(&mut stream, &out[..n]).await?;
            let transport = state
                .into_stateless_transport_mode()
                .map_err(|e| DistributedError::Protocol(format!("handshake failed: {e}")))?;
            hello.check_peer(&theirs, params.peer_rank)?;
            Ok(split(stream, Protection::Noise(Arc::new(transport))))
        }
    }
}

/// TCP transport for ring communication.
pub struct TcpTransport;

impl TcpTransport {
    /// Connect to peers in a ring topology, with fabric-aware fallback.
    ///
    /// Each rank:
    ///   1. Binds the listener on `0.0.0.0:<my_port>`. Listening on the
    ///      unspecified address means peers can dial in over *any* fabric
    ///      we have an interface on — Thunderbolt or Ethernet — without
    ///      forcing us to pick one at bind time.
    ///   2. Connects to the next peer by trying each address in
    ///      `addrs_for(next_rank)` in priority order (best fabric first).
    ///      Each address gets `max_retries / addrs.len()` retry budget;
    ///      we exhaust one before falling back to the next.
    ///   3. Authenticates both connections ([`LinkSecurity::from_config`]).
    pub async fn connect(
        config: &DistributedConfig,
    ) -> Result<(TransportSender, TransportReceiver)> {
        let security = LinkSecurity::from_config(config)?;
        let membership = RingMembership::from_config(config);
        Self::connect_ring(config, security, membership, None, Channel::Ring).await
    }

    /// Form the ring. `listener` is a socket the caller already bound for
    /// this rank; without one, this binds `0.0.0.0` on the rank's port.
    pub(crate) async fn connect_ring(
        config: &DistributedConfig,
        security: LinkSecurity,
        membership: RingMembership,
        listener: Option<TcpListener>,
        channel: Channel,
    ) -> Result<(TransportSender, TransportReceiver)> {
        let world_size = config.nodes.len();
        let rank = config.rank;
        let my_addr = config.nodes[rank];
        let connection_timeout = Duration::from_millis(config.connection_timeout_ms);

        let listener = match listener {
            Some(listener) => listener,
            None => {
                // Bind on 0.0.0.0:<my_port> so we accept on every local interface.
                // Falling back to a specific address would mean a TB cable disconnect
                // breaks the listener even though Ethernet is still up.
                let bind_addr = SocketAddr::new(
                    "0.0.0.0".parse().expect("0.0.0.0 always parses"),
                    my_addr.port(),
                );
                info!(
                    "Node {} listening on {} (advertised as {})",
                    rank, bind_addr, my_addr
                );
                TcpListener::bind(bind_addr).await?
            }
        };

        let next_rank = (rank + 1) % world_size;
        let prev_rank = (rank + world_size - 1) % world_size;
        let next_endpoints = config.addrs_for(next_rank);
        if next_endpoints.is_empty() {
            return Err(DistributedError::Config(format!(
                "no addresses configured for next peer rank {}",
                next_rank
            ))
            .into());
        }

        let dial_params = LinkParams {
            security: security.clone(),
            channel,
            membership: membership.clone(),
            peer_rank: next_rank,
        };
        let accept_params = LinkParams {
            security,
            channel,
            membership,
            peer_rank: prev_rank,
        };

        // Walk fabrics in priority order, retrying each before falling through.
        let connect_fut = Self::connect_with_fallback(
            next_rank,
            &next_endpoints,
            config.max_retries,
            &dial_params,
        );

        // Accept the connection from the previous peer.
        let accept_fut = async {
            match timeout(
                connection_timeout,
                accept_authenticated(&listener, &accept_params),
            )
            .await
            {
                Ok(result) => result,
                Err(_) => Err(DistributedError::Protocol(format!(
                    "Timeout waiting for incoming connection ({}ms)",
                    config.connection_timeout_ms
                ))),
            }
        };

        let (next_peer, prev_peer) = tokio::try_join!(
            async { connect_fut.await.map_err(anyhow::Error::from) },
            async { accept_fut.await.map_err(anyhow::Error::from) }
        )?;

        // Gradients flow one way round the ring: out on the connection to the
        // next rank, in on the connection from the previous one.
        Ok((next_peer.sender, prev_peer.receiver))
    }

    /// Round-robin connect: try each address once per round with shared
    /// exponential backoff between rounds. A transient outage on the
    /// best-fabric (e.g. a Thunderbolt cable jiggle) falls back to the
    /// next fabric within a few seconds rather than after the entire
    /// per-fabric budget is exhausted.
    ///
    /// `max_retries` caps the total number of *connect attempts across all
    /// fabrics*, not per fabric. With 2 endpoints and `max_retries = 50`,
    /// each address is tried up to ~25 times.
    pub(crate) async fn connect_with_fallback(
        next_rank: usize,
        endpoints: &[SocketAddr],
        max_retries: u32,
        params: &LinkParams,
    ) -> std::result::Result<Link, DistributedError> {
        if endpoints.is_empty() {
            return Err(DistributedError::Config(format!(
                "no endpoints configured for next peer rank {next_rank}"
            )));
        }
        let mut last_err: Option<String> = None;
        let mut attempts: u32 = 0;
        let mut round: u32 = 0;

        while attempts < max_retries {
            for (idx, addr) in endpoints.iter().enumerate() {
                if attempts >= max_retries {
                    break;
                }
                attempts += 1;

                match TcpStream::connect(addr).await {
                    Ok(stream) => match client_handshake(stream, params).await {
                        Ok(link) => {
                            info!(
                                "Connected to next peer {} at {} (fabric {}/{}, round {}, {})",
                                next_rank,
                                addr,
                                idx + 1,
                                endpoints.len(),
                                round + 1,
                                if params.security.is_encrypted() {
                                    "encrypted"
                                } else {
                                    "PLAINTEXT"
                                },
                            );
                            return Ok(link);
                        }
                        // The peer is reachable and authenticated us, but the
                        // cluster is configured inconsistently: retrying
                        // cannot fix that.
                        Err(e) if is_membership_error(&e) => return Err(e),
                        Err(e) => {
                            debug!(
                                "Handshake with peer {} at {} failed: {}",
                                next_rank, addr, e
                            );
                            last_err = Some(e.to_string());
                        }
                    },
                    Err(e) => {
                        if round == 0 {
                            debug!("Waiting for peer {} at {} ({})", next_rank, addr, e);
                        }
                        last_err = Some(e.to_string());
                    }
                }
            }

            // Finished one round across every fabric — sleep before retrying.
            round += 1;
            let backoff = (INITIAL_BACKOFF_MS * 2u64.pow(round.min(6))).min(MAX_BACKOFF_MS);
            tokio::time::sleep(Duration::from_millis(backoff)).await;
        }

        Err(match last_err {
            Some(reason) => DistributedError::ConnectionFailed {
                addr: endpoints[0],
                reason: format!("gave up after {max_retries} attempts; last error: {reason}"),
            },
            None => DistributedError::MaxRetriesExceeded {
                addr: endpoints[0],
                max_retries,
            },
        })
    }
}

/// Errors that mean "the peer answered and we disagree about the cluster",
/// as opposed to "the peer is not up yet".
fn is_membership_error(e: &DistributedError) -> bool {
    matches!(
        e,
        DistributedError::ClusterMismatch(_) | DistributedError::VersionMismatch { .. }
    )
}

/// Accept connections until one completes the handshake. Peers that cannot
/// authenticate (or never finish) are dropped and the wait goes on; a peer
/// that authenticates but disagrees about the cluster is an error.
pub(crate) async fn accept_authenticated(
    listener: &TcpListener,
    params: &LinkParams,
) -> Result<Link, DistributedError> {
    loop {
        let (stream, addr) = listener.accept().await?;
        match server_handshake(stream, params).await {
            Ok(link) => {
                info!("Accepted authenticated connection from {}", addr);
                return Ok(link);
            }
            Err(e) if is_membership_error(&e) => return Err(e),
            Err(e) => warn!("Rejected incoming connection from {}: {}", addr, e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn in_memory_transport_roundtrip() {
        let (mut sender, mut receiver) = in_memory_channel(4);
        sender.send(b"hello").await.expect("send");

        let mut buf = [0u8; 5];
        receiver.recv(&mut buf).await.expect("recv");
        assert_eq!(&buf, b"hello");
    }

    #[tokio::test]
    async fn in_memory_transport_recv_vec_roundtrip() {
        let (mut sender, mut receiver) = in_memory_channel(4);
        sender.send(b"payload").await.expect("send");

        let data = receiver.recv_vec().await.expect("recv_vec");
        assert_eq!(data, b"payload");
    }

    #[tokio::test]
    async fn in_memory_transport_len_mismatch_errors() {
        let (mut sender, mut receiver) = in_memory_channel(4);
        sender.send(b"toolong").await.expect("send");

        let mut buf = [0u8; 3];
        let err = receiver.recv(&mut buf).await.expect_err("length mismatch");
        assert!(err.to_string().contains("Expected 3 bytes, got 7"));
    }
}

#[cfg(test)]
#[path = "transport_tests.rs"]
mod link_tests;
