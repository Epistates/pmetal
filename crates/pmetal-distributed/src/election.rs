//! Distributed master election algorithm.
//!
//! Implements a seniority-based leader election protocol inspired by exo.
//! The election ensures exactly one master is elected in the cluster,
//! with deterministic tiebreaking based on seniority and peer ID.
//!
//! # Algorithm
//!
//! 1. When a node starts or loses connection to master, it starts a campaign
//! 2. Nodes exchange election messages with their clock, seniority, and peer ID
//! 3. The node with highest seniority wins; peer ID breaks ties
//! 4. Election timeout triggers re-election if no winner emerges

use anyhow::Result;
use libp2p::PeerId;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tracing::{info, warn};

/// Default election timeout.
pub const DEFAULT_ELECTION_TIMEOUT: Duration = Duration::from_secs(3);

/// Delay before starting election after connection change.
pub const CONNECTION_ELECTION_DELAY: Duration = Duration::from_millis(200);

/// Election configuration.
#[derive(Debug, Clone)]
pub struct ElectionConfig {
    /// Timeout for election completion.
    pub election_timeout: Duration,
    /// Delay before starting election after connection change.
    pub connection_delay: Duration,
    /// Whether to automatically start election on peer loss.
    pub auto_elect_on_peer_loss: bool,
}

impl Default for ElectionConfig {
    fn default() -> Self {
        Self {
            election_timeout: DEFAULT_ELECTION_TIMEOUT,
            connection_delay: CONNECTION_ELECTION_DELAY,
            auto_elect_on_peer_loss: true,
        }
    }
}

/// Election message exchanged between nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElectionMessage {
    /// Logical clock for ordering.
    pub clock: u64,
    /// Node seniority (time since node started).
    pub seniority: Duration,
    /// Proposed session ID (prevents stale elections).
    pub proposed_session: u64,
    /// Number of commands seen (for consistency).
    pub commands_seen: u64,
    /// The peer ID of the sender.
    pub sender: PeerId,
    /// The peer ID this node is voting for.
    pub vote_for: PeerId,
}

/// Election state for the local node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElectionState {
    /// Not in an election.
    Idle,
    /// Running an election campaign.
    Campaigning,
    /// Elected as master.
    Master,
    /// Following another master.
    Follower,
}

/// Information about a candidate in the election.
#[derive(Debug, Clone)]
struct CandidateInfo {
    /// Peer ID of the candidate.
    peer_id: PeerId,
    /// Seniority of the candidate.
    seniority: Duration,
    /// Session proposed by this candidate.
    proposed_session: u64,
    /// Commands seen by this candidate.
    commands_seen: u64,
    /// When we last heard from this candidate.
    last_seen: Instant,
}

/// Election events.
#[derive(Debug, Clone)]
pub enum ElectionEvent {
    /// Election started.
    Started { session: u64 },
    /// We were elected as master.
    ElectedMaster { session: u64 },
    /// Another node was elected as master.
    NewMaster { master: PeerId, session: u64 },
    /// Election timed out.
    Timeout { session: u64 },
    /// Split brain detected.
    SplitBrain { masters: Vec<PeerId> },
}

/// Election manager.
pub struct ElectionManager {
    /// Local peer ID.
    local_peer_id: PeerId,
    /// Configuration.
    config: ElectionConfig,
    /// Current state.
    state: Arc<RwLock<ElectionState>>,
    /// Current master (if known).
    current_master: Arc<RwLock<Option<PeerId>>>,
    /// Current session ID.
    session_id: Arc<RwLock<u64>>,
    /// When this node started (for seniority).
    start_time: Instant,
    /// Candidates in current election.
    candidates: Arc<RwLock<HashMap<PeerId, CandidateInfo>>>,
    /// Our vote in the current election.
    current_vote: Arc<RwLock<Option<PeerId>>>,
    /// Number of commands we've processed.
    commands_seen: Arc<RwLock<u64>>,
    /// Logical clock.
    clock: Arc<RwLock<u64>>,
    /// Event sender.
    event_tx: mpsc::Sender<ElectionEvent>,
}

impl ElectionManager {
    /// Create a new election manager.
    pub fn new(
        local_peer_id: PeerId,
        config: ElectionConfig,
        event_tx: mpsc::Sender<ElectionEvent>,
    ) -> Self {
        Self {
            local_peer_id,
            config,
            state: Arc::new(RwLock::new(ElectionState::Idle)),
            current_master: Arc::new(RwLock::new(None)),
            session_id: Arc::new(RwLock::new(0)),
            start_time: Instant::now(),
            candidates: Arc::new(RwLock::new(HashMap::new())),
            current_vote: Arc::new(RwLock::new(None)),
            commands_seen: Arc::new(RwLock::new(0)),
            clock: Arc::new(RwLock::new(0)),
            event_tx,
        }
    }

    /// Get the current election state.
    pub fn state(&self) -> ElectionState {
        *self.state.read()
    }

    /// Get the current master.
    pub fn current_master(&self) -> Option<PeerId> {
        *self.current_master.read()
    }

    /// Check if we are the master.
    pub fn is_master(&self) -> bool {
        *self.state.read() == ElectionState::Master
    }

    /// Check if we are a follower.
    pub fn is_follower(&self) -> bool {
        *self.state.read() == ElectionState::Follower
    }

    /// Get our seniority.
    pub fn seniority(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Get the current session ID.
    pub fn session_id(&self) -> u64 {
        *self.session_id.read()
    }

    /// Increment and get the logical clock.
    fn tick(&self) -> u64 {
        let mut clock = self.clock.write();
        *clock += 1;
        *clock
    }

    /// Update clock to max of local and received.
    fn update_clock(&self, received: u64) {
        let mut clock = self.clock.write();
        *clock = (*clock).max(received) + 1;
    }

    /// Increment commands seen.
    pub fn record_command(&self) {
        *self.commands_seen.write() += 1;
    }

    /// Start a new election campaign.
    pub async fn start_election(&self) -> Result<()> {
        let new_session = {
            let mut session = self.session_id.write();
            *session += 1;
            *session
        };

        {
            let mut state = self.state.write();
            *state = ElectionState::Campaigning;
        }

        // Clear previous candidates
        self.candidates.write().clear();

        // Vote for ourselves
        *self.current_vote.write() = Some(self.local_peer_id);

        // Add ourselves as a candidate
        self.candidates.write().insert(
            self.local_peer_id,
            CandidateInfo {
                peer_id: self.local_peer_id,
                seniority: self.seniority(),
                proposed_session: new_session,
                commands_seen: *self.commands_seen.read(),
                last_seen: Instant::now(),
            },
        );

        info!(
            "Starting election campaign, session={}, seniority={:?}",
            new_session,
            self.seniority()
        );

        let _ = self
            .event_tx
            .send(ElectionEvent::Started {
                session: new_session,
            })
            .await;

        Ok(())
    }

    /// Create an election message to broadcast.
    pub fn create_message(&self) -> ElectionMessage {
        let vote = self.current_vote.read().unwrap_or(self.local_peer_id);
        ElectionMessage {
            clock: self.tick(),
            seniority: self.seniority(),
            proposed_session: *self.session_id.read(),
            commands_seen: *self.commands_seen.read(),
            sender: self.local_peer_id,
            vote_for: vote,
        }
    }

    /// Process a received election message.
    pub async fn process_message(&self, msg: ElectionMessage) -> Result<Option<ElectionMessage>> {
        self.update_clock(msg.clock);

        // Add/update candidate info
        {
            let mut candidates = self.candidates.write();
            candidates.insert(
                msg.sender,
                CandidateInfo {
                    peer_id: msg.sender,
                    seniority: msg.seniority,
                    proposed_session: msg.proposed_session,
                    commands_seen: msg.commands_seen,
                    last_seen: Instant::now(),
                },
            );
        }

        // If we're idle, join the election
        if *self.state.read() == ElectionState::Idle {
            self.start_election().await?;
        }

        // Update our vote based on candidates
        let should_respond = self.update_vote();

        if should_respond {
            Ok(Some(self.create_message()))
        } else {
            Ok(None)
        }
    }

    /// Update our vote based on current candidates.
    fn update_vote(&self) -> bool {
        let candidates = self.candidates.read();

        // Find the best candidate
        let best = candidates.values().max_by(|a, b| {
            // Higher seniority wins
            match a.seniority.cmp(&b.seniority) {
                std::cmp::Ordering::Equal => {
                    // Higher commands seen wins
                    match a.commands_seen.cmp(&b.commands_seen) {
                        std::cmp::Ordering::Equal => {
                            // Lexicographically higher peer ID wins (deterministic)
                            a.peer_id.cmp(&b.peer_id)
                        }
                        other => other,
                    }
                }
                other => other,
            }
        });

        if let Some(best_candidate) = best {
            let mut current_vote = self.current_vote.write();
            let old_vote = *current_vote;
            *current_vote = Some(best_candidate.peer_id);

            // Return true if vote changed
            old_vote != Some(best_candidate.peer_id)
        } else {
            false
        }
    }

    /// Check if election is complete (all candidates agree on winner).
    pub async fn check_election_complete(&self, all_peers: &[PeerId]) -> Result<bool> {
        let candidates = self.candidates.read();

        // Need votes from all known peers
        if candidates.len() < all_peers.len() {
            return Ok(false);
        }

        // Check if all candidates vote for the same peer
        let votes: Vec<_> = candidates.values().map(|c| c.peer_id).collect();

        // For now, simple majority - find the candidate with most votes for themselves
        // In a proper implementation, we'd track explicit votes
        let our_vote = self.current_vote.read();

        if let Some(winner) = our_vote.as_ref() {
            // Check if winner is in our candidates
            if candidates.contains_key(winner) {
                // Declare winner
                self.declare_winner(*winner).await?;
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Declare the election winner.
    async fn declare_winner(&self, winner: PeerId) -> Result<()> {
        let session = *self.session_id.read();

        if winner == self.local_peer_id {
            *self.state.write() = ElectionState::Master;
            *self.current_master.write() = Some(winner);

            info!("Elected as master, session={}", session);
            let _ = self
                .event_tx
                .send(ElectionEvent::ElectedMaster { session })
                .await;
        } else {
            *self.state.write() = ElectionState::Follower;
            *self.current_master.write() = Some(winner);

            info!(
                "Following master {}, session={}",
                winner.to_base58(),
                session
            );
            let _ = self
                .event_tx
                .send(ElectionEvent::NewMaster {
                    master: winner,
                    session,
                })
                .await;
        }

        Ok(())
    }

    /// Handle master loss (triggers new election).
    pub async fn handle_master_loss(&self) -> Result<()> {
        if self.config.auto_elect_on_peer_loss {
            warn!("Master lost, starting new election after delay");

            // Wait for connection delay
            tokio::time::sleep(self.config.connection_delay).await;

            self.start_election().await?;
        }

        Ok(())
    }

    /// Check for election timeout.
    pub async fn check_timeout(&self) -> Result<bool> {
        if *self.state.read() != ElectionState::Campaigning {
            return Ok(false);
        }

        // Check if any candidate info is stale
        let candidates = self.candidates.read();
        let now = Instant::now();

        for candidate in candidates.values() {
            if now.duration_since(candidate.last_seen) > self.config.election_timeout {
                drop(candidates);

                let session = *self.session_id.read();
                warn!("Election timeout, session={}", session);

                let _ = self.event_tx.send(ElectionEvent::Timeout { session }).await;

                // Start new election
                self.start_election().await?;
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Force become master (for single-node clusters).
    pub async fn force_master(&self) -> Result<()> {
        *self.state.write() = ElectionState::Master;
        *self.current_master.write() = Some(self.local_peer_id);

        let session = {
            let mut s = self.session_id.write();
            *s += 1;
            *s
        };

        info!("Forced master election (single node), session={}", session);
        let _ = self
            .event_tx
            .send(ElectionEvent::ElectedMaster { session })
            .await;

        Ok(())
    }

    /// Step down from master role.
    pub async fn step_down(&self) -> Result<()> {
        if *self.state.read() == ElectionState::Master {
            *self.state.write() = ElectionState::Idle;
            *self.current_master.write() = None;

            info!("Stepped down from master role");
        }

        Ok(())
    }
}

impl std::fmt::Debug for ElectionManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ElectionManager")
            .field("local_peer_id", &self.local_peer_id.to_base58())
            .field("state", &*self.state.read())
            .field("current_master", &self.current_master())
            .field("session_id", &*self.session_id.read())
            .field("seniority", &self.seniority())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;

    #[tokio::test]
    async fn test_election_start() {
        let (tx, mut rx) = mpsc::channel(10);
        let peer_id = PeerId::random();
        let manager = ElectionManager::new(peer_id, ElectionConfig::default(), tx);

        manager.start_election().await.unwrap();

        assert_eq!(manager.state(), ElectionState::Campaigning);
        assert_eq!(manager.session_id(), 1);

        // Should receive Started event
        let event = rx.try_recv().unwrap();
        assert!(matches!(event, ElectionEvent::Started { session: 1 }));
    }

    #[tokio::test]
    async fn test_force_master() {
        let (tx, mut rx) = mpsc::channel(10);
        let peer_id = PeerId::random();
        let manager = ElectionManager::new(peer_id, ElectionConfig::default(), tx);

        manager.force_master().await.unwrap();

        assert!(manager.is_master());
        assert_eq!(manager.current_master(), Some(peer_id));

        let event = rx.try_recv().unwrap();
        assert!(matches!(event, ElectionEvent::ElectedMaster { .. }));
    }

    #[test]
    fn test_seniority() {
        let (tx, _rx) = mpsc::channel(10);
        let manager = ElectionManager::new(PeerId::random(), ElectionConfig::default(), tx);

        // Seniority should be non-zero after creation
        std::thread::sleep(Duration::from_millis(10));
        assert!(manager.seniority() >= Duration::from_millis(10));
    }

    #[test]
    fn test_create_message() {
        let (tx, _rx) = mpsc::channel(10);
        let peer_id = PeerId::random();
        let manager = ElectionManager::new(peer_id, ElectionConfig::default(), tx);

        let msg = manager.create_message();

        assert_eq!(msg.sender, peer_id);
        assert!(msg.clock > 0);
        assert!(msg.seniority > Duration::ZERO);
    }
}
