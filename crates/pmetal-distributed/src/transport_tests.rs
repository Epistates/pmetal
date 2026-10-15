//! Framing, handshake and tamper-resistance of the TCP transport, over
//! loopback only.

use super::*;
use crate::testing::{fresh_security, tcp_link_pair};
use tokio::net::TcpListener;

fn pattern(len: usize, salt: u8) -> Vec<u8> {
    (0..len)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(salt))
        .collect()
}

/// Sizes at and around every framing boundary.
fn boundary_sizes() -> Vec<usize> {
    let chunk = MAX_FRAME_BODY - TAG_LEN;
    let mut sizes = vec![0, 1, 7, 8, 9, 1000];
    for edge in [chunk - 8, chunk, 2 * chunk - 8, 2 * chunk, 3 * chunk + 5] {
        sizes.extend([edge - 1, edge, edge + 1]);
    }
    sizes.push(3_000_000);
    sizes
}

async fn round_trips(security: LinkSecurity) {
    let (mut a, mut b) = tcp_link_pair(security).await;
    for (i, len) in boundary_sizes().into_iter().enumerate() {
        let message = pattern(len, i as u8);
        let send = a.sender.send(&message);
        let mut received = vec![0u8; len];
        let recv = b.receiver.recv(&mut received);
        let (sent, got) = tokio::join!(send, recv);
        sent.expect("send");
        got.expect("recv");
        assert_eq!(received, message, "size {len}");
    }
    // And the variable-size receive.
    for (i, len) in boundary_sizes().into_iter().enumerate() {
        let message = pattern(len, 100 + i as u8);
        let (sent, got) = tokio::join!(a.sender.send(&message), b.receiver.recv_vec());
        sent.expect("send");
        assert_eq!(got.expect("recv_vec"), message, "size {len}");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn authenticated_links_carry_messages_of_every_size() {
    round_trips(fresh_security()).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn plaintext_links_carry_messages_of_every_size() {
    round_trips(LinkSecurity::Plaintext).await;
}

/// A connection that cannot produce the secret is dropped at the handshake,
/// and the acceptor goes on to admit a legitimate peer.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_peer_without_the_secret_is_rejected_and_the_acceptor_carries_on() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let secret = ClusterSecret::generate();
    let membership = |rank| RingMembership {
        rank,
        world_size: 2,
        digest: [3; 32],
    };
    let params = |security: LinkSecurity, rank, peer_rank| LinkParams {
        security,
        channel: Channel::Ring,
        membership: membership(rank),
        peer_rank,
    };
    let accept_params = params(LinkSecurity::Authenticated(secret.clone()), 1, 0);
    let acceptor =
        tokio::spawn(async move { accept_authenticated(&listener, &accept_params).await });

    // Wrong secret.
    let intruder = params(LinkSecurity::Authenticated(ClusterSecret::generate()), 0, 1);
    let stream = TcpStream::connect(addr).await.unwrap();
    let err = client_handshake(stream, &intruder).await.err().unwrap();
    assert!(matches!(err, DistributedError::Auth(_)), "{err}");

    // Not our protocol at all.
    let mut garbage = TcpStream::connect(addr).await.unwrap();
    garbage
        .write_all(b"GET / HTTP/1.1\r\nHost: x\r\n\r\n")
        .await
        .unwrap();
    drop(garbage);

    // The real thing still gets in.
    let honest = params(LinkSecurity::Authenticated(secret), 0, 1);
    let stream = TcpStream::connect(addr).await.unwrap();
    let mut link = client_handshake(stream, &honest).await.expect("honest");
    let mut theirs = acceptor.await.unwrap().expect("accepted");
    let (sent, got) = tokio::join!(link.sender.send(b"hello"), async {
        let mut buf = [0u8; 5];
        theirs.receiver.recv(&mut buf).await.map(|()| buf)
    });
    sent.unwrap();
    assert_eq!(&got.unwrap(), b"hello");
}

/// Two machines that hold the secret but were configured with different
/// rings find out at the handshake, with the reason.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn disagreeing_ring_membership_fails_the_handshake_with_the_reason() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let security = fresh_security();
    let params = |rank, peer_rank, digest| LinkParams {
        security: security.clone(),
        channel: Channel::Ring,
        membership: RingMembership {
            rank,
            world_size: 2,
            digest,
        },
        peer_rank,
    };
    let accept_params = params(1, 0, [2; 32]);
    let acceptor =
        tokio::spawn(async move { accept_authenticated(&listener, &accept_params).await });
    let stream = TcpStream::connect(addr).await.unwrap();
    let dial_err = client_handshake(stream, &params(0, 1, [1; 32]))
        .await
        .err()
        .unwrap();
    assert!(
        matches!(&dial_err, DistributedError::ClusterMismatch(m) if m.contains("membership")),
        "{dial_err}"
    );
    let accept_err = acceptor.await.unwrap().err().unwrap();
    assert!(
        matches!(&accept_err, DistributedError::ClusterMismatch(m) if m.contains("membership")),
        "{accept_err}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_peer_that_is_the_wrong_rank_is_refused() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let security = fresh_security();
    let params = |rank, peer_rank| LinkParams {
        security: security.clone(),
        channel: Channel::Ring,
        membership: RingMembership {
            rank,
            world_size: 3,
            digest: [5; 32],
        },
        peer_rank,
    };
    // The acceptor (rank 1) expects rank 0 but the dialer claims to be rank 2.
    let accept_params = params(1, 0);
    let acceptor =
        tokio::spawn(async move { accept_authenticated(&listener, &accept_params).await });
    let stream = TcpStream::connect(addr).await.unwrap();
    let _ = client_handshake(stream, &params(2, 1)).await;
    let err = acceptor.await.unwrap().err().unwrap();
    assert!(
        matches!(&err, DistributedError::ClusterMismatch(m) if m.contains("rank")),
        "{err}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_channel_mismatch_is_refused() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let security = fresh_security();
    let params = |channel| LinkParams {
        security: security.clone(),
        channel,
        membership: RingMembership {
            rank: 0,
            world_size: 1,
            digest: [0; 32],
        },
        peer_rank: 0,
    };
    let accept_params = params(Channel::Activations);
    let acceptor =
        tokio::spawn(async move { accept_authenticated(&listener, &accept_params).await });
    let stream = TcpStream::connect(addr).await.unwrap();
    let err = client_handshake(stream, &params(Channel::Ring))
        .await
        .err()
        .unwrap();
    assert!(matches!(err, DistributedError::ClusterMismatch(_)), "{err}");
    assert!(acceptor.await.unwrap().is_err());
}

/// What a recorded-and-modified byte stream does to the receiver.
#[derive(Clone, Copy, Debug)]
enum Attack {
    FlipBit,
    ReplayFrame,
    ReorderFrames,
    DropFrame,
    TruncateMidFrame,
}

/// Split a byte stream of data frames into the frames.
fn split_frames(stream: &[u8]) -> Vec<Vec<u8>> {
    let mut frames = Vec::new();
    let mut at = 0;
    while at < stream.len() {
        let len = u16::from_le_bytes([stream[at], stream[at + 1]]) as usize;
        frames.push(stream[at..at + 2 + len].to_vec());
        at += 2 + len;
    }
    frames
}

/// Dial a receiver through a proxy that rewrites the data the dialer sends,
/// and return what the receiver makes of each of `messages`.
async fn attack(attack: Attack, messages: &[Vec<u8>]) -> Vec<Result<Vec<u8>, String>> {
    let security = fresh_security();
    let membership = |rank| RingMembership {
        rank,
        world_size: 2,
        digest: [9; 32],
    };
    let params = |rank, peer_rank| LinkParams {
        security: security.clone(),
        channel: Channel::Ring,
        membership: membership(rank),
        peer_rank,
    };

    let server_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let server_addr = server_listener.local_addr().unwrap();
    let proxy_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let proxy_addr = proxy_listener.local_addr().unwrap();

    let accept_params = params(1, 0);
    let acceptor = tokio::spawn(async move {
        accept_authenticated(&server_listener, &accept_params)
            .await
            .expect("accepted")
    });

    // The proxy relays the handshake in both directions, then buffers the
    // dialer's data, damages it, and delivers what is left.
    tokio::spawn(async move {
        let (client, _) = proxy_listener.accept().await.unwrap();
        let server = TcpStream::connect(server_addr).await.unwrap();
        let (mut client_r, mut client_w) = client.into_split();
        let (mut server_r, mut server_w) = server.into_split();
        tokio::spawn(async move {
            let _ = tokio::io::copy(&mut server_r, &mut client_w).await;
        });
        // The handshake message goes through at once (the server's reply is
        // what lets the dialer start sending); only the data is held back.
        let mut len = [0u8; 2];
        client_r.read_exact(&mut len).await.unwrap();
        let mut handshake = len.to_vec();
        handshake.resize(2 + u16::from_le_bytes(len) as usize, 0);
        client_r.read_exact(&mut handshake[2..]).await.unwrap();
        server_w.write_all(&handshake).await.unwrap();
        let mut recorded = Vec::new();
        client_r.read_to_end(&mut recorded).await.unwrap();
        let mut frames = split_frames(&recorded);
        let mut out = Vec::new();
        match attack {
            Attack::FlipBit => {
                let frame = &mut frames[0];
                let last = frame.len() - 1;
                frame[last] ^= 0x01;
            }
            Attack::ReplayFrame => {
                let first = frames[0].clone();
                frames.insert(1, first);
            }
            Attack::ReorderFrames => frames.swap(0, 1),
            Attack::DropFrame => {
                frames.remove(0);
            }
            Attack::TruncateMidFrame => {
                let last = frames.last_mut().unwrap();
                last.truncate(last.len() / 2);
            }
        }
        for frame in frames {
            out.extend_from_slice(&frame);
        }
        let _ = server_w.write_all(&out).await;
        let _ = server_w.shutdown().await;
    });

    let stream = TcpStream::connect(proxy_addr).await.unwrap();
    let mut dialer = client_handshake(stream, &params(0, 1)).await.unwrap();
    let mut victim = acceptor.await.unwrap();
    for message in messages {
        dialer.sender.send(message).await.unwrap();
    }
    drop(dialer);

    let mut results = Vec::new();
    for message in messages {
        let mut buf = vec![0u8; message.len()];
        results.push(
            victim
                .receiver
                .recv(&mut buf)
                .await
                .map(|()| buf)
                .map_err(|e| e.to_string()),
        );
    }
    results
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_modified_frame_is_detected() {
    let messages = vec![pattern(300, 1), pattern(300, 2)];
    let results = attack(Attack::FlipBit, &messages).await;
    let err = results[0].as_ref().expect_err("tampering must be caught");
    assert!(err.contains("authentication"), "{err}");
    // The stream is unusable afterwards.
    assert!(
        results[1]
            .as_ref()
            .expect_err("poisoned")
            .contains("poisoned")
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_replayed_frame_is_detected() {
    let messages = vec![pattern(300, 1), pattern(300, 2)];
    let results = attack(Attack::ReplayFrame, &messages).await;
    assert_eq!(results[0].as_ref().unwrap(), &messages[0]);
    let err = results[1].as_ref().expect_err("replay must be caught");
    assert!(err.contains("authentication"), "{err}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn reordered_frames_are_detected() {
    let messages = vec![pattern(300, 1), pattern(300, 2)];
    let results = attack(Attack::ReorderFrames, &messages).await;
    let err = results[0].as_ref().expect_err("reordering must be caught");
    assert!(err.contains("authentication"), "{err}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_dropped_frame_is_detected() {
    let messages = vec![pattern(300, 1), pattern(300, 2)];
    let results = attack(Attack::DropFrame, &messages).await;
    let err = results[0].as_ref().expect_err("a gap must be caught");
    assert!(err.contains("authentication"), "{err}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_truncated_stream_is_an_error() {
    let messages = vec![pattern(300, 1), pattern(200_000, 2)];
    let results = attack(Attack::TruncateMidFrame, &messages).await;
    assert_eq!(results[0].as_ref().unwrap(), &messages[0]);
    let err = results[1].as_ref().expect_err("truncation must be caught");
    assert!(
        err.contains("closed") || err.contains("authentication"),
        "{err}"
    );
}

/// The length a sender declares is checked against the buffer before any
/// payload is written to it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_message_of_the_wrong_size_never_touches_the_buffer() {
    let (mut a, mut b) = tcp_link_pair(fresh_security()).await;
    a.sender.send(&pattern(100_000, 4)).await.unwrap();
    let mut buf = vec![0xAAu8; 10];
    let err = b.receiver.recv(&mut buf).await.expect_err("size mismatch");
    assert!(err.to_string().contains("Expected 10 bytes"), "{err}");
    assert!(buf.iter().all(|&x| x == 0xAA));

    let err = b.receiver.recv(&mut buf).await.expect_err("poisoned");
    assert!(err.to_string().contains("poisoned"), "{err}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_closed_peer_is_reported_as_such() {
    let (a, mut b) = tcp_link_pair(fresh_security()).await;
    drop(a);
    let mut buf = [0u8; 4];
    let err = b.receiver.recv(&mut buf).await.expect_err("closed");
    assert!(err.to_string().contains("closed"), "{err}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn connect_falls_back_when_first_fabric_unreachable() {
    // Bind a real listener on 127.0.0.1; this is the *fallback* address.
    // The "primary" address points at a closed port.
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let working_addr = listener.local_addr().unwrap();
    let unreachable: SocketAddr = "127.0.0.1:1".parse().unwrap();
    let endpoints = vec![unreachable, working_addr];

    let security = fresh_security();
    let params = |rank, peer_rank| LinkParams {
        security: security.clone(),
        channel: Channel::Ring,
        membership: RingMembership {
            rank,
            world_size: 2,
            digest: [1; 32],
        },
        peer_rank,
    };
    let accept_params = params(1, 0);
    let accepting =
        tokio::spawn(async move { accept_authenticated(&listener, &accept_params).await });
    let dial_params = params(0, 1);
    let connect = tokio::spawn(async move {
        // Tight retry budget so the unreachable fabric exhausts quickly.
        TcpTransport::connect_with_fallback(1, &endpoints, 4, &dial_params).await
    });

    accepting
        .await
        .unwrap()
        .expect("accepted over the fallback");
    connect.await.unwrap().expect("connected over the fallback");
}
