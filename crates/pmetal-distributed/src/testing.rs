//! Loopback rings for tests. Nothing here leaves 127.0.0.1.

use crate::collective::RingCollective;
use crate::config::DistributedConfig;
use crate::secret::ClusterSecret;
use crate::transport::{
    Channel, Link, LinkParams, LinkSecurity, RingMembership, TcpTransport, accept_authenticated,
    client_handshake,
};
use std::net::SocketAddr;
use tokio::net::{TcpListener, TcpStream};

/// A fresh random secret, as a security setting.
pub(crate) fn fresh_security() -> LinkSecurity {
    LinkSecurity::Authenticated(ClusterSecret::generate())
}

/// Bind `n` listeners on free loopback ports.
pub(crate) async fn bind_listeners(n: usize) -> (Vec<TcpListener>, Vec<SocketAddr>) {
    let mut listeners = Vec::with_capacity(n);
    let mut addrs = Vec::with_capacity(n);
    for _ in 0..n {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        addrs.push(listener.local_addr().expect("local_addr"));
        listeners.push(listener);
    }
    (listeners, addrs)
}

/// Form a ring of `n` ranks over loopback TCP, every rank using `security`,
/// and return the ranks' collectives in rank order.
pub(crate) async fn tcp_ring(n: usize, security: LinkSecurity) -> Vec<RingCollective> {
    let (listeners, addrs) = bind_listeners(n).await;
    let mut tasks = Vec::new();
    for (rank, listener) in listeners.into_iter().enumerate() {
        let mut config = DistributedConfig::new(addrs.clone(), rank);
        config.connection_timeout_ms = 20_000;
        config.max_retries = 20;
        let security = security.clone();
        tasks.push(tokio::spawn(async move {
            let membership = RingMembership::from_config(&config);
            let (sender, receiver) = TcpTransport::connect_ring(
                &config,
                security,
                membership,
                Some(listener),
                Channel::Ring,
            )
            .await
            .expect("ring formed");
            RingCollective::new(rank, config.nodes.len(), sender, receiver)
        }));
    }
    let mut ring = Vec::with_capacity(n);
    for task in tasks {
        ring.push(task.await.expect("ring task"));
    }
    ring
}

/// One authenticated connection over loopback: `(dialer's end, acceptor's end)`.
pub(crate) async fn tcp_link_pair(security: LinkSecurity) -> (Link, Link) {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let membership = |rank| RingMembership {
        rank,
        world_size: 2,
        digest: [7; 32],
    };
    let dial = LinkParams {
        security: security.clone(),
        channel: Channel::Ring,
        membership: membership(0),
        peer_rank: 1,
    };
    let accept = LinkParams {
        security,
        channel: Channel::Ring,
        membership: membership(1),
        peer_rank: 0,
    };
    let accepting = tokio::spawn(async move {
        accept_authenticated(&listener, &accept)
            .await
            .expect("accepted")
    });
    let stream = TcpStream::connect(addr).await.expect("connect");
    let dialer = client_handshake(stream, &dial).await.expect("handshake");
    (dialer, accepting.await.expect("accept task"))
}
