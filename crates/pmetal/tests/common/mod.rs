//! HTTP helpers for the tests that drive a real `pmetal serve`.

#![allow(dead_code)]

use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// POST `body` to `path` on the server at `port`; the status and the body
/// (de-chunked).
pub async fn post(port: u16, path: &str, body: &Value) -> (u16, String) {
    let body = body.to_string();
    let mut stream = tokio::net::TcpStream::connect(("127.0.0.1", port))
        .await
        .unwrap();
    stream
        .write_all(
            format!(
                "POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n\
                 Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .as_bytes(),
        )
        .await
        .unwrap();
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).await.unwrap();
    let raw = String::from_utf8(raw).unwrap();
    let (head, mut rest) = raw.split_once("\r\n\r\n").unwrap();
    let status = head.split(' ').nth(1).unwrap().parse().unwrap();
    if !head
        .to_ascii_lowercase()
        .contains("transfer-encoding: chunked")
    {
        return (status, rest.to_owned());
    }
    let mut body = String::new();
    loop {
        let (size, tail) = rest.split_once("\r\n").unwrap();
        let size = usize::from_str_radix(size.trim(), 16).unwrap();
        if size == 0 {
            return (status, body);
        }
        body.push_str(&tail[..size]);
        rest = &tail[size + 2..];
    }
}

/// The JSON payloads of a server-sent event stream.
pub fn sse_data(body: &str) -> Vec<Value> {
    body.lines()
        .filter_map(|line| line.strip_prefix("data: ").or(line.strip_prefix("data:")))
        .filter_map(|data| serde_json::from_str(data).ok())
        .collect()
}

/// Wait until the server at `port` takes connections.
pub async fn wait_for(port: u16) {
    for _ in 0..100 {
        if tokio::net::TcpStream::connect(("127.0.0.1", port))
            .await
            .is_ok()
        {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
}
