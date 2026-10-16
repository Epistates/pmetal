//! `chat` and `decide` against a server on this machine, and the commands that
//! run to completion.

mod common;

use std::time::{Duration, Instant};

use common::Harness;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
use tokio::net::TcpListener;

struct Received {
    request_line: String,
    body: String,
}

/// Serve one request on a free local port and report what it was.
async fn serve_once(
    status: &'static str,
    body: &'static str,
) -> (u16, tokio::task::JoinHandle<Received>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let handle = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let mut raw = Vec::new();
        let mut buf = [0u8; 4096];
        let (head_len, content_length) = loop {
            let n = stream.read(&mut buf).await.unwrap();
            raw.extend_from_slice(&buf[..n]);
            let text = String::from_utf8_lossy(&raw);
            if let Some(end) = text.find("\r\n\r\n") {
                let length = text[..end]
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().unwrap())
                    })
                    .unwrap_or(0);
                break (end + 4, length);
            }
        };
        while raw.len() < head_len + content_length {
            let n = stream.read(&mut buf).await.unwrap();
            raw.extend_from_slice(&buf[..n]);
        }
        let text = String::from_utf8(raw).unwrap();
        let response = format!(
            "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\
             connection: close\r\n\r\n{body}",
            body.len()
        );
        stream.write_all(response.as_bytes()).await.unwrap();
        Received {
            request_line: text.lines().next().unwrap().to_string(),
            body: text[head_len..].to_string(),
        }
    });
    (port, handle)
}

#[tokio::test]
async fn a_redirect_from_the_server_is_not_followed() {
    // Following it would send the request wherever the response says.
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let mut buf = [0u8; 4096];
        let _ = stream.read(&mut buf).await.unwrap();
        stream
            .write_all(
                b"HTTP/1.1 307 Temporary Redirect\r\nlocation: http://127.0.0.1:9/elsewhere\r\n\
                  content-length: 0\r\nconnection: close\r\n\r\n",
            )
            .await
            .unwrap();
    });
    let error = pmetal_mcp::serve_client::post_json(
        port.into(),
        "/x",
        &json!({}),
        &turbomcp::RequestContext::new(),
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("307"), "{error}");
    server.await.unwrap();
}

#[tokio::test]
async fn a_reply_larger_than_the_limit_is_refused() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let mut buf = [0u8; 4096];
        let _ = stream.read(&mut buf).await.unwrap();
        let _ = stream
            .write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 40000000\r\nconnection: close\r\n\r\n")
            .await;
        let chunk = vec![b'x'; 1 << 20];
        for _ in 0..40 {
            if stream.write_all(&chunk).await.is_err() {
                break;
            }
        }
    });
    let error = pmetal_mcp::serve_client::post_json(
        port.into(),
        "/x",
        &json!({}),
        &turbomcp::RequestContext::new(),
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("larger than"), "{error}");
    let _ = server.await;
}

const REQUEST: &str =
    r#"{"model":"clef-flash","state":"door open","questions":{"q":{"type":"noul"}}}"#;

#[tokio::test]
async fn chat_returns_the_reply_as_text_with_the_details_beside_it() {
    let reply = r#"{"model":"qwen","choices":[{"message":{"role":"assistant","content":"Hello there"},"finish_reason":"stop"}],"usage":{"total_tokens":9}}"#;
    let (port, server) = serve_once("200 OK", reply).await;
    let h = Harness::new("exit 0");
    let result = h
        .call(
            "chat",
            json!({"message": "hi", "port": port, "system": "be brief", "temperature": 0.2, "max_tokens": 16}),
        )
        .await;
    assert_ne!(result["isError"], json!(true), "{result}");
    assert_eq!(result["content"][0]["text"], "Hello there");
    assert_eq!(result["structuredContent"]["finish_reason"], "stop");
    assert_eq!(result["structuredContent"]["usage"]["total_tokens"], 9);

    let received = server.await.unwrap();
    assert_eq!(received.request_line, "POST /v1/chat/completions HTTP/1.1");
    let sent: Value = serde_json::from_str(&received.body).unwrap();
    assert_eq!(
        sent["messages"][0],
        json!({"role": "system", "content": "be brief"})
    );
    assert_eq!(
        sent["messages"][1],
        json!({"role": "user", "content": "hi"})
    );
    assert_eq!(sent["temperature"], 0.2);
    assert_eq!(sent["max_tokens"], 16);
}

#[tokio::test]
async fn chat_only_talks_to_ports_where_a_server_can_be() {
    let h = Harness::new("exit 0");
    for port in [22, 80, 1023, 70_000] {
        let result = h.call("chat", json!({"message": "hi", "port": port})).await;
        assert_eq!(result["isError"], json!(true), "{port}: {result}");
    }
    // Out-of-range temperatures are the schema's to refuse.
    let message = h
        .call_err("chat", json!({"message": "hi", "temperature": 9.0}))
        .await;
    assert!(message.contains("temperature"), "{message}");
}

#[tokio::test]
async fn decide_asks_a_running_server() {
    let answer = r#"{"model":"clef-flash","answers":{"q":{"yes":0.9,"no":0.1}}}"#;
    let (port, server) = serve_once("200 OK", answer).await;
    let h = Harness::new("exit 0");
    let result = h
        .call("decide", json!({"request": REQUEST, "port": port}))
        .await;
    assert_eq!(result["content"][0]["text"], answer);
    let received = server.await.unwrap();
    assert_eq!(received.request_line, "POST /v1/systemone HTTP/1.1");
    assert_eq!(
        serde_json::from_str::<Value>(&received.body).unwrap(),
        serde_json::from_str::<Value>(REQUEST).unwrap()
    );
}

#[tokio::test]
async fn decide_reports_the_server_refusal() {
    let (port, server) = serve_once(
        "400 Bad Request",
        r#"{"error":{"message":"q: criteria must not be empty"}}"#,
    )
    .await;
    let h = Harness::new("exit 0");
    let result = h
        .call("decide", json!({"request": REQUEST, "port": port}))
        .await;
    assert_eq!(result["isError"], json!(true));
    assert_eq!(result["_meta"]["io.turbomcp/errorKind"], "invalid_params");
    assert!(
        result["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("criteria must not be empty")
    );
    server.await.unwrap();
}

#[tokio::test]
async fn decide_needs_exactly_one_of_model_and_port() {
    let h = Harness::new("exit 0");
    for args in [
        json!({"request": REQUEST}),
        json!({"request": REQUEST, "model": "m", "port": 18999}),
        json!({"request": REQUEST, "max_length": 512, "port": 18999}),
        json!({"request": REQUEST, "port": 70000}),
        json!({"request": REQUEST, "port": 0}),
        json!({"request": "not json", "port": 18999}),
    ] {
        let result = h.call("decide", args.clone()).await;
        assert_eq!(result["isError"], json!(true), "{args}: {result}");
    }
}

#[tokio::test]
async fn decide_stages_the_request_privately_and_removes_it() {
    // The fake `decide` copies the request file it is given, to see it exists
    // while the command runs and that it is not readable by others.
    let h = Harness::new(
        r#"while [ $# -gt 0 ]; do [ "$1" = "--request" ] && f="$2"; shift; done
cat "$f"
ls -l "$f" | cut -c1-10 > "$(dirname "$0")/mode"
echo "$f" > "$(dirname "$0")/staged-path""#,
    );
    let result = h
        .call(
            "decide",
            json!({"request": REQUEST, "model": "Cloudflare/clef-flash"}),
        )
        .await;
    assert_ne!(result["isError"], json!(true), "{result}");
    assert_eq!(
        result["content"][0]["text"].as_str().unwrap().trim(),
        REQUEST
    );
    assert_eq!(
        std::fs::read_to_string(h.dir.join("mode")).unwrap().trim(),
        "-rw-------"
    );
    let staged = std::fs::read_to_string(h.dir.join("staged-path")).unwrap();
    assert!(!std::path::Path::new(staged.trim()).exists());
}

#[tokio::test]
async fn a_command_that_fails_returns_its_error_with_secrets_removed() {
    let h = Harness::new("echo 'denied for hf_AbCdEfGhIjKlMnOpQrStUvWx' >&2; exit 2");
    let message = h
        .call_err(
            "generate",
            json!({"model": "Qwen/Qwen3-0.6B", "prompt": "hi"}),
        )
        .await;
    assert!(message.contains("denied for hf_[REDACTED]"), "{message}");
    assert!(!message.contains("AbCdEf"), "{message}");
}

#[tokio::test]
async fn a_command_that_runs_to_completion_returns_what_it_printed() {
    let h = Harness::new(r#"echo "ran: $*""#);
    let result = h
        .call(
            "generate",
            json!({"model": "Qwen/Qwen3-0.6B", "prompt": "hi there", "max_tokens": 8}),
        )
        .await;
    let text = result["content"][0]["text"].as_str().unwrap();
    assert!(text.starts_with("ran: infer "), "{text}");
    assert!(
        text.contains("--max-tokens 8") && text.contains("hi there"),
        "{text}"
    );
}

#[tokio::test]
async fn a_cancelled_request_stops_the_command() {
    use std::sync::Arc;
    use tokio_util::sync::CancellationToken;
    use turbomcp::McpHandler as _;
    use turbomcp::RequestContext;

    let h = Harness::new("echo $$ > \"$(dirname \"$0\")/pid\"\nsleep 300");
    let token = CancellationToken::new();
    let ctx = RequestContext::stdio().with_cancellation_token(Arc::new(token.clone()));
    let server = h.server.clone();
    let args = json!({"model": "Qwen/Qwen3-0.6B", "prompt": "hi"});
    let call = tokio::spawn(async move { server.call_tool("generate", args, &ctx).await });

    let pid_file = h.dir.join("pid");
    for _ in 0..100 {
        if pid_file.exists()
            && !std::fs::read_to_string(&pid_file)
                .unwrap()
                .trim()
                .is_empty()
        {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let started = Instant::now();
    token.cancel();
    let result = call.await.unwrap().unwrap();
    assert_eq!(result.is_error, Some(true));
    assert!(started.elapsed() < Duration::from_secs(5));
    let pid = std::fs::read_to_string(&pid_file).unwrap();
    let alive = std::process::Command::new("kill")
        .args(["-0", pid.trim()])
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success());
    assert!(!alive, "the cancelled command is still running");
}

#[tokio::test]
async fn commands_that_run_to_completion_take_turns() {
    // Each may load a model, so two at once could exhaust memory.
    let limits = pmetal_mcp::jobs::JobLimits {
        max_blocking: 1,
        ..Default::default()
    };
    let h = Harness::with_limits(
        "d=$(dirname \"$0\")\necho start >> \"$d/order\"\nsleep 0.4\necho end >> \"$d/order\"",
        limits,
    );
    let args = json!({"model": "Qwen/Qwen3-0.6B", "prompt": "hi"});
    let (a, b) = tokio::join!(h.call("generate", args.clone()), h.call("generate", args));
    assert_ne!(a["isError"], json!(true), "{a}");
    assert_ne!(b["isError"], json!(true), "{b}");
    let order = std::fs::read_to_string(h.dir.join("order")).unwrap();
    assert_eq!(
        order.split_whitespace().collect::<Vec<_>>(),
        ["start", "end", "start", "end"]
    );
}

mod blocking {
    use std::sync::{Arc, Mutex};

    use pmetal_mcp::util::{pmetal_command, run_command};
    use turbomcp::__macro_support::turbomcp_core::{McpSession, SessionFuture};
    use turbomcp::RequestContext;

    use super::*;

    /// A session that records the notifications the server sends.
    #[derive(Debug, Default)]
    struct Recorder(Mutex<Vec<(String, Value)>>);

    impl McpSession for Recorder {
        fn call<'a>(&'a self, method: &'a str, _params: Value) -> SessionFuture<'a, Value> {
            Box::pin(async move {
                Err(turbomcp::McpError::capability_not_supported(
                    method.to_string(),
                ))
            })
        }
        fn notify<'a>(&'a self, method: &'a str, params: Value) -> SessionFuture<'a, ()> {
            Box::pin(async move {
                self.0.lock().unwrap().push((method.to_string(), params));
                Ok(())
            })
        }
    }

    fn command(h: &Harness) -> tokio::process::Command {
        pmetal_command(&h.dir.join("fake-pmetal"))
    }

    #[tokio::test]
    async fn a_long_command_sends_progress_to_a_client_that_asked() {
        let h = Harness::new("sleep 1.2; echo done");
        let recorder = Arc::new(Recorder::default());
        let ctx = RequestContext::stdio()
            .with_progress_token(json!("tok-1"))
            .with_session(recorder.clone());
        let out = run_command(
            command(&h),
            &["infer".to_string()],
            &ctx,
            Duration::from_secs(30),
            Duration::from_millis(200),
        )
        .await
        .unwrap();
        assert_eq!(out.trim(), "done");

        let sent = recorder.0.lock().unwrap();
        let progress: Vec<&Value> = sent
            .iter()
            .filter(|(method, _)| method == "notifications/progress")
            .map(|(_, params)| params)
            .collect();
        assert!(progress.len() >= 3, "{progress:?}");
        assert!(progress.iter().all(|p| p["progressToken"] == "tok-1"));
        // "The progress value MUST increase with each notification."
        let values: Vec<f64> = progress
            .iter()
            .map(|p| p["progress"].as_f64().unwrap())
            .collect();
        assert!(values.windows(2).all(|w| w[1] > w[0]), "{values:?}");
        assert!(progress[0]["message"].as_str().unwrap().contains("infer"));
    }

    #[tokio::test]
    async fn no_progress_is_sent_to_a_client_that_did_not_ask() {
        let h = Harness::new("sleep 0.6");
        let recorder = Arc::new(Recorder::default());
        let ctx = RequestContext::stdio().with_session(recorder.clone());
        run_command(
            command(&h),
            &["infer".to_string()],
            &ctx,
            Duration::from_secs(30),
            Duration::from_millis(100),
        )
        .await
        .unwrap();
        assert!(recorder.0.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn a_command_past_its_time_limit_is_stopped() {
        let h = Harness::new("echo $$ > \"$(dirname \"$0\")/pid\"\nsleep 300");
        let started = Instant::now();
        let error = run_command(
            command(&h),
            &["infer".to_string()],
            &RequestContext::new(),
            Duration::from_millis(600),
            Duration::from_secs(5),
        )
        .await
        .unwrap_err();
        assert!(started.elapsed() < Duration::from_secs(10));
        assert!(
            error.to_string().contains("did not finish in 0s")
                || error.to_string().contains("was stopped"),
            "{error}"
        );
        let pid = std::fs::read_to_string(h.dir.join("pid")).unwrap();
        let alive = std::process::Command::new("kill")
            .args(["-0", pid.trim()])
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|s| s.success());
        assert!(!alive, "the command outlived its limit");
    }

    #[tokio::test]
    async fn output_is_bounded() {
        // 20 MiB of standard output; the result keeps the first 8.
        let h = Harness::new("head -c 20971520 /dev/zero | tr '\\0' y");
        let out = run_command(
            command(&h),
            &["infer".to_string()],
            &RequestContext::new(),
            Duration::from_secs(60),
            Duration::from_secs(5),
        )
        .await
        .unwrap();
        assert!(out.len() < 9 * 1024 * 1024, "{}", out.len());
        assert!(out.contains("more bytes of output dropped"));
    }
}
