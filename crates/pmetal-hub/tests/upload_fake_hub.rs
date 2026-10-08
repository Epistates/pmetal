//! `upload_model` against a fake Hub on loopback.
//!
//! The fake speaks the three endpoints a small-file upload touches (repo
//! create, preupload, commit), records every request, and the test checks
//! what reached it: the repository settings, the files that were offered and
//! committed (and the ones that must never be), their bytes, the commit
//! message, and the token on every request.

#![allow(unsafe_code)] // `std::env::set_var` (edition 2024)

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use base64::Engine as _;
use pmetal_hub::{UploadOptions, upload_model};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpListener;

#[derive(Debug, Clone)]
struct Recorded {
    method: String,
    path: String,
    authorization: Option<String>,
    body: Vec<u8>,
}

/// Serve requests until the test ends, answering the upload endpoints.
async fn serve(listener: TcpListener, base: String, log: Arc<Mutex<Vec<Recorded>>>) {
    loop {
        let Ok((stream, _)) = listener.accept().await else {
            return;
        };
        let base = base.clone();
        let log = log.clone();
        tokio::spawn(async move {
            let (read, mut write) = stream.into_split();
            let mut reader = BufReader::new(read);
            loop {
                let mut request_line = String::new();
                if reader.read_line(&mut request_line).await.unwrap_or(0) == 0 {
                    return;
                }
                let mut parts = request_line.split_whitespace();
                let method = parts.next().unwrap_or_default().to_owned();
                let target = parts.next().unwrap_or_default().to_owned();
                let path = target.split('?').next().unwrap_or_default().to_owned();

                let mut content_length = 0usize;
                let mut authorization = None;
                loop {
                    let mut line = String::new();
                    reader.read_line(&mut line).await.unwrap();
                    let line = line.trim_end();
                    if line.is_empty() {
                        break;
                    }
                    let (name, value) = line.split_once(':').unwrap();
                    match name.to_ascii_lowercase().as_str() {
                        "content-length" => content_length = value.trim().parse().unwrap(),
                        "authorization" => authorization = Some(value.trim().to_owned()),
                        _ => {}
                    }
                }
                let mut body = vec![0u8; content_length];
                reader.read_exact(&mut body).await.unwrap();

                let (status, reply) = match (method.as_str(), path.as_str()) {
                    ("POST", "/api/repos/create") => {
                        ("200 OK", json!({ "url": format!("{base}/me/tiny") }))
                    }
                    ("POST", "/api/models/me/tiny/preupload/main") => {
                        let request: Value = serde_json::from_slice(&body).unwrap();
                        let files: Vec<Value> = request["files"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|f| json!({ "path": f["path"], "uploadMode": "regular" }))
                            .collect();
                        ("200 OK", json!({ "files": files }))
                    }
                    ("POST", "/api/models/me/tiny/commit/main") => (
                        "200 OK",
                        json!({
                            "commitUrl": format!("{base}/me/tiny/commit/c0ffee"),
                            "commitOid": "c0ffee",
                        }),
                    ),
                    _ => ("404 Not Found", json!({ "error": "unexpected request" })),
                };
                log.lock().unwrap().push(Recorded {
                    method,
                    path,
                    authorization,
                    body,
                });

                let reply = serde_json::to_vec(&reply).unwrap();
                let head = format!(
                    "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
                    reply.len()
                );
                if write.write_all(head.as_bytes()).await.is_err()
                    || write.write_all(&reply).await.is_err()
                {
                    return;
                }
            }
        });
    }
}

fn write(dir: &std::path::Path, rel: &str, bytes: &[u8]) {
    let path = dir.join(rel);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, bytes).unwrap();
}

#[tokio::test]
async fn upload_creates_the_repo_and_commits_only_the_model_files() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let log = Arc::new(Mutex::new(Vec::new()));
    tokio::spawn(serve(listener, base.clone(), log.clone()));

    // No other test in this binary reads the environment.
    unsafe {
        std::env::set_var("HF_ENDPOINT", &base);
        std::env::set_var("HF_TOKEN", "hf_fake_write_token");
    }

    let model = tempfile::tempdir().unwrap();
    let dir = model.path();
    let config = br#"{"model_type":"qwen3"}"#;
    let tokenizer = br#"{"version":"1.0"}"#;
    let adapter = [7u8, 0, 255, 3, 9, 1];
    write(dir, "config.json", config);
    write(dir, "tokenizer.json", tokenizer);
    write(dir, "adapter/adapter_model.safetensors", &adapter);
    // Never published: Finder metadata, an AppleDouble twin, a stray .git,
    // and a directory the caller excludes.
    write(dir, ".DS_Store", b"finder");
    write(dir, "._config.json", b"appledouble");
    write(dir, "adapter/._adapter_model.safetensors", b"appledouble");
    write(dir, ".git/HEAD", b"ref: refs/heads/main");
    write(
        dir,
        "checkpoints/step-10/adapter_model.safetensors",
        b"stale",
    );

    let options = UploadOptions {
        private: true,
        commit_message: Some("Publish the tiny adapter".into()),
        ignore: vec!["checkpoints/**".into()],
        ..Default::default()
    };
    let report = upload_model(dir, "me/tiny", None, &options).await.unwrap();

    assert_eq!(report.repo_url, format!("{base}/me/tiny"));
    assert_eq!(
        report.commit_url.as_deref(),
        Some(format!("{base}/me/tiny/commit/c0ffee").as_str())
    );

    let log = log.lock().unwrap().clone();
    let paths: Vec<&str> = log.iter().map(|r| r.path.as_str()).collect();
    assert_eq!(
        paths,
        [
            "/api/repos/create",
            "/api/models/me/tiny/preupload/main",
            "/api/models/me/tiny/commit/main",
        ],
        "unexpected request sequence: {log:?}"
    );
    for request in &log {
        assert_eq!(request.method, "POST");
        assert_eq!(
            request.authorization.as_deref(),
            Some("Bearer hf_fake_write_token"),
            "{} was not authenticated",
            request.path
        );
    }

    let create: Value = serde_json::from_slice(&log[0].body).unwrap();
    assert_eq!(create["name"], "tiny");
    assert_eq!(create["organization"], "me");
    assert_eq!(create["private"], true);
    assert_eq!(create["type"], "model");

    let preupload: Value = serde_json::from_slice(&log[1].body).unwrap();
    let mut offered: Vec<&str> = preupload["files"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| f["path"].as_str().unwrap())
        .collect();
    offered.sort_unstable();
    assert_eq!(
        offered,
        [
            "adapter/adapter_model.safetensors",
            "config.json",
            "tokenizer.json",
        ]
    );

    let lines: Vec<Value> = log[2]
        .body
        .split(|b| *b == b'\n')
        .filter(|l| !l.is_empty())
        .map(|l| serde_json::from_slice(l).unwrap())
        .collect();
    assert_eq!(lines[0]["key"], "header");
    assert_eq!(lines[0]["value"]["summary"], "Publish the tiny adapter");
    let committed: BTreeMap<&str, Vec<u8>> = lines[1..]
        .iter()
        .map(|line| {
            assert_eq!(line["key"], "file");
            let value = &line["value"];
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(value["content"].as_str().unwrap())
                .unwrap();
            (value["path"].as_str().unwrap(), bytes)
        })
        .collect();
    assert_eq!(
        committed,
        BTreeMap::from([
            ("adapter/adapter_model.safetensors", adapter.to_vec()),
            ("config.json", config.to_vec()),
            ("tokenizer.json", tokenizer.to_vec()),
        ])
    );
}

#[tokio::test]
async fn upload_refuses_a_repo_id_without_an_owner() {
    let model = tempfile::tempdir().unwrap();
    let err = upload_model(model.path(), "tiny", None, &UploadOptions::default())
        .await
        .unwrap_err();
    assert!(err.to_string().contains("owner/name"), "{err}");
}
