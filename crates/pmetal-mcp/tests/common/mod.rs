//! Shared test machinery: a server driven through its JSON-RPC entry point, and
//! a stand-in for the `pmetal` binary.

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use pmetal_mcp::PmetalMcpServer;
use pmetal_mcp::jobs::{JobLimits, JobManager};
use pmetal_mcp::policy::PathPolicy;
use serde_json::{Value, json};
use turbomcp::McpHandlerExt as _;
use turbomcp::RequestContext;

/// A fresh directory that is removed when this is dropped. It derefs to the
/// canonical path, so it compares equal to what the policy returns.
pub struct Scratch {
    path: PathBuf,
    _guard: tempfile::TempDir,
}

impl std::ops::Deref for Scratch {
    type Target = Path;

    fn deref(&self) -> &Path {
        &self.path
    }
}

impl AsRef<Path> for Scratch {
    fn as_ref(&self) -> &Path {
        &self.path
    }
}

pub fn scratch_dir(name: &str) -> Scratch {
    let guard = tempfile::Builder::new()
        .prefix(&format!("pmetal-mcp-test-{name}-"))
        .tempdir()
        .unwrap();
    let path = guard.path().canonicalize().unwrap();
    Scratch {
        path,
        _guard: guard,
    }
}

/// A script that stands in for `pmetal`: `body` runs with the subcommand and its
/// arguments as `$@`.
pub fn fake_pmetal(dir: &Path, body: &str) -> PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let path = dir.join("fake-pmetal");
    std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    path
}

pub struct Harness {
    pub server: PmetalMcpServer,
    /// The directory the policy allows.
    pub dir: PathBuf,
    next_id: AtomicU64,
    /// The directories above, removed when the harness drops (after the server).
    _scratch: Vec<Scratch>,
}

impl Harness {
    /// A server whose `pmetal` is the script `body`, allowed to use only a new
    /// directory, with default job limits.
    pub fn new(body: &str) -> Self {
        Self::with_limits(body, JobLimits::default())
    }

    pub fn with_limits(body: &str, limits: JobLimits) -> Self {
        let scratch = scratch_dir("harness");
        let dir = scratch.to_path_buf();
        let policy = PathPolicy::new(dir.clone(), vec![dir]);
        Self::build(body, limits, vec![scratch], policy)
    }

    /// As `new`, but the server's working directory is not an allowed root, as
    /// when a launcher starts it in `/`.
    pub fn with_cwd_outside_the_roots(body: &str) -> Self {
        let scratch = scratch_dir("harness");
        let cwd = scratch_dir("cwd");
        let policy = PathPolicy::new(cwd.to_path_buf(), vec![scratch.to_path_buf()]);
        Self::build(body, JobLimits::default(), vec![scratch, cwd], policy)
    }

    /// `scratch[0]` is the directory the policy allows.
    fn build(body: &str, limits: JobLimits, scratch: Vec<Scratch>, policy: PathPolicy) -> Self {
        let dir = scratch[0].to_path_buf();
        let binary = fake_pmetal(&dir, body);
        Self {
            server: PmetalMcpServer::with(JobManager::with_binary(binary, limits), policy),
            dir,
            next_id: AtomicU64::new(1),
            _scratch: scratch,
        }
    }

    /// The whole JSON-RPC response to a request.
    pub async fn request(&self, method: &str, params: Value) -> Value {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        self.server
            .handle_request(
                json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}),
                RequestContext::stdio(),
            )
            .await
            .expect("the request is well-formed JSON-RPC")
    }

    /// `result` of a request that succeeded.
    pub async fn result(&self, method: &str, params: Value) -> Value {
        let response = self.request(method, params).await;
        assert!(
            response.get("error").is_none(),
            "{method} failed at the protocol level: {response}"
        );
        response["result"].clone()
    }

    /// The `tools/call` result, which may be an `isError` result.
    pub async fn call(&self, tool: &str, args: Value) -> Value {
        self.result("tools/call", json!({"name": tool, "arguments": args}))
            .await
    }

    /// A call that must succeed; its structured content.
    pub async fn call_ok(&self, tool: &str, args: Value) -> Value {
        let result = self.call(tool, args).await;
        assert!(
            !result["isError"].as_bool().unwrap_or(false),
            "{tool} failed: {result}"
        );
        result["structuredContent"].clone()
    }

    /// A call that must come back as a tool error; its message.
    pub async fn call_err(&self, tool: &str, args: Value) -> String {
        let result = self.call(tool, args).await;
        assert_eq!(
            result["isError"],
            json!(true),
            "{tool} should have failed: {result}"
        );
        result["content"][0]["text"]
            .as_str()
            .unwrap_or("")
            .to_string()
    }

    /// Poll `job_status` until `done` holds of its result.
    pub async fn wait_for_job(&self, job_id: &str, done: impl Fn(&Value) -> bool) -> Value {
        for _ in 0..200 {
            let status = self.call_ok("job_status", json!({"job_id": job_id})).await;
            if done(&status) {
                return status;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        panic!("job {job_id} never reached the expected state");
    }

    /// Path inside the allowed directory.
    pub fn path(&self, name: &str) -> String {
        self.dir.join(name).display().to_string()
    }
}

/// A minimal valid `train` call.
pub fn train_args(h: &Harness) -> Value {
    json!({
        "model": "Qwen/Qwen3-0.6B",
        "dataset": h.path("data.jsonl"),
        "output_dir": h.path("out"),
    })
}
