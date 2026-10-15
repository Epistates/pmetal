//! A fake Hugging Face Hub on loopback, for tests that must not touch the
//! network.
//!
//! It speaks the endpoints a download touches: repository info, the recursive
//! file tree, and `resolve` (HEAD and GET). Repositories are git-like: named
//! branches point at commits, a commit is a map from path to bytes. Every
//! request is recorded, and faults (a failing status for the next N requests,
//! a body cut short) and access rules (private, gated) can be injected.

#![allow(dead_code, unreachable_pub)]

pub mod fixtures;

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpListener;

pub fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// A 40-hex commit id derived from a label, so tests can name commits.
pub fn commit_id(label: &str) -> String {
    sha256_hex(label.as_bytes())[..40].to_owned()
}

#[derive(Debug, Clone)]
pub struct Recorded {
    pub method: String,
    pub path: String,
    pub query: String,
    pub authorization: Option<String>,
    pub body: Vec<u8>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Kind {
    Model,
    Dataset,
}

#[derive(Clone, Default)]
pub struct Commit {
    pub files: BTreeMap<String, Vec<u8>>,
}

/// Who may read a repository.
#[derive(Clone, Default)]
pub enum Access {
    #[default]
    Open,
    /// Private: every request needs this exact token (401 otherwise).
    Private(String),
    /// Gated: the token must be one of these, else 403.
    Gated(Vec<String>),
}

#[derive(Clone, Default)]
pub struct Repo {
    pub branches: BTreeMap<String, String>,
    pub commits: BTreeMap<String, Commit>,
    pub access: Access,
    /// Extra fields merged into the repository info response.
    pub info_extra: Value,
}

impl Repo {
    /// A repository whose `main` branch points at one commit holding `files`.
    pub fn with_files(files: &[(&str, &[u8])]) -> Self {
        let mut repo = Repo::default();
        repo.commit("main", "c1", files);
        repo
    }

    /// Point `branch` at a new commit (identified by `label`) holding `files`.
    pub fn commit(&mut self, branch: &str, label: &str, files: &[(&str, &[u8])]) -> String {
        let sha = commit_id(label);
        self.commits.insert(
            sha.clone(),
            Commit {
                files: files
                    .iter()
                    .map(|(p, b)| ((*p).to_owned(), b.to_vec()))
                    .collect(),
            },
        );
        self.branches.insert(branch.to_owned(), sha.clone());
        sha
    }

    pub fn head(&self, branch: &str) -> String {
        self.branches[branch].clone()
    }
}

struct Fault {
    path_contains: String,
    /// Only requests with this method, when set.
    method: Option<&'static str>,
    status: u16,
    remaining: usize,
}

type Hook = Box<dyn FnOnce(&mut HashMap<(Kind, String), Repo>) + Send>;

#[derive(Default)]
struct Inner {
    repos: HashMap<(Kind, String), Repo>,
    /// Run once, right after the first `resolve` GET whose path contains the
    /// key: a push that lands in the middle of a download.
    hooks: Vec<(String, Option<Hook>)>,
    log: Vec<Recorded>,
    faults: Vec<Fault>,
    /// `resolve` GETs whose path contains the first and ends with the second
    /// answer with half the body and then end the connection.
    truncate: Vec<(String, String)>,
}

#[derive(Clone)]
pub struct FakeHub {
    pub base: String,
    inner: Arc<Mutex<Inner>>,
}

impl FakeHub {
    pub async fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let inner = Arc::new(Mutex::new(Inner::default()));
        tokio::spawn(serve(listener, inner.clone()));
        Self { base, inner }
    }

    pub fn add(&self, kind: Kind, id: &str, repo: Repo) {
        self.inner
            .lock()
            .unwrap()
            .repos
            .insert((kind, id.to_owned()), repo);
    }

    pub fn add_model(&self, id: &str, repo: Repo) {
        self.add(Kind::Model, id, repo);
    }

    pub fn add_dataset(&self, id: &str, repo: Repo) {
        self.add(Kind::Dataset, id, repo);
    }

    /// Mutate a repository in place (push a commit while a test is running).
    pub fn edit(&self, kind: Kind, id: &str, f: impl FnOnce(&mut Repo)) {
        let mut inner = self.inner.lock().unwrap();
        f(inner.repos.get_mut(&(kind, id.to_owned())).unwrap());
    }

    /// Answer the next `times` requests whose path contains `path_contains`
    /// with `status`.
    pub fn fail(&self, path_contains: &str, status: u16, times: usize) {
        self.inner.lock().unwrap().faults.push(Fault {
            path_contains: path_contains.to_owned(),
            method: None,
            status,
            remaining: times,
        });
    }

    /// [`fail`](Self::fail) for `GET` requests only: the metadata `HEAD`
    /// succeeds and the transfer that follows does not.
    pub fn fail_get(&self, path_contains: &str, status: u16, times: usize) {
        self.inner.lock().unwrap().faults.push(Fault {
            path_contains: path_contains.to_owned(),
            method: Some("GET"),
            status,
            remaining: times,
        });
    }

    /// After the first download whose path contains `path_contains`, run `f`
    /// on the hub's repositories (for example, push a new commit).
    pub fn after_download(
        &self,
        path_contains: &str,
        f: impl FnOnce(&mut HashMap<(Kind, String), Repo>) + Send + 'static,
    ) {
        self.inner
            .lock()
            .unwrap()
            .hooks
            .push((path_contains.to_owned(), Some(Box::new(f))));
    }

    /// Serve only half the bytes of GETs of `file` in the repository `repo_id`
    /// (at any revision), then end the connection.
    pub fn truncate_downloads(&self, repo_id: &str, file: &str) {
        self.inner
            .lock()
            .unwrap()
            .truncate
            .push((format!("/{repo_id}/resolve/"), format!("/{file}")));
    }

    pub fn log(&self) -> Vec<Recorded> {
        self.inner.lock().unwrap().log.clone()
    }

    pub fn clear_log(&self) {
        self.inner.lock().unwrap().log.clear();
    }

    /// Recorded requests whose path contains `needle`.
    pub fn requests_to(&self, needle: &str) -> Vec<Recorded> {
        self.log()
            .into_iter()
            .filter(|r| r.path.contains(needle))
            .collect()
    }

    /// Paths of the files fetched with GET from `resolve` URLs, in order.
    pub fn downloaded(&self) -> Vec<String> {
        self.log()
            .into_iter()
            .filter(|r| r.method == "GET" && r.path.contains("/resolve/"))
            .map(|r| r.path)
            .collect()
    }
}

struct Response {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
    /// Declare this length even though `body` is shorter or empty.
    declared_len: Option<usize>,
    /// End the connection after the body (a dropped link).
    hang_up: bool,
    /// Never answer: hold the connection open in silence.
    stall: bool,
}

impl Response {
    fn json(status: u16, value: Value) -> Self {
        Self {
            status,
            headers: vec![("Content-Type".into(), "application/json".into())],
            body: serde_json::to_vec(&value).unwrap(),
            declared_len: None,
            hang_up: false,
            stall: false,
        }
    }

    fn error(status: u16, code: &str, message: &str) -> Self {
        let mut r = Self::json(status, json!({ "error": message }));
        r.headers.push(("X-Error-Code".into(), code.into()));
        r
    }
}

fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%'
            && let Some(v) = s
                .get(i + 1..i + 3)
                .and_then(|h| u8::from_str_radix(h, 16).ok())
        {
            out.push(v);
            i += 3;
            continue;
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn authorized(access: &Access, authorization: Option<&str>) -> Result<(), Box<Response>> {
    let bearer = authorization.and_then(|a| a.strip_prefix("Bearer "));
    match access {
        Access::Open => Ok(()),
        Access::Private(token) => {
            if bearer == Some(token.as_str()) {
                Ok(())
            } else {
                Err(Box::new(Response::error(
                    401,
                    "RepoNotFound",
                    "Invalid username or password.",
                )))
            }
        }
        Access::Gated(tokens) => match bearer {
            Some(t) if tokens.iter().any(|a| a == t) => Ok(()),
            _ => Err(Box::new(Response::error(
                403,
                "GatedRepo",
                "Access to this repository is restricted; accept the license to continue.",
            ))),
        },
    }
}

fn resolve_rev<'a>(repo: &'a Repo, rev: &str) -> Option<(String, &'a Commit)> {
    let sha = repo
        .branches
        .get(rev)
        .cloned()
        .or_else(|| repo.commits.contains_key(rev).then(|| rev.to_owned()))?;
    repo.commits.get(&sha).map(|c| (sha, c))
}

fn tree_entry(path: &str, bytes: &[u8]) -> Value {
    // Everything over 1 KiB is "LFS", like the real Hub's large files.
    let mut entry = json!({
        "type": "file",
        "oid": sha256_hex(bytes)[..40],
        "size": bytes.len(),
        "path": path,
    });
    if bytes.len() > 1024 {
        entry["lfs"] = json!({
            "oid": sha256_hex(bytes),
            "size": bytes.len(),
            "pointerSize": 134,
        });
    }
    entry
}

fn handle(
    inner: &mut Inner,
    method: &str,
    path: &str,
    query: &str,
    authorization: Option<&str>,
) -> Response {
    // Injected faults come first.
    for fault in &mut inner.faults {
        if fault.remaining > 0
            && path.contains(&fault.path_contains)
            && fault.method.is_none_or(|m| m == method)
        {
            fault.remaining -= 1;
            // Status 0 is a Hub that has stopped answering.
            if fault.status == 0 {
                let mut r = Response::error(503, "Injected", "stalled");
                r.stall = true;
                return r;
            }
            let mut r = Response::error(fault.status, "Injected", "injected fault");
            if fault.status == 429 {
                r.headers.push(("Retry-After".into(), "1".into()));
            }
            return r;
        }
    }
    let segments: Vec<String> = path
        .trim_start_matches('/')
        .split('/')
        .map(percent_decode)
        .collect();
    let seg: Vec<&str> = segments.iter().map(String::as_str).collect();

    // /api/models?search=…&limit=…&filter=…&pipeline_tag=…
    if seg == ["api", "models"] {
        let params: Vec<(String, String)> = query
            .split('&')
            .filter(|p| !p.is_empty())
            .map(|p| {
                let (k, v) = p.split_once('=').unwrap_or((p, ""));
                (percent_decode(k), percent_decode(v))
            })
            .collect();
        let get = |k: &str| params.iter().find(|(a, _)| a == k).map(|(_, v)| v.as_str());
        let filters: Vec<&str> = params
            .iter()
            .filter(|(k, _)| k == "filter")
            .map(|(_, v)| v.as_str())
            .collect();
        let search = get("search").unwrap_or("").to_lowercase();
        let limit: usize = get("limit").and_then(|l| l.parse().ok()).unwrap_or(1000);
        let mut ids: Vec<&(Kind, String)> = inner
            .repos
            .keys()
            .filter(|(k, _)| *k == Kind::Model)
            .collect();
        ids.sort();
        let mut out = Vec::new();
        for key in ids {
            let repo = &inner.repos[key];
            let mut entry = json!({ "id": key.1 });
            if let (Some(extra), Some(obj)) = (repo.info_extra.as_object(), entry.as_object_mut()) {
                obj.extend(extra.clone());
            }
            let tags: Vec<&str> = entry["tags"]
                .as_array()
                .map(|a| a.iter().filter_map(Value::as_str).collect())
                .unwrap_or_default();
            let matches = key.1.to_lowercase().contains(&search)
                && filters.iter().all(|f| tags.contains(f))
                && get("pipeline_tag").is_none_or(|p| entry["pipeline_tag"] == p)
                && get("author").is_none_or(|a| key.1.starts_with(&format!("{a}/")));
            if matches {
                out.push(entry);
            }
        }
        out.truncate(limit);
        return Response::json(200, Value::Array(out));
    }

    // /api/{models|datasets}/{owner}/{name}[/revision/{rev}|/tree/{rev}[/dir]]
    if seg.first() == Some(&"api") && seg.len() >= 4 {
        let kind = match seg[1] {
            "models" => Kind::Model,
            "datasets" => Kind::Dataset,
            _ => return Response::error(404, "NotFound", "unknown api route"),
        };
        let id = format!("{}/{}", seg[2], seg[3]);
        let Some(repo) = inner.repos.get(&(kind, id.clone())) else {
            return Response::error(404, "RepoNotFound", "Repository not found");
        };
        if let Err(r) = authorized(&repo.access, authorization) {
            return *r;
        }
        let (op, rev) = match seg.get(4).copied() {
            None => ("info", "main".to_owned()),
            Some("revision") => ("info", seg.get(5).copied().unwrap_or("main").to_owned()),
            Some("tree") => ("tree", seg.get(5).copied().unwrap_or("main").to_owned()),
            Some(_) => return Response::error(404, "NotFound", "unknown api route"),
        };
        let Some((sha, commit)) = resolve_rev(repo, &rev) else {
            return Response::error(404, "RevisionNotFound", "Revision not found");
        };
        return match op {
            "info" => {
                let siblings: Vec<Value> = commit
                    .files
                    .keys()
                    .map(|p| json!({ "rfilename": p }))
                    .collect();
                let mut info = json!({
                    "id": id,
                    "sha": sha,
                    "siblings": siblings,
                    "private": matches!(repo.access, Access::Private(_)),
                    "gated": matches!(repo.access, Access::Gated(_)),
                });
                if let (Some(extra), Some(obj)) =
                    (repo.info_extra.as_object(), info.as_object_mut())
                {
                    for (k, v) in extra {
                        obj.insert(k.clone(), v.clone());
                    }
                }
                Response::json(200, info)
            }
            _ => {
                let prefix = seg.get(6..).map(|s| s.join("/")).unwrap_or_default();
                let recursive = query.contains("recursive=true");
                let mut entries = Vec::new();
                let mut dirs = std::collections::BTreeSet::new();
                for (p, bytes) in &commit.files {
                    let rel = if prefix.is_empty() {
                        p.as_str()
                    } else if let Some(r) = p.strip_prefix(&format!("{prefix}/")) {
                        r
                    } else {
                        continue;
                    };
                    if !recursive && let Some((dir, _)) = rel.split_once('/') {
                        dirs.insert(if prefix.is_empty() {
                            dir.to_owned()
                        } else {
                            format!("{prefix}/{dir}")
                        });
                        continue;
                    }
                    entries.push(tree_entry(p, bytes));
                }
                for d in dirs {
                    entries.push(json!({ "type": "directory", "oid": "0".repeat(40), "path": d }));
                }
                Response::json(200, Value::Array(entries))
            }
        };
    }

    // /{owner}/{name}/resolve/{rev}/{file...}            (models)
    // /datasets/{owner}/{name}/resolve/{rev}/{file...}
    let (kind, rest) = if seg.first() == Some(&"datasets") {
        (Kind::Dataset, &seg[1..])
    } else {
        (Kind::Model, &seg[..])
    };
    if rest.len() >= 5 && rest[2] == "resolve" {
        let id = format!("{}/{}", rest[0], rest[1]);
        let rev = rest[3];
        let file = rest[4..].join("/");
        let Some(repo) = inner.repos.get(&(kind, id)) else {
            return Response::error(404, "RepoNotFound", "Repository not found");
        };
        if let Err(r) = authorized(&repo.access, authorization) {
            return *r;
        }
        let Some((sha, commit)) = resolve_rev(repo, rev) else {
            return Response::error(404, "RevisionNotFound", "Revision not found");
        };
        let Some(bytes) = commit.files.get(&file) else {
            let mut r = Response::error(404, "EntryNotFound", "Entry not found");
            r.headers.push(("X-Repo-Commit".into(), sha));
            return r;
        };
        let etag = sha256_hex(bytes);
        let mut headers = vec![
            ("ETag".to_owned(), format!("\"{etag}\"")),
            ("X-Repo-Commit".to_owned(), sha),
            ("X-Linked-Size".to_owned(), bytes.len().to_string()),
        ];
        if bytes.len() > 1024 {
            headers.push(("X-Linked-Etag".to_owned(), format!("\"{etag}\"")));
        }
        let truncated = method == "GET"
            && inner
                .truncate
                .iter()
                .any(|(repo, file)| path.contains(repo.as_str()) && path.ends_with(file.as_str()));
        let body = if method == "HEAD" {
            Vec::new()
        } else if truncated {
            bytes[..bytes.len() / 2].to_vec()
        } else {
            bytes.clone()
        };
        return Response {
            status: 200,
            headers,
            declared_len: (method == "HEAD" || truncated).then_some(bytes.len()),
            body,
            hang_up: truncated,
            stall: false,
        };
    }

    Response::error(404, "NotFound", "unexpected request")
}

async fn serve(listener: TcpListener, inner: Arc<Mutex<Inner>>) {
    loop {
        let Ok((stream, _)) = listener.accept().await else {
            return;
        };
        let inner = inner.clone();
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
                let (path, query) = target
                    .split_once('?')
                    .map_or((target.as_str(), ""), |(p, q)| (p, q));
                let (path, query) = (path.to_owned(), query.to_owned());

                let mut content_length = 0usize;
                let mut authorization = None;
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).await.unwrap_or(0) == 0 {
                        return;
                    }
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

                let response = {
                    let mut guard = inner.lock().unwrap();
                    guard.log.push(Recorded {
                        method: method.clone(),
                        path: path.clone(),
                        query: query.clone(),
                        authorization: authorization.clone(),
                        body,
                    });
                    let response =
                        handle(&mut guard, &method, &path, &query, authorization.as_deref());
                    if method == "GET" && path.contains("/resolve/") {
                        let Inner { repos, hooks, .. } = &mut *guard;
                        for (needle, hook) in hooks.iter_mut() {
                            if path.contains(needle.as_str())
                                && let Some(f) = hook.take()
                            {
                                f(repos);
                            }
                        }
                    }
                    response
                };

                let reason = match response.status {
                    200 => "OK",
                    401 => "Unauthorized",
                    403 => "Forbidden",
                    404 => "Not Found",
                    429 => "Too Many Requests",
                    503 => "Service Unavailable",
                    _ => "Status",
                };
                if response.stall {
                    tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
                    return;
                }
                let declared = response.declared_len.unwrap_or(response.body.len());
                let mut head = format!("HTTP/1.1 {} {reason}\r\n", response.status);
                for (k, v) in &response.headers {
                    head.push_str(&format!("{k}: {v}\r\n"));
                }
                head.push_str(&format!("Content-Length: {declared}\r\n\r\n"));
                if write.write_all(head.as_bytes()).await.is_err()
                    || write.write_all(&response.body).await.is_err()
                {
                    return;
                }
                if response.hang_up {
                    let _ = write.shutdown().await;
                    return;
                }
            }
        });
    }
}
