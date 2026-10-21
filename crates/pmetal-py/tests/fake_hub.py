"""A small stand-in for the Hugging Face Hub's HTTP API, served on localhost.

It answers what a model download asks: repository info (the commit), the file
tree, and the files, honouring ``Authorization``. A test points ``endpoint=`` at
it, so downloads, verification, errors and progress run for real without a
network.
"""

from __future__ import annotations

import hashlib
import http.server
import json
import re
import threading
from pathlib import Path


def files_of(directory: Path) -> list[Path]:
    """The files a repository serves: not the AppleDouble companions an external volume adds."""
    return [p for p in sorted(directory.rglob("*")) if p.is_file() and not p.name.startswith("._")]


class FakeHub:
    """Serves repositories from directories: ``repos[repo_id] = (commit, directory)``."""

    def __init__(self) -> None:
        self.repos: dict[str, tuple[str, Path]] = {}
        #: Repository ids that need this bearer token.
        self.private: dict[str, str] = {}
        #: Every request: (method, path, authorization header).
        self.requests: list[tuple[str, str, str | None]] = []
        #: Paths to answer with a server error, once each.
        self.fail_once: set[str] = set()
        #: A pause before each file body, seconds (to make a download last).
        self.delay = 0.0
        #: Files whose served bytes differ from what the tree lists (same length).
        self.corrupt: set[str] = set()
        #: What a model search answers with.
        self.search_results: list[dict[str, object]] = []
        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def __enter__(self) -> FakeHub:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()

    def add(self, repo_id: str, directory: Path, commit: str | None = None, token: str | None = None) -> str:
        commit = commit or hashlib.sha1(repo_id.encode()).hexdigest()
        self.repos[repo_id] = (commit, directory)
        if token:
            self.private[repo_id] = token
        return commit

    def _handler(self) -> type[http.server.BaseHTTPRequestHandler]:
        hub = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: object) -> None:
                pass

            def _send(
                self, status: int, body: bytes = b"", headers: dict[str, str] | None = None, head: bool = False
            ) -> None:
                self.send_response(status)
                for key, value in (headers or {}).items():
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if not head:
                    self.wfile.write(body)

            def _json(self, status: int, payload: object) -> None:
                self._send(status, json.dumps(payload).encode(), {"Content-Type": "application/json"})

            def _handle(self, head: bool) -> None:
                method = "HEAD" if head else "GET"
                path = self.path.split("?")[0]
                auth = self.headers.get("Authorization")
                hub.requests.append((method, path, auth))
                if path in hub.fail_once:
                    hub.fail_once.discard(path)
                    return self._send(503, b'{"error":"try again"}', head=head)

                if path == "/api/models":
                    return self._json(200, hub.search_results)

                match = re.match(r"^/api/models/(?P<id>[^/]+/[^/]+)(?:/revision/(?P<rev>[^/]+))?$", path)
                if match:
                    repo = self._repo(match["id"], auth)
                    if repo is None:
                        return
                    commit, directory = repo
                    files = [{"rfilename": str(p.relative_to(directory))} for p in files_of(directory)]
                    return self._json(
                        200,
                        {
                            "id": match["id"],
                            "sha": commit,
                            "siblings": files,
                            "tags": [],
                            "private": match["id"] in hub.private,
                        },
                    )

                match = re.match(r"^/api/models/(?P<id>[^/]+/[^/]+)/tree/(?P<rev>[^/]+)(?:/(?P<sub>.*))?$", path)
                if match:
                    repo = self._repo(match["id"], auth)
                    if repo is None:
                        return
                    _, directory = repo
                    entries = []
                    for p in files_of(directory):
                        data = p.read_bytes()
                        entry: dict[str, object] = {
                            "type": "file",
                            "oid": hashlib.sha1(data).hexdigest(),
                            "size": len(data),
                            "path": str(p.relative_to(directory)),
                        }
                        if p.suffix == ".safetensors":
                            entry["lfs"] = {
                                "oid": hashlib.sha256(data).hexdigest(),
                                "size": len(data),
                                "pointerSize": 134,
                            }
                        entries.append(entry)
                    return self._json(200, entries)

                match = re.match(r"^/(?P<id>[^/]+/[^/]+)/resolve/(?P<rev>[^/]+)/(?P<file>.+)$", path)
                if match:
                    repo = self._repo(match["id"], auth)
                    if repo is None:
                        return
                    commit, directory = repo
                    target = directory / match["file"]
                    if not target.is_file():
                        return self._send(404, b'{"error":"Entry not found"}', {"X-Error-Code": "EntryNotFound"}, head)
                    data = target.read_bytes()
                    etag = hashlib.sha1(data).hexdigest()
                    if match["file"] in hub.corrupt:
                        data = data[:-1] + bytes([data[-1] ^ 0xFF])
                    headers = {
                        "ETag": f'"{etag}"',
                        "X-Repo-Commit": commit,
                        "Content-Type": "application/octet-stream",
                        "Accept-Ranges": "bytes",
                    }
                    if hub.delay and not head:
                        import time

                        time.sleep(hub.delay)
                    rng = self.headers.get("Range")
                    if rng and (m := re.match(r"bytes=(\d+)-(\d*)", rng)) and not head:
                        start = int(m[1])
                        end = int(m[2]) if m[2] else len(data) - 1
                        headers["Content-Range"] = f"bytes {start}-{end}/{len(data)}"
                        return self._send(206, data[start : end + 1], headers)
                    return self._send(200, data, headers, head)

                self._send(404, b'{"error":"not found"}', head=head)

            def _repo(self, repo_id: str, auth: str | None) -> tuple[str, Path] | None:
                if repo_id not in hub.repos:
                    self._send(404, b'{"error":"Repository not found"}', {"X-Error-Code": "RepoNotFound"})
                    return None
                needed = hub.private.get(repo_id)
                if needed and auth != f"Bearer {needed}":
                    self._send(401, b'{"error":"Invalid credentials"}', {"X-Error-Code": "RepoNotFound"})
                    return None
                return hub.repos[repo_id]

            def do_GET(self) -> None:  # noqa: N802
                self._handle(head=False)

            def do_HEAD(self) -> None:  # noqa: N802
                self._handle(head=True)

        return Handler
