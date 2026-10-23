#!/usr/bin/env python3
"""Check every pinned dependency for malware, advisories and its age.

Reads the three Cargo lockfiles and the GUI's bun.lock and reports:

  malware     any OSV record (MAL-*) for a pinned version, in any ecosystem
  npm         any other OSV advisory for a pinned npm version; the Rust
              lockfiles' advisories are `just deny`'s job, which knows which
              crates the macOS build actually compiles
  age         versions published less than --min-age-days ago (default 7):
              most malicious releases are caught and pulled within days, so a
              fresh version is the riskiest one to have pinned
  scripts     npm packages that declare install scripts (bun runs none of them,
              as package.json trusts no dependency, but a new one is worth a look)

Exits non-zero on malware or an npm advisory, and on young versions too with
--fail-on-young. Needs only the standard library and network access to
api.osv.dev and api.deps.dev.
"""

import argparse
import concurrent.futures
import json
import re
import sys
import time
import tomllib
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CARGO_LOCKS = ["Cargo.lock", "crates/pmetal-gui/src-tauri/Cargo.lock", "fuzz/Cargo.lock"]
BUN_LOCK = "crates/pmetal-gui/bun.lock"
HEADERS = {"User-Agent": "pmetal-audit-lockfiles", "Content-Type": "application/json"}


def fetch(url, body=None):
    request = urllib.request.Request(url, data=body, headers=HEADERS)
    for attempt in range(6):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read())
        except Exception:
            if attempt == 5:
                raise
            time.sleep(1 + attempt)


def pinned():
    """(ecosystem, name, version) for every registry package in the lockfiles."""
    packages = set()
    for lock in CARGO_LOCKS:
        for package in tomllib.loads((ROOT / lock).read_text()).get("package", []):
            if package.get("source", "").startswith("registry+"):
                packages.add(("crates.io", package["name"], package["version"]))
    scripts = []
    # bun.lock is JSON with trailing commas.
    text = re.sub(r",(\s*[}\]])", r"\1", (ROOT / BUN_LOCK).read_text())
    for entry in json.loads(text)["packages"].values():
        name, _, version = entry[0].rpartition("@")
        packages.add(("npm", name, version))
        meta = entry[2] if len(entry) > 2 and isinstance(entry[2], dict) else {}
        if meta.get("scripts") or meta.get("hasInstallScript"):
            scripts.append(entry[0])
    return sorted(packages), sorted(scripts)


def advisories(packages):
    found = []
    for start in range(0, len(packages), 500):
        chunk = packages[start : start + 500]
        queries = [
            {"package": {"ecosystem": eco, "name": name}, "version": version}
            for eco, name, version in chunk
        ]
        body = json.dumps({"queries": queries}).encode()
        results = fetch("https://api.osv.dev/v1/querybatch", body)["results"]
        for package, result in zip(chunk, results):
            found += [(*package, vuln["id"]) for vuln in result.get("vulns", [])]
    return found


def published(package):
    eco, name, version = package
    system = {"crates.io": "cargo", "npm": "npm"}[eco]
    url = "https://api.deps.dev/v3/systems/{}/packages/{}/versions/{}".format(
        system, urllib.parse.quote(name, safe=""), urllib.parse.quote(version, safe="")
    )
    try:
        stamp = fetch(url).get("publishedAt")
    except Exception:
        stamp = None
    return package, stamp and datetime.fromisoformat(stamp.replace("Z", "+00:00"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--min-age-days", type=int, default=7)
    parser.add_argument("--fail-on-young", action="store_true")
    args = parser.parse_args()

    packages, scripts = pinned()
    npm = sum(1 for p in packages if p[0] == "npm")
    print(f"{len(packages)} pinned versions ({len(packages) - npm} crates, {npm} npm)")

    found = advisories(packages)
    malware = [f for f in found if f[3].startswith("MAL-")]
    npm_advisories = [f for f in found if f[0] == "npm" and not f[3].startswith("MAL-")]
    crate_advisories = [f for f in found if f[0] == "crates.io" and not f[3].startswith("MAL-")]
    print(f"\nmalware records: {len(malware)}")
    for f in malware:
        print("   ", *f)
    print(f"npm advisories: {len(npm_advisories)}")
    for f in npm_advisories:
        print("   ", *f)
    print(f"crate advisories (any platform; `just deny` decides): {len(crate_advisories)}")
    for f in crate_advisories:
        print("   ", *f)

    cutoff = datetime.now(timezone.utc) - timedelta(days=args.min_age_days)
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        stamps = list(pool.map(published, packages))
    young = sorted((t, p) for p, t in stamps if t and t > cutoff)
    unknown = [p for p, t in stamps if t is None]
    print(f"\nyounger than {args.min_age_days} days: {len(young)}")
    for stamp, (eco, name, version) in young:
        print(f"    {stamp:%Y-%m-%d %H:%M}  {eco:9} {name} {version}")
    if unknown:
        print(f"publish time unknown: {len(unknown)}")
        for package in unknown:
            print("   ", *package)

    print(f"\nnpm packages with install scripts: {len(scripts)}")
    for spec in scripts:
        print("   ", spec)

    failed = bool(malware or npm_advisories or (args.fail_on_young and young))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
