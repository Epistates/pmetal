# Security Policy

## Supported Versions

The latest release and the `main` branch receive security fixes. PMetal is pre-1.0, so older releases are not patched: update to the latest release.

## Reporting a Vulnerability

If you discover a security vulnerability within PMetal, please do not disclose it publicly. Instead, report it privately to the maintainers via email at `security@epistates.com` or through GitHub's private vulnerability reporting feature.

We take all security issues seriously and will respond to your initial report within 48 hours. The maintainers will work with you to verify the issue and prepare a fix before public disclosure.

## Security Practices

### Fuzzing
The GGUF decoders have fuzz targets (`gguf_reader`, `gguf_dequant` and `gguf_quantize`, under `fuzz/`) that must not panic on malformed input. CI runs `gguf_reader` with `cargo-fuzz` on every push and pull request to `main` and daily on a schedule; run the others locally with `cargo +nightly fuzz run <target>`.

### Error Handling
Library crates return typed errors (`thiserror`) instead of calling `.unwrap()` or `.expect()` on fallible operations, so a malformed file or request is an error the caller can handle, not a panic. Unsafe code is denied workspace-wide by lint and allowed only in the modules that need it for FFI (the Metal, Neural Engine and tensor runtime layers).

### Supply Chain
Every dependency is checked against `deny.toml` by `cargo-deny` in CI: licenses (permissive only, with a short list of named exceptions), RustSec advisories, banned crates and allowed sources, for the main workspace and the GUI. A scheduled job checks the lockfiles against the advisory database daily. The repository also carries a `cargo-vet` configuration under `supply-chain/`.

### Defaults That Limit Exposure
- `pmetal serve` listens on loopback unless you pass `--host`, can require API keys (`--api-key`, or `PMETAL_API_KEY`) and allows browser origins only when named (`--cors-origin`), and bounds request bodies and request time.
- Links between ranks of a multi-Mac cluster are authenticated and encrypted with a shared cluster secret (`pmetal cluster keygen`).
- PMetal never executes code from a model repository: `trust_remote_code` does not exist, and checkpoints published as PyTorch pickles are read with a restricted unpickler.
- The desktop app keeps the Hugging Face token in the macOS Keychain and grants its web view only the commands it uses. Jobs started from the app, the TUI or the MCP server run under a supervisor that redacts credentials from their logs.
