# Security Policy

## Supported Versions

Currently, the `main` branch and all tagged releases are supported with security updates. We recommend keeping your deployment on the latest stable release.

## Reporting a Vulnerability

If you discover a security vulnerability within PMetal, please do not disclose it publicly. Instead, report it privately to the maintainers via email at `security@epistates.com` or through GitHub's private vulnerability reporting feature.

We take all security issues seriously and will respond to your initial report within 48 hours. The maintainers will work with you to verify the issue and prepare a fix before public disclosure.

## Security Practices

PMetal follows "Google-Grade" security and robustness standards:

### Continuous Fuzzing
All parser endpoints, including the GGUF reader, are subjected to continuous fuzz testing via `cargo-fuzz`. This ensures that malformed inputs cannot cause panics, memory leaks, or execution of arbitrary code. Continuous fuzzing runs automatically on every Pull Request and via scheduled GitHub Actions.

### Explicit Error Handling
The use of `.unwrap()` and `.expect()` is strictly prohibited in library code (`pmetal-*` crates). All fallible operations must return a robust `Result` type, appropriately categorized using the `thiserror` crate. Panics are only considered acceptable in the CLI binary if a catastrophic or unrecoverable environmental state is encountered.

### Supply Chain Security
Dependencies are regularly audited for known vulnerabilities using `cargo-audit`. We follow a policy of minimal dependencies, particularly avoiding dependencies with known security histories or poor maintenance records.
