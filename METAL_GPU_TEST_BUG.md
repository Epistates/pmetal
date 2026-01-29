# Bug Report: Metal Command Buffer Race Condition in pmetal-distill Tests

**Date**: 2026-01-28
**Severity**: Low (workaround in place)
**Component**: `pmetal-distill` crate
**Platform**: macOS ARM64 (Apple Silicon)
**Status**: Mitigated

---

## Summary

GPU-accelerated tests in `pmetal-distill` crash with Metal command buffer assertion failures when run in parallel. This is due to Metal command buffer race conditions when multiple tests access GPU resources concurrently.

## Root Cause

Metal command buffers are not thread-safe for concurrent access. When multiple tests run in parallel:
1. Multiple tests attempt to use the same Metal command queue/buffer concurrently
2. Command buffers may be committed by one test while another also tries to commit
3. Completion handlers may be added after buffer commit

## Error Messages

```
-[IOGPUMetalCommandBuffer validate]:214: failed assertion `commit an already committed command buffer'
-[_MTLCommandBuffer commit]:690: failed assertion `commit an already committed command buffer'
-[_MTLCommandBuffer addCompletedHandler:]:1011: failed assertion `Completed handler provided after commit call'
```

## Mitigation

**The justfile already enforces single-threaded test execution:**

```just
# Run all tests (single-threaded for Metal GPU compatibility)
test:
    cargo test -- --test-threads=1
```

All standard development commands (`just test`, `just ci`, etc.) run tests with `--test-threads=1`.

## When This Matters

The bug only manifests when running tests with parallel execution:

```bash
# This crashes:
cargo test -p pmetal-distill --lib

# This works (and is what `just test` does):
cargo test -p pmetal-distill --lib -- --test-threads=1
```

## Additional Mitigations Applied

Added `serial_test` dependency and `#[serial]` attributes to GPU-intensive tests in:
- `losses/kl_divergence.rs`
- `losses/jensen_shannon.rs`
- `losses/hidden_state.rs`
- `losses/soft_cross_entropy.rs`

This provides defense-in-depth for the GPU tests specifically.

## Future Improvements

If parallel test execution becomes necessary:
1. Implement per-test Metal command queue isolation
2. Add a global GPU test mutex
3. Use `serial_test` crate more extensively

## Related

- Metal Best Practices: Command buffers should not be shared across threads without synchronization
- MLX uses Metal for GPU acceleration on Apple Silicon

---

**Resolution**: Using `just test` or `cargo test -- --test-threads=1` avoids the issue.
