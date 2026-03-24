# pmetal info

Show device information — GPU, ANE, bandwidth, and NAX capabilities.

Display detailed information about your Apple Silicon hardware.

## Usage

```bash
pmetal info
```

## Output

Shows:

- **GPU family** (Apple7–Apple10)
- **Device tier** (Base/Pro/Max/Ultra)
- **GPU core count**
- **ANE core count** and availability
- **NAX support** (M5+)
- **Memory bandwidth** (GB/s, with measured-vs-fallback source)
- **Metal features** (dynamic caching, mesh shaders)
- **UltraFusion topology** (Ultra chips)
- **UltraFusion local executor scaffold** (Ultra chips): per-die slices, local stage count, and heuristic interconnect bandwidth

## See Also

- [pmetal memory](/cli/memory/) — Memory usage details
- [Hardware Support](/hardware/apple-silicon/) — Full hardware matrix
