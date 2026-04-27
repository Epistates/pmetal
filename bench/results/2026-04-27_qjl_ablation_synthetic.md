# QJL ablation — synthetic Gaussian smoke run

Date: 2026-04-27
Harness: `cargo bench --bench turboquant_qjl_ablation -p pmetal-bridge --features tq-ablation`
Build: head + Phase 0 split (commit `fbe850d`) + Phase A harness

## What this is / isn't

This is the **first-order sanity check** that the ablation knob is wired
through every encode chokepoint and produces measurable score drift on
synthetic Gaussian K/V tensors. It is **not** the wikitext-2 perplexity
A/B sweep that gates Phase C's default-flip decision — that requires real
model weights and lives in `pmetal-models/benches/turboquant_qjl_ablation.rs`
(see TODO in the harness file).

Decision criterion for the real measurement:

| ΔPPL across all (model, bits, ctx) cells | Phase C action |
|---|---|
| < 0.5% | reproduces — flip Variant F to default |
| ≥ 0.5% on any cell | does not reproduce — ship Variant F as opt-in only |

## Synthetic results

Workload: head_dim ∈ {128, 256}, bits ∈ {2, 3, 4}, seq=1024, q_heads=kv_heads=4.
Compares attention output `[B, q_heads, 1, D]` between qjl_disabled=false and qjl_disabled=true.

| head | bits | seq  | mean \|Δ\| | max \|Δ\|  | cos(out)  |
|------|------|------|------------|------------|-----------|
| 128  | 4    | 1024 | 0.001825   | 0.008579   | 0.999995  |
| 128  | 3    | 1024 | 0.003030   | 0.012637   | 0.999987  |
| 128  | 2    | 1024 | 0.003840   | 0.024594   | 0.999974  |
| 256  | 4    | 1024 | 0.001355   | 0.009117   | 0.999997  |
| 256  | 3    | 1024 | 0.002362   | 0.012220   | 0.999992  |
| 256  | 2    | 1024 | 0.003368   | 0.018865   | 0.999980  |

## First-order read

Cosine similarity between QJL-on and QJL-off outputs is ≥ 0.99997 across
all bit-widths and head dims on synthetic Gaussian data. This is consistent
with the reference paper's "≈ 0 contribution" claim and supports the case
for Phase C default-flip — but synthetic Gaussians are an easy case
(no heavy tails, no structured residuals), so a real-model run is still
required.

The 2-bit cells show ~2× the score drift of 4-bit, which is expected:
lower-resolution codebooks leave larger residuals for QJL to correct, so
removing QJL has a larger relative effect. If the real-model run shows the
2-bit cells fail the 0.5% gate while 4-bit passes, Phase C should ship the
default flip *only* for 4-bit configs.

## Sanity check

Re-running without the `tq-ablation` feature flag produces Δ=0 across every
cell (the toggle is a no-op in non-feature builds), confirming the
ablation knob is load-bearing.

## Next step

Land `pmetal-models/benches/turboquant_qjl_ablation.rs` for the real
wikitext-2 sweep across Qwen3-4B, Llama4 dense, GptOss, and Qwen3.5-MoE.
That bench replaces the synthetic Gaussian forward pass with
`token_logprobs(forward(prompt, cache))` over a fixed wikitext-2 prefix at
contexts 4K and 16K.
