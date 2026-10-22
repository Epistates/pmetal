# pmetal-runtime

PMetal's tensor runtime for Apple Silicon: arrays, ops, autodiff, compiled
graphs, native inference engines and custom Metal kernels.

## Overview

Every PMetal crate computes with the array type defined here. `InlineArray` is
a stack-allocated handle to a lazy compute graph (MLX underneath, reached
through `extern "C"` calls into this crate's own C++), with no per-op heap
allocation. MLX is an implementation detail behind the Rust API: it is kept
where it is the best choice and replaced by PMetal's own Metal kernels where
they are faster or more correct, and it never appears in the signatures of the
crates above this one.

This is the bottom of the workspace. `pmetal-cache`, `pmetal-models`,
`pmetal-lora`, `pmetal-trainer`, `pmetal-data`, `pmetal-distill`,
`pmetal-distributed`, `pmetal-merge`, `pmetal-vocoder`, `pmetal-serve`, the
`pmetal` CLI and the Python bindings all depend on it; it depends on no other
workspace crate. It is an unsafe-FFI crate by nature (`unsafe_code` is allowed
here and denied everywhere else in the workspace).

## Features

- **Arrays and ops**: `InlineArray` with the full op set (elementwise,
  reductions, shape ops, indexing, linear algebra, random, fast attention,
  normalisation and RoPE kernels), safetensors I/O and zero-copy interop
- **Autodiff**: `value_and_grad` and `checkpoint_apply`, and `compat::checkpointed`,
  real gradient checkpointing that drops a layer's activations and recomputes them
- **Compiled graphs**: `CompiledFn` traces a Rust closure once and replays the
  fused graph on every call
- **Model-building API** (`compat`): `Array`, `Dtype`, `Module` and
  `ModuleParameters`, `Linear` (with a LoRA adapter and a packed `LinearQuant`
  as properties of the layer), embeddings, norms and activations, and the
  optimizers (`AdamW`, `Sgd`, `MomentumSgd`, `Lion`, `Adafactor`, `Muon`)
- **Operations a model is built from** (`nn`): scaled dot-product attention with
  masks, sinks and softcapping, the gated delta rule, the cut cross-entropy
  loss, rotary embeddings at explicit positions and the mixture-of-experts combine
- **Native inference engines**: standalone Qwen3 / Qwen3.5 (dense and MoE),
  Gemma 4, DeepSeek V2/V3, GPT-OSS and Llama 4 engines with compiled decode,
  packed-weight (MLX affine, NVFP4/FP8 ModelOpt) loading and KV caches
- **Decoding**: `DecodeSampler` and `SamplingParams` (temperature, top-k,
  top-p, min-p, repetition, frequency and presence penalties),
  `LogitsProcessor` for constrained decoding
- **TurboQuant KV cache**: rotation-based KV quantization with attention
  computed straight from the packed cache
- **MLX-format quantization** (`mlx_quant`) with per-tensor sensitivity analysis
- **Distributed** (`distributed`): process groups and collectives over the ring
  and Thunderbolt backends
- **Error channel**: a C++ exception in an op is recorded, not fatal;
  `check_unobserved_error` and the `try_*` ops surface it as `RuntimeError`

## Installation

```toml
[dependencies]
pmetal-runtime = "0.7"
```

The first build compiles MLX (see [Building](#building)).

## Usage

```rust
use pmetal_runtime::{InlineArray, check_unobserved_error};
use pmetal_runtime::inline_array::value_and_grad;

// Ops build a lazy graph; reading a value evaluates it.
let w = InlineArray::from_f32_slice(&[1.0, 2.0, 3.0], &[3]);
let x = InlineArray::from_f32_slice(&[0.5, 0.5, 2.0], &[3]);

// loss = sum(w * x), so d loss / d w = x. The loss function sees the
// parameters first, then the inputs.
let (mut loss, mut grads) =
    value_and_grad(|a| a[0].multiply(&a[1]).sum(None), &[w], &[x]);
assert_eq!(loss.to_f32_vec(1).unwrap(), [7.5]);
assert_eq!(grads[0].to_f32_vec(3).unwrap(), [0.5, 0.5, 2.0]);

// An op that failed in C++ left an error behind; nothing did here.
check_unobserved_error().unwrap();
```

A closure traced once and replayed:

```rust
use pmetal_runtime::{CompiledFn, InlineArray};

// SwiGLU, compiled: silu(gate) * up.
let mut swiglu = CompiledFn::new(1, true, |inputs| {
    let (gate, up) = (&inputs[0], &inputs[1]);
    vec![gate.multiply(&gate.sigmoid()).multiply(up)]
})
.unwrap();

let gate = InlineArray::from_f32_slice(&[0.0, 1.0], &[2]);
let up = InlineArray::from_f32_slice(&[2.0, 2.0], &[2]);
let mut out = swiglu.call(&[&gate, &up]).unwrap();
let values = out[0].to_f32_vec(2).unwrap();
assert_eq!(values[0], 0.0);
assert!((values[1] - 1.4621172).abs() < 1e-5);
```

## Modules

| Module | Description |
|--------|-------------|
| `inline_array` | `InlineArray`, its ops, autodiff, safetensors I/O, `MetalKernel`, allocator controls |
| `compat` | The model-building API: `Array`, `Dtype`, `Module`, layers, LoRA, optimizers, checkpointing |
| `nn` | Attention, gated delta rule, cut cross-entropy, rotary embeddings, MoE combine |
| `rope` | Rotary frequencies for every `rope_type` and the one rotation that applies them |
| `compile` | `CompiledFn`, a Rust closure compiled into one fused graph |
| `decode` | Samplers, penalties, prefill and the generation loop |
| `logits_processor` | `LogitsProcessor`, `LogitBias`, `AllowedTokens` |
| `turboquant`, `turboquant_dispatch` | The TurboQuant KV cache and its attention dispatch |
| `mlx_quant` | MLX-format weight quantization |
| `native_checkpoint`, `native_loader`, `native_weight`, `native_moe`, `native_common` | What the native engines share: checkpoint reading, packed weights, expert dispatch |
| `qwen3_native`, `gemma4_native`, `deepseek_native`, `gpt_oss_native`, `llama4_native` | The native inference engines |
| `optimizer`, `training` | `AdamW` and parameter groups over `InlineArray`s; the language-model loss and gradient clipping |
| `distributed` | Process groups and collectives |
| `memory` | Allocator counters, system memory and the estimates the trainers size a run with |
| `metallib` | Locating and caching the `mlx.metallib` the runtime loads |
| `error`, `try_ops`, `scalar` | The error channel, checked ops, and dtype-safe scalars |

## Feature flags

| Feature | Default | Description |
|---------|---------|-------------|
| `metal` | yes | The Metal backend and its compiled kernels |
| `accelerate` | yes | Apple's Accelerate framework for the CPU paths |
| `strict-errors` | no | Tests only: an op whose C++ side throws panics at that op, naming it, instead of returning a placeholder; also seeds a thread's random key from its name, so a test does not take its inputs from the clock. Every workspace crate turns it on in its own `[dev-dependencies]` |

## Building

The build script compiles MLX from source (CMake and a C++ toolchain are
required, plus the Metal toolchain for the metallib) and links it **statically**,
so a binary built from this crate runs without any MLX library beside it. The
first build is slow; these variables control it:

- `PMETAL_MLX_PREFIX`: a directory in which a finished MLX build is kept,
  one slot per MLX revision and feature set, and reused by later builds
- `PMETAL_MLX_STATIC=0`: link MLX as a shared library instead (many test
  binaries linking one dylib build faster than many linking a static archive)
- `PMETAL_MLX_LIB_DIR`: link an MLX you built yourself instead
- `PMETAL_METALLIB_PATH`: the `mlx.metallib` to load, ahead of every other location

An application that outlives its build tree carries a copy of the metallib and
points the runtime at it with `metallib::locate` and `inline_array::set_metallib_path`
before its first GPU op.

## Requirements

- Rust 1.95 or later
- macOS 14 (Sonoma) or later on Apple Silicon (M1 or newer)
- Xcode command-line tools, CMake, and the Metal toolchain
  (`xcodebuild -downloadComponent MetalToolchain`)

## License

MIT OR Apache-2.0
