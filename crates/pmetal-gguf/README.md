# pmetal-gguf

GGUF file format support for llama.cpp and Ollama compatibility.

## Overview

This crate provides reading and writing support for the GGUF (GPT-Generated Unified Format) file format, enabling compatibility with llama.cpp, Ollama, and other GGUF-compatible inference engines.

## Features

- **GGUF Reading**: Parse GGUF files and extract metadata/tensors
- **GGUF Writing**: Create GGUF files from SafeTensors/PyTorch models
- **Tensor Dequantization**: Convert quantized tensors to full precision
- **Metadata Handling**: Read/write model metadata and tokenizer info

## Usage

### Reading GGUF Files

```rust
use pmetal_gguf::GgufContent;

// Load GGUF file
let gguf = GgufContent::from_file("model.gguf")?;

// Access metadata
println!("Architecture: {}", gguf.metadata.get("general.architecture")?);
println!("Context length: {}", gguf.metadata.get("llama.context_length")?);

// Iterate tensors
for (name, tensor) in gguf.tensors() {
    println!("{}: {:?}", name, tensor.shape());
}
```

### Dequantizing Tensors

```rust
use pmetal_gguf::{GgufContent, dequantize};

let gguf = GgufContent::from_file("model-q4.gguf")?;

// Dequantize a specific tensor
let weights = gguf.get_tensor("model.layers.0.self_attn.q_proj.weight")?;
let fp32_weights = dequantize(&weights)?;
```

### Converting to GGUF

```rust
use pmetal_gguf::{GgufWriter, Quantization};

let mut writer = GgufWriter::new("output.gguf")?;

// Set metadata
writer.set_metadata("general.architecture", "llama")?;
writer.set_metadata("general.name", "My Model")?;

// Add tensors with optional quantization
writer.add_tensor("model.embed_tokens.weight", &embeddings, Quantization::None)?;
writer.add_tensor("model.layers.0.self_attn.q_proj.weight", &weights, Quantization::Q4_K)?;

writer.finish()?;
```

## Supported Quantization Types

| Type | Bits | Description |
|------|------|-------------|
| F32 | 32 | Full precision |
| F16 | 16 | Half precision |
| Q8_0 | 8 | 8-bit quantization |
| Q4_0 | 4 | 4-bit quantization |
| Q4_K | 4 | K-quant (higher quality) |
| Q5_K | 5 | K-quant |
| Q6_K | 6 | K-quant |

## Modules

| Module | Description |
|--------|-------------|
| `reader` | GGUF file parsing |
| `writer` | GGUF file creation |
| `dequant` | Tensor dequantization |
| `metadata` | Metadata handling |

## License

MIT OR Apache-2.0
