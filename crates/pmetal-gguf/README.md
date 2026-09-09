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

```rust,no_run
use pmetal_gguf::GgufContent;

fn inspect(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let gguf = GgufContent::from_file(path)?;

    println!("Architecture: {:?}", gguf.architecture());
    println!("Context length: {:?}", gguf.get_metadata("llama.context_length"));

    for name in gguf.tensor_names() {
        if let Some(info) = gguf.get_tensor_info(name) {
            println!("{name}: {:?}", info.dimensions);
        }
    }
    Ok(())
}
```

### Dequantizing Tensors

`dequantize` takes the raw tensor bytes plus the type and shape the header recorded.

```rust,no_run
use std::fs::File;

use pmetal_gguf::{GgufContent, dequant};

fn dequantize_one(path: &str, name: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;
    let gguf = GgufContent::from_file(path)?;

    let info = gguf
        .get_tensor_info(name)
        .ok_or("tensor not present in this file")?;
    let shape: Vec<i32> = info.dimensions.iter().map(|&d| d as i32).collect();
    let dtype = info.dtype;

    let bytes = gguf.read_tensor_data(&mut file, name)?;
    Ok(dequant::dequantize(&bytes, dtype, &shape)?)
}
```

### Converting to GGUF

```rust,no_run
use std::fs::File;

use pmetal_gguf::GgufBuilder;

fn write(path: &str, embeddings: Vec<f32>) -> Result<(), Box<dyn std::error::Error>> {
    let mut builder = GgufBuilder::with_model("llama", "My Model");
    builder.add_u32("llama.context_length", 4096);
    builder.add_f32_tensor("token_embd.weight", vec![4096, 32000], embeddings);

    let mut file = File::create(path)?;
    builder.write(&mut file)?;
    Ok(())
}
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
| `quantize` | GGUF file creation and quantization |
| `dequant` | Tensor dequantization |
| `dynamic` | Dynamic quantization |
| `imatrix` | Importance matrix support |
| `k_quants` | K-quant implementations |
| `iq_quants` | IQ-quant implementations |
| `vec_dot` | Vectorized dot product kernels |

## License

MIT OR Apache-2.0
