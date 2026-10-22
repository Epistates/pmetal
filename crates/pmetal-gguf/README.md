# pmetal-gguf

Reading, writing, dequantizing and quantizing GGUF files, the single-file format for GGML-family
inference runtimes. Implements the GGUF specification (versions 1 to 3) and the ggml block formats.

It is plain Rust with no GPU or runtime dependency. `pmetal-models` loads GGUF checkpoints through
it, `pmetal-data` builds a tokenizer from the one a file carries, `pmetal-hub` verifies GGUF
downloads and writes the `config.json` they lack, and the `pmetal` CLI writes quantized models
with it.

## Features

- **Reading**: a parser that checks every count and length against the bytes the file has left and
  every tensor against the file size and alignment, so a hostile or truncated file is an error, not
  an allocation or a panic. `GgufModel` memory-maps a model, including **split** sets
  (`name-00001-of-00003.gguf`), and serves tensor bytes as slices of the mapping.
- **Dequantization**: every tensor type a file can carry (F32/F16/BF16, the integer types,
  Q4_0 to Q8_1, Q2_K to Q8_K, IQ1_S/M, IQ2_XXS/XS/S, IQ3_XXS/S, IQ4_NL/XS, TQ1_0/TQ2_0, MXFP4,
  NVFP4, Q1_0), bit-exact against the reference decoder and parallel over blocks.
- **Quantization**: the legacy and K-quant formats, IQ4_NL/IQ4_XS, the ternary, 1-bit and 4-bit
  float formats, with optional per-channel importance (an imatrix) steering the scale search.
- **Writing**: `GgufBuilder` for small files, `GgufStreamWriter` / `GgufFileWriter` to stream a
  multi-gigabyte model tensor by tensor, atomically.
- **Metadata**: tokenizer and `config.json` extraction from `tokenizer.ggml.*` and
  `<arch>.*` keys.

## Installation

```toml
[dependencies]
pmetal-gguf = "0.7"
```

## Usage

### Round trip in memory

Quantize a tensor, write a GGUF file to bytes, read it back and decode it. `TensorInfo::dimensions`
are row-major, outermost first, as `dequant` takes them.

```rust
use std::io::Cursor;

use pmetal_gguf::{GgmlType, GgufBuilder, GgufContent, dequant, quantize};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Two rows of 64 weights, quantized to Q8_0 (blocks of 32).
    let (rows, cols) = (2usize, 64usize);
    let weights: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.05).sin()).collect();
    let packed = quantize::quantize_tensor(&weights, GgmlType::Q8_0, cols, None)?;

    let mut builder = GgufBuilder::with_model("llama", "tiny");
    builder.add_raw_tensor(
        "blk.0.ffn_up.weight",
        vec![rows as u64, cols as u64],
        GgmlType::Q8_0,
        packed,
    );
    let mut file = Cursor::new(builder.build_to_bytes()?);

    let gguf = GgufContent::read(&mut file)?;
    assert_eq!(gguf.architecture(), Some("llama"));
    let info = gguf.get_tensor_info("blk.0.ffn_up.weight").unwrap();
    let raw = gguf.read_tensor_data(&mut file, "blk.0.ffn_up.weight")?;
    let shape: Vec<i32> = info.dimensions.iter().map(|&d| d as i32).collect();
    let decoded = dequant::dequantize(&raw, info.dtype, &shape)?;

    let worst = weights.iter().zip(&decoded).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
    assert!(worst < 0.01);
    Ok(())
}
```

### Reading GGUF Files

```rust,no_run
use pmetal_gguf::GgufContent;

fn inspect(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let gguf = GgufContent::from_file(path)?;

    println!("Architecture: {:?}", gguf.architecture());
    println!("Context length: {:?}", gguf.get_u64("llama.context_length"));

    for name in gguf.tensor_names() {
        if let Some(info) = gguf.get_tensor_info(name) {
            println!("{name}: {:?}", info.dimensions);
        }
    }
    Ok(())
}
```

### Loading a model, split or not

`GgufModel` opens any shard of a split set, presents one tensor namespace and borrows tensor bytes
from the mapping; `dequantize_into` decodes straight into your buffer.

```rust,no_run
use pmetal_gguf::{GgufModel, dequant};

fn load(path: &str, name: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let model = GgufModel::open(path)?;
    let info = model.tensor_info(name).ok_or("tensor not present in this model")?;
    let shape: Vec<i32> = info.dimensions.iter().map(|&d| d as i32).collect();

    let mut weights = vec![0.0f32; info.n_elements()];
    dequant::dequantize_into(model.tensor_data(name)?, info.dtype, &shape, &mut weights)?;
    Ok(weights)
}
```

### Writing a quantized model

Declare the tensors, then supply them one at a time; the file appears atomically when finished.

```rust,no_run
use std::collections::BTreeMap;

use pmetal_gguf::{GgmlType, GgufFileWriter, MetadataValue, TensorInfo, quantize};

fn write(path: &str, weights: &[f32], rows: usize, cols: usize) -> Result<(), Box<dyn std::error::Error>> {
    let mut metadata = BTreeMap::new();
    metadata.insert("general.architecture".to_string(), MetadataValue::String("llama".into()));

    // `importance` is optional: one weight per input channel (column).
    let quantized = quantize::quantize_tensor(weights, GgmlType::Q4K, cols, None)?;

    let tensors = vec![TensorInfo::new("blk.0.ffn_up.weight", vec![rows as u64, cols as u64], GgmlType::Q4K)];
    let mut file = GgufFileWriter::create(path, &metadata, tensors, 32)?;
    file.write_tensor("blk.0.ffn_up.weight", &quantized)?;
    file.finish()?;
    Ok(())
}
```

For small files `GgufBuilder::with_model(..)` collects everything in memory and writes it with
`write_file` (atomic) or `build_to_bytes`.

## Tensor types

All types decode. "Encode" lists what `quantize::quantize_tensor` writes; the rest are read-only
(a quantizer for the IQ1/IQ2/IQ3 grid formats needs the reference's grid-neighbour search).

| Family | Types | Bits/weight | Encode |
|--------|-------|-------------|--------|
| Float / integer | F32, F16, BF16, F64, I8..I64 | 8 to 64 | F32, F16, BF16 |
| Legacy | Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1 | 4.5 to 9 | all |
| K-quants | Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K | 2.6 to 9.1 | all |
| IQ | IQ4_NL, IQ4_XS | 4.5, 4.25 | both |
| IQ | IQ1_S, IQ1_M, IQ2_XXS/XS/S, IQ3_XXS/S | 1.6 to 3.4 | no |
| Ternary / 1-bit | TQ1_0, TQ2_0, Q1_0 | 1.1 to 2.1 | all |
| 4-bit float | MXFP4, NVFP4 | 4.25, 4.5 | both |

The repacked layouts (`Q4_0_4_4` and friends) interleave rows for a particular kernel and are not
decoded.

## Modules

| Module | Description |
|--------|-------------|
| `reader` | GGUF header parsing and validation (`GgufContent`) |
| `GgufModel` | Memory-mapped model, split sets |
| `dequant` | Tensor dequantization |
| `quantize` | Quantization, importance weighting, row-length fallbacks |
| `dynamic` | Per-tensor type selection and bit-budget allocation |
| `imatrix` | Importance matrix (legacy `.dat` and GGUF) |
| `k_quants`, `iq_quants` | Block formats |
| `tokenizer`, `config` | Metadata extraction |

## Feature flags

None.

## Requirements

- Rust 1.95 or later; any platform

## License

MIT OR Apache-2.0
