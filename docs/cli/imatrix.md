# pmetal imatrix

Measure an importance matrix on calibration data.

Run a model over calibration text and record, for every weight matrix, how large the activations are in each input channel. The quantizers use that to spend their few levels where the activations are large (`pmetal quantize --imatrix`). One measurement serves every quantization of the model.

## Usage

```bash
pmetal imatrix \
  --model <MODEL> \
  --dataset <CALIBRATION_DATA> \
  --output imatrix.gguf \
  [OPTIONS]
```

## Examples

```bash
# From a text file
pmetal imatrix \
  --model Qwen/Qwen3-0.6B \
  --dataset calibration.txt \
  --output qwen3-0.6b.imatrix.gguf

# From a JSONL dataset, 128 chunks of 2048 tokens, in the legacy .dat format
pmetal imatrix \
  --model ./output \
  --dataset calibration.jsonl \
  --ctx-size 2048 --chunks 128 \
  --output imatrix.dat

# Use it
pmetal quantize \
  --model Qwen/Qwen3-0.6B \
  --output qwen3-0.6b-iq4_xs.gguf --method iq4_xs \
  --imatrix qwen3-0.6b.imatrix.gguf
```

## Options

| Flag | Description |
|------|-------------|
| `--model` | The model to measure: a Hub id or a local directory (safetensors) |
| `--dataset` | Calibration data: a text file (one document), or a JSONL or Parquet dataset, local or a Hub dataset id |
| `--output` | Where to write the matrix: a `.gguf` file (the imatrix format of current tools) or, for a `.dat` path, the legacy format (default `./imatrix.gguf`) |
| `--ctx-size` | Tokens per chunk, the context each forward runs at (default 512) |
| `--chunks` | Chunks to run (default 0, all the data makes) |
| `--batch-size` | Chunks per forward pass (default 4) |

## What is measured

The documents are tokenized as the model would see them (conversation rows through the model's chat template, plain text with the tokenizer's special tokens), laid end to end and cut into non-overlapping chunks of `--ctx-size` tokens; each chunk starts with the BOS token when the tokenizer adds one. A text file is one document; a dataset has one per row.

The model's own forward runs over the chunks, and the input of every weight matrix (every attention and feed-forward projection, the output head, and for a model with tied embeddings the embedding matrix, which is the head) is squared and summed per input channel, in `f32` on the GPU. The output logits are not computed. A mixture-of-experts layer's routed experts are measured per expert, counting only the tokens routed to each. Memory is the model's own plus a vector per matrix, however much data is given.

Matrices the forward reaches by a path that does not report its input are listed in a warning and quantize without importance. Use a few hundred chunks of text like the text the model will run on; a matrix measured on very little text describes that text more than the model.

## See Also

- [pmetal quantize](/cli/quantize/) — Quantize with the matrix
- [Quantization](/models/quantization/) — Detailed quantization guide
