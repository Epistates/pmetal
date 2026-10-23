# pmetal tokenize

Tokenize a text corpus into binary shards for pretraining.

The corpus is read in the formats `pmetal train` reads (JSONL, JSON, Parquet, Arrow, CSV, or a directory of shards), one document per row, from a text column. Each output shard is a flat little-endian binary file named `shard_00000.bin`, `shard_00001.bin`, and so on: for every document, a `u32` token count followed by its `u32` token IDs. [`pmetal pretrain`](/cli/pretrain/), [`pmetal train-draft`](/cli/train-draft/) and [`pmetal train-mtp`](/cli/train-mtp/) read them.

Rows with a missing or empty text column are skipped and counted in the summary. The command prints the number of documents, shards and tokens when it finishes.

## Usage

```bash
pmetal tokenize \
  --input <CORPUS> \
  --output <OUTPUT_DIR> \
  --tokenizer <TOKENIZER>
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `-i`, `--input` | required | Corpus file |
| `-o`, `--output` | required | Output directory for the shard files |
| `-t`, `--tokenizer` | required | Tokenizer: a `tokenizer.json` file, a model directory holding one, or a Hub model ID (only its `tokenizer.json` is fetched, never the weights) |
| `--text-column` | `text` | Column holding the text |
| `--docs-per-shard` | `10000` | Maximum documents per shard |

## Examples

```bash
# A JSONL corpus with a "text" column
pmetal tokenize \
  --input corpus.jsonl \
  --output ./tok \
  --tokenizer Qwen/Qwen3-0.6B

# A different column, smaller shards
pmetal tokenize \
  --input corpus.jsonl \
  --output ./tok \
  --tokenizer Qwen/Qwen3-0.6B \
  --text-column body --docs-per-shard 1000
```

Use the tokenizer of the model the shards will train: the shards hold token IDs, so another vocabulary trains on the wrong tokens.

## See Also

- [pmetal pretrain](/cli/pretrain/): Pretrain from scratch on the shards
- [pmetal train-draft](/cli/train-draft/) and [pmetal train-mtp](/cli/train-mtp/): Speculative-decoding drafters
