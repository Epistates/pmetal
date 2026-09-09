# pmetal embed-train

Sentence-transformer fine-tuning for BERT/encoder models with contrastive losses.

Fine-tune sentence embedding models (BERT, encoder architectures) with contrastive learning objectives. Supports pair and triplet datasets with configurable pooling and normalization.

## Usage

```bash
pmetal embed-train \
  --model <MODEL> \
  --dataset <DATASET> \
  --output <OUTPUT_DIR> \
  [OPTIONS]
```

## Examples

```bash
# InfoNCE contrastive training with pairs
pmetal embed-train \
  --model BAAI/bge-small-en-v1.5 \
  --dataset pairs.jsonl \
  --output ./output/embeddings \
  --loss infonce

# Triplet training with margin
pmetal embed-train \
  --model BAAI/bge-small-en-v1.5 \
  --dataset triplets.jsonl \
  --loss triplet

# CoSENT with mean pooling
pmetal embed-train \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --dataset pairs.jsonl \
  --loss cosent --pooling mean
```

## Dataset Formats

**Pair JSONL** (for InfoNCE/CoSENT):
```json
{"text_a": "What is ML?", "text_b": "Machine learning is...", "label": 1}
```

**Triplet JSONL** (for Triplet loss):
```json
{"anchor": "What is ML?", "positive": "Machine learning is...", "negative": "Cooking recipes..."}
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | *required* | BERT/encoder model path or HuggingFace ID |
| `--dataset` | *required* | Training dataset (pair or triplet JSONL) |
| `--output` | `./output` | Output directory |
| `--loss` | `infonce` | Loss function: `infonce`, `triplet`, `cosent` |
| `--pooling` | `cls` | Pooling strategy: `cls`, `mean`, `last_token` |
| `--no-normalize` | `false` | Skip L2 normalization of embeddings |
| `--learning-rate` | `2e-5` | Learning rate |
| `--batch-size` | `32` | Batch size |
| `--epochs` | `3` | Training epochs |

## See Also

- [Training Overview](/training/overview/) — All training methods
