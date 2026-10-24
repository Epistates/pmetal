# pmetal embed-train

Train a sentence-embedding model (BERT / encoder-only) with contrastive losses.

Fine-tunes the whole encoder, or LoRA adapters with `--lora`, on pairs or triplets of texts. `--model` is a local model directory (a plain encoder checkpoint or a sentence-transformers one); `pmetal download` prints the snapshot directory of a Hub model. A sentence-transformers model keeps its pipeline in the output (`modules.json`, the pooling and normalize modules), so `sentence-transformers` and `pmetal` both load what is written.

Ctrl-C (SIGINT) or SIGTERM finishes the step in flight, saves what is trained so far and exits with status 130. `--log-events` (a global option) writes the run's events as JSONL.

## Usage

```bash
pmetal embed-train \
  --model <MODEL_DIR> \
  --dataset <DATASET> \
  [OPTIONS]
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `-m`, `--model` | required | Path to the encoder model directory |
| `-d`, `--dataset` | required | Training dataset: JSONL pairs or triplets (see below) |
| `-o`, `--output` | `./output-embed` | Output directory |
| `--loss` | `info_nce` | `info_nce`, `mnrl`, `triplet`, `cosent` or `cosine_similarity` |
| `--pooling` | `auto` | `auto` (the pooling the model was trained with, mean without one), `mean`, `cls`, `max`, `last_token` or `weighted_mean` |
| `--temperature` | `0.05` | Temperature for InfoNCE, and 1/scale for CoSENT |
| `--margin` | `0.3` | Margin for the triplet loss |
| `--learning-rate` | `2e-5` | Learning rate |
| `--batch-size` | `32` | Batch size |
| `--epochs` | `3` | Training epochs |
| `--max-seq-len` | `512` | Maximum input length, capped at the model's own `max_seq_length` when it declares one |
| `--weight-decay` | `0.01` | AdamW weight decay |
| `--no-normalize` | `false` | Skip L2 normalization of the embeddings before the loss |
| `--log-every` | `10` | Log progress every N steps |
| `--seed` | `42` | Random seed (dataset shuffling and LoRA initialization) |
| `--lora` | `false` | Train LoRA adapters on every linear layer and save them as an adapter instead of fine-tuning the whole encoder |
| `--lora-r` | `16` | LoRA rank (with `--lora`) |
| `--lora-alpha` | `32.0` | LoRA alpha (with `--lora`) |

## Dataset Formats

The format is detected from the first row.

**Pairs** (`info_nce`, `mnrl`, `cosent`, `cosine_similarity`). The key pairs `text_a`/`text_b`, `sentence1`/`sentence2`, `query`/`positive` and `premise`/`hypothesis` are accepted. An optional `label` (or `score`) is a similarity in [0, 1]; a pair without one is a positive pair.

```json
{"text_a": "What is ML?", "text_b": "Machine learning is...", "label": 0.9}
{"query": "What is ML?", "positive": "Machine learning is..."}
```

**Triplets** (`triplet`):

```json
{"anchor": "What is ML?", "positive": "Machine learning is...", "negative": "Cooking recipes..."}
```

## Examples

```bash
# InfoNCE contrastive training with pairs
pmetal embed-train \
  --model /path/to/bge-small-en-v1.5 \
  --dataset pairs.jsonl \
  --output ./output/embeddings \
  --loss info_nce

# Triplet training with a margin
pmetal embed-train \
  --model /path/to/bge-small-en-v1.5 \
  --dataset triplets.jsonl \
  --loss triplet --margin 0.3

# CoSENT with mean pooling and LoRA adapters
pmetal embed-train \
  --model /path/to/all-MiniLM-L6-v2 \
  --dataset scored_pairs.jsonl \
  --loss cosent --pooling mean --lora
```

## See Also

- [Training Overview](/training/overview/): All training methods
- [pmetal download](/cli/download/): Fetch an encoder model
