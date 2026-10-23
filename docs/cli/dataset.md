# pmetal dataset

Look at, convert and prepare training data.

Every subcommand reads a dataset as `pmetal train` does: a JSONL, JSON, Parquet, Arrow, CSV or TSV
file, or a directory (its train split, every shard), with rows in any layout `pmetal dataset
formats` lists. A Hub dataset ID is fetched as Parquet. Text is formatted with the model's own chat
template, and lengths are counted with its tokenizer, so what the commands measure and write is what
training sees. Only the tokenizer and chat template of a model are fetched, never its weights.

## Usage

```bash
pmetal dataset <SUBCOMMAND> [OPTIONS]
```

| Subcommand | What it does |
|------------|--------------|
| [`analyze`](#analyze) | Sample count, layouts found, length statistics |
| [`validate`](#validate) | Check that every row can be trained on |
| [`preview`](#preview-and-download) | Print the first rows of a dataset |
| [`download`](#preview-and-download) | Fetch a Hub dataset as JSONL |
| [`convert`](#convert) | Any readable format to JSONL, with columns renamed |
| [`filter`](#filter) | Keep rows by length, duplicates, a pattern or completeness |
| [`split`](#split-merge-and-sample) | Train, validation and test sets |
| [`merge`](#split-merge-and-sample) | Several datasets into one |
| [`sample`](#split-merge-and-sample) | A random sample |
| [`template`](#template) | Format conversations with a chat template |
| [`prepare`](#prepare) | The whole pipeline, from a dataset to `train.jsonl` and `val.jsonl` |
| [`formats`](#formats) | List the row layouts and chat templates |

## analyze

Sample count, the layouts found, and character and token length statistics. With `--model`, tokens
are counted as training sees each row (conversations through the model's chat template).

| Option | Default | Description |
|--------|---------|-------------|
| `-p`, `--path` | required | Dataset file or directory |
| `-m`, `--model` | none | Model whose tokenizer counts tokens and whose chat template renders conversations |
| `--detailed` | `false` | Per-sample statistics |

```bash
pmetal dataset analyze --path train.jsonl --model Qwen/Qwen3-0.6B
pmetal dataset analyze --path data.parquet
```

## validate

Checks that every row reads, has text, and (with `--model`) fits `--max-seq-len` tokens. Rows that
cannot be trained on make the command exit with status 1; rows that are only too long are a note,
since they are truncated.

| Option | Default | Description |
|--------|---------|-------------|
| `-p`, `--path` | required | Dataset file or directory |
| `-m`, `--model` | none | Model whose tokenizer (and chat template) measures the samples |
| `--max-seq-len` | `2048` | Maximum sequence length to check against |

```bash
pmetal dataset validate --path train.jsonl --model Qwen/Qwen3-0.6B --max-seq-len 4096
```

## preview and download

| Option | Default | Description |
|--------|---------|-------------|
| `<DATASET_ID>` | required | Hub dataset ID (`preview` also takes a local path) |
| `--split` | `train` | Dataset split, for a Hub dataset |
| `-n`, `--num` | `5` | Samples to show (`preview` only) |
| `-o`, `--output` | none | Output JSONL path (`download` only) |
| `--revision` | the default branch | Branch or revision (`download` only) |

```bash
pmetal dataset preview tatsu-lab/alpaca --num 3
pmetal dataset download tatsu-lab/alpaca --split train --output alpaca.jsonl
```

`download` writes the rows as they are, whatever their columns (message lists with tool calls
included).

## convert

Any readable format to JSONL, with columns renamed as `target=source` pairs.

| Option | Default | Description |
|--------|---------|-------------|
| `-i`, `--input` | required | Input file or directory (Parquet, JSON, JSONL, Arrow, CSV, TSV) |
| `-o`, `--output` | required | Output JSONL file |
| `--columns` | none | Rename columns: `"target=source,target=source"`, e.g. `"text=content"` takes `text` from the column `content` |
| `--shuffle` | `false` | Shuffle the output |
| `--seed` | `42` | Seed for shuffling |

```bash
pmetal dataset convert --input data.parquet --output data.jsonl
pmetal dataset convert --input data.csv --output data.jsonl --columns "text=content" --shuffle --seed 7
```

## filter

Keep rows by token length (`--min-tokens`, `--max-tokens`; need `--model`), exact duplicates
(`--dedup`), a regular expression over the sample's text (`--pattern`, `--invert`), or
completeness (`--complete-only`: a conversation needs a user and an assistant turn). A row that no
layout reads stops the command and names itself.

| Option | Default | Description |
|--------|---------|-------------|
| `-i`, `--input` | required | Input dataset (file or directory) |
| `-o`, `--output` | required | Output JSONL path |
| `-m`, `--model` | none | Model for token-based filtering |
| `--min-tokens` | none | Minimum token count (needs `--model`) |
| `--max-tokens` | none | Maximum token count (needs `--model`) |
| `--dedup` | `false` | Remove duplicates (exact match of their text) |
| `--pattern` | none | Regular expression matched against the sample's text; keeps the matching samples (a conversation reads as `Role: content` blocks) |
| `--invert` | `false` | Keep the samples that do not match `--pattern` |
| `--complete-only` | `false` | Keep only conversations that have both a user and an assistant turn |

```bash
pmetal dataset filter -i raw.jsonl -o filtered.jsonl --model Qwen/Qwen3-0.6B --max-tokens 2048 --dedup
```

## split, merge and sample

| `split` option | Default | Description |
|----------------|---------|-------------|
| `-i`, `--input` | required | Input dataset |
| `-o`, `--output-dir` | required | Output directory for the split files |
| `--val-ratio` | `0.1` | Validation share (0 to 1) |
| `--test-ratio` | `0.0` | Test share (0 to 1) |
| `--seed` | `42` | Seed for the split |
| `--stratify` | none | Keep the proportions of this column's values in every split (e.g. `category`) |

| `merge` option | Default | Description |
|----------------|---------|-------------|
| `-i`, `--inputs` | required | Input datasets (files or directories) |
| `-o`, `--output` | required | Output JSONL path |
| `--weights` | none | One weight per input, comma-separated (`2.0` repeats a dataset, `0.5` keeps half of it) |
| `--interleave` | `false` | Take a row of each dataset in turn |
| `--shuffle` | `false` | Shuffle after merging |
| `--seed` | `42` | Seed for shuffling and weighted sampling |

| `sample` option | Default | Description |
|-----------------|---------|-------------|
| `-i`, `--input` | required | Input dataset |
| `-o`, `--output` | required | Output JSONL path |
| `-n`, `--num` | required | Number of samples to take |
| `--seed` | `42` | Random seed |

```bash
pmetal dataset split --input all.jsonl --output-dir ./splits --val-ratio 0.1 --test-ratio 0.1
pmetal dataset split --input all.jsonl --output-dir ./splits --stratify category
pmetal dataset merge --inputs a.jsonl b.jsonl --output both.jsonl --weights 1.0,0.5 --shuffle
pmetal dataset sample --input all.jsonl --output small.jsonl --num 1000
```

`--stratify COLUMN` divides each value of the column on its own, so every set keeps the proportions
of the whole. A merge weight repeats a dataset (`2.0` is twice) or thins it (`0.5` keeps half, chosen
with `--seed`). The same seed gives the same output.

## template

Format conversations with a chat template and write `{"text": ...}` rows.

**A `text` row is trained whole.** A trainer reading these rows puts loss on every token, the user's
turn and the system prompt included. To train on the answer alone, pass `--mask-prompt`, which writes
`{"prompt": ..., "completion": ...}` rows instead.

| Option | Default | Description |
|--------|---------|-------------|
| `-i`, `--input` | required | Input dataset (file or directory) |
| `-o`, `--output` | required | Output JSONL path |
| `-m`, `--model` | none | Model whose chat template formats the conversations |
| `--template` | the model's own, else `chatml` | A built-in template: `chatml`, `llama3`, `llama2`, `mistral`, `qwen`, `phi`, `gemma`, `raw` (no template, `Role: content` blocks) or `auto` (the model's own; needs `--model`) |
| `--system` | none | System message for conversations that have none |
| `--add-generation-prompt` | `false` | Write each conversation only up to where the model starts its last answer |
| `--mask-prompt` | `false` | Write `{"prompt": ..., "completion": ...}` rows, the last assistant turn being the completion, so a trainer trains on the completion only |

```bash
# The model's own chat template
pmetal dataset template -i data.jsonl -o templated.jsonl --model Qwen/Qwen3-0.6B

# A built-in template instead
pmetal dataset template -i data.jsonl -o templated.jsonl --template llama3

# Rows a trainer learns the answers of, not the questions
pmetal dataset template -i data.jsonl -o masked.jsonl --model Qwen/Qwen3-0.6B --mask-prompt
```

Instruction rows (Alpaca, problem/solution, prompt/completion) are formatted as the conversation
they stand for. A BOS token the tokenizer adds on its own is not also written into the text.

## prepare

The whole way from a dataset (Hub ID or local path) to `train.jsonl` and `val.jsonl`: format with the
model's chat template, drop samples longer than `--max-seq-len` tokens (measured as they will be
trained on) and duplicates, shuffle with `--seed`, split, and report token percentiles.

A conversation is written as `{"prompt": ..., "completion": ...}` rows, one per assistant turn: the
prompt is everything before the turn, and training learns only the completion, so the system prompt
and the user's words are context, not something the model is taught to say. Plain text stays a
`{"text": ...}` document, trained whole.

| Option | Default | Description |
|--------|---------|-------------|
| `<DATASET>` | required | Hub dataset ID or local path |
| `-o`, `--output-dir` | required | Output directory |
| `-m`, `--model` | required | Model whose tokenizer measures the samples and whose chat template formats them |
| `--template` | the model's own | A built-in chat template in place of the model's (the values of `dataset template`) |
| `--max-seq-len` | `2048` | Drop samples longer than this many tokens |
| `--val-ratio` | `0.05` | Validation share |
| `--seed` | `42` | Random seed |
| `--no-dedup` | `false` | Keep duplicates |
| `--columns` | none | Rename columns first: `"target=source,..."` (for example `instruction=problem`) |

```bash
pmetal dataset prepare tatsu-lab/alpaca --output-dir ./alpaca_prepared \
  --model Qwen/Qwen3-0.6B --max-seq-len 2048 --val-ratio 0.05
```

## formats

Lists the row layouts and chat templates. No options.

## Supported Formats

| Format | Extensions | Read | Write |
|--------|-----------|------|-------|
| JSONL | `.jsonl` | Yes | Yes |
| JSON | `.json` | Yes | No |
| Parquet | `.parquet` | Yes | No |
| Arrow | `.arrow` | Yes | No |
| CSV, TSV | `.csv`, `.tsv` | Yes | No |

The commands write JSONL.

## See Also

- [pmetal train](/cli/train/): Train on a dataset
- [pmetal tokenize](/cli/tokenize/): Tokenize a corpus for pretraining
