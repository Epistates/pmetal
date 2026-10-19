# pmetal dataset

Look at, convert and prepare training data.

Every subcommand reads a dataset as `pmetal train` does: a JSONL, JSON, Parquet, Arrow, CSV or TSV
file, or a directory (its train split, every shard), with rows in any layout `pmetal dataset
formats` lists. A Hub dataset ID is fetched as Parquet. Text is formatted with the model's own chat
template, and lengths are counted with its tokenizer, so what the commands measure and write is what
training sees. Only the tokenizer and chat template of a model are fetched, never its weights.

## Subcommands

### analyze

Sample count, the layouts found, and character and token length statistics. With `--model`, tokens
are counted as training sees each row (conversations through the model's chat template).

```bash
pmetal dataset analyze --path train.jsonl --model Qwen/Qwen3-0.6B
pmetal dataset analyze --path data.parquet
```

### validate

Checks that every row reads, has text, and (with `--model`) fits `--max-seq-len` tokens. Rows that
cannot be trained on make the command exit with status 1; rows that are only too long are a note,
since they are truncated.

```bash
pmetal dataset validate --path train.jsonl --model Qwen/Qwen3-0.6B --max-seq-len 4096
```

### preview, download

```bash
pmetal dataset preview tatsu-lab/alpaca --num 3
pmetal dataset download tatsu-lab/alpaca --split train --output alpaca.jsonl
```

`download` writes the rows as they are, whatever their columns (message lists with tool calls
included).

### convert

Any readable format to JSONL, with columns renamed as `target=source` pairs.

```bash
pmetal dataset convert --input data.parquet --output data.jsonl
pmetal dataset convert --input data.csv --output data.jsonl --columns "text=content" --shuffle --seed 7
```

### filter

Keep rows by token length (`--min-tokens`, `--max-tokens`; need `--model`), exact duplicates
(`--dedup`), a regular expression over the sample's text (`--pattern`, `--invert`), or
completeness (`--complete-only`: a conversation needs a user and an assistant turn). A row that no
layout reads stops the command and names itself.

```bash
pmetal dataset filter -i raw.jsonl -o filtered.jsonl --model Qwen/Qwen3-0.6B --max-tokens 2048 --dedup
```

### split, merge, sample

```bash
pmetal dataset split --input all.jsonl --output-dir ./splits --val-ratio 0.1 --test-ratio 0.1
pmetal dataset split --input all.jsonl --output-dir ./splits --stratify category
pmetal dataset merge --inputs a.jsonl b.jsonl --output both.jsonl --weights 1.0,0.5 --shuffle
pmetal dataset sample --input all.jsonl --output small.jsonl --num 1000
```

`--stratify COLUMN` divides each value of the column on its own, so every set keeps the proportions
of the whole. A merge weight repeats a dataset (`2.0` is twice) or thins it (`0.5` keeps half, chosen
with `--seed`); `--interleave` takes a row of each dataset in turn. The same seed gives the same
output.

### template

Format conversations with a chat template and write `{"text": ...}` rows.

```bash
# The model's own chat template
pmetal dataset template -i data.jsonl -o templated.jsonl --model Qwen/Qwen3-0.6B

# A built-in template instead
pmetal dataset template -i data.jsonl -o templated.jsonl --template llama3
```

Instruction rows (Alpaca, problem/solution, prompt/completion) are formatted as the conversation
they stand for. `--system` gives a system message to conversations that have none.
`--add-generation-prompt` writes each conversation only up to where the model starts its last answer.
`--mask-prompt` writes `{"prompt": ..., "completion": ...}` rows, the last assistant turn being the
completion, which a trainer reads with only the completion carrying loss.

`--template` names a built-in template (`chatml`, `llama3`, `llama2`, `mistral`, `qwen`, `phi`,
`gemma`), `raw` for no template (`Role: content` blocks), or `auto` for the model's own. Without it,
the model's own is used, and ChatML without a model. A BOS token the tokenizer adds on its own is not
also written into the text.

### prepare

The whole way from a dataset (Hub ID or local path) to `train.jsonl` and `val.jsonl`: format with the
model's chat template, drop samples longer than `--max-seq-len` tokens (measured on the text as it
will be trained on) and duplicates, shuffle with `--seed`, split, and report token percentiles.

```bash
pmetal dataset prepare tatsu-lab/alpaca --output-dir ./alpaca_prepared \
  --model Qwen/Qwen3-0.6B --max-seq-len 2048 --val-ratio 0.05
```

`--columns "target=source,..."` renames columns first (for example `instruction=problem`).

### formats

Lists the row layouts and chat templates.

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

- [pmetal train](/cli/train/): train on a dataset
