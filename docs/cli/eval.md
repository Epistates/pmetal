# pmetal eval

Evaluate a model: perplexity on a dataset, or a benchmark task.

Perplexity is `exp` of the average negative log-likelihood per token of the dataset's text, rendered
with the model's chat template and cut at `--max-seq-len`. Lower is better. The dataset is read in
the formats `pmetal train` reads; `--lora` scores a model with an adapter on top, to compare it with
the base model on the same data. With `--json` the result is `model`, `dataset`, `num_samples`,
`total_tokens`, `avg_nll` and `perplexity`.

Plain text (and a task's prompts) start with the beginning-of-sequence token the model expects, once.
The tokenizer adds it when it is built to (Llama, Gemma 3); a model whose tokenizer does not, but
whose tokenizer config sets `add_bos_token` or whose chat template opens with `bos_token` (Gemma 4),
is given it all the same. A model with none (Qwen) gets none, and text that already opens with the
token is not given a second.

## Usage

```bash
# Perplexity
pmetal eval --model <MODEL> --dataset <DATASET> [OPTIONS]

# A benchmark
pmetal eval --model <MODEL> --task <TASK> [OPTIONS]
```

## Examples

```bash
# Perplexity, with and without a LoRA adapter
pmetal eval --model Qwen/Qwen3-0.6B --dataset eval.jsonl
pmetal eval --model Qwen/Qwen3-0.6B --dataset eval.jsonl \
  --lora ./output/adapter_model.safetensors

# GSM8K on 50 problems, with the results in a file
pmetal eval --model Qwen/Qwen3-0.6B --task gsm8k --limit 50 --output results/gsm8k.json

# MMLU, 5-shot, 100 questions drawn from across the subjects
pmetal eval --model Qwen/Qwen3-0.6B --task mmlu --limit 100

# ARC-Challenge with the per-item logs kept next to the results
pmetal eval --model Qwen/Qwen3-0.6B --task arc_challenge \
  --output results/arc.json --log-samples

# Your own questions: question/answer rows, or question/choices rows
pmetal eval --model Qwen/Qwen3-0.6B --task custom --dataset questions.jsonl --fewshot 3
```

## Tasks

A task is how a benchmark turns a dataset row into a prompt, what it asks of the model, and how it
scores the answer. These follow the widely used definitions, so a score here means what the same
benchmark's score means elsewhere.

| Task | Dataset (Hugging Face) | Asks | Reported |
|------|------------------------|------|----------|
| `gsm8k` | `openai/gsm8k` (`main`, test) | A greedy generation of up to `--max-gen-tokens` tokens, stopped at the next `Question:`, after the 5-shot chain-of-thought examples (`Question: ...\nAnswer: ... #### N`) | `exact_match,strict-match` (the number after `#### `) and `exact_match,flexible-extract` (the last number in the text); commas, `$`, everything up to `#### ` and a final `.` are ignored, and case |
| `mmlu` | `cais/mmlu` (`all`, test; the `dev` split for the examples) | The log-likelihood of ` A`, ` B`, ` C`, ` D` after the subject's description, its 5 `dev` questions and the question with its lettered choices | `acc`, overall and per subject |
| `arc_challenge` | `allenai/ai2_arc` (`ARC-Challenge`, test; train for the examples) | The log-likelihood of each choice's text after `Question: ...\nAnswer:` (25-shot) | `acc`; `acc_norm`, the log-likelihood over the choice's length in characters |
| `hellaswag` | `Rowan/hellaswag` (validation; train for the examples) | The log-likelihood of each ending after the activity label and context, with the dataset's `[title]`/bracket markup removed (10-shot) | `acc`, `acc_norm` |
| `ifeval-lite` | `google/IFEval` | A greedy generation (up to 1280 tokens) of the prompt as it is | `prompt_level_strict_acc`, `inst_level_strict_acc`, `prompt_level_loose_acc`, `inst_level_loose_acc` |
| `custom` | your `--dataset` | A generation, or the log-likelihood of choices | `exact_match`, or `acc` and `acc_norm` |

Every metric comes with its standard error (the sample standard deviation over √n).

- **Few-shot.** `--fewshot N` sets the examples shown before each question; without it a task uses
  its usual number (gsm8k 5, mmlu 5, arc_challenge 25, hellaswag 10, the others 0). MMLU's examples
  are the subject's own `dev` questions, in order. The others draw them at random from the task's
  training split with `--seed`: the same seed gives the same examples every run, but not the ones
  another implementation drew.
- **Limit.** `--limit N` (`--num-samples`) scores N items. A task of one subject scores its first
  items; MMLU, whose first items are all one subject, draws them from every subject with `--seed`.
- **Data.** The standard datasets are downloaded file by file (not whole repositories) into the Hub
  cache. `--dataset FILE` replaces the task's test data with a local file in the task's layout, for
  an offline run or a variant of the data.
- **Batching.** Log-likelihoods run in batches of up to `--batch-size` sequences, longest first.
  MMLU's four one-token choices are read from one forward of the question.
- **ifeval-lite** checks every IFEval instruction type except `language:response_language`, which
  needs a language detector: a prompt with one is skipped and counted as skipped. The checks for
  English capitals and lower case look at the case, not the language, and `capital_word_frequency`
  counts runs of word characters where the reference uses a Penn Treebank tokenizer.

### Custom data

One JSON object per line. A question is `question` (or `prompt`, `input`, `query`, `context`).

```json
{"question": "What is the capital of France?", "answer": "Paris"}
{"question": "Name a primary colour.", "answer": ["red", "blue", "yellow"]}
{"question": "Which is a mammal?", "choices": ["trout", "whale", "heron"], "answer": 1}
```

A row with `choices` is multiple choice: `answer` is the choice's index, its letter (`"B"`) or its
text, and a row is scored by the log-likelihood of each choice. A row without is a generation,
scored by exact match against any answer after lower-casing and trimming quotes, space and closing
punctuation; the model stops at a blank line. All rows must be of one kind. With `--fewshot` the
examples are the file's other rows. An optional `subject` groups the results.

## Results

`--output results.json` writes (and `--json` prints) the run:

```json
{
  "model": "Qwen/Qwen3-0.6B",
  "task": "gsm8k",
  "config": {"fewshot": 5, "limit": 50, "seed": 1234, "batch_size": 16, "max_gen_tokens": 256, "max_seq_len": 8192},
  "results": {
    "task": "gsm8k", "fewshot": 5, "samples": 50, "skipped": 0,
    "metrics": {
      "exact_match,strict-match": {"value": 0.34, "stderr": 0.0677},
      "exact_match,flexible-extract": {"value": 0.42, "stderr": 0.0705}
    }
  },
  "elapsed_seconds": 96.4
}
```

Tasks with subjects add `groups`, the same metrics per subject. `--log-samples` also writes
`results.samples.jsonl`, one line per item: its prompt, what the model said or the log-likelihood
of each choice, and the verdict.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | *required* | Model ID or local path |
| `--dataset` | — | Perplexity: the text to score. With `--task`: a local file of the task's data |
| `--task` | — | `gsm8k`, `mmlu`, `arc_challenge`, `hellaswag`, `ifeval-lite` or `custom`; without it, perplexity |
| `--fewshot` | the task's | Examples shown before each question |
| `--lora` | — | LoRA adapter to attach: a directory written by `pmetal train`, or its weights file |
| `--max-seq-len` | `0` | Longest sequence in tokens (0 = 1024 for perplexity, 8192 for a task); a longer one loses the start of its context |
| `--num-samples`, `--limit` | `0` | Score this many samples (0 = all) |
| `--seed` | `1234` | Which items a limit keeps and which few-shot examples are drawn |
| `--batch-size` | `16` | Most sequences in one scoring forward pass |
| `--max-gen-tokens` | `256` | Most tokens a generation task produces per item |
| `--output` | — | Write the results as JSON to this file |
| `--log-samples` | `false` | Keep each item's prompt, output and verdict beside `--output` |
| `--json` | `false` | Print the results as JSON |

## See Also

- [pmetal train](/cli/train/) — Train a model
- [pmetal infer](/cli/infer/) — Run inference
