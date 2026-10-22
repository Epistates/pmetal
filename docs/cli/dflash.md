# pmetal dflash

Block-diffusion speculative decoding with a DFlash draft.

A Qwen3 target runs alongside a DFlash draft that proposes a block of tokens in one pass; the target verifies the block in a single forward pass. The output is the target's own: greedy at `--temperature 0`, otherwise drawn from the target's distribution at that temperature. The draft only changes the speed.

Train a draft for your own target with [`pmetal train-draft`](/cli/train-draft/). For DFlash inside the general inference command (including the Apple Neural Engine path) see `--draft-model` on [`pmetal infer`](/cli/infer/).

## Usage

```bash
pmetal dflash \
  --target <QWEN3_TARGET> \
  --draft <DRAFT> \
  --prompt <PROMPT> \
  [OPTIONS]
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--target` | required | Target Qwen3 model (Hub ID or local path) |
| `--draft` | required | DFlash draft model (Hub ID or local path) |
| `-p`, `--prompt` | required | Input prompt |
| `--max-new-tokens` | `128` | Maximum tokens to generate after the prompt |
| `--temperature` | `0.0` | Sampling temperature; `0` is greedy |
| `--speculative-tokens` | the draft's `block_size` | Override the draft block size |
| `--draft-fp8` | `false` | Quantize the draft's linear weights to FP8 (E4M3) on load |
| `--tree-budget` | `0` | Verify a draft tree of this many nodes (`0` verifies one linear block). Needs a target on the native runtime; a DFlash 2 draft proposes a single path and verifies linearly |
| `--no-chat` | `false` | Skip the model's chat template and tokenize the prompt verbatim |
| `--compare-greedy` | `false` | Also decode the prompt greedily, one token per forward pass, and report its speed and whether the DFlash tokens match |
| `--json` | `false` | Emit a JSON report of tokens and metrics instead of plain text |

## Examples

```bash
# Greedy decoding with a published draft
pmetal dflash \
  --target Qwen/Qwen3-4B \
  --draft z-lab/Qwen3-4B-DFlash-b16 \
  --prompt "Explain monotonic queues." \
  --max-new-tokens 256

# Check the speedup and that the tokens match plain greedy decoding
pmetal dflash \
  --target Qwen/Qwen3-4B \
  --draft z-lab/Qwen3-4B-DFlash-b16 \
  --prompt "Explain monotonic queues." \
  --compare-greedy --json
```

## See Also

- [pmetal train-draft](/cli/train-draft/): Train a DFlash draft checkpoint
- [pmetal infer](/cli/infer/): Speculative decoding options in the general inference command
