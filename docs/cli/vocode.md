# pmetal vocode

Synthesize a waveform from a mel spectrogram with a neural vocoder.

`pmetal vocode` runs a BigVGAN generator (Lee et al., 2023) on Apple silicon. The released
checkpoints load unchanged, from the Hub or from a directory.

## Usage

```bash
pmetal vocode \
  --model <VOCODER> \
  --input <WAV_OR_MEL> \
  --output <OUT.wav> \
  [--chunk-frames <N>]
```

## Input

| Input | What happens |
|-------|--------------|
| A `.wav` file | Mixed down to mono, resampled to the model's rate if it differs, turned into a log-mel spectrogram by the model's own front end, and synthesized back. A round trip: it shows what the vocoder does to a recording. |
| A safetensors file | Holds the log-mel spectrogram of a text-to-speech model as `mel`, `[bins, frames]` (or `[batch, bins, frames]`); a file with one tensor needs no name. The bin count must be the model's. |

The output is a 16-bit mono WAV at the model's sampling rate, `frames * hop` samples long.

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model`, `-m` | required | A BigVGAN release (`nvidia/bigvgan_v2_24khz_100band_256x`) or a directory with its `config.json` and `bigvgan_generator.pt` or `model.safetensors` |
| `--input`, `-i` | required | The WAV file or mel spectrogram |
| `--output`, `-o` | required | The WAV file to write |
| `--chunk-frames` | `0` | Synthesize this many mel frames at a time; the output is the same and memory follows the chunk. `0` does it in one pass |

## Examples

```bash
# Re-synthesize a recording
pmetal vocode -m nvidia/bigvgan_v2_24khz_100band_256x -i speech.wav -o resynth.wav

# Vocode a long mel spectrogram with bounded memory
pmetal vocode -m ./bigvgan -i mel.safetensors -o speech.wav --chunk-frames 256
```

Hub downloads use the same cache, token, mirror and offline settings as every other command
(`HF_HOME`, `HF_TOKEN`, `HF_ENDPOINT`, `HF_HUB_OFFLINE`).
