# pmetal-vocoder

BigVGAN neural vocoder for Apple Silicon: mel spectrogram to waveform, with the
released checkpoints loading as they are, and the training recipe to fine-tune
them.

## Overview

BigVGAN (Lee et al., 2023) is a GAN vocoder built from transposed-convolution
upsamplers, each followed by a bank of parallel residual blocks whose
activations are *anti-aliased* Snake functions: a periodic nonlinearity that
models harmonics, evaluated at twice the sample rate and low-pass filtered so
the harmonics it creates cannot fold back as aliasing.

```text
mel [B, n_mels, frames]
  -> Conv1d(7)
  -> [ ConvTranspose1d(rate)  ->  mean of parallel AMP blocks ] x stages
  -> anti-aliased Snake -> Conv1d(7) -> tanh or clamp
  -> audio [B, 1, frames * hop_size]
```

Every released configuration is a preset (`BigVGANConfig::v2_24khz_100band`,
`v2_44khz_128band_512x`, `base_24khz_100band`, ...), and a release's
`config.json` deserialises as is.

It computes on `pmetal-runtime` arrays and resolves Hugging Face ids through
`pmetal-hub`; the `pmetal` CLI exposes it as `pmetal vocode`.

## Installation

```toml
[dependencies]
pmetal-vocoder = "0.7"
```

## Usage

A generator with random weights and a small configuration runs without any download. The
output is `frames * hop_size` samples long:

```rust
use pmetal_runtime::compat::{Dtype, ops};
use pmetal_vocoder::{BigVGAN, BigVGANConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = BigVGANConfig::from_json(
        r#"{"resblock": "1", "num_mels": 8, "hop_size": 8, "n_fft": 32, "win_size": 32,
            "sampling_rate": 16000, "upsample_rates": [4, 2], "upsample_kernel_sizes": [8, 4],
            "upsample_initial_channel": 16, "resblock_kernel_sizes": [3, 5],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 2, 3]], "use_tanh_at_final": false,
            "use_bias_at_final": false, "activation": "snakebeta", "snake_logscale": true,
            "fmin": 0, "fmax": null}"#,
    )?;
    let vocoder = BigVGAN::new(config)?;

    // mel: [batch, mel_bins, frames] -> audio: [batch, 1, samples]
    let mel = ops::zeros(&[1, 8, 5], Dtype::Float32);
    let audio = vocoder.forward(&mel)?;
    assert_eq!(audio.shape(), [1, 1, 5 * 8]);
    Ok(())
}
```

Synthesis from a released model (a Hugging Face id, or a local directory with
`config.json` and `bigvgan_generator.pt` or `model.safetensors`):

```rust,no_run
use pmetal_runtime::compat::Array;
use pmetal_vocoder::{BigVGAN, Result};

/// `mel` is `[batch, mel_bins, frames]`; the result is `[batch, 1, samples]`.
async fn vocode(mel: &Array) -> Result<Array> {
    let vocoder = BigVGAN::from_pretrained("nvidia/bigvgan_v2_24khz_100band_256x").await?;
    vocoder.forward(mel)
}
```

The mel spectrogram a model expects comes from its own front end, which
reproduces the reference's (reflect padding, Hann window, `sqrt(re^2 + im^2 + 1e-9)`
magnitude, Slaney filterbank, `log(max(x, 1e-5))`):

```rust,no_run
use pmetal_runtime::compat::Array;
use pmetal_vocoder::audio::MelFrontEnd;
use pmetal_vocoder::{BigVGAN, Result};

/// Re-synthesise `wav` (`[samples]`, at the model's sampling rate).
fn copy_synthesis(vocoder: &BigVGAN, wav: &Array) -> Result<Array> {
    let mel = MelFrontEnd::new(&vocoder.config)?.forward(wav)?;
    vocoder.forward(&mel)
}
```

Long or live input is synthesised as a stream. The generator is convolutional,
so a piece given enough left context and held back until enough right context
has arrived reproduces the one-shot waveform exactly; memory follows the piece,
not the recording:

```rust,no_run
use pmetal_runtime::compat::Array;
use pmetal_vocoder::{BigVGAN, Result, StreamingSynthesizer};

fn stream(vocoder: &BigVGAN, chunks: &[Array]) -> Result<Vec<Array>> {
    let mut stream = StreamingSynthesizer::new(vocoder, 1);
    let mut audio = Vec::new();
    for chunk in chunks {
        // Each chunk is `[1, mel_bins, frames]`; audio trails by
        // `stream.latency_frames()` frames.
        audio.extend(stream.push(chunk)?);
    }
    audio.extend(stream.finish()?);
    Ok(audio)
}
```

### Fine-tuning

```rust,no_run
use pmetal_runtime::compat::Array;
use pmetal_vocoder::discriminator::{BigVGANDiscriminator, DiscriminatorConfig};
use pmetal_vocoder::train::{GanTrainer, TrainingConfig};
use pmetal_vocoder::{BigVGAN, Result};
use std::path::Path;

fn fine_tune(dir: &Path, batches: &[(Array, Array)]) -> Result<()> {
    let json = std::fs::read_to_string(dir.join("config.json"))?;
    let generator = BigVGAN::from_dir(dir)?;
    let mut discriminator = BigVGANDiscriminator::new(&DiscriminatorConfig::from_json(&json)?)?;
    discriminator.load_checkpoint(&dir.join("bigvgan_discriminator_optimizer.pt"))?;

    let mut trainer = GanTrainer::new(generator, discriminator, TrainingConfig::from_json(&json)?)?;
    for (mel, wav) in batches {
        // mel: [batch, mel_bins, frames]; wav: [batch, 1, frames * hop_size]
        let metrics = trainer.step(mel, wav)?;
        println!("step {}: mel error {}", metrics.step, metrics.mel_error);
    }
    trainer.save_checkpoint(&dir.join("fine_tuned.safetensors"))
}
```

## What is here

| Module | Contents |
|--------|----------|
| `generator` | `BigVGAN`: forward, `generate`, streaming and chunked synthesis, receptive field, loading and saving |
| `nn` | `Conv1d` / `ConvTranspose1d` / `Conv2d` with weight norm, Snake and SnakeBeta, anti-aliased `Activation1d` with its fused kernel, `AMPBlock` (both layouts), Kaiser-sinc and `firwin2` filters |
| `discriminator` | Multi-period, multi-resolution, multi-band and multi-scale sub-band CQT discriminators |
| `loss` | Least-squares GAN, feature matching, single-scale and multi-scale mel losses |
| `audio` | STFT and iSTFT, mel filterbank and the BigVGAN front end, CQT, sample-rate conversion |
| `checkpoint` | `.pt` (restricted unpickler) and safetensors readers into a strict `StateDict` |
| `train` | `GanTrainer`: the adversarial training loop, learning-rate schedule, gradient clipping, checkpoints |

## Correctness

Every component is checked against values computed by the reference
implementation (and librosa, torchaudio and nnAudio where it relies on them),
not against shapes. Released checkpoints reproduce the reference waveform on a
fixed mel to 2e-5 (SNR above 100 dB) for the base and v2 24 kHz models, and the
released discriminators agree with the reference to 1e-5, including the CQT
discriminator's derived buffers.

Layers are channels-last (`[batch, time, channels]`), MLX's native layout;
`BigVGAN::forward` keeps the reference's channels-first interface.

## Performance

`BigVGAN::prepare_for_inference` (called by `from_dir` and `from_pretrained`)
folds weight normalisation into the weights and runs each anti-aliased
activation as one fused Metal kernel instead of about twenty tiny ones.
Provisional numbers on an M4 Max with other work loading the machine, random
weights, best of three:

| Model | Audio | Op-by-op | Fused (float32) | Fused (float16) |
|-------|-------|----------|-----------------|-----------------|
| BigVGAN-v2 24 kHz | 1.0 s | 68 ms | 34 ms | 29 ms |
| BigVGAN-v2 24 kHz | 5.0 s | 469 ms | 168 ms | 133 ms |
| BigVGAN-base 24 kHz | 1.0 s | 48 ms | 14 ms | 11 ms |

Float16 is the half-precision choice (SNR 43 dB against the float32 waveform;
bfloat16 gives 26 dB). The fused kernel has no gradient: training uses the
differentiable op path, and `prepare_for_training` switches back. Run
`cargo run --release -p pmetal-vocoder --example synth_bench` to measure on
your machine.

## Not here

- Spectral norm in the discriminators (`use_spectral_norm`); no released model
  uses it, and loading says so.
- The CUDA-only options of the reference.

## Feature flags

None.

## Requirements

- Rust 1.95 or later
- macOS 14 or later on Apple Silicon

## License

MIT OR Apache-2.0
