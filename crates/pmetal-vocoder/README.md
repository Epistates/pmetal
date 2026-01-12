# pmetal-vocoder

BigVGAN neural vocoder for text-to-speech synthesis.

## Overview

This crate provides a Rust implementation of BigVGAN, a high-fidelity neural vocoder that converts mel spectrograms to audio waveforms. It's designed for integration with speech synthesis systems.

## Architecture

BigVGAN uses an anti-aliased multi-periodicity architecture:

```
Mel Spectrogram → Conv1d → [AMP Blocks × N] → Conv1d → Audio Waveform
                              ↓
                     Snake Activations
                     Anti-aliased Upsampling
                     Multi-period Processing
```

## Features

- **High Fidelity**: 24kHz+ audio generation
- **256× Upsampling**: Mel frames to audio samples
- **Snake Activations**: Learnable periodic activations
- **Anti-Aliased Upsampling**: Reduced aliasing artifacts
- **Multi-Resolution Discriminators**: MPD, CQT-D for training

## Components

### Generator
- Anti-Aliased Multi-Periodicity (AMP) blocks
- Snake/SnakeBeta activations for periodic signals
- Configurable upsampling ratios

### Discriminators (for training)
- **MPD**: Multi-Period Discriminator
- **MSD**: Multi-Scale Discriminator
- **CQT-D**: Constant-Q Transform Discriminator

### Audio Processing
- STFT/iSTFT for spectral analysis
- Mel filterbank computation
- Resampling utilities

## Usage

```rust
use pmetal_vocoder::{BigVGANGenerator, VocoderConfig};

// Load vocoder
let config = VocoderConfig::default();
let vocoder = BigVGANGenerator::new(config)?;

// Convert mel spectrogram to audio
let mel = /* mel spectrogram [batch, mel_bins, frames] */;
let audio = vocoder.forward(&mel)?;
// audio: [batch, 1, samples]
```

## Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `upsample_rates` | Upsampling factors | [8, 8, 2, 2] |
| `upsample_kernel_sizes` | Kernel sizes | [16, 16, 4, 4] |
| `resblock_kernel_sizes` | ResBlock kernels | [3, 7, 11] |
| `resblock_dilation_sizes` | Dilation rates | [[1,3,5], ...] |

## Modules

| Module | Description |
|--------|-------------|
| `generator` | BigVGAN generator architecture |
| `discriminator` | Training discriminators |
| `nn` | Custom neural network layers |
| `audio` | Audio processing utilities |
| `loss` | Training loss functions |

## Status

This crate is under active development. Core architecture is implemented but some utilities are pending completion.

## License

MIT OR Apache-2.0
