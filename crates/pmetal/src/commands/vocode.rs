//! `pmetal vocode`: a mel spectrogram (or a recording) to a waveform.

use std::path::Path;
use std::time::Instant;

use anyhow::Context;
use pmetal_core::jobs::VocodeSpec;
use pmetal_runtime::compat::Array;
use pmetal_vocoder::audio::{MelFrontEnd, Resampler, read_wav, to_pcm16, write_wav_pcm16};
use pmetal_vocoder::{BigVGAN, BigVGANConfig};

/// What the input file holds.
enum Input {
    /// A recording: re-synthesized through the model's own mel front end.
    Wav,
    /// A log-mel spectrogram, `[bins, frames]`.
    Mel,
}

fn input_kind(path: &Path) -> Input {
    let is_wav = path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("wav") || e.eq_ignore_ascii_case("wave"));
    if is_wav { Input::Wav } else { Input::Mel }
}

/// The log-mel spectrogram `[1, bins, frames]` the input stands for under
/// `config`.
fn mel_for(path: &Path, config: &BigVGANConfig) -> anyhow::Result<Array> {
    let mel = match input_kind(path) {
        Input::Wav => {
            let bytes = std::fs::read(path)
                .with_context(|| format!("--input: could not read '{}'", path.display()))?;
            let wav = read_wav(&bytes).with_context(|| {
                format!("--input: '{}' is not a usable WAV file", path.display())
            })?;
            let mut samples = wav.mono();
            anyhow::ensure!(
                !samples.is_empty(),
                "--input: '{}' holds no audio",
                path.display()
            );
            let rate = config.sampling_rate;
            if i64::from(wav.sample_rate) != i64::from(rate) {
                tracing::info!(
                    from = wav.sample_rate,
                    to = rate,
                    "Resampling the input to the model's rate"
                );
                let resampler = Resampler::new(wav.sample_rate as i32, rate)?;
                let wave = Array::from_f32_slice(&samples, &[samples.len() as i32]);
                let resampled = resampler.forward(&wave)?;
                samples = resampled
                    .as_dtype(pmetal_runtime::compat::Dtype::Float32)
                    .as_slice::<f32>()
                    .to_vec();
            }
            let wave = Array::from_f32_slice(&samples, &[samples.len() as i32]);
            MelFrontEnd::new(config)?.forward(&wave)?
        }
        Input::Mel => {
            let tensors = pmetal_vocoder::checkpoint::load_safetensors(path)
                .with_context(|| format!("--input: could not read '{}'", path.display()))?;
            let mut names: Vec<&String> = tensors.keys().collect();
            names.sort();
            let mel = match (tensors.get("mel"), names.as_slice()) {
                (Some(mel), _) => mel.clone(),
                (None, [only]) => tensors.get(only).expect("listed").clone(),
                (None, _) => anyhow::bail!(
                    "--input: '{}' holds {names:?}; name the log-mel spectrogram `mel`",
                    path.display()
                ),
            };
            match mel.ndim() {
                2 => mel.reshape(&[1, mel.dim(0), mel.dim(1)]),
                3 => mel,
                n => anyhow::bail!(
                    "--input: the mel spectrogram is {n}-dimensional {:?}, not [bins, frames]",
                    mel.shape()
                ),
            }
        }
    };
    anyhow::ensure!(
        mel.dim(1) == config.num_mels,
        "the input has {} mel bins and the model takes {}",
        mel.dim(1),
        config.num_mels
    );
    anyhow::ensure!(mel.dim(2) > 0, "the input has no frames");
    Ok(mel)
}

/// Run `pmetal vocode`.
pub(crate) async fn run_vocode(spec: VocodeSpec) -> anyhow::Result<()> {
    let model = BigVGAN::from_pretrained(&spec.model)
        .await
        .map_err(|e| anyhow::anyhow!("could not load the vocoder '{}': {e}", spec.model))?;
    let rate = model.config.sampling_rate;
    let mel = mel_for(Path::new(&spec.input), &model.config)?;
    let frames = mel.dim(2);

    let start = Instant::now();
    let audio = if spec.chunk_frames > 0 {
        model.synthesize_chunked(&mel, spec.chunk_frames as i32)?
    } else {
        model.forward(&mel)?
    };
    let samples = to_pcm16(&audio)?;
    pmetal_runtime::check_unobserved_error()
        .map_err(|e| anyhow::anyhow!("synthesis failed: {e}"))?;
    let elapsed = start.elapsed().as_secs_f64();

    if let Some(parent) = Path::new(&spec.output)
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("--output: could not create '{}'", parent.display()))?;
    }
    std::fs::write(&spec.output, write_wav_pcm16(&samples, rate as u32))
        .with_context(|| format!("--output: could not write '{}'", spec.output))?;

    let seconds = samples.len() as f64 / f64::from(rate);
    println!(
        "Wrote {} ({frames} frames, {seconds:.2} s of audio at {rate} Hz) in {elapsed:.2} s, {:.1}x real time",
        spec.output,
        seconds / elapsed.max(1e-9)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_runtime::compat::random;

    const TINY: &str = r#"{"resblock":"1","num_mels":8,"upsample_rates":[4,2],
        "upsample_kernel_sizes":[8,4],"upsample_initial_channel":8,
        "resblock_kernel_sizes":[3],"resblock_dilation_sizes":[[1,3]],
        "sampling_rate":8000,"n_fft":32,"hop_size":8,"win_size":32}"#;

    /// A tiny vocoder in a directory, as a release is laid out.
    fn tiny_release(dir: &Path) {
        random::seed(5);
        let config = BigVGANConfig::from_json(TINY).unwrap();
        let mut model = BigVGAN::new(config).unwrap();
        model.fuse_weight_norm();
        std::fs::write(dir.join("config.json"), TINY).unwrap();
        model
            .save_safetensors(&dir.join("model.safetensors"))
            .unwrap();
    }

    fn sine_wav(path: &Path, rate: u32, samples: usize) {
        let pcm: Vec<i16> = (0..samples)
            .map(|i| ((i as f32 * 0.3).sin() * 12000.0) as i16)
            .collect();
        std::fs::write(path, write_wav_pcm16(&pcm, rate)).unwrap();
    }

    fn spec(dir: &Path, input: &str) -> VocodeSpec {
        VocodeSpec {
            model: dir.to_str().unwrap().into(),
            input: dir.join(input).to_str().unwrap().into(),
            output: dir.join("out.wav").to_str().unwrap().into(),
            chunk_frames: 0,
        }
    }

    fn output(dir: &Path) -> pmetal_vocoder::audio::Wav {
        read_wav(&std::fs::read(dir.join("out.wav")).unwrap()).unwrap()
    }

    #[tokio::test]
    async fn a_recording_comes_back_as_a_wav_of_the_same_length_at_the_models_rate() {
        let dir = tempfile::tempdir().unwrap();
        tiny_release(dir.path());
        sine_wav(&dir.path().join("in.wav"), 8000, 800);
        run_vocode(spec(dir.path(), "in.wav")).await.unwrap();
        let wav = output(dir.path());
        assert_eq!((wav.sample_rate, wav.channels), (8000, 1));
        // 800 samples is 100 frames of hop 8, which the model turns back into 800.
        assert_eq!(wav.frames(), 800);
        pmetal_runtime::check_unobserved_error().unwrap();
    }

    #[tokio::test]
    async fn a_recording_at_another_rate_is_resampled_to_the_models() {
        let dir = tempfile::tempdir().unwrap();
        tiny_release(dir.path());
        sine_wav(&dir.path().join("in.wav"), 16_000, 1600);
        run_vocode(spec(dir.path(), "in.wav")).await.unwrap();
        let wav = output(dir.path());
        assert_eq!(wav.sample_rate, 8000);
        // 1600 samples at 16 kHz is 800 at 8 kHz.
        assert_eq!(wav.frames(), 800);
    }

    #[tokio::test]
    async fn a_mel_file_synthesizes_the_same_in_chunks() {
        let dir = tempfile::tempdir().unwrap();
        tiny_release(dir.path());
        random::seed(8);
        let mel = pmetal_runtime::compat::random::uniform_range(
            -4.0,
            1.0,
            &[8, 40],
            pmetal_runtime::compat::Dtype::Float32,
        );
        Array::save_safetensors(
            dir.path().join("mel.safetensors").to_str().unwrap(),
            &[("mel", &mel)],
        )
        .unwrap();

        run_vocode(spec(dir.path(), "mel.safetensors"))
            .await
            .unwrap();
        let whole = output(dir.path());
        assert_eq!(whole.frames(), 40 * 8);

        let mut chunked = spec(dir.path(), "mel.safetensors");
        chunked.chunk_frames = 7;
        run_vocode(chunked).await.unwrap();
        let pieces = output(dir.path());
        assert_eq!(pieces.frames(), whole.frames());
        let worst = whole
            .samples
            .iter()
            .zip(&pieces.samples)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-3, "chunked synthesis differs by {worst}");
        pmetal_runtime::check_unobserved_error().unwrap();
    }

    #[tokio::test]
    async fn an_input_the_model_cannot_take_is_refused_with_what_it_found() {
        let dir = tempfile::tempdir().unwrap();
        tiny_release(dir.path());
        let wrong =
            pmetal_runtime::compat::ops::zeros(&[5, 10], pmetal_runtime::compat::Dtype::Float32);
        Array::save_safetensors(
            dir.path().join("mel.safetensors").to_str().unwrap(),
            &[("mel", &wrong)],
        )
        .unwrap();
        let err = run_vocode(spec(dir.path(), "mel.safetensors"))
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("5 mel bins") && err.contains("takes 8"),
            "{err}"
        );

        std::fs::write(dir.path().join("bad.wav"), b"RIFFxxxxWAVE").unwrap();
        let err = format!(
            "{:#}",
            run_vocode(spec(dir.path(), "bad.wav")).await.unwrap_err()
        );
        assert!(err.contains("bad.wav") && err.contains("WAV"), "{err}");
        assert!(
            !dir.path().join("out.wav").exists(),
            "nothing is written on failure"
        );
    }
}
