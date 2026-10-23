//! `pmetal download`: fetch a model from the Hub with progress and checks.

use std::sync::Arc;

use crate::cli::hub::DownloadArgs;
use indicatif::{ProgressBar, ProgressStyle};
use pmetal_hub::{
    DownloadOptions, HubOptions, ProgressCallback, TransferProgress, Verification, WeightFormat,
    download_model_with, plan_model_download, split_quant_selector, verify_model_dir,
};

/// The download options a request stands for. A `:QUANT` suffix on the model
/// and `--quant` both select a GGUF quantization and so imply the GGUF format.
fn options_for(request: &DownloadArgs) -> (String, DownloadOptions) {
    let (id, suffix_quant) = split_quant_selector(&request.model);
    let quant = request
        .quant
        .clone()
        .or_else(|| suffix_quant.map(str::to_owned));
    let mut weights: WeightFormat = request.format.into();
    if quant.is_some() && weights == WeightFormat::Auto {
        weights = WeightFormat::Gguf;
    }
    let options = DownloadOptions {
        hub: HubOptions {
            offline: request.offline.then_some(true),
            ..HubOptions::default()
        },
        revision: request.revision.clone(),
        weights,
        gguf_quant: quant,
        verify: if request.verify_sha256 {
            Verification::Sha256
        } else {
            Verification::Size
        },
        ..DownloadOptions::default()
    };
    (id.to_owned(), options)
}

fn human_rate(bytes_per_sec: f64) -> String {
    format!(
        "{}/s",
        pmetal_core::format::human_size(bytes_per_sec as u64)
    )
}

fn human_duration(d: std::time::Duration) -> String {
    let secs = d.as_secs();
    match secs {
        0..=59 => format!("{secs}s"),
        60..=3599 => format!("{}m{:02}s", secs / 60, secs % 60),
        _ => format!("{}h{:02}m", secs / 3600, (secs % 3600) / 60),
    }
}

/// A byte progress bar with rate and ETA, and the callback that drives it from
/// a transfer's progress snapshots. Used by downloads and uploads.
pub(crate) fn transfer_bar() -> anyhow::Result<(ProgressBar, ProgressCallback)> {
    let bar = ProgressBar::new(0);
    bar.set_style(
        ProgressStyle::with_template("{spinner} [{bar:30}] {bytes}/{total_bytes}  {msg}")?
            .progress_chars("=> "),
    );
    let shown = bar.clone();
    let callback: ProgressCallback = Arc::new(move |p: &TransferProgress| {
        shown.set_length(p.total_bytes);
        shown.set_position(p.bytes_done);
        let rate = p.bytes_per_sec.map(human_rate).unwrap_or_default();
        let eta = p
            .eta
            .filter(|_| !p.finished)
            .map(|d| format!("ETA {}", human_duration(d)))
            .unwrap_or_default();
        let what = p.current_file.as_deref().filter(|f| *f == "committing");
        shown.set_message(format!(
            "{}/{} files  {rate}  {eta}  {}",
            p.files_done,
            p.total_files,
            what.unwrap_or_default()
        ));
    });
    Ok((bar, callback))
}

/// Run `pmetal download`.
pub(crate) async fn run_download(request: DownloadArgs) -> anyhow::Result<()> {
    let (model, mut options) = options_for(&request);

    if request.dry_run {
        let plan = plan_model_download(&model, &options).await?;
        println!("{} at commit {}", plan.repo_id, plan.commit);
        for file in &plan.model.files {
            println!(
                "  {:>10}  {}",
                pmetal_core::format::human_size(file.size),
                file.path
            );
        }
        println!(
            "{} files, {}",
            plan.model.files.len(),
            pmetal_core::format::human_size(plan.model.total_bytes())
        );
        return Ok(());
    }

    let (bar, callback) = transfer_bar()?;
    options.progress = Some(callback);

    let path = download_model_with(&model, &options).await;
    bar.finish_and_clear();
    let path = path?;

    let report = verify_model_dir(&path);
    println!("Model downloaded to: {}", path.display());
    println!(
        "  {} weight file(s), {}",
        report.weight_files.len(),
        pmetal_core::format::human_size(report.weight_bytes)
    );
    Ok(())
}

/// Run `pmetal upload`.
pub(crate) async fn run_upload(args: crate::cli::hub::UploadArgs) -> anyhow::Result<()> {
    tracing::info!(path = %args.path, repo = %args.repo, "Uploading model");
    let (bar, progress) = transfer_bar()?;
    let options = pmetal_hub::UploadOptions {
        private: args.private,
        revision: args.revision,
        commit_message: args.message,
        create_pr: args.create_pr,
        ignore: args.exclude,
        progress: Some(progress),
        ..Default::default()
    };
    let report = pmetal_hub::upload_model(&args.path, &args.repo, None, &options).await;
    bar.finish_and_clear();
    let report = report?;
    println!("Uploaded {} to {}", args.path, report.repo_url);
    if let Some(commit) = report.commit_url {
        println!("Commit: {commit}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::hub::WeightsArg;

    fn request(model: &str) -> DownloadArgs {
        DownloadArgs {
            model: model.to_owned(),
            revision: None,
            format: WeightsArg::Auto,
            quant: None,
            verify_sha256: false,
            dry_run: false,
            offline: false,
        }
    }

    #[test]
    fn a_quant_suffix_selects_gguf() {
        let (id, options) = options_for(&request("org/model-GGUF:Q4_K_M"));
        assert_eq!(id, "org/model-GGUF");
        assert_eq!(options.weights, WeightFormat::Gguf);
        assert_eq!(options.gguf_quant.as_deref(), Some("Q4_K_M"));
    }

    #[test]
    fn the_flag_beats_the_suffix_and_explicit_formats_stand() {
        let mut r = request("org/model:Q8_0");
        r.quant = Some("Q4_K_M".into());
        r.format = WeightsArg::Safetensors;
        let (_, options) = options_for(&r);
        assert_eq!(options.gguf_quant.as_deref(), Some("Q4_K_M"));
        assert_eq!(options.weights, WeightFormat::Safetensors);
    }

    #[test]
    fn flags_map_to_options() {
        let mut r = request("org/model");
        r.verify_sha256 = true;
        r.offline = true;
        r.revision = Some("v1".into());
        let (_, options) = options_for(&r);
        assert_eq!(options.verify, Verification::Sha256);
        assert_eq!(options.hub.offline, Some(true));
        assert_eq!(options.revision.as_deref(), Some("v1"));
    }

    #[test]
    fn durations_read_naturally() {
        use std::time::Duration;
        assert_eq!(human_duration(Duration::from_secs(5)), "5s");
        assert_eq!(human_duration(Duration::from_secs(125)), "2m05s");
        assert_eq!(human_duration(Duration::from_secs(7500)), "2h05m");
    }
}
