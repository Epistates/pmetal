//! `pmetal verify`: is this model directory complete?

use std::path::PathBuf;

use pmetal_hub::{ModelDirReport, verify_model_dir};
use serde_json::{Value, json};

use crate::cli::hub::VerifyArgs;

/// The directory `model` names: a path, or a Hub ID whose complete copy is in
/// the Hub cache.
fn locate(model: &str) -> anyhow::Result<PathBuf> {
    let path = PathBuf::from(model);
    if path.exists() {
        return Ok(path);
    }
    if pmetal_hub::is_hf_id(model) {
        return pmetal_hub::find_cached_model(model).ok_or_else(|| {
            anyhow::anyhow!(
                "'{model}' has no complete copy in the Hub cache; fetch it with \
                 `pmetal download {model}`"
            )
        });
    }
    anyhow::bail!("'{model}' is neither a directory nor a Hub model ID (org/name)")
}

/// The report as JSON: what a script reads.
fn report_json(model: &str, path: &std::path::Path, report: &ModelDirReport) -> Value {
    let safetensors = report.safetensors.as_ref();
    json!({
        "model": model,
        "path": path.display().to_string(),
        "complete": report.is_complete(),
        "loadable": report.is_loadable_language_model(),
        "format": report.format.map(|f| format!("{f:?}")),
        "weight_files": report.weight_files.len(),
        "weight_bytes": report.weight_bytes,
        "tensors": safetensors.map(|s| s.tensors),
        "problems": report.problems,
        "warnings": report.warnings,
    })
}

/// The report for a person.
fn report_text(model: &str, path: &std::path::Path, report: &ModelDirReport) -> String {
    let mut out = format!("{model}\n  at {}\n", path.display());
    match report.format {
        Some(format) => out.push_str(&format!(
            "  {format:?} weights: {} file(s), {}",
            report.weight_files.len(),
            pmetal_hub::fit::format_bytes(report.weight_bytes)
        )),
        None => out.push_str("  no weights"),
    }
    if let Some(summary) = &report.safetensors {
        out.push_str(&format!(", {} tensors", summary.tensors));
    }
    out.push('\n');
    for problem in &report.problems {
        out.push_str(&format!("  problem: {problem}\n"));
    }
    for warning in &report.warnings {
        out.push_str(&format!("  warning: {warning}\n"));
    }
    out.push_str(if report.is_loadable_language_model() {
        "  complete: a language model can load it\n"
    } else if report.is_complete() {
        "  complete, but see the warnings\n"
    } else {
        "  INCOMPLETE\n"
    });
    out
}

/// Run `pmetal verify`. An incomplete or damaged model is an error (exit
/// status 1) after the report is printed.
pub(crate) fn run_verify(args: VerifyArgs) -> anyhow::Result<()> {
    let path = locate(&args.model)?;
    let report = verify_model_dir(&path);
    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&report_json(&args.model, &path, &report))?
        );
    } else {
        print!("{}", report_text(&args.model, &path, &report));
    }
    anyhow::ensure!(
        report.is_complete(),
        "{} is incomplete or damaged ({} problem(s))",
        args.model,
        report.problems.len()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{Dtype, TensorView, serialize};

    fn write_weights(path: &std::path::Path) {
        let data = [0u8; 16];
        let view = TensorView::new(Dtype::F32, vec![2, 2], &data).unwrap();
        std::fs::write(path, serialize([("w", view)], None).unwrap()).unwrap();
    }

    fn args(model: &std::path::Path, json: bool) -> VerifyArgs {
        VerifyArgs {
            model: model.to_str().unwrap().into(),
            json,
        }
    }

    #[test]
    fn a_complete_directory_verifies() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), "{}").unwrap();
        write_weights(&dir.path().join("model.safetensors"));
        run_verify(args(dir.path(), false)).unwrap();
        let report = verify_model_dir(dir.path());
        let json = report_json("m", dir.path(), &report);
        assert_eq!(json["complete"], true);
        assert_eq!(json["tensors"], 1);
        assert!(report_text("m", dir.path(), &report).contains("a language model can load it"));
    }

    #[test]
    fn a_missing_shard_is_reported_and_fails() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), "{}").unwrap();
        write_weights(&dir.path().join("model-00001-of-00002.safetensors"));
        std::fs::write(
            dir.path().join("model.safetensors.index.json"),
            r#"{"weight_map":{"a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors"}}"#,
        )
        .unwrap();
        let err = run_verify(args(dir.path(), true)).unwrap_err().to_string();
        assert!(
            err.contains("incomplete") && err.contains("1 problem"),
            "{err}"
        );
        let report = verify_model_dir(dir.path());
        let text = report_text("m", dir.path(), &report);
        assert!(
            text.contains("model-00002-of-00002.safetensors is missing"),
            "{text}"
        );
        assert!(text.contains("INCOMPLETE"), "{text}");
    }

    #[test]
    fn something_that_is_not_a_model_says_what_it_is() {
        let err = locate("definitely-not-a-model").unwrap_err().to_string();
        assert!(
            err.contains("neither a directory nor a Hub model ID"),
            "{err}"
        );
    }
}
