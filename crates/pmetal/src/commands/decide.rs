//! `pmetal decide`: answer one `/v1/systemone` request with a decision model.

use std::io::Read;
use std::time::Instant;

use pmetal_models::decision::DecisionModel;

/// Load the decision model `model_id`, answer the request body in
/// `request_path` (`-` reads stdin), and print the response body.
pub(crate) async fn run_decide(
    model_id: &str,
    request_path: &str,
    max_length: usize,
    compact: bool,
) -> anyhow::Result<()> {
    let request_text = if request_path == "-" {
        let mut text = String::new();
        std::io::stdin().read_to_string(&mut text)?;
        text
    } else {
        std::fs::read_to_string(request_path)
            .map_err(|e| anyhow::anyhow!("cannot read request {request_path}: {e}"))?
    };
    let request: serde_json::Value = serde_json::from_str(&request_text)
        .map_err(|e| anyhow::anyhow!("request is not JSON: {e}"))?;
    // Refuse a malformed request before spending a model load on it.
    pmetal_models::decision::systemone::validate_request(&request)?;

    let model_path = pmetal_hub::resolve_model_path(model_id, None, None).await?;
    if !pmetal_models::decision::is_decision_model(&model_path) {
        anyhow::bail!(
            "{model_id} is not a decision model: it has no joint_head_config.json and \
             joint_head.safetensors"
        );
    }
    let started = Instant::now();
    let mut model = DecisionModel::load(&model_path)?;
    tracing::info!(
        "Loaded decision model {} in {:.1}s",
        model_path.display(),
        started.elapsed().as_secs_f64()
    );

    let started = Instant::now();
    let response = model.systemone(&request, max_length)?;
    tracing::info!(
        "Answered in {:.0} ms ({} input tokens)",
        started.elapsed().as_secs_f64() * 1e3,
        response["usage"]["input_tokens"]
    );

    let text = if compact {
        serde_json::to_string(&response)?
    } else {
        serde_json::to_string_pretty(&response)?
    };
    println!("{text}");
    Ok(())
}
