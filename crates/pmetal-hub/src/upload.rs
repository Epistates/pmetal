//! Model uploading to HuggingFace Hub.

use pmetal_core::Result;
use std::path::Path;

/// Upload a model to HuggingFace Hub.
pub async fn upload_model<P: AsRef<Path>>(
    _model_path: P,
    _repo_id: &str,
    _token: &str,
) -> Result<()> {
    // TODO: Implement model upload
    tracing::info!("Model upload not yet implemented");
    Ok(())
}
