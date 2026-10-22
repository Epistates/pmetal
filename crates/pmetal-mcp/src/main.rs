#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // All logging MUST go to stderr — stdout is the MCP protocol channel.
    pmetal_core::logging::init(
        pmetal_core::logging::LogOptions::new("mcp")
            .output(pmetal_core::logging::LogOutput::Stderr)
            .default_filter("error"),
    );

    tracing::info!("starting pmetal-mcp server");
    pmetal_mcp::run_stdio().await
}
