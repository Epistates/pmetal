//! Test support: what the real argument parser says about a command the
//! interface would launch.

use clap::Parser;

use crate::cli::{Cli, Commands};
use crate::tui::jobs::LaunchPlan;

/// The command `plan` stands for, read by `Cli::try_parse_from`: the parser the
/// process it launches will use.
pub fn parse(plan: &LaunchPlan) -> Result<Commands, String> {
    let mut argv = vec!["pmetal".to_string(), plan.subcommand.clone()];
    argv.extend(plan.argv.iter().cloned());
    Cli::try_parse_from(&argv)
        .map(|cli| cli.command)
        .map_err(|e| e.to_string())
}

/// `plan` is a command `pmetal` accepts.
#[track_caller]
pub fn assert_parses(plan: &LaunchPlan) -> Commands {
    parse(plan).unwrap_or_else(|e| {
        panic!(
            "`{}` is not a command pmetal accepts:\n{e}",
            plan.command_line()
        )
    })
}
