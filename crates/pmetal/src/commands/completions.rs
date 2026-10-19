//! `pmetal completions`: the shell's tab completion for every command and flag.

use clap::CommandFactory;
use clap_complete::Shell;

use crate::cli::Cli;

/// Write the completion script for `shell` to `out`.
pub(crate) fn write_completions(shell: Shell, out: &mut dyn std::io::Write) {
    clap_complete::generate(shell, &mut Cli::command(), "pmetal", out);
}

/// Run `pmetal completions <shell>`: the script on standard output.
pub(crate) fn run_completions(shell: Shell) {
    write_completions(shell, &mut std::io::stdout());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn script(shell: Shell) -> String {
        let mut out = Vec::new();
        write_completions(shell, &mut out);
        String::from_utf8(out).unwrap()
    }

    /// A script knows the commands and the flags of this build, not a list
    /// somebody typed: a flag added to a command is in the next script.
    #[test]
    fn every_shell_completes_commands_and_flags() {
        for shell in [
            Shell::Bash,
            Shell::Zsh,
            Shell::Fish,
            Shell::Elvish,
            Shell::PowerShell,
        ] {
            let text = script(shell);
            for word in ["train", "verify", "completions", "max-steps", "log-events"] {
                assert!(text.contains(word), "{shell}: no `{word}`");
            }
        }
        assert!(script(Shell::Bash).contains("_pmetal"));
    }
}
