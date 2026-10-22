# pmetal completions

Print the shell completion script for pmetal.

The script is generated from the command line parser itself, so every command and flag of the
installed build is completed, and a flag added in a later version is in the next script.

## Usage

```bash
pmetal completions <SHELL>
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `<SHELL>` | required | `bash`, `zsh`, `fish`, `elvish` or `powershell` |

## Examples

```bash
# zsh: put the script on your fpath
pmetal completions zsh > ~/.zfunc/_pmetal

# bash
pmetal completions bash > /usr/local/etc/bash_completion.d/pmetal

# fish
pmetal completions fish > ~/.config/fish/completions/pmetal.fish
```

Restart the shell (or run `compinit` for zsh) to pick it up.

## See Also

- [Installation](/installation/): Installing pmetal
