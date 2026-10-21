"""``python -m pmetal ...``: the ``pmetal`` command line, the one jobs run."""

from __future__ import annotations

import os
import sys

from ._jobs import CliNotFoundError, find_cli


def main() -> None:
    """Replace this process with the ``pmetal`` command line, given the same arguments."""
    try:
        cli = find_cli()
    except CliNotFoundError as error:
        sys.exit(f"python -m pmetal: {error}")
    os.execv(cli, [str(cli), *sys.argv[1:]])  # noqa: S606 - the command line is the point


if __name__ == "__main__":
    main()
