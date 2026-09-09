"""Bootstrap guard for `make setup`.

Plain Python so it works identically on Windows cmd.exe, PowerShell and POSIX
shells -- none of which agree on `test -f` / `{ }` grouping syntax.
"""

from __future__ import annotations

import pathlib
import sys


def main() -> int:
    if not pathlib.Path("requirements.lock").exists():
        print(
            "requirements.lock is missing.\n"
            "Run this once:  make setup-dev && make lock\n"
            "Then commit requirements.lock and re-run:  make setup"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
