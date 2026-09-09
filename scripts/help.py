"""Print available Makefile targets.

Cross-platform replacement for the original `grep | awk` pipeline, which
does not exist on stock Windows.
"""

from __future__ import annotations

import pathlib
import re

PATTERN = re.compile(r"^([a-zA-Z][a-zA-Z0-9_-]*):.*?## (.*)$")


def main() -> None:
    for line in pathlib.Path("Makefile").read_text(encoding="utf-8").splitlines():
        m = PATTERN.match(line)
        if m:
            print(f"  {m.group(1):<16} {m.group(2)}")


if __name__ == "__main__":
    main()
