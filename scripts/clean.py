"""Remove tool caches. Cross-platform equivalent of `rm -rf`, which does not
exist on stock Windows.
"""

from __future__ import annotations

import pathlib
import shutil

TARGETS = [".pytest_cache", ".ruff_cache", ".mypy_cache"]


def main() -> None:
    for t in TARGETS:
        shutil.rmtree(t, ignore_errors=True)
    for p in pathlib.Path(".").rglob("__pycache__"):
        shutil.rmtree(p, ignore_errors=True)


if __name__ == "__main__":
    main()
