"""Compress the LLM response cache for committing.

Uses Python's stdlib gzip module rather than the `gzip` CLI, which is not
installed by default on Windows.
"""

from __future__ import annotations

import gzip
import pathlib
import shutil
import sys

SRC = pathlib.Path("artifacts/llm_cache.jsonl")
DST = pathlib.Path("artifacts/llm_cache.jsonl.gz")


def main() -> int:
    if not SRC.exists():
        print(f"{SRC} not found -- nothing to freeze.")
        return 1
    DST.parent.mkdir(parents=True, exist_ok=True)
    with SRC.open("rb") as f_in, gzip.open(DST, "wb", compresslevel=9) as f_out:
        shutil.copyfileobj(f_in, f_out)
    size_kb = DST.stat().st_size / 1024
    print(f"wrote {DST} ({size_kb:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
