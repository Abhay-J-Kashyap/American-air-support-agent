"""Ingest the raw Customer Support on Twitter CSV down to one brand's slice.

Two-pass design, and the two passes can't be collapsed into one:

Pass 1 builds a lightweight id -> parent-id graph over the WHOLE file. A
customer's opening tweet doesn't know yet whether its thread will ever reach
the brand -- that's only knowable once the full reply graph exists. So we
have to see everything before we can decide what to keep.

Pass 2 re-reads the CSV and keeps only rows whose tweet_id fell into the
brand-connected set found in pass 1.

Both passes are chunked; peak memory is the id graph (a few numeric columns
for ~3M rows, tens of MB) plus the brand's own slice (the only thing fully
materialized with text), not the whole 500MB file.
"""

from __future__ import annotations

import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from aa_agent.pii import scrub

RAW_COLUMNS = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
]

# response_tweet_id (children, as declared by the parent) and
# in_response_to_tweet_id (parent, as declared by the child) are supposed to
# mirror each other but do not always agree in the real dataset -- that's
# the kind of noise the brief calls "messy real-world". We treat
# in_response_to_tweet_id as the single source of truth and derive children
# by inverting it, rather than trusting both directions and reconciling
# disagreements. See DECISIONS.md #16.
_INDEX_COLUMNS = ["tweet_id", "author_id", "in_response_to_tweet_id"]

_MULTIPART_SUFFIX_RE = re.compile(r"\s*\(\d+/\d+\)\s*$")

# twcs.csv uses Twitter's native timestamp format, not ISO-8601.
TWITTER_TS_FORMAT = "%a %b %d %H:%M:%S %z %Y"


@dataclass(frozen=True)
class IngestStats:
    raw_rows: int
    brand_connected_ids: int
    kept_rows: int
    duplicate_ids_dropped: int
    undated_rows_dropped: int
    multipart_merges: int
    training_rows: int
    eval_rows: int


def load_id_index(csv_path: Path, chunksize: int = 100_000) -> pd.DataFrame:
    """Pass 1: tweet_id / author_id / parent-id for every row, nothing else.

    Deliberately narrow. Loading `text` here for 3M rows is the difference
    between a ~60MB working set and something that pressures an 8GB laptop.
    """
    chunks = []
    for chunk in pd.read_csv(
        csv_path,
        usecols=_INDEX_COLUMNS,
        dtype={"tweet_id": "int64", "author_id": "string"},
        chunksize=chunksize,
    ):
        chunk["in_response_to_tweet_id"] = pd.to_numeric(
            chunk["in_response_to_tweet_id"], errors="coerce"
        ).astype("Int64")
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)


def find_brand_connected_ids(index_df: pd.DataFrame, brand_author_id: str) -> frozenset[int]:
    """BFS out from every brand-authored tweet, walking both to parents and
    to children, over the whole reply graph. Returns every tweet_id in any
    thread that touches the brand anywhere.

    Tolerates dangling references (a parent id that no row in the file
    actually has) by construction: such an id is visited as a bare node with
    no further edges, not looked up and crashed on.
    """
    brand_ids = set(index_df.loc[index_df["author_id"] == brand_author_id, "tweet_id"])
    if not brand_ids:
        raise ValueError(
            f"no rows found with author_id == {brand_author_id!r}. "
            "Check config.project.brand matches the dataset's exact author_id string."
        )

    parent_of: dict[int, int | None] = {}
    children_of: dict[int, set[int]] = defaultdict(set)
    for tid, pid in zip(index_df["tweet_id"], index_df["in_response_to_tweet_id"], strict=True):
        pid_int = int(pid) if pd.notna(pid) else None
        parent_of[int(tid)] = pid_int
        if pid_int is not None:
            children_of[pid_int].add(int(tid))

    visited: set[int] = set()
    frontier: deque[int] = deque(brand_ids)
    while frontier:
        tid = frontier.popleft()
        if tid in visited:
            continue
        visited.add(tid)
        parent = parent_of.get(tid)
        if parent is not None and parent not in visited:
            frontier.append(parent)
        for child in children_of.get(tid, ()):
            if child not in visited:
                frontier.append(child)

    return frozenset(visited)


def _parse_created_at(raw: pd.Series) -> pd.Series:
    """Parse the dataset's timestamps.

    twcs.csv uses Twitter's native format ("Tue Oct 31 22:10:47 +0000 2017"),
    not ISO-8601. Without an explicit format pandas falls back to dateutil
    and parses each of ~88k values individually -- 3x slower, and it emits a
    UserWarning on every run.

    Anything the strict format rejects gets a second pass through the
    inferring parser rather than going straight to NaT, so a stray
    differently-formatted row is recovered instead of silently dropped by
    the undated-rows filter. Only values that fail BOTH are counted as
    undated.
    """
    parsed = pd.to_datetime(raw, format=TWITTER_TS_FORMAT, errors="coerce", utc=True)
    missing = parsed.isna() & raw.notna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(raw[missing], errors="coerce", utc=True)
    return parsed


def stream_filter_csv(
    csv_path: Path, keep_ids: frozenset[int], chunksize: int = 100_000
) -> tuple[pd.DataFrame, int, int]:
    """Pass 2: re-read the raw file, keep only rows in `keep_ids`.

    Returns (df, duplicate_ids_dropped, unparseable_timestamps_dropped).

    Both classes of bad row are dropped HERE rather than defended against
    downstream, because both are silent corrupters rather than crashers:

    * Duplicate tweet_ids make `set_index("tweet_id")` non-unique, so a
      later `.loc[parent_id]` returns a DataFrame instead of a Series and
      the multipart merge dies with pandas' "truth value of a Series is
      ambiguous" -- a confusing error a long way from its cause.
    * Unparseable timestamps become NaT, and `NaT <= cutoff` is False, so
      those rows land in the EVAL window without warning. Silently
      polluting the held-out set is the worst outcome available here.
    """
    kept = []
    for chunk in pd.read_csv(csv_path, usecols=RAW_COLUMNS, chunksize=chunksize):
        kept.append(chunk[chunk["tweet_id"].isin(keep_ids)])
    df = pd.concat(kept, ignore_index=True)
    df["in_response_to_tweet_id"] = pd.to_numeric(
        df["in_response_to_tweet_id"], errors="coerce"
    ).astype("Int64")
    df["created_at"] = _parse_created_at(df["created_at"])

    before = len(df)
    df = df.drop_duplicates(subset="tweet_id", keep="first")
    dupes_dropped = before - len(df)

    before = len(df)
    df = df[df["created_at"].notna()]
    undated_dropped = before - len(df)

    df = df.sort_values("created_at").reset_index(drop=True)
    return df, dupes_dropped, undated_dropped


def join_multipart_replies(df: pd.DataFrame, brand_author_id: str) -> tuple[pd.DataFrame, int]:
    """Merge a brand reply split across consecutive tweets ("...(1/2)" then
    a same-author follow-up replying to it) back into one logical message.

    The merged row keeps the SECOND part's tweet_id and points at the FIRST
    part's parent -- not the first part's id. Anything downstream (e.g. the
    customer's next reply) points at the last part actually posted, so
    collapsing the pair has to preserve that edge or the thread graph breaks.

    Known edge case, accepted rather than handled: if some other row's
    in_response_to_tweet_id pointed at the first part specifically (rare --
    would mean a reply arrived in the few seconds between the two halves
    being posted), that row becomes dangling after the merge. Dangling
    parents are already tolerated everywhere else in this module.
    """
    by_id = df.set_index("tweet_id", drop=False)
    if not by_id.index.is_unique:
        # stream_filter_csv already dedupes; this catches a caller that
        # skipped it. Failing here with a named cause beats failing three
        # lines later with pandas' "truth value of a Series is ambiguous".
        raise ValueError(
            "duplicate tweet_id values present -- deduplicate before merging "
            "multipart replies (stream_filter_csv does this)."
        )
    to_drop: set[int] = set()
    merges = 0

    brand_rows = df[df["author_id"] == brand_author_id].sort_values("created_at")
    for _, part2 in brand_rows.iterrows():
        parent_id = part2["in_response_to_tweet_id"]
        if pd.isna(parent_id) or int(parent_id) not in by_id.index:
            continue
        part1 = by_id.loc[int(parent_id)]
        if (
            part1["author_id"] != brand_author_id
            or int(part1["tweet_id"]) in to_drop
            or not _MULTIPART_SUFFIX_RE.search(str(part1["text"]))
        ):
            continue

        merged_text = (
            _MULTIPART_SUFFIX_RE.sub("", str(part1["text"])).rstrip() + " " + str(part2["text"])
        )
        by_id.loc[int(part2["tweet_id"]), "text"] = merged_text.strip()
        by_id.loc[int(part2["tweet_id"]), "in_response_to_tweet_id"] = part1[
            "in_response_to_tweet_id"
        ]
        to_drop.add(int(part1["tweet_id"]))
        merges += 1

    result = by_id[~by_id["tweet_id"].isin(to_drop)].reset_index(drop=True)
    return result, merges


def reconstruct_threads(df: pd.DataFrame) -> pd.DataFrame:
    """Tag each row with `thread_id`: the root ancestor's tweet_id, found by
    walking in_response_to_tweet_id up until it's null or points outside
    this slice. Forks (two brand replies to the same customer tweet) share a
    thread_id by design -- this stage groups the conversation tree, it does
    not pick a single linear path through it. Branch selection is a
    retrieval-index concern, not an ingestion concern.
    """
    parent_of = dict(zip(df["tweet_id"], df["in_response_to_tweet_id"], strict=True))
    valid_ids = set(parent_of)

    root_cache: dict[int, int] = {}

    def root_of(tid: int) -> int:
        """Walk up to the root, tolerating cycles.

        Reply graphs scraped from a live API are not guaranteed acyclic --
        a self-reply, an id reused across a pagination boundary, or a
        malformed row is enough to produce a -> b -> a. Without the `seen`
        guard this loop never terminates and `make data` hangs with no
        output. On detecting a cycle we collapse it to min(cycle), which is
        entry-order independent so the same file always yields the same
        thread_ids regardless of row ordering.
        """
        if tid in root_cache:
            return root_cache[tid]
        path: list[int] = []
        seen: set[int] = set()
        current = tid
        while True:
            if current in seen:
                current = min(seen)  # cycle: deterministic collapse
                break
            seen.add(current)
            path.append(current)
            parent = parent_of.get(current)
            if parent is None or pd.isna(parent) or int(parent) not in valid_ids:
                break
            nxt = int(parent)
            if nxt in root_cache:
                current = root_cache[nxt]
                break
            current = nxt
        for node in path:
            root_cache[node] = current
        return current

    df = df.copy()
    df["thread_id"] = [root_of(int(tid)) for tid in df["tweet_id"]]
    return df


def assign_time_split(df: pd.DataFrame, split_quantile: float) -> pd.DataFrame:
    """Time-based train/eval split (DECISIONS.md #4): index on the earlier
    window, evaluate on the later one, so a storm-driven burst of
    near-identical tweets can't land on both sides of the boundary.
    """
    df = df.copy()
    cutoff = df["created_at"].quantile(split_quantile)
    df["in_training_window"] = df["created_at"] <= cutoff
    return df


def scrub_text_column(df: pd.DataFrame, brand_handles: frozenset[str]) -> pd.DataFrame:
    df = df.copy()
    df["text_scrubbed"] = df["text"].astype(str).map(lambda t: scrub(t, brand_handles))
    return df


def run_ingest(
    raw_csv: Path,
    out_parquet: Path,
    brand_author_id: str,
    split_quantile: float,
    chunksize: int = 100_000,
) -> IngestStats:
    index_df = load_id_index(raw_csv, chunksize=chunksize)
    keep_ids = find_brand_connected_ids(index_df, brand_author_id)
    raw_rows = len(index_df)
    del index_df  # the full-file index is not needed past this point

    df, dupes, undated = stream_filter_csv(raw_csv, keep_ids, chunksize=chunksize)
    df, merges = join_multipart_replies(df, brand_author_id)
    df = reconstruct_threads(df)
    df = assign_time_split(df, split_quantile)
    df = scrub_text_column(df, brand_handles=frozenset({brand_author_id.lower()}))

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)

    return IngestStats(
        raw_rows=raw_rows,
        brand_connected_ids=len(keep_ids),
        kept_rows=len(df),
        duplicate_ids_dropped=dupes,
        undated_rows_dropped=undated,
        multipart_merges=merges,
        training_rows=int(df["in_training_window"].sum()),
        eval_rows=int((~df["in_training_window"]).sum()),
    )
