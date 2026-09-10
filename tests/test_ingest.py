from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from aa_agent.ingest import (
    assign_time_split,
    find_brand_connected_ids,
    join_multipart_replies,
    load_id_index,
    reconstruct_threads,
    run_ingest,
    stream_filter_csv,
)

BRAND = "AmericanAir"

# Column order matches the real Kaggle twcs.csv exactly.
# Covers: a clean thread: a customer tweet whose parent id doesn't exist
# anywhere (dangling, and NOT brand-connected -- must be excluded); a brand
# tweet whose cited parent is missing (dangling, but trivially included
# because the row itself is brand-authored); a fork (two brand replies to
# one customer tweet); an orphan brand broadcast with no parent at all; a
# two-part brand reply that must be rejoined; and a complete thread for a
# different brand that must be excluded entirely.
RAW_CSV = """tweet_id,author_id,inbound,created_at,text,response_tweet_id,in_response_to_tweet_id
100,11,True,2026-01-01T10:00:00Z,"@AmericanAir my bag is missing, email jane@x.com or call 555-123-9999, confirmation AB12CD",101,
101,AmericanAir,False,2026-01-01T10:05:00Z,"So sorry! DM us your confirmation.",,100
200,22,True,2026-01-01T10:10:00Z,"random unrelated tweet",,999999
250,AmericanAir,False,2026-01-01T11:00:00Z,"We have noted your concern.",,888888
300,33,True,2026-01-01T12:00:00Z,"@AmericanAir seat issue",301,
301,AmericanAir,False,2026-01-01T12:02:00Z,"Reply A",,300
302,AmericanAir,False,2026-01-01T12:03:00Z,"Reply B duplicate agent",,300
400,AmericanAir,False,2026-01-01T09:00:00Z,"System-wide delays today due to weather",,
500,44,True,2026-01-01T13:00:00Z,"@AmericanAir why was my flight cancelled",501,
501,AmericanAir,False,2026-01-01T13:02:00Z,"Sorry to hear that! (1/2)",502,500
502,AmericanAir,False,2026-01-01T13:03:00Z,"We have escalated this to our team.",,501
600,55,True,2026-01-01T14:00:00Z,"@sprintcare my phone is broken",601,
601,sprintcare,False,2026-01-01T14:05:00Z,"Sorry, DM us.",,600
"""


@pytest.fixture()
def raw_csv_path(tmp_path: Path) -> Path:
    p = tmp_path / "twcs.csv"
    p.write_text(RAW_CSV, encoding="utf-8")
    return p


# ---------------------------------------------------------------- pass 1 --


def test_load_id_index_reads_every_row(raw_csv_path):
    idx = load_id_index(raw_csv_path)
    assert len(idx) == 13


def test_brand_connected_ids_includes_every_touched_thread(raw_csv_path):
    idx = load_id_index(raw_csv_path)
    keep = find_brand_connected_ids(idx, BRAND)
    for expected in (100, 101, 250, 300, 301, 302, 400, 500, 501, 502):
        assert expected in keep


def test_brand_connected_ids_excludes_unrelated_dangling_thread(raw_csv_path):
    """Tweet 200's parent (999999) doesn't exist anywhere in the file, and
    200 itself isn't brand-authored -- the dangling reference must not
    accidentally pull it in."""
    idx = load_id_index(raw_csv_path)
    keep = find_brand_connected_ids(idx, BRAND)
    assert 200 not in keep


def test_brand_connected_ids_excludes_other_brand_entirely(raw_csv_path):
    idx = load_id_index(raw_csv_path)
    keep = find_brand_connected_ids(idx, BRAND)
    assert 600 not in keep
    assert 601 not in keep


def test_missing_brand_raises_a_clear_error(raw_csv_path):
    idx = load_id_index(raw_csv_path)
    with pytest.raises(ValueError, match="author_id"):
        find_brand_connected_ids(idx, "NotARealBrand")


# ---------------------------------------------------------------- pass 2 --


def test_stream_filter_keeps_exactly_the_connected_real_rows(raw_csv_path):
    idx = load_id_index(raw_csv_path)
    keep = find_brand_connected_ids(idx, BRAND)
    df, _dupes, _undated = stream_filter_csv(raw_csv_path, keep)

    # Every surviving row must be in the connected set...
    assert set(df["tweet_id"]).issubset(keep)

    # ...and 888888 is a dangling reference cited by tweet 250, not an
    # actual row in the file -- find_brand_connected_ids correctly includes
    # it as a graph node (visited to avoid crashing on the lookup), but
    # stream_filter_csv correctly can't return a row that doesn't exist.
    real_ids_in_file = {100, 101, 200, 250, 300, 301, 302, 400, 500, 501, 502, 600, 601}
    assert set(df["tweet_id"]) == keep & real_ids_in_file


# -------------------------------------------------------------- multipart --


def test_multipart_reply_merges_and_skips_the_dropped_parent():
    full = pd.DataFrame(
        {
            "tweet_id": [500, 501, 502],
            "author_id": ["44", BRAND, BRAND],
            "inbound": [True, False, False],
            "created_at": pd.to_datetime(
                ["2026-01-01T13:00:00Z", "2026-01-01T13:02:00Z", "2026-01-01T13:03:00Z"]
            ),
            "text": [
                "why was my flight cancelled",
                "Sorry to hear that! (1/2)",
                "We have escalated this to our team.",
            ],
            "response_tweet_id": [None, 502, None],
            "in_response_to_tweet_id": pd.array([pd.NA, 500, 501], dtype="Int64"),
        }
    )
    merged, n = join_multipart_replies(full, BRAND)
    assert n == 1
    assert set(merged["tweet_id"]) == {500, 502}
    row = merged.loc[merged["tweet_id"] == 502].iloc[0]
    assert row["in_response_to_tweet_id"] == 500  # points at part1's parent, not part1
    assert "(1/2)" not in row["text"]
    assert "Sorry to hear that!" in row["text"]
    assert "escalated" in row["text"]


def test_non_multipart_brand_replies_are_left_alone():
    full = pd.DataFrame(
        {
            "tweet_id": [300, 301],
            "author_id": ["33", BRAND],
            "inbound": [True, False],
            "created_at": pd.to_datetime(["2026-01-01T12:00:00Z", "2026-01-01T12:02:00Z"]),
            "text": ["seat issue", "Reply A"],
            "response_tweet_id": [301, None],
            "in_response_to_tweet_id": pd.array([pd.NA, 300], dtype="Int64"),
        }
    )
    merged, n = join_multipart_replies(full, BRAND)
    assert n == 0
    assert len(merged) == 2


# ---------------------------------------------------------------- threads --


def test_fork_shares_a_single_thread_id():
    df = pd.DataFrame(
        {
            "tweet_id": [300, 301, 302],
            "in_response_to_tweet_id": pd.array([pd.NA, 300, 300], dtype="Int64"),
        }
    )
    tagged = reconstruct_threads(df)
    assert tagged.loc[tagged["tweet_id"] == 301, "thread_id"].iloc[0] == 300
    assert tagged.loc[tagged["tweet_id"] == 302, "thread_id"].iloc[0] == 300


def test_orphan_broadcast_is_its_own_root():
    df = pd.DataFrame(
        {"tweet_id": [400], "in_response_to_tweet_id": pd.array([pd.NA], dtype="Int64")}
    )
    tagged = reconstruct_threads(df)
    assert tagged.iloc[0]["thread_id"] == 400


def test_parent_outside_the_slice_becomes_its_own_root():
    """250's cited parent (888888) was never in this slice -- reconstruction
    must not crash looking it up, and 250 becomes a root of one."""
    df = pd.DataFrame(
        {"tweet_id": [250], "in_response_to_tweet_id": pd.array([888888], dtype="Int64")}
    )
    tagged = reconstruct_threads(df)
    assert tagged.iloc[0]["thread_id"] == 250


def test_deep_chain_resolves_to_the_true_root():
    df = pd.DataFrame(
        {
            "tweet_id": [1, 2, 3, 4],
            "in_response_to_tweet_id": pd.array([pd.NA, 1, 2, 3], dtype="Int64"),
        }
    )
    tagged = reconstruct_threads(df)
    assert set(tagged["thread_id"]) == {1}


# ------------------------------------------------- messy real-data cases --


@pytest.mark.timeout(10)
def test_cyclic_parent_refs_terminate():
    """A scraped reply graph is not guaranteed acyclic. Without a cycle
    guard this hangs forever and `make data` produces no output at all."""
    df = pd.DataFrame(
        {"tweet_id": [1, 2], "in_response_to_tweet_id": pd.array([2, 1], dtype="Int64")}
    )
    tagged = reconstruct_threads(df)
    assert set(tagged["thread_id"]) == {1}  # collapsed to min(cycle)


@pytest.mark.timeout(10)
def test_self_referential_tweet_terminates():
    df = pd.DataFrame({"tweet_id": [7], "in_response_to_tweet_id": pd.array([7], dtype="Int64")})
    assert reconstruct_threads(df).iloc[0]["thread_id"] == 7


def test_cycle_collapse_is_entry_order_independent():
    """min(cycle) rather than "whichever node we re-entered on", so the
    same file yields the same thread_ids regardless of row order."""
    a = pd.DataFrame(
        {"tweet_id": [1, 2], "in_response_to_tweet_id": pd.array([2, 1], dtype="Int64")}
    )
    b = pd.DataFrame(
        {"tweet_id": [2, 1], "in_response_to_tweet_id": pd.array([1, 2], dtype="Int64")}
    )
    assert set(reconstruct_threads(a)["thread_id"]) == set(reconstruct_threads(b)["thread_id"])


def test_duplicate_tweet_ids_are_dropped_in_pass_two(tmp_path):
    """Duplicates make set_index non-unique, so a later .loc[parent] returns
    a DataFrame and the multipart merge dies with pandas' opaque "truth
    value of a Series is ambiguous"."""
    csv = tmp_path / "dupes.csv"
    csv.write_text(
        "tweet_id,author_id,inbound,created_at,text,response_tweet_id,in_response_to_tweet_id\n"
        '100,11,True,2026-01-01T10:00:00Z,"first",,\n'
        '100,11,True,2026-01-01T10:00:00Z,"duplicate row",,\n'
        '101,AmericanAir,False,2026-01-01T10:05:00Z,"reply",,100\n',
        encoding="utf-8",
    )
    df, dupes, undated = stream_filter_csv(csv, frozenset({100, 101}))
    assert dupes == 1
    assert undated == 0
    assert df["tweet_id"].is_unique
    assert df.loc[df["tweet_id"] == 100, "text"].iloc[0] == "first"  # keep='first'


def test_multipart_merge_rejects_duplicate_ids_with_a_named_error():
    df = pd.DataFrame(
        {
            "tweet_id": [501, 501],
            "author_id": [BRAND, BRAND],
            "inbound": [False, False],
            "created_at": pd.to_datetime(["2026-01-01T13:02:00Z", "2026-01-01T13:03:00Z"]),
            "text": ["Sorry! (1/2)", "dup"],
            "response_tweet_id": [None, None],
            "in_response_to_tweet_id": pd.array([pd.NA, pd.NA], dtype="Int64"),
        }
    )
    with pytest.raises(ValueError, match="duplicate tweet_id"):
        join_multipart_replies(df, BRAND)


def test_unparseable_timestamps_are_dropped_not_silently_held_out(tmp_path):
    """NaT <= cutoff is False, so undated rows would land in the EVAL
    window unannounced -- silently polluting the held-out set."""
    csv = tmp_path / "baddate.csv"
    csv.write_text(
        "tweet_id,author_id,inbound,created_at,text,response_tweet_id,in_response_to_tweet_id\n"
        '100,11,True,2026-01-01T10:00:00Z,"ok",,\n'
        '101,11,True,not-a-date,"undated",,\n',
        encoding="utf-8",
    )
    df, dupes, undated = stream_filter_csv(csv, frozenset({100, 101}))
    assert undated == 1
    assert set(df["tweet_id"]) == {100}


def test_real_twitter_timestamp_format_parses(tmp_path):
    """The fixtures elsewhere in this file use ISO-8601, which the real
    twcs.csv does NOT -- it uses Twitter's native format. Those fixtures
    only passed because errors="coerce" fell back to dateutil. This test
    pins the actual format so a strict-parsing regression is caught."""
    csv = tmp_path / "real_fmt.csv"
    csv.write_text(
        "tweet_id,author_id,inbound,created_at,text,response_tweet_id,in_response_to_tweet_id\n"
        '100,11,True,Tue Oct 31 22:10:47 +0000 2017,"@AmericanAir help",,\n'
        '101,AmericanAir,False,Tue Oct 31 22:11:45 +0000 2017,"Sorry!",,100\n',
        encoding="utf-8",
    )
    df, _dupes, undated = stream_filter_csv(csv, frozenset({100, 101}))
    assert undated == 0, "real Twitter timestamps must not be treated as undated"
    assert len(df) == 2
    assert df["created_at"].dt.year.tolist() == [2017, 2017]


def test_odd_format_timestamp_recovered_by_fallback(tmp_path):
    """A row not matching the strict format gets a second pass rather than
    being silently dropped as undated."""
    csv = tmp_path / "mixed.csv"
    csv.write_text(
        "tweet_id,author_id,inbound,created_at,text,response_tweet_id,in_response_to_tweet_id\n"
        '100,11,True,Tue Oct 31 22:10:47 +0000 2017,"native",,\n'
        '101,AmericanAir,False,2017-10-31T22:11:45Z,"iso instead",,100\n',
        encoding="utf-8",
    )
    df, _dupes, undated = stream_filter_csv(csv, frozenset({100, 101}))
    assert undated == 0
    assert len(df) == 2


# ------------------------------------------------------------------ split --


def test_time_split_is_monotonic_and_near_target_fraction():
    times = pd.date_range("2026-01-01", periods=100, freq="h", tz="UTC")
    df = pd.DataFrame({"tweet_id": range(100), "created_at": times})
    split = assign_time_split(df, split_quantile=0.7)
    train = split[split["in_training_window"]]
    held_out = split[~split["in_training_window"]]
    assert train["created_at"].max() <= held_out["created_at"].min()
    assert 0.6 <= len(train) / len(split) <= 0.8


# -------------------------------------------------------------- end to end --


def test_run_ingest_end_to_end(raw_csv_path, tmp_path):
    out = tmp_path / "american_air.parquet"
    stats = run_ingest(raw_csv_path, out, brand_author_id=BRAND, split_quantile=0.7)

    assert out.exists()
    result = pd.read_parquet(out)

    ids = set(result["tweet_id"])
    assert 600 not in ids and 601 not in ids  # other brand excluded
    assert 200 not in ids  # unrelated dangling ref excluded
    assert 501 not in ids  # multipart parent merged away
    assert 502 in ids  # multipart child retained, now merged

    merged_row = result.loc[result["tweet_id"] == 502].iloc[0]
    assert "escalated" in merged_row["text"]

    pii_row = result.loc[result["tweet_id"] == 100].iloc[0]
    assert "jane@x.com" in pii_row["text"]  # original preserved for audit
    assert "[EMAIL]" in pii_row["text_scrubbed"]
    assert "AB12CD" not in pii_row["text_scrubbed"]

    assert "thread_id" in result.columns
    assert "in_training_window" in result.columns
    assert stats.kept_rows == len(result)
    assert stats.multipart_merges == 1
    assert stats.brand_connected_ids >= 10
