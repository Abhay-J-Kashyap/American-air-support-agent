from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from aa_agent.embed import content_hash, embed_cached, l2_normalize
from aa_agent.intents import (
    IntentCluster,
    KSweepResult,
    build_label_prompt,
    extract_first_inbound,
    fit_clusters,
    parse_label_response,
    pick_best_k,
    representatives,
    silhouette_is_monotonic_decreasing,
    structure_is_weak,
    sweep_k,
    write_taxonomy,
)


class StubEmbedder:
    """Deterministic fake embedder. Lets every test below run without
    downloading an ONNX model -- CI must not depend on a model download."""

    name = "stub"
    dim = 8

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        rng = np.random.default_rng(0)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            seed = abs(hash(t)) % (2**31)
            out[i] = np.random.default_rng(seed).normal(size=self.dim)
        del rng
        return out


# ------------------------------------------------------------------ embed --


def test_content_hash_is_order_sensitive():
    assert content_hash(["a", "b"], "m", "float16") != content_hash(["b", "a"], "m", "float16")


def test_content_hash_varies_with_model_and_dtype():
    base = content_hash(["a"], "m1", "float16")
    assert base != content_hash(["a"], "m2", "float16")
    assert base != content_hash(["a"], "m1", "float32")


def test_embed_cached_hits_disk_on_second_call(tmp_path):
    emb = StubEmbedder()
    texts = ["bag missing", "flight delayed"]
    first = embed_cached(texts, emb, tmp_path)
    assert emb.calls == 1
    second = embed_cached(texts, emb, tmp_path)
    assert emb.calls == 1, "second call must come from disk, not the model"
    np.testing.assert_allclose(first, second, rtol=1e-2)


def test_embed_cached_returns_float32_despite_float16_storage(tmp_path):
    out = embed_cached(["x"], StubEmbedder(), tmp_path, dtype="float16")
    assert out.dtype == np.float32


def test_embed_cached_rejects_wrong_row_count(tmp_path):
    class Broken(StubEmbedder):
        def encode(self, texts):
            return np.zeros((len(texts) + 1, self.dim), dtype=np.float32)

    with pytest.raises(ValueError, match="vectors for"):
        embed_cached(["a", "b"], Broken(), tmp_path)


def test_embed_cached_writes_atomically_no_tmp_file_left_behind(tmp_path):
    embed_cached(["x", "y"], StubEmbedder(), tmp_path)
    tmp_files = list(tmp_path.glob("*.npy.tmp"))
    assert tmp_files == [], "temp file must be renamed away, not left behind"


def test_embed_cached_self_heals_from_a_corrupt_cache_file(tmp_path):
    """Regression: np.save is not atomic. A run killed mid-write leaves a
    truncated .npy at the real cache path. Before this fix, the next run
    would crash on np.load with a confusing numpy error far from its cause.
    Now it's treated as a miss and re-embedded."""
    from aa_agent.embed import content_hash

    texts = ["bag missing"]
    key = content_hash(texts, "stub", "float16")
    corrupt = tmp_path / f"emb_{key}.npy"
    corrupt.write_bytes(b"not a valid npy file")

    emb = StubEmbedder()
    out = embed_cached(texts, emb, tmp_path)
    assert emb.calls == 1, "corrupt file must trigger a real re-embed, not silently fail"
    assert out.shape == (1, emb.dim)
    assert not corrupt.read_bytes().startswith(b"not a valid")  # overwritten with a real one


def test_l2_normalize_unit_length_and_zero_safe():
    X = np.array([[3.0, 4.0], [0.0, 0.0]])
    out = l2_normalize(X)
    assert np.isclose(np.linalg.norm(out[0]), 1.0)
    assert not np.isnan(out[1]).any(), "zero vector must not produce NaN"


# ------------------------------------------------------- first-turn subset --


def _thread_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tweet_id": [1, 2, 3, 4, 5],
            "thread_id": [1, 1, 1, 4, 4],
            "inbound": [True, False, True, True, False],
            "created_at": pd.to_datetime(
                [
                    "2017-01-01T10:00:00Z",
                    "2017-01-01T10:05:00Z",
                    "2017-01-01T10:10:00Z",
                    "2017-06-01T09:00:00Z",
                    "2017-06-01T09:05:00Z",
                ]
            ),
            "text_scrubbed": ["opening", "brand reply", "follow up", "later opening", "reply"],
            "in_training_window": [True, True, True, False, False],
        }
    )


def test_extract_first_inbound_takes_one_row_per_thread():
    out = extract_first_inbound(_thread_frame(), training_only=False)
    assert len(out) == 2
    assert set(out["thread_id"]) == {1, 4}


def test_extract_first_inbound_takes_the_earliest_not_the_first_row():
    df = _thread_frame()
    df = df.iloc[::-1].reset_index(drop=True)  # reverse row order
    out = extract_first_inbound(df, training_only=False)
    assert out.loc[out["thread_id"] == 1, "text_scrubbed"].iloc[0] == "opening"


def test_extract_first_inbound_keeps_rows_intact_when_text_is_null():
    """Regression: groupby().first() returns the first NON-NULL value per
    column independently, which stitches a fake row together from two real
    ones -- tweet_id from the opening message, text from a later one. The
    extracted row must correspond to exactly one real tweet."""
    df = pd.DataFrame(
        {
            "tweet_id": [10, 11],
            "thread_id": [1, 1],
            "inbound": [True, True],
            "created_at": pd.to_datetime(["2017-01-01T10:00:00Z", "2017-01-01T10:05:00Z"]),
            "text_scrubbed": [None, "SECOND MESSAGE TEXT"],
            "in_training_window": [True, True],
        }
    )
    out = extract_first_inbound(df, training_only=False)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["tweet_id"] == 10
    assert row["text_scrubbed"] != "SECOND MESSAGE TEXT", "text borrowed from a different tweet"


def test_extract_first_inbound_excludes_brand_replies():
    out = extract_first_inbound(_thread_frame(), training_only=False)
    assert "brand reply" not in set(out["text_scrubbed"])


def test_extract_first_inbound_respects_the_training_window():
    """The taxonomy must not be derived from held-out data."""
    out = extract_first_inbound(_thread_frame(), training_only=True)
    assert set(out["thread_id"]) == {1}


def test_extract_first_inbound_handles_empty_input():
    empty = _thread_frame().iloc[0:0]
    assert extract_first_inbound(empty).empty


# --------------------------------------------------------------- clustering --


def _separated_blobs(seed: int = 0) -> np.ndarray:
    """Three well-separated blobs -- a sweep should prefer k=3."""
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    return np.vstack([c + rng.normal(scale=0.3, size=(40, 2)) for c in centers])


def test_sweep_prefers_the_true_cluster_count():
    results = sweep_k(_separated_blobs(), [2, 3, 4, 5], seed=0)
    assert pick_best_k(results) == 3


def test_sweep_skips_k_larger_than_the_dataset():
    X = np.random.default_rng(0).normal(size=(5, 2))
    results = sweep_k(X, [2, 3, 99], seed=0)
    assert 99 not in [r.k for r in results]


def test_sweep_inertia_falls_monotonically_with_k():
    """Inertia can never select k on its own -- this pins why."""
    results = sweep_k(_separated_blobs(), [2, 3, 4, 5], seed=0)
    inertias = [r.inertia for r in results]
    assert inertias == sorted(inertias, reverse=True)


def test_pick_best_k_rejects_an_empty_sweep():
    with pytest.raises(ValueError, match="empty sweep"):
        pick_best_k([])


def test_fit_clusters_is_deterministic_under_a_fixed_seed():
    X = _separated_blobs()
    a, _ = fit_clusters(X, 3, seed=42)
    b, _ = fit_clusters(X, 3, seed=42)
    np.testing.assert_array_equal(a, b)


def test_representatives_are_nearest_to_centroid():
    X = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 0.0]])
    labels = np.array([0, 0, 0])
    centroids = np.array([[0.0, 0.0]])
    texts = ["closest", "near", "far outlier"]
    reps = representatives(X, labels, centroids, texts, n_per_cluster=2)
    assert reps[0] == ["closest", "near"]


def test_representatives_cover_every_cluster():
    X = _separated_blobs()
    labels, centroids = fit_clusters(X, 3, seed=0)
    reps = representatives(X, labels, centroids, [f"m{i}" for i in range(len(X))], 5)
    assert sorted(reps) == [0, 1, 2]
    assert all(len(v) == 5 for v in reps.values())


def test_representatives_handles_cluster_smaller_than_n():
    X = np.array([[0.0, 0.0], [9.0, 9.0], [9.1, 9.1]])
    labels = np.array([0, 1, 1])
    centroids = np.array([[0.0, 0.0], [9.05, 9.05]])
    reps = representatives(X, labels, centroids, ["a", "b", "c"], n_per_cluster=10)
    assert len(reps[0]) == 1


# ----------------------------------------------------------- LLM labelling --


def test_prompt_includes_every_message_numbered():
    msgs = ["bag lost", "flight late"]
    prompt = build_label_prompt(msgs)
    user = prompt[-1]["content"]
    assert "1. bag lost" in user and "2. flight late" in user


def test_parse_plain_json():
    label, desc = parse_label_response('{"label": "baggage_issue", "description": "Lost bags."}')
    assert label == "baggage_issue"
    assert desc == "Lost bags."


def test_parse_strips_markdown_fences():
    raw = '```json\n{"label": "refund_request", "description": "Wants money back."}\n```'
    assert parse_label_response(raw)[0] == "refund_request"


def test_parse_recovers_json_embedded_in_prose():
    """Small models add preamble despite explicit instructions not to."""
    raw = 'Sure! Here is the label:\n{"label": "seat_change", "description": "Seat swap."}\nHope that helps.'
    assert parse_label_response(raw)[0] == "seat_change"


def test_parse_raises_when_there_is_no_json():
    with pytest.raises(ValueError, match="no JSON object"):
        parse_label_response("I cannot help with that request.")


def test_parse_raises_on_missing_label_field():
    with pytest.raises(ValueError, match="label"):
        parse_label_response('{"description": "no label here"}')


# --------------------------------------------------------------- artifacts --


def test_write_taxonomy_emits_both_files_and_appends_other(tmp_path):
    clusters = [IntentCluster(0, "baggage_issue", "Lost bags.", 120, ["m1", "m2"])]
    sweep = [KSweepResult(k=3, silhouette=0.4, inertia=100.0)]
    write_taxonomy(clusters, sweep, tmp_path)

    taxonomy = json.loads((tmp_path / "taxonomy.json").read_text())
    labels = [i["label"] for i in taxonomy["intents"]]
    assert "baggage_issue" in labels
    assert labels[-1] == "other", "an escape-hatch intent must always exist"

    audit = json.loads((tmp_path / "cluster_audit.json").read_text())
    assert audit["k_sweep"][0]["k"] == 3
    assert audit["clusters"][0]["size"] == 120
    assert audit["clusters"][0]["representatives"] == ["m1", "m2"]


# ------------------------------------------- reasoning-model output parsing --


def test_parse_ignores_braces_inside_a_think_block():
    """gpt-oss and qwen emit reasoning that routinely contains braces. An
    earlier version grabbed the FIRST {...} span and picked up reasoning
    debris instead of the answer."""
    raw = (
        '<think>Maybe {"label": "wrong_guess"} fits? No, reconsider.</think>\n'
        '{"label": "baggage_issue", "description": "Lost bags."}'
    )
    assert parse_label_response(raw)[0] == "baggage_issue"


def test_parse_prefers_the_last_json_object():
    raw = '{"label": "first_draft"} then revised: {"label": "final_answer", "description": "d"}'
    assert parse_label_response(raw)[0] == "final_answer"


def test_parse_handles_nested_braces():
    raw = '{"label": "refund", "description": "x", "meta": {"conf": 0.9}}'
    assert parse_label_response(raw)[0] == "refund"


def test_parse_handles_braces_inside_string_values():
    raw = '{"label": "seat", "description": "user typed { in the message"}'
    label, desc = parse_label_response(raw)
    assert label == "seat"
    assert "{" in desc


def test_parse_skips_a_json_object_that_lacks_a_label():
    raw = '{"description": "no label"} {"label": "real_one", "description": "d"}'
    assert parse_label_response(raw)[0] == "real_one"


def test_parse_prefers_fenced_block_over_surrounding_prose():
    raw = 'Analysis: consider {"label": "noise"}.\n```json\n{"label": "correct", "description": "d"}\n```'
    assert parse_label_response(raw)[0] == "correct"


def test_iter_json_objects_finds_multiple_spans():
    from aa_agent.intents import _iter_json_objects

    assert len(_iter_json_objects('{"a":1} text {"b":2}')) == 2


def test_iter_json_objects_tolerates_unbalanced_braces():
    from aa_agent.intents import _iter_json_objects

    assert _iter_json_objects('}}} {"a":1}') == ['{"a":1}']


# --------------------------------------------------- cluster-quality guards --


def test_structure_is_weak_flags_near_zero_silhouette():
    """Real run produced 0.02-0.04 across all k -- essentially no structure."""
    weak = [KSweepResult(k=k, silhouette=0.03, inertia=100.0) for k in range(4, 17)]
    assert structure_is_weak(weak)


def test_structure_is_weak_passes_genuine_clusters():
    strong = [KSweepResult(k=3, silhouette=0.62, inertia=100.0)]
    assert not structure_is_weak(strong)


def test_monotonic_decreasing_detected():
    """When silhouette only falls, argmax returns the smallest k swept --
    an artifact of --k-min, not a property of the data."""
    results = [
        KSweepResult(k=4, silhouette=0.042, inertia=3544.0),
        KSweepResult(k=5, silhouette=0.036, inertia=3487.0),
        KSweepResult(k=6, silhouette=0.029, inertia=3436.0),
    ]
    assert silhouette_is_monotonic_decreasing(results)
    assert pick_best_k(results) == 4, "argmax lands on the range's lower bound"


def test_monotonic_not_flagged_when_there_is_a_real_peak():
    results = [
        KSweepResult(k=2, silhouette=0.30, inertia=100.0),
        KSweepResult(k=3, silhouette=0.62, inertia=90.0),
        KSweepResult(k=4, silhouette=0.41, inertia=80.0),
    ]
    assert not silhouette_is_monotonic_decreasing(results)
    assert pick_best_k(results) == 3
