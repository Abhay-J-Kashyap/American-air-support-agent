from __future__ import annotations

import pandas as pd
import pytest

from aa_agent.golden import (
    GoldenExample,
    append_label,
    attach_reference_replies,
    build_golden_sample,
    compile_golden_set,
    label_progress,
    load_labels,
    validate_patterns,
)

BRAND = "AmericanAir"


def _corpus(n_threads: int = 600) -> pd.DataFrame:
    """Synthetic corpus: 70% training / 30% eval, each thread one inbound
    opener plus one brand reply. A handful carry rare-intent markers."""
    rows = []
    rare_texts = {
        50: "i need a refund for my cancelled flight",
        51: "please rebook me on the next flight",
        52: "my wheelchair request was ignored at the gate",
        53: "my aadvantage miles are missing",
        54: "check-in failed and my boarding pass wont load",
        55: "can i upgrade my seat",
        56: "my lawyer will be in touch about this",
        57: "thanks a lot for losing my bag, wonderful",
        58: "my flight was late and also my bag is missing",
        59: "flying for a funeral, please help",
    }
    for t in range(n_threads):
        tid = 1000 + t * 10
        training = t < int(n_threads * 0.7)
        ts = pd.Timestamp("2017-01-01", tz="UTC") + pd.Timedelta(hours=t)
        rows.append(
            {
                "tweet_id": tid,
                "thread_id": tid,
                "author_id": str(t),
                "inbound": True,
                "created_at": ts,
                "text_scrubbed": rare_texts.get(t % 100, f"generic complaint number {t}"),
                "in_training_window": training,
            }
        )
        rows.append(
            {
                "tweet_id": tid + 1,
                "thread_id": tid,
                "author_id": BRAND,
                "inbound": False,
                "created_at": ts + pd.Timedelta(minutes=5),
                "text_scrubbed": f"reply to thread {tid}",
                "in_training_window": training,
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------- sampling ----


def test_patterns_all_compile():
    validate_patterns()


def test_sample_sizes_match_request():
    examples, stats = build_golden_sample(_corpus(), BRAND, 60, 40, seed=1)
    assert stats.stratum_a == 60
    assert stats.stratum_b == 40
    assert len(examples) == 100


def test_sample_comes_only_from_the_eval_window():
    """Sampling from the training window would evaluate the system on
    messages its own retrieval index contains."""
    df = _corpus()
    examples, _ = build_golden_sample(df, BRAND, 60, 40, seed=1)
    eval_ids = set(df[~df["in_training_window"]]["tweet_id"])
    assert all(e.tweet_id in eval_ids for e in examples)


def test_strata_do_not_overlap():
    examples, _ = build_golden_sample(_corpus(), BRAND, 60, 40, seed=1)
    a = {e.tweet_id for e in examples if e.stratum == "A"}
    b = {e.tweet_id for e in examples if e.stratum == "B"}
    assert not (a & b)


def test_stratum_a_is_uniform_random_only():
    """Stratum A must carry no selection bias -- it's the only stratum that
    can support a population estimate."""
    examples, _ = build_golden_sample(_corpus(), BRAND, 60, 40, seed=1)
    reasons = {e.sample_reason for e in examples if e.stratum == "A"}
    assert reasons == {"uniform_random"}


def test_stratum_b_reaches_rare_categories():
    """Round-robin exists so a common category can't exhaust the budget
    before a rare one is reached."""
    examples, stats = build_golden_sample(_corpus(), BRAND, 60, 60, seed=1)
    reasons = {e.sample_reason for e in examples if e.stratum == "B"}
    assert len(reasons) >= 5, f"expected several categories, got {reasons}"
    assert "random_topup" not in reasons or len(reasons) > 5


def test_adversarial_categories_are_sampled():
    examples, _ = build_golden_sample(_corpus(), BRAND, 50, 60, seed=3)
    reasons = {e.sample_reason for e in examples}
    assert reasons & {"legal_threat", "sarcasm", "multi_intent", "bereavement"}


def test_sampling_is_deterministic_under_a_fixed_seed():
    a, _ = build_golden_sample(_corpus(), BRAND, 50, 50, seed=7)
    b, _ = build_golden_sample(_corpus(), BRAND, 50, 50, seed=7)
    assert [e.tweet_id for e in a] == [e.tweet_id for e in b]


def test_different_seeds_give_different_samples():
    a, _ = build_golden_sample(_corpus(), BRAND, 50, 50, seed=7)
    b, _ = build_golden_sample(_corpus(), BRAND, 50, 50, seed=8)
    assert [e.tweet_id for e in a] != [e.tweet_id for e in b]


def test_too_small_a_pool_raises_clearly():
    with pytest.raises(ValueError, match="need at least"):
        build_golden_sample(_corpus(20), BRAND, 100, 100, seed=1)


def test_reference_replies_are_attached():
    examples, _ = build_golden_sample(_corpus(), BRAND, 40, 20, seed=1)
    assert any(e.reference_reply for e in examples)


def test_threads_without_a_brand_reply_get_none_not_a_crash():
    df = _corpus()
    df = df[df["author_id"] != BRAND]  # strip every reply
    examples, _ = build_golden_sample(df, BRAND, 40, 20, seed=1)
    assert all(e.reference_reply is None for e in examples)


def test_attach_reference_uses_the_earliest_reply():
    df = _corpus(100)
    extra = df[df["author_id"] == BRAND].iloc[0].copy()
    extra["tweet_id"] = 999999
    extra["created_at"] = extra["created_at"] - pd.Timedelta(minutes=1)
    extra["text_scrubbed"] = "EARLIEST REPLY"
    df2 = pd.concat([df, pd.DataFrame([extra])], ignore_index=True)
    sample = df2[(df2["thread_id"] == extra["thread_id"]) & (df2["inbound"])]
    out = attach_reference_replies(sample, df2, BRAND)
    assert out["reference_reply"].iloc[0] == "EARLIEST REPLY"


# ---------------------------------------------------------- persistence ----


def test_append_and_reload_roundtrip(tmp_path):
    path = tmp_path / "labels.jsonl"
    e = GoldenExample(1, 1, "text", "2017-01-01", "A", "uniform_random")
    e.label_intent = "baggage_issue"
    e.label_action = "escalate"
    append_label(e, path)
    labels = load_labels(path)
    assert labels[1].label_intent == "baggage_issue"


def test_relabelling_supersedes_the_earlier_entry(tmp_path):
    """Append-only storage means a correction is a newer line, not an edit."""
    path = tmp_path / "labels.jsonl"
    first = GoldenExample(1, 1, "t", "2017-01-01", "A", "uniform_random")
    first.label_intent = "wrong_label"
    first.label_action = "auto"
    append_label(first, path)

    second = GoldenExample(1, 1, "t", "2017-01-01", "A", "uniform_random")
    second.label_intent = "corrected_label"
    second.label_action = "escalate"
    append_label(second, path)

    labels = load_labels(path)
    assert len(labels) == 1
    assert labels[1].label_intent == "corrected_label"


def test_torn_line_does_not_destroy_earlier_labels(tmp_path):
    """Two hours of labelling must survive a killed process."""
    path = tmp_path / "labels.jsonl"
    e = GoldenExample(1, 1, "t", "2017-01-01", "A", "uniform_random")
    e.label_intent = "baggage_issue"
    e.label_action = "auto"
    append_label(e, path)
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"tweet_id": 2, "thread_i')
    labels = load_labels(path)
    assert 1 in labels


def test_progress_counts_only_fully_labelled(tmp_path):
    path = tmp_path / "labels.jsonl"
    examples = [GoldenExample(i, i, f"t{i}", "2017-01-01", "A", "uniform_random") for i in range(5)]
    partial = GoldenExample(0, 0, "t0", "2017-01-01", "A", "uniform_random")
    partial.label_intent = "baggage_issue"  # no action set
    append_label(partial, path)
    done, total = label_progress(examples, load_labels(path))
    assert (done, total) == (0, 5)


def test_compile_merges_labels_onto_examples(tmp_path):
    path = tmp_path / "labels.jsonl"
    examples = [GoldenExample(i, i, f"t{i}", "2017-01-01", "A", "uniform_random") for i in range(3)]
    labelled = GoldenExample(1, 1, "t1", "2017-01-01", "A", "uniform_random")
    labelled.label_intent = "refund_or_compensation"
    labelled.label_action = "escalate"
    append_label(labelled, path)

    out = compile_golden_set(examples, load_labels(path))
    assert len(out) == 3
    assert out.loc[out["tweet_id"] == 1, "label_intent"].iloc[0] == "refund_or_compensation"
    assert out.loc[out["tweet_id"] == 0, "label_intent"].isna().all()


def test_compile_preserves_stratum_for_separate_reporting():
    """A+B must never be reported as one population -- the stratum column
    is what makes separate reporting possible."""
    examples, _ = build_golden_sample(_corpus(), BRAND, 30, 20, seed=1)
    out = compile_golden_set(examples, {})
    assert set(out["stratum"]) == {"A", "B"}
    assert (out["stratum"] == "A").sum() == 30
