from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from aa_agent.taxonomy import RiskTier, Taxonomy, load_taxonomy

REAL = Path("config/taxonomy.yaml")


def _minimal(**overrides) -> dict:
    base = {
        "intents": [
            {
                "label": "baggage_issue",
                "description": "Lost bag.",
                "source": "cluster",
                "risk_tier": "review",
            },
            {
                "label": "other",
                "description": "Anything else.",
                "source": "cluster",
                "risk_tier": "review",
            },
        ]
    }
    base.update(overrides)
    return base


# ------------------------------------------------------- the shipped file --


def test_real_taxonomy_loads():
    tax = load_taxonomy(REAL)
    assert len(tax.intents) >= 10


def test_real_taxonomy_marks_the_two_known_high_risk_intents():
    """Refund commits money; accessibility_medical is safety-critical and at
    0.4% prevalence no learned classifier will reliably catch it."""
    escalate = set(load_taxonomy(REAL).always_escalate_labels())
    assert {"refund_or_compensation", "accessibility_medical"} <= escalate


def test_real_taxonomy_records_provenance_for_every_intent():
    """cluster vs keyword carry different evidence; the report must not
    blur them, so neither may be left unset."""
    for intent in load_taxonomy(REAL).intents:
        assert intent.source is not None


def test_cluster_shares_partition_the_sample():
    """The structural invariant that actually matters: k-means assigns each
    message exactly one cluster, so cluster-sourced shares must sum to ~1.
    If this drifts, a merge has double-counted or dropped a cluster.

    (An earlier version of this test asserted 'anything under 6% must be
    keyword-sourced'. That threshold was derived from understated keyword
    figures -- computed with single keywords on the wrong denominator -- and
    broke as soon as the numbers were corrected. Encoding a real invariant
    rather than an incidental one.)
    """
    from aa_agent.taxonomy import IntentSource

    total = sum(
        i.approx_share
        for i in load_taxonomy(REAL).intents
        if i.source is IntentSource.CLUSTER and i.approx_share
    )
    assert 0.97 <= total <= 1.03, f"cluster shares sum to {total:.3f}, expected ~1.0"


def test_keyword_shares_are_not_a_distribution():
    """Keyword shares are overlapping subsets on a different denominator.
    They must never be mistaken for a partition."""
    from aa_agent.taxonomy import IntentSource

    total = sum(
        i.approx_share
        for i in load_taxonomy(REAL).intents
        if i.source is IntentSource.KEYWORD and i.approx_share
    )
    assert total < 1.0


def test_suspect_figures_are_flagged_in_the_file():
    """checkin_boarding's 8.3% is inflated by \\bgate\\b matching delay
    complaints. A figure we don't trust must carry that warning into the
    artifact, not just live in a chat log."""
    tax = load_taxonomy(REAL)
    note = tax.get("checkin_boarding").note or ""
    assert "SUSPECT" in note


def test_prompt_block_lists_every_label():
    tax = load_taxonomy(REAL)
    block = tax.prompt_block()
    for label in tax.labels:
        assert label in block


# ---------------------------------------------------------- validation ----


def test_duplicate_labels_rejected():
    bad = _minimal()
    bad["intents"].append(dict(bad["intents"][0]))
    with pytest.raises(ValueError, match="duplicate intent labels"):
        Taxonomy.model_validate(bad)


def test_missing_other_rejected():
    bad = {"intents": [i for i in _minimal()["intents"] if i["label"] != "other"]}
    bad["intents"].append(
        {"label": "extra", "description": "d", "source": "cluster", "risk_tier": "review"}
    )
    with pytest.raises(ValueError, match="'other' intent"):
        Taxonomy.model_validate(bad)


@pytest.mark.parametrize("bad_label", ["Baggage Issue", "baggage-issue", "BaggageIssue"])
def test_non_snake_case_labels_rejected(bad_label):
    bad = _minimal()
    bad["intents"][0]["label"] = bad_label
    with pytest.raises(ValueError, match="snake_case"):
        Taxonomy.model_validate(bad)


def test_unknown_risk_tier_rejected():
    bad = _minimal()
    bad["intents"][0]["risk_tier"] = "probably_fine"
    with pytest.raises(ValueError):
        Taxonomy.model_validate(bad)


def test_lookup_helpers():
    tax = Taxonomy.model_validate(_minimal())
    assert tax.risk_tier("baggage_issue") is RiskTier.REVIEW
    with pytest.raises(KeyError, match="unknown intent"):
        tax.get("nonexistent")


def test_missing_file_error_names_the_recovery_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="discover-intents"):
        load_taxonomy(tmp_path / "nope.yaml")


def test_discover_intents_output_does_not_overwrite_curated_file(tmp_path):
    """Regression guard: the proposal and the curated taxonomy are separate
    artifacts. A discover-intents re-run must never clobber the file the
    classifier and escalation policy are built against."""
    from aa_agent.intents import IntentCluster, KSweepResult, write_taxonomy

    curated_before = REAL.read_text(encoding="utf-8")
    write_taxonomy(
        [IntentCluster(0, "some_new_cluster", "d", 10, ["m"])],
        [KSweepResult(k=1, silhouette=0.1, inertia=1.0)],
        tmp_path,
    )
    assert REAL.read_text(encoding="utf-8") == curated_before
    assert (tmp_path / "taxonomy.json").exists()


def test_shipped_yaml_is_parseable_as_plain_yaml():
    """Catches a stray tab or bad indent before pydantic even sees it."""
    assert isinstance(yaml.safe_load(REAL.read_text(encoding="utf-8")), dict)


def test_shares_are_documented_as_non_exclusive():
    """The shares total ~110% because cluster shares partition the corpus
    while keyword shares are overlapping subsets drawn from within them.
    That's correct, but it MUST be documented or a reader will mistake the
    list for a distribution."""
    text = REAL.read_text(encoding="utf-8")
    assert "DO NOT SUM TO 1" in text
    total = sum(i.approx_share for i in load_taxonomy(REAL).intents if i.approx_share)
    assert total > 1.0, "if this ever sums to <=1 the warning comment is now wrong"
