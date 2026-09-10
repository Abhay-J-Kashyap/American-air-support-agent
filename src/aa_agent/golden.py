"""Golden set construction: sampling, storage, resume.

Deliberately separate from the terminal UI in `cli.py` so the sampling
frame and the persistence layer are testable without simulating keystrokes.

The two-stratum design is the whole point (see SAMPLING.md):

  Stratum A -- uniform random from the EVAL window. The only stratum from
               which population-level metrics may be quoted.
  Stratum B -- stratified across intents plus adversarial picks. Gives the
               rare intents enough examples to measure at all. Emphatically
               NOT a population estimate; reporting a metric over A+B
               combined as though it described real traffic would be the
               single most misleading thing this project could do.

Labels are written append-only, one JSON object per line, flushed after
every example. Two hours of labelling must never be lost to a crash.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Patterns used ONLY to over-sample rare intents into stratum B. They are a
# sampling aid, never a label: the human decides the intent. A message
# matching /refund/ may well be venting rather than requesting one.
RARE_INTENT_PATTERNS: dict[str, str] = {
    "refund_or_compensation": r"refund|voucher|compensat|reimburs|money back",
    "booking_change": r"cancel|rebook|reschedul|change my flight|change our flight",
    "seat_or_upgrade": r"\bseat|upgrade",
    "checkin_boarding": r"check.?in|boarding pass",
    "loyalty_miles": r"aadvantage|miles|elite|status match",
    "accessibility_medical": r"wheelchair|medical|disab|service animal",
}

# Signals for genuinely hard cases. A golden set of only easy examples
# flatters the system and hides exactly the failures worth finding.
ADVERSARIAL_PATTERNS: dict[str, str] = {
    "legal_threat": r"\blawyer|\bsue\b|lawsuit|attorney|dot complaint|department of transportation",
    "sarcasm": r"thanks a lot|great job|wonderful|fantastic.*(?:delay|cancel|lost)|love how",
    "multi_intent": r"\band also\b|second(?:ly)?|two (?:issues|problems)|on top of that",
    "unaccompanied_minor": r"unaccompanied|my (?:son|daughter|kid|child) (?:is|was) fly",
    "bereavement": r"funeral|passed away|bereave|died",
}


@dataclass
class GoldenExample:
    """One labelled example. `label_*` fields are filled by the human."""

    tweet_id: int
    thread_id: int
    text: str
    created_at: str
    stratum: str
    sample_reason: str
    reference_reply: str | None = None
    label_intent: str | None = None
    label_action: str | None = None
    label_reason: str | None = None
    reference_is_good: bool | None = None
    notes: str = ""

    @property
    def is_labelled(self) -> bool:
        return self.label_intent is not None and self.label_action is not None


@dataclass
class SampleStats:
    stratum_a: int
    stratum_b: int
    total: int
    by_reason: dict[str, int] = field(default_factory=dict)


def _first_inbound_eval_window(df: pd.DataFrame) -> pd.DataFrame:
    """First-turn customer messages from the HELD-OUT window only.

    The golden set must come from the eval window: sampling it from the
    training window would evaluate the system on messages its own retrieval
    index contains, which inflates every score (DECISIONS.md #4).
    """
    work = df[df["inbound"].astype(str).str.lower().isin({"true", "1"})]
    work = work[~work["in_training_window"]]
    if work.empty:
        return work.copy()
    return (
        work.sort_values("created_at")
        .drop_duplicates(subset="thread_id", keep="first")
        .reset_index(drop=True)
    )


def attach_reference_replies(sample: pd.DataFrame, df: pd.DataFrame, brand: str) -> pd.DataFrame:
    """Attach the brand's actual first reply in each sampled thread.

    Kept as a REFERENCE, not a target. A large share are content-free
    ("DM us"), which is why the labeller also asks whether the reference
    was any good -- without that field, cases where the agent beats the
    reference are indistinguishable from cases where it failed.
    """
    replies = df[(df["author_id"] == brand) & (df["thread_id"].isin(set(sample["thread_id"])))]
    if replies.empty:
        sample = sample.copy()
        sample["reference_reply"] = None
        return sample
    first_reply = (
        replies.sort_values("created_at")
        .drop_duplicates(subset="thread_id", keep="first")
        .set_index("thread_id")["text_scrubbed"]
    )
    sample = sample.copy()
    sample["reference_reply"] = sample["thread_id"].map(first_reply)
    return sample


def build_golden_sample(
    df: pd.DataFrame,
    brand: str,
    n_stratum_a: int = 100,
    n_stratum_b: int = 100,
    seed: int = 20260909,
) -> tuple[list[GoldenExample], SampleStats]:
    """Draw the two strata. Stratum B never overlaps stratum A."""
    pool = _first_inbound_eval_window(df)
    if len(pool) < n_stratum_a + n_stratum_b:
        raise ValueError(
            f"eval window has only {len(pool)} first-turn messages; "
            f"need at least {n_stratum_a + n_stratum_b}"
        )

    rng = np.random.default_rng(seed)

    a_idx = rng.choice(len(pool), size=n_stratum_a, replace=False)
    stratum_a = pool.iloc[a_idx]

    remaining = pool.drop(pool.index[a_idx])
    text = remaining["text_scrubbed"].astype(str).str.lower()

    # Round-robin across categories rather than filling each quota in turn,
    # so a common category cannot exhaust the budget before a rare one is
    # reached. accessibility_medical (~0.5%) would otherwise get nothing.
    buckets: dict[str, list[int]] = {}
    for name, pattern in {**RARE_INTENT_PATTERNS, **ADVERSARIAL_PATTERNS}.items():
        hits = remaining.index[text.str.contains(pattern, regex=True, na=False)].tolist()
        rng.shuffle(hits)
        if hits:
            buckets[name] = hits

    chosen: dict[int, str] = {}
    while len(chosen) < n_stratum_b and buckets:
        for name in list(buckets):
            if len(chosen) >= n_stratum_b:
                break
            while buckets[name]:
                idx = buckets[name].pop()
                if idx not in chosen:
                    chosen[idx] = name
                    break
            if not buckets[name]:
                del buckets[name]

    # Top up with random draws if the patterns could not fill the quota.
    if len(chosen) < n_stratum_b:
        leftovers = [i for i in remaining.index if i not in chosen]
        rng.shuffle(leftovers)
        for idx in leftovers[: n_stratum_b - len(chosen)]:
            chosen[idx] = "random_topup"

    stratum_b = remaining.loc[list(chosen)]

    stratum_a = attach_reference_replies(stratum_a, df, brand)
    stratum_b = attach_reference_replies(stratum_b, df, brand)

    examples: list[GoldenExample] = []
    for _, row in stratum_a.iterrows():
        examples.append(_to_example(row, "A", "uniform_random"))
    for idx, row in stratum_b.iterrows():
        examples.append(_to_example(row, "B", chosen[idx]))

    by_reason: dict[str, int] = {}
    for e in examples:
        by_reason[e.sample_reason] = by_reason.get(e.sample_reason, 0) + 1

    stats = SampleStats(
        stratum_a=len(stratum_a),
        stratum_b=len(stratum_b),
        total=len(examples),
        by_reason=by_reason,
    )
    return examples, stats


def _to_example(row: pd.Series, stratum: str, reason: str) -> GoldenExample:
    ref = row.get("reference_reply")
    return GoldenExample(
        tweet_id=int(row["tweet_id"]),
        thread_id=int(row["thread_id"]),
        text=str(row["text_scrubbed"]),
        created_at=str(row["created_at"]),
        stratum=stratum,
        sample_reason=reason,
        reference_reply=None if ref is None or pd.isna(ref) else str(ref),
    )


# ------------------------------------------------------------ persistence --


def save_examples(examples: list[GoldenExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for e in examples:
            fh.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")


def load_examples(path: Path) -> list[GoldenExample]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(GoldenExample(**json.loads(line)))
            except (json.JSONDecodeError, TypeError):
                continue  # torn line from a killed process
    return out


def append_label(example: GoldenExample, path: Path) -> None:
    """Append one labelled example, flushed immediately.

    Append-and-flush per example rather than a bulk save at the end: two
    hours of labelling must survive a crash, a closed terminal, or a laptop
    lid. `load_labels` keeps the LAST entry per tweet_id, so relabelling an
    example simply appends a newer record.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")
        fh.flush()


def load_labels(path: Path) -> dict[int, GoldenExample]:
    """Latest label per tweet_id, so a re-label supersedes an earlier one."""
    latest: dict[int, GoldenExample] = {}
    for e in load_examples(path):
        latest[e.tweet_id] = e
    return latest


def label_progress(
    examples: list[GoldenExample], labels: dict[int, GoldenExample]
) -> tuple[int, int]:
    done = sum(1 for e in examples if e.tweet_id in labels and labels[e.tweet_id].is_labelled)
    return done, len(examples)


def compile_golden_set(
    examples: list[GoldenExample], labels: dict[int, GoldenExample]
) -> pd.DataFrame:
    """Merge sampled examples with their labels into a single frame."""
    rows = []
    for e in examples:
        merged = labels.get(e.tweet_id, e)
        rows.append(asdict(merged))
    return pd.DataFrame(rows)


def validate_patterns() -> None:
    """Fail fast on a malformed regex rather than mid-sample."""
    for name, pattern in {**RARE_INTENT_PATTERNS, **ADVERSARIAL_PATTERNS}.items():
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"invalid pattern for {name!r}: {exc}") from exc
