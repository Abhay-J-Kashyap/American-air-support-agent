"""Intent discovery: cluster real customer messages, then label the clusters.

Deliberately NOT a hand-written taxonomy. The brief asks for intents defined
*from the data*, so the pipeline is: extract first-turn customer messages ->
embed -> k-means with a silhouette sweep -> pull representatives nearest each
centroid -> have an LLM propose a label per cluster -> human edit pass.

The LLM never sees the whole corpus and never picks the number of clusters.
It only names groups the geometry already found, which keeps the taxonomy
grounded in message distribution rather than in the model's priors about
what airline complaints "should" look like.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from aa_agent.llm.client import LLMClient

# Silhouette is O(n^2) in memory and time. On 30k+ points it will hang or
# OOM, so it is always computed on a subsample. This is a scoring heuristic
# for choosing k, not a reported metric -- an approximation is fine.
SILHOUETTE_SAMPLE = 5_000


@dataclass(frozen=True)
class KSweepResult:
    k: int
    silhouette: float
    inertia: float


@dataclass
class IntentCluster:
    cluster_id: int
    label: str
    description: str
    size: int
    representatives: list[str]
    raw_response: str = ""


def extract_first_inbound(df: pd.DataFrame, training_only: bool = True) -> pd.DataFrame:
    """One row per thread: the customer's opening message.

    Intent is a property of what the customer first asked, not of every
    follow-up turn. Clustering all inbound messages would over-represent
    long argumentative threads and fill the taxonomy with clusters like
    "still waiting" and "any update?" -- conversational states, not intents.

    Training window only by default: the taxonomy must be derived without
    looking at the held-out evaluation window (DECISIONS.md #4).
    """
    work = df[df["inbound"].astype(str).str.lower().isin({"true", "1"})]
    if training_only:
        work = work[work["in_training_window"]]
    if work.empty:
        return work.copy()
    # drop_duplicates, NOT groupby().first(): groupby's first() returns the
    # first NON-NULL value for each column INDEPENDENTLY, so a thread whose
    # opening message has a null text would emit a Frankenstein row --
    # tweet_id and created_at from the real first message, text silently
    # borrowed from a later one. drop_duplicates keeps whole rows intact.
    first = work.sort_values("created_at").drop_duplicates(subset="thread_id", keep="first")
    return first.reset_index(drop=True)


def sweep_k(
    X: np.ndarray, k_values: list[int], seed: int, sample: int = SILHOUETTE_SAMPLE
) -> list[KSweepResult]:
    """Fit k-means for each k and score it.

    Both silhouette and inertia are recorded. Inertia always falls as k
    rises so it can never select k on its own, but it is kept because the
    shape of the curve is what a human reads when overruling the silhouette
    pick -- and the brief expects a human edit pass, not blind automation.
    """
    results = []
    for k in k_values:
        if k >= len(X):
            continue
        km = KMeans(n_clusters=k, random_state=seed, n_init=3)
        labels = km.fit_predict(X)
        n_sample = min(sample, len(X))
        score = float(
            silhouette_score(X, labels, sample_size=n_sample, random_state=seed)
            if len(set(labels)) > 1
            else -1.0
        )
        results.append(KSweepResult(k=k, silhouette=score, inertia=float(km.inertia_)))
    return results


# Silhouette interpretation (Rousseeuw): >0.70 strong, 0.50-0.70 reasonable,
# 0.25-0.50 weak, <0.25 no substantial structure. Short-text embeddings
# routinely land near zero because semantic space is a CONTINUUM, not a set
# of discrete blobs -- k-means still returns clusters, but the boundaries
# are imposed rather than discovered.
WEAK_STRUCTURE_THRESHOLD = 0.25


def pick_best_k(results: list[KSweepResult]) -> int:
    if not results:
        raise ValueError("empty sweep: no k values were evaluated")
    return max(results, key=lambda r: r.silhouette).k


def structure_is_weak(results: list[KSweepResult]) -> bool:
    """True when no k in the sweep shows substantial cluster structure."""
    return bool(results) and max(r.silhouette for r in results) < WEAK_STRUCTURE_THRESHOLD


def silhouette_is_monotonic_decreasing(results: list[KSweepResult]) -> bool:
    """True when silhouette only falls as k rises.

    Then argmax necessarily returns the smallest k swept, so the "chosen"
    k is an artifact of the range's lower bound rather than a property of
    the data -- and reporting it as an optimum would be misleading.
    """
    scores = [r.silhouette for r in sorted(results, key=lambda r: r.k)]
    return len(scores) > 1 and all(b <= a for a, b in zip(scores, scores[1:], strict=False))


def fit_clusters(X: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(X)
    return labels, km.cluster_centers_


def representatives(
    X: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    texts: list[str],
    n_per_cluster: int = 12,
) -> dict[int, list[str]]:
    """The n messages closest to each centroid.

    Nearest-to-centroid rather than a random sample from the cluster: the
    LLM gets the cluster's core meaning rather than its fringe, which
    produces sharper labels. The trade-off is that a label can look cleaner
    than the cluster actually is -- so the audit trail written alongside the
    taxonomy keeps cluster sizes and silhouette scores visible.
    """
    out: dict[int, list[str]] = {}
    for cid in sorted(set(int(c) for c in labels)):
        mask = labels == cid
        idx = np.flatnonzero(mask)
        d = np.linalg.norm(X[idx] - centroids[cid], axis=1)
        nearest = idx[np.argsort(d)[:n_per_cluster]]
        out[cid] = [texts[i] for i in nearest]
    return out


_LABEL_SYSTEM = (
    "You label clusters of real customer support messages sent to an airline "
    "on Twitter. Reply with JSON only, no prose and no markdown fences."
)

_LABEL_TEMPLATE = """These {n} messages were grouped together by a clustering algorithm.

{messages}

Give this cluster a short intent label and a one-sentence description.

Rules:
- label: snake_case, 1-3 words, describes what the CUSTOMER wants
- description: one sentence, under 20 words
- Describe what these messages actually have in common, even if that is
  something vague like general praise or an unclear complaint. Do not invent
  a specific airline issue that is not present in the messages.

Respond with exactly: {{"label": "...", "description": "..."}}"""


def build_label_prompt(messages: list[str]) -> list[dict[str, str]]:
    numbered = "\n".join(f"{i + 1}. {m}" for i, m in enumerate(messages))
    return [
        {"role": "system", "content": _LABEL_SYSTEM},
        {
            "role": "user",
            "content": _LABEL_TEMPLATE.format(n=len(messages), messages=numbered),
        },
    ]


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_HARMONY_RE = re.compile(
    r"<\|channel\|>analysis<\|message\|>.*?(?:<\|end\|>|<\|start\|>)", re.DOTALL
)


def _iter_json_objects(text: str) -> list[str]:
    """Every balanced {...} span in `text`, in order.

    Brace-counting rather than a regex: JSON objects nest, and a regex
    cannot match balanced delimiters. Quote-aware so a brace inside a
    string value doesn't throw off the depth count.
    """
    spans: list[str] = []
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for i, ch in enumerate(text):
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                spans.append(text[start : i + 1])
                start = -1
            elif depth < 0:
                depth = 0
    return spans


def parse_label_response(text: str) -> tuple[str, str]:
    """Extract label/description from an LLM response.

    Written defensively because the models available on free tiers are
    reasoning models (gpt-oss, qwen) that emit analysis before their
    answer, despite being told to return JSON only:

    * Reasoning blocks are stripped first -- they routinely contain braces,
      and an earlier version that grabbed the FIRST {...} span picked up a
      fragment of the reasoning instead of the actual answer.
    * Candidate objects are then tried LAST-first, since the real answer
      comes after any analysis.
    * Brace-counting rather than find("{")/rfind("}"), which spans from the
      first brace to the last and yields garbage when there are several.
    """
    cleaned = _THINK_RE.sub("", text)
    cleaned = _HARMONY_RE.sub("", cleaned).strip()

    if "```" in cleaned:
        parts = cleaned.split("```")
        # Odd indices are fenced blocks; prefer them, they're the intended payload.
        for block in parts[1::2]:
            block = block.removeprefix("json").strip()
            if "{" in block:
                cleaned = block
                break

    candidates = _iter_json_objects(cleaned)
    if not candidates:
        raise ValueError(f"no JSON object in response: {text[:300]!r}")

    for span in reversed(candidates):
        try:
            obj = json.loads(span)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        label = str(obj.get("label", "")).strip()
        if label:
            return label, str(obj.get("description", "")).strip()

    raise ValueError(f"no JSON object with a 'label' field: {text[:300]!r}")


_RETRY_SUFFIX = (
    "\n\nIMPORTANT: Output ONLY the raw JSON object. No reasoning, no "
    "explanation, no markdown. Start your reply with { and end it with }."
)


def label_clusters(
    client: LLMClient,
    reps: dict[int, list[str]],
    sizes: dict[int, int],
) -> list[IntentCluster]:
    clusters = []
    for cid, messages in sorted(reps.items()):
        prompt = build_label_prompt(messages)
        resp = client.complete("classifier", prompt)
        raw = resp.text
        try:
            label, description = parse_label_response(raw)
        except ValueError:
            # One retry with an explicit format reminder. Reasoning models
            # often comply on a second, blunter ask -- and a retry is far
            # cheaper than a human hand-labelling a 2,000-message cluster.
            strict = [
                *prompt[:-1],
                {**prompt[-1], "content": prompt[-1]["content"] + _RETRY_SUFFIX},
            ]
            raw = client.complete("classifier", strict).text
            try:
                label, description = parse_label_response(raw)
            except ValueError as exc:
                # Never swallow the cause: the raw response is retained on
                # the cluster so a failure is diagnosable from the artifact
                # rather than requiring a re-run to reproduce.
                label = f"unlabelled_cluster_{cid}"
                description = f"PARSE FAILED: {exc}"
        clusters.append(
            IntentCluster(
                cluster_id=cid,
                label=label,
                description=description,
                size=sizes.get(cid, 0),
                representatives=messages,
                raw_response=raw,
            )
        )
    return clusters


def write_taxonomy(clusters: list[IntentCluster], sweep: list[KSweepResult], out_dir: Path) -> None:
    """Write the taxonomy plus the audit trail that produced it.

    The audit trail is a deliverable, not a debug artifact: it is what shows
    the taxonomy was derived from the data rather than asserted.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    taxonomy = {
        "intents": [
            {"label": c.label, "description": c.description, "cluster_id": c.cluster_id}
            for c in clusters
        ]
        + [{"label": "other", "description": "Does not fit any discovered intent."}]
    }
    (out_dir / "taxonomy.json").write_text(json.dumps(taxonomy, indent=2), encoding="utf-8")

    (out_dir / "cluster_audit.json").write_text(
        json.dumps(
            {
                "k_sweep": [asdict(r) for r in sweep],
                "chosen_k": len(clusters),
                "clusters": [asdict(c) for c in clusters],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
