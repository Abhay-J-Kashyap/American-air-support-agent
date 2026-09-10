"""The curated intent taxonomy.

Separate from `intents.py` on purpose. `intents.py` DISCOVERS a proposal and
rewrites its output on every run; this module loads the human-curated file
that downstream stages actually depend on. Conflating them would mean a
`discover-intents` re-run silently clobbers the taxonomy the classifier,
retriever and escalation policy are all built against.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

TAXONOMY_PATH = Path("config/taxonomy.yaml")


class RiskTier(str, Enum):
    """Drives escalation. Ordered by severity."""

    AUTO_OK = "auto_ok"
    REVIEW = "review"
    ALWAYS_ESCALATE = "always_escalate"


class IntentSource(str, Enum):
    CLUSTER = "cluster"
    KEYWORD = "keyword"


class Intent(BaseModel):
    label: str
    description: str
    source: IntentSource
    risk_tier: RiskTier
    approx_share: float | None = None
    note: str | None = None

    @field_validator("label")
    @classmethod
    def label_is_snake_case(cls, v: str) -> str:
        if not v.replace("_", "").isalnum() or v != v.lower():
            raise ValueError(f"label must be lowercase snake_case, got {v!r}")
        return v


class Taxonomy(BaseModel):
    intents: list[Intent] = Field(min_length=2)

    @field_validator("intents")
    @classmethod
    def labels_are_unique(cls, v: list[Intent]) -> list[Intent]:
        labels = [i.label for i in v]
        dupes = {x for x in labels if labels.count(x) > 1}
        if dupes:
            raise ValueError(f"duplicate intent labels: {sorted(dupes)}")
        return v

    @field_validator("intents")
    @classmethod
    def has_other_escape_hatch(cls, v: list[Intent]) -> list[Intent]:
        # Without an explicit `other`, a classifier is forced to assign every
        # message to a real intent, which manufactures confident wrong labels
        # for the genuinely unclassifiable.
        if not any(i.label == "other" for i in v):
            raise ValueError("taxonomy must include an 'other' intent")
        return v

    @property
    def labels(self) -> list[str]:
        return [i.label for i in self.intents]

    def get(self, label: str) -> Intent:
        for i in self.intents:
            if i.label == label:
                return i
        raise KeyError(f"unknown intent {label!r}; have {self.labels}")

    def risk_tier(self, label: str) -> RiskTier:
        return self.get(label).risk_tier

    def always_escalate_labels(self) -> list[str]:
        return [i.label for i in self.intents if i.risk_tier is RiskTier.ALWAYS_ESCALATE]

    def prompt_block(self) -> str:
        """Rendered intent list for a classification prompt."""
        return "\n".join(f"- {i.label}: {i.description}" for i in self.intents)


def load_taxonomy(path: Path | None = None) -> Taxonomy:
    path = path or TAXONOMY_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `discover-intents` to generate a proposal, "
            "then curate it into this file by hand."
        )
    return Taxonomy.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
