"""Core data types for edshield.

Labels follow the PIILO / Kaggle "PII Data Detection" schema so that models
trained on that corpus drop straight into the runtime. Rule-based detectors
add a few labels the corpus does not cover (SSN, DATE).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional

# The seven PIILO labels (student-writing corpus, CC BY 4.0).
PIILO_LABELS = [
    "NAME_STUDENT",
    "EMAIL",
    "USERNAME",
    "ID_NUM",
    "PHONE_NUM",
    "URL_PERSONAL",
    "STREET_ADDRESS",
]

# Extra labels produced only by the rule layer.
RULE_ONLY_LABELS = ["SSN", "DATE"]

ALL_LABELS = PIILO_LABELS + RULE_ONLY_LABELS

# Labels where the model is the authority whenever one is loaded: the rules
# for these are recall-oriented fallbacks that over-flag ordinary essay text.
# Everything else stays with the rules, with the model as a second opinion.
MODEL_AUTHORITY_LABELS = ["NAME_STUDENT", "ID_NUM", "STREET_ADDRESS"]


@dataclass
class Entity:
    label: str
    text: str
    start: int
    end: int
    confidence: float = 1.0
    source: str = "rules"  # "rules" | "model"

    def overlaps(self, other: "Entity") -> bool:
        return self.start < other.end and other.start < self.end

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AnalysisResult:
    text: str
    entities: List[Entity] = field(default_factory=list)
    model_name: Optional[str] = None

    def by_label(self) -> dict:
        out: dict = {}
        for e in self.entities:
            out.setdefault(e.label, []).append(e)
        return out

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "entities": [e.to_dict() for e in self.entities],
            "model_name": self.model_name,
        }


@dataclass
class DeidResult:
    original_text: str
    deidentified_text: str
    entities: List[Entity]
    method: str
    policy: str
    replacements: dict = field(default_factory=dict)  # original -> surrogate

    def to_dict(self) -> dict:
        return {
            "deidentified_text": self.deidentified_text,
            "entities": [e.to_dict() for e in self.entities],
            "method": self.method,
            "policy": self.policy,
            "n_entities": len(self.entities),
        }
