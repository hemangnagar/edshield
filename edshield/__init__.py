"""edshield — local-first student-privacy layer for AI in education.

    from edshield import extract_pii, deidentify

    r = extract_pii("Hi, I'm Maya Chen, my email is maya.c@gmail.com")
    print([(e.label, e.text) for e in r.entities])

    print(deidentify("...", policy="coppa").deidentified_text)

Nothing here uses the network. With `edshield[hf]` installed and a model on
disk, `extract_pii` adds neural detection of student names; without it the
rule layer runs alone and says so with a warning.
"""

from __future__ import annotations

import os
import warnings
from typing import Iterable, List, Optional

from . import ner
from .ner import ModelUnavailableError
from .types import Entity, AnalysisResult, DeidResult, PIILO_LABELS, ALL_LABELS, MODEL_AUTHORITY_LABELS
from .rules import detect_rules, resolve_overlaps, propagate_names
from .deid import apply_deidentification, check_no_leak, available_policies, load_policy

__version__ = "0.1.0"
__all__ = [
    "analyze_text",
    "extract_pii",
    "deidentify",
    "check_no_leak",
    "available_policies",
    "load_policy",
    "Entity",
    "AnalysisResult",
    "DeidResult",
    "PIILO_LABELS",
    "ALL_LABELS",
    "MODEL_AUTHORITY_LABELS",
    "ModelUnavailableError",
]


def _model_entities(
    text: str, model_name: Optional[str], device: str, threshold: float, o_threshold: Optional[float]
) -> Optional[List[Entity]]:
    """Model detections, or None when no model ran. An empty list means it
    ran and found nothing.

    A model the caller asked for by name (or through EDSHIELD_MODEL) must
    work: failure raises ModelUnavailableError. With no model named, the
    default is tried and, if it is not available, the rules run alone and a
    RuntimeWarning says so.
    """
    if model_name == "rules":
        return None
    requested = model_name or os.environ.get("EDSHIELD_MODEL")
    try:
        return ner.detect_model(
            text, model_name=model_name, device=device, threshold=threshold, o_threshold=o_threshold
        )
    except Exception as exc:  # noqa: BLE001
        if requested:
            if isinstance(exc, ModelUnavailableError):
                raise
            raise ModelUnavailableError(f"model '{requested}' failed: {exc}") from exc
        warnings.warn(
            f"edshield: no model available ({exc}); running rules only, which misses most names. "
            "Pass model_name='rules' to choose this on purpose.",
            RuntimeWarning,
            stacklevel=3,
        )
        return None


def analyze_text(
    text: str,
    model_name: Optional[str] = None,
    device: str = "cpu",
    labels: Optional[Iterable[str]] = None,
    threshold: float = 0.5,
    use_rules: bool = True,
    propagate: bool = True,
    model_authority: Optional[Iterable[str]] = MODEL_AUTHORITY_LABELS,
    o_threshold: Optional[float] = None,
) -> AnalysisResult:
    """Detect PII entities. Combines rule detectors with the model (if any).

    When a model ran, the labels in `model_authority` come from the model
    only and the rules are not run for them. Without a model the rules cover
    every label. Pass `model_authority=()` to union both layers on all labels.

    `o_threshold` switches the model from argmax to the recall-first rule
    "entity whenever P(O) < o_threshold"; see `edshield.ner.decode`.
    """
    model_ents = _model_entities(text, model_name, device, threshold, o_threshold)
    model_ran = model_ents is not None
    model_ents = model_ents or []
    if labels:
        wanted = set(labels)
        model_ents = [e for e in model_ents if e.label in wanted]

    ents: List[Entity] = []
    if use_rules:
        rule_labels = list(labels) if labels else list(ALL_LABELS)
        if model_ran:
            deferred = set(model_authority or ())
            rule_labels = [l for l in rule_labels if l not in deferred]
        if rule_labels:
            ents.extend(detect_rules(text, rule_labels))
    ents.extend(model_ents)
    if propagate:
        ents = propagate_names(text, ents)
    used = model_name if model_ran else "rules"
    return AnalysisResult(text=text, entities=resolve_overlaps(ents), model_name=used)


def extract_pii(text: str, model_name: Optional[str] = None, **kw) -> AnalysisResult:
    """Alias for analyze_text restricted to PII labels."""
    return analyze_text(text, model_name=model_name, labels=kw.pop("labels", ALL_LABELS), **kw)


def deidentify(
    text: str,
    method: Optional[str] = None,
    policy: str = "ferpa",
    model_name: Optional[str] = None,
    device: str = "cpu",
    date_shift_days: Optional[int] = None,
    seed: Optional[int] = None,
    verify: bool = True,
    o_threshold: Optional[float] = None,
) -> DeidResult:
    """Detect then transform. `method=None` uses the policy's default method.
    `verify=True` raises if any acted-on value is still present verbatim.
    With `o_threshold`, the policy's confidence floor applies to 1 - P(O)."""
    res = analyze_text(text, model_name=model_name, device=device, o_threshold=o_threshold)
    out = apply_deidentification(
        text, res.entities, method=method, policy=policy, date_shift_days=date_shift_days, seed=seed
    )
    if verify and out.leaks:
        raise RuntimeError(f"De-identification leak detected: {out.leaks}")
    return out
