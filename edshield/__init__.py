"""edshield — local-first student-privacy layer for AI in education.

    from edshield import extract_pii, deidentify

    r = extract_pii("Hi, I'm Maya Chen, my email is maya.c@gmail.com")
    print([(e.label, e.text) for e in r.entities])

    print(deidentify("...", policy="coppa").deidentified_text)

Nothing here needs a network. With `edshield[hf]` installed and a model on
disk, `extract_pii` adds neural detection of student names; without it the
rule layer runs alone.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

from .types import Entity, AnalysisResult, DeidResult, PIILO_LABELS, ALL_LABELS
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
]


def _model_entities(text: str, model_name: Optional[str], device: str, threshold: float) -> List[Entity]:
    if model_name == "rules":
        return []
    try:
        from .ner import detect_model
        return detect_model(text, model_name=model_name, device=device, threshold=threshold)
    except Exception as exc:  # noqa: BLE001 - deliberate: never fail closed on the model layer
        import logging
        logging.getLogger("edshield").info("model layer unavailable (%s); using rules only", exc)
        return []


def analyze_text(
    text: str,
    model_name: Optional[str] = None,
    device: str = "cpu",
    labels: Optional[Iterable[str]] = None,
    threshold: float = 0.5,
    use_rules: bool = True,
    propagate: bool = True,
) -> AnalysisResult:
    """Detect PII entities. Combines rule detectors with the model (if any)."""
    ents: List[Entity] = []
    if use_rules:
        ents.extend(detect_rules(text, labels))
    model_ents = _model_entities(text, model_name, device, threshold)
    if labels:
        wanted = set(labels)
        model_ents = [e for e in model_ents if e.label in wanted]
    ents.extend(model_ents)
    if propagate:
        ents = propagate_names(text, ents)
    used = model_name if model_ents else "rules"
    return AnalysisResult(text=text, entities=resolve_overlaps(ents), model_name=used)


def extract_pii(text: str, model_name: Optional[str] = None, **kw) -> AnalysisResult:
    """Alias for analyze_text restricted to PII labels."""
    return analyze_text(text, model_name=model_name, labels=kw.pop("labels", ALL_LABELS), **kw)


def deidentify(
    text: str,
    method: str = "mask",
    policy: str = "ferpa",
    model_name: Optional[str] = None,
    device: str = "cpu",
    date_shift_days: Optional[int] = None,
    seed: Optional[int] = None,
    verify: bool = True,
) -> DeidResult:
    """Detect then transform. `verify=True` raises if any acted-on entity
    still appears verbatim in the output."""
    res = analyze_text(text, model_name=model_name, device=device)
    out = apply_deidentification(
        text, res.entities, method=method, policy=policy, date_shift_days=date_shift_days, seed=seed
    )
    if verify:
        leaks = check_no_leak(out.deidentified_text, out.entities)
        if leaks:
            raise RuntimeError(f"De-identification leak detected: {leaks}")
    return out
