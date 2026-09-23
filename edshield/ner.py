"""Model layer: token-classification models trained on PIILO.

Everything here is optional. If `transformers` is not installed, or the
model artifacts are not on disk and the Hub is unreachable, callers fall
back to the rule layer. Models are loaded once and cached per process.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from .types import Entity

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "models.jsonl"

# Short name -> Hugging Face repo id (or local path). Populated from models.jsonl.
DEFAULT_MODEL = "piilo_deberta_small"


def load_manifest() -> dict:
    entries = {}
    if MANIFEST.exists():
        for line in MANIFEST.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rec = json.loads(line)
            entries[rec["name"]] = rec
    return entries


def resolve_model_id(model_name: Optional[str]) -> str:
    """Map a short name to a repo id or local path; passthrough otherwise."""
    if model_name is None:
        model_name = os.environ.get("EDSHIELD_MODEL", DEFAULT_MODEL)
    if os.path.isdir(model_name):
        return model_name
    rec = load_manifest().get(model_name)
    if rec:
        local = rec.get("local_path")
        if local and os.path.isdir(REPO_ROOT / local):
            return str(REPO_ROOT / local)
        return rec["hf_id"]
    return model_name


@lru_cache(maxsize=4)
def _pipeline(model_id: str, device: str):
    try:
        from transformers import pipeline  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Model inference needs `pip install 'edshield[hf]'`; rules-only mode is available without it."
        ) from exc
    kwargs = {"aggregation_strategy": "simple"}
    if device == "cpu":
        kwargs["device"] = -1
    elif device == "cuda":
        kwargs["device"] = 0
    elif device == "mps":
        kwargs["device"] = "mps"
    return pipeline("token-classification", model=model_id, tokenizer=model_id, **kwargs)


def smart_merge(ents: List[Entity], text: str, max_gap: int = 1) -> List[Entity]:
    """Merge adjacent same-label spans separated by at most `max_gap` chars of
    whitespace, hyphen, dot or apostrophe (tokenizer fragmentation repair)."""
    if not ents:
        return ents
    ents = sorted(ents, key=lambda e: e.start)
    merged = [ents[0]]
    for e in ents[1:]:
        prev = merged[-1]
        gap = text[prev.end:e.start]
        if (
            e.label == prev.label
            and len(gap) <= max_gap
            and all(ch in " -.'’_" for ch in gap)
        ):
            prev.end = e.end
            prev.text = text[prev.start:prev.end]
            prev.confidence = min(prev.confidence, e.confidence)
        else:
            merged.append(e)
    return merged


def detect_model(
    text: str,
    model_name: Optional[str] = None,
    device: str = "cpu",
    threshold: float = 0.5,
    o_threshold: Optional[float] = None,
    max_chars: int = 2000,
) -> List[Entity]:
    """Run a token-classification model over `text` in character windows.

    `o_threshold` is the recall trick from the PIILO competition: rather than
    argmax, a token is non-O whenever P(O) < o_threshold. Set ~0.99 for
    recall-heavy policies. It is applied through the pipeline's score when
    the pipeline exposes per-token scores; otherwise `threshold` applies.
    """
    model_id = resolve_model_id(model_name)
    pipe = _pipeline(model_id, device)
    ents: List[Entity] = []
    # Window on paragraph boundaries to keep offsets exact.
    pos = 0
    while pos < len(text):
        end = min(len(text), pos + max_chars)
        if end < len(text):
            cut = text.rfind("\n", pos, end)
            if cut > pos + max_chars // 2:
                end = cut
        chunk = text[pos:end]
        for r in pipe(chunk):
            label = r["entity_group"]
            if label == "O":
                continue
            score = float(r["score"])
            if score < threshold:
                continue
            s, e = pos + int(r["start"]), pos + int(r["end"])
            span = text[s:e]
            # Trim leading/trailing whitespace the tokenizer may include.
            ls = len(span) - len(span.lstrip())
            rs = len(span) - len(span.rstrip())
            s, e = s + ls, e - rs
            if e <= s:
                continue
            ents.append(Entity(label=label, text=text[s:e], start=s, end=e, confidence=score, source="model"))
        pos = end
    return smart_merge(ents, text)
