"""Model layer: token-classification models trained on PIILO.

Everything here is optional. Models are loaded once and cached per process,
and only from disk: the local path in models.jsonl, a directory you pass, or
the Hugging Face cache. Nothing is downloaded unless EDSHIELD_ALLOW_DOWNLOAD=1
is set or `allow_download=True` is passed.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .types import Entity

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "models.jsonl"

# Short name -> Hugging Face repo id (or local path). Populated from models.jsonl.
DEFAULT_MODEL = "piilo_deberta_small"

MAX_TOKENS = 1024  # the context length the models are trained with


class ModelUnavailableError(RuntimeError):
    """The requested model could not be loaded or run."""


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


def downloads_allowed() -> bool:
    return os.environ.get("EDSHIELD_ALLOW_DOWNLOAD", "").strip().lower() in {"1", "true", "yes"}


_LOAD_FAILURES: Dict[tuple, Exception] = {}


@lru_cache(maxsize=4)
def _load(model_id: str, device: str, allow_download: bool):
    try:
        import torch  # noqa: WPS433,F401
        from transformers import AutoModelForTokenClassification, AutoTokenizer  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover
        raise ModelUnavailableError(
            "Model inference needs `pip install 'edshield[hf]'`; rules-only mode is available without it."
        ) from exc
    local_only = not allow_download
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=local_only)
    model = AutoModelForTokenClassification.from_pretrained(model_id, local_files_only=local_only)
    model.to(device).eval()
    return tok, model


def _load_once(model_id: str, device: str, allow_download: bool):
    """Like _load, but a failure is remembered so it is not retried per call."""
    key = (model_id, device, allow_download)
    if key in _LOAD_FAILURES:
        raise _LOAD_FAILURES[key]
    try:
        return _load(*key)
    except Exception as exc:  # noqa: BLE001
        if not isinstance(exc, ModelUnavailableError):
            hint = "" if allow_download else " (downloads are off; set EDSHIELD_ALLOW_DOWNLOAD=1 to fetch it)"
            wrapped = ModelUnavailableError(f"cannot load model '{model_id}'{hint}: {exc}")
            wrapped.__cause__ = exc
            exc = wrapped
        _LOAD_FAILURES[key] = exc
        raise exc


def _predict(chunk: str, model_id: str, device: str, allow_download: bool):
    """Token offsets, per-token class probabilities and the id->label map for `chunk`."""
    import torch  # noqa: WPS433

    tok, model = _load_once(model_id, device, allow_download)
    enc = tok(chunk, return_offsets_mapping=True, return_tensors="pt")
    if enc["input_ids"].shape[1] > MAX_TOKENS and len(chunk) > 200:
        # Unusually dense text: split at whitespace and run the halves.
        cut = _cut(chunk, len(chunk) // 4, len(chunk) // 2)
        o1, p1, id2label = _predict(chunk[:cut], model_id, device, allow_download)
        o2, p2, _ = _predict(chunk[cut:], model_id, device, allow_download)
        return o1 + [(s + cut, e + cut) for s, e in o2], p1 + p2, id2label
    offsets = [tuple(o) for o in enc.pop("offset_mapping")[0].tolist()]
    with torch.no_grad():
        logits = model(**{k: v.to(model.device) for k, v in enc.items()}).logits[0]
    probs = torch.softmax(logits.float(), dim=-1).cpu().tolist()
    return offsets, probs, dict(model.config.id2label)


# --- Windowing ------------------------------------------------------------

def _cut(text: str, lo: int, hi: int, prefer_newline: bool = True) -> int:
    """Best place to cut in (lo, hi]: after a newline, else after whitespace, else hi."""
    nl = text.rfind("\n", lo, hi) if prefer_newline else -1
    if nl != -1:
        return nl + 1
    for i in range(hi, lo, -1):
        if text[i - 1].isspace():
            return i
    return hi


def windows(text: str, max_chars: int = 2000, overlap: int = 200) -> List[Tuple[int, int, int, int]]:
    """Split `text` into overlapping windows cut at whitespace.

    Returns (start, end, keep_from, keep_to) per window. The keep ranges tile
    the text exactly, and each boundary between them sits in the middle of an
    overlap, so whatever is kept from a window was seen with context on both
    sides.
    """
    n = len(text)
    overlap = max(0, min(overlap, max_chars // 4))
    spans: List[Tuple[int, int]] = []
    pos = 0
    while True:
        end = min(n, pos + max_chars)
        if end < n:
            end = _cut(text, pos + max_chars // 2, end)
        spans.append((pos, end))
        if end >= n:
            break
        back = end - overlap
        pos = max(pos + 1, _cut(text, max(pos, back - overlap), back, prefer_newline=False))
    out = []
    keep_from = 0
    for i, (s, e) in enumerate(spans):
        keep_to = n if i == len(spans) - 1 else (spans[i + 1][0] + e) // 2
        out.append((s, e, keep_from, keep_to))
        keep_from = keep_to
    return out


# --- Decoding -------------------------------------------------------------

_JOINERS = " -.'’_"


def decode(
    text: str,
    offsets: Sequence[Tuple[int, int]],
    probs: Sequence[Sequence[float]],
    id2label: Dict[int, str],
    threshold: float = 0.5,
    o_threshold: Optional[float] = None,
) -> List[Entity]:
    """Turn per-token probabilities into entity spans.

    By default a token takes its most likely class and a span is kept when
    its mean probability reaches `threshold`. With `o_threshold` (the recall
    trick from the PIILO competition) a token is an entity whenever
    P(O) < o_threshold, labelled with its most likely non-O class, and the
    span's confidence is the probability that it is an entity at all, 1 - P(O).
    """
    o_id = next(i for i, l in id2label.items() if l == "O")
    tokens = []  # (start, end, label, score)
    for (s, e), p in zip(offsets, probs):
        if s == e:
            continue  # special token
        while s < e and text[s].isspace():
            s += 1
        if s == e:
            continue
        if o_threshold is not None:
            if p[o_id] >= o_threshold:
                continue
            best = max((i for i in range(len(p)) if i != o_id), key=lambda i: p[i])
            score = 1.0 - p[o_id]
        else:
            best = max(range(len(p)), key=lambda i: p[i])
            if best == o_id:
                continue
            score = p[best]
        tokens.append((s, e, id2label[best].split("-", 1)[-1], score))

    # Join word pieces and adjacent words of the same label.
    groups: List[list] = []  # [start, end, label, [scores]]
    for s, e, label, score in tokens:
        if groups:
            g = groups[-1]
            gap = text[g[1]:s]
            if g[2] == label and len(gap) <= 1 and all(ch in _JOINERS for ch in gap):
                g[1] = max(g[1], e)
                g[3].append(score)
                continue
        groups.append([s, e, label, [score]])

    ents: List[Entity] = []
    for s, e, label, scores in groups:
        conf = sum(scores) / len(scores)
        if o_threshold is None and conf < threshold:
            continue
        # Cover the whole word: masking half a name leaves the other half behind.
        while s > 0 and text[s - 1].isalnum():
            s -= 1
        while e < len(text) and text[e].isalnum():
            e += 1
        if ents and ents[-1].label == label and s <= ents[-1].end:
            ents[-1].end = max(ents[-1].end, e)
            ents[-1].text = text[ents[-1].start:ents[-1].end]
            ents[-1].confidence = min(ents[-1].confidence, conf)
            continue
        ents.append(Entity(label=label, text=text[s:e], start=s, end=e, confidence=conf, source="model"))
    return ents


def detect_model(
    text: str,
    model_name: Optional[str] = None,
    device: str = "cpu",
    threshold: float = 0.5,
    o_threshold: Optional[float] = None,
    max_chars: int = 2000,
    overlap: int = 200,
    allow_download: Optional[bool] = None,
) -> List[Entity]:
    """Run a token-classification model over `text` in overlapping windows.

    See `decode` for `threshold` and `o_threshold`. Raises
    ModelUnavailableError if the model cannot be loaded.
    """
    model_id = resolve_model_id(model_name)
    if allow_download is None:
        allow_download = downloads_allowed()
    ents: List[Entity] = []
    for start, end, keep_from, keep_to in windows(text, max_chars, overlap):
        chunk = text[start:end]
        if not chunk.strip():
            continue
        offsets, probs, id2label = _predict(chunk, model_id, device, allow_download)
        for e in decode(chunk, offsets, probs, id2label, threshold, o_threshold):
            e.start += start
            e.end += start
            if keep_from <= (e.start + e.end) // 2 < keep_to:
                ents.append(e)
    return ents
