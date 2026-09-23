"""De-identification: turn detected entities into masked, surrogate, hashed
or date-shifted text under a named policy.

Policies live in edshield/policies/*.yaml and say which labels to act on,
which method to use per label, and the minimum confidence to honour.
"""

from __future__ import annotations

import hashlib
import os
import random
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .types import Entity, DeidResult

POLICY_DIR = Path(__file__).resolve().parent / "policies"


def load_policy(name_or_path: str) -> dict:
    p = Path(name_or_path)
    if not p.exists():
        p = POLICY_DIR / f"{name_or_path}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"Unknown policy '{name_or_path}'. Built-ins: {available_policies()}")
    with open(p) as fh:
        return yaml.safe_load(fh)


def available_policies() -> List[str]:
    return sorted(p.stem for p in POLICY_DIR.glob("*.yaml"))


# --- Surrogates -------------------------------------------------------------

class SurrogateFactory:
    """Consistent fake replacements: the same original value always maps to
    the same surrogate within one document (or one session if reused)."""

    def __init__(self, seed: Optional[int] = None, locale: str = "en_US"):
        try:
            from faker import Faker  # noqa: WPS433
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("`replace` needs `pip install faker`") from exc
        self.fake = Faker(locale)
        if seed is not None:
            self.fake.seed_instance(seed)
        self._memo: Dict[str, str] = {}

    def for_entity(self, ent: Entity) -> str:
        key = f"{ent.label}::{ent.text.strip().lower()}"
        if key in self._memo:
            return self._memo[key]
        f = self.fake
        n_words = len(ent.text.split())
        if ent.label == "NAME_STUDENT":
            val = f.first_name() if n_words == 1 else f"{f.first_name()} {f.last_name()}"
        elif ent.label == "EMAIL":
            val = f.free_email()
        elif ent.label == "USERNAME":
            val = f.user_name()
        elif ent.label == "ID_NUM":
            digits = sum(ch.isdigit() for ch in ent.text)
            val = "".join(random.choice("0123456789") for _ in range(max(6, digits)))
        elif ent.label == "PHONE_NUM":
            val = f.numerify("555-###-####")
        elif ent.label == "URL_PERSONAL":
            val = f"https://example.com/{f.user_name()}"
        elif ent.label == "STREET_ADDRESS":
            val = f.street_address()
        elif ent.label == "SSN":
            val = f.numerify("900-##-####")
        elif ent.label == "DATE":
            val = f.date(pattern="%m/%d/%Y")
        else:
            val = f"[{ent.label}]"
        self._memo[key] = val
        return val


def _hash(value: str, salt: str, length: int = 10) -> str:
    return hashlib.sha256((salt + value).encode("utf-8")).hexdigest()[:length]


_DATE_FORMATS = ["%m/%d/%Y", "%m-%d-%Y", "%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%B %d %Y"]


def _shift_date(value: str, days: int) -> str:
    for fmt in _DATE_FORMATS:
        try:
            d = datetime.strptime(value.replace("Sept", "Sep"), fmt)
            return (d + timedelta(days=days)).strftime(fmt)
        except ValueError:
            continue
    return "[DATE]"


# --- Main entry -------------------------------------------------------------

def apply_deidentification(
    text: str,
    entities: List[Entity],
    method: str = "mask",
    policy: str = "ferpa",
    date_shift_days: Optional[int] = None,
    seed: Optional[int] = None,
    salt: Optional[str] = None,
) -> DeidResult:
    pol = load_policy(policy)
    rules: dict = pol.get("labels", {})
    default_method = pol.get("default_method", method)
    min_conf = float(pol.get("min_confidence", 0.0))
    salt = salt or os.environ.get("EDSHIELD_HASH_SALT", "edshield")
    if date_shift_days is None:
        date_shift_days = int(pol.get("date_shift_days", 0)) or random.randint(-365, 365)

    factory: Optional[SurrogateFactory] = None
    replacements: Dict[str, str] = {}

    def render(ent: Entity) -> str:
        nonlocal factory
        cfg = rules.get(ent.label, {})
        # Precedence: an explicit per-label method in the policy always wins;
        # otherwise the caller's method; otherwise the policy default.
        if isinstance(cfg, dict) and cfg.get("method"):
            m = cfg["method"]
        else:
            m = method or default_method
        if m == "mask":
            out = cfg.get("mask", f"[{ent.label}]") if isinstance(cfg, dict) else f"[{ent.label}]"
        elif m == "replace":
            if factory is None:
                factory = SurrogateFactory(seed=seed)
            out = factory.for_entity(ent)
        elif m == "hash":
            out = _hash(ent.text, salt)
        elif m == "shift_dates":
            out = _shift_date(ent.text, date_shift_days) if ent.label == "DATE" else f"[{ent.label}]"
        elif m == "keep":
            out = ent.text
        else:
            raise ValueError(f"Unknown method '{m}'")
        replacements[ent.text] = out
        return out

    # Filter by policy: label must be enabled and confidence above threshold.
    acted: List[Entity] = []
    for e in entities:
        cfg = rules.get(e.label)
        enabled = cfg is not None and (cfg is True or (isinstance(cfg, dict) and cfg.get("enabled", True)))
        label_min = float(cfg.get("min_confidence", min_conf)) if isinstance(cfg, dict) else min_conf
        if enabled and e.confidence >= label_min:
            acted.append(e)

    acted.sort(key=lambda e: e.start)
    out_parts: List[str] = []
    cursor = 0
    for e in acted:
        out_parts.append(text[cursor:e.start])
        out_parts.append(render(e))
        cursor = e.end
    out_parts.append(text[cursor:])
    return DeidResult(
        original_text=text,
        deidentified_text="".join(out_parts),
        entities=acted,
        method=method,
        policy=policy,
        replacements=replacements,
    )


def check_no_leak(deidentified_text: str, entities: List[Entity]) -> List[str]:
    """Return original entity strings that still appear verbatim in the output."""
    leaks = []
    for e in entities:
        if len(e.text.strip()) >= 3 and re.search(re.escape(e.text.strip()), deidentified_text):
            leaks.append(e.text)
    return leaks
