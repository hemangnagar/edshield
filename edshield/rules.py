"""Deterministic PII detectors.

These need no model and no network. They cover the identifiers that are
convention-bound (emails, phones, URLs, IDs) where regex + validation beats
a neural model on precision, and they give the runtime a floor when a model
is unavailable. Student names are the hard case and are left to the model
layer; the rules only catch names introduced by explicit cues
("My name is ...").
"""

from __future__ import annotations

import re
from typing import Iterable, List

from .types import Entity

# --- Patterns -------------------------------------------------------------

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# US-centric phone patterns; international forms with + prefix.
PHONE_RE = re.compile(
    r"(?<![\w.])(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?!\w)(?!\.\d)"
    r"|(?<![\w.])\+\d{1,3}[\s.-]?\d{2,4}(?:[\s.-]?\d{2,4}){2,3}(?!\w)(?!\.\d)"
)

SSN_RE = re.compile(r"(?<!\d)(?!000|666|9\d\d)\d{3}[- ](?!00)\d{2}[- ](?!0000)\d{4}(?!\d)")

URL_RE = re.compile(
    r"\b(?:https?://|www\.)[^\s<>\"']+|\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:com|net|org|io|me|edu|co|dev|app|info)(?:/[^\s<>\"']*)?",
    re.IGNORECASE,
)

# Handles and usernames: @handle, "username: xyz", "user: xyz"
USERNAME_CUES = r"username|user name|handle|login|user id|userid|screen name|gamertag|discord|instagram|insta|snap|snapchat|tiktok|twitter|github|roblox|xbox|psn"
USERNAME_RE = re.compile(
    r"(?<![\w@])@([A-Za-z0-9_.]{3,30})\b"                                                    # @handle
    rf"|\b(?:{USERNAME_CUES})(?![./])\s*(?:is|:|=|-)\s*@?([A-Za-z0-9_][A-Za-z0-9_.]{{2,29}})\b"  # cue + separator + value
    rf"|\b(?:{USERNAME_CUES})\b[^.\n@]{{0,40}}?\b(?:is|:)\s*@?([A-Za-z0-9_][A-Za-z0-9_.]{{2,29}})\b"  # cue ... is value
    rf"|\b(?:{USERNAME_CUES})(?![./])\s+@?([A-Za-z]*[0-9_.][A-Za-z0-9_.]*)\b",              # cue + value that looks like a handle
    re.IGNORECASE,
)

# Student/other ID numbers: label-cued or long alphanumeric tokens.
ID_CUE_RE = re.compile(
    r"\b(?:student\s*(?:id|number|no\.?|#)|id\s*(?:number|no\.?|#)?|osis|sid|lunch\s*(?:number|#)|badge|library\s*card|account\s*(?:number|no\.?|#)|roll\s*(?:number|no\.?))\s*(?:is|:|=|-|#)?\s*([A-Z0-9][A-Z0-9-]{4,24})\b",
    re.IGNORECASE,
)
ID_BARE_RE = re.compile(r"(?<![\w-])(?=[A-Z0-9-]{7,24}(?![\w-]))(?=[A-Z-]*\d)[A-Z0-9-]{7,24}(?![\w-])")

STREET_SUFFIXES = (
    "expressway|extensions|throughway|trafficway|crossroad|extension|junctions|mountains|stravenue|underpass|causeway|crescent|crossing|junction|motorway|mountain|overpass|parkways|turnpike|villages|centers|circles|corners|estates|freeway|gardens|gateway|harbors|heights|highway|islands|landing|meadows|mission|orchard|parkway|passage|prairie|springs|squares|station|streets|terrace|valleys|viaduct|village|avenue|branch|bridge|brooks|bypass|canyon|center|circle|cliffs|common|corner|course|courts|divide|drives|estate|fields|forest|forges|garden|greens|groves|harbor|hollow|island|knolls|lights|manors|meadow|plains|points|radial|rapids|ridges|shoals|shores|skyway|spring|square|stream|street|summit|tunnel|unions|valley|alley|brook|burgs|cliff|court|coves|creek|crest|curve|drive|falls|ferry|field|flats|fords|forge|forks|glens|green|grove|haven|hills|inlet|knoll|lakes|light|locks|lodge|manor|mills|mount|parks|pines|place|plain|plaza|point|ports|ranch|rapid|ridge|river|roads|route|shoal|shore|spurs|trace|track|trail|union|views|ville|vista|walks|wells|blvd|burg|camp|cape|club|cove|dale|fall|flat|ford|fork|fort|glen|hill|isle|keys|lake|land|lane|loaf|lock|loop|mall|mews|mill|neck|oval|park|pass|path|pike|pine|pkwy|port|ramp|rest|road|spur|view|walk|wall|ways|well|ave|cir|dam|hwy|key|row|rue|run|ter|trl|via|way|ct|dr|ln|pl|rd|sq|st"
)
STREET_ADDRESS_RE = re.compile(
    rf"\b\d{{1,6}}\s+(?:[A-Z][a-zA-Z0-9]*\.?\s+){{0,4}}(?:{STREET_SUFFIXES})\.?"
    r"(?:\s*,?\s*(?:apt|apartment|unit|suite|ste|#)\.?\s*\w+)?"
    r"(?:\s*,?\s*[A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)?\s*,?\s*[A-Z]{2}\s*\d{5}(?:-\d{4})?)?",
    re.IGNORECASE,
)

DATE_RE = re.compile(
    r"\b(?:0?[1-9]|1[0-2])[/.-](?:0?[1-9]|[12]\d|3[01])[/.-](?:19|20)\d{2}\b"
    r"|\b(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])\b"
    r"|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+(?:19|20)\d{2}\b",
    re.IGNORECASE,
)

# Names introduced by an explicit cue. Strong cues accept a single first
# name ("this is Marcus"); weak cues need two or three capitalised tokens.
_NAME_TOKENS = r"[A-Z][a-z'’-]+(?:\s+(?:[A-Z][a-z'’-]+|[A-Z]\.?)){0,2}"
NAME_CUE_RE = re.compile(
    rf"\b(?i:my name is|my name's|i am|i'm|im|this is|it's|its|name:|sincerely,?|thanks,?|thank you,?|regards,?|love,?|yours,?)\s+({_NAME_TOKENS})"
    rf"|\b(?i:written by|by|from|interviewed|with|and)\s+([A-Z][a-z'’-]+(?:\s+(?:[A-Z][a-z'’-]+|[A-Z]\.?)){{1,2}})"
)
NAME_STOPWORDS = {
    "The", "This", "That", "What", "When", "Where", "Why", "How", "Which", "Who",
    "In", "On", "At", "For", "From", "With", "About", "Going", "Not", "Very",
    "Really", "Also", "Then", "Now", "Here", "There", "So", "But", "And", "Or",
    "A", "An", "It", "We", "They", "He", "She", "You", "I", "Sure", "Sorry",
    "Hi", "Hello", "Hey", "Oh", "Thanks", "Yeah", "Well", "Um", "Hmm", "Wait", "Ms", "Mr", "Mrs", "Dr", "Miss", "Mx", "Tutor", "Student", "Teacher", "Okay", "Ok",
    "Yes", "No", "Just", "Still", "Only", "All", "My", "Your", "Our", "Their", "Good",
}

# URLs at reference domains are not personal.
URL_ALLOWLIST = (
    "wikipedia.org", "khanacademy.org", "google.com", "youtube.com", "youtu.be",
    "britannica.com", "nasa.gov", ".gov", "nytimes.com", "bbc.com", "cnn.com",
    "coursera.org", "edx.org", "doi.org", "jstor.org", "arxiv.org", "scholar.google",
    "quizlet.com", "desmos.com", "wolframalpha.com", "stackoverflow.com",
)

PERSONAL_URL_HINTS = (
    "linkedin.com/in/", "facebook.com/", "instagram.com/", "twitter.com/", "x.com/",
    "tiktok.com/@", "github.com/", "youtube.com/@", "youtube.com/c/", "medium.com/@",
    "wordpress.com", "blogspot.com", "wixsite.com", "carrd.co", "about.me", "linktr.ee",
)


def _url_is_personal(url: str) -> bool:
    u = url.lower()
    if any(h in u for h in PERSONAL_URL_HINTS):
        return True
    if any(a in u for a in URL_ALLOWLIST):
        return False
    # Default: unknown domains in student writing are treated as personal.
    return True


def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


# --- Detection ------------------------------------------------------------

def _add(ents: List[Entity], label: str, m_start: int, m_end: int, text: str, conf: float) -> None:
    span = text[m_start:m_end]
    # Trim trailing punctuation that regexes sometimes swallow.
    while span and span[-1] in ".,;:)]}'\"":
        span = span[:-1]
        m_end -= 1
    if not span.strip():
        return
    ents.append(Entity(label=label, text=span, start=m_start, end=m_end, confidence=conf, source="rules"))


def detect_rules(text: str, labels: Iterable[str] | None = None) -> List[Entity]:
    """Run every rule detector over `text` and return non-overlapping entities."""
    wanted = set(labels) if labels else None
    ents: List[Entity] = []

    def want(label: str) -> bool:
        return wanted is None or label in wanted

    if want("EMAIL"):
        for m in EMAIL_RE.finditer(text):
            _add(ents, "EMAIL", m.start(), m.end(), text, 0.99)

    if want("SSN"):
        for m in SSN_RE.finditer(text):
            _add(ents, "SSN", m.start(), m.end(), text, 0.97)

    if want("PHONE_NUM"):
        for m in PHONE_RE.finditer(text):
            digits = re.sub(r"\D", "", m.group())
            if 10 <= len(digits) <= 15:
                _add(ents, "PHONE_NUM", m.start(), m.end(), text, 0.95)

    if want("URL_PERSONAL"):
        for m in URL_RE.finditer(text):
            # Domains inside an email address are handled by the EMAIL rule.
            if "@" in text[max(0, m.start() - 1):m.start()]:
                continue
            if _url_is_personal(m.group()):
                _add(ents, "URL_PERSONAL", m.start(), m.end(), text, 0.9)

    if want("USERNAME"):
        for m in USERNAME_RE.finditer(text):
            g = next((i for i in (1, 2, 3, 4) if m.group(i)), None)
            if g is None:
                continue
            # Skip handles that are actually part of an email.
            if EMAIL_RE.search(text[max(0, m.start() - 1):m.end() + 1]):
                continue
            _add(ents, "USERNAME", m.start(g), m.end(g), text, 0.9)

    if want("ID_NUM"):
        for m in ID_CUE_RE.finditer(text):
            _add(ents, "ID_NUM", m.start(1), m.end(1), text, 0.92)
        for m in ID_BARE_RE.finditer(text):
            tok = m.group()
            digits = re.sub(r"\D", "", tok)
            conf = 0.6
            if len(digits) >= 13 and _luhn_ok(digits):
                conf = 0.95  # card-like
            elif len(digits) >= 7:
                conf = 0.75
            _add(ents, "ID_NUM", m.start(), m.end(), text, conf)

    if want("STREET_ADDRESS"):
        for m in STREET_ADDRESS_RE.finditer(text):
            _add(ents, "STREET_ADDRESS", m.start(), m.end(), text, 0.85)

    if want("DATE"):
        for m in DATE_RE.finditer(text):
            _add(ents, "DATE", m.start(), m.end(), text, 0.8)

    if want("NAME_STUDENT"):
        for m in NAME_CUE_RE.finditer(text):
            g = 1 if m.group(1) else 2
            name = m.group(g)
            if name.split()[0] in NAME_STOPWORDS:
                continue
            _add(ents, "NAME_STUDENT", m.start(g), m.end(g), text, 0.7)

    return resolve_overlaps(ents)


def resolve_overlaps(ents: List[Entity]) -> List[Entity]:
    """Keep the higher-confidence, then longer, entity when spans overlap.

    Model entities win ties against rule entities of the same length.
    """
    ordered = sorted(
        ents,
        key=lambda e: (-e.confidence, -(e.end - e.start), 0 if e.source == "model" else 1, e.start),
    )
    kept: List[Entity] = []
    for e in ordered:
        if not any(e.overlaps(k) for k in kept):
            kept.append(e)
    kept.sort(key=lambda e: e.start)
    return kept


def propagate_names(text: str, ents: List[Entity], min_token_len: int = 3) -> List[Entity]:
    """Tag every other occurrence of a detected student name (full name and
    each capitalised name token) so a name caught once is caught everywhere."""
    names = [e for e in ents if e.label == "NAME_STUDENT"]
    if not names:
        return ents
    out = list(ents)
    seen = {(e.start, e.end) for e in ents}
    needles = set()
    for e in names:
        full = e.text.strip()
        needles.add((full, e.confidence))
        for tok in full.split():
            tok = tok.strip(".")
            if len(tok) >= min_token_len and tok[0].isupper() and tok not in NAME_STOPWORDS:
                needles.add((tok, e.confidence * 0.9))
    for needle, conf in sorted(needles, key=lambda n: -len(n[0])):
        for m in re.finditer(r"(?<![\w'’])" + re.escape(needle) + r"(?![\w'’])", text):
            if (m.start(), m.end()) in seen:
                continue
            # skip if inside an email/url/username span already found
            if any(k.start <= m.start() and m.end() <= k.end for k in ents if k.label != "NAME_STUDENT"):
                continue
            out.append(Entity("NAME_STUDENT", needle, m.start(), m.end(), conf, source="propagated"))
            seen.add((m.start(), m.end()))
    return out
