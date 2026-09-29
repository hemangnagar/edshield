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

NAME_LABELS = ["NAME_STUDENT", "NAME_RELATED"]

# --- Patterns -------------------------------------------------------------

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# US-centric phone patterns; international forms with + prefix.
PHONE_RE = re.compile(
    r"(?<![\w.])(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?:[ \t]?(?:x|ext\.?)[ \t]?\d{1,6})?(?!\w)(?!\.\d)"
    r"|(?<![\w.])\+\d{1,3}[\s.-]?\d{2,4}(?:[\s.-]?\d{2,4}){2,3}(?!\w)(?!\.\d)"
)

SSN_RE = re.compile(r"(?<!\d)(?!000|666|9\d\d)\d{3}[- ](?!00)\d{2}[- ](?!0000)\d{4}(?!\d)")

URL_RE = re.compile(r"\b(?:https?://|www\.)[^\s<>\"']+", re.IGNORECASE)
# Bare domains are case-sensitive and need a boundary after the TLD, so a
# missing space after a full stop ("5.Approach", "solution.Education") is
# not read as a domain.
BARE_DOMAIN_RE = re.compile(
    r"(?<![\w@./-])(?=[a-z0-9-]*[a-z])[a-z0-9-]{2,}(?:\.[a-z0-9-]+)*\.(?:com|net|org|io|me|edu|co|dev|app|info)(?![A-Za-z0-9-])(?:/[^\s<>\"']*)?"
)

# Handles and usernames: @handle, "username: xyz", "user: xyz"
# Strong cues only ever introduce an account name. Platform cues are also
# ordinary words ("handle the issue", "twitter is great"), so after them the
# value has to look like a handle or follow an explicit ":" / "=".
USERNAME_STRONG_CUES = r"username|user name|user id|userid|screen name|gamertag"
USERNAME_PLATFORM_CUES = r"handle|login|discord|instagram|insta|snap|snapchat|tiktok|twitter|github|roblox|xbox|psn"
USERNAME_CUES = rf"{USERNAME_STRONG_CUES}|{USERNAME_PLATFORM_CUES}"
USERNAME_RE = re.compile(
    r"(?<![\w@])@([A-Za-z0-9_.]{3,30})\b"                                                    # @handle
    rf"|\b(?:{USERNAME_CUES})(?![./\w])[ \t]*[:=][ \t]*@?([A-Za-z0-9_][A-Za-z0-9_.]{{2,29}})\b"  # cue: value
    rf"|\b(?:{USERNAME_STRONG_CUES})\b[^.\n@]{{0,40}}?\bis[ \t]+@?([A-Za-z0-9_][A-Za-z0-9_.]{{2,29}})\b"  # strong cue ... is value
    rf"|\b(?:{USERNAME_PLATFORM_CUES})(?![./\w])(?:[ \t]+(?:is|-))?[ \t]+@?([A-Za-z]*[0-9_.][A-Za-z0-9_.]*)\b"  # platform cue + handle-like value
    rf"|\b(?:{USERNAME_PLATFORM_CUES})\b[^.\n@]{{0,40}}?\bis[ \t]+@?([A-Za-z]*[0-9_][A-Za-z0-9_.]*)\b",          # platform cue ... is handle-like value
    re.IGNORECASE,
)
USERNAME_STOPWORDS = {
    "not", "the", "a", "an", "my", "your", "our", "is", "was", "and", "or", "but", "already", "taken",
    "wrong", "required", "invalid", "incorrect", "missing", "what", "something", "still", "also", "now",
}

# Student/other ID numbers: label-cued or long alphanumeric tokens.
# The cue must end at a word boundary ("id" is not a cue inside "identify")
# and the value must contain a digit ("ID badges" is not an ID).
ID_CUE_RE = re.compile(
    r"\b(?:student\s*(?:id|number|no\.?|#)|id\s*(?:number|no\.?|#)?|osis|sid|lunch\s*(?:number|#)|badge|library\s*card|account\s*(?:number|no\.?|#)|roll\s*(?:number|no\.?))"
    r"(?![A-Za-z])\s*(?:is|:|=|-|#)?\s*(?=[A-Z0-9-]*\d)([A-Z0-9][A-Z0-9-]{4,24})\b",
    re.IGNORECASE,
)
ID_BARE_RE = re.compile(r"(?<![\w-])(?=[A-Z0-9-]{7,24}(?![\w-]))(?=[A-Z-]*\d)[A-Z0-9-]{7,24}(?![\w-])")
ID_BARE_MIN_DIGITS = 5
# Digit groups that are dates, number ranges or round numbers, not IDs.
NOT_AN_ID_RE = re.compile(r"\d{1,4}-\d{1,4}|\d{1,2}-\d{1,2}-\d{2,4}|\d{4}-\d{1,2}-\d{1,2}|\d+0{3}")

STREET_SUFFIXES = (
    "expressway|extensions|throughway|trafficway|crossroad|extension|junctions|mountains|stravenue|underpass|causeway|crescent|crossing|junction|motorway|mountain|overpass|parkways|turnpike|villages|centers|circles|corners|estates|freeway|gardens|gateway|harbors|heights|highway|islands|landing|meadows|mission|orchard|parkway|passage|prairie|springs|squares|station|streets|terrace|valleys|viaduct|village|avenue|branch|bridge|brooks|bypass|canyon|center|circle|cliffs|common|corner|course|courts|divide|drives|estate|fields|forest|forges|garden|greens|groves|harbor|hollow|island|knolls|lights|manors|meadow|plains|points|radial|rapids|ridges|shoals|shores|skyway|spring|square|stream|street|summit|tunnel|unions|valley|alley|brook|burgs|cliff|court|coves|creek|crest|curve|drive|falls|ferry|field|flats|fords|forge|forks|glens|green|grove|haven|hills|inlet|knoll|lakes|light|locks|lodge|manor|mills|mount|parks|pines|place|plain|plaza|point|ports|ranch|rapid|ridge|river|roads|route|shoal|shore|spurs|trace|track|trail|union|views|ville|vista|walks|wells|blvd|burg|camp|cape|club|cove|dale|fall|flat|ford|fork|fort|glen|hill|isle|keys|lake|land|lane|loaf|lock|loop|mall|mews|mill|neck|oval|park|pass|path|pike|pine|pkwy|port|ramp|rest|road|spur|view|walk|wall|ways|well|ave|cir|dam|hwy|key|row|rue|run|ter|trl|via|way|ct|dr|ln|pl|rd|sq|st"
)
# House number, one to four capitalised street-name words (or an ordinal),
# then a suffix that ends at a word boundary. Case matters for the name
# words: "5 people in the park" and "4 stakeholders" are not addresses.
STREET_ADDRESS_RE = re.compile(
    rf"\b\d{{1,6}}[ \t]+(?:(?:[A-Z][a-zA-Z'’]*\.?|\d{{1,3}}(?:st|nd|rd|th))[ \t]+){{1,4}}(?i:{STREET_SUFFIXES})\b\.?"
    r"(?:[ \t]*,?[ \t]*(?i:apt|apartment|unit|suite|ste|#)\.?[ \t]*\w+)?"
    r"(?:\s*,?\s*[A-Z][a-zA-Z]+(?:[ \t][A-Z][a-zA-Z]+)?[ \t]*,?[ \t]*[A-Z]{2}[ \t]*\d{5}(?:-\d{4})?)?"
)

DATE_RE = re.compile(
    r"\b(?:0?[1-9]|1[0-2])[/.-](?:0?[1-9]|[12]\d|3[01])[/.-](?:19|20)\d{2}\b"
    r"|\b(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])\b"
    r"|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+(?:19|20)\d{2}\b",
    re.IGNORECASE,
)

# Names introduced by an explicit cue. Strong cues accept a single first
# name ("this is Marcus"); weak cues need two or three capitalised tokens.
# A name never runs across a line break, so a heading followed by a sentence
# ("Challenge and Selection", blank line, "I ...") is not read as one.
_NAME_TOKENS = r"[A-Z][a-z'’-]+(?:[ \t]+(?:[A-Z][a-z'’-]+|[A-Z]\.?)){0,2}"
NAME_CUE_RE = re.compile(
    rf"\b(?i:my name is|my name's|i am|i'm|im|this is|name:)[ \t]+({_NAME_TOKENS})"
    rf"|\b(?i:sincerely,?|thanks,?|thank you,?|regards,?|love,?|yours,?)\s+({_NAME_TOKENS})"
    rf"|\b(?i:written by|interviewed by|by)[ \t]+([A-Z][a-z'’-]+(?:[ \t]+(?:[A-Z][a-z'’-]+|[A-Z]\.?)){{1,2}})"
)
NAME_STOPWORDS = {
    "The", "This", "That", "What", "When", "Where", "Why", "How", "Which", "Who",
    "In", "On", "At", "For", "From", "With", "About", "Going", "Not", "Very",
    "Really", "Also", "Then", "Now", "Here", "There", "So", "But", "And", "Or",
    "A", "An", "It", "We", "They", "He", "She", "You", "I", "Sure", "Sorry",
    "Hi", "Hello", "Hey", "Oh", "Thanks", "Yeah", "Well", "Um", "Hmm", "Wait", "Ms", "Mr", "Mrs", "Dr", "Miss", "Mx", "Tutor", "Student", "Teacher", "Okay", "Ok",
    "Yes", "No", "Just", "Still", "Only", "All", "My", "Your", "Our", "Their", "Good",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December", "Spring", "Summer", "Fall", "Winter",
}

# --- Indirect and persistent identifiers ------------------------------------
# FERPA counts family members' names and indirect identifiers (date and place
# of birth, anything that identifies a student in combination); COPPA counts
# persistent identifiers and geolocation. None of these are PIILO labels.

_OCTET = r"(?:25[0-5]|2[0-4]\d|1?\d?\d)"
IPV4_RE = re.compile(rf"(?<![\w.])(?:{_OCTET}\.){{3}}{_OCTET}(?!\.?\d)(?!\w)")
IPV6_RE = re.compile(r"(?<![\w:])(?:[0-9A-Fa-f]{1,4}:){3,7}(?::?[0-9A-Fa-f]{1,4}){1,4}(?![\w:])")
MAC_RE = re.compile(r"(?<![\w:-])(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}(?![\w:-])")
UUID_RE = re.compile(r"(?<![\w-])[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}(?![\w-])")
GEO_RE = re.compile(r"(?<![\w.])-?\d{1,2}\.\d{3,8},[ \t]*-?\d{1,3}\.\d{3,8}(?!\.?\d)(?!\w)")

AGE_RE = re.compile(
    r"\b(\d{1,2})[ -]?(?:years?|yrs?)[ -]old\b"
    r"|\b(\d{1,2})[ \t]?(?:yo|y/o)\b"
    r"|\b(?:age|aged|turning|turned|turn)[ \t]+(\d{1,2})\b"
    r"|\b(?:i am|i'm|i’m|im)[ \t]+(\d{1,2})(?=[ \t]*(?:[.,!?;]|$|and\b|but\b|so\b|btw\b))",
    re.IGNORECASE | re.MULTILINE,
)
_MONTHS = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*"
BIRTHDAY_RE = re.compile(
    rf"\b(?:birthday|bday|born)\b[^.\n]{{0,20}}?\b({_MONTHS}\.?[ \t]+\d{{1,2}}(?:st|nd|rd|th)?)\b(?!,?[ \t]+(?:19|20)\d{{2}})",
    re.IGNORECASE,
)

_PERSON = r"[A-Z][a-z'’-]+(?:[ \t]+[A-Z][a-z'’-]+)?"
RELATIONS = (
    "mom|mum|mommy|mother|dad|daddy|father|stepmom|stepdad|stepmother|stepfather|parent|guardian|brother|sister|"
    "sibling|grandma|grandpa|grandmother|grandfather|aunt|auntie|uncle|cousin|best friend|friend|bff|teacher|tutor|"
    "coach|principal|counselor|neighbor|neighbour|classmate|partner|boyfriend|girlfriend"
)
NAME_RELATED_RE = re.compile(
    rf"\b(?i:my|our|his|her|their)[ \t]+(?:[a-z]+[ \t]+)?(?i:{RELATIONS})s?,?[ \t]+(?:(?i:is|named|called|name is)[ \t]+)?({_PERSON})"
)
NAME_TITLE_RE = re.compile(rf"\b(?:Mr|Mrs|Ms|Miss|Mx|Dr|Coach|Principal|Professor|Prof)\.?[ \t]+({_PERSON})")

SCHOOL_TYPES = (
    "Elementary School|Middle School|High School|Junior High|Primary School|Secondary School|Charter School|"
    "Preparatory School|Elementary|Academy|School|College|University|Institute"
)
SCHOOL_KIND_WORDS = {
    "High", "Middle", "Elementary", "Junior", "Senior", "Primary", "Secondary", "Charter", "Preparatory",
    "Public", "Private", "Grade", "Summer", "Sunday", "Business", "Law", "Medical", "Graduate", "Community",
}
SCHOOL_RE = re.compile(
    rf"\b((?:[A-Z][A-Za-z'’.-]*[ \t]+){{1,4}})(?:{SCHOOL_TYPES})\b"
    r"|\b(?:University|College|School|Academy|Institute)[ \t]+of[ \t]+(?:the[ \t]+)?[A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+){0,2}"
)

US_STATES = (
    "AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|"
    "OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY|DC"
)
_PLACE = r"[A-Z][a-z]+(?:[ \t][A-Z][a-z]+){0,2}"
LOCATION_RE = re.compile(
    rf"\b({_PLACE},[ \t]*(?:{US_STATES}))\b"
    rf"|\b(?i:i live in|we live in|i'm from|i’m from|i am from|im from|we moved to|i moved to|i was born in|born in|my hometown is)[ \t]+({_PLACE})"
)

# URLs at reference domains are not personal.
URL_ALLOWLIST = (
    ".edu", ".ac.uk", "who.int", "un.org", "forbes.com", "hbr.org", "ted.com",
    "researchgate.net", "sciencedirect.com", "springer.com", "investopedia.com",
    "mckinsey.com", "ideo.com", "ideou.com", "interaction-design.org", "mindtools.com",
    "miro.com", "mural.co",
    "wikipedia.org", "khanacademy.org", "google.com",
    "britannica.com", "nasa.gov", ".gov", "nytimes.com", "bbc.com", "cnn.com",
    "coursera.org", "edx.org", "doi.org", "jstor.org", "arxiv.org", "scholar.google",
    "quizlet.com", "desmos.com", "wolframalpha.com", "stackoverflow.com",
)

PERSONAL_URL_HINTS = (
    "linkedin.com/in/", "facebook.com/", "instagram.com/", "twitter.com/", "x.com/",
    "tiktok.com/@", "github.com/", "youtube.com/", "youtu.be/", "medium.com/@",
    "wordpress.com", "blogspot.com", "wixsite.com", "carrd.co", "about.me", "linktr.ee",
)


# Either the writer claims it ("my store is ...") or it has just been called
# somebody's site ("the website of ...", then the link on its own line).
OWN_SITE_CUE_RE = re.compile(
    r"\b(?:my|our)\b[^.\n]{0,30}$"
    r"|\b(?:web-?site|web site|homepage|home page|blog|portfolio|profile|channel)\b[\s\S]{0,150}$",
    re.IGNORECASE,
)


def _url_is_personal(url: str, before: str = "") -> bool:
    """`before` is the text leading up to the URL."""
    u = url.lower()
    if any(h in u for h in PERSONAL_URL_HINTS):
        return True
    if any(a in u for a in URL_ALLOWLIST):
        return False
    # A bare site name ("draw.io", "www.klaxoon.com") is a product or a
    # publication unless the writer calls it theirs ("my store is ...").
    path = re.sub(r"^(?:https?://)?[^/]+", "", u).strip("/.,;:)]}'\"")
    if not path:
        return bool(OWN_SITE_CUE_RE.search(before))
    # Default: a page on an unknown domain in student writing is treated as personal.
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
        for m in list(URL_RE.finditer(text)) + list(BARE_DOMAIN_RE.finditer(text)):
            # Domains inside an email address are handled by the EMAIL rule.
            if "@" in text[max(0, m.start() - 1):m.start()]:
                continue
            if _url_is_personal(m.group(), text[max(0, m.start() - 170):m.start()]):
                _add(ents, "URL_PERSONAL", m.start(), m.end(), text, 0.9)

    if want("USERNAME"):
        for m in USERNAME_RE.finditer(text):
            g = next((i for i in (1, 2, 3, 4, 5) if m.group(i)), None)
            if g is None:
                continue
            # Skip handles that are actually part of an email.
            if EMAIL_RE.search(text[max(0, m.start() - 1):m.end() + 1]):
                continue
            value = m.group(g).rstrip(".")
            if len(value) < 3 or not any(ch.isalpha() for ch in value):
                continue
            if g == 2 and value.isalpha() and value[0].isupper():
                continue  # "Instagram: Familiarity and ..." is a heading, not a handle
            if g == 3 and value.lower() in USERNAME_STOPWORDS:
                continue
            _add(ents, "USERNAME", m.start(g), m.end(g), text, 0.9)

    if want("ID_NUM"):
        for m in ID_CUE_RE.finditer(text):
            _add(ents, "ID_NUM", m.start(1), m.end(1), text, 0.92)
        for m in ID_BARE_RE.finditer(text):
            tok = m.group()
            digits = re.sub(r"\D", "", tok)
            if len(digits) < ID_BARE_MIN_DIGITS or NOT_AN_ID_RE.fullmatch(tok):
                continue
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
            g = next(i for i in (1, 2, 3) if m.group(i))
            name = m.group(g)
            if name.split()[0] in NAME_STOPWORDS:
                continue
            _add(ents, "NAME_STUDENT", m.start(g), m.end(g), text, 0.7)

    if want("NAME_RELATED"):
        for m in list(NAME_RELATED_RE.finditer(text)) + list(NAME_TITLE_RE.finditer(text)):
            if m.group(1).split()[0] in NAME_STOPWORDS:
                continue
            _add(ents, "NAME_RELATED", m.start(1), m.end(1), text, 0.7)

    if want("SCHOOL"):
        for m in SCHOOL_RE.finditer(text):
            start = m.start()
            if m.group(1):
                # "At Lincoln Middle School": the sentence opener is not part of the name.
                words = list(re.finditer(r"\S+", m.group(1)))
                while words and words[0].group() in NAME_STOPWORDS:
                    words.pop(0)
                # "In High School you get more homework" names no school.
                if all(w.group() in SCHOOL_KIND_WORDS for w in words):
                    continue
                start = m.start(1) + words[0].start()
            _add(ents, "SCHOOL", start, m.end(), text, 0.75)

    if want("LOCATION"):
        for m in LOCATION_RE.finditer(text):
            g = 1 if m.group(1) else 2
            if m.group(g).split()[0] in NAME_STOPWORDS:
                continue
            _add(ents, "LOCATION", m.start(g), m.end(g), text, 0.75)

    if want("AGE"):
        for m in AGE_RE.finditer(text):
            g = next(i for i in (1, 2, 3, 4) if m.group(i))
            if int(m.group(g)) > 0:
                _add(ents, "AGE", m.start(g), m.end(g), text, 0.8)

    if want("DATE"):
        for m in BIRTHDAY_RE.finditer(text):
            _add(ents, "DATE", m.start(1), m.end(1), text, 0.8)

    if want("IP_ADDRESS"):
        for m in IPV4_RE.finditer(text):
            _add(ents, "IP_ADDRESS", m.start(), m.end(), text, 0.95)
        for m in IPV6_RE.finditer(text):
            _add(ents, "IP_ADDRESS", m.start(), m.end(), text, 0.9)

    if want("DEVICE_ID"):
        for m in MAC_RE.finditer(text):
            _add(ents, "DEVICE_ID", m.start(), m.end(), text, 0.95)
        for m in UUID_RE.finditer(text):
            _add(ents, "DEVICE_ID", m.start(), m.end(), text, 0.95)

    if want("GEO"):
        for m in GEO_RE.finditer(text):
            _add(ents, "GEO", m.start(), m.end(), text, 0.9)

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


def propagate_names(
    text: str, ents: List[Entity], min_token_len: int = 3, min_model_confidence: float = 0.75
) -> List[Entity]:
    """Tag every other occurrence of a detected name (full name and each
    capitalised name token) so a name caught once is caught everywhere.
    Student names are handled first, so a name that is both keeps that label.

    A name the model was unsure of is still flagged where the model found it,
    but is not spread through the document: one weak guess ("Hood", in
    "Little Red Riding Hood") should not become six."""
    names = [
        e for e in ents
        if e.label in NAME_LABELS and (e.source != "model" or e.confidence >= min_model_confidence)
    ]
    if not names:
        return ents
    out = list(ents)
    seen = {(e.start, e.end) for e in ents}
    needles = {}  # text -> (label, confidence); the first label wins
    for e in sorted(names, key=lambda e: NAME_LABELS.index(e.label)):
        full = e.text.strip()
        needles.setdefault(full, (e.label, e.confidence))
        for tok in full.split():
            tok = tok.strip(".")
            if len(tok) >= min_token_len and tok[0].isupper() and tok not in NAME_STOPWORDS:
                needles.setdefault(tok, (e.label, e.confidence * 0.9))
    for needle, (label, conf) in sorted(needles.items(), key=lambda n: -len(n[0])):
        # "Priya's" is a mention of Priya; "O'Brien" is not a mention of Brien.
        for m in re.finditer(r"(?<![\w'’])" + re.escape(needle) + r"(?!\w|['’](?!s\b)\w)", text):
            if (m.start(), m.end()) in seen:
                continue
            # skip if inside an email/url/username span already found
            if any(k.start <= m.start() and m.end() <= k.end for k in ents if k.label not in NAME_LABELS):
                continue
            out.append(Entity(label, needle, m.start(), m.end(), conf, source="propagated"))
            seen.add((m.start(), m.end()))
    return out
