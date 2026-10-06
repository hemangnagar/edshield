"""demo/index.html carries a JavaScript port of edshield/rules.py. This runs
that JavaScript under Node and checks it agrees with the Python rules, so the
two cannot drift apart unnoticed. Skipped when Node is not installed."""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from edshield.rules import detect_rules, propagate_names, resolve_overlaps

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")

DEMO = Path(__file__).resolve().parent.parent / "demo" / "index.html"

SENTENCES = [
    # ordinary prose
    "We were able to identify the root cause during the ideation phase.",
    "Everyone has to wear ID badges, and the id cards were checked.",
    "COVID-19 changed how school worked in 2019-2020.",
    "The restaurant seats 100-200 people and we saved 250-300 hours.",
    "The city has 12000000 residents.",
    "There were 5 people in the park and 3 stars on the flag.",
    "We interviewed 4 stakeholders and spent 10 million on the first run.",
    "Challenge and Selection\n\nI work in a small team.",
    "I went with New York friends and United States history.",
    "It's Monday and this is Spring Break.",
    "My login is broken and nobody could handle the management issues.",
    "I think twitter is great for news and it is hard to handle 3 sensors.",
    "Instagram: Familiarity and narrow reach",
    "It has truly succeeded.  5.Approach: we built a prototype.",
    "See https://www.forbes.com/sites/story and http://tutorials.istudy.psu.edu/conceptmaps/ for more.",
    # real identifiers
    "My ID: 4471882.",
    "Student no. S12345678 was late.",
    "The form says 860632713425 at the top.",
    "Mobile: (820)913-3241x894",
    "Call 703-555-0142 ext 12 after school.",
    "We live at 591 Smith Centers Apt. 656 now.",
    "SSN 123-45-6789, lives at 1420 Maple Ridge Ct, Vienna, VA 22182",
    "Send it to 12 W 5th Ave please.",
    "Hi, this is Marcus.",
    "Written by Dennis Boone for class.",
    "Sincerely,\nPriya Raman",
    "My username on the class site is zjones.",
    "username: marcus_hoops",
    "my discord is marcus_77",
    "my login for the school portal is aruiz2013 if u need it, id number S20130447.",
    "Reach me at sam.lee@school.org or @samlee_art on insta",
    "Watch it at https://www.youtube.com/watch?v=2sOzgGAeiQV today.",
    "My store is holalili.com and it is new.",
    "I'm including the website of a colleague for anyone interested.\n\nhttp://www.moore.com/\n\nShe works shifts.",
    "The tool I used was draw.io and we met on www.klaxoon.com every week. Visit our site at www.theatfc.org.",
    "Tools such as www.mindmup.com, https://coggle.it/, http://wisemapping.com/ and https://hodge-ramsey.com/tagmain.html.",
    "See https://en.wikipedia.org/wiki/Cell and my page priyawrites.wordpress.com",
    "Due 03/14/2012 and again on March 3, 2013.",
    "My name is Priya Raman. Priya's poster won. Mr. O'Raman and Ramanathan were there.",
    "My   name is Ann Lee. Ann went to the Annual fair with Ann's brother.",
    # indirect and persistent identifiers
    "I got 42 but my brother Jordan said that's wrong. Later Jordan's friend left.",
    "My friend Daniel Okafor helped me count trays.",
    "my mom is Sarah and my teacher Ms. Patel said to use the rubric",
    "Ask Coach Ramirez or Dr. Nguyen about it.",
    "I'm in 8th grade at Rachel Carson Middle School.",
    "At Lincoln High School we have a big gym. In High School you get more homework.",
    "She studies at the University of Toledo now.",
    "We moved to Cedar Falls last summer. The game was in Round Rock, TX this year.",
    "We live at 1420 Maple Ridge Ct, Vienna, VA 22182 now.",
    "I am 11 years old and I like soccer.\nim 12 btw\nI'm turning 13 in May. my 9-year-old brother",
    "I am 3 problems behind on the worksheet. My friend and I went to the park.",
    "My birthday is March 3 and I want a bike. I was born on March 3, 2012.",
    "The log shows 192.168.1.44 at login. We read chapters 3.1.2 and 4.10.",
    "Connected from 2001:0db8:85a3:0000:0000:8a2e:0370:7334 at 10:30:15.",
    "The tablet's MAC is 3C:22:FB:9A:10:5E and id 38400000-8cf0-11bd-b23e-10b96e40000d.",
    "Pinned at 38.9012, -77.2653.",
    # dates, ages and school names the way children type them
    "my birthday party is the 8th of August, can u come. recital is August 8th. tryouts got moved to March 3 and jul 27.",
    "if you are 12 like me nobody listens. 15m and i need help with algebra. brb 5m. the pool is 25m long and we are 3 problems behind.",
    "we moved so now its parker middle school for me. my old high school had a pool. in middle school you switch classes.",
    "May 5 kids came to the party. I read 20 pages on March 3 nights in a row. you may 5 us later.",
]

HARNESS = """
const sentences = JSON.parse(require("fs").readFileSync(0, "utf8"));
const out = sentences.map(t => resolve(propagateNames(t, detectRules(t)))
  .map(e => [e.label, e.text, e.start, e.end]));
console.log(JSON.stringify(out));
"""


def demo_rules_js() -> str:
    html = DEMO.read_text(encoding="utf-8")
    m = re.search(r"// -+ Rule detectors.*?(?=// -+ Policies)", html, re.S)
    assert m, "rule section not found in demo/index.html"
    return m.group()


def demo_section(title: str) -> str:
    html = DEMO.read_text(encoding="utf-8")
    m = re.search(r"// -+ " + re.escape(title) + r".*?(?=// -+ )", html, re.S)
    assert m, f"section {title!r} not found in demo/index.html"
    return m.group()


# (text, [(token piece, label, score), ...]) as the browser model reports them: one entry
# per token, in order, with no positions. "▁" marks a token that starts a word.
MODEL_CASES = [
    # a short piece that also occurs earlier in another word ("p" in "project")
    ("For this project email me at priya.raman08@gmail.com today",
     [("▁For", "O", .99), ("▁this", "O", .99), ("▁project", "O", .99), ("▁email", "O", .99),
      ("▁me", "O", .99), ("▁at", "O", .99), ("▁p", "B-EMAIL", .98), ("riya", "I-EMAIL", .99),
      (".", "I-EMAIL", .99), ("raman", "I-EMAIL", .99), ("08", "I-EMAIL", .99), ("@", "I-EMAIL", .99),
      ("gmail", "I-EMAIL", .99), (".", "I-EMAIL", .99), ("com", "I-EMAIL", .99), ("▁today", "O", .99)]),
    # a username in pieces, each of which also occurs inside an earlier word
    ("my teacher said use the rubric. my login is aruiz2013 ok",
     [("▁my", "O", .99), ("▁teacher", "O", .99), ("▁said", "O", .99), ("▁use", "O", .99),
      ("▁the", "O", .99), ("▁rubric", "O", .99), (".", "O", .99), ("▁my", "O", .99),
      ("▁login", "O", .99), ("▁is", "O", .99), ("▁a", "B-USERNAME", .99), ("ru", "B-USERNAME", .99),
      ("iz2013", "B-USERNAME", .99), ("▁ok", "O", .99)]),
    # word pieces, two adjacent words, and a name only partly tagged
    ("I met Priyanka Ramanathan and Daniel today",
     [("▁I", "O", .99), ("▁met", "O", .99), ("▁Pri", "B-NAME_STUDENT", .97), ("yanka", "B-NAME_STUDENT", .95),
      ("▁Raman", "I-NAME_STUDENT", .96), ("athan", "O", .60), ("▁and", "O", .99),
      ("▁Daniel", "B-NAME_STUDENT", .99), ("▁today", "O", .99)]),
    # below the threshold, a one-letter name, and a piece of a URL inside brackets
    ("By C. see (https://coursera.org/share/b24116a7) and Hood",
     [("▁By", "O", .99), ("▁C", "B-NAME_STUDENT", .98), (".", "O", .99), ("▁see", "O", .99),
      ("▁(", "O", .99), ("https", "O", .9), ("://", "O", .9), ("coursera", "B-URL_PERSONAL", .66),
      (".", "O", .9), ("org", "O", .9), ("/", "O", .9), ("share", "O", .9), ("/", "O", .9), ("b24116a7", "B-ID_NUM", .56),
      (")", "O", .99), ("▁and", "O", .99), ("▁Hood", "B-NAME_STUDENT", .45)]),
]


def python_decode(text, pieces):
    from edshield import ner
    labels = ["O"] + sorted({l for _, l, _ in pieces if l != "O"})
    id2label = dict(enumerate(labels))
    offsets, probs, cursor = [], [], 0
    for piece, label, score in pieces:
        piece = piece.lstrip("▁")
        start = text.index(piece, cursor)
        assert text[cursor:start].strip() == "", "test pieces must follow each other in the text"
        cursor = start + len(piece)
        p = [0.0] * len(labels)
        p[labels.index(label)] = score
        p[1 if label == "O" else 0] += 1 - score
        offsets.append((start, cursor))
        probs.append(p)
    return [[e.label, e.text, e.start, e.end] for e in ner.decode(text, offsets, probs, id2label)]


def test_demo_model_decoding_agrees_with_python():
    harness = """
const cases = JSON.parse(require("fs").readFileSync(0, "utf8"));
console.log(JSON.stringify(cases.map(([text, pieces]) =>
  decodeTokens(text, alignTokens(text, pieces.map(([word, entity, score]) => ({word, entity, score}))))
    .map(e => [e.label, e.text, e.start, e.end]))));
"""
    code = 'const NAME_LABELS = ["NAME_STUDENT","NAME_RELATED"];\n' + demo_section("Model decoding") + harness
    proc = subprocess.run([NODE, "-e", code], input=json.dumps(MODEL_CASES), capture_output=True,
                          text=True, encoding="utf-8", timeout=60)
    assert proc.returncode == 0, proc.stderr
    got = json.loads(proc.stdout)
    assert got[0] == [["EMAIL", "priya.raman08@gmail.com", 29, 52]]
    assert got[1] == [["USERNAME", "aruiz2013", 44, 53]]
    for (text, pieces), js in zip(MODEL_CASES, got):
        assert js == python_decode(text, pieces), f"demo and Python disagree on: {text!r}"


def test_demo_rules_agree_with_python():
    proc = subprocess.run(
        [NODE, "-e", demo_rules_js() + HARNESS],
        input=json.dumps(SENTENCES), capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    js = json.loads(proc.stdout)
    for text, got in zip(SENTENCES, js):
        ents = resolve_overlaps(propagate_names(text, detect_rules(text)))
        want = [[e.label, e.text, e.start, e.end] for e in ents]
        assert got == want, f"demo and Python disagree on: {text!r}"


def test_demo_overlap_resolution_agrees_with_python():
    from edshield import Entity
    text = "we moved to Brookfield tbh and Ada Okoye-Lund came too"
    ents = [
        ("LOCATION", 12, 26, 0.95, "model"), ("STREET_ADDRESS", 23, 26, 0.98, "model"),
        ("NAME_RELATED", 31, 45, 0.6, "model"), ("NAME_STUDENT", 35, 40, 0.9, "model"),
        ("USERNAME", 12, 17, 0.5, "rules"), ("NAME_STUDENT", 41, 45, 0.55, "propagated"),
        ("NAME_RELATED", 41, 50, 0.5, "model"), ("LOCATION", 20, 30, 0.4, "rules"),
        ("NAME_RELATED", 33, 45, 0.3, "rules"),
    ]
    harness = """
const [text, ents] = JSON.parse(require("fs").readFileSync(0, "utf8"));
console.log(JSON.stringify(resolve(ents.map(([label, start, end, confidence, source]) =>
  ({label, text: text.slice(start, end), start, end, confidence, source}))).map(e => [e.label, e.text, e.start, e.end])));
"""
    proc = subprocess.run([NODE, "-e", demo_rules_js() + harness], input=json.dumps([text, ents]),
                          capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert proc.returncode == 0, proc.stderr
    want = resolve_overlaps([Entity(l, text[s:e], s, e, c, src) for l, s, e, c, src in ents])
    assert json.loads(proc.stdout) == [[e.label, e.text, e.start, e.end] for e in want]
    got = json.loads(proc.stdout)
    assert ["LOCATION", "Brookfield", 12, 22] in got
    assert all(s < e and text[s:e] == t for _, t, s, e in got)


# (text, [(token piece, {class: probability}), ...]): full distributions, for recall-first decoding
RECALL_FIRST_CASES = [
    # a name the model leans against, next to one it is sure of
    ("thanks from Noor Haddad and Ines today",
     [("▁thanks", {"O": .999}), ("▁from", {"O": .995}), ("▁Noor", {"O": .97, "B-NAME_STUDENT": .03}),
      ("▁Had", {"O": .87, "I-NAME_STUDENT": .12}), ("dad", {"O": .40, "I-NAME_STUDENT": .55}), ("▁and", {"O": .999}),
      ("▁Ines", {"O": .01, "B-NAME_STUDENT": .98}), ("▁today", {"O": .992})]),
    # the best class other than O changes inside a word, and a weak tail joins a sure town
    ("im in Brookfield tbh my user is kiwi_42",
     [("▁im", {"O": .999}), ("▁in", {"O": .999}), ("▁Brook", {"O": .0, "B-LOCATION": 1.0}),
      ("field", {"O": .0, "B-LOCATION": .99}), ("▁tb", {"O": .10, "I-LOCATION": .70, "I-STREET_ADDRESS": .16}),
      ("h", {"O": .02, "I-STREET_ADDRESS": .44, "I-LOCATION": .34, "B-LOCATION": .18}), ("▁my", {"O": .999}),
      ("▁user", {"O": .995}), ("▁is", {"O": .999}), ("▁kiwi", {"O": .985, "B-USERNAME": .01}),
      ("_", {"O": .86, "B-USERNAME": .10}), ("42", {"O": .91, "B-USERNAME": .07, "B-ID_NUM": .02})]),
    # nothing under the threshold; an initial alone is still not a name
    ("By C. we read the rubric", [("▁By", {"O": .999}), ("▁C", {"O": .5, "B-NAME_STUDENT": .5}), (".", {"O": .999}),
                                  ("▁we", {"O": .999}), ("▁read", {"O": .999}), ("▁the", {"O": .999}), ("▁rubric", {"O": .995})]),
]


def test_demo_recall_first_decoding_agrees_with_python():
    from edshield import ner
    harness = """
const cases = JSON.parse(require("fs").readFileSync(0, "utf8"));
console.log(JSON.stringify(cases.map(([text, pieces]) => {
  const rows = pieces.map(([word, dist]) => {
    const all = Object.entries(dist).sort((a, b) => b[1] - a[1]), nonO = all.filter(([l]) => l !== "O");
    return {word, entity: all[0][0], score: all[0][1], po: dist.O, alt: nonO[0] ? nonO[0][0] : "B-NAME_STUDENT"};
  });
  return decodeTokens(text, alignTokens(text, rows), 0.5, 0.99).map(e => [e.label, e.text, e.start, e.end, +e.confidence.toFixed(6)]);
})));
"""
    code = 'const NAME_LABELS = ["NAME_STUDENT","NAME_RELATED"];\n' + demo_section("Model decoding") + harness
    proc = subprocess.run([NODE, "-e", code], input=json.dumps(RECALL_FIRST_CASES), capture_output=True,
                          text=True, encoding="utf-8", timeout=60)
    assert proc.returncode == 0, proc.stderr
    got = json.loads(proc.stdout)
    labels = ["O"] + sorted({l for _, pieces in RECALL_FIRST_CASES for _, dist in pieces for l in dist if l != "O"})
    for (text, pieces), js in zip(RECALL_FIRST_CASES, got):
        offsets, probs, cursor = [], [], 0
        for piece, dist in pieces:
            piece = piece.lstrip("▁")
            start = text.index(piece, cursor)
            cursor = start + len(piece)
            offsets.append((start, cursor))
            probs.append([dist.get(l, 0.0) for l in labels])
        want = ner.decode(text, offsets, probs, dict(enumerate(labels)), o_threshold=0.99)
        assert js == [[e.label, e.text, e.start, e.end, round(e.confidence, 6)] for e in want], f"disagree on: {text!r}"
    assert [e[:2] for e in got[0]] == [["NAME_STUDENT", "Noor Haddad"], ["NAME_STUDENT", "Ines"]]
    assert got[2] == []
