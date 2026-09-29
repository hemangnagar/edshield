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
    "See https://en.wikipedia.org/wiki/Cell and my page priyawrites.wordpress.com",
    "Due 03/14/2012 and again on March 3, 2013.",
    "My name is Priya Raman. Priya's poster won. Mr. O'Raman and Ramanathan were there.",
    "My   name is Ann Lee. Ann went to the Annual fair with Ann's brother.",
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
