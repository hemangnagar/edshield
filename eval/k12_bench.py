"""Synthetic K-12 writing with every identifier type edshield covers, in PIILO
JSON format. All values come from Faker or fixed lists; no real people.

PIILO is essays by adult online learners. This set is there to measure the
kinds of text a school product actually sees: short essays by children,
tutoring transcripts and chatbot messages.

Two styles, reported separately because they answer different questions:

  cued  identifiers appear with the wording the rules look for ("My name is
        ...", "I live in ..."). A regression check: the rules should be near
        perfect here, and that says little about real text.
  hard  the same identifiers the way children type them: no cue words,
        lowercase names, run-together phone numbers, "name at gmail dot com".
        These were written to be realistic, not to match the rules.

    python eval/k12_bench.py --n 400 --style hard --out data/k12_hard.json --seed 1
    python eval/evaluate.py --input data/k12_hard.json --model rules --labels all
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from faker import Faker

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
MARK_RE = re.compile(r"⟦([A-Z_]+)\|(.*?)⟧", re.S)


def m(label: str, value) -> str:
    """Mark `value` as an identifier of type `label` inside a template."""
    return f"⟦{label}|{value}⟧"


FILLER = {
    "essay": [
        "My favorite part of the book was when the dog found his way home.",
        "I think recycling is important because it keeps the ocean clean.",
        "First we planted the seeds and then we watered them every day.",
        "The experiment did not work the first time so we tried again.",
        "In conclusion, I learned that teamwork makes hard things easier.",
        "The main character was brave even when she was scared.",
        "Our class voted and most people wanted a longer recess.",
        "I used three sources for this report and wrote notes on cards.",
    ],
    "transcript": [
        "Tutor: What do you think the question is asking?",
        "Student: I think we multiply first and then add.",
        "Tutor: Nice work. Can you explain how you got that?",
        "Student: I got 42 but I'm not sure that's right.",
        "Tutor: Let's slow down and look at the units.",
        "Student: Oh, because the two triangles are similar?",
        "Tutor: Try problems 4 through 8 before next time.",
        "Student: Okay that makes sense now.",
    ],
    "chat": [
        "can u help me with my essay its due tmrw",
        "i dont get fractions at all lol",
        "whats a good hook for a story about space",
        "ok thx that helps",
        "wait how do i cite a website",
        "is it ok if my paragraph is only 4 sentences",
        "my teacher said to use the rubric but idk where it is",
        "can you make it sound less boring",
    ],
}

SCHOOL_KINDS = ["Elementary School", "Middle School", "High School", "Academy", "Elementary"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]
STATES = ["VA", "TX", "OH", "CA", "NY", "FL", "WA", "IL", "GA", "NC", "MI", "CO"]


def person(fake: Faker, rng: random.Random) -> dict:
    """Everything that stays the same for one student across a document."""
    first, last = fake.first_name(), fake.last_name()
    city = fake.city()
    while not re.fullmatch(r"[A-Z][a-z]+(?: [A-Z][a-z]+)?", city):
        city = fake.city()
    user = f"{first.lower()}{rng.choice(['_', '.', ''])}{rng.choice([last.lower()[:4], 'plays', 'draws', 'xd'])}{rng.randint(7, 99)}"
    month, day, year = rng.choice(MONTHS), rng.randint(1, 28), rng.randint(2008, 2018)
    return {
        "dob": f"{month} {day}, {year}", "bday": f"{month} {day}",
        "bday_spoken": f"{day}th of {month}", "bday_chat": f"{month[:3].lower()} {day}",
        "first": first, "last": last, "full": f"{first} {last}",
        "friend": fake.first_name(), "friend_full": f"{fake.first_name()} {fake.last_name()}",
        "sibling": fake.first_name(), "parent": fake.first_name(),
        "teacher": fake.last_name(), "title": rng.choice(["Ms.", "Mr.", "Mrs.", "Dr.", "Coach"]),
        "school": f"{fake.last_name()} {rng.choice(SCHOOL_KINDS)}",
        "city": city, "state": rng.choice(STATES),
        "age": rng.randint(7, 17),
        "email": f"{first.lower()}.{last.lower()}{rng.randint(1, 99)}@{rng.choice(['gmail.com', 'yahoo.com', 'outlook.com'])}",
        "user": user,
        "phone": fake.numerify(rng.choice(["###-###-####", "(###) ###-####", "###.###.####"])),
        "phone_digits": fake.numerify("571#######"),
        "id": fake.numerify(rng.choice(["#######", "S########", "##-######"])),
        "street": fake.street_address(),
        "street_name": f"{fake.last_name()} {rng.choice(['Street', 'Court', 'Avenue', 'Lane', 'Drive'])}",
        "url": f"https://{rng.choice(['youtube.com/@', 'www.tiktok.com/@', 'github.com/'])}{user}",
        "site": f"{first.lower()}{rng.choice(['writes', 'draws', 'codes'])}.{rng.choice(['wordpress.com', 'blogspot.com'])}",
        "ip": fake.ipv4(), "mac": fake.mac_address().upper(), "uuid": fake.uuid4(),
        "geo": f"{rng.uniform(25, 48):.4f}, {rng.uniform(-122, -71):.4f}",
        "ssn": fake.numerify("5##-##-####").replace("5", rng.choice("1234"), 1),
    }


def cued(p: dict, rng: random.Random):
    """Identifiers introduced the way the rules expect."""
    return {
        "essay": [
            f"My name is {m('NAME_STUDENT', p['full'])} and I am in {rng.choice(['3rd', '5th', '7th', '9th'])} grade.",
            f"I am {m('AGE', p['age'])} years old and I like soccer.",
            f"I go to {m('SCHOOL', p['school'])} and my teacher is {p['title']} {m('NAME_RELATED', p['teacher'])}.",
            f"My brother {m('NAME_RELATED', p['sibling'])} helped me with the poster.",
            f"My best friend {m('NAME_RELATED', p['friend_full'])} was in my group.",
            f"We live at {m('STREET_ADDRESS', p['street'])} near the park.",
            f"I live in {m('LOCATION', p['city'])} with my family.",
            f"Our team played in {m('LOCATION', p['city'] + ', ' + p['state'])} this year.",
            f"I was born on {m('DATE', p['dob'])}.",
            f"My birthday is {m('DATE', p['bday'])} and I want a bike.",
            f"You can email me at {m('EMAIL', p['email'])} for the slides.",
            f"My student ID is {m('ID_NUM', p['id'])} if the office needs it.",
            f"I posted my project at {m('URL_PERSONAL', p['url'])} last night.",
            f"My blog is {m('URL_PERSONAL', p['site'])} and I write there every week.",
        ],
        "transcript": [
            f"Student: Hi, this is {m('NAME_STUDENT', p['first'])}.",
            f"Student: My name is {m('NAME_STUDENT', p['full'])}.",
            f"Student: I'm {m('AGE', p['age'])} years old.",
            f"Student: My mom {m('NAME_RELATED', p['parent'])} said you can call her at {m('PHONE_NUM', p['phone'])}.",
            f"Student: My email is {m('EMAIL', p['email'])}.",
            f"Student: My username is {m('USERNAME', p['user'])}.",
            f"Student: My teacher {p['title']} {m('NAME_RELATED', p['teacher'])} gave us this worksheet.",
            f"Student: I go to {m('SCHOOL', p['school'])}.",
            f"Student: We moved to {m('LOCATION', p['city'])} last summer.",
            f"Student: My student number is {m('ID_NUM', p['id'])}.",
        ],
        "chat": [
            f"my name is {m('NAME_STUDENT', p['full'])}",
            f"I am {m('AGE', p['age'])} years old",
            f"my discord is @{m('USERNAME', p['user'])}",
            f"my insta is @{m('USERNAME', p['user'])}",
            f"you can text me at {m('PHONE_NUM', p['phone'])}",
            f"my email is {m('EMAIL', p['email'])}",
            f"my sister {m('NAME_RELATED', p['sibling'])} is so annoying",
            f"device log: ip {m('IP_ADDRESS', p['ip'])} mac {m('DEVICE_ID', p['mac'])}",
            f"ad id {m('DEVICE_ID', p['uuid'])} location {m('GEO', p['geo'])}",
            f"my ssn is {m('SSN', p['ssn'])} is that bad to share",
        ],
    }


def hard(p: dict, rng: random.Random):
    """The same identifiers the way children actually type them."""
    lf, lfr = p["first"].lower(), p["friend"].lower()
    email_spoken = f"{p['first'].lower()}{p['last'].lower()} at gmail dot com"
    return {
        "essay": [
            f"{m('NAME_STUDENT', p['full'])}\n{p['title']} {m('NAME_RELATED', p['teacher'])}\nPeriod {rng.randint(1, 7)}",
            f"By {m('NAME_STUDENT', p['first'])}, age {m('AGE', p['age'])}",
            f"{m('NAME_RELATED', p['friend'])} and I built the volcano together.",
            f"Then {m('NAME_RELATED', p['friend'])} said we should add more baking soda.",
            f"At {m('SCHOOL', p['school'].split()[0])} we have a garden behind the gym.",
            f"It gets really cold here in {m('LOCATION', p['city'])} so the plants died.",
            f"Our house on {m('STREET_ADDRESS', p['street_name'])} has a big yard.",
            f"When you are {m('AGE', p['age'])} like me you can not drive yet.",
            f"I was born the {m('DATE', p['bday_spoken'])} and my party is next week.",
            f"{m('NAME_STUDENT', p['first'])}'s report on volcanoes",
        ],
        "transcript": [
            f"Tutor: Great job today, {m('NAME_STUDENT', p['first'])}!",
            f"Tutor: Say hi to {m('NAME_RELATED', p['parent'])} for me.",
            f"Student: {m('NAME_RELATED', p['sibling'])} already showed me that trick.",
            f"Student: you can call {m('PHONE_NUM', p['phone_digits'])} its my moms phone",
            f"Student: its {m('EMAIL', email_spoken)}",
            f"Student: find me as {m('USERNAME', p['user'])} on there",
            f"Student: {m('ID_NUM', p['id'])} is what it says on my card",
            f"Tutor: How is {m('SCHOOL', p['school'].split()[0])} going this year?",
            f"Student: we just moved from {m('LOCATION', p['city'])}",
            f"Student: I turn {m('AGE', p['age'] + 1)} next month",
        ],
        "chat": [
            f"im {m('NAME_STUDENT', lf)} btw",
            f"its {m('NAME_STUDENT', lf)} again lol",
            f"me and {m('NAME_RELATED', lfr)} r doing a project",
            f"{m('AGE', p['age'])}m here",
            f"{m('AGE', p['age'])} yr old and i still dont get it",
            f"add me {m('USERNAME', p['user'])}",
            f"text me {m('PHONE_NUM', p['phone_digits'])}",
            f"{m('EMAIL', email_spoken)}",
            f"i go to {m('SCHOOL', p['school'].lower())}",
            f"im in {m('LOCATION', p['city'].lower())} rn",
            f"my bday is {m('DATE', p['bday_chat'])}",
            f"{m('NAME_STUDENT', p['first'].upper())} {m('NAME_STUDENT', p['last'].upper())} period 3",
        ],
    }


def to_piilo(doc_id: int, text_marked: str, genre: str, style: str) -> dict:
    text, char_labels, pos = [], [], 0
    for mk in MARK_RE.finditer(text_marked):
        plain = text_marked[pos:mk.start()]
        text.append(plain)
        char_labels += ["O"] * len(plain)
        text.append(mk.group(2))
        char_labels += [mk.group(1)] * len(mk.group(2))
        pos = mk.end()
    text.append(text_marked[pos:])
    char_labels += ["O"] * len(text_marked[pos:])
    text = "".join(text)
    tokens, ws, labels = [], [], []
    prev = None  # (label, end) of the previous token
    for t in TOKEN_RE.finditer(text):
        lab = char_labels[t.start()]
        if lab == "O":
            labels.append("O")
            prev = None
        else:
            inside = prev and prev[0] == lab and all(c == lab for c in char_labels[prev[1]:t.start()])
            labels.append(("I-" if inside else "B-") + lab)
            prev = (lab, t.end())
        tokens.append(t.group())
        ws.append(t.end() < len(text) and text[t.end()].isspace())
    return {"document": doc_id, "genre": genre, "style": style, "full_text": text,
            "tokens": tokens, "trailing_whitespace": ws, "labels": labels}


def generate(n: int, style: str, seed: int, pii_rate: float = 0.45):
    fake = Faker("en_US")
    fake.seed_instance(seed)
    rng = random.Random(seed)
    docs = []
    for i in range(n):
        genre = rng.choice(["essay", "transcript", "chat"])
        p = person(fake, rng)
        pool = (cued if style == "cued" else hard)(p, rng)[genre]
        lines = []
        for _ in range(rng.randint(5, 9)):
            lines.append(rng.choice(pool) if rng.random() < pii_rate else rng.choice(FILLER[genre]))
        sep = " " if genre == "essay" else "\n"
        docs.append(to_piilo(i, sep.join(lines), genre, style))
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--style", choices=["cued", "hard"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    docs = generate(a.n, a.style, a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(docs), encoding="utf-8")
    n_ents = sum(l.startswith("B-") for d in docs for l in d["labels"])
    print(f"wrote {len(docs)} {a.style} docs with {n_ents} identifiers to {a.out}")


if __name__ == "__main__":
    main()
