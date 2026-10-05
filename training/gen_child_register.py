"""Generate child-register training documents in PIILO format.

PIILO is adult coursework. Children write differently: lowercase chat, short
transcript turns, journal entries, and they mention the town they live in, the
street, the school and the people around them in passing, without the cues
adults use. This script writes synthetic documents in that register so the
model can learn those mentions, with two labels PIILO does not have, LOCATION
and SCHOOL, next to the PIILO ones.

    python training/gen_child_register.py --n 2500 --seed 7 --out data/piilo/child_register.json

The output goes to training only (`prepare_piilo.py --extra`). The sentence
frames here are written for this file. They are not taken from
`eval/k12_bench.py` or from any evaluation set, and `tests/test_child_register.py`
fails if a frame shares a five-word run with the benchmark generator.

What is labelled:
  NAME_STUDENT    the writer and the children and family around them
  LOCATION        a town or city tied to the writer (lives, moved, visits family)
  SCHOOL          the writer's school, by full name or the bare name
  STREET_ADDRESS  a street or full address tied to the writer
  EMAIL, USERNAME, PHONE_NUM, ID_NUM
What is not: places, people and schools mentioned as general knowledge
("the capital of France is Paris", "we read about Rosa Parks"), ages, grades
and dates (the rule layer covers ages and dates).

All values come from Faker and the lists below; no real people.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import List, Tuple

from faker import Faker

TOKEN_RE = re.compile(r"\n+|\w+|[^\w\s]", re.UNICODE)
SLOT_RE = re.compile(r"<([A-Za-z_]+)>")

# Slot -> label. Lowercase slots are filled but not labelled.
LABELLED = {
    "ME": "NAME_STUDENT", "ME_FIRST": "NAME_STUDENT", "ME_LAST": "NAME_STUDENT", "KID": "NAME_STUDENT",
    "FAMILY": "NAME_STUDENT", "KID_FULL": "NAME_STUDENT", "KID_LAST": "NAME_STUDENT",
    "TOWN": "LOCATION", "TOWN2": "LOCATION",
    "SCHOOL": "SCHOOL", "SCHOOL_BARE": "SCHOOL",
    "STREET": "STREET_ADDRESS", "ADDRESS": "STREET_ADDRESS",
    "EMAIL": "EMAIL", "USER": "USERNAME", "PHONE": "PHONE_NUM", "ID": "ID_NUM",
}

TOWNS = [
    "Akron", "Albany", "Amarillo", "Ann Arbor", "Asheville", "Athens", "Aurora", "Bakersfield", "Bangor", "Baton Rouge",
    "Bend", "Billings", "Biloxi", "Boise", "Boulder", "Bozeman", "Brownsville", "Buffalo", "Burlington", "Cedar Rapids",
    "Chattanooga", "Cheyenne", "Chico", "Columbia", "Concord", "Corpus Christi", "Dayton", "Decatur", "Des Moines",
    "Dothan", "Dubuque", "Duluth", "Durham", "El Paso", "Elgin", "Erie", "Eugene", "Fargo", "Fayetteville", "Flagstaff",
    "Flint", "Fort Wayne", "Fresno", "Gainesville", "Galveston", "Gary", "Grand Rapids", "Green Bay", "Greenville",
    "Hartford", "Hattiesburg", "Helena", "Hilo", "Huntsville", "Iowa City", "Ithaca", "Jackson", "Joplin", "Juneau",
    "Kalamazoo", "Kenosha", "Knoxville", "Lafayette", "Lansing", "Laredo", "Las Cruces", "Lexington", "Lincoln",
    "Little Rock", "Lubbock", "Macon", "Madison", "Medford", "Mesa", "Midland", "Missoula", "Mobile", "Modesto",
    "Muncie", "Naperville", "Nashua", "Newark", "Norfolk", "Ocala", "Ogden", "Olympia", "Omaha", "Oshkosh", "Paducah",
    "Pasadena", "Peoria", "Plano", "Pocatello", "Provo", "Pueblo", "Racine", "Rapid City", "Reading", "Redding", "Reno",
    "Roanoke", "Rochester", "Rockford", "Roswell", "Saginaw", "Salem", "Salinas", "San Marcos", "Santa Fe", "Savannah",
    "Scranton", "Shreveport", "Sioux Falls", "South Bend", "Spokane", "Springfield", "Stockton", "Syracuse", "Tacoma",
    "Tallahassee", "Tempe", "Toledo", "Topeka", "Tucson", "Tulsa", "Tupelo", "Tyler", "Utica", "Visalia", "Waco",
    "Walla Walla", "Waterloo", "Wichita", "Wilmington", "Winona", "Yakima", "Yonkers", "York", "Youngstown", "Yuma",
]
# Places, people and schools as general knowledge: never labelled.
WORLD_PLACES = ["France", "Egypt", "Japan", "Brazil", "Mexico", "Kenya", "India", "Antarctica", "Australia", "Italy",
                "Paris", "London", "Tokyo", "Rome", "Cairo", "Texas", "Alaska", "Hawaii", "Canada", "China",
                "the Amazon", "Mount Everest", "the Pacific Ocean", "Mars", "the Sahara", "Greece", "Peru", "Florida"]
FAMOUS = ["Rosa Parks", "Abraham Lincoln", "Harriet Tubman", "Albert Einstein", "Marie Curie", "George Washington",
          "Martin Luther King", "Amelia Earhart", "Neil Armstrong", "Roald Dahl", "Dr. Seuss", "Jane Goodall",
          "Thomas Edison", "Helen Keller", "Sacagawea", "Benjamin Franklin", "Frida Kahlo", "Shakespeare"]
STREET_NAMES = ["Maple", "Oak", "Cedar", "Birch", "Elm", "Pine", "Walnut", "Willow", "Hickory", "Juniper", "Magnolia",
                "Sycamore", "Chestnut", "Spruce", "Poplar", "Dogwood", "Laurel", "Aspen", "Linden", "Hawthorne",
                "Washington", "Jefferson", "Lincoln", "Madison", "Franklin", "Jackson", "Grant", "Monroe", "Adams",
                "Lake", "River", "Hill", "Park", "Church", "Mill", "Spring", "Sunset", "Highland", "Prospect", "Orchard",
                "Meadow", "Ridge", "Valley", "Forest", "Garden", "Summit", "Harbor", "Canyon", "Prairie", "Quarry",
                "2nd", "3rd", "4th", "5th", "7th", "9th", "12th", "21st", "33rd", "Main", "Broad", "Market", "Division"]
STREET_KINDS = ["Street", "St", "St.", "Avenue", "Ave", "Ave.", "Road", "Rd", "Drive", "Dr", "Lane", "Ln", "Court", "Ct",
                "Place", "Way", "Boulevard", "Blvd", "Circle", "Terrace", "Trail"]
SCHOOL_KINDS = ["Elementary", "Elementary School", "Middle School", "High School", "Academy", "Junior High",
                "Primary School", "Intermediate School", "Charter School", "Prep", "Day School", "Magnet School",
                "School", "K-8", "Montessori"]
SUBJECTS = ["math", "science", "reading", "social studies", "spanish", "art", "music", "history", "writing", "pe"]
TEACHER_TITLES = ["Mr.", "Ms.", "Mrs.", "Miss", "Coach", "Dr."]

# --- frames. `<UPPER>` slots are labelled, `<lower>` slots are filled and left as O. -------------

CHAT_PII = [
    "this is <ME_FIRST>", "<ME_FIRST> here", "hey its <ME_FIRST> from <subject> class", "yo <KID> are u there",
    "my name's <ME> if u need it for the form", "sign me up as <ME>", "ask <KID> she has the answers",
    "<KID> said the quiz is friday", "tell <KID> i said hi", "can <KID> join the call too",
    "me <KID> and <KID> are a group", "ugh <KID> keeps spamming the group chat", "my cousin <FAMILY> can drive us",
    "my sister <FAMILY> took my charger", "<FAMILY> says i have to log off at 9", "grandma <FAMILY> is visiting",
    "i'm at my dads in <TOWN> this weekend", "we're driving to <TOWN> for thanksgiving",
    "do u live near <TOWN> too", "the <TOWN> library closes early", "its raining so hard in <TOWN>",
    "i used to live in <TOWN2> before <TOWN>", "my grandparents are from <TOWN>", "anyone else from <TOWN>",
    "stuck in <TOWN> traffic ugh", "<TOWN> is so boring nothing to do", "back home in <TOWN> finally",
    "we got a snow day at <SCHOOL>", "does <SCHOOL> have a robotics club", "<SCHOOL_BARE> lost again last night",
    "i switched to <SCHOOL> this year", "everybody at <SCHOOL_BARE> has that app", "<SCHOOL> picture day is tmrw",
    "u go to <SCHOOL_BARE> right", "the <SCHOOL_BARE> bus was late", "i hate the food at <SCHOOL>",
    "im on <STREET> by the gas station", "meet at the corner of <STREET>", "the party is at <ADDRESS>",
    "our apartment is on <STREET>", "drop it off at <ADDRESS> pls", "i live right off <STREET>",
    "my moms number is <PHONE>", "call <PHONE> if im late", "my new number <PHONE>", "<EMAIL> thats where to send stuff",
    "email: <EMAIL>", "<EMAIL> is the one my mom checks", "follow <USER>", "on discord look for <USER>",
    "im <USER> on roblox", "dm <USER> on insta", "lunch number is <ID>", "my student number <ID> isnt working",
    "the portal says <ID> is invalid", "<ME_LAST> <ME_FIRST> 3rd hour", "from <ME> grade <grade>",
]
CHAT_NEUTRAL = [
    "what did u get for number 4", "idk how to do this", "is the test tomorrow or thursday", "omg same",
    "can someone explain fractions", "brb dinner", "this worksheet is impossible", "lol ok", "wait which chapter",
    "i forgot my book at home", "did anyone finish the lab", "thx", "ok ill try that", "nvm i got it",
    "my wifi keeps dropping", "ugh i have practice till 6", "are we allowed to use a calculator",
    "we watched a video about <place> in class", "our project is on <famous>", "i think the answer is 12",
    "middle school is so much harder", "wish it was summer already", "who else has <subject> homework",
    "my teacher said <place> has the biggest desert", "do we need to know <famous> for the quiz", "gtg my mom is calling",
    "can you check my essay", "how many paragraphs does it need", "i am so tired today", "we have a sub today",
    "<n> more days till break", "i got <n> out of 20", "page <n> is missing from my packet",
]
TUTOR_PII = [
    "Tutor: Thanks for joining, <ME_FIRST>.", "Tutor: <ME_FIRST>, can you read the next line?",
    "Tutor: The recap goes out to <EMAIL> tonight.", "Tutor: Is <PHONE> still the best number for your parents?",
    "Tutor: I have you down as <ME>, is that spelled right?", "Tutor: Which school is that, <SCHOOL>?",
    "Tutor: Has <SCHOOL_BARE> covered decimals yet?", "Tutor: Tell <FAMILY> the next session is Thursday.",
    "Tutor: Are you still in <TOWN> or did you move already?", "Tutor: I worked with <KID> on this last week.",
    "Tutor: Exactly, <ME_FIRST>. Lots of people miss that step.", "Tutor: Your ID on the roster is <ID>, correct?",
    "Student: I'm <ME_FIRST>.", "Student: <ME>.", "Student: my teacher at <SCHOOL> does it differently",
    "Student: we started this unit at <SCHOOL_BARE> last week", "Student: <FAMILY>, my older brother, did the first two with me",
    "Student: <KID> told me the answer but I want to understand it", "Student: my mom wants the report, she uses <EMAIL>",
    "Student: her cell is <PHONE>", "Student: I think my login is <USER>", "Student: it says student number, so <ID>?",
    "Student: it is an hour later here because of <TOWN> time", "Student: I'm visiting my aunt in <TOWN> this week",
    "Student: I ride the bus from <STREET>", "Student: the library on <STREET> has it",
    "Student: since July our house is in <TOWN>", "Student: I was at <SCHOOL> before but now I'm homeschooled",
    "Student: my dad <FAMILY> said to ask about the schedule", "Student: it's <ADDRESS>, do you need the zip?",
    "Student: <SCHOOL_BARE> gives numbers, not letter grades", "Student: I do swim team in <TOWN> on Tuesdays",
    "Parent: This is <FAMILY>, I'm sitting in today.", "Parent: We're at <ADDRESS> if you need to mail anything.",
]
TUTOR_NEUTRAL = [
    "Tutor: Compare the left side with the right side.", "Tutor: Take your time.", "Tutor: Let's check that with a drawing.",
    "Tutor: What would happen if we doubled it?", "Tutor: Say the problem back to me in your own words.",
    "Tutor: Good. What comes next?", "Tutor: That's a common mix-up, no worries.", "Tutor: Try it with smaller numbers.",
    "Tutor: We have about ten minutes left.", "Tutor: Did you get a chance to finish the practice set?",
    "Student: I always forget which one goes on top", "Student: is it <n>?", "Student: oh wait I see it",
    "Student: can we do another one like that", "Student: I don't understand the word problem",
    "Student: we learned about <place> in <subject>", "Student: my report is on <famous>", "Student: that was easier",
    "Student: do I have to show my work", "Student: I got a <n> on the last quiz", "Student: ok",
    "Student: why do you flip the second fraction", "Student: I'm in grade <grade> so we haven't done that yet",
]
PROSE_PII = [
    "My name is not on the cover so I will write it here: <ME>.", "This journal belongs to <ME_FIRST>.",
    "<ME>, room <n>.", "Written for <subject> by <ME>.", "<KID> is my best friend and we sit together at lunch.",
    "<KID> and I made a poster for the science fair.", "My little brother <FAMILY> always wants to play with my things.",
    "On Saturday my aunt <FAMILY> took us to the lake.", "I interviewed my grandfather, <FAMILY>, for this project.",
    "My partner <KID_FULL> did the drawings.", "At <SCHOOL> the class I like best is <subject>.",
    "<SCHOOL> has a new playground this year.", "The students at <SCHOOL_BARE> collected cans for the food bank.",
    "Before <SCHOOL_BARE> I went to a much smaller school.", "Everyone at <SCHOOL> wears a uniform on Mondays.",
    "Our town, <TOWN>, has one stoplight and a diner.", "I have lived in <TOWN> my whole life.",
    "Last year my family moved to <TOWN> because of my mom's job.", "My cousins live far away in <TOWN>.",
    "Each August my grandparents host everyone in <TOWN>.", "<TOWN> gets very hot in July.",
    "People in <TOWN> know each other because it is small.", "I was nervous when we left <TOWN2> for <TOWN>.",
    "Our place is on <STREET>, close to the park.", "My bus stop is at the end of <STREET>.",
    "There is a big oak tree in front of our house at <ADDRESS>.", "My friend lives two doors down on <STREET>.",
    "I walk down <STREET> to get to school.", "Replies can go to <EMAIL>.",
    "You can call my mom at <PHONE> about the field trip.", "My library card number is <ID>.",
    "In the game I play as <USER>, and my castle took a week.",
]
PROSE_NEUTRAL = [
    "The book was about a girl who trains a dragon.", "I think recess should be longer because we need to move.",
    "First we mixed the vinegar and then we waited.", "My favorite animal is the octopus because it is smart.",
    "We learned that <place> is very far from here.", "<famous> was brave and never gave up.",
    "In the story the boy travels to <place> by boat.", "I worked hard on this and I hope you like it.",
    "The experiment showed that plants need light.", "When I grow up I want to be a vet.",
    "Finishing the last page took me the longest.", "My group could not agree on a topic at first.",
    "I read <n> pages every night before bed.", "Next time I would measure more carefully.",
    "It was the best day of the whole year.", "Middle school will be different from elementary school.",
    "Our class is learning about <famous> this month.", "I am in grade <grade> and I like <subject> the most.",
    "The capital of <place> is hard to spell.", "Then it started to rain and we ran inside.",
]
ASSISTANT_PII = [
    "hi im <ME_FIRST> can u help with my homework", "this problem is from my <subject> class at <SCHOOL>",
    "write a poem about my town <TOWN>", "can you write a letter to my friend <KID>",
    "make a story where <ME_FIRST> and <KID> find a treasure", "i need a speech for student council at <SCHOOL_BARE>",
    "what is there to do in <TOWN> for kids", "how far is <TOWN> from <TOWN2>",
    "help me write to my grandma <FAMILY>", "my address is <ADDRESS> how do i write it on an envelope",
    "is <STREET> a good name for a street in my story, its where i live", "add <ME> as the author",
    "email it to <EMAIL>", "<USER> is my username, got a cooler one",
]
ASSISTANT_NEUTRAL = [
    "what is 3/4 plus 1/8", "explain photosynthesis like im 10", "who was <famous>", "what language do they speak in <place>",
    "give me 5 facts about <place>", "how do you spell necessary", "can you check my paragraph",
    "why is the sky blue", "what should i write about for my <subject> project", "how long is a marathon",
]
HEADERS = [
    "Name: <ME>\nClass: <subject>\nSchool: <SCHOOL>", "<ME>\n<SCHOOL>\nGrade <grade>", "NAME <ME>\nDATE ____\nTEACHER <title> <KID_LAST>",
    "By <ME>", "<ME_FIRST> <ME_LAST>, <SCHOOL_BARE>", "Student: <ME> / ID: <ID>", "<ME>, <TOWN>",
    "From: <ME>\nTo: <title> <KID_LAST>\nRe: homework", "Reading log - <ME_FIRST>", "<SCHOOL> - <subject> - <ME>",
]


class Persona:
    def __init__(self, fake: Faker, rng: random.Random):
        self.rng = rng
        self.fake = fake
        self.first, self.last = fake.first_name(), fake.last_name()
        self.kids = [fake.first_name() for _ in range(3)]
        self.family = [fake.first_name() for _ in range(2)]
        self.kid_last = fake.last_name()
        towns = rng.sample(TOWNS, 2) if rng.random() < 0.6 else [fake.city(), fake.city()]
        self.town, self.town2 = towns
        self.street = f"{rng.choice(STREET_NAMES)} {rng.choice(STREET_KINDS)}"
        self.address = f"{rng.randint(1, 9999)} {self.street}"
        if rng.random() < 0.3:
            self.address += f", {self.town}"
        base = rng.choice([fake.last_name(), rng.choice(STREET_NAMES[:30]), self.town.split()[0], f"St. {fake.first_name()}"])
        self.school_bare = base
        self.school = f"{base} {rng.choice(SCHOOL_KINDS)}"
        self.email = rng.choice([fake.free_email(), f"{self.first.lower()}{self.last.lower()}{rng.randint(1, 99)}@{fake.free_email_domain()}"])
        self.user = rng.choice([fake.user_name(), f"{self.first.lower()}_{rng.randint(10, 9999)}", f"xX{self.last}Xx", f"{rng.choice(['cool', 'pro', 'the', 'lil'])}{self.first.lower()}{rng.randint(1, 99)}"])
        self.phone = fake.numerify(rng.choice(["###-###-####", "(###) ###-####", "##########", "### ### ####", "###.###.####"]))
        self.id = fake.bothify(rng.choice(["#######", "##-#####", "S#######", "??######", "######", "#########"])).upper()

    def value(self, slot: str) -> str:
        rng = self.rng
        return {
            "ME": f"{self.first} {self.last}", "ME_FIRST": self.first, "ME_LAST": self.last,
            "KID": rng.choice(self.kids), "KID_FULL": f"{rng.choice(self.kids)} {self.kid_last}", "KID_LAST": self.kid_last,
            "FAMILY": rng.choice(self.family), "TOWN": self.town, "TOWN2": self.town2,
            "SCHOOL": self.school, "SCHOOL_BARE": self.school_bare, "STREET": self.street, "ADDRESS": self.address,
            "EMAIL": self.email, "USER": self.user, "PHONE": self.phone, "ID": self.id,
            "place": rng.choice(WORLD_PLACES), "famous": rng.choice(FAMOUS), "subject": rng.choice(SUBJECTS),
            "n": str(rng.randint(2, 40)), "grade": str(rng.randint(2, 9)), "title": rng.choice(TEACHER_TITLES),
        }[slot]


def fill(frame: str, p: Persona, rng: random.Random, style: str) -> Tuple[str, List[Tuple[int, int, str]]]:
    """Fill the slots of one frame. `style` changes the casing of the whole line
    the way a child would type it: 'lower', 'upper' (headers only) or 'asis'."""
    out, spans, pos = [], [], 0
    for m in SLOT_RE.finditer(frame):
        literal = frame[pos:m.start()]
        out.append(literal.lower() if style == "lower" else literal.upper() if style == "upper" else literal)
        value = p.value(m.group(1))
        if style == "lower":
            value = value.lower()
        elif style == "upper" and m.group(1) not in {"EMAIL", "USER"}:
            value = value.upper()
        start = sum(len(x) for x in out)
        out.append(value)
        label = LABELLED.get(m.group(1))
        if label:
            spans.append((start, start + len(value), label))
        pos = m.end()
    tail = frame[pos:]
    out.append(tail.lower() if style == "lower" else tail.upper() if style == "upper" else tail)
    return "".join(out), spans


def make_doc(fake: Faker, rng: random.Random, pii_rate: float) -> Tuple[str, List[Tuple[int, int, str]]]:
    p = Persona(fake, rng)
    register = rng.choices(["chat", "tutor", "prose", "assistant"], weights=[35, 25, 28, 12])[0]
    pii, neutral = {"chat": (CHAT_PII, CHAT_NEUTRAL), "tutor": (TUTOR_PII, TUTOR_NEUTRAL),
                    "prose": (PROSE_PII, PROSE_NEUTRAL), "assistant": (ASSISTANT_PII, ASSISTANT_NEUTRAL)}[register]
    clean = rng.random() < 0.12  # a document with no identifiers at all
    lines: List[Tuple[str, list]] = []
    if register == "prose" and not clean and rng.random() < 0.35:
        style = "upper" if rng.random() < 0.2 else "asis"
        lines.append(fill(rng.choice(HEADERS), p, rng, style))
    for _ in range(rng.randint(3, 9) if register != "assistant" else rng.randint(1, 3)):
        frame = rng.choice(pii) if (not clean and rng.random() < pii_rate) else rng.choice(neutral)
        if register in {"chat", "assistant"}:
            style = "lower" if rng.random() < 0.7 else "asis"
        elif register == "tutor" and frame.startswith("Student:") and rng.random() < 0.3:
            text, spans = fill(frame[len("Student: "):], p, rng, "lower")
            lines.append(("Student: " + text, [(s + 9, e + 9, l) for s, e, l in spans]))
            continue
        else:
            style = "asis"
        lines.append(fill(frame, p, rng, style))
    # Header lines keep their own line; prose sentences usually run on, everything else is one line per turn.
    nl = chr(10)
    sep = " " if register == "prose" and rng.random() < 0.7 else nl
    text, spans = "", []
    for i, (line, sp) in enumerate(lines):
        if i:
            text += nl if nl in line or nl in lines[i - 1][0] else sep
        cursor = len(text)
        text += line
        spans.extend((s + cursor, e + cursor, l) for s, e, l in sp)
    return text, spans


def to_piilo(doc_id: int, text: str, spans: List[Tuple[int, int, str]]) -> dict:
    char_labels = ["O"] * len(text)
    starts = set()
    for s, e, label in spans:
        starts.add(s)
        for i in range(s, e):
            char_labels[i] = label
    tokens, ws, labels = [], [], []
    prev = None
    for m in TOKEN_RE.finditer(text):
        tok = m.group()
        lab = "O" if tok.startswith("\n") else char_labels[m.start()]
        if lab == "O":
            labels.append("O")
            prev = None
        else:
            begins = prev != lab or m.start() in starts
            labels.append(f"{'B' if begins else 'I'}-{lab}")
            prev = lab
        tokens.append(tok)
        ws.append(m.end() < len(text) and text[m.end()] == " ")
    return {"document": doc_id, "full_text": text, "tokens": tokens, "trailing_whitespace": ws, "labels": labels}


def generate(n: int, seed: int, pii_rate: float = 0.5) -> List[dict]:
    fake = Faker("en_US")
    fake.seed_instance(seed)
    rng = random.Random(seed)
    return [to_piilo(i, *make_doc(fake, rng, pii_rate)) for i in range(n)]


def all_frames() -> List[str]:
    return (CHAT_PII + CHAT_NEUTRAL + TUTOR_PII + TUTOR_NEUTRAL + PROSE_PII + PROSE_NEUTRAL
            + ASSISTANT_PII + ASSISTANT_NEUTRAL + HEADERS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--pii_rate", type=float, default=0.5)
    a = ap.parse_args()
    docs = generate(a.n, a.seed, a.pii_rate)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(docs), encoding="utf-8")
    counts: dict = {}
    for d in docs:
        for lab in d["labels"]:
            if lab.startswith("B-"):
                counts[lab[2:]] = counts.get(lab[2:], 0) + 1
    print(f"wrote {len(docs)} documents to {a.out}; spans: {dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()
