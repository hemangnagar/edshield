import pytest

from edshield.rules import detect_rules, propagate_names, resolve_overlaps


def labels(text):
    return {(e.label, e.text) for e in detect_rules(text)}


def test_email_and_username_do_not_collide():
    got = labels("Reach me at sam.lee@school.org or @samlee_art on insta")
    assert ("EMAIL", "sam.lee@school.org") in got
    assert ("USERNAME", "samlee_art") in got
    assert not any(l == "URL_PERSONAL" for l, _ in got)


def test_phone_beats_id_and_survives_trailing_period():
    got = labels("Text me at 703-555-0142.")
    assert ("PHONE_NUM", "703-555-0142") in got
    assert not any(l == "ID_NUM" for l, _ in got)


def test_student_id_cue():
    got = labels("My student ID is 4471882 and my lunch number is 88213")
    assert ("ID_NUM", "4471882") in got
    assert ("ID_NUM", "88213") in got


def test_reference_urls_are_not_personal():
    got = labels("See https://en.wikipedia.org/wiki/Cell and my page priyawrites.wordpress.com")
    assert ("URL_PERSONAL", "priyawrites.wordpress.com") in got
    assert not any("wikipedia" in t for _, t in got)


def test_name_cue_full_name():
    got = labels("My name is Priya Raman and I like math.")
    assert ("NAME_STUDENT", "Priya Raman") in got


def test_name_cue_rejects_stopwords():
    assert not any(l == "NAME_STUDENT" for l, _ in labels("I am Going Home today."))


def test_ssn_and_address():
    got = labels("SSN 123-45-6789, lives at 1420 Maple Ridge Ct, Vienna, VA 22182")
    assert ("SSN", "123-45-6789") in got
    assert any(l == "STREET_ADDRESS" for l, _ in got)


def test_no_pii_clean_text():
    assert detect_rules("Photosynthesis converts light energy into chemical energy in chloroplasts.") == []


# --- ordinary prose must stay clean -----------------------------------------
# Each sentence is modelled on a false positive the rules produced on real
# student essays.

@pytest.mark.parametrize("text", [
    # ID_NUM: "id" inside a word, a cue with no number, ranges, dates, round numbers
    "We were able to identify the root cause during the ideation phase.",
    "It was identified that the visual identity needed work.",
    "Everyone has to wear ID badges, and the id cards were checked.",
    "COVID-19 changed how school worked in 2019-2020.",
    "The restaurant seats 100-200 people and we saved 250-300 hours.",
    "The report was filed on 14-09-2021 and again on 2021-09-14.",
    "The city has 12000000 residents.",
    # STREET_ADDRESS: numbers followed by ordinary words
    "There were 5 people in the park and 3 stars on the flag.",
    "We interviewed 4 stakeholders and spent 10 million on the first run.",
    "In week 3 of this course we waited 20 minutes for the way home.",
    # NAME_STUDENT: headings, weak cues, days and seasons
    "Challenge and Selection\n\nI work in a small team.",
    "Insight and Approach\n\nThe tool helped us.",
    "I went with New York friends and United States history.",
    "It's Monday and this is Spring Break.",
    # USERNAME: platform words used as ordinary words
    "My login is broken and nobody could handle the management issues.",
    "I think twitter is great for news and it is hard to handle 3 sensors.",
    "Instagram: Familiarity and narrow reach",
    # URL_PERSONAL: a missing space after a full stop, and reference sites
    "It has truly succeeded.  5.Approach: we built a prototype.",
    "That can be a solution.Education systems differ.",
    "See https://www.forbes.com/sites/story and http://tutorials.istudy.psu.edu/conceptmaps/ for more.",
])
def test_ordinary_prose_is_not_flagged(text):
    assert {(l, t) for l, t in labels(text) if l != "DATE"} == set()


# --- and real identifiers must still be caught -------------------------------

@pytest.mark.parametrize("text,expected", [
    ("My ID: 4471882.", ("ID_NUM", "4471882")),
    ("Student no. S12345678 was late.", ("ID_NUM", "S12345678")),
    ("Roll number 143860010348 is on the sheet.", ("ID_NUM", "143860010348")),
    ("The form says 860632713425 at the top.", ("ID_NUM", "860632713425")),
    ("Mobile: (820)913-3241x894", ("PHONE_NUM", "(820)913-3241x894")),
    ("Call 703-555-0142 ext 12 after school.", ("PHONE_NUM", "703-555-0142 ext 12")),
    ("We live at 591 Smith Centers Apt. 656 now.", ("STREET_ADDRESS", "591 Smith Centers Apt. 656")),
    ("Send it to 12 W 5th Ave please.", ("STREET_ADDRESS", "12 W 5th Ave")),
    ("Hi, this is Marcus.", ("NAME_STUDENT", "Marcus")),
    ("Written by Dennis Boone for class.", ("NAME_STUDENT", "Dennis Boone")),
    ("Sincerely,\nPriya Raman", ("NAME_STUDENT", "Priya Raman")),
    ("My username on the class site is zjones.", ("USERNAME", "zjones")),
    ("username: marcus_hoops", ("USERNAME", "marcus_hoops")),
    ("my discord is marcus_77", ("USERNAME", "marcus_77")),
    ("Watch it at https://www.youtube.com/watch?v=2sOzgGAeiQV today.", ("URL_PERSONAL", "https://www.youtube.com/watch?v=2sOzgGAeiQV")),
    ("My store is holalili.com and it is new.", ("URL_PERSONAL", "holalili.com")),
])
def test_real_identifiers_are_still_caught(text, expected):
    assert expected in labels(text)


def test_propagation_catches_possessive_but_not_name_fragments():
    text = "My name is Priya Raman. Priya's poster won. Mr. O'Raman and Ramanathan were there."
    ents = resolve_overlaps(propagate_names(text, detect_rules(text)))
    found = sorted((e.start, e.text) for e in ents if e.label == "NAME_STUDENT")
    assert [t for _, t in found] == ["Priya Raman", "Priya"]
    assert text[found[1][0]:].startswith("Priya's")
