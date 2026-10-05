"""Indirect and persistent identifiers: the people and places around a
student (FERPA) and device, network and location identifiers (COPPA)."""
import pytest

import edshield
from edshield.rules import detect_rules


def found(text):
    return {(e.label, e.text) for e in edshield.analyze_text(text, model_name="rules").entities}


@pytest.mark.parametrize("text,expected", [
    # family, friends, teachers
    ("I got 42 but my brother Jordan said that's wrong.", ("NAME_RELATED", "Jordan")),
    ("My friend Daniel Okafor helped me count trays.", ("NAME_RELATED", "Daniel Okafor")),
    ("my mom is Sarah and she works nights", ("NAME_RELATED", "Sarah")),
    ("My little sister Maya broke it.", ("NAME_RELATED", "Maya")),
    ("my teacher Ms. Patel said to use the rubric", ("NAME_RELATED", "Patel")),
    ("Ask Coach Ramirez or Dr. Nguyen about it.", ("NAME_RELATED", "Ramirez")),
    ("Ask Coach Ramirez or Dr. Nguyen about it.", ("NAME_RELATED", "Nguyen")),
    # schools and places
    ("I'm in 8th grade at Rachel Carson Middle School.", ("SCHOOL", "Rachel Carson Middle School")),
    ("At Lincoln High School we have a big gym.", ("SCHOOL", "Lincoln High School")),
    ("She studies at the University of Toledo now.", ("SCHOOL", "University of Toledo")),
    ("We moved to Cedar Falls last summer.", ("LOCATION", "Cedar Falls")),
    ("I live in Vienna with my family.", ("LOCATION", "Vienna")),
    ("The game was in Round Rock, TX this year.", ("LOCATION", "Round Rock, TX")),
    # age and birthday
    ("I am 11 years old and I like soccer.", ("AGE", "11")),
    ("im 12 btw", ("AGE", "12")),
    ("I'm turning 13 in May.", ("AGE", "13")),
    ("my 9-year-old brother", ("AGE", "9")),
    ("My birthday is March 3 and I want a bike.", ("DATE", "March 3")),
    # device, network, location
    ("The log shows 192.168.1.44 at login.", ("IP_ADDRESS", "192.168.1.44")),
    ("Connected from 2001:0db8:85a3:0000:0000:8a2e:0370:7334 today.", ("IP_ADDRESS", "2001:0db8:85a3:0000:0000:8a2e:0370:7334")),
    ("The tablet's MAC is 3C:22:FB:9A:10:5E.", ("DEVICE_ID", "3C:22:FB:9A:10:5E")),
    ("Advertising id 38400000-8cf0-11bd-b23e-10b96e40000d was sent.", ("DEVICE_ID", "38400000-8cf0-11bd-b23e-10b96e40000d")),
    ("Pinned at 38.9012, -77.2653 on the map.", ("GEO", "38.9012, -77.2653")),
])
def test_indirect_identifiers_are_found(text, expected):
    assert expected in found(text)


@pytest.mark.parametrize("text", [
    "I scored 12 points and my answer was 42.",
    "We read chapters 3.1.2 and 4.10 of the book.",
    "The ratio was 1.5 and the time was 10:30:15.",
    "In High School you get more homework.",
    "My teacher said the quiz is on Friday.",
    "My friend and I went to the park.",
    "I am 3 problems behind on the worksheet.",
    "The Design Thinking course was useful.",
    "We live in a small apartment.",
])
def test_ordinary_text_has_no_indirect_identifiers(text):
    new = {"NAME_RELATED", "SCHOOL", "LOCATION", "AGE", "IP_ADDRESS", "DEVICE_ID", "GEO"}
    assert {(l, t) for l, t in found(text) if l in new} == set()


def test_related_names_propagate_and_keep_their_label():
    text = "My brother Jordan helped. Later Jordan's friend left. My name is Priya and Priya agreed."
    got = sorted((e.start, e.label, e.text) for e in edshield.analyze_text(text, model_name="rules").entities)
    assert [(l, t) for _, l, t in got] == [
        ("NAME_RELATED", "Jordan"), ("NAME_RELATED", "Jordan"), ("NAME_STUDENT", "Priya"), ("NAME_STUDENT", "Priya")]


def test_address_wins_over_the_city_inside_it():
    got = found("We live at 1420 Maple Ridge Ct, Vienna, VA 22182 now.")
    assert ("STREET_ADDRESS", "1420 Maple Ridge Ct, Vienna, VA 22182") in got
    assert not any(l == "LOCATION" for l, _ in got)


@pytest.mark.parametrize("policy", ["ferpa", "coppa", "research"])
def test_every_policy_acts_on_the_new_labels(policy):
    text = ("My name is Priya Raman. My brother Jordan goes to Lincoln High School in Round Rock, TX. "
            "I am 11 years old. Device 3C:22:FB:9A:10:5E at 192.168.1.44, pinned at 38.9012, -77.2653.")
    out = edshield.deidentify(text, policy=policy, model_name="rules", seed=1).deidentified_text
    for value in ["Priya", "Jordan", "Lincoln", "Round Rock", "11 years", "3C:22", "192.168", "38.9012"]:
        assert value not in out, (policy, value, out)


def test_labels_are_selectable():
    ents = detect_rules("My brother Jordan is 9 years old.", labels=["AGE"])
    assert [(e.label, e.text) for e in ents] == [("AGE", "9")]


@pytest.mark.parametrize("text,expected", [
    # dates as children say or type them, without a year
    ("I was born the 8th of August and my party is next week.", ("DATE", "8th of August")),
    ("the field trip is on the 3rd of March", ("DATE", "3rd of March")),
    ("recital is August 8th so i cant come", ("DATE", "August 8th")),
    ("party is jul 27 dont forget", ("DATE", "jul 27")),
    ("tryouts got moved to March 3", ("DATE", "March 3")),
    # ages without "years old"
    ("When you are 12 like me you can not drive yet.", ("AGE", "12")),
    ("you're 11?? i thought u were older", ("AGE", "11")),
    ("14m here, anyone want to study", ("AGE", "14m")),
    ("13f looking for a math buddy", ("AGE", "13f")),
    # school names in lowercase
    ("i go to johnson middle school", ("SCHOOL", "johnson middle school")),
    ("i go to riley elementary and i hate fractions", ("SCHOOL", "riley elementary")),
    ("we played against st. mary's high school", ("SCHOOL", "st. mary's high school")),
])
def test_child_register_dates_ages_schools_are_found(text, expected):
    assert expected in found(text)


@pytest.mark.parametrize("text", [
    "brb 5m",
    "see you in 10m",
    "the pool is 25m long and i swam 6m",
    "it took 15m to finish",
    "we are 3 problems behind and you are 12 points ahead",
    "May 5 kids came to the party",
    "you may 5 us later",
    "my old high school had a pool",
    "in middle school you switch classes",
    "the new elementary school opens next year",
    "I read 20 pages on March 3 nights in a row",
])
def test_child_register_rules_leave_ordinary_text_alone(text):
    assert {(lab, t) for lab, t in found(text) if lab in {"DATE", "AGE", "SCHOOL"}} == set()
