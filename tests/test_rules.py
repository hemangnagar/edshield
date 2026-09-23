import edshield
from edshield.rules import detect_rules


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
