import random

import pytest
import edshield
from edshield import Entity
from edshield.deid import apply_deidentification, check_no_leak

T = ("Hi, my name is Priya Raman. Email priya.raman08@gmail.com, phone 703-555-0142, "
     "student ID 4471882, insta @priya.draws, born 03/14/2012.")


def test_mask_ferpa_no_leak():
    r = edshield.deidentify(T, policy="ferpa", model_name="rules")
    assert "Priya" not in r.deidentified_text
    assert "4471882" not in r.deidentified_text
    assert check_no_leak(r.deidentified_text, r.entities) == []
    assert r.leaks == []


def test_coppa_masks_contact_even_when_replace_requested():
    r = edshield.deidentify(T, policy="coppa", method="replace", model_name="rules", seed=1)
    assert "[EMAIL]" in r.deidentified_text
    assert "[PHONE_NUM]" in r.deidentified_text
    # the name has no per-label method in coppa, so the caller's `replace` applies
    assert "[CHILD]" not in r.deidentified_text
    assert "Priya" not in r.deidentified_text


def test_research_surrogates_consistent():
    text = "My name is Priya Raman. Priya Raman wrote this. Priya Raman also drew the cover."
    r = edshield.deidentify(text, policy="research", model_name="rules", seed=3)
    surrogate = r.replacements["Priya Raman"]
    assert surrogate != "Priya Raman" and not surrogate.startswith("[")
    assert r.deidentified_text.count(surrogate) == 3
    assert "Priya" not in r.deidentified_text


def test_dates_shift_preserve_format():
    r = edshield.deidentify("Due 03/14/2012.", policy="ferpa", model_name="rules", date_shift_days=10)
    assert "03/24/2012" in r.deidentified_text


def test_hash_is_stable():
    a = edshield.deidentify(T, policy="ferpa", method="hash", model_name="rules").deidentified_text
    b = edshield.deidentify(T, policy="ferpa", method="hash", model_name="rules").deidentified_text
    assert a.split("Email ")[1].split(",")[0] == b.split("Email ")[1].split(",")[0]


def test_unknown_policy():
    with pytest.raises(FileNotFoundError):
        edshield.deidentify(T, policy="nope", model_name="rules")


# --- method resolution ------------------------------------------------------

def test_policy_default_method_is_used_when_caller_gives_none():
    r = edshield.deidentify("My name is Priya Raman.", policy="research", model_name="rules", seed=1)
    assert r.method == "replace"
    assert "[" not in r.deidentified_text and "Priya" not in r.deidentified_text
    assert edshield.deidentify("My name is Priya Raman.", policy="ferpa", model_name="rules").method == "mask"


def test_caller_method_overrides_policy_default():
    r = edshield.deidentify("My name is Priya Raman.", policy="research", method="mask", model_name="rules")
    assert r.deidentified_text == "My name is [NAME_STUDENT]."


def test_policy_can_forbid_method_override(tmp_path):
    p = tmp_path / "locked.yaml"
    p.write_text("name: locked\ndefault_method: mask\nallow_method_override: false\n"
                 "labels:\n  NAME_STUDENT: {enabled: true}\n")
    ent = Entity("NAME_STUDENT", "Priya", 0, 5, 0.9)
    r = apply_deidentification("Priya wrote this.", [ent], method="replace", policy=str(p))
    assert r.deidentified_text == "[NAME_STUDENT] wrote this."


# --- verifier ---------------------------------------------------------------

def test_name_inside_a_longer_word_is_not_a_leak():
    r = edshield.deidentify("My name is Ann Lee. Ann went to the Annual fair.", policy="ferpa", model_name="rules")
    assert r.deidentified_text == "My name is [STUDENT]. [STUDENT] went to the Annual fair."


def test_shifted_date_equal_to_another_original_is_not_a_leak():
    r = edshield.deidentify("Draft 03/14/2012, final 03/24/2012.", policy="ferpa", model_name="rules", date_shift_days=10)
    assert r.deidentified_text == "Draft 03/24/2012, final 04/03/2012."


def test_value_left_in_place_is_a_leak():
    # only the first mention is handed over; the second stays in the text
    ent = Entity("NAME_STUDENT", "Priya", 0, 5, 0.9)
    r = apply_deidentification("Priya wrote this. Thanks, Priya!", [ent], policy="ferpa")
    assert r.leaks == ["Priya"]


def test_deidentify_refuses_to_return_a_leak(monkeypatch):
    monkeypatch.setattr(edshield, "propagate_names", lambda text, ents: ents)
    with pytest.raises(RuntimeError, match="leak"):
        edshield.deidentify("My name is Priya Raman. Later Priya Raman left.", policy="ferpa", model_name="rules")


def test_check_no_leak_whole_words_only():
    ents = [Entity("NAME_STUDENT", "Ann", 0, 3, 0.9)]
    assert check_no_leak("the Annual fair, Joanne", ents) == []
    assert check_no_leak("it was Ann's idea", ents) == ["Ann"]


# --- dates and seeds --------------------------------------------------------

def test_zero_date_shift_is_rejected():
    with pytest.raises(ValueError, match="date_shift_days"):
        edshield.deidentify("Due 03/14/2012.", policy="ferpa", model_name="rules", date_shift_days=0)


def test_random_date_shift_is_never_zero():
    for seed in range(400):
        r = edshield.deidentify("Due 03/14/2012.", policy="ferpa", model_name="rules", seed=seed)
        assert "03/14/2012" not in r.deidentified_text


def test_seed_makes_output_reproducible():
    text = "My student ID is 4471882, due 03/14/2012. My name is Priya Raman."
    outs = set()
    for state in (1, 2, 3):
        random.seed(state)  # the caller's global RNG must not matter
        outs.add(edshield.deidentify(text, policy="research", model_name="rules", seed=7).deidentified_text)
    assert len(outs) == 1
    other = edshield.deidentify(text, policy="research", model_name="rules", seed=8).deidentified_text
    assert other not in outs
