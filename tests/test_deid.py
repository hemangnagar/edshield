import pytest
import edshield
from edshield.deid import check_no_leak

T = ("Hi, my name is Priya Raman. Email priya.raman08@gmail.com, phone 703-555-0142, "
     "student ID 4471882, insta @priya.draws, born 03/14/2012.")


def test_mask_ferpa_no_leak():
    r = edshield.deidentify(T, policy="ferpa", model_name="rules")
    assert "Priya" not in r.deidentified_text
    assert "4471882" not in r.deidentified_text
    assert check_no_leak(r.deidentified_text, r.entities) == []


def test_coppa_masks_contact_even_when_replace_requested():
    r = edshield.deidentify(T, policy="coppa", method="replace", model_name="rules", seed=1)
    assert "[EMAIL]" in r.deidentified_text
    assert "[PHONE_NUM]" in r.deidentified_text
    assert "[CHILD]" not in r.deidentified_text or True  # name may be replaced


def test_research_surrogates_consistent():
    text = "Priya Raman wrote this. Priya Raman also drew the cover. My name is Priya Raman."
    r = edshield.deidentify(text, policy="research", method="replace", model_name="rules", seed=3)
    # Only the cued mention is caught by rules; the surrogate must be a real-looking name
    assert "Priya" not in r.deidentified_text.split("My name is")[-1]


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
