"""Model authority: with a model loaded, NAME_STUDENT / ID_NUM / STREET_ADDRESS
come from the model only; the rules keep the labels they are precise on.

The model is stubbed, so these run without torch or weights.
"""
import pytest

import edshield
import edshield.ner
from edshield import Entity


def stub_model(monkeypatch, found=()):
    """Replace the model with one that 'finds' each (label, substring) in `found`."""
    def fake(text, **kw):
        out = []
        for label, sub in found:
            s = text.index(sub)
            out.append(Entity(label, sub, s, s + len(sub), 0.98, source="model"))
        return out
    monkeypatch.setattr(edshield.ner, "detect_model", fake)


def broken_model(monkeypatch):
    def fake(text, **kw):
        raise RuntimeError("no weights on disk")
    monkeypatch.setattr(edshield.ner, "detect_model", fake)


def got(text, **kw):
    kw.setdefault("model_name", "stub")
    return {(e.label, e.text, e.source) for e in edshield.analyze_text(text, **kw).entities}


ESSAY = ("In 2019 the team moved to 12 Design Thinking Way and cited report 20190412771 "
         "by Steve Blank.")


def test_rules_alone_overflag_the_essay():
    labels = {l for l, _, _ in got(ESSAY, model_name="rules")}
    assert {"ID_NUM", "STREET_ADDRESS", "NAME_STUDENT"} <= labels


def test_model_silences_rule_false_positives(monkeypatch):
    stub_model(monkeypatch)
    assert got(ESSAY) == set()


@pytest.mark.parametrize("label,text,value", [
    ("NAME_STUDENT", "My name is Priya Raman and I like math.", "Priya Raman"),
    ("ID_NUM", "My student ID is 4471882.", "4471882"),
    ("STREET_ADDRESS", "I live at 1420 Maple Ridge Ct near the park.", "1420 Maple Ridge Ct"),
])
def test_authority_labels_come_from_model_only(monkeypatch, label, text, value):
    stub_model(monkeypatch)
    assert got(text) == set()
    stub_model(monkeypatch, [(label, value)])
    assert got(text) == {(label, value, "model")}


def test_rules_keep_their_labels_when_model_finds_nothing(monkeypatch):
    stub_model(monkeypatch)
    text = ("Email sam.lee@school.org, call 703-555-0142, insta @samlee_art, "
            "page samwrites.wordpress.com, SSN 123-45-6789, born 03/14/2012.")
    assert got(text) == {
        ("EMAIL", "sam.lee@school.org", "rules"),
        ("PHONE_NUM", "703-555-0142", "rules"),
        ("USERNAME", "samlee_art", "rules"),
        ("URL_PERSONAL", "samwrites.wordpress.com", "rules"),
        ("SSN", "123-45-6789", "rules"),
        ("DATE", "03/14/2012", "rules"),
    }


def test_model_is_second_opinion_on_rule_labels(monkeypatch):
    # a handle with no cue: the rules miss it, the model adds it
    stub_model(monkeypatch, [("USERNAME", "coolkid_77")])
    assert ("USERNAME", "coolkid_77", "model") in got("Add coolkid_77 to the group chat.")


def test_deferred_rule_cannot_crowd_out_a_kept_rule(monkeypatch):
    # "id: <url>" makes the cued ID rule claim the URL's span and win the
    # overlap; with the model in charge of ID_NUM the URL must survive.
    text = "My id: priya2012.wordpress.com"
    assert ("ID_NUM", "priya2012", "rules") in got(text, model_name="rules")
    stub_model(monkeypatch)
    assert got(text) == {("URL_PERSONAL", "priya2012.wordpress.com", "rules")}


def test_default_model_missing_falls_back_to_rules_with_a_warning(monkeypatch):
    broken_model(monkeypatch)
    monkeypatch.delenv("EDSHIELD_MODEL", raising=False)
    text = "My name is Priya Raman, student ID 4471882."
    with pytest.warns(RuntimeWarning, match="rules only"):
        r = edshield.analyze_text(text)
    assert r.model_name == "rules"
    assert {(e.label, e.text) for e in r.entities} == {("NAME_STUDENT", "Priya Raman"), ("ID_NUM", "4471882")}


def test_requested_model_that_fails_raises(monkeypatch):
    broken_model(monkeypatch)
    with pytest.raises(edshield.ModelUnavailableError, match="no weights on disk"):
        edshield.analyze_text("My name is Priya Raman.", model_name="stub")
    with pytest.raises(edshield.ModelUnavailableError):
        edshield.deidentify("My name is Priya Raman.", model_name="stub")
    monkeypatch.setenv("EDSHIELD_MODEL", "from_env")
    with pytest.raises(edshield.ModelUnavailableError):
        edshield.analyze_text("My name is Priya Raman.")


def test_rules_mode_never_touches_the_model(monkeypatch, recwarn):
    broken_model(monkeypatch)
    r = edshield.analyze_text("My name is Priya Raman.", model_name="rules")
    assert r.model_name == "rules" and len(recwarn) == 0


def test_model_name_reported_even_when_model_finds_nothing(monkeypatch):
    stub_model(monkeypatch)
    assert edshield.analyze_text("Nothing to see here.", model_name="stub").model_name == "stub"


def test_propagation_follows_model_names_only(monkeypatch):
    text = "Maya Chen wrote this with Jordan Banks. Later Maya and Jordan presented it."
    stub_model(monkeypatch, [("NAME_STUDENT", "Maya Chen")])
    res = got(text)
    assert ("NAME_STUDENT", "Maya Chen", "model") in res
    assert ("NAME_STUDENT", "Maya", "propagated") in res
    assert not any("Jordan" in t for _, t, _ in res)  # rule-cued name, not the model's


def test_empty_authority_restores_union(monkeypatch):
    stub_model(monkeypatch)
    res = got("My name is Priya Raman, student ID 4471882.", model_authority=())
    assert ("NAME_STUDENT", "Priya Raman", "rules") in res
    assert ("ID_NUM", "4471882", "rules") in res


def test_labels_filter_still_applies(monkeypatch):
    stub_model(monkeypatch, [("NAME_STUDENT", "Priya Raman")])
    res = got("My name is Priya Raman, email priya@school.org.", labels=["EMAIL"])
    assert res == {("EMAIL", "priya@school.org", "rules")}


def test_deidentify_uses_model_names_and_does_not_leak(monkeypatch):
    stub_model(monkeypatch, [("NAME_STUDENT", "Priya Raman")])
    text = "Priya Raman here. Email priya.raman08@gmail.com. Priya says hi."
    out = edshield.deidentify(text, policy="ferpa", model_name="stub").deidentified_text
    assert "Priya" not in out and "gmail" not in out
