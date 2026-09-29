import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

import edshield  # noqa: E402
import edshield.ner  # noqa: E402
from edshield.service import app  # noqa: E402

client = TestClient(app)
TEXT = "My name is Priya Raman, email priya@school.org."


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["policies"] == ["coppa", "ferpa", "research"]


def test_extract_and_deidentify():
    r = client.post("/pii/extract", json={"text": TEXT, "model_name": "rules"})
    assert r.status_code == 200
    assert {(e["label"], e["text"]) for e in r.json()["entities"]} == {
        ("NAME_STUDENT", "Priya Raman"), ("EMAIL", "priya@school.org")}
    r = client.post("/pii/deidentify", json={"text": TEXT, "model_name": "rules", "policy": "ferpa"})
    assert r.status_code == 200
    assert r.json()["deidentified_text"] == "My name is [STUDENT], email [EMAIL]."


def test_response_never_echoes_the_original_text():
    r = client.post("/pii/deidentify", json={"text": TEXT, "model_name": "rules"})
    assert "original_text" not in r.json()


@pytest.mark.parametrize("policy", ["nope", "pyproject.toml", "ci/github-actions-ci.yml", "../ferpa", "C:/secrets.yaml"])
def test_policy_must_be_a_built_in_name(policy):
    r = client.post("/pii/deidentify", json={"text": TEXT, "model_name": "rules", "policy": policy})
    assert r.status_code == 400 and "Unknown policy" in r.json()["detail"]


@pytest.mark.parametrize("model", ["someone/some-model", "C:/models/x", ".", "../models"])
@pytest.mark.parametrize("path", ["/pii/extract", "/pii/deidentify"])
def test_model_must_be_rules_or_in_the_manifest(monkeypatch, path, model):
    def never(*a, **k):
        raise AssertionError("the model layer must not be reached")
    monkeypatch.setattr(edshield.ner, "detect_model", never)
    r = client.post(path, json={"text": TEXT, "model_name": model})
    assert r.status_code == 400 and "Unknown model" in r.json()["detail"]


def test_bad_method_and_labels_are_rejected():
    r = client.post("/pii/deidentify", json={"text": TEXT, "model_name": "rules", "method": "rot13"})
    assert r.status_code == 422
    r = client.post("/pii/extract", json={"text": TEXT, "model_name": "rules", "labels": ["NAME"]})
    assert r.status_code == 400


def test_unavailable_model_is_503(monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("no weights on disk")
    monkeypatch.setattr(edshield.ner, "detect_model", broken)
    r = client.post("/pii/extract", json={"text": TEXT, "model_name": "piilo_deberta_small"})
    assert r.status_code == 503


def test_unverified_output_is_422_and_carries_no_text(monkeypatch):
    monkeypatch.setattr(edshield, "propagate_names", lambda text, ents: ents)
    body = {"text": "My name is Priya Raman. Later Priya Raman left.", "model_name": "rules"}
    r = client.post("/pii/deidentify", json=body)
    assert r.status_code == 422
    assert "Priya" not in r.text
