"""Model layer without a model: windowing, decoding and loading are tested
with the forward pass stubbed, so no torch or weights are needed."""
import re

import pytest

import edshield
from edshield import ner

ID2LABEL = {0: "O", 1: "B-NAME_STUDENT", 2: "I-NAME_STUDENT", 3: "B-USERNAME", 4: "I-USERNAME"}
FILLER = "The project taught me a lot about how to listen to other people. "
NAME = "Priyanka Ramanathan"


def one_hot(i, p=0.98):
    rest = (1 - p) / (len(ID2LABEL) - 1)
    return [p if j == i else rest for j in range(len(ID2LABEL))]


def fake_predict(calls=None):
    """A 'model' that tags NAME only when it sees the whole name in its window,
    and otherwise tags whatever fragment it sees as a USERNAME."""
    def predict(chunk, model_id, device, allow_download):
        if calls is not None:
            calls.append(chunk)
        offsets, probs = [], []
        words = list(re.finditer(r"\S+", chunk))
        for m in words:
            word = m.group().strip(".,")
            at_edge = m is words[0] or m is words[-1]
            if word in NAME.split() and NAME in chunk:
                cls = 1 if word == NAME.split()[0] else 2
            elif word in NAME.split() or (at_edge and any(word in part for part in NAME.split())):
                cls = 3  # half a name, or a name cut off by the window edge
            else:
                cls = 0
            offsets.append((m.start(), m.start() + len(word)))
            probs.append(one_hot(cls))
        return offsets, probs, ID2LABEL
    return predict


# --- windows ------------------------------------------------------------------

def test_short_text_is_one_window():
    assert ner.windows("hello world") == [(0, 11, 0, 11)]
    assert ner.windows("") == [(0, 0, 0, 0)]


@pytest.mark.parametrize("text", [FILLER * 100, "x" * 7000, ("word " * 30 + "\n") * 60])
def test_windows_overlap_and_keep_ranges_tile_the_text(text):
    ws = ner.windows(text, max_chars=2000, overlap=200)
    assert len(ws) > 1
    assert ws[0][0] == 0 and ws[-1][1] == len(text)
    assert ws[0][2] == 0 and ws[-1][3] == len(text)
    for (s, e, kf, kt), (s2, e2, kf2, kt2) in zip(ws, ws[1:]):
        assert e - s <= 2000
        assert s < s2 < e            # overlap, and progress
        assert kt == kf2             # keep ranges tile with no gap
        assert s2 <= kt <= e         # the boundary lies inside the overlap


def test_windows_cut_between_words():
    text = FILLER * 100
    for s, e, _, _ in ner.windows(text)[:-1]:
        assert text[e - 1].isspace()
        assert s == 0 or text[s - 1].isspace()


# --- decode -------------------------------------------------------------------

def test_decode_joins_word_pieces_and_adjacent_words():
    text = "I met Priya Raman today"
    offsets = [(0, 1), (2, 5), (6, 9), (9, 11), (12, 17), (18, 23)]
    probs = [one_hot(0), one_hot(0), one_hot(1), one_hot(1), one_hot(2), one_hot(0)]
    ents = ner.decode(text, offsets, probs, ID2LABEL)
    assert [(e.label, e.text, e.start, e.end) for e in ents] == [("NAME_STUDENT", "Priya Raman", 6, 17)]
    assert ents[0].source == "model" and ents[0].confidence == pytest.approx(0.98)


def test_decode_covers_the_whole_word():
    text = "I met Priyanka today"
    ents = ner.decode(text, [(6, 9)], [one_hot(1)], ID2LABEL)  # only "Pri" was tagged
    assert [(e.text, e.start, e.end) for e in ents] == [("Priyanka", 6, 14)]


def test_decode_threshold_drops_weak_spans():
    text = "I met Priya today"
    weak = [0.55, 0.45, 0.0, 0.0, 0.0]  # argmax is O
    unsure = [0.3, 0.4, 0.1, 0.1, 0.1]  # argmax is NAME at 0.4
    assert ner.decode(text, [(6, 11)], [weak], ID2LABEL) == []
    assert ner.decode(text, [(6, 11)], [unsure], ID2LABEL, threshold=0.5) == []
    assert len(ner.decode(text, [(6, 11)], [unsure], ID2LABEL, threshold=0.3)) == 1


def test_o_threshold_turns_unsure_tokens_into_entities():
    text = "I met Priya today"
    p = [0.90, 0.07, 0.01, 0.01, 0.01]  # argmax says O, but P(O) < 0.99
    assert ner.decode(text, [(6, 11)], [p], ID2LABEL) == []
    ents = ner.decode(text, [(6, 11)], [p], ID2LABEL, o_threshold=0.99)
    assert [(e.label, e.text) for e in ents] == [("NAME_STUDENT", "Priya")]
    assert ents[0].confidence == pytest.approx(0.10)
    sure_o = [0.995, 0.002, 0.001, 0.001, 0.001]
    assert ner.decode(text, [(6, 11)], [sure_o], ID2LABEL, o_threshold=0.99) == []


def test_decode_skips_special_tokens_and_leading_space():
    text = "Hi Priya"
    ents = ner.decode(text, [(0, 0), (0, 2), (2, 8), (0, 0)], [one_hot(1), one_hot(0), one_hot(1), one_hot(1)], ID2LABEL)
    assert [(e.text, e.start) for e in ents] == [("Priya", 3)]


def test_decode_covers_the_whole_url_from_any_piece_of_it():
    text = "See (https://coursera.org/share/b24116a7056d612f). Done"
    piece = text.index("coursera")
    ents = ner.decode(text, [(piece, piece + 8)], [one_hot(3)], {**ID2LABEL, 3: "B-URL_PERSONAL"})
    assert [(e.label, e.text) for e in ents] == [("URL_PERSONAL", "https://coursera.org/share/b24116a7056d612f")]


def test_decode_drops_a_one_letter_name():
    text = "Mind Mapping By C. Challenge"
    s = text.index("C.")
    assert ner.decode(text, [(s, s + 1)], [one_hot(1)], ID2LABEL) == []


# --- propagation of model names -------------------------------------------------

def test_unsure_model_names_are_flagged_but_not_spread(monkeypatch):
    text = "Little Red Riding Hood met a wolf. Riding Hood ran. Priya wrote it and Priya drew it."

    def predict(chunk, *a):
        hood, priya = chunk.index("Hood"), chunk.index("Priya")
        weak = [0.40, 0.60, 0.0, 0.0, 0.0]
        return [(hood, hood + 4), (priya, priya + 5)], [weak, one_hot(1)], ID2LABEL
    monkeypatch.setattr(ner, "_predict", predict)
    ents = edshield.analyze_text(text, model_name="stub", use_rules=False).entities
    assert [(e.text, e.source) for e in ents] == [("Hood", "model"), ("Priya", "model"), ("Priya", "propagated")]


# --- detect_model -------------------------------------------------------------

@pytest.mark.parametrize("position", [1985, 1990, 1995, 2000, 2005])
def test_name_at_a_window_boundary_is_found_once_and_whole(monkeypatch, position):
    monkeypatch.setattr(ner, "_predict", fake_predict())
    lead = "I worked with my classmate "
    text = (FILLER * 40)[:position - len(lead)] + lead + NAME + " on the prototype. " + FILLER * 5
    ents = ner.detect_model(text, model_name="stub")
    assert [(e.label, e.text) for e in ents] == [("NAME_STUDENT", NAME)]
    assert text[ents[0].start:ents[0].end] == NAME


def test_offsets_are_exact_in_every_window(monkeypatch):
    monkeypatch.setattr(ner, "_predict", fake_predict())
    text = "".join(f"{FILLER * 12}{NAME} spoke next. " for _ in range(6))
    ents = ner.detect_model(text, model_name="stub")
    assert len(ents) == 6
    assert all(text[e.start:e.end] == NAME == e.text for e in ents)


def test_o_threshold_reaches_the_decoder(monkeypatch):
    def predict(chunk, *a):
        s = chunk.index("Priya")
        return [(s, s + 5)], [[0.90, 0.07, 0.01, 0.01, 0.01]], ID2LABEL
    monkeypatch.setattr(ner, "_predict", predict)
    assert edshield.analyze_text("I met Priya today", model_name="stub", use_rules=False).entities == []
    r = edshield.analyze_text("I met Priya today", model_name="stub", use_rules=False, o_threshold=0.99)
    assert [(e.label, e.text) for e in r.entities] == [("NAME_STUDENT", "Priya")]


# --- loading ------------------------------------------------------------------

def test_downloads_are_off_unless_opted_in(monkeypatch):
    seen = []

    def load(model_id, device, allow_download):
        seen.append(allow_download)
        raise OSError("not found locally")
    monkeypatch.setattr(ner, "_load", load)
    monkeypatch.setattr(ner, "_LOAD_FAILURES", {})
    monkeypatch.delenv("EDSHIELD_ALLOW_DOWNLOAD", raising=False)
    with pytest.raises(ner.ModelUnavailableError, match="downloads are off"):
        ner._load_once("some/model", "cpu", ner.downloads_allowed())
    monkeypatch.setenv("EDSHIELD_ALLOW_DOWNLOAD", "1")
    with pytest.raises(ner.ModelUnavailableError):
        ner._load_once("some/model", "cpu", ner.downloads_allowed())
    assert seen == [False, True]


def test_a_failed_load_is_not_retried(monkeypatch):
    attempts = []

    def load(model_id, device, allow_download):
        attempts.append(model_id)
        raise OSError("not found locally")
    monkeypatch.setattr(ner, "_load", load)
    monkeypatch.setattr(ner, "_LOAD_FAILURES", {})
    for _ in range(3):
        with pytest.raises(ner.ModelUnavailableError):
            ner._load_once("some/model", "cpu", False)
    assert attempts == ["some/model"]


def test_resolve_model_id(tmp_path, monkeypatch):
    assert ner.resolve_model_id(str(tmp_path)) == str(tmp_path)
    assert ner.resolve_model_id("someone/some-model") == "someone/some-model"
    monkeypatch.setenv("EDSHIELD_MODEL", str(tmp_path))
    assert ner.resolve_model_id(None) == str(tmp_path)
