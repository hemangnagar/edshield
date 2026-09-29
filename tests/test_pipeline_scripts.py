"""The data split and the scorer decide what every reported number means."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(rel):
    spec = importlib.util.spec_from_file_location(Path(rel).stem, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


prepare = load("training/prepare_piilo.py")
evaluate = load("eval/evaluate.py")


def doc(i, positive, source="real"):
    labels = ["B-NAME_STUDENT", "O"] if positive else ["O", "O"]
    return {"document": i, "tokens": ["Priya", "wrote"], "trailing_whitespace": [True, False],
            "labels": labels, "source": source}


REAL = [doc(i, positive=i % 3 == 0) for i in range(300)]
SYNTH = [doc(i, positive=True, source="synthetic") for i in range(100)]  # ids collide with real ones


# --- split --------------------------------------------------------------------

def test_validation_is_real_documents_only():
    train, val = prepare.split(REAL, SYNTH, val_frac=0.1)
    assert len(val) == 30
    assert all(d["source"] == "real" for d in val)
    assert sum(d["source"] == "synthetic" for d in train) == len(SYNTH)


def test_validation_keeps_the_natural_mix_and_training_is_thinned():
    train, val = prepare.split(REAL, neg_ratio=0.3, val_frac=0.2)
    val_neg = sum(not prepare.is_positive(d) for d in val) / len(val)
    assert 0.5 < val_neg < 0.8  # about two thirds of REAL is negative
    rest_neg = sum(not prepare.is_positive(d) for d in REAL) - sum(not prepare.is_positive(d) for d in val)
    assert sum(not prepare.is_positive(d) for d in train) == int(rest_neg * 0.3)


def test_no_document_is_in_both_splits_and_every_positive_is_used():
    train, val = prepare.split(REAL, SYNTH)
    ids = lambda ds: {d["document"] for d in ds if d["source"] == "real"}
    assert ids(train) & ids(val) == set()
    positives = {d["document"] for d in REAL if prepare.is_positive(d)}
    assert positives <= ids(train) | ids(val)


def test_split_is_reproducible_and_leaves_input_alone():
    before = [d["document"] for d in REAL]
    a = prepare.split(REAL, SYNTH, seed=1)
    b = prepare.split(REAL, SYNTH, seed=1)
    c = prepare.split(REAL, SYNTH, seed=2)
    assert a == b and a[1] != c[1]
    assert [d["document"] for d in REAL] == before


# --- scorer -------------------------------------------------------------------

def counts(gold, pred):
    tp, fp, fn, missed = evaluate.score(gold, pred)
    return sum(tp.values()), sum(fp.values()), sum(fn.values()), missed


def test_two_predictions_on_one_gold_span_are_not_a_false_positive():
    gold = [(0, 11, "NAME_STUDENT")]
    pred = [(0, 5, "NAME_STUDENT"), (6, 11, "NAME_STUDENT")]
    assert counts(gold, pred) == (1, 0, 0, [])


def test_one_prediction_covering_two_gold_spans_finds_both():
    gold = [(0, 5, "NAME_STUDENT"), (6, 11, "NAME_STUDENT")]
    assert counts(gold, [(0, 11, "NAME_STUDENT")]) == (2, 0, 0, [])


def test_wrong_label_is_both_a_miss_and_a_false_positive():
    gold = [(0, 5, "NAME_STUDENT")]
    assert counts(gold, [(0, 5, "USERNAME")]) == (0, 1, 1, gold)


def test_unmatched_spans():
    gold = [(0, 5, "EMAIL"), (20, 25, "EMAIL")]
    pred = [(0, 5, "EMAIL"), (40, 45, "EMAIL")]
    assert counts(gold, pred) == (1, 1, 1, [(20, 25, "EMAIL")])


def test_rebuild_round_trips_text_and_spans():
    d = {"tokens": ["Hi", ",", "Priya", "Raman", "!"], "trailing_whitespace": [False, True, True, False, False],
         "labels": ["O", "O", "B-NAME_STUDENT", "I-NAME_STUDENT", "O"]}
    text, spans = evaluate.rebuild(d)
    assert text == "Hi, Priya Raman!"
    assert [(text[s:e], l) for s, e, l in spans] == [("Priya Raman", "NAME_STUDENT")]
