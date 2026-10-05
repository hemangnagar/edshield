"""The child-register training generator: well-formed PIILO records, and frames
that share no five-word run with the benchmark generator (training text worded
like the hard set would make that set circular)."""

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))

import gen_child_register as g  # noqa: E402


def _literals(path: Path):
    out = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.JoinedStr):
            out.append("".join(v.value if isinstance(v, ast.Constant) else " SLOT " for v in node.values))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            out.append(node.value)
    return [s for s in out if len(s.split()) >= 3]


def _runs(strings, n):
    found = set()
    for text in strings:
        text = re.sub(r"<[A-Za-z_]+>|\{[^}]*\}", " SLOT ", text)
        words = [w if w == "SLOT" else w.lower() for w in re.findall(r"[A-Za-z']+", text)]
        found.update(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
    return found


def test_documents_are_well_formed_piilo_records():
    docs = g.generate(200, seed=3)
    labels = set()
    for d in docs:
        assert len(d["tokens"]) == len(d["labels"]) == len(d["trailing_whitespace"])
        rebuilt = "".join(t + (" " if w else "") for t, w in zip(d["tokens"], d["trailing_whitespace"]))
        assert rebuilt == d["full_text"]
        prev = "O"
        for lab in d["labels"]:
            if lab.startswith("I-"):
                assert prev[2:] == lab[2:], "an I- tag must continue a span of the same label"
            prev = lab
        labels.update(lab[2:] for lab in d["labels"] if lab != "O")
    assert {"NAME_STUDENT", "LOCATION", "SCHOOL", "STREET_ADDRESS"} <= labels
    assert any(all(lab == "O" for lab in d["labels"]) for d in docs), "some documents carry no identifiers"


def test_same_seed_same_documents():
    assert g.generate(20, seed=5) == g.generate(20, seed=5)


def test_frames_share_no_five_word_run_with_the_benchmark_generator():
    bench = _runs(_literals(ROOT / "eval" / "k12_bench.py"), 5)
    mine = _runs(g.all_frames(), 5)
    assert not (mine & bench), sorted(mine & bench)[:10]
