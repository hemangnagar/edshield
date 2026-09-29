"""Evaluate any detector (rules, model, or both) on PIILO-format JSON.

Reports span-level precision, recall and F5 (the competition's metric weights
recall 5:1, but scores tokens; this scores spans, and any overlap with a gold
span of the same label counts as finding it) and, most importantly for a
privacy tool, the leak count: gold entities that the detector missed entirely.

Evaluate a model on documents it was not trained on. prepare_piilo.py writes
the held-out split next to the training data:

    python eval/evaluate.py --input data/piilo_hf/validation.json --model rules
    python eval/evaluate.py --input data/piilo_hf/validation.json --model models/piilo-deberta-v3-small
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import edshield


def rebuild(doc):
    text, spans = [], []  # spans: (start, end, label)
    pos = 0
    cur = None
    for t, ws, lab in zip(doc["tokens"], doc["trailing_whitespace"], doc["labels"]):
        s, e = pos, pos + len(t)
        text.append(t)
        pos = e
        if ws:
            text.append(" ")
            pos += 1
        if lab.startswith("B-") or (lab.startswith("I-") and (cur is None or cur[2] != lab[2:])):
            if cur:
                spans.append(tuple(cur))
            cur = [s, e, lab[2:]]
        elif lab.startswith("I-") and cur:
            cur[1] = e
        else:
            if cur:
                spans.append(tuple(cur))
                cur = None
    if cur:
        spans.append(tuple(cur))
    return "".join(text), spans


def score(gold, pred):
    """Compare (start, end, label) spans. Returns (tp, fp, fn) Counters by label
    and the gold spans that were missed.

    A gold span is found if any prediction of its label overlaps it. A
    prediction is a false positive only if it overlaps no gold span of its
    label, so two predictions covering one gold name are not penalised.
    """
    tp = Counter(); fp = Counter(); fn = Counter()
    missed = []

    def overlaps(a, b):
        return a[2] == b[2] and a[0] < b[1] and b[0] < a[1]

    for g in gold:
        if any(overlaps(g, p) for p in pred):
            tp[g[2]] += 1
        else:
            fn[g[2]] += 1
            missed.append(g)
    for p in pred:
        if not any(overlaps(g, p) for g in gold):
            fp[p[2]] += 1
    return tp, fp, fn, missed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--model", default="rules")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--o_threshold", type=float, default=None, help="entity whenever P(O) < this (default: argmax)")
    ap.add_argument("--labels", choices=["piilo", "all"], default="piilo",
                    help="score the seven PIILO labels, or every label edshield produces")
    ap.add_argument("--report", default=None, help="write JSON report here")
    a = ap.parse_args()

    docs = json.loads(Path(a.input).read_text(encoding="utf-8"))
    if a.limit:
        docs = docs[: a.limit]
    labels = edshield.PIILO_LABELS if a.labels == "piilo" else edshield.ALL_LABELS

    tp = Counter(); fp = Counter(); fn = Counter()
    leaks = []
    exposed = Counter()  # gold spans that no flag of any label touched: what would actually get through
    for d in docs:
        text, gold = rebuild(d)
        gold = [g for g in gold if g[2] in labels]
        pred = edshield.analyze_text(text, model_name=a.model, device=a.device, o_threshold=a.o_threshold).entities
        pred_spans = [(e.start, e.end, e.label) for e in pred if e.label in labels]
        d_tp, d_fp, d_fn, missed = score(gold, pred_spans)
        tp += d_tp; fp += d_fp; fn += d_fn
        for gs, ge, gl in missed:
            untouched = not any(e.start < ge and gs < e.end for e in pred)
            exposed[gl] += untouched
            leaks.append({"document": d["document"], "label": gl, "text": text[gs:ge], "exposed": untouched})

    def prf(label=None):
        t = tp[label] if label else sum(tp.values())
        p_ = fp[label] if label else sum(fp.values())
        n = fn[label] if label else sum(fn.values())
        prec = t / (t + p_) if t + p_ else 0.0
        rec = t / (t + n) if t + n else 0.0
        b2 = 25.0
        f5 = (1 + b2) * prec * rec / (b2 * prec + rec) if prec + rec else 0.0
        return {"tp": t, "fp": p_, "fn": n, "precision": round(prec, 4), "recall": round(rec, 4), "f5": round(f5, 4)}

    n_gold = sum(tp.values()) + sum(fn.values())
    n_exposed = sum(exposed.values())
    report = {"model": a.model, "docs": len(docs), "overall": prf(), "per_label": {l: prf(l) for l in labels},
              "identifiers": n_gold, "exposed": n_exposed, "exposed_by_label": dict(exposed),
              "removed_under_any_label": round(1 - n_exposed / n_gold, 4) if n_gold else None,
              "leaks": leaks[:200]}
    print(f"model={a.model} docs={len(docs)}")
    print(f"overall  P={report['overall']['precision']:.3f} R={report['overall']['recall']:.3f} F5={report['overall']['f5']:.3f}  leaks={sum(fn.values())}")
    # A name found but labelled as the wrong kind of name is still removed from the text.
    print(f"exposed  {n_exposed} of {n_gold} identifiers were touched by no flag of any label")
    for l, m in report["per_label"].items():
        if m["tp"] + m["fn"] + m["fp"]:
            print(f"  {l:15} P={m['precision']:.3f} R={m['recall']:.3f} F5={m['f5']:.3f} (n={m['tp']+m['fn']}, exposed={exposed[l]})")
    if a.report:
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
