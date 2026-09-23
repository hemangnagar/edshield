"""Evaluate any detector (rules, model, or both) on PIILO-format JSON.

Reports the competition's micro F5 at token level plus span-level P/R and,
most importantly for a privacy tool, the leak count: gold entities that the
detector missed entirely.

    python eval/evaluate.py --input data/piilo/train.json --model rules
    python eval/evaluate.py --input data/piilo/train.json --model models/piilo-deberta-v3-small
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--model", default="rules")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--report", default=None, help="write JSON report here")
    a = ap.parse_args()

    docs = json.loads(Path(a.input).read_text())
    if a.limit:
        docs = docs[: a.limit]

    tp = Counter(); fp = Counter(); fn = Counter()
    leaks = []
    for d in docs:
        text, gold = rebuild(d)
        pred = edshield.analyze_text(text, model_name=a.model, device=a.device).entities
        pred_spans = [(e.start, e.end, e.label) for e in pred if e.label in edshield.PIILO_LABELS]
        matched = set()
        for gs, ge, gl in gold:
            hit = None
            for i, (ps, pe, pl) in enumerate(pred_spans):
                if pl == gl and ps < ge and gs < pe:  # overlap counts (competition is token-level)
                    hit = i
                    break
            if hit is None:
                fn[gl] += 1
                leaks.append({"document": d["document"], "label": gl, "text": text[gs:ge]})
            else:
                tp[gl] += 1
                matched.add(hit)
        for i, (_, _, pl) in enumerate(pred_spans):
            if i not in matched:
                fp[pl] += 1

    def prf(label=None):
        t = tp[label] if label else sum(tp.values())
        p_ = fp[label] if label else sum(fp.values())
        n = fn[label] if label else sum(fn.values())
        prec = t / (t + p_) if t + p_ else 0.0
        rec = t / (t + n) if t + n else 0.0
        b2 = 25.0
        f5 = (1 + b2) * prec * rec / (b2 * prec + rec) if prec + rec else 0.0
        return {"tp": t, "fp": p_, "fn": n, "precision": round(prec, 4), "recall": round(rec, 4), "f5": round(f5, 4)}

    report = {"model": a.model, "docs": len(docs), "overall": prf(), "per_label": {l: prf(l) for l in edshield.PIILO_LABELS}, "leaks": leaks[:200]}
    print(f"model={a.model} docs={len(docs)}")
    print(f"overall  P={report['overall']['precision']:.3f} R={report['overall']['recall']:.3f} F5={report['overall']['f5']:.3f}  leaks={sum(fn.values())}")
    for l, m in report["per_label"].items():
        print(f"  {l:15} P={m['precision']:.3f} R={m['recall']:.3f} F5={m['f5']:.3f} (n={m['tp']+m['fn']})")
    if a.report:
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
