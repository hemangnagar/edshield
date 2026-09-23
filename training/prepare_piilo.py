"""Prepare the PIILO / Kaggle "PII Data Detection" corpus for training.

Get the data (CC BY 4.0, The Learning Agency Lab):

    pip install kaggle
    kaggle competitions download -c pii-detection-removal-from-educational-data
    unzip pii-detection-removal-from-educational-data.zip -d data/piilo

The JSON has one record per document with pre-tokenised text:
    {"document": 7, "full_text": "...", "tokens": [...],
     "trailing_whitespace": [true, ...], "labels": ["O", "B-NAME_STUDENT", ...]}

This script:
  * keeps every positive document and a configurable fraction of the
    all-"O" documents (the corpus is ~69% negatives; models train faster and
    recall improves with 1:2 or so),
  * splits by document into train / validation,
  * writes a `datasets` DatasetDict ready for train.py.

Usage:
    python training/prepare_piilo.py --input data/piilo/train.json --out data/piilo_hf
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

LABELS = ["NAME_STUDENT", "EMAIL", "USERNAME", "ID_NUM", "PHONE_NUM", "URL_PERSONAL", "STREET_ADDRESS"]
BIO = ["O"] + [f"{p}-{l}" for l in LABELS for p in ("B", "I")]
LABEL2ID = {l: i for i, l in enumerate(BIO)}


def load(path: Path):
    with open(path) as fh:
        return json.load(fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Kaggle train.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--neg_ratio", type=float, default=0.3, help="fraction of all-O docs to keep")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--extra", nargs="*", default=[], help="extra JSON files in the same format (synthetic data)")
    a = ap.parse_args()

    from datasets import Dataset, DatasetDict  # noqa: WPS433

    random.seed(a.seed)
    docs = load(Path(a.input))
    for extra in a.extra:
        docs += load(Path(extra))

    pos = [d for d in docs if any(l != "O" for l in d["labels"])]
    neg = [d for d in docs if all(l == "O" for l in d["labels"])]
    random.shuffle(neg)
    neg = neg[: int(len(neg) * a.neg_ratio)]
    keep = pos + neg
    random.shuffle(keep)
    n_val = int(len(keep) * a.val_frac)
    val, train = keep[:n_val], keep[n_val:]

    def to_rows(ds):
        return {
            "document": [d["document"] for d in ds],
            "tokens": [d["tokens"] for d in ds],
            "trailing_whitespace": [d["trailing_whitespace"] for d in ds],
            "ner_tags": [[LABEL2ID[l] for l in d["labels"]] for d in ds],
        }

    dd = DatasetDict({"train": Dataset.from_dict(to_rows(train)), "validation": Dataset.from_dict(to_rows(val))})
    dd.save_to_disk(a.out)
    (Path(a.out) / "labels.json").write_text(json.dumps(BIO))
    print(f"docs total={len(docs)} positive={len(pos)} negatives kept={len(neg)} -> train={len(train)} val={len(val)}")
    print(f"saved to {a.out}")


if __name__ == "__main__":
    main()
