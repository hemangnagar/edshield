"""Prepare the PIILO / Kaggle "PII Data Detection" corpus for training.

Get the data (CC BY 4.0, The Learning Agency Lab):

    pip install kaggle
    kaggle competitions download -c pii-detection-removal-from-educational-data
    unzip pii-detection-removal-from-educational-data.zip -d data/piilo

The JSON has one record per document with pre-tokenised text:
    {"document": 7, "full_text": "...", "tokens": [...],
     "trailing_whitespace": [true, ...], "labels": ["O", "B-NAME_STUDENT", ...]}

This script:
  * holds out a validation set of real documents in their natural mix
    (the corpus is ~69% all-"O" documents), so validation precision means
    what it will mean in use,
  * builds the training set from the remaining real documents, keeping
    every positive one and a configurable fraction of the all-"O" ones
    (models train faster and recall improves with 1:2 or so),
  * adds synthetic documents (`--extra`) to the training set only,
  * writes a `datasets` DatasetDict ready for train.py, plus
    validation.json in PIILO format for eval/evaluate.py.

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
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def is_positive(doc) -> bool:
    return any(l != "O" for l in doc["labels"])


def split(docs, extra=(), neg_ratio: float = 0.3, val_frac: float = 0.1, seed: int = 42):
    """Return (train, validation).

    Validation is a random sample of the real `docs`, untouched. Training is
    the rest of the real docs with the all-"O" ones thinned to `neg_ratio`,
    plus every document in `extra`.
    """
    rng = random.Random(seed)
    docs = list(docs)
    rng.shuffle(docs)
    n_val = int(len(docs) * val_frac)
    val, rest = docs[:n_val], docs[n_val:]
    pos = [d for d in rest if is_positive(d)]
    neg = [d for d in rest if not is_positive(d)]
    neg = neg[: int(len(neg) * neg_ratio)]
    train = pos + neg + list(extra)
    rng.shuffle(train)
    return train, val


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

    docs = load(Path(a.input))
    extra = [d for path in a.extra for d in load(Path(path))]
    train, val = split(docs, extra, neg_ratio=a.neg_ratio, val_frac=a.val_frac, seed=a.seed)

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
    keys = ("document", "tokens", "trailing_whitespace", "labels")
    (Path(a.out) / "validation.json").write_text(
        json.dumps([{k: d[k] for k in keys} for d in val]), encoding="utf-8"
    )
    n_pos = sum(is_positive(d) for d in val)
    print(f"real docs={len(docs)} synthetic={len(extra)} -> train={len(train)} "
          f"val={len(val)} ({n_pos} with PII, all real)")
    print(f"saved to {a.out}")


if __name__ == "__main__":
    main()
