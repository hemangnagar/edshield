"""Fine-tune an encoder for PIILO token classification.

Defaults reproduce the competition's strong baseline: DeBERTa-v3, long
context via stride, and evaluation with the recall-weighted F5 metric.

    python training/train.py --data data/piilo_hf --base microsoft/deberta-v3-small \
        --out models/piilo-deberta-v3-small --epochs 3 --lr 2e-5 --max_len 1024

On a single consumer GPU, -small trains in well under an hour on the
positive-heavy split produced by prepare_piilo.py. On Apple Silicon use
--device mps. Requires `pip install 'edshield[train]'`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", default="microsoft/deberta-v3-small")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--o_weight", type=float, default=0.15, help="loss weight for the O class (recall trick)")
    ap.add_argument("--o_threshold", type=float, default=0.99, help="eval: token is entity if P(O) < this")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true")
    a = ap.parse_args()

    import torch
    from datasets import load_from_disk
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(a.seed)
    dd = load_from_disk(a.data)
    bio = json.loads((Path(a.data) / "labels.json").read_text())
    id2label = dict(enumerate(bio))
    label2id = {l: i for i, l in id2label.items()}

    tok = AutoTokenizer.from_pretrained(a.base)

    def rebuild_text(ex):
        text, char_labels = [], []
        for t, ws, lab in zip(ex["tokens"], ex["trailing_whitespace"], ex["ner_tags"]):
            text.append(t)
            char_labels.extend([lab] * len(t))
            if ws:
                text.append(" ")
                char_labels.append(0)
        return "".join(text), char_labels

    def tokenize(ex):
        text, char_labels = rebuild_text(ex)
        enc = tok(text, return_offsets_mapping=True, truncation=True, max_length=a.max_len,
                  stride=a.stride, return_overflowing_tokens=True)
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for ids, mask, offs in zip(enc["input_ids"], enc["attention_mask"], enc["offset_mapping"]):
            labs = []
            for (s, e) in offs:
                if s == e:
                    labs.append(-100)
                    continue
                # label of first non-space char in the token
                lab = 0
                for c in range(s, e):
                    if c < len(char_labels) and not text[c].isspace():
                        lab = char_labels[c]
                        break
                labs.append(lab)
            out["input_ids"].append(ids)
            out["attention_mask"].append(mask)
            out["labels"].append(labs)
        return out

    cols = dd["train"].column_names
    tds = dd["train"].map(tokenize, remove_columns=cols, batched=False)
    # map() returns lists-of-lists per example; flatten windows
    tds = tds.flatten_indices() if hasattr(tds, "flatten_indices") else tds

    def explode(ds):
        rows = {"input_ids": [], "attention_mask": [], "labels": []}
        for ex in ds:
            for i in range(len(ex["input_ids"])):
                for k in rows:
                    rows[k].append(ex[k][i])
        from datasets import Dataset
        return Dataset.from_dict(rows)

    train_ds = explode(tds)
    val_ds = explode(dd["validation"].map(tokenize, remove_columns=cols, batched=False))

    model = AutoModelForTokenClassification.from_pretrained(
        a.base, num_labels=len(bio), id2label=id2label, label2id=label2id, dtype=torch.float32
    )

    class_weights = torch.ones(len(bio))
    class_weights[0] = a.o_weight

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(logits.device, logits.dtype), ignore_index=-100)
            loss = loss_fct(logits.view(-1, len(bio)), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(p):
        probs = torch.softmax(torch.tensor(p.predictions), dim=-1).numpy()
        preds = probs.argmax(-1)
        # recall trick: anything with P(O) below threshold is an entity
        o_prob = probs[..., 0]
        alt = probs[..., 1:].argmax(-1) + 1
        preds = np.where(o_prob < a.o_threshold, alt, preds)
        y = p.label_ids
        mask = y != -100
        tp = fp = fn = 0
        for pr, gt in zip(preds[mask], y[mask]):
            pr_ent = id2label[int(pr)].split("-")[-1] if pr != 0 else "O"
            gt_ent = id2label[int(gt)].split("-")[-1] if gt != 0 else "O"
            if gt_ent != "O" and pr_ent == gt_ent:
                tp += 1
            elif pr_ent != "O" and pr_ent != gt_ent:
                fp += 1
                if gt_ent != "O":
                    fn += 1
            elif gt_ent != "O" and pr_ent == "O":
                fn += 1
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        beta2 = 25.0
        f5 = (1 + beta2) * prec * rec / (beta2 * prec + rec) if prec + rec else 0.0
        return {"precision": prec, "recall": rec, "f5": f5}

    args = TrainingArguments(
        output_dir=a.out,
        learning_rate=a.lr,
        per_device_train_batch_size=a.bs,
        per_device_eval_batch_size=a.bs,
        num_train_epochs=a.epochs,
        weight_decay=0.01,
        warmup_steps=int(0.1 * a.epochs * len(train_ds) / a.bs),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f5",
        fp16=a.fp16,
        logging_steps=50,
        report_to="none",
        seed=a.seed,
    )
    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForTokenClassification(tok),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(a.out)
    tok.save_pretrained(a.out)
    metrics = trainer.evaluate()
    (Path(a.out) / "eval_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
