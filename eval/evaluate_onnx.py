"""Evaluate an exported ONNX file with the scoring of evaluate.py.

The browser demo runs the quantized export, not the PyTorch model, so its
accuracy has to be measured on that file. This swaps the model call in
edshield.ner for ONNX Runtime and leaves windowing, decoding, model authority
and the rules exactly as they are:

    python eval/evaluate_onnx.py --onnx demo/models/piilo-deberta-v3-small-onnx/onnx/model_quantized.onnx \
        --input data/piilo_hf_v2/validation.json

The tokenizer and config are read from the folder above `onnx/` unless
--model-dir is given. Every other argument is passed on to evaluate.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--threads", type=int, default=0)
    a, rest = ap.parse_known_args()

    import numpy as np
    import onnxruntime as ort
    from transformers import AutoTokenizer

    from edshield import ner

    onnx_path = Path(a.onnx)
    model_dir = Path(a.model_dir) if a.model_dir else onnx_path.parent.parent
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    id2label = {int(i): l for i, l in config["id2label"].items()}
    opts = ort.SessionOptions()
    if a.threads:
        opts.intra_op_num_threads = a.threads
    sess = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])

    def predict(chunk, model_id, device, allow_download):
        enc = tok(chunk, return_offsets_mapping=True, return_tensors="np")
        if enc["input_ids"].shape[1] > ner.MAX_TOKENS and len(chunk) > 200:
            cut = ner._cut(chunk, len(chunk) // 4, len(chunk) // 2)
            o1, p1, _ = predict(chunk[:cut], model_id, device, allow_download)
            o2, p2, _ = predict(chunk[cut:], model_id, device, allow_download)
            return o1 + [(s + cut, e + cut) for s, e in o2], p1 + p2, id2label
        offsets = [tuple(o) for o in enc["offset_mapping"][0].tolist()]
        feed = {"input_ids": enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64)}
        logits = sess.run(["logits"], feed)[0][0].astype(np.float64)
        logits -= logits.max(axis=-1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=-1, keepdims=True)
        return offsets, probs.tolist(), id2label

    ner._predict = predict

    import evaluate

    sys.argv = [sys.argv[0], "--model", "onnx"] + rest
    evaluate.main()


if __name__ == "__main__":
    main()
