"""Export a trained model to ONNX (fp32 + INT8) and lay it out for
Transformers.js so the browser demo can load it from a local folder.

    python training/export_onnx.py --model models/piilo-deberta-v3-small \
        --out demo/models/piilo-deberta-v3-small-onnx

Requires `pip install 'edshield[hf,onnx]'`. Output layout:
    <out>/config.json, tokenizer files, onnx/model.onnx, onnx/model_quantized.onnx
which is what `@huggingface/transformers` expects when given a local path.

The INT8 model uses dynamic quantization with no CPU-specific instructions,
so the same file runs on a Chromebook, an ARM laptop and a desktop. After
export the script runs both files against the PyTorch model and prints how
closely they agree; quantization changes the numbers, so check the agreement
before shipping the small file.
"""
from __future__ import annotations

import argparse
from pathlib import Path

SAMPLE = (
    "My name is Priya Raman and I'm in 8th grade. My friend Daniel Okafor helped me count trays. "
    "You can email me at priya.raman08@gmail.com or text 703-555-0142. My student ID is 4471882."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args()

    import numpy as np
    import onnxruntime as ort
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    out = Path(a.out)
    onnx_dir = out / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForTokenClassification.from_pretrained(a.model, dtype=torch.float32).eval()
    model.config.return_dict = False
    model.config.save_pretrained(out)
    tok.save_pretrained(out)

    enc = tok(SAMPLE, return_tensors="pt")
    fp32 = onnx_dir / "model.onnx"
    torch.onnx.export(
        model,
        (enc["input_ids"], enc["attention_mask"]),
        str(fp32),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"},
        },
        opset_version=a.opset,
        dynamo=False,
    )
    int8 = onnx_dir / "model_quantized.onnx"
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QUInt8)

    # Agreement with PyTorch, on the sample and on a longer input than was traced.
    print(f"exported to {out}")
    for text in (SAMPLE, SAMPLE * 6):
        enc = tok(text, return_tensors="pt")
        with torch.no_grad():
            ref = model(enc["input_ids"], enc["attention_mask"])[0][0].numpy()
        feed = {"input_ids": enc["input_ids"].numpy(), "attention_mask": enc["attention_mask"].numpy()}
        for path in (fp32, int8):
            sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            got = sess.run(["logits"], feed)[0][0]
            agree = float((got.argmax(-1) == ref.argmax(-1)).mean())
            print(f"  {path.name:22} {path.stat().st_size / 1e6:6.1f} MB  tokens={len(ref):4}  "
                  f"max |logit diff|={float(np.abs(got - ref).max()):.3f}  same label on {agree:.1%} of tokens")


if __name__ == "__main__":
    main()
