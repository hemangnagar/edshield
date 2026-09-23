"""Export a trained model to ONNX (fp32 + INT8) and lay it out for
Transformers.js so the browser demo can load it from a local folder.

    python training/export_onnx.py --model models/piilo-deberta-v3-small \
        --out models/piilo-deberta-v3-small-onnx

Requires `pip install 'edshield[onnx]'`. Output layout:
    <out>/config.json, tokenizer files, onnx/model.onnx, onnx/model_quantized.onnx
which is exactly what `@huggingface/transformers` expects when given a local path.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args()

    from optimum.onnxruntime import ORTModelForTokenClassification, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from transformers import AutoTokenizer

    out = Path(a.out)
    onnx_dir = out / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    model = ORTModelForTokenClassification.from_pretrained(a.model, export=True)
    model.save_pretrained(out)
    AutoTokenizer.from_pretrained(a.model).save_pretrained(out)

    # Move model.onnx into onnx/ (Transformers.js layout) and quantize.
    src = out / "model.onnx"
    if src.exists():
        shutil.move(str(src), onnx_dir / "model.onnx")
    quantizer = ORTQuantizer.from_pretrained(out, file_name="onnx/model.onnx")
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    quantizer.quantize(save_dir=out, quantization_config=qconfig, file_suffix="quantized")
    q = out / "model_quantized.onnx"
    if q.exists():
        shutil.move(str(q), onnx_dir / "model_quantized.onnx")
    print(f"exported to {out}")
    for p in sorted(onnx_dir.iterdir()):
        print(f"  {p.name}  {p.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
