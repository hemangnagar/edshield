---
license: cc-by-4.0
language: en
library_name: transformers.js
pipeline_tag: token-classification
base_model: edshield/piilo-deberta-v3-small
tags:
  - pii
  - de-identification
  - education
  - student-privacy
  - edshield
  - onnx
---

# piilo-deberta-v3-small-onnx

ONNX export of [edshield/piilo-deberta-v3-small](https://huggingface.co/edshield/piilo-deberta-v3-small), for ONNX Runtime and for the browser through Transformers.js. It finds personal identifiers in student writing, and is the model the [edshield](https://github.com/hemangnagar/edshield) browser demo runs on the device.

| File | Size | What it is |
|---|---|---|
| `onnx/model.onnx` | 566 MB | Full precision (fp32) |
| `onnx/model_quantized.onnx` | 205 MB | INT8, with the first two encoder layers left at full precision |

Seven labels, in BIO form: `NAME_STUDENT`, `EMAIL`, `USERNAME`, `ID_NUM`, `PHONE_NUM`, `URL_PERSONAL`, `STREET_ADDRESS`.

## Use

```js
import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.2";

const pipe = await pipeline("token-classification", "edshield/piilo-deberta-v3-small-onnx", { dtype: "q8" });
const tokens = await pipe("Hi, this is Marcus. My email is marcus.t2012@gmail.com", { ignore_labels: [] });
```

This gives one label per token. Turning tokens into spans, adding the rules and applying a policy is done by the JavaScript in edshield's `demo/index.html`, which is a port of the Python library and is tested against it. The model file is downloaded once and cached by the browser; the text you run it on does not leave the device.

## Results

Span-level, on 680 PIILO essays held out from training, measured with edshield's `eval/evaluate_onnx.py`, which runs these files under ONNX Runtime with the same rules, windows and decoding as the Python library. "Got through" counts identifiers that no flag of any label touched.

| Test set | Detector | Precision | Recall | F5 | Got through |
|---|---|---|---|---|---|
| PIILO held-out, 680 essays | Rules + `model_quantized.onnx` | 0.639 | 1.000 | 0.979 | 0 of 165 |
| PIILO held-out, 680 essays | Rules + full PyTorch model | 0.642 | 1.000 | 0.979 | 0 of 165 |
| K-12 synthetic, hard | Rules + `model_quantized.onnx` | | | | 324 of 1,433 (23%) |
| K-12 synthetic, hard | Rules + full PyTorch model | | | | 319 of 1,433 (22%) |

Read these with caution:

1. **The quantized file is an approximation of the full model.** Quantizing every layer gave a 172 MB file that let 7 of the 165 identifiers through. Leaving the first two encoder layers at full precision removed those misses at a cost of 33 MB.
2. **That setting was chosen by its result on the PIILO essays,** so the PIILO row is a little optimistic. The K-12 hard set played no part in the choice.
3. **There is no measurement on real writing by children.** PIILO is writing by adult online learners, and the K-12 sets are synthetic. On child-style text about one identifier in five gets through.
4. **The PIILO set is small.** 143 of its 165 identifiers are names.
5. **These figures are from ONNX Runtime on a desktop CPU.** The browser runs the same file through ONNX Runtime's WebAssembly build; the demo's output is checked against the Python library on three synthetic samples, not on the full test set.

The full reports are in the repository under `eval/results/`, and [docs/COVERAGE.md](https://github.com/hemangnagar/edshield/blob/main/docs/COVERAGE.md) has the breakdown by identifier type.

## What this model is not

It is not a compliance certificate, and using it does not by itself make a product meet FERPA, COPPA or any other law. Those laws place duties on the school or the operator. This model is one technical safeguard inside that, and it does not remove all personal information.

## Export

`training/export_onnx.py` in the repository: `torch.onnx.export` at opset 17, then dynamic quantization with ONNX Runtime, `--keep-layers 2`. No CPU-specific instructions are used, so the same file runs on a Chromebook, an ARM laptop and a desktop.

## Licence and attribution

The weights are released under CC BY 4.0, the licence of the training data.

Trained on the PIILO corpus (the dataset of the Kaggle competition "The Learning Agency Lab - PII Data Detection"), created by [The Learning Agency Lab](https://the-learning-agency-lab.com/) with Vanderbilt University and released under CC BY 4.0.

The base model, DeBERTa-v3-small, is by Microsoft and released under the MIT licence. The edshield code is Apache-2.0.
