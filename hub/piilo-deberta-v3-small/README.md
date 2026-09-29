---
license: cc-by-4.0
language: en
library_name: transformers
pipeline_tag: token-classification
base_model: microsoft/deberta-v3-small
tags:
  - pii
  - de-identification
  - education
  - student-privacy
  - edshield
---

# piilo-deberta-v3-small

A token-classification model that finds personal identifiers in student writing. It is the model layer of [edshield](https://github.com/hemangnagar/edshield), a library that removes student identifiers from text on the device, before the text reaches any language model.

DeBERTa-v3-small (142M parameters) fine-tuned on the PIILO corpus. Seven labels, in BIO form: `NAME_STUDENT`, `EMAIL`, `USERNAME`, `ID_NUM`, `PHONE_NUM`, `URL_PERSONAL`, `STREET_ADDRESS`.

## Use

The model is meant to be used through edshield, which adds rules, overlapping windows for long documents, name propagation, policies and a check that nothing it acted on is left in the output.

```bash
pip install "edshield[hf]"
```

```python
import os
os.environ["EDSHIELD_ALLOW_DOWNLOAD"] = "1"   # edshield downloads nothing unless told to

from edshield import deidentify
text = "Hi, this is Marcus. My email is marcus.t2012@gmail.com and my Discord is @marcus_hoops."
print(deidentify(text, policy="coppa").deidentified_text)
```

After the first run the model is in the local Hugging Face cache and the variable is no longer needed. The model also loads with `transformers` directly (`AutoModelForTokenClassification`), in which case you get token labels only.

For the browser, see [edshield/piilo-deberta-v3-small-onnx](https://huggingface.co/edshield/piilo-deberta-v3-small-onnx).

## Results

Span-level, measured with edshield's `eval/evaluate.py` on 680 PIILO essays held out from training, in the corpus's natural mix (581 of them contain no identifier). The detector is edshield's rules plus this model, which is how the model is used. "Got through" counts identifiers that no flag of any label touched, so they would remain in the output.

| Test set | Detector | Precision | Recall | F5 | Got through |
|---|---|---|---|---|---|
| PIILO held-out, 680 essays | Rules + this model | 0.642 | 1.000 | 0.979 | 0 of 165 |
| PIILO held-out, 680 essays | Rules only | 0.538 | 0.388 | 0.392 | 101 of 165 |
| K-12 synthetic, hard | Rules + this model | | | | 319 of 1,433 (22%) |
| K-12 synthetic, hard | Rules only | | | | 1,100 of 1,433 (77%) |

| Rules + this model, by label | Precision | Recall | n |
|---|---|---|---|
| NAME_STUDENT | 0.656 | 1.000 | 143 |
| URL_PERSONAL | 0.381 | 1.000 | 8 |
| ID_NUM | 0.700 | 1.000 | 7 |
| EMAIL | 1.000 | 1.000 | 4 |
| USERNAME | 1.000 | 1.000 | 2 |
| STREET_ADDRESS | 0.500 | 1.000 | 1 |

Read these with caution:

1. **There is no measurement on real writing by children.** PIILO is writing by adult online learners. The K-12 sets are synthetic and were written by the same people who wrote the detector.
2. **On child-style text about one identifier in five gets through.** What gets through on the hard set is mostly lowercase schools and towns, ages in chat shorthand, spoken dates and streets without a house number.
3. **The PIILO set is small.** 143 of its 165 identifiers are names; most other types have fewer than ten examples, so their rows say little.
4. **Precision is low on purpose.** Most false alarms are names of people other than the essay's author, which PIILO does not label but which a privacy tool should remove.
5. **Some development choices were made on the validation set.** Three fixes for false alarms were chosen by reading false alarms on these same 680 essays.

The full reports and the breakdown by identifier type are in the repository under `eval/results/` and in [docs/COVERAGE.md](https://github.com/hemangnagar/edshield/blob/main/docs/COVERAGE.md).

## What this model is not

It is not a compliance certificate, and using it does not by itself make a product meet FERPA, COPPA or any other law. Those laws place duties on the school or the operator: consent, notice, contracts, retention, security, and the judgement that a record is de-identified. This model is one technical safeguard inside that. It does not remove all personal information. Review a sample of output from your own students' writing before relying on it.

Never test it by pasting real student data into a hosted service. Use synthetic samples.

## Training

| | |
|---|---|
| Base model | `microsoft/deberta-v3-small` |
| Data | PIILO corpus, 6,807 essays. 680 held out for validation. Training used the rest, keeping every essay with an identifier and 30% of those without, plus synthetic documents for the rare labels: 4,430 documents in all |
| Epochs | 3 |
| Learning rate | 2e-5, 10% warm-up, weight decay 0.01 |
| Batch size | 2 |
| Context | 1,024 tokens, stride 128 |
| Loss | Cross-entropy with the `O` class weighted 0.15, to favour recall |
| Precision | fp32 |
| Hardware | One GTX 1060 6 GB, about 42 minutes |

Scripts: `training/prepare_piilo.py` and `training/train.py` in the repository.

## Licence and attribution

The weights are released under CC BY 4.0, the licence of the training data.

Trained on the PIILO corpus (the dataset of the Kaggle competition "The Learning Agency Lab - PII Data Detection"), created by [The Learning Agency Lab](https://the-learning-agency-lab.com/) with Vanderbilt University and released under CC BY 4.0.

The base model, DeBERTa-v3-small, is by Microsoft and released under the MIT licence. The edshield code is Apache-2.0.
