# edshield

**Local-first student-privacy layer for AI in education.**
Detects and removes student PII from essays, tutoring transcripts and chat messages before the text reaches any language model. Runs on a laptop CPU, Apple Silicon, or inside a Chromebook browser. No cloud, no student text leaves the device. Apache-2.0.

```python
from edshield import extract_pii, deidentify

text = "Hi, this is Marcus. My email is marcus.t2012@gmail.com and my Discord is @marcus_hoops."

print([(e.label, e.text) for e in extract_pii(text).entities])
# [('NAME_STUDENT', 'Marcus'), ('EMAIL', 'marcus.t2012@gmail.com'), ('USERNAME', 'marcus_hoops')]

print(deidentify(text, policy="coppa").deidentified_text)
# Hi, this is [CHILD]. My email is [EMAIL] and my Discord is @[USERNAME].
```

## Why

Every AI tutor, writing assistant and classroom chatbot sends student text to a model. FERPA, COPPA and a growing set of state laws (NY Ed Law 2-d, Illinois SOPPA, California SOPIPA) say identifiable student data cannot be handed to a third party without consent, and district AI policies are now saying it outright. Today each vendor and each research group rebuilds the same de-identification pipeline privately. edshield is the shared, open one.

The design follows [OpenMed](https://github.com/maziyarpanahi/openmed): small fine-tuned encoders for the domain, a rule layer for the identifiers where regex plus validation beats a neural model, policy profiles named after the regulation, and runtimes for Python, ONNX and the browser.

## What it does

| Layer | Covers | Needs a model? |
|---|---|---|
| Rules | EMAIL, PHONE_NUM, URL_PERSONAL, USERNAME, ID_NUM, STREET_ADDRESS, SSN, DATE, and names introduced with a cue ("my name is…", a signature) | No |
| Rules, beyond PIILO | NAME_RELATED (family, friends, teachers), SCHOOL, LOCATION, AGE, IP_ADDRESS, DEVICE_ID, GEO | No |
| Model | The authority for NAME_STUDENT, ID_NUM and STREET_ADDRESS, plus a second opinion on every other label | Yes (PIILO-trained encoder) |
| Propagation | Once a name is found, every other mention of it in the document is caught | No |
| Policies | `ferpa`, `coppa`, `research` decide which labels to act on, the confidence floor, and the method per label (mask, surrogate, hash, date-shift) | No |
| Verifier | Refuses to return output if any acted-on value still appears verbatim | No |
| Audit record | Policy, version, detector and counts for every document, with no student data in it | No |

When a model is loaded, the rules for NAME_STUDENT, ID_NUM and STREET_ADDRESS are switched off: they are recall-oriented fallbacks that over-flag ordinary essay text. Without a model the rules cover every label. `analyze_text(..., model_authority=())` runs both layers on everything.

**Loading a model.** Models load from disk only: a directory you pass, the `local_path` in `models.jsonl`, or the Hugging Face cache. Nothing is downloaded unless you set `EDSHIELD_ALLOW_DOWNLOAD=1`. If you name a model (`model_name=...` or `EDSHIELD_MODEL`) and it cannot be loaded, edshield raises `ModelUnavailableError` rather than quietly doing less. If you name none and the default is not installed, the rules run alone and a `RuntimeWarning` says so; pass `model_name="rules"` to choose that on purpose.

**Recall-first decoding.** `analyze_text(..., o_threshold=0.99)` marks a token as an entity whenever P(O) < 0.99 instead of taking the most likely class. It is off by default: on held-out PIILO essays it lowered precision from 0.69 to 0.57 with recall already at 1.00.

Label schema is the seven types of the [PIILO corpus](https://the-learning-agency-lab.com/learning-exchange/piilo-dataset/) (The Learning Agency Lab, CC BY 4.0), so models trained on it drop straight in.

## Install

```bash
pip install -e .                 # rules, policies, CLI. No torch.
pip install -e ".[hf]"           # + PyTorch model inference
pip install -e ".[service]"      # + REST service
pip install -e ".[train,onnx]"   # + training and ONNX/browser export
```

```bash
edshield redact essay.txt --policy ferpa
edshield extract essay.txt --model rules
edshield serve --port 8080       # POST /pii/extract, POST /pii/deidentify
```

## Demo

```bash
python -m http.server 8000 -d demo
# open http://localhost:8000
```

Three synthetic samples (essay, tutoring transcript, chatbot message), three policies, and a detector switch. "Rules only" runs entirely from the page's own JavaScript. "Rules + on-device model" loads an ONNX model from `demo/models/` through Transformers.js and runs it in the browser, so the same layer works on a Chromebook with no backend.

## Benchmarks

All numbers are span-level from `eval/evaluate.py`; F5 weights recall 5:1, as the PIILO competition did.

**Real student essays.** 680 PIILO documents held out from training of `piilo-deberta-v3-small` (DeBERTa-v3-small, 3 epochs), in the corpus's natural mix: 581 of them contain no PII at all.

| Detector | Precision | Recall | F5 | Missed entities |
|---|---|---|---|---|
| Rules only | 0.538 | 0.388 | 0.392 | 101 of 165 |
| Rules + model | 0.642 | 1.000 | 0.979 | 0 of 165 |

| Rules + model, by label | Precision | Recall | n |
|---|---|---|---|
| NAME_STUDENT | 0.656 | 1.000 | 143 |
| URL_PERSONAL | 0.381 | 1.000 | 8 |
| ID_NUM | 0.700 | 1.000 | 7 |
| EMAIL | 1.000 | 1.000 | 4 |
| USERNAME | 1.000 | 1.000 | 2 |
| STREET_ADDRESS | 0.500 | 1.000 | 1 |

The rare labels have a handful of examples each, so their rows say little. Most name false positives are names of people other than the essay's author (personas, lecturers, friends), which PIILO does not label but which a privacy tool should remove. Reports are in `eval/results/`.

**Synthetic K-12 writing.** PIILO is adult writing, so `eval/k12_bench.py` generates short essays, tutoring transcripts and chat messages in children's registers, covering every label. "Got through" counts identifiers that no flag of any label touched.

| Set | Detector | Identifiers | Got through |
|---|---|---|---|
| Cued: worded the way the rules expect | Rules + model | 1,445 | 0 |
| Hard: the way children type | Rules + model | 1,433 | 319 (22%) |
| Hard | Rules only | 1,433 | 1,100 (77%) |

The hard set is the honest baseline. What gets through is mostly lowercase schools and towns, ages in chat shorthand, spoken dates and streets without a house number. [docs/COVERAGE.md](docs/COVERAGE.md) has the breakdown and maps each identifier type in FERPA and COPPA to what edshield does.

**Synthetic transcripts and essays.** Rules only, 500 documents (`eval/synthetic_bench.py --n 500 --seed 1`). The generator's sentences use the same cues the rules look for, so read this as a regression check, not as expected accuracy on real text:

| Label | Precision | Recall | F5 |
|---|---|---|---|
| EMAIL | 1.000 | 1.000 | 1.000 |
| PHONE_NUM | 1.000 | 1.000 | 1.000 |
| USERNAME | 1.000 | 1.000 | 1.000 |
| URL_PERSONAL | 1.000 | 1.000 | 1.000 |
| ID_NUM | 0.926 | 1.000 | 0.997 |
| STREET_ADDRESS | 1.000 | 0.945 | 0.947 |
| NAME_STUDENT | 1.000 | 0.761 | 0.768 |
| **overall** | **0.994** | **0.903** | **0.906** |

Names are the gap, and the reason the model layer exists: rules only catch names the writer introduces, so a friend mentioned in passing is missed.

## Train a model

```bash
pip install kaggle
kaggle competitions download -c pii-detection-removal-from-educational-data
unzip pii-detection-removal-from-educational-data.zip -d data/piilo

python eval/synthetic_bench.py --n 2000 --out data/synthetic.json      # augmentation for rare labels
python training/prepare_piilo.py --input data/piilo/train.json --out data/piilo_hf --extra data/synthetic.json
python training/train.py --data data/piilo_hf --base microsoft/deberta-v3-small --out models/piilo-deberta-v3-small --epochs 3
python eval/evaluate.py --input data/piilo_hf/validation.json --model models/piilo-deberta-v3-small --device cuda
python training/export_onnx.py --model models/piilo-deberta-v3-small --out demo/models/piilo-deberta-v3-small-onnx
```

`train.py` uses the recall tricks that won the competition: down-weighted O class, a P(O) threshold instead of argmax at inference, long context with stride, synthetic augmentation. `models.jsonl` is the manifest; add a line per published model.

## Layout

```
edshield/         runtime: rules.py, ner.py, deid.py, policies/, cli.py, service.py
training/         prepare_piilo.py, train.py, export_onnx.py
eval/             evaluate.py (F5 + leak count), synthetic_bench.py, results/
demo/             single-file browser demo; drop exported models in demo/models/
tests/            pytest
docs/             COVERAGE.md: what is detected, how well, and what can be claimed
.github/          GitHub Actions workflow: tests and the rules benchmark on every push
models.jsonl      model manifest
```

## Roadmap

- [x] Rule layer with validators, name propagation, leak verifier
- [x] FERPA / COPPA / research policies
- [x] Browser demo, REST service, CLI
- [ ] First PIILO-trained encoders (DeBERTa-v3-small/base, ModernBERT-base) published to the Hub
- [ ] ONNX INT8 export and browser benchmark on a Chromebook
- [ ] Transcript models: teacher/tutor discourse moves (TalkMoves, NCTE), argumentative elements (PERSUADE)
- [ ] MLX backend and a Swift package
- [ ] MCP server and agent skills

## For research contributors

The corpus is public, the metric is defined, and the winning recipes are documented. That makes this a good place for a first applied-ML paper: fine-tune a model on PIILO, compare it with ChatGPT and Presidio on `eval/evaluate.py`, measure what leaks, and publish. Open an issue with the experiment you want to run; results that beat the current manifest entry get merged and credited in the model card.

## What this is not

Running edshield does not by itself make a product FERPA- or COPPA-compliant. It removes direct identifiers with measured recall; the institution still owns the reasonable-determination review, the consent process, and the data-handling agreement. Never paste real student data into a cloud-hosted agent to test this; use the synthetic samples.

## Credits

[The Learning Agency Lab](https://the-learning-agency-lab.com/) and Vanderbilt University for the PIILO corpus; [OpenMed](https://github.com/maziyarpanahi/openmed) for the pattern; Hugging Face `transformers` and Transformers.js; Faker.

## License

Apache-2.0. Model weights carry the license of their training data (PIILO is CC BY 4.0).
