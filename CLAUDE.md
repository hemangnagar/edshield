# edshield

Local-first student-privacy layer: detects and removes student PII from essays, tutoring transcripts and chat before text reaches any LLM. Public pet project (github.com/hemangnagar/edshield, Apache-2.0), modelled on OpenMed. Near-term goal: a working offline demo to show an edtech contact.

State saved 2026-09-29. Update the "Current state" and "Open decisions" sections when they change.

## Rules for working here

- Never put real student data in the repo or in prompts. Synthetic samples only.
- Never write "FERPA compliant" or "COPPA compliant" claims. `docs/COVERAGE.md` has the wording the evidence supports. Compliance belongs to the deploying organisation; the truthful route to a claim is third-party certification.
- Recall matters more than precision. A change that causes a leak is worse than one that adds false alarms. Re-run the held-out evaluation after touching rules or decoding.
- Python rules (`edshield/rules.py`) and the demo's JavaScript (`demo/index.html`) must stay in parity. `tests/test_demo_parity.py` enforces it and needs Node.
- The demo must run offline apart from one CDN script tag.
- Do not commit `data/piilo*`, `models/`, `demo/models/*` or `*.onnx` (gitignored).
- Do not tune rules to `eval/k12_bench.py`'s hard set; that makes the test circular.
- Verify before explaining a number. Report when long jobs finish.

## Current state

- Pull requests 1 and 2 were merged into `main` on 2026-09-29. `main` has everything through the browser model fix (merge commit `1079fcb`).
- New work goes on branch `worktree-model-authority` and reaches `main` through a new pull request, which the user creates and merges on GitHub.
- Worktree: `C:\edshield\.claude\worktrees\model-authority`. Main checkout: `C:\edshield`.
- 183 tests pass locally. CI (`.github/workflows/ci.yml`) is green on Python 3.10 and 3.12.
- The main checkout still has staged changes (`data/synthetic.json`, `eval/results/deberta_small_piilo.json`, `training/train.py`). All three are now in the branch; discard them there before pulling `main`.

## What is where

| Path | What |
|---|---|
| `edshield/rules.py` | Regex detectors, name propagation, overlap resolution |
| `edshield/ner.py` | Model inference: overlapping windows, `decode()`, offline loading |
| `edshield/__init__.py` | `analyze_text`, `extract_pii`, `deidentify`; model authority |
| `edshield/deid.py` | Masking, surrogates, date shift, leak verifier, audit record |
| `edshield/policies/*.yaml` | `ferpa`, `coppa`, `research` |
| `edshield/service.py`, `cli.py` | FastAPI service, command line |
| `training/` | `prepare_piilo.py`, `train.py`, `export_onnx.py` |
| `eval/` | `evaluate.py`, `evaluate_onnx.py` (scores an exported file), `synthetic_bench.py`, `k12_bench.py`, `results/` |
| `demo/index.html` | Single-file browser demo with a JS port of the rules and decoding |
| `docs/COVERAGE.md` | FERPA and COPPA identifier types mapped to coverage, with results |
| `models.jsonl` | Model manifest |

Files outside git, in the main checkout:

| Path | What |
|---|---|
| `C:\edshield\models\piilo-deberta-v3-small-v2` | Current model, trained on the corrected split |
| `C:\edshield\models\piilo-deberta-v3-small` | Original model; its validation numbers were inflated |
| `C:\edshield\demo\models\piilo-deberta-v3-small-onnx` | ONNX export (fp32 566 MB, INT8 205 MB); the worktree links to it. `model_quantized.all-layers.bak` is the earlier 172 MB file |
| `C:\edshield\data\piilo\train.json` | PIILO corpus, 6,807 documents |
| `C:\edshield\data\piilo_hf_v2` | Current split: train 4,430, validation 680 real documents, plus `validation.json` |

## Design decisions already made

- **Model authority.** When a model ran, `NAME_STUDENT`, `ID_NUM` and `STREET_ADDRESS` come from the model only and the rules are not run for them. Without a model the rules cover everything.
- **Labels.** Seven PIILO labels, plus rule-only `SSN`, `DATE`, `NAME_RELATED`, `SCHOOL`, `LOCATION`, `AGE`, `IP_ADDRESS`, `DEVICE_ID`, `GEO`.
- **Other people's names are flagged.** PIILO labels only the author, so this lowers measured precision. It is deliberate.
- **Propagation.** A name is spread through the document only if the model's confidence is 0.75 or more.
- **URLs.** A bare site name is personal only if the writer claims it or it has just been called a website, blog, page, profile or channel.
- **Model failure is loud.** A named model that cannot load raises `ModelUnavailableError`. With none named, rules run with a `RuntimeWarning`.
- **No downloads** unless `EDSHIELD_ALLOW_DOWNLOAD=1`.
- **`o_threshold`** works but is off by default: it lowered precision with no recall gain.
- **Browser model.** The INT8 export leaves the first two encoder layers at full precision (`--keep-layers 2`). Quantizing them caused the misses; the embedding table quantizes without loss. The setting was chosen on the PIILO validation set.
- **Validation** is real documents only, in their natural mix. Synthetic documents go to training only.

## Results

Span-level, from `eval/evaluate.py`. "Got through" means no flag of any label touched the identifier.

| Test set | Detector | Precision | Recall | Got through |
|---|---|---|---|---|
| PIILO held-out, 680 essays | Rules + model | 0.642 | 1.000 | 0 of 165 |
| PIILO held-out | Rules + INT8 model (browser), 205 MB | 0.639 | 1.000 | 0 of 165 |
| PIILO held-out | Rules + INT8 model, every layer quantized, 172 MB | 0.692 | 0.952 | 7 of 165 |
| PIILO held-out | Rules only | 0.538 | 0.388 | 101 of 165 |
| K-12 synthetic, cued | Rules + model | | | 0 of 1,445 |
| K-12 synthetic, hard | Rules + model | | | 319 of 1,433 (22%) |
| K-12 synthetic, hard | Rules + INT8 model (browser), 205 MB | | | 324 of 1,433 (23%) |
| K-12 synthetic, hard | Rules only | | | 1,100 of 1,433 (77%) |

Caveats: PIILO is adult writing. The K-12 sets are synthetic. The PIILO false-alarm fixes were chosen by reading false alarms on the same validation set.

## Open decisions

1. Retrain with child-style synthetic text, worded differently from the test set, to close the 22% gap on the hard set. About 42 minutes. Afterwards re-export the browser file and score it with `eval/evaluate_onnx.py`.
2. Publish to the Hugging Face Hub. On hold. Steps: account, `edshield` organisation, write token, `hf auth login`, `hf upload`, model card with CC BY 4.0 and attribution.

Known small issues: the model labels `priyawrites.wordpress.com` as `EMAIL` (still removed); "I'll be 15 soon" is not caught as an age.

## Environment

Windows 10, GTX 1060 6 GB, 7.9 GB RAM, C: about 98% full. Python 3.11 venv at `C:\edshield\.venv`.

- torch must stay on `2.14.0+cu126`. Later CUDA builds have no kernels for this card.
- fp32 only; no fp16 on this card.
- transformers 5.x: no `warmup_ratio`; `from_pretrained` needs `dtype=torch.float32`.
- optimum 2.x has no ONNX exporter; `export_onnx.py` uses `torch.onnx.export`.
- Git has no identity configured. Commit with `git -c user.name="Hemang Nagar" -c user.email="hi@hemangnagar.dev" commit ...`.
- `gh` is not installed. CI status: `https://api.github.com/repos/hemangnagar/edshield/actions/runs`.
- Headless Edge is at `C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe`.
- Before training, close Chrome and other apps. Background jobs are stopped when memory runs critically low.
- Scripts in `eval/` and `training/` import `edshield` from the editable install, which points at `C:\edshield`. From the worktree, put the worktree on `sys.path` first or the old code runs.

## Commands

```bash
# tests
C:\edshield\.venv\Scripts\python.exe -m pytest -q

# held-out evaluation with the model, on the GPU
python eval/evaluate.py --input C:/edshield/data/piilo_hf_v2/validation.json --model C:/edshield/models/piilo-deberta-v3-small-v2 --device cuda

# held-out evaluation of the browser file, on the CPU (about 6 minutes)
python eval/evaluate_onnx.py --onnx C:/edshield/demo/models/piilo-deberta-v3-small-onnx/onnx/model_quantized.onnx --input C:/edshield/data/piilo_hf_v2/validation.json

# synthetic K-12 sets
python eval/k12_bench.py --n 400 --style hard --seed 1 --out data/k12_hard.json
python eval/evaluate.py --input data/k12_hard.json --model rules --labels all

# rebuild the split, retrain, export
python training/prepare_piilo.py --input data/piilo/train.json --extra data/synthetic.json --out data/piilo_hf_v2
python training/train.py --data data/piilo_hf_v2 --base microsoft/deberta-v3-small --out models/piilo-deberta-v3-small-v2 --epochs 3 --bs 2
python training/export_onnx.py --model models/piilo-deberta-v3-small-v2 --out demo/models/piilo-deberta-v3-small-onnx

# demo: open http://localhost:8000, Detector = "Rules + on-device model"
python -m http.server 8000 -d demo
```
