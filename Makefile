.PHONY: install test bench demo
install: ; pip install -e ".[dev]"
test:    ; pytest -q
# data/synthetic.json is the 2000-document training augmentation; the benchmark writes its own file.
bench:   ; python eval/synthetic_bench.py --n 500 --out data/synthetic_bench.json --seed 1 && python eval/evaluate.py --input data/synthetic_bench.json --model rules --report eval/results/rules_synthetic.json
demo:    ; python -m http.server 8000 -d demo
