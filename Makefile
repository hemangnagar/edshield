.PHONY: install test bench demo
install: ; pip install -e ".[dev]"
test:    ; pytest -q
bench:   ; python eval/synthetic_bench.py --n 500 --out data/synthetic.json --seed 1 && python eval/evaluate.py --input data/synthetic.json --model rules --report eval/results/rules_synthetic.json
demo:    ; python -m http.server 8000 -d demo
