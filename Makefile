# Development loop. Correctness runs before timing, always.

PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: setup test bench bench-e2e report report-check profile profile-nsys clean

setup:                ## create the venv and install everything
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e '.[e2e,dev]' 

test:                 ## numerical correctness; skips absent backends
	$(PY) -m pytest

bench: test           ## op-level sweep -> results/ops_*.json
	$(PY) -m hag.bench_ops --dtype fp16

bench-e2e:            ## tokens/sec + memory -> results/e2e_*.json
	$(PY) -m hag.bench_e2e --model $(or $(MODEL),Qwen/Qwen2.5-0.5B)

report:               ## regenerate the README tables from results/
	$(PY) -m hag.report

report-check:         ## fail if the README has drifted from results/
	$(PY) -m hag.report --check

profile:              ## per-kernel profile; needs no apt package
	$(PY) -m hag.profile_torch --model $(or $(MODEL),Qwen/Qwen2.5-0.5B)

profile-nsys:         ## Nsight Systems timeline (CUDA, needs nsys installed)
	./scripts/profile_nsys.sh $(or $(MODEL),Qwen/Qwen2.5-1.5B)

clean:
	rm -rf .pytest_cache **/__pycache__ profiles
