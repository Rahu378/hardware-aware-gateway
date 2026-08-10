#!/usr/bin/env bash
# One-shot Colab run: clone, install, verify, benchmark, report.
#
# Exists because driving this from notebook cells kept failing in ways that
# still looked like success. Two specific traps this closes:
#
#   * `set -e`. Notebook cells run independently, so a failing benchmark cell
#     does not stop the download cell that follows it. The first Colab run of
#     this repo raised ModuleNotFoundError four times and still offered an
#     archive at the end.
#   * Absolute paths. A repeated `%cd` in a re-run cell descends into a nested
#     clone, and a later relative `zip` then archives a different checkout than
#     the one that was just built.
#
# Usage inside Colab:
#   !bash scripts/colab_run.sh
# or from a clean runtime with nothing checked out yet:
#   !curl -sL https://raw.githubusercontent.com/Rahu378/hardware-aware-gateway/main/scripts/colab_run.sh | bash

set -euo pipefail

REPO_URL="https://github.com/Rahu378/hardware-aware-gateway.git"
WORK="${HAG_WORK:-/content/hag-run}"
MODEL="${HAG_MODEL:-Qwen/Qwen2.5-1.5B}"

echo "=== 1/6  clone into ${WORK} ==="
rm -rf "$WORK"
git clone -q "$REPO_URL" "$WORK"
cd "$WORK"

echo "=== 2/6  install ==="
pip install -q -e '.[e2e,dev]'
# The import must succeed in a *subprocess*, which is what every step below is.
python -c "import hag; print('import OK:', hag.__file__)"

echo "=== 3/6  correctness ==="
python -m pytest -q

echo "=== 4/6  op-level sweep ==="
python -m hag.bench_ops --backend cuda --dtype fp16

echo "=== 5/6  profile + end-to-end (${MODEL}) ==="
python -m hag.profile_torch --model "$MODEL" --prompt-tokens 512 --new-tokens 32
python -m hag.bench_e2e --model "$MODEL" --prompt-tokens 512 --new-tokens 128 --repeats "${HAG_REPEATS:-16}"

echo "=== 6/6  report ==="
python -m hag.report

echo
echo "=== results written ==="
ls -la "$WORK/results"
cd "$WORK" && zip -qr /content/artifacts.zip results profiles
echo
echo "Archive: /content/artifacts.zip"
echo "Every step above succeeded, or this script stopped before reaching here."
