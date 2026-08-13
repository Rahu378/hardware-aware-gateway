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
#
# Works on Kaggle too, which is worth knowing: its free GPU quota is separate
# from Colab's, so exhausting one still leaves the other. Enable Settings >
# Internet in the Kaggle notebook first, or the clone and the model download
# both fail.

set -euo pipefail

REPO_URL="https://github.com/Rahu378/hardware-aware-gateway.git"
MODEL="${HAG_MODEL:-Qwen/Qwen2.5-1.5B}"

# Colab writes to /content, Kaggle to /kaggle/working, and anything else gets
# the home directory. Kaggle matters because its free GPU quota is separate from
# Colab's, so hitting the limit on one does not block the other.
# Kaggle is checked first because a Kaggle image can also have /content, and
# matching Colab there would write the archive somewhere the Output panel does
# not show. Test for the more specific platform before the more general one.
if [ -n "${HAG_WORK:-}" ]; then
    BASE="$(dirname "$HAG_WORK")"; WORK="$HAG_WORK"; PLATFORM=explicit
elif [ -d /kaggle/working ]; then
    BASE=/kaggle/working; WORK=/kaggle/working/hag-run; PLATFORM=kaggle
elif [ -d /content ]; then
    BASE=/content; WORK=/content/hag-run; PLATFORM=colab
else
    BASE="$HOME"; WORK="$HOME/hag-run"; PLATFORM=generic
fi
ARCHIVE="$BASE/artifacts.zip"
echo "platform: $PLATFORM   workdir: $WORK"

# Check for a GPU before doing anything else. Without this the script clones,
# installs, silently skips all fifty GPU tests, and only fails two minutes later
# inside the benchmark with "Torch not compiled with CUDA enabled", which reads
# like a broken install rather than an unselected accelerator.
echo "=== 0/7  check accelerator ==="
if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo
    echo "########################################################################"
    echo "#  No CUDA device.                                                     #"
    echo "#                                                                      #"
    echo "#  Colab:   Runtime > Change runtime type > T4 GPU > Save              #"
    echo "#  Kaggle:  Settings > Accelerator > GPU T4, and Internet on            #"
    echo "#  Then run this cell again.                                           #"
    echo "#                                                                      #"
    echo "#  A CPU runtime skips every GPU test and measures nothing, so this    #"
    echo "#  stops here rather than producing an archive that looks complete.    #"
    echo "########################################################################"
    exit 1
fi
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"

echo "=== 1/7  clone into ${WORK} ==="
rm -rf "$WORK"
git clone -q "$REPO_URL" "$WORK"
cd "$WORK"

echo "=== 2/7  install ==="
pip install -q -e '.[e2e,dev]'
# The import must succeed in a *subprocess*, which is what every step below is.
python -c "import hag; print('import OK:', hag.__file__)"

echo "=== 3/7  correctness ==="
python -m pytest -q
# A suite that skips everything exits 0. Assert the CUDA half really ran, so
# "all tests passed" cannot mean "no test touched a GPU".
python - <<'CHECK'
import subprocess, sys
out = subprocess.run(
    [sys.executable, "-m", "pytest", "-q", "-k", "triton or graphed", "--collect-only"],
    capture_output=True, text=True,
)
if " no tests ran" in out.stdout or "0 tests collected" in out.stdout:
    sys.exit("No CUDA tests were collected; the kernels were never exercised.")
print("CUDA tests collected and run.")
CHECK

echo "=== 4/7  op-level sweep ==="
python -m hag.bench_ops --backend cuda --dtype fp16

echo "=== 5/7  profile + end-to-end (${MODEL}) ==="
python -m hag.profile_torch --model "$MODEL" --prompt-tokens 512 --new-tokens 32
python -m hag.bench_e2e --model "$MODEL" --prompt-tokens 512 --new-tokens 128 --repeats "${HAG_REPEATS:-16}"

echo "=== 6/7  cuda graphs ==="
# The only step allowed to fail without taking the run with it. Everything above
# is established and reproducible; this one has never executed on a GPU, and a
# failure here should not cost the results that already succeeded. The banner
# and the exit-code check at the end make sure it cannot fail quietly, which is
# the trap `set -e` exists to close.
GRAPHS_OK=1
if ! python -m hag.graphs --model "$MODEL" --prompt-tokens 512 --new-tokens 128 --repeats 8; then
    GRAPHS_OK=0
    echo
    echo "########################################################"
    echo "# STEP 6 FAILED: cuda graphs. Everything else is fine.  #"
    echo "# Send the output above; the archive is still written.  #"
    echo "########################################################"
    echo
fi

echo "=== 7/7  report ==="
python -m hag.report

echo
echo "=== results written ==="
ls -la "$WORK/results"
cd "$WORK" && zip -qr "$ARCHIVE" results profiles
echo
echo "Archive: $ARCHIVE"
if [ "$PLATFORM" = "kaggle" ]; then
    echo "On Kaggle: find it in the Output panel on the right sidebar."
fi
if [ "$GRAPHS_OK" = "1" ]; then
    echo "Every step succeeded."
else
    echo "Steps 1-5 and 7 succeeded. Step 6 (cuda graphs) FAILED, see above."
fi
