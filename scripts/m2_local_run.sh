#!/usr/bin/env bash
# M2.1 + M2.2 experiments on the local CPU machine (Fedora), in a bounded order.
#
#   conda activate qwen3-scratch
#   pip install -e ".[dev,hf,notebooks]"      # once: transformers etc. (reference only)
#   bash scripts/m2_local_run.sh               # all steps
#   bash scripts/m2_local_run.sh benchmark     # one step: hardware|download|tests|parity|run|benchmark
#   QWEN3_CKPT=/path/to/dir bash scripts/m2_local_run.sh parity   # use an existing directory
#
# Everything is written to results/m2/ (logs in results/m2/logs/). Close the browser
# and other large programs first: the float32 parity phases need ~4 GiB peak RSS
# each (one model at a time). Checkpoint download (~1.2 GB) goes to the normal
# Hugging Face cache (~/.cache/huggingface), never into the repository.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=results/m2
LOG=$OUT/logs
mkdir -p "$LOG"
THREADS=${THREADS:-2}            # Ryzen 3 3250U: 2 physical cores (4 SMT threads)
STEPS=${1:-all}
STATUS=$OUT/steps.txt
: > "$STATUS.new"

ckpt_dir() {
  python - <<'PY'
from qwen3_scratch.pretrained import download_checkpoint
print(download_checkpoint())
PY
}

step() {  # step <name> <command...>
  local name=$1; shift
  if [[ $STEPS != all && $STEPS != "$name" ]]; then return 0; fi
  echo "=== [$name] $*" | tee "$LOG/$name.log"
  local t0=$SECONDS
  "$@" 2>&1 | tee -a "$LOG/$name.log"
  local code=${PIPESTATUS[0]}
  local verdict=PASSED; [[ $code -ne 0 ]] && verdict="FAILED (exit $code)"
  echo "$name: $verdict in $((SECONDS - t0))s" | tee -a "$STATUS.new"
}

step hardware python scripts/hardware_report.py
step download bash -c "python - <<'PY'
import hashlib, json, pathlib
from qwen3_scratch.pretrained import download_checkpoint, QWEN3_0_6B_BASE_REVISION
d = download_checkpoint()
files = {p.name: p.stat().st_size for p in sorted(d.iterdir())}
h = hashlib.sha256()
with open(d / 'model.safetensors', 'rb') as f:
    for chunk in iter(lambda: f.read(1 << 24), b''):
        h.update(chunk)
info = {'dir_name': d.name, 'revision': QWEN3_0_6B_BASE_REVISION, 'files': files,
        'model_safetensors_sha256': h.hexdigest()}
pathlib.Path('results/m2/checkpoint.json').write_text(json.dumps(info, indent=2) + '\n')
print(json.dumps(info, indent=2))
PY"
CKPT=${QWEN3_CKPT:-$(ckpt_dir 2>/dev/null | tail -1)}   # QWEN3_CKPT: use a local dir instead
export QWEN3_WEIGHTS_DIR=$CKPT
step tests python -m pytest -q -m "weights and not slow" -p no:cacheprovider
step parity python scripts/verify_pretrained.py --model-dir "$CKPT" --dtype float32 \
  --threads "$THREADS" --max-new-tokens 16 --uncached-tokens 8
step run bash -c "
python scripts/run_pretrained_qwen3.py --model-dir '$CKPT' --threads $THREADS \
  --prompt 'Artificial intelligence is' --max-new-tokens 32 &&
python scripts/run_pretrained_qwen3.py --model-dir '$CKPT' --threads $THREADS \
  --prompt 'Hà Nội là thủ đô của' --max-new-tokens 32 &&
python scripts/run_pretrained_qwen3.py --model-dir '$CKPT' --threads $THREADS \
  --prompt 'The three primary colors are' --max-new-tokens 32 --do-sample \
  --temperature 0.7 --top-k 50 --top-p 0.9 --seed 0"
step benchmark python scripts/benchmark_generation.py --model-dir "$CKPT" --dtype float32 \
  --threads "$THREADS" --prompt-lens 16 64 128 256 --new-tokens 16 32 64 --repeats 2 \
  --uncached-max-prompt 128 --max-case-seconds 240 --tag local_fp32 --plot

cat "$STATUS.new" >> "$STATUS"; rm -f "$STATUS.new"
echo; echo "summary:"; cat "$STATUS"
