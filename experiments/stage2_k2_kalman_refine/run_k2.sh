#!/usr/bin/env bash
# Manual K2 launch only. No resume, and no Baseline-B or smoke checkpoint writes.
set -euo pipefail
cd "$(dirname "$0")/../.."
EXP=experiments/stage2_k2_kalman_refine
CKPT_DIR="$EXP/ckpt"
mkdir -p "$CKPT_DIR" logs
if [[ -n "$(find "$CKPT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing to start: $CKPT_DIR is not empty" >&2
    exit 1
fi
/usr/local/miniconda3/envs/gft/bin/python - "$EXP/audit/preflight.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding='utf-8') as f:
    report = json.load(f)
if not report.get('ready_for_training'):
    raise SystemExit('K2 preflight did not pass')
PY
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
LOG="logs/stage2_k2_$(date -u +%Y%m%d_%H%M%S).log"
printf '%s\n' "$LOG" > "$EXP/current_k2_log.txt"
echo "Logging to $LOG"
/usr/local/miniconda3/envs/gft/bin/python -m torch.distributed.run \
    --nproc_per_node=1 --master_port=29518 --max_restarts=0 \
    "$EXP/train_k2.py" 2>&1 | tee "$LOG"
