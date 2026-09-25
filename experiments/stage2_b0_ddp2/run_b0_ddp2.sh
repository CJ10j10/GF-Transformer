#!/usr/bin/env bash
# Manual matched B0-D2 control launch; no resume or legacy checkpoint writes.
set -euo pipefail
cd "$(dirname "$0")/../.."
EXP=experiments/stage2_b0_ddp2
CKPT_DIR="$EXP/ckpt"
mkdir -p "$CKPT_DIR" "$EXP/results" logs
if [[ -n "$(find "$CKPT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing to start: $CKPT_DIR is not empty" >&2
    exit 1
fi
if [[ -e "$EXP/results/validation_history.csv" ]]; then
    echo "Refusing to start: validation history already exists" >&2
    exit 1
fi
/usr/local/miniconda3/envs/gft/bin/python - "$EXP/audit/preflight.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding='utf-8') as f:
    report = json.load(f)
if not report.get('ready_for_training') or report.get('world_size') != 2:
    raise SystemExit('B0-D2 preflight did not pass')
PY
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
LOG="logs/stage2_b0_ddp2_$(date -u +%Y%m%d_%H%M%S).log"
printf '%s\n' "$LOG" > "$EXP/current_log.txt"
echo "Logging to $LOG"
/usr/local/miniconda3/envs/gft/bin/python -m torch.distributed.run \
    --nproc_per_node=2 --master_port=29521 --max_restarts=0 \
    "$EXP/train_b0_ddp2.py" 2>&1 | tee "$LOG"
