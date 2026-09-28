#!/usr/bin/env bash
# Manual formal launch of one preregistered B0/K2 seed. Never resumes.
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo "Usage: $0 {b0|k2} {11|23}" >&2
    exit 2
fi
VARIANT=$1
RUN_SEED=$2
case "$VARIANT:$RUN_SEED" in
    b0:11) MASTER_PORT=29531 ;;
    k2:11) MASTER_PORT=29532 ;;
    b0:23) MASTER_PORT=29533 ;;
    k2:23) MASTER_PORT=29534 ;;
    *) echo 'Only the preregistered B0/K2 seeds 11 and 23 are allowed' >&2; exit 2 ;;
esac
cd "$(dirname "$0")/../.."
RUN="experiments/stage2_matched_seeds/runs/${VARIANT}_seed${RUN_SEED}"
mkdir -p "$RUN/ckpt" "$RUN/results" logs
if [[ -n "$(find "$RUN/ckpt" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing to start: $RUN/ckpt is not empty" >&2
    exit 1
fi
if [[ -e "$RUN/results/validation_history.csv" ]]; then
    echo "Refusing to start: validation history exists in $RUN" >&2
    exit 1
fi
/usr/local/miniconda3/envs/gft/bin/python - "$RUN/audit/preflight.json" "$VARIANT" "$RUN_SEED" <<'PY'
import json, sys
with open(sys.argv[1], encoding='utf-8') as handle:
    report = json.load(handle)
if not (report.get('ready_for_training') and report.get('variant') == sys.argv[2]
        and report.get('train_seed') == int(sys.argv[3])
        and report.get('world_size') == 2):
    raise SystemExit('Matching two-GPU preflight has not passed')
PY
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
LOG="logs/stage2_${VARIANT}_ddp2_seed${RUN_SEED}_$(date -u +%Y%m%d_%H%M%S).log"
printf '%s\n' "$LOG" > "$RUN/current_log.txt"
echo "Logging to $LOG"
/usr/local/miniconda3/envs/gft/bin/python -m torch.distributed.run \
    --nproc_per_node=2 --master_port="$MASTER_PORT" --max_restarts=0 \
    experiments/stage2_matched_seeds/train_matched_seed.py \
    --variant "$VARIANT" --seed "$RUN_SEED" 2>&1 | tee "$LOG"
