#!/usr/bin/env bash
# Manual formal K2b seed-3 launch. Refuses resume and requires passing preflight.
set -euo pipefail
cd "$(dirname "$0")/../.."
RUN=experiments/stage2_k2b_post_gf/runs/seed3
mkdir -p "$RUN/ckpt" "$RUN/results" logs
if [[ -n "$(find "$RUN/ckpt" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing to start: $RUN/ckpt is not empty" >&2
    exit 1
fi
if [[ -e "$RUN/results/validation_history.csv" ]]; then
    echo "Refusing to start: validation history already exists" >&2
    exit 1
fi
/usr/local/miniconda3/envs/gft/bin/python - "$RUN/audit/preflight.json" <<'PY'
import hashlib, json, pathlib, sys
with open(sys.argv[1], encoding='utf-8') as handle:
    report = json.load(handle)
if not (report.get('ready_for_training') and report.get('variant') == 'k2b'
        and report.get('train_seed') == 3 and report.get('world_size') == 2
        and report.get('per_gpu_batch') == 2 and report.get('grad_accum') == 1):
    raise SystemExit('Matching K2b seed-3 two-GPU preflight has not passed')
for name, expected in report['code_sha256'].items():
    actual = hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f'Code changed after K2b preflight: {name}')
PY
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
LOG="logs/stage2_k2b_ddp2_seed3_$(date -u +%Y%m%d_%H%M%S).log"
printf '%s\n' "$LOG" > "$RUN/current_log.txt"
echo "Logging to $LOG"
/usr/local/miniconda3/envs/gft/bin/python -m torch.distributed.run \
    --nproc_per_node=2 --master_port=29536 --max_restarts=0 \
    experiments/stage2_matched_seeds/train_matched_seed.py \
    --variant k2b --seed 3 2>&1 | tee "$LOG"
