#!/usr/bin/env bash
# Safe short smoke only; does not start formal epochs or save checkpoints.
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo "Usage: $0 {b0|k2} {11|23}" >&2
    exit 2
fi
VARIANT=$1
RUN_SEED=$2
case "$VARIANT:$RUN_SEED" in
    b0:11) MASTER_PORT=29541 ;;
    k2:11) MASTER_PORT=29542 ;;
    b0:23) MASTER_PORT=29543 ;;
    k2:23) MASTER_PORT=29544 ;;
    *) echo 'Only the preregistered B0/K2 seeds 11 and 23 are allowed' >&2; exit 2 ;;
esac
cd "$(dirname "$0")/../.."
mkdir -p logs
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
LOG="logs/stage2_${VARIANT}_ddp2_seed${RUN_SEED}_preflight_$(date -u +%Y%m%d_%H%M%S).log"
echo "Preflight logging to $LOG"
/usr/local/miniconda3/envs/gft/bin/python -m torch.distributed.run \
    --nproc_per_node=2 --master_port="$MASTER_PORT" --max_restarts=0 \
    experiments/stage2_matched_seeds/preflight_matched_seed.py \
    --variant "$VARIANT" --seed "$RUN_SEED" 2>&1 | tee "$LOG"
