#!/usr/bin/env bash
# Explicit Baseline-B launch command. Run only when formal training is authorized.
set -euo pipefail
cd "$(dirname "$0")/../.."
CKPT_DIR=experiments/stage2_fixdata/ckpt_repo_bs4
mkdir -p "$CKPT_DIR" logs
if [[ -n "$(find "$CKPT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing to start: $CKPT_DIR is not empty" >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
LOG="logs/stage2_repo_bs4_$(date -u +%Y%m%d_%H%M%S).log"
echo "Logging to $LOG"
/usr/local/miniconda3/envs/gft/bin/python -m torch.distributed.run \
    --nproc_per_node=1 --master_port=29517 --max_restarts=0 \
    train_segformer_cls.py 2>&1 | tee "$LOG"
