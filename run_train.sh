#!/bin/bash
# ===========================================================================
# GF-Transformer Training — Launcher for 4× RTX 4090 DDP
# ===========================================================================
# Usage:
#   bash run_train.sh              # Run both stages
#   bash run_train.sh loc           # Run Stage 1 only
#   bash run_train.sh cls           # Run Stage 2 only
# ===========================================================================

set -euo pipefail

cd "$(dirname "$0")"
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate gft

# ── Config ─────────────────────────────────────────────────────────────
NUM_GPUS=4
MASTER_PORT=${MASTER_PORT:-29500}
OMP_NUM_THREADS=4

export OMP_NUM_THREADS MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

STAGE=${1:-all}

# ── Stage 1: Building Localization ─────────────────────────────────────
if [[ "$STAGE" == "all" ]] || [[ "$STAGE" == "loc" ]]; then
    echo "============================================================================"
    echo " STAGE 1: Building Localization (GFformer_one)"
    echo " 150 epochs | per-GPU BS=12 | eff BS=48 | ~2.7h expected"
    echo "============================================================================"
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
        train_segformer_loc.py 2>&1 | tee -a logs/stage1_loc.log
    echo "Stage 1 done."
fi

# ── Stage 2: Damage Classification ─────────────────────────────────────
if [[ "$STAGE" == "all" ]] || [[ "$STAGE" == "cls" ]]; then
    echo "============================================================================"
    echo " STAGE 2: Damage Classification (GFformer_two)"
    echo " 30 epochs | per-GPU BS=6 | eff BS=24 | ~1.9h expected"
    echo "============================================================================"
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
        train_segformer_cls.py 2>&1 | tee -a logs/stage2_cls.log
    echo "Stage 2 done."
fi

echo "============================================================================"
echo " TRAINING COMPLETE"
echo " Checkpoints: tune_weight/"
echo " Logs:        logs/"
echo " TensorBoard: runs/"
echo "============================================================================"
