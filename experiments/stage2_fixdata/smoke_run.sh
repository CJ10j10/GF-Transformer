#!/bin/bash
# Gate 3: Stage-2 smoke test — runs until the FIRST validation completes,
# then stops the training. Single GPU (torchrun nproc=1).
# If the runner dies before validation, prints the log tail and exits 1.
set -u
cd /workspace/GF-Transformer

LOG=logs/stage2_smoke_fix.log
CKPT_DIR=experiments/stage2_fixdata/ckpt
: > "$LOG"

# torch 1.9 has no torchrun binary — use python -m torch.distributed.run
# --max_restarts=0: fail fast instead of elastic-agent auto-retrying on OOM
setsid /usr/local/miniconda3/envs/gft/bin/python -m torch.distributed.run \
    --nproc_per_node=1 --master_port=29507 --max_restarts=0 \
    train_segformer_cls.py > "$LOG" 2>&1 &
PGID=$!

while ! grep -aq "Val Score" "$LOG"; do
    sleep 15
    if ! kill -0 "$PGID" 2>/dev/null; then
        echo "RUNNER EXITED BEFORE FIRST VALIDATION"
        tail -40 "$LOG"
        exit 1
    fi
done

sleep 3
kill -TERM -- "-$PGID" 2>/dev/null
sleep 15
kill -KILL -- "-$PGID" 2>/dev/null

# Keep the smoke checkpoint out of the real checkpoint dir.
mkdir -p experiments/stage2_fixdata/ckpt_smoke
mv -f "$CKPT_DIR"/* experiments/stage2_fixdata/ckpt_smoke/ 2>/dev/null

echo "SMOKE STOPPED AFTER FIRST VALIDATION"
echo "--- first validation ---"
grep -a "Val Score" "$LOG" | head -3
echo "--- epoch 0 train line ---"
grep -a "epoch: 0; lr" "$LOG" | head -2
