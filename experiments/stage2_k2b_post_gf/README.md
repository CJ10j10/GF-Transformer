# K2b: GF prior, post-disaster observation

K2b tests a separate hypothesis from K2. At the three used scales, GF output is the prior. A 1×1 projection maps the post-disaster feature to the GF channel count. The update is

```text
observation = obs_proj(post_feat)
innovation = observation - global_feat
P = softplus(p_net(global_feat)) + 1e-6
R = softplus(r_net(abs(innovation))) + 1e-6
K = P / (P + R)
refined = global_feat + K * innovation
```

The canonical update acts from initialization, so K2b logits need not equal B0 logits at initialization. The three actual forward-path shapes at a 512×512 crop are:

| Insertion | Post feature | GF prior / refined feature |
| --- | --- | --- |
| GF1 → K2b1 → CSGF1 | 64×128×128 | 128×128×128 |
| GF2 → K2b2 → CSGF2 | 128×64×64 | 320×64×64 |
| GF3 → K2b3 → CSGF3 | 320×32×32 | 512×32×32 |

There is no change to GF, CSGF, loss, augmentation, sampling, or encoder initialization. The existing K2 `post-pre` path is retained unchanged. The fourth GF block is unused and has no Kalman layer.

The initial K2b run uses preregistered training seed **3** to pair with the frozen B0-D2 seed-3 checkpoint. B0/K2 seed-23 runs are paused. The fixed validation split still uses seed 3, with 917 images and the frozen localization masks. Best checkpoint selection uses the highest single-view F1s at epochs 1, 3, ..., 49; minor-damage F1 is reported separately. Later K2b seeds 11 and 23 remain in the existing seed plan and require their own preflights before training.

Formal protocol: 2×RTX4090, per-GPU batch 2, accumulation 1, global batch 4, FP32, 50 epochs, AdamW lr 2e-4 weight decay 1e-6, MultiStepLR milestones [3,9] gamma 0.5, crop 512, validation batch 1. Checkpoints and compact validation history are isolated under `runs/seed3/`. Training refuses to resume or overwrite these files. The preflight runs only 20 real optimizer updates without saving a checkpoint.

## Manual launch after preflight passes

```bash
tmux new -s stage2_k2b_seed3
cd /workspace/GF-Transformer
bash experiments/stage2_k2b_post_gf/run_k2b.sh
```

Detach with `Ctrl-b d`; reconnect with `tmux attach -t stage2_k2b_seed3`. In another shell, inspect the latest log and compact validation history:

```bash
cd /workspace/GF-Transformer
RUN=experiments/stage2_k2b_post_gf/runs/seed3
LOG=$(cat "$RUN/current_log.txt")
tail -c 300000 "$LOG" | tr '\r' '\n' | grep -E 'optimizer_updates|Val Score:|score_best:|Matched-seed done' | tail -n 30
cat "$RUN/results/validation_history.csv" | tail -n 5
```

After training, independently re-evaluate the best checkpoint with:

```bash
CUDA_VISIBLE_DEVICES=0 /usr/local/miniconda3/envs/gft/bin/python \
  experiments/stage2_matched_seeds/eval_matched_seed.py --variant k2b --seed 3
```

Do not commit raw tqdm logs or large checkpoints; commit compact CSV/JSON results after independent evaluation.
