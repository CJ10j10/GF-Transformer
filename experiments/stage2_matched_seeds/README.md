# Preregistered matched-seed screening

This branch prepares four **manual** 2 × RTX 4090 training runs. No 50-epoch run is launched during preparation. The B0-D2 seed-3 result is frozen at tag `b0-ddp2-frozen`; its provenance is in `../stage2_b0_ddp2/results/freeze_manifest.json`.

## Seeds and comparisons

`seed_plan.json` preregisters `[3, 11, 23]` for B0, K1, K2, K2b, and K3. Seed 3 B0 and K2 have already completed; this screening adds B0 and K2 at seeds 11 and 23. The same seed value is used for PyTorch, NumPy, Python random, and the DDP sampler, with rank offset for process RNG. The **train/validation split remains fixed at seed 3** and has SHA-256 `7e3c7e8cf1762512697055f6a7bfd5d51050a6d03fc4d93e0d86e1ffaa2c480e`. Changing the training seed must never change the 917 validation image IDs or masks.

At each seed, the primary comparison is K2 minus B0 best-checkpoint single-view F1s. The best checkpoint is selected by the highest unrounded F1s among the 25 checks at epochs 1, 3, ..., 49. Report all three seed-pair differences, including seed 3, without dropping unfavorable runs. F1_1 (minor damage) is a prespecified secondary endpoint; checkpoint selection does not use it. K2b remains an independent hypothesis regardless of K2 screening outcome.

## Frozen training protocol

Each run uses the existing GF/CSGF, loss, augmentation, oversampling, and Stage1 encoder checkpoint. Only B0 versus K2 model selection and the preregistered run seed differ. Per GPU batch 2, two GPUs, accumulation 1, global batch 4, 50 epochs, 3351 updates per epoch, AdamW lr 2e-4 and weight decay 1e-6, MultiStepLR milestones [3,9] gamma 0.5, crop 512 × 512, FP32, validation batch 1, one original view. Rank 0 evaluates on the same 917 masks and writes one compact CSV row per validation. Every run has its own `runs/<variant>_seed<seed>/ckpt/`, `audit/`, and `results/`; no resume or legacy checkpoint read/write is allowed.

The preflights perform 20 real optimizer updates on two GPUs and one validation sample per run, without saving a checkpoint. Reports under each `audit/preflight.json` must have `ready_for_training: true` before the formal launcher accepts a run.

## Manual launch with tmux

Each command below occupies both GPUs. Run them **one at a time**. To start the first run:

```bash
tmux new -s stage2_seed_screening
cd /workspace/GF-Transformer
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate gft
bash experiments/stage2_matched_seeds/run_matched_seed.sh b0 11
```

After it finishes, use the same tmux session for the remaining runs:

```bash
bash experiments/stage2_matched_seeds/run_matched_seed.sh k2 11
bash experiments/stage2_matched_seeds/run_matched_seed.sh b0 23
bash experiments/stage2_matched_seeds/run_matched_seed.sh k2 23
```

Detach with `Ctrl-b d`; reconnect with `tmux attach -t stage2_seed_screening`. The launcher records a timestamped `tee` log path in each run's `current_log.txt`. To inspect a run from another shell:

```bash
cd /workspace/GF-Transformer
RUN=experiments/stage2_matched_seeds/runs/b0_seed11
LOG=$(cat "$RUN/current_log.txt")
tail -c 300000 "$LOG" | tr '\r' '\n' | grep -E 'optimizer_updates|Val Score:|score_best:|Matched-seed done' | tail -n 30
cat "$RUN/results/validation_history.csv" | tail -n 3
```

An individual run should take roughly 14–15 hours based on completed B0-D2/K2-D2 runs; all four sequential runs could take about 58–60 hours. These are estimates, not a schedule or an instruction to start them automatically. Raw tqdm logs and large checkpoints stay local; compact validation histories and result JSON can be committed after training.

## After the four runs

Independently re-evaluate each best checkpoint with `eval_matched_seed.py --variant <b0|k2> --seed <11|23>` (single GPU, read-only for checkpoints). After all four JSON results exist, run `summarize_screening.py`. It writes a compact per-seed CSV and JSON with the explicitly defined **K2 minus B0** deltas; the earlier seed-3 paired bootstrap JSON uses **B0 minus K2**, so read each result's `delta_definition`.
