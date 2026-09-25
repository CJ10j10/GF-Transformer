# B0-D2 matched control: training handoff

Status: prepared for manual launch only. The formal 50-epoch job has **not** been started.

## Comparison and protocol

This branch starts from the completed K2-D2 code so the control uses the same Stage 2 training and validation implementation. `train_b0_ddp2.py` changes the model construction to `GFformer_two(use_kalman=False)`: it contains no Kalman layers or Kalman parameters. This is the original GF + CSGF path and is strictly compatible with the frozen Baseline-B checkpoint. The B0-D2 run starts from the **same frozen Stage 1 encoder checkpoint** as K2-D2, never from the trained Baseline-B or K2 checkpoint.

| Setting | B0-D2 |
| --- | --- |
| GPUs | 2 × RTX 4090 24 GiB, DDP |
| Batch | 2 images per GPU, accumulation 1, global batch 4 |
| Training | 50 epochs, 3351 optimizer updates per epoch |
| Optimizer | AdamW, learning rate 2e-4, weight decay 1e-6, one parameter group |
| Scheduler | MultiStepLR milestones [3, 9], gamma 0.5, step once per epoch |
| Precision/crop | FP32, 512 × 512 |
| Data | original 13,407-entry oversampled train index list, same augmentation/loss |
| Validation | fixed 917 images, one original view, rank 0, batch 1; original `train_segformer_cls.validate` metric and verified localization masks |
| Initial weights | `experiments/stage1_fixdata_eval/ckpt/GFformer_loc_fixdata_ep57_valdice0.8830_sha_cd940989.pt` encoder transfer |
| Checkpoints | `experiments/stage2_b0_ddp2/ckpt/` only; no resume |

The two GPU run has the same global batch and update count as K2-D2. It is a separate training run, so numerical equality with single GPU Baseline-B is not expected. The preflight checks the model's baseline parameter set, K2 shared-parameter initialization, 100% Stage 1 encoder transfer, trainable backbone, finite gradients/loss/parameters, optimizer update count, both GPU memory peaks, and a single validation sample. Its report is `audit/preflight.json`. The launcher refuses to start unless that report is ready and the checkpoint directory is empty.

## Manual launch

In a shell, create a tmux session and then run the following inside it:

```bash
tmux new -s stage2_b0_ddp2
cd /workspace/GF-Transformer
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate gft
bash experiments/stage2_b0_ddp2/run_b0_ddp2.sh
```

The launcher writes the current log path to `experiments/stage2_b0_ddp2/current_log.txt` and uses a UTC timestamp with `tee`. Detach with `Ctrl-b d` and return with `tmux attach -t stage2_b0_ddp2`.

To view recent epoch and validation lines from another shell:

```bash
cd /workspace/GF-Transformer
LOG=$(cat experiments/stage2_b0_ddp2/current_log.txt)
tail -c 300000 "$LOG" | tr '\r' '\n' | grep -E 'optimizer_updates|Val Score:|score_best:|B0 DDP2 done' | tail -n 30
```

The formal trainer writes `results/validation_history.csv` one row per validation epoch, including `F1b`, `F1d`, `F1s`, all four damage class F1 values, and the running best score. Commit this small CSV after completion. The raw tqdm log, checkpoint, and current-log pointer are ignored by Git.

## After B0-D2 training

Evaluate the best B0-D2 and frozen K2-D2 checkpoints on the *same* fixed 917 images, same localization masks, and same single-view metric. Compare both best checkpoints and the validation histories. For uncertainty, collect each image's TP/FP/FN for building and four damage classes. In each paired bootstrap replicate, resample the **same 917 image indices** for both models, sum the sufficient statistics by class across sampled images, then compute the original global F1 formulas (`2TP/(2TP+FP+FN)`), four-class harmonic `F1d`, and `F1s = 0.3 F1b + 0.7 F1d`. Report the paired difference distribution and interval. Do not average image-level F1; that would change the metric.

Existing compact histories: `experiments/stage2_k2_kalman_refine/results/ddp2_validation_history.csv` and `experiments/stage2_k2_kalman_refine/results/baseline_b_validation_history.csv`.
