# B0-D2 matched control: training handoff

Status: completed. The formal 50-epoch B0-D2 job finished successfully in 14.60 hours on 2026-09-26 UTC. The best checkpoint is epoch 13; its fixed 917-image single-view validation was independently reproduced on 2026-09-28.

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

## Original manual launch (archival)

The following was used for this completed run. The launcher now refuses to restart because the checkpoint directory is nonempty. In a shell, create a tmux session and then run the following inside it:

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

The formal trainer writes `results/validation_history.csv` one row per validation epoch, including `F1b`, `F1d`, `F1s`, all four damage class F1 values, and the running best score. This small CSV is committed with the evaluation artifacts. The raw tqdm log, checkpoint, and current-log pointer are ignored by Git.

## Completed result

The frozen B0-D2 best checkpoint is `ckpt/GFformer_cls_3_b0_ddp2_best14` (epoch 13, SHA-256 `3c10b6f1ea807ee77dddc11743bbf73e471ab6ad8a2473fc424116721f8a1193`). It reproduced F1b 0.8720, F1d 0.7115, F1s 0.7596, and minor-damage F1 0.5276. See `results/ddp2_single_view.json` for all class metrics and provenance. The compact `results/validation_history.csv` has 25 validation rows; no raw tqdm log or 216 MB checkpoint is committed.

Against the matched K2-D2 best checkpoint, B0-D2 is higher by 0.0097 F1s, 0.0139 F1d, and 0.0252 minor-damage F1. Against the original single GPU Baseline-B best, it is higher by 0.0116 F1s and 0.0359 minor-damage F1. These are validation differences between selected checkpoints; they do not measure variation across training seeds.

## Paired validation comparison

`paired_bootstrap.py` independently collected each image's TP/FP/FN for building and four damage classes from the best B0-D2 and K2-D2 checkpoints on the same fixed 917 images. Each of 10,000 paired replicates resampled the **same image indices** for both models, summed sufficient statistics by class, and then computed the unchanged global F1 formulas (`2TP/(2TP+FP+FN)`), harmonic `F1d`, and `F1s = 0.3 F1b + 0.7 F1d`. It never averaged image-level F1. The compact 38 KB per-image counts and full summary are in `results/paired_per_image_sufficient_statistics.npz` and `results/paired_bootstrap.json`.

B0 minus K2 observed F1s = +0.0097; paired 95% percentile interval [-0.0031, +0.0259]. Minor-damage F1 = +0.0252; interval [-0.0057, +0.0620]. Both intervals include zero. B0 is the higher-scoring checkpoint in this run, but the paired validation sample does not establish a stable model-level advantage. This bootstrap covers image sampling for already selected checkpoints; it does not include training-seed variation or the effect of choosing the best checkpoints on the same validation split.

Existing compact histories: `experiments/stage2_k2_kalman_refine/results/ddp2_validation_history.csv` and `experiments/stage2_k2_kalman_refine/results/baseline_b_validation_history.csv`.
