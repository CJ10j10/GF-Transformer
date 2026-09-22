#!/usr/bin/env python3
"""
Independent re-evaluation of the Stage 1 (building localization) checkpoint.

Purpose
-------
Verify the checkpoint that Stage 2 will use, with the EXACT same validation
protocol as the original training run (train_segformer_loc.py):
  * same file listing order (data/xBD/train + data/xBD/tier3, sorted,
    filter '*_pre_disaster.png')
  * same random split: train_test_split(all_idxs, test_size=0.1,
    random_state=3, shuffle=True)  -- NO new random split
  * same data loading: full-resolution image (1024x1024), no augmentation,
    preprocess_inputs() normalization
  * same decision threshold: sigmoid output > 0.5

Outputs per-requirement metrics:
  TP, FP, FN, Precision, Recall, F1, IoU  (global, pixel-level over the
  whole validation split) plus the training-style metric (mean per-image
  Dice) to cross-check against the checkpoint's recorded best_score.

Gate: F1 (global) must be >= 0.85.

Usage:
  /usr/local/miniconda3/envs/gft/bin/python eval_stage1.py
"""

import os
import sys
import json
import hashlib
import datetime

import numpy as np
import cv2
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'model'))

from gfmodel import GFformer_one
from utils import preprocess_inputs, dice

# ── The ONLY Stage-1 checkpoint this evaluation (and Stage 2) may use ──
CKPT_PATH = os.path.join(
    os.path.dirname(__file__), 'ckpt',
    'GFformer_loc_fixdata_ep57_valdice0.8830_sha_cd940989.pt')

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
RESULTS_JSON = os.path.join(RESULTS_DIR, 'metrics.json')

DATA_BASE = os.path.join(REPO_ROOT, 'data', 'xBD')
TRAIN_DIRS = [os.path.join(DATA_BASE, 'train'), os.path.join(DATA_BASE, 'tier3')]

SEED = 3                 # identical to train_segformer_loc.py
TEST_SIZE = 0.1          # identical to train_segformer_loc.py
THRESHOLD = 0.5          # identical to validate() in train_segformer_loc.py
BATCH_SIZE = 4           # identical to the training run's val_batch_size
F1_GATE = 0.85           # requirement: F1 must not be below this


def build_file_list():
    """Reconstruct all_files EXACTLY as train_segformer_loc.py does."""
    all_files = []
    for d in TRAIN_DIRS:
        for f in sorted(os.listdir(os.path.join(d, 'images'))):
            if '_pre_disaster.png' in f:
                all_files.append(os.path.join(d, 'images', f))
    return all_files


def checkpoint_info(path):
    ck = torch.load(path, map_location='cpu')
    sha = hashlib.sha256(open(path, 'rb').read()).hexdigest()
    return {
        'path': os.path.abspath(path),
        'sha256': sha,
        'size_mb': round(os.path.getsize(path) / 1024 / 1024, 2),
        'recorded_epoch': ck.get('epoch'),
        'recorded_best_score': ck.get('best_score'),
        'state_dict_keys': len(ck.get('state_dict', {})),
        'ckpt': ck,
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cv2.setNumThreads(0)
    torch.cuda.set_device(0)

    # ── 1. Checkpoint identity ──────────────────────────────────────────
    info = checkpoint_info(CKPT_PATH)
    print('=' * 76)
    print(' CHECKPOINT UNDER TEST')
    print('=' * 76)
    print(f"  path:               {info['path']}")
    print(f"  sha256:             {info['sha256']}")
    print(f"  size:               {info['size_mb']} MB")
    print(f"  recorded epoch:     {info['recorded_epoch']}")
    print(f"  recorded best_score:{info['recorded_best_score']:.6f}")
    print(f"  state_dict keys:    {info['state_dict_keys']}")

    # ── 2. Validation split — same code as training, NO re-randomization ─
    all_files = build_file_list()
    all_idxs = np.arange(len(all_files))
    train_idxs, val_idxs = train_test_split(
        all_idxs, test_size=TEST_SIZE, random_state=SEED, shuffle=True)

    print('=' * 76)
    print(' VALIDATION SPLIT (reconstructed from training code)')
    print('=' * 76)
    print(f"  total files:        {len(all_files)}")
    print(f"  val size:           {len(val_idxs)}")
    print(f"  split params:       test_size={TEST_SIZE}, random_state={SEED}")
    # Sanity: the original run logged exactly these numbers.
    assert len(all_files) == 9168, f'unexpected file count {len(all_files)}'
    assert len(val_idxs) == 917, f'unexpected val size {len(val_idxs)}'
    print('  [OK] split matches the training run (9168 total / 917 val)')

    # ── 3. Model ────────────────────────────────────────────────────────
    model = GFformer_one().cuda()
    model.load_state_dict(info['ckpt']['state_dict'], strict=True)
    model.eval()
    del info['ckpt']
    print('=' * 76)
    print(' MODEL')
    print('=' * 76)
    print('  GFformer_one, state_dict loaded strictly, eval mode')

    # ── 4. Evaluate (full-res, no augmentation, threshold 0.5) ─────────
    tp = fp = fn = tn = 0
    per_image_dice = []

    val_files = [all_files[i] for i in val_idxs]
    print('=' * 76)
    print(f' EVALUATING {len(val_files)} validation images (batch {BATCH_SIZE})')
    print('=' * 76)

    with torch.no_grad():
        for start in tqdm(range(0, len(val_files), BATCH_SIZE), desc='Eval'):
            imgs, gts = [], []
            for img_path in val_files[start:start + BATCH_SIZE]:
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                msk0 = cv2.imread(img_path.replace('/images/', '/masks/'), cv2.IMREAD_UNCHANGED)
                assert img.shape[:2] == (1024, 1024), f'{img_path}: {img.shape}'
                gt = (msk0 > 127)  # same binarization as training ValData
                img = preprocess_inputs(img)
                imgs.append(torch.from_numpy(img.transpose(2, 0, 1)).float())
                gts.append(gt)
            x = torch.stack(imgs).cuda()
            out = torch.sigmoid(model(x)[:, 0, ...]).cpu().numpy()

            for gt, pred in zip(gts, out):
                pred_b = pred > THRESHOLD
                tp += int((pred_b & gt).sum())
                fp += int((pred_b & ~gt).sum())
                fn += int((~pred_b & gt).sum())
                tn += int((~pred_b & ~gt).sum())
                per_image_dice.append(dice(gt, pred_b))  # utils.dice, as in training

    # ── 5. Metrics ──────────────────────────────────────────────────────
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    mean_dice = float(np.mean(per_image_dice))

    metrics = {
        'checkpoint': {
            'path': info['path'],
            'sha256': info['sha256'],
            'size_mb': info['size_mb'],
            'recorded_epoch': info['recorded_epoch'],
            'recorded_best_score': info['recorded_best_score'],
        },
        'split': {
            'total_files': len(all_files),
            'val_size': len(val_idxs),
            'test_size': TEST_SIZE,
            'random_state': SEED,
        },
        'eval': {
            'threshold': THRESHOLD,
            'batch_size': BATCH_SIZE,
            'full_resolution': True,
            'augmentation': 'none',
            'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
            'precision': precision,
            'recall': recall,
            'F1': f1,
            'IoU': iou,
            'mean_per_image_dice': mean_dice,
            'per_image_dice_std': float(np.std(per_image_dice)),
            'per_image_dice_min': float(np.min(per_image_dice)),
            'per_image_dice_max': float(np.max(per_image_dice)),
        },
        'gate': {
            'requirement': f'F1 >= {F1_GATE}',
            'passed': f1 >= F1_GATE,
        },
        'evaluated_at': datetime.datetime.now().isoformat(timespec='seconds'),
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_JSON, 'w') as f:
        json.dump(metrics, f, indent=2)

    print('=' * 76)
    print(' METRICS')
    print('=' * 76)
    print(f"  TP:        {tp}")
    print(f"  FP:        {fp}")
    print(f"  FN:        {fn}")
    print(f"  TN:        {tn}")
    print(f"  Precision: {precision:.6f}")
    print(f"  Recall:    {recall:.6f}")
    print(f"  F1:        {f1:.6f}")
    print(f"  IoU:       {iou:.6f}")
    print(f"  Mean per-image Dice (training metric): {mean_dice:.6f}")
    print(f"    vs checkpoint recorded best_score:   {info['recorded_best_score']:.6f}")
    print('=' * 76)
    if f1 >= F1_GATE:
        print(f" GATE: PASS  (F1 {f1:.6f} >= {F1_GATE})")
    else:
        print(f" GATE: FAIL  (F1 {f1:.6f} < {F1_GATE})")
    print(f" Results written to: {RESULTS_JSON}")
    print('=' * 76)


if __name__ == '__main__':
    main()
