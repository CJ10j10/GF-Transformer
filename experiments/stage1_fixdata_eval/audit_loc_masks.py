#!/usr/bin/env python3
"""
Gate 1: audit the localization masks written to disk by inference_loc.py.

Reads back the PNGs in loc_masks/ for the FIXED 917-image validation split
(same split construction as training / eval_stage1.py — no re-randomization)
and checks:

  * missing : file absent or unreadable          (expected 0)
  * broken  : wrong shape or values outside {0,255}  (expected 0)
  * mean per-image Dice, global F1, IoU vs the ground-truth masks
              (expected mean Dice ~0.883, global F1 ~0.863)

Results are saved to results/loc_mask_audit.json.
"""

import os
import sys
import json
import datetime

import numpy as np
import cv2
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# Reuse the EXACT split construction from the independent re-evaluation
# (which itself replicates train_segformer_loc.py).
from eval_stage1 import build_file_list, SEED, TEST_SIZE  # noqa: E402
from utils import dice  # noqa: E402  (same metric as training validate())

LOC_DIR = os.path.join(HERE, 'loc_masks')
OUT_JSON = os.path.join(HERE, 'results', 'loc_mask_audit.json')


def main():
    all_files = build_file_list()
    all_idxs = np.arange(len(all_files))
    _, val_idxs = train_test_split(all_idxs, test_size=TEST_SIZE,
                                   random_state=SEED, shuffle=True)
    assert len(all_files) == 9168, f'unexpected total {len(all_files)}'
    assert len(val_idxs) == 917, f'unexpected val size {len(val_idxs)}'

    total_masks = len([f for f in os.listdir(LOC_DIR) if f.endswith('.png')])

    missing, broken = [], []
    tp = fp = fn = 0
    per_image_dice = []

    for i in val_idxs:
        img_path = all_files[i]
        name = os.path.basename(img_path).replace('.png', '_part1.png')
        mp = os.path.join(LOC_DIR, name)

        if not os.path.exists(mp):
            missing.append({'file': img_path, 'reason': 'absent'}); continue
        m = cv2.imread(mp, cv2.IMREAD_UNCHANGED)
        if m is None:
            missing.append({'file': img_path, 'reason': 'unreadable'}); continue
        if m.shape != (1024, 1024) or not np.isin(m, (0, 255)).all():
            broken.append({'file': img_path, 'shape': list(m.shape),
                           'values': sorted(set(np.unique(m).tolist()))}); continue

        gt = cv2.imread(img_path.replace('/images/', '/masks/'), cv2.IMREAD_UNCHANGED) > 127
        pb = m > 127
        tp += int((pb & gt).sum())
        fp += int((pb & ~gt).sum())
        fn += int((~pb & gt).sum())
        per_image_dice.append(dice(gt, pb))

    mean_dice = float(np.mean(per_image_dice))
    f1 = 2 * tp / (2 * tp + fp + fn)
    iou = tp / (tp + fp + fn)

    audit = {
        'split': {'total_files': len(all_files), 'val_size': len(val_idxs),
                  'test_size': TEST_SIZE, 'random_state': SEED},
        'loc_masks_dir': os.path.abspath(LOC_DIR),
        'masks_on_disk': total_masks,
        'missing': {'count': len(missing), 'samples': missing[:10]},
        'broken': {'count': len(broken), 'samples': broken[:10]},
        'metrics': {
            'TP': tp, 'FP': fp, 'FN': fn,
            'mean_per_image_dice': mean_dice,
            'global_F1': f1,
            'global_IoU': iou,
            'per_image_dice_min': float(np.min(per_image_dice)),
            'per_image_dice_max': float(np.max(per_image_dice)),
        },
        'expectations': {
            'missing_eq_0': len(missing) == 0,
            'broken_eq_0': len(broken) == 0,
            'mean_dice_approx_0.883': abs(mean_dice - 0.883) < 0.01,
            'global_f1_approx_0.863': abs(f1 - 0.863) < 0.01,
        },
        'audited_at': datetime.datetime.now().isoformat(timespec='seconds'),
    }

    with open(OUT_JSON, 'w') as f:
        json.dump(audit, f, indent=2)

    print('=' * 72)
    print(' GATE 1: localization mask audit (fixed 917-image val split)')
    print('=' * 72)
    print(f"  masks on disk:   {total_masks} (expected 9168)")
    print(f"  missing:         {len(missing)}  (required 0)")
    print(f"  broken:          {len(broken)}  (required 0)")
    print(f"  mean Dice:       {mean_dice:.6f}  (expected ~0.883)")
    print(f"  global F1:       {f1:.6f}  (expected ~0.863)")
    print(f"  global IoU:      {iou:.6f}")
    ok = (len(missing) == 0 and len(broken) == 0 and
          abs(mean_dice - 0.883) < 0.01 and abs(f1 - 0.863) < 0.01)
    print(f"  GATE 1: {'PASS' if ok else 'FAIL'}")
    print(f"  results -> {OUT_JSON}")
    print('=' * 72)


if __name__ == '__main__':
    main()
