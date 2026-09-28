#!/usr/bin/env python3
"""Audit and seal the fixed B0-D2 vs K2-D2 validation provenance."""

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.stage2_fixdata.eval_repo_bs4 import sha256_file
from experiments.stage2_b0_ddp2.paired_bootstrap import metrics_from_counts

RESULTS = ROOT / 'experiments/stage2_b0_ddp2/results'
B0_JSON = RESULTS / 'ddp2_single_view.json'
K2_JSON = ROOT / 'experiments/stage2_k2_kalman_refine/results/ddp2_single_view.json'
BOOT_JSON = RESULTS / 'paired_bootstrap.json'
COUNTS_NPZ = RESULTS / 'paired_per_image_sufficient_statistics.npz'
SPLIT_SHA = '7e3c7e8cf1762512697055f6a7bfd5d51050a6d03fc4d93e0d86e1ffaa2c480e'


def main():
    b0 = json.loads(B0_JSON.read_text())
    k2 = json.loads(K2_JSON.read_text())
    boot = json.loads(BOOT_JSON.read_text())
    assert b0['mode'] == k2['mode'] == 'single'
    assert b0['views'] == k2['views'] == ['original']
    assert b0['validation_images'] == k2['validation_images'] == boot['validation_images'] == 917
    assert b0['validation_split_seed'] == k2['validation_split_seed'] == 3
    assert b0['validation_split_sha256'] == k2['validation_split_sha256'] == boot['validation_split_sha256'] == SPLIT_SHA
    assert b0['localization_masks'] == k2['localization_masks'] == 'experiments/stage1_fixdata_eval/loc_masks'
    assert b0['metric_code'] == k2['metric_code'] == 'train_segformer_cls.validate'
    assert b0['metric_code_sha256'] == k2['metric_code_sha256'] == boot['metric_code_sha256']
    assert sha256_file(ROOT / 'train_segformer_cls.py') == b0['metric_code_sha256']
    assert sha256_file(ROOT / b0['checkpoint']) == b0['checkpoint_sha256'] == boot['b0_checkpoint_sha256']
    assert sha256_file(ROOT / k2['checkpoint']) == k2['checkpoint_sha256'] == boot['k2_checkpoint_sha256']
    assert boot['resamples'] == 10000 and boot['seed'] == 20260928

    with np.load(COUNTS_NPZ) as data:
        b0_counts = data['b0_tp_fp_fn']
        k2_counts = data['k2_tp_fp_fn']
        files = data['validation_files'].tolist()
    assert b0_counts.shape == k2_counts.shape == (917, 5, 3)
    assert np.array_equal(b0_counts[:, 4, :], k2_counts[:, 4, :]), 'Localization building counts differ'
    assert hashlib.sha256('\n'.join(files).encode()).hexdigest() == SPLIT_SHA
    for path in files:
        assert Path(path).is_file()
    b0_metrics = metrics_from_counts(b0_counts.sum(axis=0))
    k2_metrics = metrics_from_counts(k2_counts.sum(axis=0))
    for key, item in boot['results'].items():
        observed = float(b0_metrics[key] - k2_metrics[key])
        assert abs(observed - item['observed_delta_b0_minus_k2']) < 1e-12
        assert abs(float(b0_metrics[key]) - b0['metrics'][key]) < 0.00005
        assert abs(float(k2_metrics[key]) - k2['metrics'][key]) < 0.00005

    mask_dir = ROOT / b0['localization_masks']
    mask_manifest = hashlib.sha256()
    latest_mask_mtime = 0.0
    for image in files:
        mask = mask_dir / (Path(image).stem + '_part1.png')
        assert mask.is_file(), mask
        latest_mask_mtime = max(latest_mask_mtime, mask.stat().st_mtime)
        mask_manifest.update(str(mask.relative_to(ROOT)).encode() + b'\0')
        mask_manifest.update(sha256_file(mask).encode() + b'\n')
    mask_sha = mask_manifest.hexdigest()
    # The same immutable mask files predate both independent evaluations.
    assert latest_mask_mtime < min(B0_JSON.stat().st_mtime, K2_JSON.stat().st_mtime)

    boot.update({
        'delta_definition': 'B0-D2 minus K2-D2',
        'bootstrap_rng_seed': boot['seed'],
        'n_boot': boot['resamples'],
        'image_count': len(files),
        'image_ids_sha256': SPLIT_SHA,
        'localization_masks': b0['localization_masks'],
        'validation_mask_manifest_sha256': mask_sha,
    })
    BOOT_JSON.write_text(json.dumps(boot, indent=2, sort_keys=True) + '\n')
    manifest = {
        'status': 'FROZEN',
        'scope': 'B0-D2 and K2-D2 selected checkpoint validation comparison',
        'delta_definition': boot['delta_definition'],
        'b0_checkpoint': b0['checkpoint'],
        'b0_checkpoint_sha256': b0['checkpoint_sha256'],
        'b0_best_epoch_1based': b0['checkpoint_epoch'],
        'k2_checkpoint': k2['checkpoint'],
        'k2_checkpoint_sha256': k2['checkpoint_sha256'],
        'k2_best_epoch_1based': k2['checkpoint_epoch'],
        'independent_reevaluation': True,
        'same_image_ids': True,
        'image_count': len(files),
        'image_ids_sha256': SPLIT_SHA,
        'same_localization_masks': True,
        'localization_masks': b0['localization_masks'],
        'validation_mask_manifest_sha256': mask_sha,
        'masks_predate_both_reevaluations': True,
        'same_metric_code': True,
        'metric_code': b0['metric_code'],
        'metric_code_sha256': b0['metric_code_sha256'],
        'paired_counts_agree_with_independent_metrics': True,
        'building_counts_identical_for_every_image': True,
        'bootstrap_rng_seed': boot['seed'],
        'n_boot': boot['resamples'],
        'bootstrap_summary': str(BOOT_JSON.relative_to(ROOT)),
        'per_image_counts': str(COUNTS_NPZ.relative_to(ROOT)),
    }
    output = RESULTS / 'freeze_manifest.json'
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(f'B0-D2 FROZEN: {output}')
    print(f'image_ids_sha256={SPLIT_SHA} mask_manifest_sha256={mask_sha}')


if __name__ == '__main__':
    main()
