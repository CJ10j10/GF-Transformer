#!/usr/bin/env python3
"""Independently re-evaluate the frozen K2 DDP2 best on Baseline-B's protocol."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import train_segformer_cls as T
from experiments.stage2_fixdata.eval_repo_bs4 import METRIC_PATTERN, sha256_file

CKPT = ROOT / 'experiments/stage2_k2_kalman_refine/ckpt_ddp2/GFformer_cls_3_k2_ddp2_best14'
RESULTS = ROOT / 'experiments/stage2_k2_kalman_refine/results'
BASELINE = ROOT / 'experiments/stage2_fixdata/results_repo_bs4/single_view.json'
METRICS = ('F1b', 'F1d', 'F1s', 'F1_0', 'F1_1', 'F1_2', 'F1_3')


def evaluate():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for production validation')
    if not CKPT.is_file():
        raise FileNotFoundError(CKPT)
    baseline = json.loads(BASELINE.read_text())
    torch.cuda.set_device(0)
    T.cudnn.benchmark = True
    _, val_idxs = T.train_test_split(
        np.arange(len(T.all_files)), test_size=0.1, random_state=3)
    assert len(T.all_files) == 9168 and len(val_idxs) == 917
    val_files = [T.all_files[i] for i in val_idxs]
    import hashlib
    split_sha256 = hashlib.sha256('\n'.join(val_files).encode()).hexdigest()
    if split_sha256 != baseline['validation_split_sha256']:
        raise RuntimeError('Validation split differs from Baseline-B')
    loc_folder = str(Path(T.LOC_FOLDER).relative_to(ROOT))
    if loc_folder != baseline['localization_masks']:
        raise RuntimeError('Localization mask directory differs from Baseline-B')
    missing = [fn for fn in val_files if not (
        Path(T.LOC_FOLDER) / Path(fn).name.replace('.png', '_part1.png')).is_file()]
    if missing:
        raise RuntimeError(f'{len(missing)} validation masks missing')
    metric_sha256 = sha256_file(ROOT / 'train_segformer_cls.py')
    if metric_sha256 != baseline['metric_code_sha256']:
        raise RuntimeError('Validation metric code differs from Baseline-B')

    checkpoint_sha256 = sha256_file(CKPT)
    checkpoint = torch.load(str(CKPT), map_location='cpu')
    model = T.GFformer_two(use_kalman=True).cuda().eval()
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    loader = DataLoader(T.ValData(val_idxs), batch_size=1, shuffle=False,
                        num_workers=4, pin_memory=True)
    captured = []
    original_dprint = T.dprint
    def capture_print(*args, **kwargs):
        line = ' '.join(str(arg) for arg in args)
        if line.startswith('Val Score:'):
            captured.append(line)
        original_dprint(*args, **kwargs)
    T.dprint = capture_print
    try:
        score = T.validate(model, loader)
    finally:
        T.dprint = original_dprint
    if len(captured) != 1:
        raise RuntimeError(f'Expected one metric line, got {captured!r}')
    match = METRIC_PATTERN.fullmatch(captured[0])
    if match is None:
        raise RuntimeError(f'Unexpected metric format: {captured[0]}')
    metrics = {name: float(value) for name, value in match.groupdict().items()}
    if abs(score - metrics['F1s']) > 0.000051:
        raise RuntimeError('Rounded and unrounded scores disagree')
    if abs(score - float(checkpoint['best_score'])) > 0.00005:
        raise RuntimeError('Independent score differs from checkpoint best_score')
    if int(checkpoint['epoch']) != 27:
        raise RuntimeError(f"Unexpected best epoch: {checkpoint['epoch']}")

    delta = {key: round(metrics[key] - baseline['metrics'][key], 4)
             for key in METRICS}
    result = {
        'mode': 'single',
        'views': ['original'],
        'checkpoint': str(CKPT.relative_to(ROOT)),
        'checkpoint_sha256': checkpoint_sha256,
        'checkpoint_epoch': int(checkpoint['epoch']),
        'checkpoint_best_score': float(checkpoint['best_score']),
        'F1s_unrounded': float(score),
        'metrics': metrics,
        'baseline_checkpoint': baseline['checkpoint'],
        'baseline_checkpoint_sha256': baseline['checkpoint_sha256'],
        'baseline_metrics': baseline['metrics'],
        'absolute_delta_vs_baseline_b': delta,
        'validation_images': len(val_idxs),
        'validation_split_seed': 3,
        'validation_split_sha256': split_sha256,
        'localization_masks': loc_folder,
        'metric_code': 'train_segformer_cls.validate',
        'metric_code_sha256': metric_sha256,
        'eval_git_head': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=str(ROOT), text=True).strip(),
    }
    RESULTS.mkdir(exist_ok=True)
    output = RESULTS / 'ddp2_single_view.json'
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(f'Result saved: {output}')
    print(f'Absolute delta vs Baseline-B: {delta}')
    return result


if __name__ == '__main__':
    evaluate()
