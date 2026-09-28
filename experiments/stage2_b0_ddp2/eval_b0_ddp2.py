#!/usr/bin/env python3
"""Independently re-evaluate frozen B0-D2 best on the 917-image single-view split."""

import csv
import hashlib
import json
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

CKPT = ROOT / 'experiments/stage2_b0_ddp2/ckpt/GFformer_cls_3_b0_ddp2_best14'
RESULTS = ROOT / 'experiments/stage2_b0_ddp2/results'
HISTORY = RESULTS / 'validation_history.csv'
BASELINE = ROOT / 'experiments/stage2_fixdata/results_repo_bs4/single_view.json'
K2 = ROOT / 'experiments/stage2_k2_kalman_refine/results/ddp2_single_view.json'
METRICS = ('F1b', 'F1d', 'F1s', 'F1_0', 'F1_1', 'F1_2', 'F1_3')


def evaluate():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for production validation')
    if not CKPT.is_file():
        raise FileNotFoundError(CKPT)
    baseline = json.loads(BASELINE.read_text())
    k2 = json.loads(K2.read_text())
    torch.cuda.set_device(0)
    T.cudnn.benchmark = True
    _, val_idxs = T.train_test_split(
        np.arange(len(T.all_files)), test_size=0.1, random_state=3)
    assert len(T.all_files) == 9168 and len(val_idxs) == 917
    val_files = [T.all_files[i] for i in val_idxs]
    split_sha256 = hashlib.sha256('\n'.join(val_files).encode()).hexdigest()
    if split_sha256 != baseline['validation_split_sha256'] or split_sha256 != k2['validation_split_sha256']:
        raise RuntimeError('Validation split differs across checkpoints')
    loc_folder = str(Path(T.LOC_FOLDER).relative_to(ROOT))
    if loc_folder != baseline['localization_masks'] or loc_folder != k2['localization_masks']:
        raise RuntimeError('Localization mask directory differs across checkpoints')
    missing = [fn for fn in val_files if not (
        Path(T.LOC_FOLDER) / Path(fn).name.replace('.png', '_part1.png')).is_file()]
    if missing:
        raise RuntimeError(f'{len(missing)} validation masks missing')
    metric_sha256 = sha256_file(ROOT / 'train_segformer_cls.py')
    if metric_sha256 != baseline['metric_code_sha256'] or metric_sha256 != k2['metric_code_sha256']:
        raise RuntimeError('Validation metric code differs across checkpoints')

    checkpoint_sha256 = sha256_file(CKPT)
    checkpoint = torch.load(str(CKPT), map_location='cpu')
    model = T.GFformer_two(use_kalman=False).cuda().eval()
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
    if int(checkpoint['epoch']) != 13:
        raise RuntimeError(f"Unexpected best epoch: {checkpoint['epoch']}")
    with HISTORY.open(newline='', encoding='utf-8') as handle:
        history = list(csv.DictReader(handle))
    if len(history) != 25:
        raise RuntimeError(f'Expected 25 validation entries, got {len(history)}')
    best = max(history, key=lambda row: float(row['F1s']))
    if int(best['checkpoint_epoch_1based']) != checkpoint['epoch']:
        raise RuntimeError('History best epoch differs from checkpoint epoch')
    for key in METRICS:
        if abs(metrics[key] - float(best[key])) > 0.00005:
            raise RuntimeError(f'{key} differs from best history row')

    result = {
        'mode': 'single', 'views': ['original'],
        'checkpoint': str(CKPT.relative_to(ROOT)),
        'checkpoint_sha256': checkpoint_sha256,
        'checkpoint_epoch': int(checkpoint['epoch']),
        'checkpoint_best_score': float(checkpoint['best_score']),
        'F1s_unrounded': float(score),
        'metrics': metrics,
        'baseline_checkpoint': baseline['checkpoint'],
        'baseline_metrics': baseline['metrics'],
        'k2_checkpoint': k2['checkpoint'],
        'k2_metrics': k2['metrics'],
        'absolute_delta_vs_baseline_b': {
            key: round(metrics[key] - baseline['metrics'][key], 4) for key in METRICS},
        'absolute_delta_vs_k2_ddp2': {
            key: round(metrics[key] - k2['metrics'][key], 4) for key in METRICS},
        'validation_images': len(val_idxs),
        'validation_split_seed': 3,
        'validation_split_sha256': split_sha256,
        'localization_masks': loc_folder,
        'metric_code': 'train_segformer_cls.validate',
        'metric_code_sha256': metric_sha256,
        'eval_git_head': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=str(ROOT), text=True).strip(),
    }
    output = RESULTS / 'ddp2_single_view.json'
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(f'Result saved: {output}')
    print(f'Absolute delta vs K2-D2: {result["absolute_delta_vs_k2_ddp2"]}')
    return result


if __name__ == '__main__':
    evaluate()
