#!/usr/bin/env python3
"""Read-only Baseline-B best-checkpoint evaluation on the fixed 917-image split."""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import train_segformer_cls as T  # noqa: E402

CKPT = ROOT / 'experiments/stage2_fixdata/ckpt_repo_bs4/GFformer_cls_3_fixdata_best14'
RESULTS = ROOT / 'experiments/stage2_fixdata/results_repo_bs4'
EXPECTED_SINGLE = {'F1b': 0.8720, 'F1d': 0.6949, 'F1s': 0.7480}
METRIC_PATTERN = re.compile(
    r'^Val Score: (?P<F1s>\d+\.\d+), Dice: (?P<F1b>\d+\.\d+), '
    r'F1: (?P<F1d>\d+\.\d+), F1_0:(?P<F1_0>\d+\.\d+) '
    r'F1_1:(?P<F1_1>\d+\.\d+) F1_2:(?P<F1_2>\d+\.\d+) '
    r'F1_3:(?P<F1_3>\d+\.\d+)$')


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class FourWayTTA(nn.Module):
    """Apply one spatial transform to all six pre/post channels, then invert logits."""
    DIMS = ((), (-1,), (-2,), (-2, -1))

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, imgs):
        logits = []
        for dims in self.DIMS:
            augmented = torch.flip(imgs, dims) if dims else imgs
            out = self.model(augmented)
            logits.append(torch.flip(out, dims) if dims else out)
        return torch.stack(logits, dim=0).mean(dim=0)


def evaluate(mode):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required for the production validation path')
    if not CKPT.is_file():
        raise FileNotFoundError(CKPT)
    torch.cuda.set_device(0)
    T.cudnn.benchmark = True
    _, val_idxs = T.train_test_split(
        np.arange(len(T.all_files)), test_size=0.1, random_state=3)
    assert len(T.all_files) == 9168 and len(val_idxs) == 917
    val_files = [T.all_files[i] for i in val_idxs]
    missing_masks = [fn for fn in val_files if not (Path(T.LOC_FOLDER) /
        Path(fn).name.replace('.png', '_part1.png')).is_file()]
    if missing_masks:
        raise RuntimeError(f'{len(missing_masks)} validation localization masks missing')
    split_sha256 = hashlib.sha256('\n'.join(val_files).encode()).hexdigest()
    checkpoint_sha256 = sha256_file(CKPT)

    model = T.GFformer_two().cuda()
    checkpoint = torch.load(str(CKPT), map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    net = model if mode == 'single' else FourWayTTA(model).eval()
    loader = DataLoader(T.ValData(val_idxs), batch_size=T.VAL_BATCH,
                        shuffle=False, num_workers=4, pin_memory=True)

    captured = []
    original_dprint = T.dprint
    def capture_print(*args, **kwargs):
        line = ' '.join(str(arg) for arg in args)
        if line.startswith('Val Score:'):
            captured.append(line)
        original_dprint(*args, **kwargs)
    T.dprint = capture_print
    try:
        score = T.validate(net, loader)  # exact production metric implementation
    finally:
        T.dprint = original_dprint
    if len(captured) != 1:
        raise RuntimeError(f'Expected one metric line, got {captured!r}')
    match = METRIC_PATTERN.fullmatch(captured[0])
    if match is None:
        raise RuntimeError(f'Unexpected validation format: {captured[0]}')
    metrics = {name: float(value) for name, value in match.groupdict().items()}
    if abs(score - metrics['F1s']) > 0.000051:
        raise RuntimeError('Unrounded and logged validation scores disagree')
    if mode == 'single':
        for key, expected in EXPECTED_SINGLE.items():
            if abs(metrics[key] - expected) > 0.00005:
                raise RuntimeError(f'Single-view {key}={metrics[key]} != {expected}')
        if abs(score - checkpoint['best_score']) > 0.00005:
            raise RuntimeError('Single-view score differs from checkpoint best_score')

    result = {
        'mode': mode,
        'metrics': metrics,
        'F1s_unrounded': float(score),
        'checkpoint': str(CKPT.relative_to(ROOT)),
        'checkpoint_sha256': checkpoint_sha256,
        'checkpoint_epoch': int(checkpoint['epoch']),
        'checkpoint_best_score': float(checkpoint['best_score']),
        'validation_images': len(val_idxs),
        'validation_split_seed': 3,
        'validation_split_sha256': split_sha256,
        'localization_masks': str(Path(T.LOC_FOLDER).relative_to(ROOT)),
        'metric_code': 'train_segformer_cls.validate',
        'metric_code_sha256': sha256_file(ROOT / 'train_segformer_cls.py'),
        'views': ['original'] if mode == 'single' else [
            'original', 'horizontal_flip', 'vertical_flip', 'rotation_180'],
        'aggregation': 'mean inverse-transformed logits before softmax and argmax'
            if mode == 'tta' else 'single logits',
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                            cwd=str(ROOT), text=True).strip(),
    }
    RESULTS.mkdir(exist_ok=True)
    out_path = RESULTS / ('single_view.json' if mode == 'single' else 'tta4.json')
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(f'Result saved: {out_path}')
    if mode == 'tta':
        single = json.loads((RESULTS / 'single_view.json').read_text())
        for key in ('checkpoint_sha256', 'validation_split_sha256',
                    'localization_masks', 'metric_code_sha256'):
            if result[key] != single[key]:
                raise RuntimeError(f'TTA and single-view provenance differ: {key}')
        delta = {key: round(metrics[key] - single['metrics'][key], 4)
                 for key in ('F1b', 'F1d', 'F1s', 'F1_0', 'F1_1', 'F1_2', 'F1_3')}
        report = {'checkpoint': result['checkpoint'],
                  'checkpoint_sha256': checkpoint_sha256,
                  'single_view': single['metrics'], 'tta4': metrics,
                  'tta_absolute_delta': delta,
                  'validation_images': len(val_idxs),
                  'validation_split_sha256': split_sha256,
                  'localization_masks': result['localization_masks'],
                  'metric_code': result['metric_code'],
                  'tta_effective': delta['F1s'] > 0,
                  'frozen_baseline_protocol': 'tta4' if delta['F1s'] > 0 else 'single_view',
                  'frozen_baseline_metrics': metrics if delta['F1s'] > 0 else single['metrics']}
        report_path = RESULTS / 'baseline_report.json'
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
        print(f'Baseline report saved: {report_path}')
        print(f'TTA absolute deltas: {delta}')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('single', 'tta'), required=True)
    args = parser.parse_args()
    evaluate(args.mode)
